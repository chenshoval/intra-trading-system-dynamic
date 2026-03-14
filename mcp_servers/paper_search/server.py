"""
Local MCP server for academic paper search.
Provides tools to search arXiv and Semantic Scholar (which indexes SSRN, PubMed, etc.)
for trading/quant research papers.

Run via: uv run --with "mcp[cli]" --with httpx --with beautifulsoup4 python server.py
"""

import sys
import json
import logging
import xml.etree.ElementTree as ET
from typing import Optional
import httpx
from mcp.server.fastmcp import FastMCP

# Logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("paper-search")

mcp = FastMCP("paper-search")

# --- Constants ---
ARXIV_API_BASE = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

HEADERS = {
    "User-Agent": "PaperSearchMCP/1.0 (academic research tool)"
}

# arXiv Atom namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _parse_arxiv_entry(entry: ET.Element) -> dict:
    """Parse a single arXiv Atom entry into a clean dict."""
    title = entry.findtext(f"{ATOM_NS}title", "").strip().replace("\n", " ")
    abstract = entry.findtext(f"{ATOM_NS}summary", "").strip().replace("\n", " ")
    arxiv_id = entry.findtext(f"{ATOM_NS}id", "").strip()

    authors = []
    for author_el in entry.findall(f"{ATOM_NS}author"):
        name = author_el.findtext(f"{ATOM_NS}name", "").strip()
        if name:
            authors.append(name)

    published = entry.findtext(f"{ATOM_NS}published", "").strip()[:10]  # YYYY-MM-DD
    updated = entry.findtext(f"{ATOM_NS}updated", "").strip()[:10]

    categories = []
    for cat_el in entry.findall(f"{ATOM_NS}category"):
        term = cat_el.get("term", "")
        if term:
            categories.append(term)

    # Get PDF link
    pdf_url = ""
    for link_el in entry.findall(f"{ATOM_NS}link"):
        if link_el.get("title") == "pdf":
            pdf_url = link_el.get("href", "")

    return {
        "title": title,
        "authors": ", ".join(authors),
        "abstract": abstract[:500] + ("..." if len(abstract) > 500 else ""),
        "full_abstract": abstract,
        "arxiv_id": arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id,
        "url": arxiv_id,
        "pdf_url": pdf_url,
        "published": published,
        "updated": updated,
        "categories": ", ".join(categories),
    }


@mcp.tool()
async def search_arxiv(
    query: str,
    max_results: int = 10,
    category: str = "",
    sort_by: str = "relevance",
) -> str:
    """Search arXiv for academic papers.

    Args:
        query: Search terms (e.g. "momentum rotation equity sector")
        max_results: Number of results to return (default 10, max 50)
        category: Optional arXiv category filter (e.g. "q-fin" for quantitative finance,
                  "q-fin.PM" for portfolio management, "cs.LG" for machine learning).
                  Leave empty for all categories.
        sort_by: Sort order - "relevance" (default), "lastUpdatedDate", or "submittedDate"
    """
    max_results = min(max_results, 50)

    # Build search query — arXiv uses + for AND between terms, spaces within terms
    search_terms = query.strip()
    search_query = f"all:{search_terms}"
    if category:
        search_query += f" AND cat:{category}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                ARXIV_API_BASE,
                params=params,
                headers=HEADERS,
                timeout=30.0,
            )
            resp.raise_for_status()
    except Exception as e:
        return f"Error searching arXiv: {e}"

    # Parse Atom XML
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return f"Error parsing arXiv response: {e}"

    entries = root.findall(f"{ATOM_NS}entry")
    if not entries:
        return f"No results found for: {query}"

    results = []
    for i, entry in enumerate(entries, 1):
        paper = _parse_arxiv_entry(entry)
        results.append(
            f"[{i}] {paper['title']}\n"
            f"    Authors: {paper['authors']}\n"
            f"    Published: {paper['published']} | Categories: {paper['categories']}\n"
            f"    URL: {paper['url']}\n"
            f"    PDF: {paper['pdf_url']}\n"
            f"    Abstract: {paper['abstract']}\n"
        )

    total = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults", "?")
    header = f"arXiv search: '{query}' — {len(entries)} of {total} results\n{'='*60}\n\n"
    return header + "\n".join(results)


@mcp.tool()
async def search_papers(
    query: str,
    max_results: int = 10,
    year_from: Optional[int] = None,
    fields_of_study: str = "",
) -> str:
    """Search academic papers via Semantic Scholar (indexes SSRN, arXiv, PubMed, and more).

    Args:
        query: Search terms (e.g. "sector rotation momentum strategy")
        max_results: Number of results (default 10, max 100)
        year_from: Only return papers published from this year onwards (e.g. 2020)
        fields_of_study: Filter by field (e.g. "Economics", "Computer Science", "Business").
                         Leave empty for all fields.
    """
    max_results = min(max_results, 100)

    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,url,year,citationCount,externalIds,openAccessPdf",
    }
    if year_from:
        params["year"] = f"{year_from}-"
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                SEMANTIC_SCHOLAR_API,
                params=params,
                headers=HEADERS,
                timeout=30.0,
            )
            if resp.status_code == 429:
                return (
                    "Rate limited by Semantic Scholar. Wait a moment and try again. "
                    "For heavy use, get a free API key at https://www.semanticscholar.org/product/api#api-key-form"
                )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error searching papers: {e}"
    except Exception as e:
        return f"Error searching papers: {e}"

    data = resp.json()
    papers = data.get("data", [])
    total = data.get("total", 0)

    if not papers:
        return f"No results found for: {query}"

    results = []
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown")
        authors_list = paper.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list[:5])
        if len(authors_list) > 5:
            authors += f" + {len(authors_list)-5} more"
        year = paper.get("year", "?")
        citations = paper.get("citationCount", 0)
        abstract = paper.get("abstract", "") or ""
        abstract_short = abstract[:500] + ("..." if len(abstract) > 500 else "")
        url = paper.get("url", "")

        # Check for SSRN or DOI links
        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        ssrn_id = ext_ids.get("SSRN", "")

        pdf_info = paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url", "")

        source_line = f"    URL: {url}"
        if ssrn_id:
            source_line += f"\n    SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}"
        if doi:
            source_line += f"\n    DOI: https://doi.org/{doi}"
        if pdf_url:
            source_line += f"\n    PDF: {pdf_url}"

        results.append(
            f"[{i}] {title}\n"
            f"    Authors: {authors}\n"
            f"    Year: {year} | Citations: {citations}\n"
            f"{source_line}\n"
            f"    Abstract: {abstract_short}\n"
        )

    header = f"Paper search: '{query}' — {len(papers)} of {total} results\n{'='*60}\n\n"
    return header + "\n".join(results)


@mcp.tool()
async def read_paper(url: str) -> str:
    """Fetch and read the full content of a paper page (arXiv or Semantic Scholar).
    Returns the full abstract, metadata, and any available links.

    Args:
        url: URL of the paper (arXiv abstract page, Semantic Scholar page, or SSRN page)
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Handle arXiv URLs - use API for clean data
            if "arxiv.org" in url:
                # Extract arXiv ID
                arxiv_id = url.rstrip("/").split("/")[-1]
                # Remove version suffix for API lookup
                api_url = f"{ARXIV_API_BASE}?id_list={arxiv_id}&max_results=1"
                resp = await client.get(api_url, headers=HEADERS, timeout=30.0)
                resp.raise_for_status()

                root = ET.fromstring(resp.text)
                entries = root.findall(f"{ATOM_NS}entry")
                if not entries:
                    return f"Paper not found: {arxiv_id}"

                paper = _parse_arxiv_entry(entries[0])
                return (
                    f"Title: {paper['title']}\n"
                    f"Authors: {paper['authors']}\n"
                    f"Published: {paper['published']} | Updated: {paper['updated']}\n"
                    f"Categories: {paper['categories']}\n"
                    f"URL: {paper['url']}\n"
                    f"PDF: {paper['pdf_url']}\n"
                    f"\nFull Abstract:\n{paper['full_abstract']}\n"
                )

            # Handle Semantic Scholar URLs
            elif "semanticscholar.org" in url:
                paper_id = url.rstrip("/").split("/")[-1]
                api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
                params = {
                    "fields": "title,authors,abstract,url,year,citationCount,references.title,externalIds,openAccessPdf,tldr"
                }
                resp = await client.get(api_url, params=params, headers=HEADERS, timeout=30.0)
                if resp.status_code == 429:
                    return "Rate limited. Wait a moment and try again."
                resp.raise_for_status()

                paper = resp.json()
                title = paper.get("title", "Unknown")
                authors = ", ".join(a.get("name", "") for a in paper.get("authors", []))
                abstract = paper.get("abstract", "No abstract available")
                year = paper.get("year", "?")
                citations = paper.get("citationCount", 0)
                tldr = paper.get("tldr", {})
                tldr_text = tldr.get("text", "") if tldr else ""

                ext_ids = paper.get("externalIds", {}) or {}
                doi = ext_ids.get("DOI", "")
                ssrn_id = ext_ids.get("SSRN", "")

                refs = paper.get("references", []) or []
                ref_titles = [r.get("title", "") for r in refs[:10] if r.get("title")]

                result = (
                    f"Title: {title}\n"
                    f"Authors: {authors}\n"
                    f"Year: {year} | Citations: {citations}\n"
                    f"URL: {paper.get('url', url)}\n"
                )
                if doi:
                    result += f"DOI: https://doi.org/{doi}\n"
                if ssrn_id:
                    result += f"SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}\n"
                if tldr_text:
                    result += f"\nTL;DR: {tldr_text}\n"
                result += f"\nFull Abstract:\n{abstract}\n"
                if ref_titles:
                    result += f"\nKey References:\n"
                    for j, ref in enumerate(ref_titles, 1):
                        result += f"  {j}. {ref}\n"
                return result

            # Generic URL - try to fetch and extract text
            else:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html",
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()

                from bs4 import BeautifulSoup

                soup = BeautifulSoup(resp.text, "html.parser")

                # Remove script and style elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                title = soup.find("title")
                title_text = title.get_text().strip() if title else "Unknown"

                # Try to find abstract
                abstract_el = soup.find(class_=lambda c: c and "abstract" in c.lower() if c else False)
                abstract = abstract_el.get_text().strip() if abstract_el else ""

                # Get main text content
                body_text = soup.get_text(separator="\n", strip=True)
                # Clean up excessive whitespace
                lines = [line.strip() for line in body_text.split("\n") if line.strip()]
                clean_text = "\n".join(lines[:100])  # First 100 lines

                result = f"Title: {title_text}\n\n"
                if abstract:
                    result += f"Abstract:\n{abstract}\n\n"
                result += f"Content (first 100 lines):\n{clean_text}\n"
                return result

    except Exception as e:
        return f"Error reading paper at {url}: {e}"


@mcp.tool()
async def search_ssrn(
    query: str,
    max_results: int = 10,
    year_from: Optional[int] = None,
) -> str:
    """Search for SSRN papers specifically via Semantic Scholar.
    SSRN doesn't have a public API, so we use Semantic Scholar which indexes SSRN papers.

    Args:
        query: Search terms (e.g. "sector rotation momentum strategy")
        max_results: Number of results (default 10, max 50)
        year_from: Only return papers from this year onwards (e.g. 2020)
    """
    # Semantic Scholar indexes SSRN papers - search there and filter
    max_results = min(max_results, 50)

    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,url,year,citationCount,externalIds,openAccessPdf",
        "fieldsOfStudy": "Economics,Business",
    }
    if year_from:
        params["year"] = f"{year_from}-"

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                SEMANTIC_SCHOLAR_API,
                params=params,
                headers=HEADERS,
                timeout=30.0,
            )
            if resp.status_code == 429:
                return (
                    "Rate limited by Semantic Scholar. Wait a moment and try again. "
                    "Tip: Space out queries by a few seconds."
                )
            resp.raise_for_status()
    except Exception as e:
        return f"Error searching SSRN papers: {e}"

    data = resp.json()
    papers = data.get("data", [])
    total = data.get("total", 0)

    if not papers:
        return f"No SSRN/economics papers found for: {query}"

    results = []
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown")
        authors_list = paper.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list[:5])
        if len(authors_list) > 5:
            authors += f" + {len(authors_list)-5} more"
        year = paper.get("year", "?")
        citations = paper.get("citationCount", 0)
        abstract = paper.get("abstract", "") or ""
        abstract_short = abstract[:400] + ("..." if len(abstract) > 400 else "")
        url = paper.get("url", "")

        ext_ids = paper.get("externalIds", {}) or {}
        ssrn_id = ext_ids.get("SSRN", "")
        doi = ext_ids.get("DOI", "")

        source = ""
        if ssrn_id:
            source = f"SSRN #{ssrn_id}"
        elif doi:
            source = f"DOI: {doi}"

        pdf_info = paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url", "")

        entry = (
            f"[{i}] {title}\n"
            f"    Authors: {authors}\n"
            f"    Year: {year} | Citations: {citations}"
        )
        if source:
            entry += f" | {source}"
        entry += f"\n    URL: {url}\n"
        if ssrn_id:
            entry += f"    SSRN Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}\n"
        if pdf_url:
            entry += f"    PDF: {pdf_url}\n"
        entry += f"    Abstract: {abstract_short}\n"
        results.append(entry)

    header = f"SSRN/Economics paper search: '{query}' — {len(papers)} of {total} results\n{'='*60}\n\n"
    return header + "\n".join(results)


if __name__ == "__main__":
    mcp.run()
