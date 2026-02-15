"""SEC EDGAR Filings Pipeline — Custom data source.

Fetches SEC filings (10-K, 10-Q, 8-K) and extracts signals.
Potential use cases:
- Insider trading (Form 4) — similar to congressional but for company insiders
- Earnings surprises from 8-K filings
- Risk factor changes between 10-K filings
- Accounting red flags

Status: STUB — implement when ready to explore SEC data.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("data_pipelines.sec_filings")


class SECFilingsPipeline:
    """Fetch and process SEC EDGAR filings.

    SEC EDGAR API: https://www.sec.gov/edgar/sec-api-documentation
    - Free, no API key needed (just set User-Agent header)
    - Rate limit: 10 requests/second
    - Full text search available

    Priority filings:
    1. Form 4 (insider trading) — most actionable
    2. 8-K (material events) — earnings, M&A, management changes
    3. 10-K/10-Q (annual/quarterly) — long-term analysis
    """

    def __init__(self, user_agent: str = "TradingResearch admin@example.com"):
        self.user_agent = user_agent
        self.base_url = "https://efts.sec.gov/LATEST"

    def fetch_insider_trades(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch Form 4 (insider trading) filings.

        Returns DataFrame with: date, ticker, insider_name, title,
                                transaction_type, shares, price, value
        """
        # TODO: Implement
        # Use SEC EDGAR full-text search API or company filings API
        # https://efts.sec.gov/LATEST/search-index?q=...
        raise NotImplementedError("SEC insider trading fetcher not yet implemented.")

    def fetch_8k_events(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch 8-K (material events) filings.

        Returns DataFrame with: date, ticker, event_type, description
        """
        raise NotImplementedError("SEC 8-K fetcher not yet implemented.")

    def run(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
        filing_type: str = "form4",
        upload_to_azure: bool = False,
    ) -> pd.DataFrame:
        """Full pipeline: fetch → process → store."""
        if filing_type == "form4":
            return self.fetch_insider_trades(tickers, start_date, end_date)
        elif filing_type == "8k":
            return self.fetch_8k_events(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unknown filing type: {filing_type}")
