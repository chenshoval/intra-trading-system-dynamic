"""Data loading utilities.

Handles:
- Azure Blob Storage (for custom data + model artifacts)
- Local CSV files (for notebook experiments)
- Yahoo Finance (quick data for research)

Azure Blob is adapted from the old repo's azure_blob.py.
"""

import os
import io
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("trading.data_loaders")


# ══════════════════════════════════════════════════════════════
# Azure Blob Storage
# ══════════════════════════════════════════════════════════════

def _get_blob_service_client(account_url: Optional[str] = None):
    """Create BlobServiceClient using DefaultAzureCredential."""
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    url = account_url or os.environ.get(
        "ACCOUNT_URL", "https://tradingsystemsa.blob.core.windows.net"
    )
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=url, credential=credential)


def upload_to_blob(
    data: pd.DataFrame | bytes | str,
    blob_path: str,
    container: str = "trading-data-blob",
    account_url: Optional[str] = None,
) -> str:
    """Upload data to Azure Blob Storage.

    Args:
        data: DataFrame (saved as CSV), bytes, or string
        blob_path: Full path within the container
        container: Container name
        account_url: Storage account URL

    Returns:
        Full blob URL
    """
    client = _get_blob_service_client(account_url)
    blob_client = client.get_blob_client(container=container, blob=blob_path)

    if isinstance(data, pd.DataFrame):
        content = data.to_csv(index=False)
    elif isinstance(data, str):
        content = data
    else:
        content = data

    blob_client.upload_blob(content, overwrite=True)
    logger.info(f"Uploaded: {container}/{blob_path}")
    return f"{client.account_name}/{container}/{blob_path}"


def download_from_blob(
    blob_path: str,
    container: str = "trading-data-blob",
    account_url: Optional[str] = None,
    as_dataframe: bool = True,
) -> pd.DataFrame | bytes:
    """Download data from Azure Blob Storage.

    Args:
        blob_path: Full path within the container
        container: Container name
        account_url: Storage account URL
        as_dataframe: If True, parse CSV to DataFrame

    Returns:
        DataFrame or raw bytes
    """
    client = _get_blob_service_client(account_url)
    blob_client = client.get_blob_client(container=container, blob=blob_path)
    content = blob_client.download_blob().readall()

    if as_dataframe:
        return pd.read_csv(io.BytesIO(content))
    return content


def list_blobs(
    prefix: str = "",
    container: str = "trading-data-blob",
    account_url: Optional[str] = None,
) -> list[str]:
    """List blob names matching a prefix."""
    client = _get_blob_service_client(account_url)
    container_client = client.get_container_client(container)
    return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]


def blob_exists(
    blob_path: str,
    container: str = "trading-data-blob",
    account_url: Optional[str] = None,
) -> bool:
    """Check if a blob exists."""
    client = _get_blob_service_client(account_url)
    blob_client = client.get_blob_client(container=container, blob=blob_path)
    try:
        blob_client.get_blob_properties()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# Local CSV
# ══════════════════════════════════════════════════════════════

def load_csv(
    path: str,
    date_col: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load a local CSV file into a DataFrame.

    Automatically handles date parsing and sets date as index if requested.
    """
    kwargs = {}
    if parse_dates and date_col:
        kwargs["parse_dates"] = [date_col]

    df = pd.read_csv(path, **kwargs)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def load_data_directory(
    directory: str,
    pattern: str = "*.csv",
    date_col: str = "date",
) -> pd.DataFrame:
    """Load all CSV files from a directory into a single DataFrame."""
    import glob

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        logger.warning(f"No files matching {pattern} in {directory}")
        return pd.DataFrame()

    dfs = []
    for f in sorted(files):
        try:
            df = load_csv(f, date_col=date_col)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total rows from {len(dfs)} files")
    return combined


# ══════════════════════════════════════════════════════════════
# Yahoo Finance (quick research data)
# ══════════════════════════════════════════════════════════════

def load_yahoo(
    tickers: str | list[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download data from Yahoo Finance via yfinance.

    Args:
        tickers: Single ticker or list of tickers
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today
        interval: 1d, 1h, 5m, etc.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, ticker
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    if isinstance(tickers, str):
        tickers = [tickers]

    dfs = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if data.empty:
                logger.warning(f"No data for {ticker}")
                continue
            data = data.reset_index()
            data.columns = [c.lower() for c in data.columns]
            if "date" not in data.columns and "datetime" in data.columns:
                data = data.rename(columns={"datetime": "date"})
            data["ticker"] = ticker
            # Standardize column names
            col_map = {"adj close": "adj_close"}
            data = data.rename(columns=col_map)
            dfs.append(data)
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Downloaded {len(result)} rows for {len(dfs)} tickers from Yahoo")
    return result
