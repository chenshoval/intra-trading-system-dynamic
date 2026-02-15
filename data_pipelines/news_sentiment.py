"""News Sentiment Pipeline — Custom data not available in QuantConnect.

Fetches news articles, embeds them, and generates sentiment scores
that can be fed into QC algorithms via Custom Data API.

Pipeline: Fetch → Embed → Score → Store (Azure Blob) → Feed to QC

Status: STUB — implement when ready to add sentiment features.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("data_pipelines.news_sentiment")


class NewsSentimentPipeline:
    """Fetch news, compute sentiment, store as custom data for QC.

    Data Sources (pick one to start):
    - NewsAPI (newsapi.org) — 100 req/day free tier
    - Benzinga — requires paid API
    - Alpha Vantage News — free tier available
    - Financial Modeling Prep — news endpoint

    Embedding Models:
    - FinBERT (ProsusAI/finbert) — finance-specific, best accuracy
    - all-MiniLM-L6-v2 — general, fast, good enough to start

    Output Format (for QC Custom Data):
    CSV with columns: date, ticker, sentiment_score, sentiment_magnitude, article_count
    Upload to Azure Blob, reference in QC via CustomData class.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "ProsusAI/finbert",
    ):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def fetch_news(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
        source: str = "newsapi",
    ) -> pd.DataFrame:
        """Fetch news articles for given tickers.

        Returns DataFrame with: date, ticker, title, description, source, url
        """
        # TODO: Implement news fetching
        # Options:
        # 1. NewsAPI: requests.get("https://newsapi.org/v2/everything", params={...})
        # 2. Financial Modeling Prep: requests.get(f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}")
        raise NotImplementedError("News fetching not yet implemented. Choose a data source and implement.")

    def compute_sentiment(self, texts: list[str]) -> list[dict]:
        """Compute sentiment scores using FinBERT.

        Returns list of: {label: "positive"|"negative"|"neutral", score: float}
        """
        # TODO: Implement sentiment computation
        # from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        raise NotImplementedError("Sentiment computation not yet implemented.")

    def aggregate_daily_sentiment(
        self,
        articles_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate article-level sentiment to daily ticker-level scores.

        Output columns: date, ticker, sentiment_score (-1 to 1),
                        sentiment_magnitude, article_count
        """
        # TODO: Implement aggregation
        raise NotImplementedError("Sentiment aggregation not yet implemented.")

    def run(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
        upload_to_azure: bool = False,
    ) -> pd.DataFrame:
        """Full pipeline: fetch → embed → aggregate → store.

        Returns daily sentiment DataFrame ready for QC Custom Data.
        """
        # 1. Fetch
        articles = self.fetch_news(tickers, start_date, end_date)

        # 2. Compute sentiment
        texts = (articles["title"] + " " + articles["description"]).tolist()
        sentiments = self.compute_sentiment(texts)
        articles["sentiment_label"] = [s["label"] for s in sentiments]
        articles["sentiment_score"] = [s["score"] for s in sentiments]

        # 3. Aggregate
        daily = self.aggregate_daily_sentiment(articles)

        # 4. Store
        if upload_to_azure:
            from utils.data_loaders import upload_to_blob
            blob_path = f"custom_data/news_sentiment/{start_date}_{end_date}.csv"
            upload_to_blob(daily, blob_path, container="trading-data-blob")
            logger.info(f"Uploaded to Azure: {blob_path}")

        return daily
