"""Sentiment + Price History Strategy (Hypothesis 4)

QuantConnect algorithm that combines news sentiment with price action
to generate trading signals. Uses Tiingo News data (available in QC)
for sentiment scoring combined with technical price features.

Signal logic:
- Positive sentiment spike + price above key MAs → BUY
- Negative sentiment spike + price below key MAs → avoid/sell
- Sentiment-momentum divergence (price up but sentiment down) → caution

Data sources (all available in QuantConnect):
- Tiingo News: article-level sentiment, word counts, crawl timestamps
- Price: standard OHLCV from QC

This is a standalone strategy (Hypothesis 4) but also feeds into
the Combined Signal strategy when validated.
"""

from AlgorithmImports import *
from QuantConnect.Data.Custom.Tiingo import *
import numpy as np
from collections import defaultdict
from datetime import timedelta


class SentimentPriceAlgorithm(QCAlgorithm):
    """Trade based on news sentiment combined with price history signals."""

    def initialize(self):
        # ── Backtest Settings ──
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        # ── Parameters ──
        self.sentiment_lookback = self.get_parameter("sentiment_lookback", 3)  # days
        self.sentiment_threshold_buy = self.get_parameter("sentiment_buy", 0.3)
        self.sentiment_threshold_sell = self.get_parameter("sentiment_sell", -0.2)
        self.min_articles = self.get_parameter("min_articles", 3)  # min articles for signal
        self.max_positions = self.get_parameter("max_positions", 10)
        self.position_size_pct = self.get_parameter("position_size_pct", 0.05)
        self.holding_period = self.get_parameter("holding_period", 10)  # days

        # ── Universe — liquid large/mid caps ──
        self.target_tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
            "META", "TSLA", "JPM", "BAC", "GS",
            "JNJ", "UNH", "PFE", "XOM", "CVX",
            "WMT", "KO", "DIS", "CAT", "BA",
        ]

        # ── Add equities + Tiingo news data ──
        self.symbols = {}
        self.news_symbols = {}
        self.indicators = {}

        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol

            # Add Tiingo News data for each ticker
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

            # Price indicators
            self.indicators[ticker] = {
                "sma_20": self.sma(equity.symbol, 20),
                "sma_50": self.sma(equity.symbol, 50),
                "rsi_14": self.rsi(equity.symbol, 14),
                "ema_12": self.ema(equity.symbol, 12),
                "ema_26": self.ema(equity.symbol, 26),
            }

        # ── Sentiment tracking ──
        # Store recent articles per ticker: list of (date, sentiment_score)
        self.sentiment_history = defaultdict(list)

        # ── Position tracking ──
        self.position_exit_dates = {}  # symbol -> exit_date
        self.position_reasons = {}    # symbol -> reason string

        # ── Benchmark ──
        self.set_benchmark("SPY")

        # ── Scheduled rebalance ──
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.evaluate_signals,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 10),
            self.check_exits,
        )

        self.log(f"Sentiment + Price Strategy initialized: "
                 f"{len(self.target_tickers)} stocks, "
                 f"sentiment_buy={self.sentiment_threshold_buy}, "
                 f"sentiment_sell={self.sentiment_threshold_sell}, "
                 f"min_articles={self.min_articles}")

    def on_data(self, data: Slice):
        """Process incoming data — capture news sentiment."""
        for ticker in self.target_tickers:
            news_symbol = self.news_symbols[ticker]
            if data.contains_key(news_symbol):
                article = data[news_symbol]
                # Tiingo provides article-level data
                # We compute a simple sentiment score from the article metadata
                sentiment = self._score_article(article)
                self.sentiment_history[ticker].append({
                    "date": self.time,
                    "score": sentiment,
                    "title": getattr(article, "title", ""),
                })

                # Keep only recent history
                cutoff = self.time - timedelta(days=self.sentiment_lookback * 2)
                self.sentiment_history[ticker] = [
                    s for s in self.sentiment_history[ticker]
                    if s["date"] > cutoff
                ]

    def _score_article(self, article) -> float:
        """Score a single news article for sentiment.

        Tiingo News provides crawl data. We use a keyword-based scoring
        as a baseline. For production, replace with FinBERT or similar.

        Returns: float between -1.0 (very negative) and 1.0 (very positive)
        """
        title = getattr(article, "title", "").lower()
        description = getattr(article, "description", "").lower()
        text = f"{title} {description}"

        # Positive keywords
        positive = [
            "beat", "beats", "exceeds", "surpass", "upgrade", "upgraded",
            "buy", "outperform", "strong", "growth", "record", "surge",
            "rally", "bullish", "profit", "gain", "revenue beat",
            "raises guidance", "raises forecast", "positive",
            "above expectations", "better than expected",
            "innovation", "breakthrough", "partnership", "deal",
        ]

        # Negative keywords
        negative = [
            "miss", "misses", "below", "downgrade", "downgraded",
            "sell", "underperform", "weak", "decline", "loss",
            "plunge", "bearish", "warning", "cuts guidance",
            "lowers forecast", "negative", "below expectations",
            "worse than expected", "lawsuit", "investigation",
            "recall", "layoff", "layoffs", "bankruptcy", "fraud",
        ]

        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)
        total = pos_count + neg_count

        if total == 0:
            return 0.0

        # Normalize to [-1, 1]
        return (pos_count - neg_count) / total

    def _get_sentiment_signal(self, ticker: str) -> dict:
        """Compute aggregate sentiment signal for a ticker.

        Returns dict with:
        - score: weighted average sentiment (-1 to 1)
        - article_count: number of articles in lookback
        - trend: sentiment trend (improving/declining/flat)
        """
        cutoff = self.time - timedelta(days=self.sentiment_lookback)
        recent = [s for s in self.sentiment_history.get(ticker, []) if s["date"] > cutoff]

        if not recent:
            return {"score": 0.0, "article_count": 0, "trend": "flat"}

        # Time-weighted average (more recent = higher weight)
        scores = []
        weights = []
        for i, article in enumerate(recent):
            days_ago = (self.time - article["date"]).total_seconds() / 86400
            weight = 1.0 / (1.0 + days_ago)  # exponential decay
            scores.append(article["score"])
            weights.append(weight)

        weighted_score = np.average(scores, weights=weights)

        # Trend: compare first half vs second half
        mid = len(scores) // 2
        if mid > 0:
            first_half = np.mean(scores[:mid])
            second_half = np.mean(scores[mid:])
            if second_half > first_half + 0.1:
                trend = "improving"
            elif second_half < first_half - 0.1:
                trend = "declining"
            else:
                trend = "flat"
        else:
            trend = "flat"

        return {
            "score": weighted_score,
            "article_count": len(recent),
            "trend": trend,
        }

    def _get_price_signal(self, ticker: str) -> dict:
        """Compute price-based signal for a ticker.

        Returns dict with:
        - trend: bullish/bearish/neutral based on MA alignment
        - momentum: RSI-based momentum score
        - macd_signal: MACD crossover signal
        """
        ind = self.indicators[ticker]

        if not all(v.is_ready for v in ind.values()):
            return {"trend": "neutral", "momentum": 50, "macd_signal": 0}

        symbol = self.symbols[ticker]
        price = self.securities[symbol].price
        sma20 = ind["sma_20"].current.value
        sma50 = ind["sma_50"].current.value
        rsi_val = ind["rsi_14"].current.value
        ema12 = ind["ema_12"].current.value
        ema26 = ind["ema_26"].current.value

        # Trend: price vs MAs
        if price > sma20 > sma50:
            trend = "bullish"
        elif price < sma20 < sma50:
            trend = "bearish"
        else:
            trend = "neutral"

        # MACD signal
        macd_val = ema12 - ema26
        macd_signal = 1 if macd_val > 0 else -1

        return {
            "trend": trend,
            "momentum": rsi_val,
            "macd_signal": macd_signal,
            "price_vs_sma20": (price / sma20 - 1) if sma20 > 0 else 0,
        }

    def evaluate_signals(self):
        """Evaluate combined sentiment + price signals for all stocks."""
        signals = []

        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]

            # Skip if already in position
            if symbol in self.position_exit_dates:
                continue

            # Get signals
            sentiment = self._get_sentiment_signal(ticker)
            price = self._get_price_signal(ticker)

            # ── Combined Signal Logic ──
            #
            # STRONG BUY:  positive sentiment + bullish trend + improving sentiment
            # BUY:         positive sentiment + neutral/bullish trend
            # AVOID:       negative sentiment OR bearish trend with declining sentiment
            #

            score = 0.0
            reason_parts = []

            # Sentiment component (0 to 1)
            if sentiment["article_count"] >= self.min_articles:
                if sentiment["score"] > self.sentiment_threshold_buy:
                    score += 0.4
                    reason_parts.append(f"sentiment={sentiment['score']:.2f}")
                    if sentiment["trend"] == "improving":
                        score += 0.1
                        reason_parts.append("improving")
                elif sentiment["score"] < self.sentiment_threshold_sell:
                    score -= 0.5
                    reason_parts.append(f"neg_sentiment={sentiment['score']:.2f}")

            # Price component (0 to 1)
            if price["trend"] == "bullish":
                score += 0.3
                reason_parts.append("bullish_trend")
            elif price["trend"] == "bearish":
                score -= 0.3
                reason_parts.append("bearish_trend")

            if price["macd_signal"] > 0:
                score += 0.1
                reason_parts.append("macd_up")

            # RSI filter: avoid overbought
            if price["momentum"] > 70:
                score -= 0.2
                reason_parts.append("overbought")
            elif price["momentum"] < 30:
                score += 0.1
                reason_parts.append("oversold")

            if score > 0.3:
                signals.append({
                    "ticker": ticker,
                    "symbol": symbol,
                    "score": score,
                    "reason": " + ".join(reason_parts),
                    "sentiment": sentiment,
                    "price": price,
                })

        # ── Execute top signals ──
        signals.sort(key=lambda x: x["score"], reverse=True)

        current_positions = sum(
            1 for s in self.symbols.values() if self.portfolio[s].invested
        )

        for signal in signals:
            if current_positions >= self.max_positions:
                break

            symbol = signal["symbol"]
            ticker = signal["ticker"]

            # Position size
            allocation = self.portfolio.total_portfolio_value * self.position_size_pct
            price = self.securities[symbol].price
            if price <= 0:
                continue

            quantity = int(allocation / price)
            if quantity <= 0:
                continue

            # Execute
            self.market_order(symbol, quantity)
            exit_date = self.time + timedelta(days=self.holding_period)
            self.position_exit_dates[symbol] = exit_date
            self.position_reasons[symbol] = signal["reason"]
            current_positions += 1

            self.log(
                f"BUY {ticker}: {quantity} shares @ ${price:.2f} "
                f"(score={signal['score']:.2f}, {signal['reason']}) "
                f"— exit by {exit_date.strftime('%Y-%m-%d')}"
            )

    def check_exits(self):
        """Exit positions at holding period or on negative sentiment shift."""
        to_remove = []

        for symbol, exit_date in self.position_exit_dates.items():
            should_exit = False
            exit_reason = ""

            # Time-based exit
            if self.time >= exit_date:
                should_exit = True
                exit_reason = "holding_period"

            # Early exit on sentiment reversal
            ticker = None
            for t, s in self.symbols.items():
                if s == symbol:
                    ticker = t
                    break

            if ticker:
                sentiment = self._get_sentiment_signal(ticker)
                if (sentiment["article_count"] >= self.min_articles and
                        sentiment["score"] < self.sentiment_threshold_sell and
                        sentiment["trend"] == "declining"):
                    should_exit = True
                    exit_reason = f"sentiment_reversal({sentiment['score']:.2f})"

            if should_exit and self.portfolio[symbol].invested:
                pnl = self.portfolio[symbol].unrealized_profit
                pnl_pct = self.portfolio[symbol].unrealized_profit_percent
                self.liquidate(symbol)
                self.log(
                    f"EXIT {symbol.value}: {exit_reason} — "
                    f"P&L: ${pnl:,.2f} ({pnl_pct:.2%}) — "
                    f"entry reason: {self.position_reasons.get(symbol, 'unknown')}"
                )
                to_remove.append(symbol)

        for s in to_remove:
            del self.position_exit_dates[s]
            self.position_reasons.pop(s, None)

    def on_end_of_algorithm(self):
        """Log final summary."""
        self.log(f"\n{'='*50}")
        self.log(f"SENTIMENT + PRICE STRATEGY RESULTS")
        self.log(f"{'='*50}")
        self.log(f"Total Return: {self.portfolio.total_profit / 100_000:.2%}")
        self.log(f"Final Value: ${self.portfolio.total_portfolio_value:,.2f}")
        self.log(f"{'='*50}\n")
