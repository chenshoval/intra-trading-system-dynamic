"""Intraday Sentiment + Price Strategy (Hypothesis 4)

Combines Tiingo News sentiment with price action for intraday signals.
Buys on positive sentiment + bullish price action, exits on sentiment reversal
or holding period. More active than congressional strategy.

Data: Tiingo News (in QC DataSource) + standard OHLCV
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict


class SentimentPriceAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        # ── Parameters ──
        self.holding_period = 5            # days (shorter = more intraday-like)
        self.sentiment_lookback = 2        # days of news to consider
        self.sentiment_buy_threshold = 0.3
        self.max_positions = 10
        self.position_size_pct = 0.08      # 8% per position
        self.max_total_exposure = 0.85

        # ── Universe: liquid large caps ──
        self.target_tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
            "META", "TSLA", "JPM", "BAC", "GS",
            "JNJ", "UNH", "XOM", "CVX", "WMT",
            "KO", "DIS", "CAT", "BA", "NFLX",
        ]

        self.symbols = {}
        self.indicators = {}

        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol

            # Add Tiingo News for each ticker
            self.add_data(TiingoNews, ticker)

            # Technical indicators
            self.indicators[ticker] = {
                "sma20": self.sma(equity.symbol, 20),
                "sma50": self.sma(equity.symbol, 50),
                "rsi": self.rsi(equity.symbol, 14),
            }

        # ── Sentiment tracking ──
        self.news_scores = defaultdict(list)  # ticker -> [(date, score)]
        self.position_exit_dates = {}
        self.total_signals = 0
        self.total_trades = 0

        # ── Benchmark ──
        self.set_benchmark("SPY")

        # ── Positive and negative keywords for scoring ──
        self.positive_words = {
            "beat", "beats", "exceeds", "surpass", "upgrade", "upgraded",
            "outperform", "strong", "growth", "record", "surge", "rally",
            "bullish", "profit", "gain", "raises guidance", "positive",
            "above expectations", "better than expected", "innovation",
            "breakthrough", "partnership", "deal", "buy", "overweight",
        }
        self.negative_words = {
            "miss", "misses", "below", "downgrade", "downgraded",
            "underperform", "weak", "decline", "loss", "plunge", "bearish",
            "warning", "cuts guidance", "lowers forecast", "negative",
            "below expectations", "worse than expected", "lawsuit",
            "investigation", "recall", "layoff", "layoffs", "fraud",
            "sell", "underweight",
        }

        # ── Schedule ──
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.evaluate_and_trade,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 10),
            self.check_exits,
        )

    def on_data(self, data):
        """Capture Tiingo News as it arrives."""
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            # Check for TiingoNews data
            if data.contains_key(symbol):
                news_data = data.get(TiingoNews)
                if news_data is not None:
                    for article in news_data:
                        score = self._score_article(article)
                        self.news_scores[ticker].append({
                            "date": self.time,
                            "score": score,
                        })

        # Clean old news (keep only recent)
        cutoff = self.time - timedelta(days=self.sentiment_lookback * 3)
        for ticker in self.news_scores:
            self.news_scores[ticker] = [
                n for n in self.news_scores[ticker] if n["date"] > cutoff
            ]

    def _score_article(self, article):
        """Score a news article for sentiment. Returns -1 to 1."""
        title = str(getattr(article, "title", "")).lower()
        desc = str(getattr(article, "description", "")).lower()
        text = f"{title} {desc}"

        pos = sum(1 for w in self.positive_words if w in text)
        neg = sum(1 for w in self.negative_words if w in text)
        total = pos + neg

        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _get_sentiment(self, ticker):
        """Get aggregate sentiment for a ticker over lookback period."""
        cutoff = self.time - timedelta(days=self.sentiment_lookback)
        recent = [n for n in self.news_scores.get(ticker, []) if n["date"] > cutoff]

        if len(recent) < 2:
            return 0.0, 0  # not enough data

        scores = [n["score"] for n in recent]
        # Time-weighted: more recent = higher weight
        weights = []
        for n in recent:
            days_ago = max((self.time - n["date"]).total_seconds() / 86400, 0.1)
            weights.append(1.0 / days_ago)

        import numpy as np
        avg = np.average(scores, weights=weights) if weights else 0.0
        return float(avg), len(recent)

    def _get_price_signal(self, ticker):
        """Get price-based signal: bullish, bearish, or neutral."""
        ind = self.indicators[ticker]
        if not all(v.is_ready for v in ind.values()):
            return "neutral", 50.0

        symbol = self.symbols[ticker]
        price = self.securities[symbol].price
        sma20 = ind["sma20"].current.value
        sma50 = ind["sma50"].current.value
        rsi_val = ind["rsi"].current.value

        if price > sma20 > sma50:
            trend = "bullish"
        elif price < sma20 < sma50:
            trend = "bearish"
        else:
            trend = "neutral"

        return trend, rsi_val

    def evaluate_and_trade(self):
        """Score all stocks and trade the best signals."""
        candidates = []

        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]

            # Skip if already in position
            if symbol in self.position_exit_dates:
                continue

            sentiment, article_count = self._get_sentiment(ticker)
            trend, rsi_val = self._get_price_signal(ticker)

            # Combined score
            score = 0.0

            # Sentiment component
            if sentiment > self.sentiment_buy_threshold and article_count >= 2:
                score += 0.5
            elif sentiment < -0.2:
                score -= 0.5

            # Price component
            if trend == "bullish":
                score += 0.3
            elif trend == "bearish":
                score -= 0.3

            # RSI: avoid overbought
            if rsi_val > 70:
                score -= 0.2
            elif rsi_val < 30:
                score += 0.2

            if score >= 0.5:
                self.total_signals += 1
                candidates.append((ticker, symbol, score, sentiment, trend))

        # Sort by score, take best
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Execute
        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        current_exposure = sum(
            abs(h.holdings_value) for h in self.portfolio.values()
            if h.invested
        ) / total_value

        for ticker, symbol, score, sentiment, trend in candidates:
            if len(self.position_exit_dates) >= self.max_positions:
                break
            if current_exposure + self.position_size_pct > self.max_total_exposure:
                break

            price = self.securities[symbol].price
            if price <= 0:
                continue

            allocation = total_value * self.position_size_pct
            quantity = int(allocation / price)
            if quantity <= 0:
                continue

            self.market_order(symbol, quantity)
            self.position_exit_dates[symbol] = self.time + timedelta(days=self.holding_period)
            current_exposure += self.position_size_pct
            self.total_trades += 1

    def check_exits(self):
        """Exit on holding period or sentiment reversal."""
        to_remove = []
        for symbol, exit_date in self.position_exit_dates.items():
            should_exit = self.time >= exit_date

            # Early exit on strong negative sentiment
            if not should_exit:
                ticker = None
                for t, s in self.symbols.items():
                    if s == symbol:
                        ticker = t
                        break
                if ticker:
                    sentiment, count = self._get_sentiment(ticker)
                    if count >= 2 and sentiment < -0.3:
                        should_exit = True

            if should_exit and self.portfolio[symbol].invested:
                self.liquidate(symbol)
                to_remove.append(symbol)

        for s in to_remove:
            del self.position_exit_dates[s]

    def on_end_of_algorithm(self):
        self.debug(f"SENTIMENT RESULTS: Return={self.portfolio.total_profit / 100_000:.2%} "
                   f"Signals={self.total_signals} Trades={self.total_trades} "
                   f"Final=${self.portfolio.total_portfolio_value:,.0f}")
