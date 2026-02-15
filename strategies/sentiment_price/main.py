"""Intraday Sentiment + Price Strategy — DEBUG VERSION"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict


class SentimentPriceAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        self.holding_period = 5
        self.sentiment_lookback = 2
        self.sentiment_buy_threshold = 0.2  # was 0.3 — more aggressive
        self.max_positions = 20             # was 10
        self.position_size_pct = 0.04       # smaller per position since more positions
        self.max_total_exposure = 0.90      # was 0.85

        # ── Universe: Top 50 liquid stocks across sectors ──
        self.target_tickers = [
            # Tech
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
            "CRM", "ADBE", "ORCL", "INTC", "AMD", "AVGO", "QCOM",
            # Finance
            "JPM", "BAC", "GS", "MS", "WFC", "BRK.B", "V", "MA",
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO",
            # Consumer
            "WMT", "KO", "PEP", "PG", "COST", "HD", "NKE", "MCD",
            # Energy
            "XOM", "CVX", "COP",
            # Industrial
            "CAT", "BA", "HON", "UPS", "GE",
            # Communication
            "DIS", "CMCSA", "T", "VZ",
            # Other
            "PYPL", "SQ",
        ]

        self.symbols = {}
        self.news_symbols = {}
        self.indicators = {}

        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol

            # Add Tiingo News — store the returned symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol
            self.debug(f"Added TiingoNews for {ticker}: {news.symbol}")

            self.indicators[ticker] = {
                "sma20": self.sma(equity.symbol, 20),
                "sma50": self.sma(equity.symbol, 50),
                "rsi": self.rsi(equity.symbol, 14),
            }

        self.news_scores = defaultdict(list)
        self.position_exit_dates = {}
        self.total_signals = 0
        self.total_trades = 0
        self.data_calls = 0
        self.news_received = 0

        self.set_benchmark("SPY")

        self.positive_words = {
            "beat", "beats", "exceeds", "surpass", "upgrade", "upgraded",
            "outperform", "strong", "growth", "record", "surge", "rally",
            "bullish", "profit", "gain", "positive", "innovation",
            "breakthrough", "partnership", "deal", "buy", "overweight",
        }
        self.negative_words = {
            "miss", "misses", "below", "downgrade", "downgraded",
            "underperform", "weak", "decline", "loss", "plunge", "bearish",
            "warning", "negative", "lawsuit", "investigation", "recall",
            "layoff", "layoffs", "fraud", "sell", "underweight",
        }

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

        self.debug(">>> INIT DONE")

    def on_data(self, data):
        self.data_calls += 1

        # Debug first few calls
        if self.data_calls <= 3:
            self.debug(f">>> on_data call #{self.data_calls}, keys count={len(list(data.keys()))}")

        # Try to get TiingoNews articles
        for ticker in self.target_tickers:
            news_symbol = self.news_symbols[ticker]

            if data.contains_key(news_symbol):
                articles = data[news_symbol]
                if self.news_received < 5:
                    self.debug(f">>> NEWS for {ticker}: type={type(articles).__name__}")

                # TiingoNews might come as a single article or list
                article_list = [articles] if not hasattr(articles, '__iter__') else articles
                for article in article_list:
                    self.news_received += 1
                    score = self._score_article(article)
                    self.news_scores[ticker].append({
                        "date": self.time,
                        "score": score,
                    })

                    if self.news_received <= 5:
                        title = str(getattr(article, "title", ""))[:60]
                        self.debug(f">>> ARTICLE: {ticker} score={score:.2f} '{title}'")

        # Clean old
        if self.data_calls % 100 == 0:
            cutoff = self.time - timedelta(days=self.sentiment_lookback * 3)
            for t in self.news_scores:
                self.news_scores[t] = [n for n in self.news_scores[t] if n["date"] > cutoff]

    def _score_article(self, article):
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
        cutoff = self.time - timedelta(days=self.sentiment_lookback)
        recent = [n for n in self.news_scores.get(ticker, []) if n["date"] > cutoff]
        if len(recent) < 2:
            return 0.0, 0
        scores = [n["score"] for n in recent]
        return sum(scores) / len(scores), len(recent)

    def _get_price_signal(self, ticker):
        ind = self.indicators[ticker]
        if not all(v.is_ready for v in ind.values()):
            return "neutral", 50.0

        symbol = self.symbols[ticker]
        price = self.securities[symbol].price
        sma20 = ind["sma20"].current.value
        sma50 = ind["sma50"].current.value
        rsi_val = ind["rsi"].current.value

        if price > sma20 > sma50:
            return "bullish", rsi_val
        elif price < sma20 < sma50:
            return "bearish", rsi_val
        return "neutral", rsi_val

    def evaluate_and_trade(self):
        candidates = []
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            if symbol in self.position_exit_dates:
                continue

            sentiment, count = self._get_sentiment(ticker)
            trend, rsi_val = self._get_price_signal(ticker)

            score = 0.0
            if sentiment > self.sentiment_buy_threshold and count >= 2:
                score += 0.5
            elif sentiment < -0.2:
                score -= 0.5
            if trend == "bullish":
                score += 0.3
            elif trend == "bearish":
                score -= 0.3
            if rsi_val > 70:
                score -= 0.2
            elif rsi_val < 30:
                score += 0.2

            if score >= 0.5:
                self.total_signals += 1
                candidates.append((ticker, symbol, score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        current_exposure = sum(
            abs(h.holdings_value) for h in self.portfolio.values() if h.invested
        ) / total_value

        for ticker, symbol, score in candidates:
            if len(self.position_exit_dates) >= self.max_positions:
                break
            if current_exposure + self.position_size_pct > self.max_total_exposure:
                break

            price = self.securities[symbol].price
            if price <= 0:
                continue

            quantity = int(total_value * self.position_size_pct / price)
            if quantity <= 0:
                continue

            self.market_order(symbol, quantity)
            self.position_exit_dates[symbol] = self.time + timedelta(days=self.holding_period)
            current_exposure += self.position_size_pct
            self.total_trades += 1

    def check_exits(self):
        to_remove = []
        for symbol, exit_date in self.position_exit_dates.items():
            should_exit = self.time >= exit_date

            if not should_exit:
                for t, s in self.symbols.items():
                    if s == symbol:
                        sentiment, count = self._get_sentiment(t)
                        if count >= 2 and sentiment < -0.3:
                            should_exit = True
                        break

            if should_exit and self.portfolio[symbol].invested:
                self.liquidate(symbol)
                to_remove.append(symbol)

        for s in to_remove:
            del self.position_exit_dates[s]

    def on_end_of_algorithm(self):
        self.debug(f"RESULTS: Return={self.portfolio.total_profit / 100_000:.2%} "
                   f"Signals={self.total_signals} Trades={self.total_trades} "
                   f"NewsReceived={self.news_received} DataCalls={self.data_calls} "
                   f"Final=${self.portfolio.total_portfolio_value:,.0f}")
