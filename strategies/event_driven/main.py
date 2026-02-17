"""Event-Driven Reactor v3 — Fixed quality universe, no dynamic filter

Simplified: Use a pre-selected list of quality S&P 500 stocks.
Scan for price gaps + news events. Buy and hold 3-5 days.

Previous versions failed because dynamic universe + add_data inside
callbacks caused issues. This version uses the same pattern as our
working sentiment strategy (fixed ticker list + add_data in init).
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict


class EventDrivenReactor(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        # ── Parameters ──
        self.max_positions = 15
        self.position_size_pct = 0.06
        self.max_total_exposure = 0.90
        self.min_gap_pct = 0.03
        self.min_volume_ratio = 1.5

        # ── Fixed quality universe: top stocks with high ROE + margins ──
        # Pre-selected: large cap, high quality, diverse sectors
        self.target_tickers = [
            # Tech (high margins, high ROE)
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO",
            "CRM", "ADBE", "ORCL", "AMD", "QCOM", "NFLX", "INTU",
            # Finance
            "JPM", "GS", "MS", "V", "MA", "AXP", "BLK",
            # Healthcare
            "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "ISRG",
            # Consumer
            "COST", "HD", "MCD", "NKE", "PG", "KO", "PEP",
            # Industrial
            "CAT", "HON", "UPS", "GE", "LMT", "RTX",
            # Energy
            "XOM", "CVX", "COP",
            # Communication
            "DIS", "CMCSA",
            # Other
            "PYPL", "SQ", "ABNB", "UBER",
        ]

        self.symbols = {}
        self.news_symbols = {}
        self.prev_close = {}
        self.volume_sma = {}
        self.position_exit_dates = {}
        self.total_signals = 0
        self.total_trades = 0
        self.data_calls = 0
        self.news_events = 0
        self.gap_events = 0
        self.event_counts = defaultdict(int)

        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol

            # Add Tiingo News
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        # ── Event patterns ──
        self.event_patterns = {
            "earnings_beat": {
                "keywords": ["beats estimates", "tops estimates", "beats expectations",
                             "better than expected", "earnings beat", "revenue beat",
                             "profit beats", "eps beats", "blowout quarter",
                             "strong quarter", "record quarter", "record earnings",
                             "record revenue", "exceeds expectations"],
                "hold_days": 5,
            },
            "analyst_upgrade": {
                "keywords": ["upgraded to buy", "upgraded to outperform",
                             "upgraded to overweight", "price target raised",
                             "price target increased", "raises price target",
                             "initiates with buy", "initiates with outperform"],
                "hold_days": 3,
            },
            "guidance_raise": {
                "keywords": ["raises guidance", "raises outlook", "raises forecast",
                             "boosts guidance", "boosts outlook", "increases guidance",
                             "above prior guidance", "upside guidance"],
                "hold_days": 4,
            },
        }

        self.set_benchmark("SPY")

        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.scan_for_gaps,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 10),
            self.check_exits,
        )

        self.debug(f">>> INIT DONE: {len(self.target_tickers)} tickers")

    def on_data(self, data):
        """Process news + track prices."""
        self.data_calls += 1

        # ── News event detection ──
        for ticker in self.target_tickers:
            news_symbol = self.news_symbols[ticker]
            if not data.contains_key(news_symbol):
                continue

            article = data[news_symbol]
            title = str(getattr(article, "title", "")).lower()
            desc = str(getattr(article, "description", "")).lower()
            text = f"{title} {desc}"

            symbol = self.symbols[ticker]
            if symbol in self.position_exit_dates:
                continue

            for event_type, config in self.event_patterns.items():
                matches = sum(1 for kw in config["keywords"] if kw in text)
                if matches >= 1:
                    self.news_events += 1
                    self.total_signals += 1
                    self.event_counts[event_type] += 1
                    self._execute_trade(symbol, ticker, event_type,
                                       config["hold_days"])
                    break

        # ── Track previous close + volume ──
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            if symbol in self.securities and self.securities[symbol].price > 0:
                price = self.securities[symbol].price
                volume = self.securities[symbol].volume
                if symbol not in self.volume_sma:
                    self.volume_sma[symbol] = volume
                else:
                    self.volume_sma[symbol] = 0.95 * self.volume_sma[symbol] + 0.05 * volume
                self.prev_close[symbol] = price

    def scan_for_gaps(self):
        """Detect price gaps at market open."""
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            if symbol in self.position_exit_dates:
                continue
            if symbol not in self.prev_close:
                continue

            price = self.securities[symbol].price
            prev = self.prev_close[symbol]
            if price <= 0 or prev <= 0:
                continue

            gap_pct = (price - prev) / prev
            vol_avg = self.volume_sma.get(symbol, 0)
            current_vol = self.securities[symbol].volume

            if gap_pct >= self.min_gap_pct and vol_avg > 0:
                vol_ratio = current_vol / vol_avg
                if vol_ratio >= self.min_volume_ratio:
                    self.gap_events += 1
                    self.total_signals += 1
                    self.event_counts["price_gap"] += 1
                    self._execute_trade(symbol, ticker, "price_gap", 5)

    def _execute_trade(self, symbol, ticker, event_type, hold_days):
        if len(self.position_exit_dates) >= self.max_positions:
            return

        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        current_exposure = sum(
            abs(h.holdings_value) for h in self.portfolio.values() if h.invested
        ) / total_value

        if current_exposure + self.position_size_pct > self.max_total_exposure:
            return

        price = self.securities[symbol].price
        if price <= 0:
            return

        quantity = round(total_value * self.position_size_pct / price, 4)
        if quantity < 0.001:
            return

        self.market_order(symbol, quantity)
        self.position_exit_dates[symbol] = self.time + timedelta(days=hold_days)
        self.total_trades += 1

    def check_exits(self):
        to_remove = []
        for symbol, exit_date in self.position_exit_dates.items():
            if self.time >= exit_date:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                to_remove.append(symbol)
        for s in to_remove:
            del self.position_exit_dates[s]

    def on_end_of_algorithm(self):
        events_str = ", ".join(f"{k}={v}" for k, v in sorted(self.event_counts.items()))
        self.debug(f"RESULTS: Return={self.portfolio.total_profit / 100_000:.2%} "
                   f"Signals={self.total_signals} Trades={self.total_trades} "
                   f"NewsEvents={self.news_events} GapEvents={self.gap_events} "
                   f"DataCalls={self.data_calls} "
                   f"Final=${self.portfolio.total_portfolio_value:,.0f} "
                   f"Events=[{events_str}]")
