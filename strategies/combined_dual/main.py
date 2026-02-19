"""Combined Dual-Engine Strategy v1

Two uncorrelated alpha streams in one algorithm, with a shared SPY
trend gate that fully stops new entries during downtrends.

Engine 1 — Event Reactor (tweaked v4):
  Same proven event signals (news, gaps) but with:
  - Earnings hold extended to 15 days (capture PEAD drift)
  - NO new event trades when SPY MA(10) < MA(50) (full stop, not half-size)
  - Fixed 3% stop, fixed 4% sizing — same as the baseline that worked

Engine 2 — Cross-Sectional Momentum (NEW):
  Monthly rebalance, buy top 10 stocks by 6-month return (skip last month).
  - Equal-weight long-only positions
  - NO rebalance into new positions during SPY downtrends
  - Entirely uncorrelated to event signals

Capital split: 50% events, 50% momentum.
Each engine manages its own slice independently.

Why this beats our previous attempts:
  - v5 model FILTER killed winners → here we ADD alpha instead of shaving it
  - Trend+Events v1 SIZED down but still traded → here we FULLY STOP new entries
  - Two uncorrelated streams → combined Sharpe > either alone
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class CombinedDualEngine(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Capital allocation ──
        self.event_capital_pct = 0.50       # 50% to event engine
        self.momentum_capital_pct = 0.50    # 50% to momentum engine

        # ── Shared: SPY trend gate ──
        self.trend_fast_period = 10
        self.trend_slow_period = 50

        # ── Event engine parameters (matches v4 baseline exactly) ──
        self.event_max_positions = 10       # half of 20 since half capital
        self.event_position_pct = 0.04      # 4% of event capital
        self.event_max_exposure = 0.90      # of event capital
        self.event_stop_loss_pct = 0.03     # fixed 3% — same as baseline
        self.min_gap_pct = 0.05
        self.min_volume_ratio = 2.0

        # ── Momentum engine parameters ──
        self.mom_lookback = 126             # ~6 months trading days
        self.mom_skip_recent = 21           # skip last month (reversal)
        self.mom_top_n = 10                 # buy top 10
        self.mom_rebalance_day = 1          # first trading day of month

        # ── 50-stock universe ──
        self.target_tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO",
            "CRM", "ADBE", "ORCL", "AMD", "QCOM", "NFLX", "INTU",
            "JPM", "GS", "MS", "V", "MA", "AXP", "BLK",
            "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "ISRG",
            "COST", "HD", "MCD", "NKE", "PG", "KO", "PEP",
            "CAT", "HON", "UPS", "GE", "LMT", "RTX",
            "XOM", "CVX", "COP",
            "DIS", "CMCSA",
            "PYPL", "SQ", "ABNB", "UBER",
        ]

        # ── Data structures ──
        self.symbols = {}
        self.news_symbols = {}
        self.prev_close = {}
        self.volume_sma = {}

        # Event engine state
        self.event_positions = {}           # symbol -> exit_date
        self.event_entry_prices = {}        # symbol -> entry_price
        self.event_trade_source = {}        # symbol -> "event"

        # Momentum engine state
        self.mom_holdings = set()            # symbols currently held by momentum
        self.mom_last_rebalance = None

        # ── Counters ──
        self.event_signals = 0
        self.event_trades = 0
        self.event_stopped = 0
        self.event_skipped_trend = 0
        self.mom_rebalances = 0
        self.mom_trades = 0
        self.mom_skipped_trend = 0
        self.event_counts = defaultdict(int)

        # ── Add equities + news ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        # ── SPY for trend gate ──
        self.add_equity("SPY", Resolution.DAILY)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast_period, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow_period, Resolution.DAILY)

        # ── Event patterns (same as v4, but earnings hold extended) ──
        self.event_patterns = {
            "earnings_beat": {
                "keywords": [
                    "beats estimates", "tops estimates", "beats expectations",
                    "better than expected", "earnings beat", "revenue beat",
                    "profit beats", "eps beats", "blowout quarter",
                    "strong quarter", "record quarter", "record earnings",
                    "record revenue", "exceeds expectations",
                ],
                "hold_days": 15,            # CHANGED: 5 -> 15 (capture PEAD)
            },
            "analyst_upgrade": {
                "keywords": [
                    "upgraded to buy", "upgraded to outperform",
                    "upgraded to overweight", "price target raised",
                    "price target increased", "raises price target",
                    "initiates with buy", "initiates with outperform",
                ],
                "hold_days": 3,
            },
            "guidance_raise": {
                "keywords": [
                    "raises guidance", "raises outlook", "raises forecast",
                    "boosts guidance", "boosts outlook", "increases guidance",
                    "above prior guidance", "upside guidance",
                ],
                "hold_days": 4,
            },
        }

        # ── Benchmark + scheduling ──
        self.set_benchmark("SPY")

        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.event_scan_for_gaps,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 60),
            self.event_check_stop_losses,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 10),
            self.event_check_exits,
        )
        # Momentum rebalance: first trading day of each month
        self.schedule.on(
            self.date_rules.month_start("SPY", 0),
            self.time_rules.after_market_open("SPY", 45),
            self.momentum_rebalance,
        )

        self.debug(
            f">>> COMBINED DUAL-ENGINE v1: {len(self.target_tickers)} tickers, "
            f"event_cap={self.event_capital_pct:.0%} mom_cap={self.momentum_capital_pct:.0%} "
            f"trend={self.trend_fast_period}/{self.trend_slow_period} MA"
        )

    # ══════════════════════════════════════════════════════════════════════
    # SHARED: Trend gate
    # ══════════════════════════════════════════════════════════════════════

    def _is_uptrend(self):
        """SPY fast MA > slow MA = uptrend. Full stop on new entries in downtrend."""
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True  # warming up, allow trading
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    def _get_event_capital(self):
        """Capital available for event engine."""
        return self.portfolio.total_portfolio_value * self.event_capital_pct

    def _get_momentum_capital(self):
        """Capital available for momentum engine."""
        return self.portfolio.total_portfolio_value * self.momentum_capital_pct

    # ══════════════════════════════════════════════════════════════════════
    # ENGINE 1: Event Reactor (tweaked v4 — no model, no filters)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        """Scan news for event signals."""
        for ticker in self.target_tickers:
            news_symbol = self.news_symbols[ticker]
            if not data.contains_key(news_symbol):
                continue

            article = data[news_symbol]
            title = str(getattr(article, "title", "")).lower()
            desc = str(getattr(article, "description", "")).lower()
            text = f"{title} {desc}"

            symbol = self.symbols[ticker]
            # Skip if already in an event position
            if symbol in self.event_positions:
                continue

            for event_type, config in self.event_patterns.items():
                matches = sum(1 for kw in config["keywords"] if kw in text)
                if matches >= 1:
                    self.event_signals += 1
                    self.event_counts[event_type] += 1
                    self._event_execute(symbol, ticker, config["hold_days"])
                    break

        # Update price/volume tracking for gap scanner
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            if symbol in self.securities and self.securities[symbol].price > 0:
                price = self.securities[symbol].price
                volume = self.securities[symbol].volume
                if symbol not in self.volume_sma:
                    self.volume_sma[symbol] = volume
                else:
                    self.volume_sma[symbol] = (
                        0.95 * self.volume_sma[symbol] + 0.05 * volume
                    )
                self.prev_close[symbol] = price

    def event_scan_for_gaps(self):
        """Scan for gap-up events at market open."""
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            if symbol in self.event_positions:
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
                    self.event_signals += 1
                    self.event_counts["price_gap"] += 1
                    self._event_execute(symbol, ticker, 5)

    def _event_execute(self, symbol, ticker, hold_days):
        """Execute an event trade — full stop in downtrends."""
        # ── TREND GATE: no new event trades in downtrend ──
        if not self._is_uptrend():
            self.event_skipped_trend += 1
            return

        if len(self.event_positions) >= self.event_max_positions:
            return

        event_capital = self._get_event_capital()
        if event_capital <= 0:
            return

        # Check event-engine exposure
        event_invested = sum(
            abs(self.portfolio[s].holdings_value)
            for s in self.event_positions
            if self.portfolio[s].invested
        )
        if event_capital > 0 and (event_invested / event_capital) > self.event_max_exposure:
            return

        price = self.securities[symbol].price
        if price <= 0:
            return

        alloc = event_capital * self.event_position_pct
        quantity = int(alloc / price)
        if quantity < 1:
            return

        self.market_order(symbol, quantity)
        self.event_positions[symbol] = self.time + timedelta(days=hold_days)
        self.event_entry_prices[symbol] = price
        self.event_trade_source[symbol] = "event"
        self.event_trades += 1

    def event_check_stop_losses(self):
        """Fixed 3% stop loss — same as baseline v4."""
        to_remove = []
        for symbol, entry_price in list(self.event_entry_prices.items()):
            if symbol not in self.securities:
                continue
            current_price = self.securities[symbol].price
            if current_price <= 0 or entry_price <= 0:
                continue

            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct <= -self.event_stop_loss_pct:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.event_stopped += 1
                to_remove.append(symbol)

        for s in to_remove:
            self.event_positions.pop(s, None)
            self.event_entry_prices.pop(s, None)
            self.event_trade_source.pop(s, None)

    def event_check_exits(self):
        """Time-based exit for event trades."""
        to_remove = []
        for symbol, exit_date in self.event_positions.items():
            if self.time >= exit_date:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                to_remove.append(symbol)

        for s in to_remove:
            del self.event_positions[s]
            self.event_entry_prices.pop(s, None)
            self.event_trade_source.pop(s, None)

    # ══════════════════════════════════════════════════════════════════════
    # ENGINE 2: Cross-Sectional Momentum
    # ══════════════════════════════════════════════════════════════════════

    def momentum_rebalance(self):
        """Monthly rebalance: buy top 10 by 6-month momentum (skip last month)."""
        self.mom_rebalances += 1

        # ── TREND GATE: don't buy into new momentum positions in downtrend ──
        in_uptrend = self._is_uptrend()

        # Step 1: Compute momentum scores
        momentum_scores = {}
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            # Need enough history
            history = self.history(
                symbol, self.mom_lookback + self.mom_skip_recent, Resolution.DAILY
            )
            if history is None or history.empty:
                continue
            if len(history) < self.mom_lookback + self.mom_skip_recent:
                continue

            try:
                closes = history["close"].values
                # 6-month return, skipping the most recent month
                price_6m_ago = closes[0]                        # ~6 months ago
                price_1m_ago = closes[-self.mom_skip_recent]    # ~1 month ago

                if price_6m_ago > 0:
                    mom_return = (price_1m_ago / price_6m_ago) - 1.0
                    momentum_scores[ticker] = mom_return
            except Exception:
                continue

        if len(momentum_scores) < self.mom_top_n:
            return

        # Step 2: Rank and select top N
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [t for t, _ in ranked[:self.mom_top_n]]
        top_symbols = {self.symbols[t] for t in top_tickers}

        # Step 3: Sell positions no longer in top N
        for symbol in list(self.mom_holdings):
            if symbol not in top_symbols:
                if self.portfolio[symbol].invested:
                    # Don't sell if it's also an event position
                    if symbol not in self.event_positions:
                        self.liquidate(symbol)
                        self.mom_trades += 1
                self.mom_holdings.discard(symbol)

        # Step 4: Buy new positions (only in uptrend)
        if not in_uptrend:
            self.mom_skipped_trend += 1
            return

        mom_capital = self._get_momentum_capital()
        if mom_capital <= 0:
            return

        target_alloc = mom_capital / self.mom_top_n  # equal-weight

        for ticker in top_tickers:
            symbol = self.symbols[ticker]
            price = self.securities[symbol].price
            if price <= 0:
                continue

            # Skip if this stock is currently in an event trade
            if symbol in self.event_positions:
                continue

            target_qty = int(target_alloc / price)
            if target_qty < 1:
                continue

            current_qty = self.portfolio[symbol].quantity
            delta = target_qty - current_qty

            if delta > 0:
                self.market_order(symbol, delta)
                self.mom_trades += 1
            elif delta < 0:
                self.market_order(symbol, delta)
                self.mom_trades += 1

            self.mom_holdings.add(symbol)

        self.mom_last_rebalance = self.time

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        total_profit = self.portfolio.total_profit
        ret_pct = total_profit / 100_000

        events_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self.event_counts.items())
        )

        trend_status = "UP" if self._is_uptrend() else "DOWN"

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} "
            f"Final=${self.portfolio.total_portfolio_value:,.0f}"
        )
        self.debug(
            f"EVENT ENGINE: Signals={self.event_signals} "
            f"Trades={self.event_trades} Stopped={self.event_stopped} "
            f"SkippedTrend={self.event_skipped_trend} "
            f"Events=[{events_str}]"
        )
        self.debug(
            f"MOMENTUM ENGINE: Rebalances={self.mom_rebalances} "
            f"Trades={self.mom_trades} "
            f"SkippedTrend={self.mom_skipped_trend} "
            f"CurrentHoldings={len(self.mom_holdings)}"
        )
        self.debug(f"TREND: Final={trend_status}")
