"""Trend + Events Strategy v1

Combines event-driven trade signals (news events, gap-ups) with a SPY
trend regime overlay, seasonal position scaling, and ATR-adaptive stops.

Design principles (from 5 research papers):
- Events detect WHICH stocks to trade (micro signal)
- SPY MA crossover determines WHEN to be aggressive (macro regime)
- Position sizing adapts to regime + seasonality — never filter trades
- ATR-based stops adapt to per-stock volatility
- Edge is in the tails — let winners run, cut losers via adaptive stops

Key differences from Event-Driven Reactor v4/v5:
- SPY 10/50 MA trend overlay scales position sizes in downtrends (0.5x)
- September positions halved, November positions boosted (1.25x)
- ATR(14)-based stops instead of fixed 3% — clamped to [2%, 5%]
- No ML model — pure rule-based, no feedback loops
- Enhanced end-of-algo logging with regime and seasonal breakdowns
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict


class TrendEventsStrategy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        # ── Parameters (all configurable, no hardcoded values) ──
        self.max_positions = 20
        self.base_position_pct = 0.04       # 4% base position size
        self.max_total_exposure = 0.90
        self.stop_loss_atr_mult = 1.5       # stop at 1.5x ATR below entry
        self.fallback_stop_pct = 0.03       # fallback if ATR not ready
        self.trend_fast_period = 10         # SPY fast MA
        self.trend_slow_period = 50         # SPY slow MA
        self.downtrend_scale = 0.5          # half size in downtrend
        self.september_scale = 0.5          # half size in September
        self.november_scale = 1.25          # extra size in November
        self.min_gap_pct = 0.05             # 5% gap threshold
        self.min_volume_ratio = 2.0         # 2x avg volume threshold

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
        self.position_exit_dates = {}
        self.entry_prices = {}
        self.stop_levels = {}               # per-position ATR-based stop %
        self.atr_indicators = {}

        # ── Counters for logging ──
        self.total_signals = 0
        self.total_trades = 0
        self.stopped_out = 0
        self.data_calls = 0
        self.news_events = 0
        self.gap_events = 0
        self.event_counts = defaultdict(int)

        # ── Regime tracking ──
        self.trades_in_uptrend = 0
        self.trades_in_downtrend = 0
        self.pnl_in_uptrend = 0.0
        self.pnl_in_downtrend = 0.0
        self.monthly_trades = defaultdict(int)
        self.monthly_pnl = defaultdict(float)
        self.trade_regime_at_entry = {}     # symbol -> "up" or "down"

        # ── Add equities + news ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol
            # Register ATR(14) for each stock
            self.atr_indicators[ticker] = self.atr(equity.symbol, 14)

        # ── SPY for trend regime ──
        spy_equity = self.add_equity("SPY", Resolution.DAILY)
        self.spy_symbol = spy_equity.symbol
        self.spy_fast_ma = self.sma("SPY", self.trend_fast_period, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow_period, Resolution.DAILY)

        # ── Event patterns ──
        self.event_patterns = {
            "earnings_beat": {
                "keywords": [
                    "beats estimates", "tops estimates", "beats expectations",
                    "better than expected", "earnings beat", "revenue beat",
                    "profit beats", "eps beats", "blowout quarter",
                    "strong quarter", "record quarter", "record earnings",
                    "record revenue", "exceeds expectations",
                ],
                "hold_days": 5,
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
            self.scan_for_gaps,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 60),
            self.check_stop_losses,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 10),
            self.check_exits,
        )

        regime_status = "WARMING UP"
        self.debug(
            f">>> TREND+EVENTS v1: {len(self.target_tickers)} tickers, "
            f"trend={self.trend_fast_period}/{self.trend_slow_period} MA, "
            f"regime={regime_status}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Regime & sizing helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_regime_scale(self):
        """Scale positions based on SPY trend regime (MA crossover)."""
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return 1.0  # indicators warming up, trade normally
        if self.spy_fast_ma.current.value > self.spy_slow_ma.current.value:
            return 1.0  # uptrend — full size
        return self.downtrend_scale  # downtrend — reduced size

    def _is_uptrend(self):
        """Check if SPY is in uptrend."""
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    def _get_seasonal_scale(self):
        """Scale positions based on month-of-year seasonality."""
        month = self.time.month
        if month == 9:
            return self.september_scale     # 0.5
        elif month == 11:
            return self.november_scale      # 1.25
        return 1.0

    def _get_atr_stop(self, ticker):
        """Compute ATR-based stop loss percentage for a stock."""
        atr_ind = self.atr_indicators.get(ticker)
        if atr_ind is None or not atr_ind.is_ready:
            return self.fallback_stop_pct

        atr_value = atr_ind.current.value
        symbol = self.symbols[ticker]
        price = self.securities[symbol].price

        if price <= 0 or atr_value <= 0:
            return self.fallback_stop_pct

        stop_pct = (atr_value * self.stop_loss_atr_mult) / price
        # Clamp between 2% and 5%
        return min(max(stop_pct, 0.02), 0.05)

    # ──────────────────────────────────────────────────────────────────────
    # Data handling
    # ──────────────────────────────────────────────────────────────────────

    def on_data(self, data):
        self.data_calls += 1

        # ── Scan news for event signals ──
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
                continue  # already in a position

            for event_type, config in self.event_patterns.items():
                matches = sum(1 for kw in config["keywords"] if kw in text)
                if matches >= 1:
                    self.news_events += 1
                    self.total_signals += 1
                    self.event_counts[event_type] += 1
                    self._execute_trade(symbol, ticker, config["hold_days"])
                    break

        # ── Update price and volume tracking ──
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

    # ──────────────────────────────────────────────────────────────────────
    # Gap scanner
    # ──────────────────────────────────────────────────────────────────────

    def scan_for_gaps(self):
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
                    self._execute_trade(symbol, ticker, 5)

    # ──────────────────────────────────────────────────────────────────────
    # Trade execution
    # ──────────────────────────────────────────────────────────────────────

    def _execute_trade(self, symbol, ticker, hold_days):
        if len(self.position_exit_dates) >= self.max_positions:
            return

        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        # ── Combined position sizing ──
        regime_scale = self._get_regime_scale()
        seasonal_scale = self._get_seasonal_scale()

        effective_size = self.base_position_pct * regime_scale * seasonal_scale
        effective_size = max(effective_size, 0.01)   # minimum 1% position
        effective_size = min(effective_size, 0.06)   # maximum 6% position

        # ── Exposure check ──
        current_exposure = sum(
            abs(h.holdings_value) for h in self.portfolio.values() if h.invested
        ) / total_value

        if current_exposure + effective_size > self.max_total_exposure:
            return

        price = self.securities[symbol].price
        if price <= 0:
            return

        quantity = round(total_value * effective_size / price, 4)
        if quantity < 0.001:
            return

        self.market_order(symbol, quantity)
        self.position_exit_dates[symbol] = self.time + timedelta(days=hold_days)
        self.entry_prices[symbol] = price
        self.stop_levels[symbol] = self._get_atr_stop(ticker)
        self.total_trades += 1

        # Track regime at entry
        is_up = self._is_uptrend()
        self.trade_regime_at_entry[symbol] = "up" if is_up else "down"
        if is_up:
            self.trades_in_uptrend += 1
        else:
            self.trades_in_downtrend += 1

        # Track monthly trades
        self.monthly_trades[self.time.month] += 1

    # ──────────────────────────────────────────────────────────────────────
    # Stop losses (ATR-based)
    # ──────────────────────────────────────────────────────────────────────

    def check_stop_losses(self):
        to_remove = []
        for symbol, entry_price in list(self.entry_prices.items()):
            if symbol not in self.securities:
                continue
            current_price = self.securities[symbol].price
            if current_price <= 0 or entry_price <= 0:
                continue

            stop_pct = self.stop_levels.get(symbol, self.fallback_stop_pct)
            loss_pct = (current_price - entry_price) / entry_price

            if loss_pct <= -stop_pct:
                if self.portfolio[symbol].invested:
                    pnl = self.portfolio[symbol].unrealized_profit
                    self._record_exit_stats(symbol, pnl)
                    self.liquidate(symbol)
                    self.stopped_out += 1
                to_remove.append(symbol)

        for s in to_remove:
            self.position_exit_dates.pop(s, None)
            self.entry_prices.pop(s, None)
            self.stop_levels.pop(s, None)
            self.trade_regime_at_entry.pop(s, None)

    # ──────────────────────────────────────────────────────────────────────
    # Time-based exits
    # ──────────────────────────────────────────────────────────────────────

    def check_exits(self):
        to_remove = []
        for symbol, exit_date in self.position_exit_dates.items():
            if self.time >= exit_date:
                if self.portfolio[symbol].invested:
                    pnl = self.portfolio[symbol].unrealized_profit
                    self._record_exit_stats(symbol, pnl)
                    self.liquidate(symbol)
                to_remove.append(symbol)

        for s in to_remove:
            del self.position_exit_dates[s]
            self.entry_prices.pop(s, None)
            self.stop_levels.pop(s, None)
            self.trade_regime_at_entry.pop(s, None)

    # ──────────────────────────────────────────────────────────────────────
    # Stats tracking
    # ──────────────────────────────────────────────────────────────────────

    def _record_exit_stats(self, symbol, pnl):
        """Record P&L by regime and month for analysis."""
        regime = self.trade_regime_at_entry.get(symbol, "up")
        if regime == "up":
            self.pnl_in_uptrend += pnl
        else:
            self.pnl_in_downtrend += pnl

        self.monthly_pnl[self.time.month] += pnl

    # ──────────────────────────────────────────────────────────────────────
    # End of algorithm logging
    # ──────────────────────────────────────────────────────────────────────

    def on_end_of_algorithm(self):
        total_profit = self.portfolio.total_profit
        ret_pct = total_profit / 100_000

        events_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self.event_counts.items())
        )

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} "
            f"Signals={self.total_signals} Trades={self.total_trades} "
            f"StoppedOut={self.stopped_out} "
            f"NewsEvents={self.news_events} GapEvents={self.gap_events} "
            f"Final=${self.portfolio.total_portfolio_value:,.0f} "
            f"Events=[{events_str}]"
        )

        self.debug(
            f"REGIME: Uptrend trades={self.trades_in_uptrend} "
            f"PnL=${self.pnl_in_uptrend:,.0f} | "
            f"Downtrend trades={self.trades_in_downtrend} "
            f"PnL=${self.pnl_in_downtrend:,.0f}"
        )

        month_names = [
            "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        monthly_str = " | ".join(
            f"{month_names[m]}:{self.monthly_trades[m]}t/${self.monthly_pnl[m]:,.0f}"
            for m in range(1, 13)
            if self.monthly_trades[m] > 0
        )
        self.debug(f"MONTHLY: {monthly_str}")
