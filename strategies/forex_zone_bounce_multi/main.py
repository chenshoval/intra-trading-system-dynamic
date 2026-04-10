"""Multi-Pair FX Zone Bounce — Hypothesis 5b

Expansion of the single-pair forex_zone_bounce (EURUSD, killed for too few trades)
to 27 forex pairs. Same core thesis: buy at demand zones, sell at supply zones
when price shows rejection (pin bar / wick). Diversification across 27 pairs
provides the trade volume that a single pair couldn't.

Changes from original zone_bounce:
- 27 pairs instead of 1 (trade volume from pair diversity)
- 4H bars for zone detection (cleaner than 15min)
- 1H bars for entry detection (still catches rejections)
- Wider session filter (08:00-21:00 UTC vs 13:00-17:00)
- ADX trend filter replaces strict range filter
- ATR-based SL/TP (adaptive per pair) instead of fixed pips
- Relaxed wick ratio (0.5 vs 0.6)
- Max 8 simultaneous positions
- 5-day time stop on zombie positions

Academic basis:
- Teeple (SSRN-3667920): S/R levels are real equilibrium outcomes
- Fritz & Weinhardt (SSRN-2788997): Order book depth confirms institutional support at S/R

Test periods (walk-forward):
- Period 1: 2020-2022 (COVID + rate hikes — worst case for mean-reversion)
- Period 2: 2016-2019 (range-bound — best case)
- Period 3: 2023-2025 (recent)

Kill criteria: WR < 40% OR return < -10% on any period.
"""

from AlgorithmImports import *
import numpy as np
from collections import deque, defaultdict


class ForexZoneBounceMulti(QCAlgorithm):

    def initialize(self):
        # ══════════════════════════════════════════════════════════════
        # Test period
        # ══════════════════════════════════════════════════════════════
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE)

        # ══════════════════════════════════════════════════════════════
        # Universe: 27 Forex Pairs
        # ══════════════════════════════════════════════════════════════
        self.all_pairs = [
            # Majors
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "NZDUSD", "USDCAD",
            # EUR crosses
            "EURJPY", "EURGBP", "EURAUD", "EURNZD", "EURCHF",
            # GBP crosses
            "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD",
            # AUD crosses
            "AUDJPY", "AUDNZD", "AUDCHF",
            # Other crosses
            "NZDJPY", "NZDCHF", "NZDCAD",
            "CADJPY", "CADCHF", "CHFJPY",
            "EURCAD", "GBPCAD",
        ]

        # ══════════════════════════════════════════════════════════════
        # Zone detection parameters (from original zone_bounce, adapted for 4H)
        # ══════════════════════════════════════════════════════════════
        self.swing_lookback = 3             # bars on each side to confirm swing (was 5 on 15min, 3 on 4H is equivalent)
        self.zone_lookback = 120            # 4H bars to scan for zones (120 × 4H = 20 days)
        self.min_touches = 2                # min swing points to form a zone
        self.max_zone_touches = 4           # zone expires after this many tests
        self.zone_cluster_atr_mult = 1.5    # cluster width = 1.5 × ATR (adaptive, not fixed pips)

        # ══════════════════════════════════════════════════════════════
        # Entry parameters
        # ══════════════════════════════════════════════════════════════
        self.rejection_wick_ratio = 0.5     # wick / candle range (was 0.6 — relaxed)
        self.session_start_hour = 8         # UTC — London open
        self.session_end_hour = 21          # UTC — NY close
        self.adx_trend_threshold = 40       # ADX > this = too trendy, skip entries

        # ══════════════════════════════════════════════════════════════
        # Risk / exits
        # ══════════════════════════════════════════════════════════════
        self.sl_atr_mult = 1.0              # stop loss = 1 × ATR below zone
        self.tp_atr_mult = 2.0              # take profit = 2 × ATR from entry
        self.risk_per_trade = 0.01          # 1% of equity per trade
        self.max_positions = 8              # max simultaneous positions
        self.max_hold_bars = 30             # 30 × 1H = ~5 trading days, then exit

        # Portfolio-level drawdown protection
        self.max_dd_pct = 0.15              # 15% DD from peak → halt all new entries
        self.dd_cooldown_days = 20          # wait 20 trading days before re-entering
        self.consecutive_loss_halt = 5      # 5 losses in a row → pause 10 days
        self.consecutive_loss_cooldown = 10 # days to cool down after streak

        # IBKR minimum lot sizes by base currency
        self.ibkr_min_by_currency = {
            "USD": 25_000, "EUR": 20_000, "GBP": 20_000,
            "JPY": 2_500_000, "CHF": 25_000, "AUD": 25_000,
            "NZD": 35_000, "CAD": 25_000,
        }

        # ══════════════════════════════════════════════════════════════
        # Per-pair data structures
        # ══════════════════════════════════════════════════════════════
        self.symbols = {}
        self.candles_4h = {}               # pair -> deque of 4H candles
        self.pair_atr = {}                 # pair -> ATR(14) indicator on daily

        # Per-pair zones
        self.demand_zones = {}             # pair -> {top, bottom, touches, updated}
        self.supply_zones = {}             # pair -> {top, bottom, touches, updated}

        # Per-pair open positions
        self.open_positions = {}           # pair -> {direction, entry_price, stop, target, entry_time, bars_held}

        # Portfolio drawdown tracking
        self.peak_equity = 100_000
        self.trading_halted = False
        self.halt_until = None              # datetime when we can resume trading
        self.consecutive_losses = 0
        self.dd_halt_count = 0
        self.streak_halt_count = 0

        # ══════════════════════════════════════════════════════════════
        # Tracking
        # ══════════════════════════════════════════════════════════════
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.long_trades = 0
        self.short_trades = 0
        self.timeout_exits = 0
        self.per_pair_pnl = defaultdict(float)
        self.per_pair_trades = defaultdict(int)
        self.per_pair_wins = defaultdict(int)
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)
        self.last_month = None

        # ══════════════════════════════════════════════════════════════
        # Add forex securities + indicators + consolidators
        # ══════════════════════════════════════════════════════════════
        for pair in self.all_pairs:
            forex = self.add_forex(pair, Resolution.HOUR, Market.OANDA)
            self.symbols[pair] = forex.symbol

            # ATR on daily for sizing and zone width
            self.pair_atr[pair] = self.atr(
                forex.symbol, 14, MovingAverageType.SIMPLE, Resolution.DAILY
            )

            # 4H candle buffer for zone detection
            self.candles_4h[pair] = deque(maxlen=self.zone_lookback + 20)

            # Initialize zone tracking
            self.demand_zones[pair] = {"top": None, "bottom": None, "touches": 0, "updated": None}
            self.supply_zones[pair] = {"top": None, "bottom": None, "touches": 0, "updated": None}

            # 4H consolidator
            consolidator = QuoteBarConsolidator(timedelta(hours=4))
            consolidator.data_consolidated += self._make_4h_handler(pair)
            self.subscription_manager.add_consolidator(forex.symbol, consolidator)

        # Hourly check for entries and exits
        self.schedule.on(
            self.date_rules.every_day("EURUSD"),
            self.time_rules.every(timedelta(hours=1)),
            self.hourly_check,
        )

        self.set_warm_up(timedelta(days=30))

        self.debug(
            f">>> ZONE BOUNCE MULTI v1: {len(self.all_pairs)} pairs, "
            f"wick={self.rejection_wick_ratio}, session={self.session_start_hour}-{self.session_end_hour}UTC, "
            f"SL={self.sl_atr_mult}×ATR, TP={self.tp_atr_mult}×ATR, "
            f"max_pos={self.max_positions}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Consolidator handler factory (closure to capture pair name)
    # ══════════════════════════════════════════════════════════════════════

    def _make_4h_handler(self, pair):
        """Create a 4H bar handler bound to a specific pair."""
        def handler(sender, bar):
            self._on_4h_bar(pair, bar)
        return handler

    def _on_4h_bar(self, pair, bar):
        """Process a new 4H bar: store candle, update zones."""
        candle = {
            "time": bar.end_time,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
        }
        self.candles_4h[pair].append(candle)

        if self.is_warming_up:
            return

        # Update zones every 6 bars (~24 hours)
        if len(self.candles_4h[pair]) >= self.zone_lookback:
            zone_info = self.demand_zones[pair]
            if zone_info["updated"] is None or (bar.end_time - zone_info["updated"]).total_seconds() > 6 * 4 * 3600:
                self._update_demand_zone(pair)
                self._update_supply_zone(pair)

    # ══════════════════════════════════════════════════════════════════════
    # Zone detection (adapted from forex_zone_bounce)
    # ══════════════════════════════════════════════════════════════════════

    def _find_swing_points(self, candles, point_type="low"):
        """Find swing lows or highs in candle data."""
        points = []
        lookback = self.swing_lookback

        for i in range(lookback, len(candles) - lookback):
            val = candles[i][point_type]
            is_swing = True

            for j in range(1, lookback + 1):
                if point_type == "low":
                    if candles[i - j]["low"] <= val or candles[i + j]["low"] <= val:
                        is_swing = False
                        break
                else:
                    if candles[i - j]["high"] >= val or candles[i + j]["high"] >= val:
                        is_swing = False
                        break

            if is_swing:
                points.append(val)

        return points

    def _cluster_points(self, points, cluster_width):
        """Find the largest cluster of points within cluster_width."""
        if len(points) < self.min_touches:
            return None

        points.sort()
        best_cluster = []

        for i in range(len(points)):
            cluster = [points[i]]
            for j in range(i + 1, len(points)):
                if points[j] - points[i] <= cluster_width:
                    cluster.append(points[j])
                else:
                    break
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        if len(best_cluster) >= self.min_touches:
            return best_cluster
        return None

    def _update_demand_zone(self, pair):
        """Find demand zone from clustered swing lows on 4H candles."""
        candles = list(self.candles_4h[pair])
        if len(candles) < self.zone_lookback:
            return

        atr_val = self.pair_atr[pair].current.value if self.pair_atr[pair].is_ready else 0
        if atr_val <= 0:
            return

        cluster_width = self.zone_cluster_atr_mult * atr_val
        swing_lows = self._find_swing_points(candles, "low")
        cluster = self._cluster_points(swing_lows, cluster_width)

        if cluster:
            new_bottom = min(cluster)
            new_top = max(cluster) + 0.5 * atr_val  # zone extends half ATR above cluster

            zone = self.demand_zones[pair]
            # Reset touch count if zone moved significantly
            if zone["bottom"] is None or abs(new_bottom - zone["bottom"]) > atr_val:
                zone["touches"] = 0

            zone["bottom"] = new_bottom
            zone["top"] = new_top
            zone["updated"] = candles[-1]["time"]

    def _update_supply_zone(self, pair):
        """Find supply zone from clustered swing highs on 4H candles."""
        candles = list(self.candles_4h[pair])
        if len(candles) < self.zone_lookback:
            return

        atr_val = self.pair_atr[pair].current.value if self.pair_atr[pair].is_ready else 0
        if atr_val <= 0:
            return

        cluster_width = self.zone_cluster_atr_mult * atr_val
        swing_highs = self._find_swing_points(candles, "high")
        cluster = self._cluster_points(swing_highs, cluster_width)

        if cluster:
            new_top = max(cluster)
            new_bottom = min(cluster) - 0.5 * atr_val

            zone = self.supply_zones[pair]
            if zone["top"] is None or abs(new_top - zone["top"]) > atr_val:
                zone["touches"] = 0

            zone["top"] = new_top
            zone["bottom"] = new_bottom
            zone["updated"] = candles[-1]["time"]

    # ══════════════════════════════════════════════════════════════════════
    # Hourly check: entries, exits, position management
    # ══════════════════════════════════════════════════════════════════════

    def hourly_check(self):
        """Run every hour: check stops/TPs, then check for new entries."""
        if self.is_warming_up:
            return

        equity = self.portfolio.total_portfolio_value

        # Monthly tracking
        current_month = f"{self.time.year}-{self.time.month:02d}"
        if self.last_month is not None and current_month != self.last_month:
            month_ret = (equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)
            self.monthly_pnl[self.last_month] = equity - self.month_start_equity
            self.month_start_equity = equity
        self.last_month = current_month

        # ── Update peak equity and check drawdown ──
        if equity > self.peak_equity:
            self.peak_equity = equity

        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # ── Clean up stale positions (close failed previously due to MOO error) ──
        # If we think we closed a position but QC still shows it invested, force close it
        # Only attempt once per pair per day to avoid infinite retry loops
        if not hasattr(self, '_orphan_cleanup_dates'):
            self._orphan_cleanup_dates = {}

        today = self.time.date()
        for pair in self.all_pairs:
            symbol = self.symbols[pair]
            if pair not in self.open_positions and self.portfolio[symbol].invested:
                # Only try once per pair per day
                last_attempt = self._orphan_cleanup_dates.get(pair)
                if last_attempt == today:
                    continue
                self._orphan_cleanup_dates[pair] = today

                # Only attempt during safe hours (not market open/close)
                hour = self.time.hour
                if hour in (22, 23, 0, 17):
                    continue

                qty = self.portfolio[symbol].quantity
                self.market_order(symbol, -qty)
                self.debug(f"ORPHAN CLEANUP: {pair} had {qty:,.0f} units, closing (1 attempt/day)")

        # Check if halt should be lifted
        if self.trading_halted and self.halt_until is not None:
            if self.time >= self.halt_until:
                self.trading_halted = False
                self.halt_until = None
                self.consecutive_losses = 0
                # CRITICAL: reset peak to current equity so DD starts fresh
                self.peak_equity = equity
                self.debug(f"TRADING RESUMED: equity=${equity:,.0f}, peak reset")

        # Trigger DD halt
        if current_dd > self.max_dd_pct and not self.trading_halted:
            self.trading_halted = True
            self.halt_until = self.time + timedelta(days=self.dd_cooldown_days)
            self.dd_halt_count += 1
            # Close all open positions — try both our tracking and QC portfolio
            for pair in list(self.open_positions.keys()):
                self._exit_trade(pair, self.securities[self.symbols[pair]].price, "DD Halt")
            # Also force-close anything QC still shows as invested (safe hours only)
            hour = self.time.hour
            if hour not in (22, 23, 0, 17):
                for pair in self.all_pairs:
                    symbol = self.symbols[pair]
                    if self.portfolio[symbol].invested:
                        qty = self.portfolio[symbol].quantity
                        self.market_order(symbol, -qty)
            self.debug(
                f"DD HALT: {current_dd:.1%} > {self.max_dd_pct:.0%}. "
                f"Closed all, halting until {self.halt_until.strftime('%Y-%m-%d')}. "
                f"Equity=${equity:,.0f} Peak=${self.peak_equity:,.0f}"
            )
            return

        # ── Check existing positions (always, even if halted) ──
        for pair in list(self.open_positions.keys()):
            self._check_exit(pair)

        # ── Don't enter new trades if halted ──
        if self.trading_halted:
            return

        # ── Session filter ──
        if not self._in_session(self.time):
            return

        # ── Avoid exact market open hours (causes MOO order errors) ──
        # Forex market opens Sunday 17:00 ET = ~22:00 UTC
        # Skip the first hour after any market open/close transition
        hour = self.time.hour
        if hour == 22 or hour == 23 or hour == 0:
            return

        if len(self.open_positions) >= self.max_positions:
            return

        for pair in self.all_pairs:
            if pair in self.open_positions:
                continue
            # CRITICAL: also check if QC thinks we have a position (close may have failed)
            symbol = self.symbols[pair]
            if self.portfolio[symbol].invested:
                continue
            if len(self.open_positions) >= self.max_positions:
                break

            self._check_long_entry(pair)
            if pair not in self.open_positions:  # Only check short if we didn't just go long
                self._check_short_entry(pair)

    def _in_session(self, time):
        """Check if current time is within trading session."""
        return self.session_start_hour <= time.hour < self.session_end_hour

    # ══════════════════════════════════════════════════════════════════════
    # Entry logic
    # ══════════════════════════════════════════════════════════════════════

    def _check_long_entry(self, pair):
        """Check for bullish rejection at demand zone."""
        zone = self.demand_zones[pair]
        if zone["top"] is None:
            return

        # Zone freshness
        if zone["touches"] >= self.max_zone_touches:
            return

        symbol = self.symbols[pair]
        price = self.securities[symbol].price
        if price <= 0:
            return

        # Get latest 1H candle data from security
        bar = self.securities[symbol]
        high = float(bar.high)
        low = float(bar.low)
        open_price = float(bar.open)
        close = float(bar.close)

        # Wick into zone
        if low > zone["top"]:
            return  # Didn't reach zone
        if low < zone["bottom"] - self.sl_atr_mult * self._get_atr(pair):
            return  # Crashed through zone

        # Close above zone (rejection)
        if close < zone["top"]:
            return

        # Bullish candle
        if close <= open_price:
            return

        # Wick ratio check
        candle_range = high - low
        if candle_range == 0:
            return
        lower_wick = min(open_price, close) - low
        if (lower_wick / candle_range) < self.rejection_wick_ratio:
            return

        # ENTER LONG
        zone["touches"] += 1
        self._enter_trade(pair, 1, close)

    def _check_short_entry(self, pair):
        """Check for bearish rejection at supply zone."""
        zone = self.supply_zones[pair]
        if zone["top"] is None:
            return

        if zone["touches"] >= self.max_zone_touches:
            return

        symbol = self.symbols[pair]
        price = self.securities[symbol].price
        if price <= 0:
            return

        bar = self.securities[symbol]
        high = float(bar.high)
        low = float(bar.low)
        open_price = float(bar.open)
        close = float(bar.close)

        # Wick into supply zone
        if high < zone["bottom"]:
            return
        if high > zone["top"] + self.sl_atr_mult * self._get_atr(pair):
            return

        # Close below zone
        if close > zone["bottom"]:
            return

        # Bearish candle
        if close >= open_price:
            return

        # Wick ratio
        candle_range = high - low
        if candle_range == 0:
            return
        upper_wick = high - max(open_price, close)
        if (upper_wick / candle_range) < self.rejection_wick_ratio:
            return

        # ENTER SHORT
        zone["touches"] += 1
        self._enter_trade(pair, -1, close)

    # ══════════════════════════════════════════════════════════════════════
    # Trade execution
    # ══════════════════════════════════════════════════════════════════════

    def _get_atr(self, pair):
        """Get current ATR value for a pair."""
        if self.pair_atr[pair].is_ready:
            return self.pair_atr[pair].current.value
        return 0

    def _get_ibkr_min_units(self, pair):
        """Get IBKR minimum lot size for pair's base currency."""
        base = pair[:3]
        return self.ibkr_min_by_currency.get(base, 25_000)

    def _get_quote_currency_rate(self, pair):
        """Get the USD value of 1 unit of the quote currency.
        For EURUSD (quote=USD): returns 1.0
        For USDJPY (quote=JPY): returns ~0.009 (1/USDJPY)
        For EURGBP (quote=GBP): returns ~1.27 (GBPUSD rate)
        This is needed to convert P&L from quote currency to USD."""
        quote = pair[3:6]
        if quote == "USD":
            return 1.0
        # Try direct quote (e.g., GBPUSD for GBP)
        direct = f"{quote}USD"
        if direct in self.symbols:
            price = self.securities[self.symbols[direct]].price
            if price > 0:
                return price
        # Try inverse (e.g., USDJPY for JPY → 1/USDJPY)
        inverse = f"USD{quote}"
        if inverse in self.symbols:
            price = self.securities[self.symbols[inverse]].price
            if price > 0:
                return 1.0 / price
        return 1.0  # fallback

    def _enter_trade(self, pair, direction, entry_price):
        """Enter a trade with ATR-based stops.
        Position sizing accounts for quote currency conversion to USD."""
        atr_val = self._get_atr(pair)
        if atr_val <= 0:
            return

        symbol = self.symbols[pair]
        equity = self.portfolio.total_portfolio_value

        # Position sizing: risk 1% of equity, stop is sl_atr_mult × ATR
        # Key: stop_distance is in QUOTE currency per unit of base currency
        # P&L per unit = stop_distance × quote_to_usd_rate
        # So: units = risk_amount / (stop_distance × quote_to_usd_rate)
        risk_amount = equity * self.risk_per_trade
        stop_distance = self.sl_atr_mult * atr_val
        if stop_distance <= 0:
            return

        quote_rate = self._get_quote_currency_rate(pair)
        pnl_per_unit_at_stop = stop_distance * quote_rate

        if pnl_per_unit_at_stop <= 0:
            return

        raw_units = risk_amount / pnl_per_unit_at_stop
        min_units = self._get_ibkr_min_units(pair)

        if raw_units < min_units:
            # Check if min lot risk exceeds 2% of equity (safety cap)
            min_lot_risk = min_units * pnl_per_unit_at_stop
            if min_lot_risk > equity * 0.02:
                return  # Min lot is too risky for this pair
            raw_units = min_units

        units = max(min_units, int(raw_units / 1000) * 1000)

        # HARD CAP: max $50K notional per trade (prevents runaway sizing)
        # For GBPJPY at 170: 50000/170 = 294 units... that's too low.
        # Better: cap at max 100K base currency units regardless of pair
        max_units = 100_000
        if pair[:3] == "JPY" or pair[3:6] == "JPY":
            # JPY base pairs (like JPYUSD which doesn't exist) — n/a
            # JPY quote pairs — units are in base currency, cap still applies
            pass
        units = min(units, max_units)

        # Ensure still above IBKR minimum after cap
        if units < min_units:
            return

        # Final safety: cap risk at 1.5% of equity (stricter than before)
        actual_risk = units * pnl_per_unit_at_stop
        if actual_risk > equity * 0.015:
            units = int(equity * 0.015 / pnl_per_unit_at_stop / 1000) * 1000
            if units < min_units:
                return

        quantity = units * direction

        if quantity == 0:
            return

        # Set stops
        if direction == 1:  # Long
            stop_price = self.demand_zones[pair]["bottom"] - self.sl_atr_mult * atr_val
            target_price = entry_price + self.tp_atr_mult * atr_val
        else:  # Short
            stop_price = self.supply_zones[pair]["top"] + self.sl_atr_mult * atr_val
            target_price = entry_price - self.tp_atr_mult * atr_val

        self.market_order(symbol, quantity)
        self.total_trades += 1
        if direction == 1:
            self.long_trades += 1
        else:
            self.short_trades += 1

        self.open_positions[pair] = {
            "direction": direction,
            "entry_price": entry_price,
            "stop": stop_price,
            "target": target_price,
            "entry_time": self.time,
            "bars_held": 0,
            "quantity": quantity,
        }

        dir_str = "LONG" if direction == 1 else "SHORT"
        expected_risk_usd = abs(quantity) * pnl_per_unit_at_stop
        self.debug(
            f"{dir_str} {pair} @ {entry_price:.5f} | "
            f"SL: {stop_price:.5f} | TP: {target_price:.5f} | "
            f"Qty: {quantity:,} | Risk$: {expected_risk_usd:,.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Exit logic
    # ══════════════════════════════════════════════════════════════════════

    def _check_exit(self, pair):
        """Check if an open position should be closed."""
        if pair not in self.open_positions:
            return

        pos = self.open_positions[pair]
        symbol = self.symbols[pair]
        price = self.securities[symbol].price

        if price <= 0:
            return

        pos["bars_held"] += 1
        direction = pos["direction"]
        reason = None

        if direction == 1:  # Long
            if price <= pos["stop"]:
                reason = "Stop Loss"
            elif price >= pos["target"]:
                reason = "Take Profit"
        else:  # Short
            if price >= pos["stop"]:
                reason = "Stop Loss"
            elif price <= pos["target"]:
                reason = "Take Profit"

        # Time stop
        if pos["bars_held"] >= self.max_hold_bars:
            reason = "Time Stop"
            self.timeout_exits += 1

        if reason:
            self._exit_trade(pair, price, reason)

    def _exit_trade(self, pair, exit_price, reason):
        """Close a position and record results."""
        if pair not in self.open_positions:
            return

        pos = self.open_positions[pair]
        symbol = self.symbols[pair]

        # Calculate P&L in USD
        # Price difference is in QUOTE currency, need to convert to USD
        price_diff = (exit_price - pos["entry_price"]) * pos["direction"]
        quote_rate = self._get_quote_currency_rate(pair)
        pnl_usd = price_diff * abs(pos["quantity"]) * quote_rate
        is_win = pnl_usd > 0

        # Close position with market order (liquidate() can fail with MOO error on forex)
        close_qty = -pos["quantity"]
        order = self.market_order(symbol, close_qty)

        # If market_order returns None or invalid, try liquidate as fallback
        if order is None or (hasattr(order, 'status') and order.status == OrderStatus.INVALID):
            self.liquidate(symbol)

        # Record stats
        self.per_pair_pnl[pair] += pnl_usd
        self.per_pair_trades[pair] += 1
        if is_win:
            self.winning_trades += 1
            self.per_pair_wins[pair] += 1
            self.consecutive_losses = 0  # reset streak on win
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            # Consecutive loss halt
            if self.consecutive_losses >= self.consecutive_loss_halt and not self.trading_halted:
                self.trading_halted = True
                self.halt_until = self.time + timedelta(days=self.consecutive_loss_cooldown)
                self.streak_halt_count += 1
                self.debug(
                    f"STREAK HALT: {self.consecutive_losses} consecutive losses. "
                    f"Pausing until {self.halt_until.strftime('%Y-%m-%d')}"
                )

        dir_str = "LONG" if pos["direction"] == 1 else "SHORT"
        self.debug(
            f"EXIT {dir_str} {pair} @ {exit_price:.5f} | "
            f"Reason: {reason} | PnL: ${pnl_usd:,.0f} | Held: {pos['bars_held']}H"
        )

        del self.open_positions[pair]

    # ══════════════════════════════════════════════════════════════════════
    # OnData — check stops on every bar (more responsive than hourly)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        if self.is_warming_up:
            return

        for pair in list(self.open_positions.keys()):
            pos = self.open_positions[pair]
            symbol = self.symbols[pair]

            if not data.contains_key(symbol):
                continue

            price = self.securities[symbol].price
            if price <= 0:
                continue

            direction = pos["direction"]

            if direction == 1 and price <= pos["stop"]:
                self._exit_trade(pair, price, "Stop Loss")
            elif direction == -1 and price >= pos["stop"]:
                self._exit_trade(pair, price, "Stop Loss")
            elif direction == 1 and price >= pos["target"]:
                self._exit_trade(pair, price, "Take Profit")
            elif direction == -1 and price <= pos["target"]:
                self._exit_trade(pair, price, "Take Profit")

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        equity = self.portfolio.total_portfolio_value
        ret_pct = (equity - 100_000) / 100_000

        # Final monthly return
        if self.last_month is not None:
            month_ret = (equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)

        total_closed = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_closed * 100 if total_closed > 0 else 0

        self.debug("=" * 70)
        self.debug("FX ZONE BOUNCE MULTI — FINAL RESULTS")
        self.debug("=" * 70)
        self.debug(
            f"RETURNS: Total={ret_pct:.2%} Final=${equity:,.0f}"
        )
        self.debug(
            f"TRADES: Total={self.total_trades} Closed={total_closed} "
            f"W={self.winning_trades} L={self.losing_trades} "
            f"WR={win_rate:.0f}% Longs={self.long_trades} "
            f"Shorts={self.short_trades} TimeOuts={self.timeout_exits} "
            f"DDHalts={self.dd_halt_count} StreakHalts={self.streak_halt_count}"
        )

        # Monthly stats
        if self.monthly_returns:
            rets = np.array(self.monthly_returns)
            win_months = np.sum(rets > 0)
            loss_months = np.sum(rets <= 0)
            avg_ret = np.mean(rets)
            std = np.std(rets)
            monthly_sharpe = avg_ret / std if std > 0 else 0
            self.debug(
                f"MONTHLY: Avg={avg_ret:.2%} Best={np.max(rets):.2%} "
                f"Worst={np.min(rets):.2%} Std={std:.2%} "
                f"Win={win_months} Loss={loss_months} "
                f"WR={win_months/len(rets):.0%} Sharpe={monthly_sharpe:.2f}"
            )

        # Per-pair breakdown
        if self.per_pair_pnl:
            self.debug("\nPER-PAIR ANALYSIS (sorted by P&L):")
            sorted_pairs = sorted(
                self.per_pair_pnl.items(), key=lambda x: x[1], reverse=True
            )
            for pair, pnl in sorted_pairs:
                trades = self.per_pair_trades.get(pair, 0)
                wins = self.per_pair_wins.get(pair, 0)
                wr = wins / trades * 100 if trades > 0 else 0
                self.debug(
                    f"  {pair:8s}: PnL=${pnl:>+10,.0f} "
                    f"Trades={trades:>3d} WR={wr:>4.0f}%"
                )

        # Pairs with NO trades (zone detection didn't fire)
        no_trade_pairs = [p for p in self.all_pairs if self.per_pair_trades[p] == 0]
        if no_trade_pairs:
            self.debug(f"\nNO TRADES: {', '.join(no_trade_pairs)}")

        # Monthly PnL
        if self.monthly_pnl:
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"\nMONTHLY PNL: {pnl_str}")
