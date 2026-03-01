"""Support Zone Bounce v2 — Enhanced for Production

Forex range-trading strategy on EUR/USD. Identifies demand/supply zones
from clustered swing lows/highs, confirms range context on 1H, enters
on rejection candles (pin bars) at zone boundaries.

Enhancements over v1:
- Session filter: only trades during London/NY overlap (13:00-17:00 UTC)
- Supply zone shorts: mirrors demand logic for sell setups at swing highs
- Zone freshness: tracks zone touches, skips zones tested 4+ times
- Better logging for performance comparison

This is a SEPARATE strategy from our stock rotator — completely
uncorrelated (forex vs US equities, mean-reversion vs momentum,
intraday vs monthly). If it works, it's a second independent income
stream per the Warsaw paper principle.
"""

from AlgorithmImports import *
import numpy as np
from collections import deque, defaultdict


class SupportZoneBounceV2(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        self.SetBrokerageModel(BrokerageName.OandaBrokerage)
        self.symbol = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol

        # ── Zone Detection ──
        self.swing_lookback = 5
        self.zone_lookback = 200
        self.zone_cluster_pips = 15
        self.min_touches = 2
        self.max_zone_touches = 4           # zone expires after this many tests
        self.pip_size = 0.0001

        # ── Range Filter (1H) ──
        self.range_lookback = 50
        self.range_ratio_threshold = 0.35

        # ── Entry ──
        self.rejection_wick_ratio = 0.6

        # ── Session Filter (UTC hours) ──
        self.session_start_hour = 13        # 1 PM UTC (London afternoon / NY morning)
        self.session_end_hour = 17          # 5 PM UTC

        # ── Risk Management ──
        self.stop_loss_pips = 15
        self.take_profit_pips = 30
        self.risk_per_trade = 0.01
        self.max_positions = 1

        # ── Data Structures ──
        self.candles_15m = deque(maxlen=self.zone_lookback + 20)
        self.candles_1h = deque(maxlen=self.range_lookback + 10)

        # Demand zone (support — for longs)
        self.demand_zone_top = None
        self.demand_zone_bottom = None
        self.demand_zone_touches = 0
        self.demand_zone_updated = None

        # Supply zone (resistance — for shorts)
        self.supply_zone_top = None
        self.supply_zone_bottom = None
        self.supply_zone_touches = 0
        self.supply_zone_updated = None

        # Trade tracking
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_direction = 0            # 1=long, -1=short, 0=flat

        # ── Counters ──
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.long_trades = 0
        self.short_trades = 0
        self.skipped_session = 0
        self.skipped_trending = 0
        self.skipped_stale_zone = 0
        self.total_pnl = 0.0

        # ── Consolidators ──
        fifteen_min = QuoteBarConsolidator(timedelta(minutes=15))
        fifteen_min.DataConsolidated += self.On15MinBar
        self.SubscriptionManager.AddConsolidator(self.symbol, fifteen_min)

        one_hour = QuoteBarConsolidator(timedelta(hours=1))
        one_hour.DataConsolidated += self.On1HBar
        self.SubscriptionManager.AddConsolidator(self.symbol, one_hour)

        self.SetWarmUp(timedelta(days=10))

        self.Debug(f">>> SUPPORT ZONE BOUNCE v2: EURUSD, session {self.session_start_hour}-{self.session_end_hour} UTC")

    # ══════════════════════════════════════════════════════════════════════
    # DATA HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def On15MinBar(self, sender, bar):
        candle = {
            "time": bar.EndTime,
            "open": float(bar.Open),
            "high": float(bar.High),
            "low": float(bar.Low),
            "close": float(bar.Close),
        }
        self.candles_15m.append(candle)

        if self.IsWarmingUp:
            return

        # Update zones periodically (every ~20 bars = 5 hours)
        if len(self.candles_15m) >= self.zone_lookback:
            if (self.demand_zone_updated is None or
                (bar.EndTime - self.demand_zone_updated).total_seconds() > 20 * 15 * 60):
                self.UpdateDemandZone()
                self.UpdateSupplyZone()

        # Check entries
        self.CheckLongEntry(candle)
        self.CheckShortEntry(candle)

    def On1HBar(self, sender, bar):
        candle = {
            "time": bar.EndTime,
            "open": float(bar.Open),
            "high": float(bar.High),
            "low": float(bar.Low),
            "close": float(bar.Close),
        }
        self.candles_1h.append(candle)

    def OnData(self, data):
        if self.IsWarmingUp:
            return

        if self.Portfolio[self.symbol].Invested:
            price = self.Securities[self.symbol].Price

            if self.trade_direction == 1:  # Long
                if price <= self.stop_price:
                    self.Liquidate(self.symbol, "Long Stop Loss")
                    self._record_trade(price, False)
                elif price >= self.target_price:
                    self.Liquidate(self.symbol, "Long Take Profit")
                    self._record_trade(price, True)

            elif self.trade_direction == -1:  # Short
                if price >= self.stop_price:
                    self.Liquidate(self.symbol, "Short Stop Loss")
                    self._record_trade(price, False)
                elif price <= self.target_price:
                    self.Liquidate(self.symbol, "Short Take Profit")
                    self._record_trade(price, True)

    # ══════════════════════════════════════════════════════════════════════
    # SESSION FILTER
    # ══════════════════════════════════════════════════════════════════════

    def _in_session(self, time):
        """Check if current time is within London/NY overlap."""
        hour = time.hour
        return self.session_start_hour <= hour < self.session_end_hour

    # ══════════════════════════════════════════════════════════════════════
    # ZONE DETECTION
    # ══════════════════════════════════════════════════════════════════════

    def _find_swing_points(self, candles, point_type="low"):
        """Find swing lows or swing highs in candle data."""
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
                else:  # high
                    if candles[i - j]["high"] >= val or candles[i + j]["high"] >= val:
                        is_swing = False
                        break

            if is_swing:
                points.append(val)

        return points

    def _cluster_points(self, points):
        """Find the largest cluster of points within zone_cluster_pips."""
        if len(points) < self.min_touches:
            return None

        points.sort()
        threshold = self.zone_cluster_pips * self.pip_size
        best_cluster = []

        for i in range(len(points)):
            cluster = [points[i]]
            for j in range(i + 1, len(points)):
                if points[j] - points[i] <= threshold:
                    cluster.append(points[j])
                else:
                    break
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        if len(best_cluster) >= self.min_touches:
            return best_cluster
        return None

    def UpdateDemandZone(self):
        """Find demand zone from clustered swing lows."""
        candles = list(self.candles_15m)
        if len(candles) < self.zone_lookback:
            return

        swing_lows = self._find_swing_points(candles, "low")
        cluster = self._cluster_points(swing_lows)

        if cluster:
            new_bottom = min(cluster)
            new_top = max(cluster) + 5 * self.pip_size

            # Check if zone changed significantly
            if (self.demand_zone_bottom is None or
                abs(new_bottom - self.demand_zone_bottom) > 10 * self.pip_size):
                self.demand_zone_touches = 0  # New zone, reset touch count

            self.demand_zone_bottom = new_bottom
            self.demand_zone_top = new_top
            self.demand_zone_updated = candles[-1]["time"]

    def UpdateSupplyZone(self):
        """Find supply zone from clustered swing highs."""
        candles = list(self.candles_15m)
        if len(candles) < self.zone_lookback:
            return

        swing_highs = self._find_swing_points(candles, "high")
        cluster = self._cluster_points(swing_highs)

        if cluster:
            new_top = max(cluster)
            new_bottom = min(cluster) - 5 * self.pip_size

            if (self.supply_zone_top is None or
                abs(new_top - self.supply_zone_top) > 10 * self.pip_size):
                self.supply_zone_touches = 0

            self.supply_zone_top = new_top
            self.supply_zone_bottom = new_bottom
            self.supply_zone_updated = candles[-1]["time"]

    # ══════════════════════════════════════════════════════════════════════
    # RANGE FILTER
    # ══════════════════════════════════════════════════════════════════════

    def IsRanging(self):
        if len(self.candles_1h) < self.range_lookback:
            return True

        candles = list(self.candles_1h)[-self.range_lookback:]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]

        total_range = max(highs) - min(lows)
        net_move = abs(closes[-1] - closes[0])

        if total_range == 0:
            return True

        return (net_move / total_range) < self.range_ratio_threshold

    # ══════════════════════════════════════════════════════════════════════
    # ENTRY LOGIC
    # ══════════════════════════════════════════════════════════════════════

    def CheckLongEntry(self, candle):
        """Check for bullish rejection at demand zone."""
        if self.Portfolio[self.symbol].Invested:
            return
        if self.demand_zone_top is None:
            return

        # Session filter
        if not self._in_session(candle["time"]):
            self.skipped_session += 1
            return

        # Range filter
        if not self.IsRanging():
            self.skipped_trending += 1
            return

        # Zone freshness
        if self.demand_zone_touches >= self.max_zone_touches:
            self.skipped_stale_zone += 1
            return

        low = candle["low"]
        close = candle["close"]
        open_price = candle["open"]
        high = candle["high"]

        # Wick into zone
        if low > self.demand_zone_top:
            return
        if low < self.demand_zone_bottom - self.stop_loss_pips * self.pip_size:
            return

        # Close above zone
        if close < self.demand_zone_top:
            return

        # Bullish candle
        if close <= open_price:
            return

        # Long lower wick
        candle_range = high - low
        if candle_range == 0:
            return
        lower_wick = min(open_price, close) - low
        if (lower_wick / candle_range) < self.rejection_wick_ratio:
            return

        # ENTER LONG
        self.demand_zone_touches += 1
        self._enter_trade(close, 1)

    def CheckShortEntry(self, candle):
        """Check for bearish rejection at supply zone."""
        if self.Portfolio[self.symbol].Invested:
            return
        if self.supply_zone_bottom is None:
            return

        # Session filter
        if not self._in_session(candle["time"]):
            return

        # Range filter
        if not self.IsRanging():
            return

        # Zone freshness
        if self.supply_zone_touches >= self.max_zone_touches:
            return

        low = candle["low"]
        close = candle["close"]
        open_price = candle["open"]
        high = candle["high"]

        # Wick into supply zone
        if high < self.supply_zone_bottom:
            return
        if high > self.supply_zone_top + self.stop_loss_pips * self.pip_size:
            return

        # Close below zone
        if close > self.supply_zone_bottom:
            return

        # Bearish candle
        if close >= open_price:
            return

        # Long upper wick (rejection)
        candle_range = high - low
        if candle_range == 0:
            return
        upper_wick = high - max(open_price, close)
        if (upper_wick / candle_range) < self.rejection_wick_ratio:
            return

        # ENTER SHORT
        self.supply_zone_touches += 1
        self._enter_trade(close, -1)

    # ══════════════════════════════════════════════════════════════════════
    # TRADE EXECUTION
    # ══════════════════════════════════════════════════════════════════════

    def _enter_trade(self, entry_price, direction):
        """Enter long (direction=1) or short (direction=-1)."""
        self.entry_price = entry_price
        self.trade_direction = direction

        if direction == 1:  # Long
            self.stop_price = self.demand_zone_bottom - self.stop_loss_pips * self.pip_size
            self.target_price = entry_price + self.take_profit_pips * self.pip_size
        else:  # Short
            self.stop_price = self.supply_zone_top + self.stop_loss_pips * self.pip_size
            self.target_price = entry_price - self.take_profit_pips * self.pip_size

        # Position sizing: risk 1% of equity
        risk_per_unit = abs(entry_price - self.stop_price)
        if risk_per_unit <= 0:
            return

        equity = self.Portfolio.TotalPortfolioValue
        risk_amount = equity * self.risk_per_trade
        quantity = int(risk_amount / risk_per_unit) * direction

        if quantity == 0:
            return

        self.MarketOrder(self.symbol, quantity)
        self.total_trades += 1

        if direction == 1:
            self.long_trades += 1
        else:
            self.short_trades += 1

        dir_str = "LONG" if direction == 1 else "SHORT"
        self.Debug(f"{dir_str} @ {entry_price:.5f} | SL: {self.stop_price:.5f} | TP: {self.target_price:.5f}")

    def _record_trade(self, exit_price, is_win):
        """Record trade result and reset."""
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        if self.entry_price and self.trade_direction != 0:
            pnl_pips = (exit_price - self.entry_price) / self.pip_size * self.trade_direction
            self.total_pnl += pnl_pips

        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_direction = 0

    # ══════════════════════════════════════════════════════════════════════
    # END OF ALGORITHM
    # ══════════════════════════════════════════════════════════════════════

    def OnEndOfAlgorithm(self):
        equity = self.Portfolio.TotalPortfolioValue
        ret_pct = (equity - 100000) / 100000

        total_closed = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_closed * 100 if total_closed > 0 else 0

        self.Debug(
            f"RESULTS: Return={ret_pct:.2%} Final=${equity:,.0f} "
            f"Trades={self.total_trades} W={self.winning_trades} L={self.losing_trades} "
            f"WR={win_rate:.0f}% Longs={self.long_trades} Shorts={self.short_trades}"
        )
        self.Debug(
            f"SKIPPED: Session={self.skipped_session} Trending={self.skipped_trending} "
            f"StaleZone={self.skipped_stale_zone}"
        )
        self.Debug(f"PNL (pips): {self.total_pnl:.0f}")
