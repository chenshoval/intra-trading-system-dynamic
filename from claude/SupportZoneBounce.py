# QuantConnect Lean Algorithm
# Strategy: Support Zone Bounce (Range Trading)
# Based on: Identifying demand zones from clustered swing lows,
#           confirming range context on higher timeframe,
#           entering on bullish rejection candles within the zone.
#
# Pair: EUR/USD (easily adaptable to other forex pairs)
# Timeframes: 15-minute (entry), 1-hour (range context)

from AlgorithmImports import *
import numpy as np
from collections import deque


class SupportZoneBounceAlgorithm(QCAlgorithm):

    def Initialize(self):
        # ── Core Settings ──────────────────────────────────────────
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000)

        # Broker model for forex
        self.SetBrokerageModel(BrokerageName.OandaBrokerage)

        # Add EUR/USD with 15-minute resolution (entry timeframe)
        self.symbol = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol

        # ── Strategy Parameters ────────────────────────────────────
        # Zone Detection
        self.swing_lookback = 5           # Bars on each side to qualify as swing low
        self.zone_lookback = 200          # How many 15m bars to look back for swing lows
        self.zone_cluster_pips = 15       # Max distance (pips) to cluster swing lows
        self.min_touches = 2              # Minimum swing lows to form a valid zone
        self.pip_size = 0.0001            # Pip size for EUR/USD

        # Range Filter (1H context)
        self.range_lookback = 50          # 1H bars to assess range vs trend
        self.atr_range_threshold = 1.5    # ATR multiplier: if range < ATR * this, it's ranging

        # Entry
        self.rejection_wick_ratio = 0.6   # Lower wick must be >= 60% of total candle range

        # Risk Management
        self.stop_loss_pips = 15          # Stop below zone bottom
        self.take_profit_pips = 30        # Target above entry
        self.risk_per_trade = 0.01        # Risk 1% of equity per trade
        self.max_positions = 1            # Only 1 position at a time

        # ── Data Structures ────────────────────────────────────────
        # Rolling windows for candle data
        self.candles_15m = deque(maxlen=self.zone_lookback + 10)
        self.candles_1h = deque(maxlen=self.range_lookback + 10)

        # Zone tracking
        self.demand_zone_top = None
        self.demand_zone_bottom = None
        self.zone_last_updated = None

        # Trade tracking
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        # ── Consolidators ──────────────────────────────────────────
        # 15-minute consolidator
        fifteen_min = QuoteBarConsolidator(timedelta(minutes=15))
        fifteen_min.DataConsolidated += self.On15MinBar
        self.SubscriptionManager.AddConsolidator(self.symbol, fifteen_min)

        # 1-hour consolidator
        one_hour = QuoteBarConsolidator(timedelta(hours=1))
        one_hour.DataConsolidated += self.On1HBar
        self.SubscriptionManager.AddConsolidator(self.symbol, one_hour)

        # Warmup
        self.SetWarmUp(timedelta(days=10))

    # ════════════════════════════════════════════════════════════════
    # DATA HANDLERS
    # ════════════════════════════════════════════════════════════════

    def On15MinBar(self, sender, bar):
        """Collect 15-minute candles and check for entries."""
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

        # Update demand zone periodically (every 20 bars)
        if len(self.candles_15m) >= self.zone_lookback:
            if (self.zone_last_updated is None or
                (bar.EndTime - self.zone_last_updated).total_seconds() > 20 * 15 * 60):
                self.UpdateDemandZone()

        # Check for entry signal
        self.CheckEntry(candle)

    def On1HBar(self, sender, bar):
        """Collect 1-hour candles for range context."""
        candle = {
            "time": bar.EndTime,
            "open": float(bar.Open),
            "high": float(bar.High),
            "low": float(bar.Low),
            "close": float(bar.Close),
        }
        self.candles_1h.append(candle)

    def OnData(self, data):
        """Main data handler — manages open positions."""
        if self.IsWarmingUp:
            return

        # Manage existing position
        if self.Portfolio[self.symbol].Invested:
            price = self.Securities[self.symbol].Price

            # Check stop loss
            if price <= self.stop_price:
                self.Liquidate(self.symbol, "Stop Loss Hit")
                self.ResetTrade()

            # Check take profit
            elif price >= self.target_price:
                self.Liquidate(self.symbol, "Take Profit Hit")
                self.ResetTrade()

    # ════════════════════════════════════════════════════════════════
    # ZONE DETECTION
    # ════════════════════════════════════════════════════════════════

    def UpdateDemandZone(self):
        """
        Find clusters of swing lows in recent 15m data.
        A swing low = a bar whose low is lower than the N bars
        on either side of it.
        """
        candles = list(self.candles_15m)
        if len(candles) < self.zone_lookback:
            return

        # Step 1: Find all swing lows
        swing_lows = []
        lookback = self.swing_lookback

        for i in range(lookback, len(candles) - lookback):
            low_i = candles[i]["low"]
            is_swing = True

            for j in range(1, lookback + 1):
                if candles[i - j]["low"] <= low_i or candles[i + j]["low"] <= low_i:
                    is_swing = False
                    break

            if is_swing:
                swing_lows.append(low_i)

        if len(swing_lows) < self.min_touches:
            return

        # Step 2: Cluster swing lows that are within zone_cluster_pips of each other
        swing_lows.sort()
        best_cluster = []
        cluster_threshold = self.zone_cluster_pips * self.pip_size

        for i in range(len(swing_lows)):
            cluster = [swing_lows[i]]
            for j in range(i + 1, len(swing_lows)):
                if swing_lows[j] - swing_lows[i] <= cluster_threshold:
                    cluster.append(swing_lows[j])
                else:
                    break
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        # Step 3: Define the zone from the best cluster
        if len(best_cluster) >= self.min_touches:
            self.demand_zone_bottom = min(best_cluster)
            self.demand_zone_top = max(best_cluster) + 5 * self.pip_size  # Small buffer
            self.zone_last_updated = candles[-1]["time"]

            self.Debug(f"Zone Updated: {self.demand_zone_bottom:.5f} - "
                       f"{self.demand_zone_top:.5f} "
                       f"(touches: {len(best_cluster)})")

    # ════════════════════════════════════════════════════════════════
    # RANGE FILTER
    # ════════════════════════════════════════════════════════════════

    def IsRanging(self):
        """
        Check if 1H price action is range-bound (not trending).
        Uses the ratio of net price movement to total range.
        If price hasn't moved far net vs. the range it's oscillated
        in, we're in a range.
        """
        if len(self.candles_1h) < self.range_lookback:
            return True  # Default to allowing trades during warmup

        candles = list(self.candles_1h)[-self.range_lookback:]

        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]

        total_range = max(highs) - min(lows)
        net_move = abs(closes[-1] - closes[0])

        # If net movement is small relative to the range, we're ranging
        # A trending market would show net_move close to total_range
        if total_range == 0:
            return True

        range_ratio = net_move / total_range

        # ratio < 0.3 means price went sideways (range)
        # ratio > 0.5 means price trended
        return range_ratio < 0.35

    # ════════════════════════════════════════════════════════════════
    # ENTRY LOGIC
    # ════════════════════════════════════════════════════════════════

    def CheckEntry(self, candle):
        """
        Check if current 15m candle qualifies as a bullish rejection
        from the demand zone.
        """
        # Already in a position?
        if self.Portfolio[self.symbol].Invested:
            return

        # Do we have a valid zone?
        if self.demand_zone_top is None or self.demand_zone_bottom is None:
            return

        # Is the market ranging? (1H context filter)
        if not self.IsRanging():
            return

        low = candle["low"]
        close = candle["close"]
        open_price = candle["open"]
        high = candle["high"]

        # Condition 1: The candle's low must wick INTO the demand zone
        if low > self.demand_zone_top:
            return  # Price didn't reach the zone
        if low < self.demand_zone_bottom - self.stop_loss_pips * self.pip_size:
            return  # Price blew through the zone — not a bounce

        # Condition 2: The candle must CLOSE above the zone (rejection)
        if close < self.demand_zone_top:
            return

        # Condition 3: Bullish candle (close > open)
        if close <= open_price:
            return

        # Condition 4: Long lower wick (rejection signature)
        candle_range = high - low
        if candle_range == 0:
            return

        lower_wick = min(open_price, close) - low
        wick_ratio = lower_wick / candle_range

        if wick_ratio < self.rejection_wick_ratio:
            return

        # ── ALL CONDITIONS MET — ENTER LONG ────────────────────────
        self.EnterLong(close)

    def EnterLong(self, entry_price):
        """Place a long trade with position sizing based on risk."""
        self.entry_price = entry_price
        self.stop_price = self.demand_zone_bottom - self.stop_loss_pips * self.pip_size
        self.target_price = entry_price + self.take_profit_pips * self.pip_size

        # Position sizing: risk 1% of equity
        risk_per_unit = entry_price - self.stop_price
        if risk_per_unit <= 0:
            return

        equity = self.Portfolio.TotalPortfolioValue
        risk_amount = equity * self.risk_per_trade
        quantity = int(risk_amount / risk_per_unit)

        if quantity <= 0:
            return

        self.MarketOrder(self.symbol, quantity)

        self.Debug(f"LONG @ {entry_price:.5f} | "
                   f"SL: {self.stop_price:.5f} | "
                   f"TP: {self.target_price:.5f} | "
                   f"Qty: {quantity}")

    def ResetTrade(self):
        """Clear trade tracking variables."""
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    # ════════════════════════════════════════════════════════════════
    # END OF DAY
    # ════════════════════════════════════════════════════════════════

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}")
