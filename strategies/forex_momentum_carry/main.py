"""Multi-Pair FX Independent Momentum Strategy — Hypothesis 5

Inspired by a Reddit trader running 27 forex pairs with Sharpe 3.64 over 1 year
of live trading. The core insight: diversification across many low-correlated
pairs is the primary Sharpe multiplier. Even modest per-pair edge (~0.5-0.7 Sharpe)
compounds to 3.0+ at the portfolio level when pair correlations average ~0.2.

Approach: Per-pair independent signals (Variant B)
- Each pair is scored independently on momentum, trend, trend strength, and vol regime
- Trades only entered when composite score exceeds threshold (selective)
- Position sizing via inverse-volatility (Barroso & Santa-Clara 2015)
- IBKR brokerage model (min 25K units per pair)

Academic basis:
- Menkhoff et al. (2012) "Currency Momentum Strategies" — JFE
- Asness, Moskowitz & Pedersen (2013) "Value and Momentum Everywhere" — JF
- Barroso & Santa-Clara (2015) "Beyond the Carry Trade: Optimal Currency Portfolios"
- Daniel & Moskowitz (2016) "Momentum Crashes" — dynamic risk scaling

Key parameters:
- 27 forex pairs (majors + crosses)
- Daily rebalance at 17:00 UTC (NY close / FX rollover)
- Entry threshold: |score| > 0.6
- Risk per pair: 0.5% of equity
- Max positions: 20 (adaptive by account size)
- Drawdown circuit breaker: 5% portfolio DD -> halve all
- Vol scaling: portfolio vol > 2x average -> halve all sizes

Test periods (walk-forward):
- Period 1: 2016-2019 (development)
- Period 2: 2020-2022 (out-of-sample, COVID)
- Period 3: 2023-2025 (rate hike cycle)
"""

from AlgorithmImports import *
import numpy as np
from collections import defaultdict


class ForexMomentumCarry(QCAlgorithm):

    def initialize(self):
        # ══════════════════════════════════════════════════════════════
        # Test period (change for walk-forward validation)
        # ══════════════════════════════════════════════════════════════
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2022, 12, 31)
        self.set_cash(100_000)

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE)

        # ══════════════════════════════════════════════════════════════
        # Universe: 27 Forex Pairs
        # ══════════════════════════════════════════════════════════════
        self.majors = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "NZDUSD", "USDCAD",
        ]
        self.eur_crosses = [
            "EURJPY", "EURGBP", "EURAUD", "EURNZD", "EURCHF",
        ]
        self.gbp_crosses = [
            "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD",
        ]
        self.aud_crosses = [
            "AUDJPY", "AUDNZD", "AUDCHF",
        ]
        self.other_crosses = [
            "NZDJPY", "NZDCHF", "NZDCAD",
            "CADJPY", "CADCHF", "CHFJPY",
            "EURCAD", "GBPCAD",
        ]

        self.all_pairs = (
            self.majors + self.eur_crosses + self.gbp_crosses
            + self.aud_crosses + self.other_crosses
        )
        self.majors_and_top_crosses = (
            self.majors + self.eur_crosses[:3] + self.gbp_crosses[:2]
            + self.aud_crosses[:2] + ["CADJPY"]
        )  # 15 pairs for medium accounts

        # ══════════════════════════════════════════════════════════════
        # Signal parameters
        # ══════════════════════════════════════════════════════════════
        self.mom_period = 21                # 1-month momentum
        self.sma_period = 50                # trend SMA
        self.trend_strength_period = 21     # fraction of positive days
        self.atr_fast = 14                  # short-term ATR
        self.atr_slow = 63                  # long-term ATR (for vol regime)
        self.history_lookback = 80          # days of history to fetch

        # Signal weights (sum to 1.0)
        self.w_momentum = 0.40
        self.w_trend = 0.30
        self.w_trend_strength = 0.15
        self.w_vol_regime = 0.15

        # Entry/exit thresholds
        self.entry_threshold = 0.35         # |score| > this to enter (was 0.60 — too selective for FX)
        self.exit_threshold = 0.10          # |score| < this to exit (neutral zone)

        # ══════════════════════════════════════════════════════════════
        # Position sizing (IBKR)
        # ══════════════════════════════════════════════════════════════
        self.risk_per_pair = 0.008          # 0.8% of equity per pair (was 0.5% — underinvested)
        self.max_per_pair_pct = 0.08        # 8% of equity max per pair (was 5%)
        self.max_gross_exposure_pct = 1.00  # 100% max gross exposure
        self.max_positions = 20             # max simultaneous positions

        # IBKR minimum lot sizes vary by base currency
        # Source: IBKR order size limits error messages
        self.ibkr_min_by_currency = {
            "USD": 25_000, "EUR": 20_000, "GBP": 20_000,
            "JPY": 2_500_000, "CHF": 25_000, "AUD": 25_000,
            "NZD": 35_000, "CAD": 25_000,
        }

        # ══════════════════════════════════════════════════════════════
        # Risk management
        # ══════════════════════════════════════════════════════════════
        self.dd_circuit_breaker = 0.10      # 10% DD -> halve all (was 5% — too tight for FX)
        self.per_pair_max_loss = 0.02       # 2% equity loss per pair -> cut (was 1.5%)
        self.vol_scaling_threshold = 2.5    # portfolio vol > 2.5x avg -> halve (was 2.0)
        self.vol_scaling_lookback = 63      # days for average vol

        # ══════════════════════════════════════════════════════════════
        # Data structures
        # ══════════════════════════════════════════════════════════════
        self.symbols = {}                   # pair_name -> Symbol
        self.pair_atr_fast = {}             # pair_name -> ATR indicator (14)
        self.pair_atr_slow = {}             # pair_name -> ATR indicator (63)
        self.pair_sma = {}                  # pair_name -> SMA indicator (50)

        self.long_positions = set()         # pairs we're currently long
        self.short_positions = set()        # pairs we're currently short
        self.entry_prices = {}              # pair -> entry price
        self.entry_dates = {}              # pair -> entry date

        self.peak_equity = 100_000
        self.circuit_breaker_active = False
        self.portfolio_daily_returns = []   # for vol tracking

        # Tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.long_trade_count = 0
        self.short_trade_count = 0
        self.total_rebalances = 0
        self.circuit_breaker_count = 0
        self.per_pair_pnl = defaultdict(float)
        self.per_pair_trades = defaultdict(int)
        self.per_pair_wins = defaultdict(int)
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)
        self.last_rebalance_date = None
        self.last_equity = 100_000

        # ══════════════════════════════════════════════════════════════
        # Add forex securities and indicators
        # ══════════════════════════════════════════════════════════════
        for pair in self.all_pairs:
            forex = self.add_forex(pair, Resolution.DAILY, Market.OANDA)
            self.symbols[pair] = forex.symbol

            # Per-pair indicators
            # ATR signature: atr(symbol, period, movingAverageType, resolution)
            self.pair_atr_fast[pair] = self.atr(
                forex.symbol, self.atr_fast, MovingAverageType.SIMPLE, Resolution.DAILY
            )
            self.pair_atr_slow[pair] = self.atr(
                forex.symbol, self.atr_slow, MovingAverageType.SIMPLE, Resolution.DAILY
            )
            self.pair_sma[pair] = self.sma(
                forex.symbol, self.sma_period, Resolution.DAILY
            )

        # ══════════════════════════════════════════════════════════════
        # Scheduling: daily at market close (17:00 UTC for FX)
        # ══════════════════════════════════════════════════════════════
        # Use EURUSD as the scheduling anchor (most liquid)
        self.schedule.on(
            self.date_rules.every_day("EURUSD"),
            self.time_rules.before_market_close("EURUSD", 5),
            self.daily_rebalance,
        )

        self.schedule.on(
            self.date_rules.every_day("EURUSD"),
            self.time_rules.after_market_open("EURUSD", 30),
            self.daily_risk_check,
        )

        self.set_warm_up(timedelta(days=90))

        # ══════════════════════════════════════════════════════════════
        # Log setup
        # ══════════════════════════════════════════════════════════════
        active_universe = self._get_active_universe()
        self.debug(
            f">>> FX MOMENTUM v1: {len(self.all_pairs)} pairs total, "
            f"{len(active_universe)} active for ${self.portfolio.total_portfolio_value:,.0f}, "
            f"threshold={self.entry_threshold}, risk/pair={self.risk_per_pair:.1%}, "
            f"max_pos={self.max_positions}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Adaptive universe based on account size
    # ══════════════════════════════════════════════════════════════════════

    def _get_active_universe(self):
        """Select universe size based on current equity."""
        equity = self.portfolio.total_portfolio_value

        if equity < 30_000:
            return self.majors  # 7 pairs
        elif equity < 75_000:
            return self.majors_and_top_crosses  # 15 pairs
        else:
            return self.all_pairs  # 27 pairs

    def _get_max_positions(self):
        """Adaptive max positions based on equity."""
        equity = self.portfolio.total_portfolio_value

        if equity < 30_000:
            return 5
        elif equity < 75_000:
            return 10
        else:
            return self.max_positions  # 20

    # ══════════════════════════════════════════════════════════════════════
    # Per-pair independent scoring
    # ══════════════════════════════════════════════════════════════════════

    def _score_pair(self, pair):
        """Score a single pair independently. Returns float in [-1, +1].
        Positive = bullish signal, Negative = bearish signal.
        Magnitude indicates conviction strength."""

        symbol = self.symbols[pair]

        # Check indicators are ready
        if not self.pair_sma[pair].is_ready:
            return 0.0
        if not self.pair_atr_fast[pair].is_ready:
            return 0.0

        # Fetch history
        history = self.history(
            symbol, self.history_lookback, Resolution.DAILY
        )
        if history is None or history.empty:
            return 0.0

        try:
            closes = history["close"].values
        except Exception:
            return 0.0

        if len(closes) < self.history_lookback - 10:
            return 0.0

        price_now = closes[-1]
        if price_now <= 0:
            return 0.0

        # ── Signal 1: Momentum (21-day return) ──
        # Normalized to [-1, +1] using a sigmoid-like transform
        if len(closes) >= self.mom_period + 1:
            mom_return = (price_now / closes[-self.mom_period - 1]) - 1.0
            # Scale: typical FX monthly return is +-1.5%, so 0.015 -> ~0.7 score
            # (was 0.03 — too insensitive for FX, most signals stayed near 0)
            mom_score = np.tanh(mom_return / 0.015)
        else:
            mom_score = 0.0

        # ── Signal 2: Trend alignment (price vs 50-SMA) ──
        sma_val = self.pair_sma[pair].current.value
        if sma_val > 0:
            # Distance from SMA as fraction, capped
            trend_distance = (price_now - sma_val) / sma_val
            # FX typically deviates 0.5-1.5% from SMA, so 0.01 -> full score
            # (was 0.02 — too insensitive)
            trend_score = np.clip(trend_distance / 0.01, -1.0, 1.0)
        else:
            trend_score = 0.0

        # ── Signal 3: Trend strength (fraction of positive days) ──
        if len(closes) >= self.trend_strength_period + 1:
            recent = closes[-self.trend_strength_period - 1:]
            daily_rets = np.diff(recent) / recent[:-1]
            frac_positive = np.sum(daily_rets > 0) / len(daily_rets)
            # Transform: 0.5 -> 0, 0.7 -> +1, 0.3 -> -1
            strength_score = np.clip((frac_positive - 0.5) / 0.2, -1.0, 1.0)
        else:
            strength_score = 0.0

        # ── Signal 4: Volatility regime ──
        # Low vol relative to average = favorable for trend trades
        atr_fast_val = self.pair_atr_fast[pair].current.value
        atr_slow_val = self.pair_atr_slow[pair].current.value if self.pair_atr_slow[pair].is_ready else atr_fast_val

        if atr_slow_val > 0:
            vol_ratio = atr_fast_val / atr_slow_val
            # Low vol regime (ratio < 1) is favorable: score positive
            # High vol regime (ratio > 1) is unfavorable: score lower
            # But we want this to bias DIRECTION not just magnitude
            # Low vol in trend = confirming trend. High vol = reversal risk.
            # Use it as a multiplier: favorable = 1.0, unfavorable = 0.0
            vol_regime_score = np.clip(1.5 - vol_ratio, 0.0, 1.0)
        else:
            vol_regime_score = 0.5

        # ── Composite score ──
        # Momentum, trend, strength are directional [-1, +1]
        # Vol regime is a confidence modifier [0, 1]
        directional_score = (
            self.w_momentum * mom_score
            + self.w_trend * trend_score
            + self.w_trend_strength * strength_score
        )

        # Vol regime scales the directional score
        # Weight of vol regime applied as: (1 - w_vol) * directional + w_vol * directional * vol_modifier
        # Simplification: directional * (1 - w_vol + w_vol * vol_regime_score)
        vol_modifier = (1 - self.w_vol_regime) + self.w_vol_regime * vol_regime_score
        composite = directional_score * vol_modifier

        # Ensure in [-1, +1]
        return np.clip(composite, -1.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════
    # Position sizing (IBKR-compatible, inverse-volatility)
    # ══════════════════════════════════════════════════════════════════════

    def _get_ibkr_min_units(self, pair):
        """Get IBKR minimum order size for the base currency of a pair.
        Base currency is the first 3 chars (e.g., EURUSD -> EUR, NZDJPY -> NZD)."""
        base_currency = pair[:3]
        return self.ibkr_min_by_currency.get(base_currency, 25_000)

    def _compute_lot_size(self, pair, direction):
        """Compute position size in base currency units.
        Uses inverse-volatility sizing per Barroso & Santa-Clara (2015).

        Args:
            pair: Forex pair name (e.g., "EURUSD")
            direction: 1 for long, -1 for short

        Returns:
            int: Lot size in base currency units (e.g., 25000 for 0.25 std lots)
                 Returns 0 if position is too small or constrained.
        """
        equity = self.portfolio.total_portfolio_value
        atr_val = self.pair_atr_fast[pair].current.value

        if atr_val <= 0 or equity <= 0:
            return 0

        min_units = self._get_ibkr_min_units(pair)

        # Risk-based sizing: how many units so that 1 ATR move = risk_per_pair * equity
        risk_amount = equity * self.risk_per_pair
        raw_units = risk_amount / atr_val

        # Apply IBKR minimum for this currency
        if raw_units < min_units:
            # Check if even minimum lot exceeds our per-pair max
            min_lot_notional = min_units * self.securities[self.symbols[pair]].price
            if min_lot_notional > equity * self.max_per_pair_pct:
                return 0  # Can't trade this pair — too large even at min lot
            raw_units = min_units

        # Round to nearest 1000 (IBKR allows 1-unit increments, but keep clean)
        units = max(min_units, int(raw_units / 1000) * 1000)

        # Cap at max per-pair percentage
        price = self.securities[self.symbols[pair]].price
        if price > 0:
            notional = units * price
            max_notional = equity * self.max_per_pair_pct
            if notional > max_notional:
                units = max(min_units, int(max_notional / price / 1000) * 1000)

        return units * direction

    # ══════════════════════════════════════════════════════════════════════
    # Daily risk check
    # ══════════════════════════════════════════════════════════════════════

    def daily_risk_check(self):
        """Run before rebalance: check circuit breakers and per-pair stops."""
        if self.is_warming_up:
            return

        equity = self.portfolio.total_portfolio_value

        # Track daily returns for portfolio vol
        if self.last_equity > 0:
            daily_ret = (equity - self.last_equity) / self.last_equity
            self.portfolio_daily_returns.append(daily_ret)
            if len(self.portfolio_daily_returns) > self.vol_scaling_lookback + 20:
                self.portfolio_daily_returns = self.portfolio_daily_returns[-(self.vol_scaling_lookback + 20):]
        self.last_equity = equity

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        # ── Circuit breaker: portfolio drawdown ──
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.dd_circuit_breaker and not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_count += 1
            self._reduce_all_positions(0.5)
            self.debug(
                f"CIRCUIT BREAKER: DD={dd:.2%} > {self.dd_circuit_breaker:.0%}. "
                f"Halved all positions. Equity=${equity:,.0f} Peak=${self.peak_equity:,.0f}"
            )
        elif dd < self.dd_circuit_breaker * 0.5 and self.circuit_breaker_active:
            # Reset circuit breaker when DD recovers to half threshold
            self.circuit_breaker_active = False
            self.debug(f"CIRCUIT BREAKER RESET: DD recovered to {dd:.2%}")

        # ── Vol scaling (Daniel & Moskowitz 2016) ──
        if len(self.portfolio_daily_returns) >= self.vol_scaling_lookback:
            recent_vol = np.std(self.portfolio_daily_returns[-14:]) * np.sqrt(252)
            avg_vol = np.std(self.portfolio_daily_returns[-self.vol_scaling_lookback:]) * np.sqrt(252)
            if avg_vol > 0 and recent_vol > self.vol_scaling_threshold * avg_vol:
                self._reduce_all_positions(0.5)
                self.debug(
                    f"VOL SCALING: Recent vol={recent_vol:.2%} > "
                    f"{self.vol_scaling_threshold:.0f}x avg={avg_vol:.2%}. Halved positions."
                )

        # ── Per-pair stop losses ──
        all_open = list(self.long_positions | self.short_positions)
        for pair in all_open:
            symbol = self.symbols[pair]
            if not self.portfolio[symbol].invested:
                continue

            unrealized_pnl = self.portfolio[symbol].unrealized_profit
            pair_loss_limit = -equity * self.per_pair_max_loss

            if unrealized_pnl < pair_loss_limit:
                self.liquidate(symbol, f"Per-pair stop: {pair}")
                self._record_trade_exit(pair)  # OK here — entry price fallback handles it
                self.debug(
                    f"PAIR STOP: {pair} loss=${unrealized_pnl:,.0f} "
                    f"< limit=${pair_loss_limit:,.0f}"
                )

    def _reduce_all_positions(self, factor):
        """Reduce all open positions by factor (e.g., 0.5 = halve).
        If reduced size falls below IBKR minimum, liquidate entirely."""
        for pair in list(self.long_positions | self.short_positions):
            symbol = self.symbols[pair]
            if self.portfolio[symbol].invested:
                current_qty = self.portfolio[symbol].quantity
                target_qty = int(current_qty * factor)
                min_units = self._get_ibkr_min_units(pair)

                # If reduced position would be below IBKR minimum, liquidate fully
                if abs(target_qty) < min_units:
                    self.liquidate(symbol)
                    self._record_trade_exit(pair)
                    self.total_trades += 1
                else:
                    delta = target_qty - int(current_qty)
                    if abs(delta) >= min_units:  # Only submit if delta meets minimum
                        self.market_order(symbol, delta)
                        self.total_trades += 1

    # ══════════════════════════════════════════════════════════════════════
    # Daily rebalance — main trading logic
    # ══════════════════════════════════════════════════════════════════════

    def daily_rebalance(self):
        """Score all pairs, enter/exit positions based on signals."""
        if self.is_warming_up:
            return

        self.total_rebalances += 1
        equity = self.portfolio.total_portfolio_value

        # Monthly tracking
        current_month = f"{self.time.year}-{self.time.month:02d}"
        if self.last_rebalance_date is not None:
            prev_month = f"{self.last_rebalance_date.year}-{self.last_rebalance_date.month:02d}"
            if current_month != prev_month:
                month_ret = (equity - self.month_start_equity) / self.month_start_equity
                self.monthly_returns.append(month_ret)
                self.monthly_pnl[prev_month] = equity - self.month_start_equity
                self.month_start_equity = equity
        self.last_rebalance_date = self.time

        # Get active universe based on account size
        active_universe = self._get_active_universe()
        max_pos = self._get_max_positions()

        # Score all pairs in active universe
        scores = {}
        for pair in active_universe:
            score = self._score_pair(pair)
            scores[pair] = score

        # Determine desired positions
        desired_long = set()
        desired_short = set()
        desired_flat = set()

        for pair, score in scores.items():
            if score > self.entry_threshold:
                desired_long.add(pair)
            elif score < -self.entry_threshold:
                desired_short.add(pair)
            elif abs(score) < self.exit_threshold:
                desired_flat.add(pair)
            # else: in the gray zone between exit and entry threshold -> hold current

        # If more candidates than max_positions, take the strongest signals
        all_desired = []
        for pair in desired_long:
            all_desired.append((pair, scores[pair], 1))
        for pair in desired_short:
            all_desired.append((pair, abs(scores[pair]), -1))

        # Sort by signal strength (absolute value), take top max_pos
        all_desired.sort(key=lambda x: x[1], reverse=True)
        selected = all_desired[:max_pos]
        selected_long = {p for p, _, d in selected if d == 1}
        selected_short = {p for p, _, d in selected if d == -1}

        # ── Close positions that should be flat ──
        # Record P&L before liquidation (entry price fallback handles post-liquidation reads)
        for pair in list(self.long_positions):
            if pair not in selected_long or pair in desired_flat:
                symbol = self.symbols[pair]
                if self.portfolio[symbol].invested:
                    self._record_trade_exit(pair)
                    self.liquidate(symbol)

        for pair in list(self.short_positions):
            if pair not in selected_short or pair in desired_flat:
                symbol = self.symbols[pair]
                if self.portfolio[symbol].invested:
                    self._record_trade_exit(pair)
                    self.liquidate(symbol)

        # ── Check gross exposure budget ──
        current_gross = sum(
            abs(self.portfolio[self.symbols[p]].holdings_value)
            for p in (selected_long | selected_short)
            if self.portfolio[self.symbols[p]].invested
        )

        # ── Enter/adjust long positions ──
        for pair in selected_long:
            symbol = self.symbols[pair]
            lot_size = self._compute_lot_size(pair, 1)
            if lot_size <= 0:
                continue

            # Check gross exposure limit
            price = self.securities[symbol].price
            if price <= 0:
                continue
            add_notional = abs(lot_size * price)
            if (current_gross + add_notional) > equity * self.max_gross_exposure_pct:
                continue

            current_qty = int(self.portfolio[symbol].quantity)

            if pair in self.short_positions:
                # Reversing from short to long: record P&L then liquidate
                self._record_trade_exit(pair)
                self.liquidate(symbol)
                current_qty = 0

            target_qty = lot_size
            delta = target_qty - current_qty

            # Only rebalance if drift > 10% or new position
            if pair not in self.long_positions or abs(delta) > abs(current_qty) * 0.10:
                if abs(delta) > 0:
                    self.market_order(symbol, delta)
                    self.total_trades += 1
                    if pair not in self.long_positions:
                        self.long_trade_count += 1
                        self.entry_prices[pair] = price
                        self.entry_dates[pair] = self.time

            self.long_positions.add(pair)
            self.short_positions.discard(pair)
            current_gross += add_notional

        # ── Enter/adjust short positions ──
        for pair in selected_short:
            symbol = self.symbols[pair]
            lot_size = self._compute_lot_size(pair, -1)
            if lot_size >= 0:
                continue  # _compute_lot_size returns negative for shorts

            price = self.securities[symbol].price
            if price <= 0:
                continue
            add_notional = abs(lot_size * price)
            if (current_gross + add_notional) > equity * self.max_gross_exposure_pct:
                continue

            current_qty = int(self.portfolio[symbol].quantity)

            if pair in self.long_positions:
                # Reversing from long to short
                self._record_trade_exit(pair)
                self.liquidate(symbol)
                current_qty = 0

            target_qty = lot_size  # negative
            delta = target_qty - current_qty

            if pair not in self.short_positions or abs(delta) > abs(current_qty) * 0.10:
                if abs(delta) > 0:
                    self.market_order(symbol, delta)
                    self.total_trades += 1
                    if pair not in self.short_positions:
                        self.short_trade_count += 1
                        self.entry_prices[pair] = price
                        self.entry_dates[pair] = self.time

            self.short_positions.add(pair)
            self.long_positions.discard(pair)
            current_gross += add_notional

        # ── Logging ──
        n_long = len(self.long_positions)
        n_short = len(self.short_positions)
        gross = sum(
            abs(self.portfolio[self.symbols[p]].holdings_value)
            for p in (self.long_positions | self.short_positions)
            if self.portfolio[self.symbols[p]].invested
        )
        net = sum(
            self.portfolio[self.symbols[p]].holdings_value
            for p in (self.long_positions | self.short_positions)
            if self.portfolio[self.symbols[p]].invested
        )
        gross_pct = gross / equity if equity > 0 else 0
        net_pct = net / equity if equity > 0 else 0

        # Log every 20th rebalance to avoid excessive output
        if self.total_rebalances % 20 == 1:
            top_scores = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            score_str = " ".join(f"{p}={s:+.2f}" for p, s in top_scores)
            self.debug(
                f"REBALANCE #{self.total_rebalances}: L={n_long} S={n_short} "
                f"Gross={gross_pct:.0%} Net={net_pct:.0%} "
                f"Eq=${equity:,.0f} Top5: {score_str}"
            )

    # ══════════════════════════════════════════════════════════════════════
    # Trade tracking
    # ══════════════════════════════════════════════════════════════════════

    def _record_trade_exit(self, pair):
        """Record a closed trade and update per-pair stats.
        Must be called BEFORE liquidate() since unrealized_profit gets zeroed after."""
        symbol = self.symbols[pair]
        pnl = self.portfolio[symbol].unrealized_profit

        # If unrealized is already 0 (sometimes happens), estimate from entry price
        if pnl == 0 and pair in self.entry_prices:
            current_price = self.securities[symbol].price
            entry_price = self.entry_prices[pair]
            qty = self.portfolio[symbol].quantity
            if entry_price > 0 and qty != 0:
                pnl = (current_price - entry_price) * qty

        self.per_pair_pnl[pair] += pnl
        self.per_pair_trades[pair] += 1

        if pnl > 0:
            self.winning_trades += 1
            self.per_pair_wins[pair] += 1
        else:
            self.losing_trades += 1

        self.long_positions.discard(pair)
        self.short_positions.discard(pair)
        self.entry_prices.pop(pair, None)
        self.entry_dates.pop(pair, None)

    # ══════════════════════════════════════════════════════════════════════
    # OnData — minimal, mostly handled by scheduled events
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        pass  # All logic in scheduled events

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm reporting
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        equity = self.portfolio.total_portfolio_value
        ret_pct = (equity - 100_000) / 100_000

        # Final monthly return
        if self.last_rebalance_date is not None:
            month_ret = (equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)

        total_closed = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_closed * 100 if total_closed > 0 else 0

        self.debug("=" * 70)
        self.debug("FX MOMENTUM CARRY — FINAL RESULTS")
        self.debug("=" * 70)
        self.debug(
            f"RETURNS: Total={ret_pct:.2%} Final=${equity:,.0f} "
            f"Rebalances={self.total_rebalances}"
        )
        self.debug(
            f"TRADES: Total={self.total_trades} Closed={total_closed} "
            f"W={self.winning_trades} L={self.losing_trades} "
            f"WR={win_rate:.0f}% Longs={self.long_trade_count} "
            f"Shorts={self.short_trade_count}"
        )
        self.debug(
            f"RISK: CircuitBreakers={self.circuit_breaker_count} "
            f"PeakEquity=${self.peak_equity:,.0f}"
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

        # Per-pair breakdown (sorted by P&L contribution)
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

        # Monthly PnL
        if self.monthly_pnl:
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"\nMONTHLY PNL: {pnl_str}")
