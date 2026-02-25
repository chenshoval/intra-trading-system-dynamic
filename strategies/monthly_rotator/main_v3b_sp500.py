"""Monthly Rotator v3b — S&P 500 Long-Short with Threshold

Expands from 50 hand-picked stocks to full S&P 500 (~500 stocks).
Long top 15 from the entire universe, short bottom 10 ONLY when they
score below 0.30 threshold (meaning they're actually bad, not just
"least good").

Key differences from v3:
- Universe: S&P 500 via QuantConnect's universe selection (not 50 hand-picked)
- Short threshold: only short stocks scoring below 0.30 (adaptive, not always-on)
- No event scoring: Tiingo doesn't cover small/mid-caps well, and with 500
  stocks the 90% price-based scoring is more than enough signal
- Bull market: long 15, short 0-3 (most stocks score well)
- Bear market: long 5, short 10 (plenty of weak stocks below threshold)

Based on Kelly/Xiu: long-short decile across broad universe = Sharpe 1.72
"""

from AlgorithmImports import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV3bSP500(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.long_n = 15
        self.max_short_n = 10               # max shorts (may be less if few below threshold)
        self.short_threshold = 0.30         # only short stocks scoring below this
        self.short_size_pct = 0.02          # 2% per short position
        self.downtrend_long_n = 5
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Signal weights (no events — 4 signals sum to 1.0) ──
        self.w_momentum = 0.45
        self.w_trend = 0.30
        self.w_recent_strength = 0.15
        self.w_volatility = 0.10

        # ── Momentum parameters ──
        self.mom_lookback = 126
        self.mom_skip = 21
        self.stock_ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        # ── Universe: dynamic S&P 500 ──
        self.universe_size = 500
        self.min_price = 5.0                # filter penny stocks
        self.active_universe = []           # populated by universe selection
        self.symbols = {}

        self.long_holdings = set()
        self.short_holdings = set()
        self.last_rebalance = None
        self.is_in_cash = False

        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)

        # ── Universe selection: top 500 US equities by dollar volume ──
        self.add_universe(self._coarse_selection)
        self.universe_settings.resolution = Resolution.DAILY

        # ── SPY for trend gate ──
        self.add_equity("SPY", Resolution.DAILY)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

        self.set_benchmark("SPY")

        self.schedule.on(
            self.date_rules.month_start("SPY", 0),
            self.time_rules.after_market_open("SPY", 45),
            self.monthly_rebalance,
        )
        self.schedule.on(
            self.date_rules.every(DayOfWeek.WEDNESDAY),
            self.time_rules.after_market_open("SPY", 60),
            self.midweek_trend_check,
        )

        self.debug(
            f">>> ROTATOR v3b S&P500 LONG-SHORT: long {self.long_n}, "
            f"max short {self.max_short_n}, threshold {self.short_threshold}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Universe selection
    # ══════════════════════════════════════════════════════════════════════

    def _coarse_selection(self, coarse):
        """Select top 500 US equities by dollar volume, price > $5."""
        filtered = [
            x for x in coarse
            if x.has_fundamental_data
            and x.price > self.min_price
            and x.dollar_volume > 1_000_000
        ]
        sorted_by_volume = sorted(filtered, key=lambda x: x.dollar_volume, reverse=True)
        selected = sorted_by_volume[:self.universe_size]

        self.active_universe = [x.symbol for x in selected]
        return self.active_universe

    # ══════════════════════════════════════════════════════════════════════
    # Trend gate
    # ══════════════════════════════════════════════════════════════════════

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Stock scoring (price-based only, no events)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks(self):
        """Score all stocks in universe on momentum + trend + recent + vol."""
        scores = {}
        raw_data = {}

        for symbol in self.active_universe:
            if symbol not in self.securities:
                continue
            if not self.securities[symbol].has_data:
                continue

            history = self.history(
                symbol, self.mom_lookback + self.mom_skip + 10, Resolution.DAILY
            )
            if history is None or history.empty:
                continue
            try:
                closes = history["close"].values
            except Exception:
                continue
            if len(closes) < self.mom_lookback + self.mom_skip:
                continue

            price_now = closes[-1]
            if price_now <= 0:
                continue

            # Signal 1: 6-month momentum (skip last month)
            price_6m = closes[0]
            price_1m = closes[-self.mom_skip]
            momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0

            # Signal 2: Stock above its own 50d MA
            if len(closes) >= self.stock_ma_period:
                ma_50 = np.mean(closes[-self.stock_ma_period:])
                trend_score = 1.0 if price_now > ma_50 else 0.0
            else:
                trend_score = 0.5

            # Signal 3: Recent 1-month strength
            if len(closes) >= self.recent_period:
                price_1m_ago = closes[-self.recent_period]
                recent = (price_now / price_1m_ago) - 1.0 if price_1m_ago > 0 else 0.0
            else:
                recent = 0.0

            # Signal 4: Volatility (lower = better)
            if len(closes) >= self.vol_period:
                returns = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1.0
            else:
                vol = 1.0

            ticker = str(symbol).split(' ')[0] if ' ' in str(symbol) else str(symbol)
            raw_data[symbol] = {
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol, "ticker": ticker,
            }

        if len(raw_data) < self.long_n:
            return scores

        symbols_list = list(raw_data.keys())
        n = len(symbols_list)

        # Rank-normalize
        for signal in ["momentum", "recent"]:
            ranked = sorted(symbols_list, key=lambda s: raw_data[s][signal])
            for i, s in enumerate(ranked):
                raw_data[s][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_vol = sorted(symbols_list, key=lambda s: raw_data[s]["vol"], reverse=True)
        for i, s in enumerate(ranked_vol):
            raw_data[s]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        # Composite score
        for symbol in symbols_list:
            d = raw_data[symbol]
            score = (
                self.w_momentum * d.get("momentum_rank", 0.5)
                + self.w_trend * d["trend"]
                + self.w_recent_strength * d.get("recent_rank", 0.5)
                + self.w_volatility * d.get("vol_rank", 0.5)
            )
            scores[symbol] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance
    # ══════════════════════════════════════════════════════════════════════

    def monthly_rebalance(self):
        self.total_rebalances += 1

        current_equity = self.portfolio.total_portfolio_value
        if self.last_rebalance is not None:
            month_ret = (current_equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)
            key = f"{self.last_rebalance.year}-{self.last_rebalance.month:02d}"
            self.monthly_pnl[key] = current_equity - self.month_start_equity

        self.month_start_equity = current_equity
        self.last_rebalance = self.time

        uptrend = self._is_uptrend()
        if uptrend:
            self.months_in_uptrend += 1
        else:
            self.months_in_downtrend += 1

        scores = self._score_stocks()
        if not scores:
            self.debug(f"  No scores (universe not ready?), skipping")
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Determine long positions
        n_long = self.long_n if uptrend else self.downtrend_long_n
        target_longs = set(s for s, _ in ranked[:n_long])

        # Determine short positions — only stocks below threshold
        potential_shorts = [(s, sc) for s, sc in ranked if sc < self.short_threshold]
        # Take the worst ones, up to max_short_n
        potential_shorts.sort(key=lambda x: x[1])  # lowest score first
        target_shorts = set(s for s, _ in potential_shorts[:self.max_short_n])

        # Don't short anything we're longing
        target_shorts -= target_longs

        self.is_in_cash = False

        # Close positions no longer wanted
        for symbol in list(self.long_holdings):
            if symbol not in target_longs:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.long_holdings.discard(symbol)

        for symbol in list(self.short_holdings):
            if symbol not in target_shorts:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.short_holdings.discard(symbol)

        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        # Long positions — equal weight
        long_alloc = total_value / n_long if n_long > 0 else 0
        for symbol in target_longs:
            if symbol not in self.securities:
                continue
            price = self.securities[symbol].price
            if price <= 0:
                continue
            target_qty = int(long_alloc / price)
            if target_qty < 1:
                continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1
            self.long_holdings.add(symbol)

        # Short positions — fixed % per position
        short_alloc = total_value * self.short_size_pct
        for symbol in target_shorts:
            if symbol not in self.securities:
                continue
            price = self.securities[symbol].price
            if price <= 0:
                continue
            target_qty = -int(short_alloc / price)
            if target_qty >= 0:
                continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1
            self.short_holdings.add(symbol)

        regime = "UP" if uptrend else "DOWN"
        n_scored = len(scores)
        n_below = len(potential_shorts)
        self.debug(
            f"REBALANCE [{regime}]: scored={n_scored} long={len(target_longs)} "
            f"short={len(target_shorts)} (below {self.short_threshold}: {n_below}), "
            f"eq=${total_value:,.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        if not self.long_holdings and not self.short_holdings:
            return

        uptrend = self._is_uptrend()

        if not uptrend and not self.is_in_cash:
            self.is_in_cash = True
            self.emergency_exits += 1

            # Reduce longs to top 5 by current score
            scores = self._score_stocks()
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                keep_long = set(s for s, _ in ranked[:self.downtrend_long_n])
            else:
                keep_long = set()

            for symbol in list(self.long_holdings):
                if symbol not in keep_long:
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.long_holdings.discard(symbol)

            # Keep shorts — they're making money in downtrend
            self.debug(
                f"EMERGENCY: downtrend, longs={len(self.long_holdings)} shorts={len(self.short_holdings)}"
            )

        elif uptrend and self.is_in_cash:
            self.is_in_cash = False
            self.debug(f"RECOVERY: uptrend restored")

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        current_equity = self.portfolio.total_portfolio_value
        if self.last_rebalance is not None:
            month_ret = (current_equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)

        ret_pct = self.portfolio.total_profit / 100_000

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} Final=${current_equity:,.0f} "
            f"Rebalances={self.total_rebalances} Trades={self.total_trades} "
            f"EmergencyExits={self.emergency_exits}"
        )
        self.debug(
            f"REGIME: Up={self.months_in_uptrend} Down={self.months_in_downtrend}"
        )

        if self.monthly_returns:
            rets = np.array(self.monthly_returns)
            win_months = np.sum(rets > 0)
            loss_months = np.sum(rets <= 0)
            avg_ret = np.mean(rets)
            std = np.std(rets)
            monthly_sharpe = avg_ret / std if std > 0 else 0

            self.debug(
                f"MONTHLY: Avg={avg_ret:.2%} Best={np.max(rets):.2%} Worst={np.min(rets):.2%} "
                f"Win={win_months} Loss={loss_months} WR={win_months/len(rets):.0%} "
                f"Sharpe={monthly_sharpe:.2f}"
            )

        if self.monthly_pnl:
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"PNL: {pnl_str}")
