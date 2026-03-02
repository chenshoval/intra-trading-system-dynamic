"""Monthly Rotator v9 — Fundamentals + Dynamic Universe

The "no known issues" version. Combines:
- v8's 8-signal scoring (momentum + trend + recent + vol + events + value + quality + leverage)
- Dynamic universe: auto-selects top 50 US stocks by dollar volume each month
  (eliminates survivorship bias from hand-picked tickers)

No more static ticker list. The universe evolves naturally:
- 2016: top 50 includes stocks that were big then, not 2026 winners
- New mega-caps auto-enter as they grow (NVDA, META post-IPO, etc.)
- Declining stocks auto-exit when they drop out of top 50

Event scoring dropped (was 5% in v8): dynamic universe means we can't
pre-subscribe to TiingoNews for unknown tickers. The 5% weight is
redistributed to momentum (now 30%). This is fine — pure momentum beat
events-only in all but bear markets, and fundamentals cover the bear case.

Signal weights (7 signals, sum to 1.0):
  momentum  0.30  (was 0.25 in v8)
  trend     0.15
  recent    0.15
  vol       0.10
  value     0.10  (earnings yield)
  quality   0.10  (ROE)
  leverage  0.10  (low debt-to-equity)
"""

from AlgorithmImports import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV9Dynamic(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.top_n = 15
        self.downtrend_top_n = 5
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Universe parameters ──
        self.universe_size = 50
        self.min_price = 10.0               # filter penny stocks
        self.min_dollar_volume = 10_000_000  # $10M daily volume minimum

        # ── Signal weights (7 signals, sum to 1.0) ──
        self.w_momentum = 0.30
        self.w_trend = 0.15
        self.w_recent_strength = 0.15
        self.w_volatility = 0.10
        self.w_value = 0.10
        self.w_quality = 0.10
        self.w_leverage = 0.10

        # ── Momentum parameters ──
        self.mom_lookback = 126
        self.mom_skip = 21
        self.stock_ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        # ── Data structures ──
        self.active_universe = []           # populated by universe selection
        self.current_holdings = set()       # symbols currently held
        self.last_rebalance = None
        self.is_in_cash = False

        # ── Counters ──
        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)
        self.fundamentals_available = 0
        self.fundamentals_missing = 0
        self.universe_sizes = []            # track how many stocks pass filter each month

        # ── Universe selection ──
        self.add_universe(self._coarse_selection, self._fine_selection)
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
            f">>> MONTHLY ROTATOR v9 (dynamic universe + fundamentals): "
            f"top {self.universe_size} by dollar volume, 7 signals"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Universe selection
    # ══════════════════════════════════════════════════════════════════════

    def _coarse_selection(self, coarse):
        """First filter: top 50 US equities by dollar volume, price > $10."""
        filtered = [
            x for x in coarse
            if x.has_fundamental_data
            and x.price > self.min_price
            and x.dollar_volume > self.min_dollar_volume
        ]
        sorted_by_volume = sorted(
            filtered, key=lambda x: x.dollar_volume, reverse=True
        )
        return [x.symbol for x in sorted_by_volume[:self.universe_size]]

    def _fine_selection(self, fine):
        """Second filter: require fundamental data availability."""
        # Accept all that pass coarse — fine selection ensures fundamentals load
        return [x.symbol for x in fine]

    def on_securities_changed(self, changes):
        """Track universe additions/removals."""
        for security in changes.added_securities:
            if security.symbol not in self.active_universe:
                self.active_universe.append(security.symbol)

        for security in changes.removed_securities:
            if security.symbol in self.active_universe:
                self.active_universe.remove(security.symbol)
            # Liquidate removed stocks if we hold them
            if security.symbol in self.current_holdings:
                if self.portfolio[security.symbol].invested:
                    self.liquidate(security.symbol)
                    self.total_trades += 1
                self.current_holdings.discard(security.symbol)

    # ══════════════════════════════════════════════════════════════════════
    # Trend gate
    # ══════════════════════════════════════════════════════════════════════

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Fundamental data
    # ══════════════════════════════════════════════════════════════════════

    def _get_fundamentals(self, symbol):
        """Extract fundamental data from QC's Fundamentals property."""
        if symbol not in self.securities:
            return None

        security = self.securities[symbol]
        if not security.fundamentals:
            return None

        f = security.fundamentals

        try:
            pe_ratio = f.valuation_ratios.pe_ratio
            earnings_yield = 1.0 / pe_ratio if pe_ratio and pe_ratio > 0 else 0.0

            roe = f.operation_ratios.roe.value if f.operation_ratios.roe else 0.0

            de_ratio = (
                f.operation_ratios.total_debt_equity_ratio.value
                if f.operation_ratios.total_debt_equity_ratio
                else 1.0
            )

            self.fundamentals_available += 1
            return {
                "earnings_yield": earnings_yield,
                "roe": roe,
                "de_ratio": de_ratio,
            }
        except Exception:
            self.fundamentals_missing += 1
            return None

    # ══════════════════════════════════════════════════════════════════════
    # Stock scoring (7 signals, no events)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks(self):
        """Score all stocks in dynamic universe."""
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

            # Signal 1: Momentum (6-month, skip last month)
            price_6m = closes[0]
            price_1m = closes[-self.mom_skip]
            momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0

            # Signal 2: Trend (above 50d MA)
            if len(closes) >= self.stock_ma_period:
                ma_50 = np.mean(closes[-self.stock_ma_period:])
                trend_score = 1.0 if price_now > ma_50 else 0.0
            else:
                trend_score = 0.5

            # Signal 3: Recent strength (1-month)
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

            # Signals 5-7: Fundamentals
            fund = self._get_fundamentals(symbol)
            earnings_yield = fund["earnings_yield"] if fund else 0.0
            roe = fund["roe"] if fund else 0.0
            de_ratio = fund["de_ratio"] if fund else 1.0

            raw_data[symbol] = {
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol,
                "earnings_yield": earnings_yield, "roe": roe, "de_ratio": de_ratio,
            }

        if len(raw_data) < self.top_n:
            return scores

        symbols_list = list(raw_data.keys())
        n = len(symbols_list)

        # Rank-normalize (higher = better)
        for signal in ["momentum", "recent", "earnings_yield", "roe"]:
            ranked = sorted(symbols_list, key=lambda s: raw_data[s][signal])
            for i, s in enumerate(ranked):
                raw_data[s][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        # Volatility: lower = better
        ranked_vol = sorted(symbols_list, key=lambda s: raw_data[s]["vol"], reverse=True)
        for i, s in enumerate(ranked_vol):
            raw_data[s]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        # Debt-to-equity: lower = better
        ranked_de = sorted(symbols_list, key=lambda s: raw_data[s]["de_ratio"], reverse=True)
        for i, s in enumerate(ranked_de):
            raw_data[s]["de_ratio_rank"] = i / (n - 1) if n > 1 else 0.5

        # Composite score — 7 signals
        for symbol in symbols_list:
            d = raw_data[symbol]
            score = (
                self.w_momentum * d.get("momentum_rank", 0.5)
                + self.w_trend * d["trend"]
                + self.w_recent_strength * d.get("recent_rank", 0.5)
                + self.w_volatility * d.get("vol_rank", 0.5)
                + self.w_value * d.get("earnings_yield_rank", 0.5)
                + self.w_quality * d.get("roe_rank", 0.5)
                + self.w_leverage * d.get("de_ratio_rank", 0.5)
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

        self.universe_sizes.append(len(self.active_universe))

        scores = self._score_stocks()
        if not scores:
            self.debug(f"  No scores (universe has {len(self.active_universe)} stocks), skipping")
            return

        n_hold = self.top_n if uptrend else self.downtrend_top_n
        self.is_in_cash = False

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_symbols = set(s for s, _ in ranked[:n_hold])

        # Sell positions not in new top N
        for symbol in list(self.current_holdings):
            if symbol not in target_symbols:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.current_holdings.discard(symbol)

        # Buy new positions
        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0 or n_hold <= 0:
            return

        target_alloc = total_value / n_hold

        for symbol in target_symbols:
            if symbol not in self.securities:
                continue
            price = self.securities[symbol].price
            if price <= 0:
                continue
            target_qty = int(target_alloc / price)
            if target_qty < 1:
                continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1
            self.current_holdings.add(symbol)

        # Log top picks with tickers
        top3_info = []
        for s, sc in ranked[:3]:
            ticker = str(s).split(' ')[0] if ' ' in str(s) else str(s)
            top3_info.append(f"{ticker}:{sc:.3f}")

        regime = "UP" if uptrend else "DOWN"
        self.debug(
            f"REBALANCE [{regime}]: universe={len(self.active_universe)} "
            f"scored={len(scores)} holding={n_hold} "
            f"top3=[{', '.join(top3_info)}] eq=${total_value:,.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        if not self.current_holdings:
            return

        uptrend = self._is_uptrend()

        if not uptrend and not self.is_in_cash:
            self.is_in_cash = True
            self.emergency_exits += 1

            scores = self._score_stocks()
            keep = set()
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                keep = set(s for s, _ in ranked[:self.downtrend_top_n])

            for symbol in list(self.current_holdings):
                if symbol not in keep:
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.current_holdings.discard(symbol)

            if keep:
                total_value = self.portfolio.total_portfolio_value
                target_alloc = total_value / len(keep)
                for symbol in keep:
                    if symbol not in self.securities:
                        continue
                    price = self.securities[symbol].price
                    if price <= 0:
                        continue
                    target_qty = int(target_alloc / price)
                    current_qty = int(self.portfolio[symbol].quantity)
                    delta = target_qty - current_qty
                    if abs(delta) > 0:
                        self.market_order(symbol, delta)
                        self.total_trades += 1

            self.debug(f"EMERGENCY: downtrend, reduced to {len(self.current_holdings)} positions")

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

        avg_universe = np.mean(self.universe_sizes) if self.universe_sizes else 0

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} Final=${current_equity:,.0f} "
            f"Rebalances={self.total_rebalances} Trades={self.total_trades} "
            f"EmergencyExits={self.emergency_exits} "
            f"AvgUniverse={avg_universe:.0f} "
            f"FundAvail={self.fundamentals_available} FundMiss={self.fundamentals_missing}"
        )
        self.debug(
            f"REGIME: Up={self.months_in_uptrend} Down={self.months_in_downtrend}"
        )

        if self.monthly_returns:
            rets = np.array(self.monthly_returns)
            win_months = np.sum(rets > 0)
            loss_months = np.sum(rets <= 0)
            avg_ret = np.mean(rets)
            median_ret = np.median(rets)
            best = np.max(rets)
            worst = np.min(rets)
            std = np.std(rets)
            monthly_sharpe = avg_ret / std if std > 0 else 0

            self.debug(
                f"MONTHLY: Avg={avg_ret:.2%} Med={median_ret:.2%} "
                f"Best={best:.2%} Worst={worst:.2%} Std={std:.2%} "
                f"Win={win_months} Loss={loss_months} "
                f"WR={win_months/len(rets):.0%} Sharpe={monthly_sharpe:.2f}"
            )

        if self.monthly_pnl:
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"PNL: {pnl_str}")
