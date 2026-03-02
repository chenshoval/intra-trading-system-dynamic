"""Monthly Rotator v9b — Fundamentals + Dynamic Universe (Fixed)

Fixes v9's problems (beta 2.0, 59% drawdown):
1. Sort by MARKET CAP instead of dollar volume (stable companies, not hyped)
2. Sector caps matching v2's allocation (max 14 tech, 7 finance, etc.)
3. Require POSITIVE EARNINGS (filters out money-losing bubble stocks)

Signal weights (7 signals, sum to 1.0):
  momentum  0.30
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


class MonthlyRotatorV9bFixed(QCAlgorithm):

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
        self.min_price = 10.0
        self.min_market_cap = 10_000_000_000  # $10B minimum market cap

        # ── Sector caps (matching v2's sector balance) ──
        # MorningstarSectorCode values:
        #   101=Basic Materials, 102=Consumer Cyclical, 103=Financial Services,
        #   104=Real Estate, 205=Consumer Defensive, 206=Healthcare,
        #   207=Utilities, 308=Communication Services, 309=Energy,
        #   310=Industrials, 311=Technology
        self.sector_caps = {
            311: 14,   # Technology — max 14
            103: 7,    # Financial Services — max 7
            206: 7,    # Healthcare — max 7
            102: 4,    # Consumer Cyclical — max 4
            205: 3,    # Consumer Defensive — max 3
            310: 5,    # Industrials — max 5
            309: 3,    # Energy — max 3
            308: 3,    # Communication Services — max 3
            101: 2,    # Basic Materials — max 2
            104: 1,    # Real Estate — max 1
            207: 1,    # Utilities — max 1
        }
        self.default_sector_cap = 2         # any unlisted sector

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
        self.active_universe = []
        self.current_holdings = set()
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
        self.universe_sizes = []
        self.sector_counts = defaultdict(int)  # track sector distribution

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
            f">>> MONTHLY ROTATOR v9b (market cap + sector caps + profitable): "
            f"top {self.universe_size}, 7 signals"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Universe selection — FIXED
    # ══════════════════════════════════════════════════════════════════════

    def _coarse_selection(self, coarse):
        """Filter: price > $10, has fundamentals, sorted by MARKET CAP."""
        filtered = [
            x for x in coarse
            if x.has_fundamental_data
            and x.price > self.min_price
            and x.market_cap > self.min_market_cap
        ]
        # Sort by market cap (largest first) — NOT dollar volume
        sorted_by_cap = sorted(filtered, key=lambda x: x.market_cap, reverse=True)
        # Take top 200 candidates for fine selection to apply sector caps
        return [x.symbol for x in sorted_by_cap[:200]]

    def _fine_selection(self, fine):
        """Apply sector caps and profitability filter."""
        sector_count = defaultdict(int)
        selected = []

        # Sort by market cap within fine selection
        sorted_fine = sorted(fine, key=lambda x: x.market_cap, reverse=True)

        for stock in sorted_fine:
            if len(selected) >= self.universe_size:
                break

            # Require positive earnings (EPS > 0)
            try:
                eps = stock.earning_reports.basic_eps.value
                if eps is None or eps <= 0:
                    continue
            except Exception:
                # If we can't read EPS, skip — be conservative
                continue

            # Apply sector cap
            sector = stock.asset_classification.morningstar_sector_code
            cap = self.sector_caps.get(sector, self.default_sector_cap)
            if sector_count[sector] >= cap:
                continue

            sector_count[sector] += 1
            selected.append(stock.symbol)

        self.sector_counts = dict(sector_count)
        return selected

    def on_securities_changed(self, changes):
        """Track universe additions/removals."""
        for security in changes.added_securities:
            if security.symbol not in self.active_universe:
                self.active_universe.append(security.symbol)

        for security in changes.removed_securities:
            if security.symbol in self.active_universe:
                self.active_universe.remove(security.symbol)
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
            return {"earnings_yield": earnings_yield, "roe": roe, "de_ratio": de_ratio}
        except Exception:
            self.fundamentals_missing += 1
            return None

    # ══════════════════════════════════════════════════════════════════════
    # Stock scoring (7 signals)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks(self):
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

            price_6m = closes[0]
            price_1m = closes[-self.mom_skip]
            momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0

            if len(closes) >= self.stock_ma_period:
                ma_50 = np.mean(closes[-self.stock_ma_period:])
                trend_score = 1.0 if price_now > ma_50 else 0.0
            else:
                trend_score = 0.5

            if len(closes) >= self.recent_period:
                price_1m_ago = closes[-self.recent_period]
                recent = (price_now / price_1m_ago) - 1.0 if price_1m_ago > 0 else 0.0
            else:
                recent = 0.0

            if len(closes) >= self.vol_period:
                returns = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1.0
            else:
                vol = 1.0

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

        for signal in ["momentum", "recent", "earnings_yield", "roe"]:
            ranked = sorted(symbols_list, key=lambda s: raw_data[s][signal])
            for i, s in enumerate(ranked):
                raw_data[s][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_vol = sorted(symbols_list, key=lambda s: raw_data[s]["vol"], reverse=True)
        for i, s in enumerate(ranked_vol):
            raw_data[s]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_de = sorted(symbols_list, key=lambda s: raw_data[s]["de_ratio"], reverse=True)
        for i, s in enumerate(ranked_de):
            raw_data[s]["de_ratio_rank"] = i / (n - 1) if n > 1 else 0.5

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
            self.debug(f"  No scores (universe={len(self.active_universe)}), skipping")
            return

        n_hold = self.top_n if uptrend else self.downtrend_top_n
        self.is_in_cash = False

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_symbols = set(s for s, _ in ranked[:n_hold])

        for symbol in list(self.current_holdings):
            if symbol not in target_symbols:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.current_holdings.discard(symbol)

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

        top3_info = []
        for s, sc in ranked[:3]:
            ticker = str(s).split(' ')[0] if ' ' in str(s) else str(s)
            top3_info.append(f"{ticker}:{sc:.3f}")

        regime = "UP" if uptrend else "DOWN"
        sectors_str = str(dict(self.sector_counts))
        self.debug(
            f"REBALANCE [{regime}]: universe={len(self.active_universe)} "
            f"scored={len(scores)} holding={n_hold} "
            f"top3=[{', '.join(top3_info)}] eq=${total_value:,.0f} "
            f"sectors={sectors_str}"
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
            self.debug(f"EMERGENCY: downtrend, reduced to {len(self.current_holdings)}")
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
            f"EmergExits={self.emergency_exits} AvgUniverse={avg_universe:.0f} "
            f"FundAvail={self.fundamentals_available} FundMiss={self.fundamentals_missing}"
        )
        self.debug(f"REGIME: Up={self.months_in_uptrend} Down={self.months_in_downtrend}")

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
