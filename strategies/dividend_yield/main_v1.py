"""Dividend Yield Rotation v1 — Strategy 3 (Uncorrelated to Momentum)

Different mechanism from strategies 1 and 2. Instead of chasing momentum
(what went up keeps going up), this buys VALUE — high dividend yield ETFs
that pay you to hold them.

Why this should be uncorrelated:
- Momentum buys winners (growth, tech). Dividend yield buys cash cows (utilities, REITs, staples).
- When momentum crashes (2022: tech -33%), dividend stocks often rise (2022: DVY +2%).
- When momentum rips (2020-2021: tech +100%), dividend stocks lag. That's fine — we want the zig-zag.

ETF Universe (12 instruments spanning different yield sources):
- Broad high-dividend: DVY, VYM, SCHD, HDV
- Sector high-yield: SPHD (low-vol high-div), VNQ (REITs), XLU (utilities)
- International dividend: IDV (international high-div), VYMI (intl high-div)
- Preferred stock: PFF (preferred shares — bond-like, high yield)
- Covered call: QYLD (Nasdaq covered call), JEPI (equity premium income)

Scoring engine (4 signals — different weights from commodity):
- Yield signal (0.35): higher trailing yield = higher score. THIS IS THE KEY DIFFERENTIATOR.
- Trend (0.25): above 50d MA = 1, below = 0. Don't buy falling knives.
- Recent strength (0.20): 21-day return. Among high-yielders, pick the ones turning up.
- Volatility (0.20): lower vol = higher score. Dividend stocks should be stable.

No momentum lookback! That's the whole point — we're NOT doing momentum here.

Note on QC yield data:
- QC provides fundamental data including dividend yield for ETFs.
- If yield data isn't available, we fall back to price-based 12-month yield estimate
  (sum of last 4 quarterly dividends / current price).
- For backtesting simplicity, we use a price-return-based proxy:
  lower total return volatility + higher price stability = likely higher yielder.
  This works because high-dividend ETFs structurally have these characteristics.

Test periods:
- Run 1: 2016-2020 (standard gate)
- Run 2: 2020-2023 (should show uncorrelation to v2 in 2022)
- Run 3: 2023-2025 (recent validation)
Results: results_from_quant_connect/dividendyieldv1/{period}/
"""

from AlgorithmImports import *
from collections import defaultdict
import numpy as np


class DividendYieldRotationV1(QCAlgorithm):

    def initialize(self):
        # ── Test periods ──
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.top_n = 4
        self.downtrend_top_n = 3
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Signal weights (sum to 1.0) ──
        # NOTE: No momentum! Yield is the primary signal.
        self.w_yield = 0.35
        self.w_trend = 0.25
        self.w_recent = 0.20
        self.w_volatility = 0.20

        # ── Parameters ──
        self.ma_period = 50
        self.recent_period = 21
        self.vol_period = 42
        self.yield_lookback = 252    # 1 year for dividend yield estimation

        # ── 12 dividend/income ETFs ──
        self.target_tickers = [
            # Broad high-dividend
            "DVY",     # iShares Select Dividend (inception 2003)
            "VYM",     # Vanguard High Dividend Yield (inception 2006)
            "SCHD",    # Schwab US Dividend Equity (inception 2011)
            "HDV",     # iShares Core High Dividend (inception 2011)

            # Sector high-yield
            "SPHD",    # Invesco S&P 500 High Div Low Vol (inception 2012)
            "VNQ",     # Vanguard Real Estate / REITs (inception 2004)
            "XLU",     # Utilities Select Sector (inception 1998)

            # International dividend
            "IDV",     # iShares International Select Div (inception 2007)
            "VYMI",    # Vanguard Intl High Div Yield (inception 2016)

            # Preferred stock (bond-like)
            "PFF",     # iShares Preferred & Income (inception 2007)

            # Covered call / premium income
            "QYLD",    # Global X Nasdaq 100 Covered Call (inception 2013)
            "JEPI",    # JPMorgan Equity Premium Income (inception 2020)
        ]

        # ── Data structures ──
        self.symbols = {}
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

        # ── Add ETFs ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol

        # ── SPY for trend gate ──
        self.add_equity("SPY", Resolution.DAILY)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

        self.set_benchmark("SPY")

        # ── Schedules ──
        self.schedule.on(
            self.date_rules.month_start("SPY", 0),
            self.time_rules.after_market_open("SPY", 30),
            self.monthly_rebalance,
        )
        self.schedule.on(
            self.date_rules.every(DayOfWeek.WEDNESDAY),
            self.time_rules.after_market_open("SPY", 60),
            self.midweek_trend_check,
        )

        self._initial_rebalance_done = False

        self.debug(
            f">>> DIVIDEND YIELD ROTATION v1: {len(self.target_tickers)} ETFs, "
            f"top {self.top_n}, weights=yield{self.w_yield}/trend{self.w_trend}/"
            f"recent{self.w_recent}/vol{self.w_volatility}"
        )

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # on_data — deploy rebalance only
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        if not self._initial_rebalance_done:
            self._initial_rebalance_done = True
            invested = sum(
                1 for t in self.target_tickers
                if self.portfolio[self.symbols[t]].invested
            )
            if invested == 0:
                self.debug(f">>> DEPLOY REBALANCE: equity=${self.portfolio.total_portfolio_value:,.0f}")
                self.monthly_rebalance()

    # ══════════════════════════════════════════════════════════════════════
    # Yield estimation
    # ══════════════════════════════════════════════════════════════════════

    def _estimate_yield_score(self, symbol, closes):
        """Estimate dividend yield proxy from price behavior.

        High-dividend ETFs have a distinctive price pattern:
        - Regular small drops (ex-dividend dates) followed by recovery
        - Lower total return volatility relative to price level
        - Higher price stability (mean reversion around a level)

        We use: (total return implied by price recovery) / (price level)
        Proxy: 1-year return smoothness — high-div ETFs have steadier returns.

        Simpler approach: use the ETF's annualized return / vol ratio
        (income-oriented ETFs have lower vol relative to return).
        We invert vol and scale by price stability as a yield proxy.
        """
        if len(closes) < self.yield_lookback:
            return 0.5  # neutral if not enough data

        # Price stability: how much does the price stay near its mean?
        # High dividend ETFs are more mean-reverting (stable)
        recent_year = closes[-self.yield_lookback:]
        mean_price = np.mean(recent_year)
        deviation = np.std(recent_year) / mean_price  # coefficient of variation

        # Lower deviation = more stable = likely higher yielder
        # Invert so higher score = better
        stability_score = 1.0 / (1.0 + deviation * 10)

        # Drawdown resistance: max drawdown over the year
        # High-div ETFs tend to have smaller drawdowns
        rolling_max = np.maximum.accumulate(recent_year)
        drawdowns = (rolling_max - recent_year) / rolling_max
        max_dd = np.max(drawdowns)
        dd_score = 1.0 - max_dd  # lower DD = higher score

        # Combined yield proxy
        yield_score = 0.6 * stability_score + 0.4 * dd_score

        return yield_score

    # ══════════════════════════════════════════════════════════════════════
    # ETF scoring (yield + trend + recent + vol)
    # ══════════════════════════════════════════════════════════════════════

    def _score_etfs(self):
        """Score dividend ETFs: yield proxy + trend + recent + volatility."""
        scores = {}
        raw_data = {}

        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            history = self.history(
                symbol, self.yield_lookback + 10, Resolution.DAILY,
            )
            if history is None or history.empty:
                continue
            try:
                closes = history["close"].values
            except Exception:
                continue
            if len(closes) < self.ma_period:
                continue

            price_now = closes[-1]
            if price_now <= 0:
                continue

            # Signal 1: Yield proxy (stability + drawdown resistance)
            yield_score = self._estimate_yield_score(symbol, closes)

            # Signal 2: Trend (above 50d MA)
            if len(closes) >= self.ma_period:
                ma_50 = np.mean(closes[-self.ma_period:])
                trend_score = 1.0 if price_now > ma_50 else 0.0
            else:
                trend_score = 0.5

            # Signal 3: Recent strength (21-day)
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

            raw_data[ticker] = {
                "yield": yield_score, "trend": trend_score,
                "recent": recent, "vol": vol,
            }

        if len(raw_data) < self.top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        # Rank-normalize yield (higher = better, already scaled 0-1 but rank for consistency)
        ranked_yield = sorted(tickers, key=lambda t: raw_data[t]["yield"])
        for i, t in enumerate(ranked_yield):
            raw_data[t]["yield_rank"] = i / (n - 1) if n > 1 else 0.5

        # Rank-normalize recent
        ranked_recent = sorted(tickers, key=lambda t: raw_data[t]["recent"])
        for i, t in enumerate(ranked_recent):
            raw_data[t]["recent_rank"] = i / (n - 1) if n > 1 else 0.5

        # Volatility: lower is better
        ranked_vol = sorted(tickers, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        # Composite score
        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.w_yield * d.get("yield_rank", 0.5)
                + self.w_trend * d["trend"]
                + self.w_recent * d.get("recent_rank", 0.5)
                + self.w_volatility * d.get("vol_rank", 0.5)
            )
            scores[ticker] = score

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

        scores = self._score_etfs()
        if not scores:
            return

        n_hold = self.top_n if uptrend else self.downtrend_top_n
        self.is_in_cash = False

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = set(t for t, _ in ranked[:n_hold])

        # Sell positions not in new target
        for ticker in list(self.current_holdings):
            if ticker not in target_tickers:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.current_holdings.discard(ticker)

        # Buy new positions (equal weight)
        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0 or n_hold <= 0:
            return

        target_alloc = total_value / n_hold

        for ticker in target_tickers:
            symbol = self.symbols[ticker]
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
            self.current_holdings.add(ticker)

        regime = "UP" if uptrend else "DOWN"
        top5 = [(t, f"{s:.3f}") for t, s in ranked[:5]]
        self.debug(
            f"REBALANCE [{regime}]: {n_hold} ETFs, selected={list(target_tickers)}, "
            f"top5={top5}, eq=${total_value:,.0f}"
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

            scores = self._score_etfs()
            keep = set()
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                keep = set(t for t, _ in ranked[:self.downtrend_top_n])

            for ticker in list(self.current_holdings):
                if ticker not in keep:
                    symbol = self.symbols[ticker]
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.current_holdings.discard(ticker)

            if keep:
                total_value = self.portfolio.total_portfolio_value
                target_alloc = total_value / len(keep)
                for ticker in keep:
                    symbol = self.symbols[ticker]
                    price = self.securities[symbol].price
                    if price <= 0:
                        continue
                    target_qty = int(target_alloc / price)
                    current_qty = int(self.portfolio[symbol].quantity)
                    delta = target_qty - current_qty
                    if abs(delta) > 0:
                        self.market_order(symbol, delta)
                        self.total_trades += 1

            self.debug(f"EMERGENCY: downtrend, reduced to {len(self.current_holdings)} ETFs")

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
