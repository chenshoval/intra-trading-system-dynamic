"""Bond Momentum v1 — Strategy 4 (Crisis Hedge)

The missing piece: a strategy that goes UP when equities crash.
In March 2020 (COVID), TLT rose +4% while SPY dropped -12%.
In Sept-Oct 2022, commodity saved us but dividend and v2 both dropped.
Bonds provide the negative/zero correlation to equities during flight-to-safety.

BUT: 2022 was the exception — rate hikes crushed both stocks AND bonds.
So we don't just hold TLT blindly. We apply momentum/trend scoring
to a diversified bond universe, rotating into whatever bond type is working.

Bond Universe (12 instruments spanning duration, credit, inflation):
- Long duration: TLT (20+ yr treasury), ZROZ (25+ yr zero coupon)
- Medium duration: IEF (7-10 yr treasury), TLH (10-20 yr treasury)
- Short duration: SHY (1-3 yr treasury), SHV (short treasury)
- Inflation-protected: TIP (TIPS), SCHP (short TIPS)
- Corporate: LQD (investment grade), HYG (high yield)
- Broad: AGG (US aggregate bond), BND (total bond market)

Why this should be uncorrelated to strategies 1-3:
- v2 (equity momentum): bonds have near-zero or NEGATIVE correlation to equities
- Commodity: bonds and commodities are different macro drivers
  (bonds = rate expectations, commodities = supply/demand)
- Dividend: dividend ETFs are equities, bonds are fixed income — different asset class

Scoring engine (4 signals):
- Momentum (0.40): 6-month return, skip last month. Bond momentum is well-documented.
- Trend (0.25): above 50d MA. Don't buy bonds in a rate hike (2022).
- Recent strength (0.20): 21-day return. Captures flight-to-safety moves.
- Volatility (0.15): lower vol = higher score. Prefer stable bonds.

Key design choice: SHY/SHV (short-term treasuries) act as "cash equivalents."
When all bonds are falling (2022), the scoring engine should naturally rotate
into SHY/SHV because they have lowest vol and smallest drawdowns. This provides
automatic de-risking without a separate regime switch.

Test periods:
- Run 1: 2016-2020 (standard gate — includes 2020 COVID flight-to-safety)
- Run 2: 2020-2023 (includes 2022 rate hike — the worst bond year in history)
- Run 3: 2023-2025 (post rate-hike normalization)
Results: results_from_quant_connect/bondmomentumv1/{period}/
"""

from AlgorithmImports import *
from collections import defaultdict
import numpy as np


class BondMomentumV1(QCAlgorithm):

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
        self.w_momentum = 0.40
        self.w_trend = 0.25
        self.w_recent = 0.20
        self.w_volatility = 0.15

        # ── Momentum parameters ──
        self.mom_lookback = 126          # 6-month momentum
        self.mom_skip = 21               # skip last month
        self.ma_period = 50              # 50-day MA for trend
        self.recent_period = 21          # 21-day recent strength
        self.vol_period = 42             # 42-day volatility

        # ── 12 bond ETFs ──
        self.target_tickers = [
            # Long duration (most sensitive to rate changes, biggest flight-to-safety)
            "TLT",     # 20+ Year Treasury Bond (inception 2002)
            "ZROZ",    # 25+ Year Zero Coupon Treasury (inception 2009)

            # Medium duration
            "IEF",     # 7-10 Year Treasury Bond (inception 2002)
            "TLH",     # 10-20 Year Treasury Bond (inception 2007)

            # Short duration (cash-like, safe haven in rate hikes)
            "SHY",     # 1-3 Year Treasury Bond (inception 2002)
            "SHV",     # Short Treasury Bond (inception 2007)

            # Inflation-protected
            "TIP",     # Treasury Inflation-Protected (inception 2003)
            "SCHP",    # Schwab US TIPS (inception 2010)

            # Corporate bonds
            "LQD",     # Investment Grade Corporate (inception 2002)
            "HYG",     # High Yield Corporate (inception 2007)

            # Broad bond market
            "AGG",     # US Aggregate Bond (inception 2003)
            "BND",     # Vanguard Total Bond Market (inception 2007)
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
            f">>> BOND MOMENTUM v1: {len(self.target_tickers)} ETFs, "
            f"top {self.top_n}, weights=mom{self.w_momentum}/trend{self.w_trend}/"
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
    # ETF scoring (4 signals — same structure as commodity)
    # ══════════════════════════════════════════════════════════════════════

    def _score_etfs(self):
        """Score bond ETFs: momentum + trend + recent + volatility."""
        scores = {}
        raw_data = {}

        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            history = self.history(
                symbol, self.mom_lookback + self.mom_skip + 10, Resolution.DAILY,
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
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol,
            }

        if len(raw_data) < self.top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        # Rank-normalize continuous signals
        for signal in ["momentum", "recent"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        # Volatility: lower is better
        ranked_vol = sorted(tickers, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        # Composite score
        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.w_momentum * d.get("momentum_rank", 0.5)
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
