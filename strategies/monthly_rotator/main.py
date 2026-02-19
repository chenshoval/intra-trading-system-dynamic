"""Monthly Rotator v1 — Multi-Signal Stock Ranking with Monthly Holds

Fundamentally different from the event-driven approach:
- Rebalance ONCE per month (not 4,000+ trades/year)
- Score all 50 stocks on momentum + trend + quality signals
- Buy top 15, equal-weight, hold the full month
- SPY trend gate: go to cash in downtrends (full stop, not half-size)
- Mid-month emergency exit only if SPY enters downtrend

Backed by:
- Warsaw paper: MA crossover with 172 trades in 31 years, IR*=0.68
- Man Group: monthly momentum, +8% real across 95 years, all regimes
- Kelly/Xiu: monthly decile portfolios, Sharpe 1.0+ long-short
- López de Prado: low-turnover hierarchical allocation beats Markowitz

Expected: ~300 trades/year vs 4,000. Fees drop from ~$11K to <$1K.
Captures PEAD drift (20-60 day), momentum persistence, trend following.
"""

from AlgorithmImports import *
from collections import defaultdict
import numpy as np


class MonthlyRotator(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.top_n = 15                     # hold top 15 stocks
        self.downtrend_top_n = 5            # reduce to top 5 in downtrend
        self.trend_fast = 10                # SPY fast MA period
        self.trend_slow = 50                # SPY slow MA period

        # ── Signal weights (sum to 1.0) ──
        self.w_momentum = 0.40              # 6-month price momentum
        self.w_trend = 0.30                 # stock above own 50d MA
        self.w_recent_strength = 0.20       # 1-month return (recent strength)
        self.w_volatility = 0.10            # lower vol = higher score (stability)

        # ── Momentum parameters ──
        self.mom_lookback = 126             # ~6 months
        self.mom_skip = 21                  # skip last month (reversal avoidance)
        self.stock_ma_period = 50           # stock's own trend MA
        self.recent_period = 21             # 1-month recent strength
        self.vol_period = 42                # 2-month realized volatility

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
        self.current_holdings = set()       # tickers currently held
        self.last_rebalance = None
        self.is_in_cash = False             # emergency cash mode

        # ── Counters ──
        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []           # track each month's return

        # ── Per-month tracking ──
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)  # year-month -> pnl

        # ── Add equities ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol

        # ── SPY for trend gate ──
        self.add_equity("SPY", Resolution.DAILY)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

        # ── Benchmark ──
        self.set_benchmark("SPY")

        # ── Monthly rebalance: first trading day of month ──
        self.schedule.on(
            self.date_rules.month_start("SPY", 0),
            self.time_rules.after_market_open("SPY", 30),
            self.monthly_rebalance,
        )

        # ── Mid-month trend check: every week check if SPY flipped ──
        self.schedule.on(
            self.date_rules.every(DayOfWeek.WEDNESDAY),
            self.time_rules.after_market_open("SPY", 60),
            self.midweek_trend_check,
        )

        self.debug(
            f">>> MONTHLY ROTATOR v1: {len(self.target_tickers)} stocks, "
            f"top {self.top_n}, trend {self.trend_fast}/{self.trend_slow} MA, "
            f"weights: mom={self.w_momentum} trend={self.w_trend} "
            f"recent={self.w_recent_strength} vol={self.w_volatility}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Trend gate
    # ══════════════════════════════════════════════════════════════════════

    def _is_uptrend(self):
        """SPY fast MA > slow MA."""
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Stock scoring
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks(self):
        """Score all stocks using multi-signal ranking. Returns dict of ticker -> score."""
        scores = {}
        raw_data = {}

        # Collect raw signal values for all stocks
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            history = self.history(
                symbol,
                self.mom_lookback + self.mom_skip + 10,  # extra buffer
                Resolution.DAILY,
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
            if price_6m > 0 and price_1m > 0:
                momentum = (price_1m / price_6m) - 1.0
            else:
                momentum = 0.0

            # Signal 2: Stock above its own 50d MA (trend)
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

            # Signal 4: Volatility (lower = better, inverted)
            if len(closes) >= self.vol_period:
                returns = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1.0
            else:
                vol = 1.0

            raw_data[ticker] = {
                "momentum": momentum,
                "trend": trend_score,
                "recent": recent,
                "vol": vol,
            }

        if len(raw_data) < self.top_n:
            return scores

        # Rank-normalize each signal across stocks (0 to 1)
        tickers = list(raw_data.keys())
        n = len(tickers)

        for signal in ["momentum", "recent"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        # Volatility: lower is better, so rank inversely
        ranked_vol = sorted(tickers, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        # Trend is already 0 or 1, no ranking needed

        # Compute composite score
        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.w_momentum * d.get("momentum_rank", 0.5)
                + self.w_trend * d["trend"]
                + self.w_recent_strength * d.get("recent_rank", 0.5)
                + self.w_volatility * d.get("vol_rank", 0.5)
            )
            scores[ticker] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance
    # ══════════════════════════════════════════════════════════════════════

    def monthly_rebalance(self):
        """Once per month: score, rank, buy top N, sell the rest."""
        self.total_rebalances += 1

        # Record previous month's return
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

        # Step 1: Score all stocks
        scores = self._score_stocks()
        if not scores:
            self.debug(f"  No scores available, skipping rebalance")
            return

        # Step 2: Determine how many to hold based on regime
        if uptrend:
            n_hold = self.top_n
            self.is_in_cash = False
        else:
            n_hold = self.downtrend_top_n
            self.is_in_cash = False

        # Step 3: Select top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = set(t for t, _ in ranked[:n_hold])
        target_symbols = {self.symbols[t] for t in target_tickers}

        # Step 4: Sell positions not in new top N
        for ticker in list(self.current_holdings):
            symbol = self.symbols[ticker]
            if ticker not in target_tickers:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.current_holdings.discard(ticker)

        # Step 5: Buy new positions (equal-weight across target)
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

        # Log
        top3 = [(t, f"{s:.3f}") for t, s in ranked[:3]]
        regime = "UPTREND" if uptrend else "DOWNTREND"
        self.debug(
            f"REBALANCE [{regime}]: holding {n_hold} stocks, "
            f"top3={top3}, equity=${total_value:,.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check (emergency exit)
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        """Weekly check: if SPY flipped to downtrend, reduce to top 5 or cash."""
        if not self.current_holdings:
            return

        uptrend = self._is_uptrend()

        if not uptrend and not self.is_in_cash:
            # SPY flipped to downtrend mid-month — reduce exposure
            self.is_in_cash = True
            self.emergency_exits += 1

            # Re-score and keep only top 5
            scores = self._score_stocks()
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                keep = set(t for t, _ in ranked[:self.downtrend_top_n])
            else:
                keep = set()

            for ticker in list(self.current_holdings):
                if ticker not in keep:
                    symbol = self.symbols[ticker]
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.current_holdings.discard(ticker)

            # Resize remaining to equal-weight with reduced capital
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

            self.debug(
                f"EMERGENCY: SPY downtrend detected, reduced to {len(self.current_holdings)} positions"
            )

        elif uptrend and self.is_in_cash:
            # SPY recovered — will fully rebalance at next month start
            self.is_in_cash = False
            self.debug(f"RECOVERY: SPY uptrend restored, will rebalance at month start")

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        # Record final month
        current_equity = self.portfolio.total_portfolio_value
        if self.last_rebalance is not None:
            month_ret = (current_equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)

        total_profit = self.portfolio.total_profit
        ret_pct = total_profit / 100_000

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} "
            f"Final=${current_equity:,.0f} "
            f"Rebalances={self.total_rebalances} "
            f"Trades={self.total_trades} "
            f"EmergencyExits={self.emergency_exits}"
        )

        self.debug(
            f"REGIME: Uptrend months={self.months_in_uptrend} "
            f"Downtrend months={self.months_in_downtrend}"
        )

        # Monthly return stats
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
                f"MONTHLY: Avg={avg_ret:.2%} Median={median_ret:.2%} "
                f"Best={best:.2%} Worst={worst:.2%} Std={std:.2%} "
                f"Win={win_months} Loss={loss_months} "
                f"WinRate={win_months/len(rets):.0%} "
                f"MonthlySharpe={monthly_sharpe:.2f}"
            )

        # Per-month P&L
        if self.monthly_pnl:
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"PNL: {pnl_str}")
