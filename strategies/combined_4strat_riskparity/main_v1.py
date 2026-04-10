"""4-Strategy Risk Parity v1 — The Full System

Combines 4 uncorrelated strategies with dynamic risk parity weighting.
Each strategy contributes equal risk to the portfolio, rebalanced monthly,
with covariance estimated from a 63-day rolling window of daily returns.

The 4 strategies:
1. v2 Equity Momentum — 50 stocks, 5 signals, top 15
2. Commodity Momentum — 15 commodity ETFs, 4 signals, top 4
3. Dividend Yield Rotation — 12 high-dividend ETFs, 4 signals, top 4
4. Bond Momentum — 12 bond ETFs, 4 signals, top 4

Correlation matrix (115 months of backtest data):
              v2      Commodity  Dividend   Bond
v2            1.000   +0.340     +0.451     +0.153
Commodity     0.340   1.000      +0.308     +0.164
Dividend      0.451   0.308      1.000      +0.292
Bond          0.153   0.164      0.292      1.000

Risk parity weighting:
- Daily: track each sleeve's portfolio value, compute 63-day rolling returns
- Monthly: compute covariance matrix from daily returns, solve for equal
  risk contribution weights, rebalance all sleeves
- Fallback: if covariance estimation fails (not enough data), use equal weight (25% each)

Key design: separate estimation frequency (daily) from rebalance frequency (monthly).
This gives 252 data points/year for covariance estimation while only trading 12x/year.

Crisis coverage:
- COVID crash (2020-03) -> Bond saved (+$602 while v2 lost -$17K)
- Rate hike bear (2022) -> Commodity saved (+$34K while v2 lost -$7K)
- Sector rotation -> Dividend helps (value rotation)
- Bull markets -> v2 carries (43% CAR in 2016-2020)

Test periods:
- Run 1: 2016-2020 (standard gate)
- Run 2: 2020-2023 (includes COVID + 2022 bear)
- Run 3: 2023-2025 (recent data)
Results: results_from_quant_connect/combined4stratriskparity/{period}/
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class Combined4StratRiskParity(QCAlgorithm):

    def initialize(self):
        # ── Test periods ──
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ══════════════════════════════════════════════════════════════
        # Risk parity parameters
        # ══════════════════════════════════════════════════════════════
        self.cov_window = 63            # 63 trading days (~3 months) for covariance
        self.n_strategies = 4
        self.min_weight = 0.10          # no strategy below 10%
        self.max_weight = 0.50          # no strategy above 50%

        # ══════════════════════════════════════════════════════════════
        # Strategy parameters
        # ══════════════════════════════════════════════════════════════
        self.trend_fast = 10
        self.trend_slow = 50
        self.mom_lookback = 126
        self.mom_skip = 21
        self.ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        # Sleeve 1: v2 equity
        self.eq_top_n = 15
        self.eq_downtrend_top_n = 5
        self.eq_w_mom = 0.35
        self.eq_w_trend = 0.25
        self.eq_w_recent = 0.20
        self.eq_w_vol = 0.10
        self.eq_w_events = 0.10

        # Sleeve 2: commodity
        self.cm_top_n = 4
        self.cm_downtrend_top_n = 3
        self.cm_w_mom = 0.40
        self.cm_w_trend = 0.25
        self.cm_w_recent = 0.20
        self.cm_w_vol = 0.15

        # Sleeve 3: dividend
        self.div_top_n = 4
        self.div_downtrend_top_n = 3
        self.div_w_yield = 0.35
        self.div_w_trend = 0.25
        self.div_w_recent = 0.20
        self.div_w_vol = 0.20
        self.div_yield_lookback = 252

        # Sleeve 4: bond
        self.bond_top_n = 4
        self.bond_downtrend_top_n = 3
        self.bond_w_mom = 0.40
        self.bond_w_trend = 0.25
        self.bond_w_recent = 0.20
        self.bond_w_vol = 0.15

        # ══════════════════════════════════════════════════════════════
        # Universes
        # ══════════════════════════════════════════════════════════════
        self.equity_tickers = [
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

        self.commodity_tickers = [
            "GLD", "SLV", "PPLT", "XLE", "USO", "UNG",
            "DBA", "MOO", "WEAT", "CORN", "SOYB",
            "CPER", "DBC", "PDBC", "TLT",
        ]

        self.dividend_tickers = [
            "DVY", "VYM", "SCHD", "HDV", "SPHD", "VNQ", "XLU",
            "IDV", "VYMI", "PFF", "QYLD", "JEPI",
        ]

        self.bond_tickers = [
            "TLT", "ZROZ", "IEF", "TLH", "SHY", "SHV",
            "TIP", "SCHP", "LQD", "HYG", "AGG", "BND",
        ]

        # Event keywords (equity sleeve only)
        self.event_keywords = {
            "earnings_beat": [
                "beats estimates", "tops estimates", "beats expectations",
                "better than expected", "earnings beat", "revenue beat",
                "profit beats", "eps beats", "blowout quarter",
                "strong quarter", "record quarter", "record earnings",
                "record revenue", "exceeds expectations",
            ],
            "analyst_upgrade": [
                "upgraded to buy", "upgraded to outperform",
                "upgraded to overweight", "price target raised",
                "price target increased", "raises price target",
                "initiates with buy", "initiates with outperform",
            ],
            "guidance_raise": [
                "raises guidance", "raises outlook", "raises forecast",
                "boosts guidance", "boosts outlook", "increases guidance",
                "above prior guidance", "upside guidance",
            ],
        }

        # ══════════════════════════════════════════════════════════════
        # Data structures
        # ══════════════════════════════════════════════════════════════
        self.all_symbols = {}       # ticker -> symbol (deduped)
        self.news_symbols = {}      # ticker -> news symbol (equity only)

        self.eq_holdings = set()
        self.cm_holdings = set()
        self.div_holdings = set()
        self.bond_holdings = set()
        self.is_in_cash = False

        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        # Risk parity tracking
        self.sleeve_values = {0: [], 1: [], 2: [], 3: []}  # daily portfolio values per sleeve
        self.current_weights = [0.25, 0.25, 0.25, 0.25]     # start equal weight
        self.weight_history = []

        # General tracking
        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []
        self.last_rebalance = None
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)

        # ══════════════════════════════════════════════════════════════
        # Add all securities (deduplicated)
        # ══════════════════════════════════════════════════════════════
        all_tickers = set(self.equity_tickers + self.commodity_tickers +
                         self.dividend_tickers + self.bond_tickers)

        for ticker in all_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.all_symbols[ticker] = equity.symbol

        # News for equity sleeve
        for ticker in self.equity_tickers:
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        # SPY for trend gate
        if "SPY" not in self.all_symbols:
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
        # Daily: track sleeve values for covariance estimation
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 5),
            self.daily_track_values,
        )

        self._initial_rebalance_done = False

        self.debug(
            f">>> 4-STRAT RISK PARITY: {len(self.equity_tickers)} stocks + "
            f"{len(self.commodity_tickers)} commodity + {len(self.dividend_tickers)} dividend + "
            f"{len(self.bond_tickers)} bond ETFs, cov_window={self.cov_window}"
        )

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Daily value tracking for covariance estimation
    # ══════════════════════════════════════════════════════════════════════

    def daily_track_values(self):
        """Track each sleeve's portfolio value daily for covariance estimation."""
        for i, (holdings, tickers) in enumerate([
            (self.eq_holdings, self.equity_tickers),
            (self.cm_holdings, self.commodity_tickers),
            (self.div_holdings, self.dividend_tickers),
            (self.bond_holdings, self.bond_tickers),
        ]):
            value = 0.0
            for ticker in holdings:
                if ticker in self.all_symbols:
                    symbol = self.all_symbols[ticker]
                    value += self.portfolio[symbol].absolute_holdings_value
            self.sleeve_values[i].append(value)

            # Keep only last cov_window + 10 days
            if len(self.sleeve_values[i]) > self.cov_window + 10:
                self.sleeve_values[i] = self.sleeve_values[i][-(self.cov_window + 10):]

    # ══════════════════════════════════════════════════════════════════════
    # Risk parity weight computation
    # ══════════════════════════════════════════════════════════════════════

    def _compute_risk_parity_weights(self):
        """Compute equal risk contribution weights from rolling covariance.
        Returns weights [w1, w2, w3, w4] summing to 1.0.
        Falls back to equal weight if not enough data."""

        # Need at least cov_window days of data
        min_len = min(len(self.sleeve_values[i]) for i in range(self.n_strategies))
        if min_len < self.cov_window:
            self.debug(f"  Risk parity: only {min_len} days, need {self.cov_window}. Using equal weight.")
            return [1.0 / self.n_strategies] * self.n_strategies

        # Get last cov_window daily values for each sleeve
        returns = []
        for i in range(self.n_strategies):
            values = np.array(self.sleeve_values[i][-self.cov_window:])
            # Skip if sleeve has zero values (not yet invested)
            if np.all(values == 0) or len(values) < self.cov_window:
                self.debug(f"  Risk parity: sleeve {i} has no data. Using equal weight.")
                return [1.0 / self.n_strategies] * self.n_strategies
            # Compute daily returns (handle zeros)
            values = np.where(values == 0, 1, values)  # replace 0 with 1 to avoid div/0
            daily_ret = np.diff(values) / values[:-1]
            returns.append(daily_ret)

        returns = np.array(returns)  # shape: (4, cov_window-1)

        # Check for zero variance
        for i in range(self.n_strategies):
            if np.std(returns[i]) < 1e-10:
                self.debug(f"  Risk parity: sleeve {i} has zero variance. Using equal weight.")
                return [1.0 / self.n_strategies] * self.n_strategies

        # Compute covariance matrix
        cov_matrix = np.cov(returns)

        # Solve for risk parity weights using iterative method
        # Start with inverse-volatility weights as initial guess
        vols = np.sqrt(np.diag(cov_matrix))
        weights = (1.0 / vols)
        weights = weights / np.sum(weights)

        # Iterative risk parity: adjust weights so each strategy contributes
        # equal marginal risk to portfolio variance
        for iteration in range(50):
            # Portfolio variance
            port_var = weights @ cov_matrix @ weights
            if port_var <= 0:
                break

            # Marginal risk contribution of each strategy
            marginal_risk = cov_matrix @ weights
            risk_contribution = weights * marginal_risk
            total_risk = np.sum(risk_contribution)

            if total_risk <= 0:
                break

            # Target: each strategy contributes 1/N of total risk
            target_risk = total_risk / self.n_strategies

            # Adjust weights proportionally
            adjustment = target_risk / (risk_contribution + 1e-10)
            weights = weights * adjustment
            weights = weights / np.sum(weights)

        # Apply floor/ceiling constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / np.sum(weights)

        return weights.tolist()

    # ══════════════════════════════════════════════════════════════════════
    # Event collection (equity sleeve only)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        if not self._initial_rebalance_done:
            self._initial_rebalance_done = True
            total_invested = sum(1 for t in self.all_symbols if self.portfolio[self.all_symbols[t]].invested)
            if total_invested == 0:
                self.debug(f">>> DEPLOY REBALANCE: equity=${self.portfolio.total_portfolio_value:,.0f}")
                self.monthly_rebalance()
            return

        for ticker in self.equity_tickers:
            news_symbol = self.news_symbols[ticker]
            if not data.contains_key(news_symbol):
                continue
            article = data[news_symbol]
            title = str(getattr(article, "title", "")).lower()
            desc = str(getattr(article, "description", "")).lower()
            text = f"{title} {desc}"
            for event_type, keywords in self.event_keywords.items():
                if any(kw in text for kw in keywords):
                    self.event_counts_this_month[ticker] += 1
                    self.total_events_detected += 1
                    break

    # ══════════════════════════════════════════════════════════════════════
    # Scoring engines (one per sleeve)
    # ══════════════════════════════════════════════════════════════════════

    def _score_momentum(self, tickers, w_mom, w_trend, w_recent, w_vol,
                        top_n, include_events=False):
        """Generic momentum scoring for any universe."""
        scores = {}
        raw_data = {}

        for ticker in tickers:
            if ticker not in self.all_symbols:
                continue
            symbol = self.all_symbols[ticker]
            history = self.history(symbol, self.mom_lookback + self.mom_skip + 10, Resolution.DAILY)
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

            if len(closes) >= self.ma_period:
                ma_50 = np.mean(closes[-self.ma_period:])
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

            entry = {"momentum": momentum, "trend": trend_score, "recent": recent, "vol": vol}

            if include_events:
                entry["events"] = self.event_counts_this_month.get(ticker, 0)

            raw_data[ticker] = entry

        if len(raw_data) < top_n:
            return scores

        ticker_list = list(raw_data.keys())
        n = len(ticker_list)

        for signal in ["momentum", "recent"]:
            ranked = sorted(ticker_list, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_vol = sorted(ticker_list, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        if include_events:
            ranked_ev = sorted(ticker_list, key=lambda t: raw_data[t]["events"])
            for i, t in enumerate(ranked_ev):
                raw_data[t]["events_rank"] = i / (n - 1) if n > 1 else 0.5

        for ticker in ticker_list:
            d = raw_data[ticker]
            score = (
                w_mom * d.get("momentum_rank", 0.5)
                + w_trend * d["trend"]
                + w_recent * d.get("recent_rank", 0.5)
                + w_vol * d.get("vol_rank", 0.5)
            )
            if include_events:
                score += self.eq_w_events * d.get("events_rank", 0.5)
            scores[ticker] = score

        return scores

    def _score_dividend(self):
        """Score dividend ETFs with yield proxy."""
        scores = {}
        raw_data = {}

        for ticker in self.dividend_tickers:
            if ticker not in self.all_symbols:
                continue
            symbol = self.all_symbols[ticker]
            history = self.history(symbol, self.div_yield_lookback + 10, Resolution.DAILY)
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

            # Yield proxy
            if len(closes) >= self.div_yield_lookback:
                recent_year = closes[-self.div_yield_lookback:]
                mean_price = np.mean(recent_year)
                deviation = np.std(recent_year) / mean_price
                stability_score = 1.0 / (1.0 + deviation * 10)
                rolling_max = np.maximum.accumulate(recent_year)
                drawdowns = (rolling_max - recent_year) / rolling_max
                max_dd = np.max(drawdowns)
                dd_score = 1.0 - max_dd
                yield_score = 0.6 * stability_score + 0.4 * dd_score
            else:
                yield_score = 0.5

            if len(closes) >= self.ma_period:
                ma_50 = np.mean(closes[-self.ma_period:])
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

            raw_data[ticker] = {"yield": yield_score, "trend": trend_score, "recent": recent, "vol": vol}

        if len(raw_data) < self.div_top_n:
            return scores

        ticker_list = list(raw_data.keys())
        n = len(ticker_list)

        ranked_yield = sorted(ticker_list, key=lambda t: raw_data[t]["yield"])
        for i, t in enumerate(ranked_yield):
            raw_data[t]["yield_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_recent = sorted(ticker_list, key=lambda t: raw_data[t]["recent"])
        for i, t in enumerate(ranked_recent):
            raw_data[t]["recent_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_vol = sorted(ticker_list, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        for ticker in ticker_list:
            d = raw_data[ticker]
            score = (
                self.div_w_yield * d.get("yield_rank", 0.5)
                + self.div_w_trend * d["trend"]
                + self.div_w_recent * d.get("recent_rank", 0.5)
                + self.div_w_vol * d.get("vol_rank", 0.5)
            )
            scores[ticker] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance — all 4 sleeves with risk parity weights
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
        self.is_in_cash = False

        # Compute risk parity weights
        weights = self._compute_risk_parity_weights()
        self.current_weights = weights
        self.weight_history.append((str(self.time)[:10], weights[:]))

        total_value = self.portfolio.total_portfolio_value

        # Capital per sleeve
        eq_capital = total_value * weights[0]
        cm_capital = total_value * weights[1]
        div_capital = total_value * weights[2]
        bond_capital = total_value * weights[3]

        # Score all 4 sleeves
        eq_scores = self._score_momentum(
            self.equity_tickers, self.eq_w_mom, self.eq_w_trend, self.eq_w_recent,
            self.eq_w_vol, self.eq_top_n, include_events=True
        )
        cm_scores = self._score_momentum(
            self.commodity_tickers, self.cm_w_mom, self.cm_w_trend, self.cm_w_recent,
            self.cm_w_vol, self.cm_top_n
        )
        div_scores = self._score_dividend()
        bond_scores = self._score_momentum(
            self.bond_tickers, self.bond_w_mom, self.bond_w_trend, self.bond_w_recent,
            self.bond_w_vol, self.bond_top_n
        )

        # Determine holdings per sleeve
        eq_n = self.eq_top_n if uptrend else self.eq_downtrend_top_n
        cm_n = self.cm_top_n if uptrend else self.cm_downtrend_top_n
        div_n = self.div_top_n if uptrend else self.div_downtrend_top_n
        bond_n = self.bond_top_n if uptrend else self.bond_downtrend_top_n

        eq_targets = set(t for t, _ in sorted(eq_scores.items(), key=lambda x: x[1], reverse=True)[:eq_n]) if eq_scores else set()
        cm_targets = set(t for t, _ in sorted(cm_scores.items(), key=lambda x: x[1], reverse=True)[:cm_n]) if cm_scores else set()
        div_targets = set(t for t, _ in sorted(div_scores.items(), key=lambda x: x[1], reverse=True)[:div_n]) if div_scores else set()
        bond_targets = set(t for t, _ in sorted(bond_scores.items(), key=lambda x: x[1], reverse=True)[:bond_n]) if bond_scores else set()

        # Build combined target: ticker -> total capital allocation
        target_allocs = defaultdict(float)
        for ticker in eq_targets:
            target_allocs[ticker] += eq_capital / eq_n if eq_n > 0 else 0
        for ticker in cm_targets:
            target_allocs[ticker] += cm_capital / cm_n if cm_n > 0 else 0
        for ticker in div_targets:
            target_allocs[ticker] += div_capital / div_n if div_n > 0 else 0
        for ticker in bond_targets:
            target_allocs[ticker] += bond_capital / bond_n if bond_n > 0 else 0

        # Liquidate anything not in any target set
        all_targets = eq_targets | cm_targets | div_targets | bond_targets
        all_current = self.eq_holdings | self.cm_holdings | self.div_holdings | self.bond_holdings

        for ticker in list(all_current):
            if ticker not in all_targets:
                if ticker in self.all_symbols:
                    symbol = self.all_symbols[ticker]
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1

        # Buy/resize to target allocations
        for ticker, alloc in target_allocs.items():
            if ticker not in self.all_symbols:
                continue
            symbol = self.all_symbols[ticker]
            price = self.securities[symbol].price
            if price <= 0:
                continue
            target_qty = int(alloc / price)
            if target_qty < 1:
                continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1

        # Update holdings sets
        self.eq_holdings = eq_targets
        self.cm_holdings = cm_targets
        self.div_holdings = div_targets
        self.bond_holdings = bond_targets

        # Logging
        events = sum(self.event_counts_this_month.values())
        regime = "UP" if uptrend else "DOWN"
        self.debug(
            f"REBALANCE [{regime}]: weights=[{weights[0]:.0%},{weights[1]:.0%},{weights[2]:.0%},{weights[3]:.0%}] "
            f"EQ={eq_n}(${eq_capital:,.0f}) CM={cm_n}(${cm_capital:,.0f}) "
            f"DIV={div_n}(${div_capital:,.0f}) BOND={bond_n}(${bond_capital:,.0f}) "
            f"events={events} total=${total_value:,.0f}"
        )

        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        all_holdings = self.eq_holdings | self.cm_holdings | self.div_holdings | self.bond_holdings
        if not all_holdings:
            return

        uptrend = self._is_uptrend()

        if not uptrend and not self.is_in_cash:
            self.is_in_cash = True
            self.emergency_exits += 1

            # Reduce each sleeve to downtrend count
            total_value = self.portfolio.total_portfolio_value
            weights = self.current_weights

            # Re-score and reduce
            for i, (tickers, holdings_set, top_n_down, capital) in enumerate([
                (self.equity_tickers, self.eq_holdings, self.eq_downtrend_top_n, total_value * weights[0]),
                (self.commodity_tickers, self.cm_holdings, self.cm_downtrend_top_n, total_value * weights[1]),
                (self.dividend_tickers, self.div_holdings, self.div_downtrend_top_n, total_value * weights[2]),
                (self.bond_tickers, self.bond_holdings, self.bond_downtrend_top_n, total_value * weights[3]),
            ]):
                if i == 2:  # dividend uses different scorer
                    scores = self._score_dividend()
                elif i == 0:  # equity includes events
                    scores = self._score_momentum(tickers, self.eq_w_mom, self.eq_w_trend,
                                                   self.eq_w_recent, self.eq_w_vol, top_n_down, True)
                else:
                    w_mom = self.cm_w_mom if i == 1 else self.bond_w_mom
                    w_trend = self.cm_w_trend if i == 1 else self.bond_w_trend
                    w_recent = self.cm_w_recent if i == 1 else self.bond_w_recent
                    w_vol = self.cm_w_vol if i == 1 else self.bond_w_vol
                    scores = self._score_momentum(tickers, w_mom, w_trend, w_recent, w_vol, top_n_down)

                keep = set()
                if scores:
                    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    keep = set(t for t, _ in ranked[:top_n_down])

                # Liquidate non-keepers in this sleeve
                for ticker in list(holdings_set):
                    if ticker not in keep and ticker in self.all_symbols:
                        # Only liquidate if not held by another sleeve
                        other_holdings = set()
                        for j, hs in enumerate([self.eq_holdings, self.cm_holdings, self.div_holdings, self.bond_holdings]):
                            if j != i:
                                other_holdings |= hs
                        if ticker not in other_holdings:
                            symbol = self.all_symbols[ticker]
                            if self.portfolio[symbol].invested:
                                self.liquidate(symbol)
                                self.total_trades += 1

                # Update holdings set
                if i == 0: self.eq_holdings = keep
                elif i == 1: self.cm_holdings = keep
                elif i == 2: self.div_holdings = keep
                elif i == 3: self.bond_holdings = keep

            self.debug(f"EMERGENCY: downtrend, reduced all sleeves")

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
            f"EmergencyExits={self.emergency_exits} Events={self.total_events_detected}"
        )
        self.debug(
            f"REGIME: Up={self.months_in_uptrend} Down={self.months_in_downtrend}"
        )
        self.debug(
            f"FINAL WEIGHTS: EQ={self.current_weights[0]:.0%} CM={self.current_weights[1]:.0%} "
            f"DIV={self.current_weights[2]:.0%} BOND={self.current_weights[3]:.0%}"
        )

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

        # Log weight evolution
        if self.weight_history:
            self.debug(f"WEIGHT HISTORY ({len(self.weight_history)} rebalances):")
            for date, w in self.weight_history[-6:]:  # last 6
                self.debug(f"  {date}: EQ={w[0]:.0%} CM={w[1]:.0%} DIV={w[2]:.0%} BOND={w[3]:.0%}")

        if self.monthly_pnl:
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"PNL: {pnl_str}")
