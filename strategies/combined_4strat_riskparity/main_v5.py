"""4-Strategy v5 — Relative Ranking + Asymmetric Smoothing

The AI tree (v4) taught us WHAT matters but used static thresholds that go stale.
v5 uses the insight (which features matter) with relative logic that never expires.

Core idea: rank sleeves by recent risk-adjusted performance, then overweight
the winners. No fixed dollar thresholds — everything is relative ranking.

Asymmetric smoothing (the key innovation):
- Slope UP fast:   max +20% per month toward a winner
- Slope DOWN slow: max -8% per month away from a loser
This catches momentum quickly but gives struggling sleeves time to recover.

Example:
  Month 1: [25%, 25%, 25%, 25%]  (start equal)
  Month 2: equity scores best -> [45%, 20%, 20%, 15%]  (fast +20% to equity)
  Month 3: equity has bad month -> [37%, 23%, 23%, 17%]  (slow -8% down)
  Month 4: equity recovers    -> [57%, 18%, 15%, 10%]  (fast +20% again)

How the ranking works (learned from v4's feature importances):
1. For each sleeve, compute a "score" from last 42 days of daily returns:
   score = risk_adjusted_return * 0.6 + trend_strength * 0.4
   where:
   - risk_adjusted_return = mean(daily_returns) / std(daily_returns)  (mini-Sharpe)
   - trend_strength = fraction of positive days
2. Rank the 4 sleeves by score
3. Target weights: rank1=45%, rank2=25%, rank3=18%, rank4=12%
4. Apply asymmetric smoothing toward target

Why this works for live trading:
- No static thresholds — adapts to any market regime
- No model to retrain — the ranking is always relative
- Asymmetric smoothing prevents whipsaw while catching trends
- Caps: min 5%, max 70%

All v2 fixes retained: per-sleeve trend gates, TLT removed from commodity.

Test periods:
- Run 1: 2016-2020
- Run 2: 2020-2023
- Run 3: 2023-2025
Results: results_from_quant_connect/combined4stratriskparityv5/{period}/
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class Combined4StratRiskParityV5(QCAlgorithm):

    def initialize(self):
        # ── Test periods ──
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ══════════════════════════════════════════════════════════════
        # v5 parameters
        # ══════════════════════════════════════════════════════════════
        self.cov_window = 42
        self.min_cov_days = 30
        self.n_strategies = 4
        self.min_weight = 0.05
        self.max_weight = 0.70           # wider than v4's 60%
        self.slope_up = 0.20             # fast: +20% per month max
        self.slope_down = 0.08           # slow: -8% per month max

        # Target weights by rank (rank 1 = best performer)
        self.rank_weights = [0.45, 0.25, 0.18, 0.12]

        # ══════════════════════════════════════════════════════════════
        # Strategy parameters (same as v2-v4)
        # ══════════════════════════════════════════════════════════════
        self.trend_fast = 10
        self.trend_slow = 50
        self.mom_lookback = 126
        self.mom_skip = 21
        self.ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        self.eq_top_n = 15
        self.eq_downtrend_top_n = 5
        self.eq_w_mom = 0.35
        self.eq_w_trend = 0.25
        self.eq_w_recent = 0.20
        self.eq_w_vol = 0.10
        self.eq_w_events = 0.10

        self.cm_top_n = 4
        self.cm_downtrend_top_n = 3
        self.cm_w_mom = 0.40
        self.cm_w_trend = 0.25
        self.cm_w_recent = 0.20
        self.cm_w_vol = 0.15

        self.div_top_n = 4
        self.div_downtrend_top_n = 3
        self.div_w_yield = 0.35
        self.div_w_trend = 0.25
        self.div_w_recent = 0.20
        self.div_w_vol = 0.20
        self.div_yield_lookback = 252

        self.bond_top_n = 4
        self.bond_downtrend_top_n = 3
        self.bond_w_mom = 0.40
        self.bond_w_trend = 0.25
        self.bond_w_recent = 0.20
        self.bond_w_vol = 0.15

        # ══════════════════════════════════════════════════════════════
        # Universes (TLT removed from commodity)
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
            "CPER", "DBC", "PDBC",
        ]

        self.dividend_tickers = [
            "DVY", "VYM", "SCHD", "HDV", "SPHD", "VNQ", "XLU",
            "IDV", "VYMI", "PFF", "QYLD", "JEPI",
        ]

        self.bond_tickers = [
            "TLT", "ZROZ", "IEF", "TLH", "SHY", "SHV",
            "TIP", "SCHP", "LQD", "HYG", "AGG", "BND",
        ]

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
        self.all_symbols = {}
        self.news_symbols = {}

        self.eq_holdings = set()
        self.cm_holdings = set()
        self.div_holdings = set()
        self.bond_holdings = set()

        self.eq_in_downtrend = False
        self.cm_in_downtrend = False
        self.div_in_downtrend = False
        self.bond_in_downtrend = False

        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        self.sleeve_values = {0: [], 1: [], 2: [], 3: []}
        self.current_weights = [0.25, 0.25, 0.25, 0.25]
        self.weight_history = []

        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = {0: 0, 1: 0, 2: 0, 3: 0}
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []
        self.last_rebalance = None
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)

        # ══════════════════════════════════════════════════════════════
        # Add securities
        # ══════════════════════════════════════════════════════════════
        all_tickers = set(self.equity_tickers + self.commodity_tickers +
                         self.dividend_tickers + self.bond_tickers)

        for ticker in all_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.all_symbols[ticker] = equity.symbol

        for ticker in self.equity_tickers:
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        if "SPY" not in self.all_symbols:
            self.add_equity("SPY", Resolution.DAILY)
            self.all_symbols["SPY"] = self.symbol("SPY")

        for ticker in ["DBC", "DVY", "AGG"]:
            if ticker not in self.all_symbols:
                equity = self.add_equity(ticker, Resolution.DAILY)
                self.all_symbols[ticker] = equity.symbol

        # Per-sleeve trend gates
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)
        self.dbc_fast_ma = self.sma("DBC", self.trend_fast, Resolution.DAILY)
        self.dbc_slow_ma = self.sma("DBC", self.trend_slow, Resolution.DAILY)
        self.dvy_fast_ma = self.sma("DVY", self.trend_fast, Resolution.DAILY)
        self.dvy_slow_ma = self.sma("DVY", self.trend_slow, Resolution.DAILY)
        self.agg_fast_ma = self.sma("AGG", self.trend_fast, Resolution.DAILY)
        self.agg_slow_ma = self.sma("AGG", self.trend_slow, Resolution.DAILY)

        self.set_benchmark("SPY")

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
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 5),
            self.daily_track_values,
        )

        self._initial_rebalance_done = False

        self.debug(
            f">>> 4-STRAT v5 ASYMMETRIC: {len(self.equity_tickers)} stocks + "
            f"{len(self.commodity_tickers)} commodity + {len(self.dividend_tickers)} dividend + "
            f"{len(self.bond_tickers)} bond ETFs, "
            f"slope_up={self.slope_up:.0%} slope_down={self.slope_down:.0%} "
            f"caps={self.min_weight:.0%}/{self.max_weight:.0%}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Per-sleeve trend checks
    # ══════════════════════════════════════════════════════════════════════

    def _is_sleeve_uptrend(self, sleeve_idx):
        if sleeve_idx == 0:
            if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready: return True
            return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value
        elif sleeve_idx == 1:
            if not self.dbc_fast_ma.is_ready or not self.dbc_slow_ma.is_ready: return True
            return self.dbc_fast_ma.current.value > self.dbc_slow_ma.current.value
        elif sleeve_idx == 2:
            if not self.dvy_fast_ma.is_ready or not self.dvy_slow_ma.is_ready: return True
            return self.dvy_fast_ma.current.value > self.dvy_slow_ma.current.value
        elif sleeve_idx == 3:
            if not self.agg_fast_ma.is_ready or not self.agg_slow_ma.is_ready: return True
            return self.agg_fast_ma.current.value > self.agg_slow_ma.current.value
        return True

    # ══════════════════════════════════════════════════════════════════════
    # Daily value tracking
    # ══════════════════════════════════════════════════════════════════════

    def daily_track_values(self):
        for i, holdings in enumerate([self.eq_holdings, self.cm_holdings,
                                       self.div_holdings, self.bond_holdings]):
            value = 0.0
            for ticker in holdings:
                if ticker in self.all_symbols:
                    symbol = self.all_symbols[ticker]
                    value += self.portfolio[symbol].absolute_holdings_value
            self.sleeve_values[i].append(value)
            if len(self.sleeve_values[i]) > self.cov_window + 10:
                self.sleeve_values[i] = self.sleeve_values[i][-(self.cov_window + 10):]

    # ══════════════════════════════════════════════════════════════════════
    # v5: Relative ranking weight computation
    # ══════════════════════════════════════════════════════════════════════

    def _compute_risk_parity_weights(self):
        """Rank sleeves by recent risk-adjusted performance.
        No static thresholds — everything is relative.
        Returns target weights before asymmetric smoothing."""

        min_len = min(len(self.sleeve_values[i]) for i in range(self.n_strategies))

        if min_len < self.min_cov_days:
            self.debug(f"  v5 ranking: only {min_len} days, need {self.min_cov_days}. Keeping previous weights.")
            return self.current_weights[:]

        use_days = min(min_len, self.cov_window)

        # Compute score for each sleeve
        scores = []
        names = ['EQ', 'CM', 'DIV', 'BOND']

        for i in range(self.n_strategies):
            values = np.array(self.sleeve_values[i][-use_days:])
            if np.all(values == 0) or len(values) < self.min_cov_days:
                scores.append(0.0)
                continue

            values = np.where(values == 0, 1, values)
            daily_ret = np.diff(values) / values[:-1]

            # Mini-Sharpe: mean / std (risk-adjusted return)
            std = np.std(daily_ret)
            if std > 1e-10:
                risk_adj = np.mean(daily_ret) / std
            else:
                risk_adj = 0.0

            # Trend strength: fraction of positive days
            trend_pct = np.sum(daily_ret > 0) / len(daily_ret) if len(daily_ret) > 0 else 0.5

            # Combined score (from v4 feature importance: returns + trend matter most)
            score = risk_adj * 0.6 + trend_pct * 0.4
            scores.append(score)

        # Rank sleeves (highest score = rank 0 = gets most weight)
        ranked_indices = sorted(range(self.n_strategies), key=lambda i: scores[i], reverse=True)

        # Assign target weights by rank
        target_weights = [0.0] * self.n_strategies
        for rank, idx in enumerate(ranked_indices):
            target_weights[idx] = self.rank_weights[rank]

        # Log the ranking
        ranking_str = " > ".join(f"{names[idx]}({scores[idx]:.3f})" for idx in ranked_indices)
        self.debug(f"  v5 ranking: {ranking_str}")
        self.debug(f"  v5 target: [{target_weights[0]:.0%},{target_weights[1]:.0%},{target_weights[2]:.0%},{target_weights[3]:.0%}]")

        return target_weights

    # ══════════════════════════════════════════════════════════════════════
    # Event collection
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
    # Scoring engines (same as v2-v4)
    # ══════════════════════════════════════════════════════════════════════

    def _score_momentum(self, tickers, w_mom, w_trend, w_recent, w_vol,
                        top_n, include_events=False):
        scores = {}
        raw_data = {}
        for ticker in tickers:
            if ticker not in self.all_symbols: continue
            symbol = self.all_symbols[ticker]
            history = self.history(symbol, self.mom_lookback + self.mom_skip + 10, Resolution.DAILY)
            if history is None or history.empty: continue
            try: closes = history["close"].values
            except Exception: continue
            if len(closes) < self.mom_lookback + self.mom_skip: continue
            price_now = closes[-1]
            if price_now <= 0: continue

            price_6m, price_1m = closes[0], closes[-self.mom_skip]
            momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0
            ma_50 = np.mean(closes[-self.ma_period:]) if len(closes) >= self.ma_period else price_now
            trend_score = 1.0 if price_now > ma_50 else 0.0
            recent = ((price_now / closes[-self.recent_period]) - 1.0) if len(closes) >= self.recent_period and closes[-self.recent_period] > 0 else 0.0
            if len(closes) >= self.vol_period:
                rets = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 1.0
            else: vol = 1.0

            entry = {"momentum": momentum, "trend": trend_score, "recent": recent, "vol": vol}
            if include_events: entry["events"] = self.event_counts_this_month.get(ticker, 0)
            raw_data[ticker] = entry

        if len(raw_data) < top_n: return scores
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
            score = (w_mom * d.get("momentum_rank", 0.5) + w_trend * d["trend"]
                     + w_recent * d.get("recent_rank", 0.5) + w_vol * d.get("vol_rank", 0.5))
            if include_events: score += self.eq_w_events * d.get("events_rank", 0.5)
            scores[ticker] = score
        return scores

    def _score_dividend(self):
        scores = {}
        raw_data = {}
        for ticker in self.dividend_tickers:
            if ticker not in self.all_symbols: continue
            symbol = self.all_symbols[ticker]
            history = self.history(symbol, self.div_yield_lookback + 10, Resolution.DAILY)
            if history is None or history.empty: continue
            try: closes = history["close"].values
            except Exception: continue
            if len(closes) < self.ma_period: continue
            price_now = closes[-1]
            if price_now <= 0: continue

            if len(closes) >= self.div_yield_lookback:
                recent_year = closes[-self.div_yield_lookback:]
                deviation = np.std(recent_year) / np.mean(recent_year)
                stability = 1.0 / (1.0 + deviation * 10)
                rolling_max = np.maximum.accumulate(recent_year)
                dd_score = 1.0 - np.max((rolling_max - recent_year) / rolling_max)
                yield_score = 0.6 * stability + 0.4 * dd_score
            else: yield_score = 0.5

            ma_50 = np.mean(closes[-self.ma_period:]) if len(closes) >= self.ma_period else price_now
            trend_score = 1.0 if price_now > ma_50 else 0.0
            recent = ((price_now / closes[-self.recent_period]) - 1.0) if len(closes) >= self.recent_period and closes[-self.recent_period] > 0 else 0.0
            if len(closes) >= self.vol_period:
                rets = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 1.0
            else: vol = 1.0
            raw_data[ticker] = {"yield": yield_score, "trend": trend_score, "recent": recent, "vol": vol}

        if len(raw_data) < self.div_top_n: return scores
        ticker_list = list(raw_data.keys())
        n = len(ticker_list)
        for signal, key in [("yield", "yield_rank"), ("recent", "recent_rank")]:
            ranked = sorted(ticker_list, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked): raw_data[t][key] = i / (n - 1) if n > 1 else 0.5
        ranked_vol = sorted(ticker_list, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol): raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5
        for ticker in ticker_list:
            d = raw_data[ticker]
            scores[ticker] = (self.div_w_yield * d["yield_rank"] + self.div_w_trend * d["trend"]
                             + self.div_w_recent * d["recent_rank"] + self.div_w_vol * d["vol_rank"])
        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance with asymmetric smoothing
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

        eq_up = self._is_sleeve_uptrend(0)
        cm_up = self._is_sleeve_uptrend(1)
        div_up = self._is_sleeve_uptrend(2)
        bond_up = self._is_sleeve_uptrend(3)

        self.eq_in_downtrend = not eq_up
        self.cm_in_downtrend = not cm_up
        self.div_in_downtrend = not div_up
        self.bond_in_downtrend = not bond_up

        if eq_up: self.months_in_uptrend += 1
        else: self.months_in_downtrend += 1

        # Get target weights from relative ranking
        target_weights = self._compute_risk_parity_weights()

        # ASYMMETRIC SMOOTHING: slope up fast, slope down slow
        smoothed = []
        for i in range(self.n_strategies):
            current = self.current_weights[i]
            target = target_weights[i]
            delta = target - current

            if delta > 0:
                # Going UP toward target — fast (slope_up = 20%)
                delta = min(delta, self.slope_up)
            else:
                # Going DOWN from target — slow (slope_down = 8%)
                delta = max(delta, -self.slope_down)

            smoothed.append(current + delta)

        # Normalize and apply caps
        total_w = sum(smoothed)
        if total_w > 0:
            smoothed = [w / total_w for w in smoothed]
        smoothed = [max(self.min_weight, min(self.max_weight, w)) for w in smoothed]
        total_w = sum(smoothed)
        if total_w > 0:
            smoothed = [w / total_w for w in smoothed]

        weights = smoothed
        self.current_weights = weights
        self.weight_history.append((str(self.time)[:10], weights[:], target_weights[:]))

        total_value = self.portfolio.total_portfolio_value
        eq_capital = total_value * weights[0]
        cm_capital = total_value * weights[1]
        div_capital = total_value * weights[2]
        bond_capital = total_value * weights[3]

        # Score all 4 sleeves
        eq_scores = self._score_momentum(self.equity_tickers, self.eq_w_mom, self.eq_w_trend,
                                          self.eq_w_recent, self.eq_w_vol, self.eq_top_n, True)
        cm_scores = self._score_momentum(self.commodity_tickers, self.cm_w_mom, self.cm_w_trend,
                                          self.cm_w_recent, self.cm_w_vol, self.cm_top_n)
        div_scores = self._score_dividend()
        bond_scores = self._score_momentum(self.bond_tickers, self.bond_w_mom, self.bond_w_trend,
                                            self.bond_w_recent, self.bond_w_vol, self.bond_top_n)

        eq_n = self.eq_top_n if eq_up else self.eq_downtrend_top_n
        cm_n = self.cm_top_n if cm_up else self.cm_downtrend_top_n
        div_n = self.div_top_n if div_up else self.div_downtrend_top_n
        bond_n = self.bond_top_n if bond_up else self.bond_downtrend_top_n

        eq_targets = set(t for t, _ in sorted(eq_scores.items(), key=lambda x: x[1], reverse=True)[:eq_n]) if eq_scores else set()
        cm_targets = set(t for t, _ in sorted(cm_scores.items(), key=lambda x: x[1], reverse=True)[:cm_n]) if cm_scores else set()
        div_targets = set(t for t, _ in sorted(div_scores.items(), key=lambda x: x[1], reverse=True)[:div_n]) if div_scores else set()
        bond_targets = set(t for t, _ in sorted(bond_scores.items(), key=lambda x: x[1], reverse=True)[:bond_n]) if bond_scores else set()

        target_allocs = defaultdict(float)
        for ticker in eq_targets: target_allocs[ticker] += eq_capital / eq_n if eq_n > 0 else 0
        for ticker in cm_targets: target_allocs[ticker] += cm_capital / cm_n if cm_n > 0 else 0
        for ticker in div_targets: target_allocs[ticker] += div_capital / div_n if div_n > 0 else 0
        for ticker in bond_targets: target_allocs[ticker] += bond_capital / bond_n if bond_n > 0 else 0

        all_targets = eq_targets | cm_targets | div_targets | bond_targets
        all_current = self.eq_holdings | self.cm_holdings | self.div_holdings | self.bond_holdings

        for ticker in list(all_current):
            if ticker not in all_targets and ticker in self.all_symbols:
                symbol = self.all_symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1

        for ticker, alloc in target_allocs.items():
            if ticker not in self.all_symbols: continue
            symbol = self.all_symbols[ticker]
            price = self.securities[symbol].price
            if price <= 0: continue
            target_qty = int(alloc / price)
            if target_qty < 1: continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1

        self.eq_holdings = eq_targets
        self.cm_holdings = cm_targets
        self.div_holdings = div_targets
        self.bond_holdings = bond_targets

        events = sum(self.event_counts_this_month.values())
        sleeve_status = (f"EQ={'UP' if eq_up else 'DN'}({eq_n}) CM={'UP' if cm_up else 'DN'}({cm_n}) "
                        f"DIV={'UP' if div_up else 'DN'}({div_n}) BOND={'UP' if bond_up else 'DN'}({bond_n})")
        self.debug(
            f"REBALANCE: weights=[{weights[0]:.0%},{weights[1]:.0%},{weights[2]:.0%},{weights[3]:.0%}] "
            f"target=[{target_weights[0]:.0%},{target_weights[1]:.0%},{target_weights[2]:.0%},{target_weights[3]:.0%}] "
            f"{sleeve_status} events={events} total=${total_value:,.0f}"
        )

        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check (per-sleeve, same as v2-v4)
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        all_holdings = self.eq_holdings | self.cm_holdings | self.div_holdings | self.bond_holdings
        if not all_holdings: return

        sleeve_names = ["EQ(SPY)", "CM(DBC)", "DIV(DVY)", "BOND(AGG)"]
        sleeve_holdings = [self.eq_holdings, self.cm_holdings, self.div_holdings, self.bond_holdings]
        sleeve_downtrend = [self.eq_in_downtrend, self.cm_in_downtrend,
                           self.div_in_downtrend, self.bond_in_downtrend]
        sleeve_top_n_down = [self.eq_downtrend_top_n, self.cm_downtrend_top_n,
                            self.div_downtrend_top_n, self.bond_downtrend_top_n]

        triggered = []

        for i in range(self.n_strategies):
            uptrend = self._is_sleeve_uptrend(i)

            if not uptrend and not sleeve_downtrend[i]:
                sleeve_downtrend[i] = True
                self.emergency_exits[i] += 1
                triggered.append(sleeve_names[i])

                if i == 0:
                    scores = self._score_momentum(self.equity_tickers, self.eq_w_mom, self.eq_w_trend,
                                                   self.eq_w_recent, self.eq_w_vol, sleeve_top_n_down[i], True)
                elif i == 1:
                    scores = self._score_momentum(self.commodity_tickers, self.cm_w_mom, self.cm_w_trend,
                                                   self.cm_w_recent, self.cm_w_vol, sleeve_top_n_down[i])
                elif i == 2:
                    scores = self._score_dividend()
                else:
                    scores = self._score_momentum(self.bond_tickers, self.bond_w_mom, self.bond_w_trend,
                                                   self.bond_w_recent, self.bond_w_vol, sleeve_top_n_down[i])

                keep = set()
                if scores:
                    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    keep = set(t for t, _ in ranked[:sleeve_top_n_down[i]])

                other_holdings = set()
                for j in range(self.n_strategies):
                    if j != i: other_holdings |= sleeve_holdings[j]

                for ticker in list(sleeve_holdings[i]):
                    if ticker not in keep and ticker not in other_holdings:
                        if ticker in self.all_symbols:
                            symbol = self.all_symbols[ticker]
                            if self.portfolio[symbol].invested:
                                self.liquidate(symbol)
                                self.total_trades += 1

                if i == 0: self.eq_holdings = keep
                elif i == 1: self.cm_holdings = keep
                elif i == 2: self.div_holdings = keep
                elif i == 3: self.bond_holdings = keep

            elif uptrend and sleeve_downtrend[i]:
                sleeve_downtrend[i] = False

        self.eq_in_downtrend = sleeve_downtrend[0]
        self.cm_in_downtrend = sleeve_downtrend[1]
        self.div_in_downtrend = sleeve_downtrend[2]
        self.bond_in_downtrend = sleeve_downtrend[3]

        if triggered:
            remaining = (f"EQ={len(self.eq_holdings)} CM={len(self.cm_holdings)} "
                        f"DIV={len(self.div_holdings)} BOND={len(self.bond_holdings)}")
            self.debug(f"EMERGENCY: {', '.join(triggered)} downtrend -> reduced. Remaining: {remaining}")

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        current_equity = self.portfolio.total_portfolio_value
        if self.last_rebalance is not None:
            month_ret = (current_equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)

        ret_pct = self.portfolio.total_profit / 100_000
        total_emergencies = sum(self.emergency_exits.values())

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} Final=${current_equity:,.0f} "
            f"Rebalances={self.total_rebalances} Trades={self.total_trades} "
            f"Emergencies={total_emergencies} (EQ={self.emergency_exits[0]} CM={self.emergency_exits[1]} "
            f"DIV={self.emergency_exits[2]} BOND={self.emergency_exits[3]}) "
            f"Events={self.total_events_detected}"
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

        if self.weight_history:
            self.debug(f"WEIGHT HISTORY ({len(self.weight_history)} rebalances):")
            for entry in self.weight_history:
                date, w = entry[0], entry[1]
                target = entry[2] if len(entry) > 2 else w
                self.debug(
                    f"  {date}: ACTUAL=[{w[0]:.0%},{w[1]:.0%},{w[2]:.0%},{w[3]:.0%}] "
                    f"TARGET=[{target[0]:.0%},{target[1]:.0%},{target[2]:.0%},{target[3]:.0%}]"
                )

        if self.monthly_pnl:
            pnl_str = " | ".join(f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items()))
            self.debug(f"PNL: {pnl_str}")
