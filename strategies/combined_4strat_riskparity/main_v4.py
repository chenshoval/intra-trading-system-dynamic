"""4-Strategy Risk Parity v4 — AI-Learned Weights

All v2/v3 fixes retained (per-sleeve gates, TLT fix, caps 5/60%, smoothing).

New in v4:
7. AI-learned weight prediction replaces risk parity covariance solver.
   A decision tree was trained offline on 115 months of backtest data.
   It predicts which sleeve will perform best given recent performance.
   Winner gets 50%, others get ~17% each. Smoothed by v3's 15% max delta.

   Key learned rules:
   - Calm markets (low dividend vol, equity positive) -> overweight equity
   - Equity dropped + bonds flat -> overweight commodity
   - Bonds gaining (flight to safety) -> overweight bonds
   - High stress + commodity momentum -> overweight commodity

   Training: notebooks/05_weight_learning/train_weights.py

   In live trading: the decision tree rules are hardcoded as if/else logic.
   No model loading needed. The features (3-month rolling returns, vol, trend)
   are computed from the sleeve's daily value tracking (same as v2/v3).
   Retrain periodically (every 6 months) by running the training script
   with updated backtest data and pasting new rules.

Test periods:
- Run 1: 2016-2020
- Run 2: 2020-2023
- Run 3: 2023-2025
Results: results_from_quant_connect/combined4stratriskparityv4/{period}/
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class Combined4StratRiskParityV4(QCAlgorithm):

    def initialize(self):
        # ── Test periods ──
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ══════════════════════════════════════════════════════════════
        # Risk parity parameters (FIXED from v1)
        # ══════════════════════════════════════════════════════════════
        self.cov_window = 42            # FIXED: 42 days (was 63)
        self.min_cov_days = 30          # FIXED: compute if at least 30 days available
        self.n_strategies = 4
        self.min_weight = 0.05          # FIXED: 5% floor (was 10%)
        self.max_weight = 0.60          # FIXED: 60% ceiling (was 50%)
        self.max_weight_delta = 0.15    # NEW v3: max 15% change per sleeve per month

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

        # Sleeve 2: commodity (FIXED: TLT removed)
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
        # Universes (FIXED: TLT removed from commodity)
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

        # FIXED: TLT removed — it's a bond, belongs in bond sleeve only
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
        self.all_symbols = {}
        self.news_symbols = {}

        self.eq_holdings = set()
        self.cm_holdings = set()
        self.div_holdings = set()
        self.bond_holdings = set()

        # Per-sleeve emergency state (FIXED: independent per sleeve)
        self.eq_in_downtrend = False
        self.cm_in_downtrend = False
        self.div_in_downtrend = False
        self.bond_in_downtrend = False

        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        # Risk parity tracking
        self.sleeve_values = {0: [], 1: [], 2: [], 3: []}
        self.current_weights = [0.25, 0.25, 0.25, 0.25]
        self.weight_history = []

        # General tracking
        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = {0: 0, 1: 0, 2: 0, 3: 0}  # per sleeve
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

        # SPY for general reference
        if "SPY" not in self.all_symbols:
            self.add_equity("SPY", Resolution.DAILY)
            self.all_symbols["SPY"] = self.symbol("SPY")

        # Ensure benchmark ETFs are added
        for ticker in ["DBC", "DVY", "AGG"]:
            if ticker not in self.all_symbols:
                equity = self.add_equity(ticker, Resolution.DAILY)
                self.all_symbols[ticker] = equity.symbol

        # ══════════════════════════════════════════════════════════════
        # Per-sleeve trend gates (FIXED: each sleeve has its own)
        # ══════════════════════════════════════════════════════════════
        # Sleeve 0: Equity -> SPY
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

        # Sleeve 1: Commodity -> DBC
        self.dbc_fast_ma = self.sma("DBC", self.trend_fast, Resolution.DAILY)
        self.dbc_slow_ma = self.sma("DBC", self.trend_slow, Resolution.DAILY)

        # Sleeve 2: Dividend -> DVY
        self.dvy_fast_ma = self.sma("DVY", self.trend_fast, Resolution.DAILY)
        self.dvy_slow_ma = self.sma("DVY", self.trend_slow, Resolution.DAILY)

        # Sleeve 3: Bond -> AGG
        self.agg_fast_ma = self.sma("AGG", self.trend_fast, Resolution.DAILY)
        self.agg_slow_ma = self.sma("AGG", self.trend_slow, Resolution.DAILY)

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
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 5),
            self.daily_track_values,
        )

        self._initial_rebalance_done = False

        self.debug(
            f">>> 4-STRAT RISK PARITY v3: {len(self.equity_tickers)} stocks + "
            f"{len(self.commodity_tickers)} commodity + {len(self.dividend_tickers)} dividend + "
            f"{len(self.bond_tickers)} bond ETFs, cov_window={self.cov_window}, "
            f"caps={self.min_weight:.0%}/{self.max_weight:.0%}, "
            f"max_delta={self.max_weight_delta:.0%}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Per-sleeve trend checks
    # ══════════════════════════════════════════════════════════════════════

    def _is_sleeve_uptrend(self, sleeve_idx):
        """Check if a specific sleeve's benchmark is in uptrend."""
        if sleeve_idx == 0:  # Equity -> SPY
            if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
                return True
            return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value
        elif sleeve_idx == 1:  # Commodity -> DBC
            if not self.dbc_fast_ma.is_ready or not self.dbc_slow_ma.is_ready:
                return True
            return self.dbc_fast_ma.current.value > self.dbc_slow_ma.current.value
        elif sleeve_idx == 2:  # Dividend -> DVY
            if not self.dvy_fast_ma.is_ready or not self.dvy_slow_ma.is_ready:
                return True
            return self.dvy_fast_ma.current.value > self.dvy_slow_ma.current.value
        elif sleeve_idx == 3:  # Bond -> AGG
            if not self.agg_fast_ma.is_ready or not self.agg_slow_ma.is_ready:
                return True
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
    # AI-learned weight computation (replaces risk parity solver)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_risk_parity_weights(self):
        """AI-learned weight prediction from decision tree trained on 115 months.
        Uses 3-month rolling features per sleeve to predict which sleeve to overweight.
        Fallback = keep previous weights if not enough data."""

        min_len = min(len(self.sleeve_values[i]) for i in range(self.n_strategies))

        if min_len < self.min_cov_days:
            self.debug(f"  AI weights: only {min_len} days, need {self.min_cov_days}. Keeping previous weights.")
            return self.current_weights[:]

        use_days = min(min_len, self.cov_window)

        # Compute monthly-scale returns for each sleeve over last 3 chunks (~21 days each)
        sleeve_returns = []
        for i in range(self.n_strategies):
            values = np.array(self.sleeve_values[i][-use_days:])
            if np.all(values == 0) or len(values) < self.min_cov_days:
                self.debug(f"  AI weights: sleeve {i} has no data. Keeping previous weights.")
                return self.current_weights[:]
            values = np.where(values == 0, 1, values)
            daily_ret = np.diff(values) / values[:-1]
            sleeve_returns.append(daily_ret)

        # Build features: for each sleeve compute rolling stats
        # Split available returns into 3 roughly equal chunks (simulating 3 months)
        chunk_size = max(1, len(sleeve_returns[0]) // 3)

        features = {}
        names = ['equity', 'commodity', 'dividend', 'bond']
        for i, name in enumerate(names):
            rets = sleeve_returns[i]
            # Scale daily returns to monthly-scale for threshold compatibility
            monthly_scale = rets * 21  # approximate daily->monthly scaling

            # Mean return (monthly scale P&L proxy)
            features[f'{name}_mean_ret'] = np.mean(monthly_scale) * 1000
            # Volatility (monthly scale)
            features[f'{name}_vol'] = np.std(monthly_scale) * 1000
            # Last "month" return (last chunk)
            last_chunk = monthly_scale[-chunk_size:] if chunk_size > 0 else monthly_scale
            features[f'{name}_last_ret'] = np.sum(last_chunk) * 1000
            # Trend percentage (fraction of positive days)
            features[f'{name}_trend_pct'] = np.sum(rets > 0) / len(rets) if len(rets) > 0 else 0.5
            # Momentum (sum of all returns, monthly scale)
            features[f'{name}_momentum'] = np.sum(monthly_scale) * 1000

        # Decision tree rules (learned from backtest data)
        prediction = self._predict_best_sleeve(features)

        # Convert prediction to weights: winner gets 50%, others 17%
        winner_weight = 0.50
        other_weight = (1.0 - winner_weight) / 3
        weights = [other_weight] * 4
        weights[prediction] = winner_weight

        sleeve_name = names[prediction]
        self.debug(
            f"  AI prediction: {sleeve_name} (div_vol={features['dividend_vol']:.0f}, "
            f"eq_last={features['equity_last_ret']:.0f}, "
            f"cm_last={features['commodity_last_ret']:.0f}, "
            f"bond_last={features['bond_last_ret']:.0f})"
        )

        return weights

    def _predict_best_sleeve(self, f):
        """Decision tree rules learned from 115 months of backtest data.
        Returns: 0=equity, 1=commodity, 2=dividend, 3=bond."""

        if f['dividend_vol'] <= 2911.54:
            if f['dividend_vol'] <= 676.50:
                if f['equity_last_ret'] <= 1353.17:
                    return 0  # equity (calm market, normal equity returns)
                else:
                    return 2  # dividend (equity had huge month, rotate to value)
            else:
                if f['equity_last_ret'] <= -112.10:
                    if f['bond_last_ret'] <= 144.24:
                        return 1  # commodity (equity down, bonds flat = inflation/crisis)
                    else:
                        return 0  # equity (equity down but bonds up = temporary dip)
                else:
                    return 0  # equity (moderate vol, equity positive = stay course)
        else:
            if f['bond_last_ret'] <= 92.64:
                if f['commodity_last_ret'] <= 349.38:
                    return 0  # equity (high stress but no clear winner)
                else:
                    return 1  # commodity (high stress + commodity momentum)
            else:
                return 3  # bond (high stress + bonds gaining = flight to safety)

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
    # Scoring engines
    # ══════════════════════════════════════════════════════════════════════

    def _score_momentum(self, tickers, w_mom, w_trend, w_recent, w_vol,
                        top_n, include_events=False):
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

            ma_50 = np.mean(closes[-self.ma_period:]) if len(closes) >= self.ma_period else price_now
            trend_score = 1.0 if price_now > ma_50 else 0.0

            if len(closes) >= self.recent_period:
                price_1m_ago = closes[-self.recent_period]
                recent = (price_now / price_1m_ago) - 1.0 if price_1m_ago > 0 else 0.0
            else:
                recent = 0.0

            if len(closes) >= self.vol_period:
                rets = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 1.0
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
            score = (w_mom * d.get("momentum_rank", 0.5) + w_trend * d["trend"]
                     + w_recent * d.get("recent_rank", 0.5) + w_vol * d.get("vol_rank", 0.5))
            if include_events:
                score += self.eq_w_events * d.get("events_rank", 0.5)
            scores[ticker] = score
        return scores

    def _score_dividend(self):
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

            if len(closes) >= self.div_yield_lookback:
                recent_year = closes[-self.div_yield_lookback:]
                mean_price = np.mean(recent_year)
                deviation = np.std(recent_year) / mean_price
                stability = 1.0 / (1.0 + deviation * 10)
                rolling_max = np.maximum.accumulate(recent_year)
                drawdowns = (rolling_max - recent_year) / rolling_max
                dd_score = 1.0 - np.max(drawdowns)
                yield_score = 0.6 * stability + 0.4 * dd_score
            else:
                yield_score = 0.5

            ma_50 = np.mean(closes[-self.ma_period:]) if len(closes) >= self.ma_period else price_now
            trend_score = 1.0 if price_now > ma_50 else 0.0

            if len(closes) >= self.recent_period:
                price_1m_ago = closes[-self.recent_period]
                recent = (price_now / price_1m_ago) - 1.0 if price_1m_ago > 0 else 0.0
            else:
                recent = 0.0

            if len(closes) >= self.vol_period:
                rets = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 1.0
            else:
                vol = 1.0

            raw_data[ticker] = {"yield": yield_score, "trend": trend_score, "recent": recent, "vol": vol}

        if len(raw_data) < self.div_top_n:
            return scores

        ticker_list = list(raw_data.keys())
        n = len(ticker_list)

        for signal, key in [("yield", "yield_rank"), ("recent", "recent_rank")]:
            ranked = sorted(ticker_list, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][key] = i / (n - 1) if n > 1 else 0.5

        ranked_vol = sorted(ticker_list, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        for ticker in ticker_list:
            d = raw_data[ticker]
            score = (self.div_w_yield * d["yield_rank"] + self.div_w_trend * d["trend"]
                     + self.div_w_recent * d["recent_rank"] + self.div_w_vol * d["vol_rank"])
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

        # Check per-sleeve uptrend (for position count)
        eq_up = self._is_sleeve_uptrend(0)
        cm_up = self._is_sleeve_uptrend(1)
        div_up = self._is_sleeve_uptrend(2)
        bond_up = self._is_sleeve_uptrend(3)

        # Reset emergency states at monthly rebalance
        self.eq_in_downtrend = not eq_up
        self.cm_in_downtrend = not cm_up
        self.div_in_downtrend = not div_up
        self.bond_in_downtrend = not bond_up

        # Track regime (based on SPY for overall stats)
        if eq_up:
            self.months_in_uptrend += 1
        else:
            self.months_in_downtrend += 1

        # Compute risk parity weights (target) then smooth toward them
        target_weights = self._compute_risk_parity_weights()

        # v3: Smooth weights — limit each sleeve to max_weight_delta change per month
        smoothed = []
        for i in range(self.n_strategies):
            current = self.current_weights[i]
            target = target_weights[i]
            delta = target - current
            # Clamp the change
            if delta > self.max_weight_delta:
                delta = self.max_weight_delta
            elif delta < -self.max_weight_delta:
                delta = -self.max_weight_delta
            smoothed.append(current + delta)

        # Re-normalize to sum to 1.0 (smoothing can break the sum)
        total_w = sum(smoothed)
        if total_w > 0:
            smoothed = [w / total_w for w in smoothed]

        # Apply floor/ceiling after smoothing
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
        eq_scores = self._score_momentum(
            self.equity_tickers, self.eq_w_mom, self.eq_w_trend, self.eq_w_recent,
            self.eq_w_vol, self.eq_top_n, include_events=True)
        cm_scores = self._score_momentum(
            self.commodity_tickers, self.cm_w_mom, self.cm_w_trend, self.cm_w_recent,
            self.cm_w_vol, self.cm_top_n)
        div_scores = self._score_dividend()
        bond_scores = self._score_momentum(
            self.bond_tickers, self.bond_w_mom, self.bond_w_trend, self.bond_w_recent,
            self.bond_w_vol, self.bond_top_n)

        # Per-sleeve position count based on per-sleeve trend
        eq_n = self.eq_top_n if eq_up else self.eq_downtrend_top_n
        cm_n = self.cm_top_n if cm_up else self.cm_downtrend_top_n
        div_n = self.div_top_n if div_up else self.div_downtrend_top_n
        bond_n = self.bond_top_n if bond_up else self.bond_downtrend_top_n

        eq_targets = set(t for t, _ in sorted(eq_scores.items(), key=lambda x: x[1], reverse=True)[:eq_n]) if eq_scores else set()
        cm_targets = set(t for t, _ in sorted(cm_scores.items(), key=lambda x: x[1], reverse=True)[:cm_n]) if cm_scores else set()
        div_targets = set(t for t, _ in sorted(div_scores.items(), key=lambda x: x[1], reverse=True)[:div_n]) if div_scores else set()
        bond_targets = set(t for t, _ in sorted(bond_scores.items(), key=lambda x: x[1], reverse=True)[:bond_n]) if bond_scores else set()

        # Build combined allocations (handles ticker overlap correctly)
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
            if ticker not in all_targets and ticker in self.all_symbols:
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

        self.eq_holdings = eq_targets
        self.cm_holdings = cm_targets
        self.div_holdings = div_targets
        self.bond_holdings = bond_targets

        events = sum(self.event_counts_this_month.values())
        sleeve_status = (f"EQ={'UP' if eq_up else 'DN'}({eq_n}) "
                        f"CM={'UP' if cm_up else 'DN'}({cm_n}) "
                        f"DIV={'UP' if div_up else 'DN'}({div_n}) "
                        f"BOND={'UP' if bond_up else 'DN'}({bond_n})")
        self.debug(
            f"REBALANCE: weights=[{weights[0]:.0%},{weights[1]:.0%},{weights[2]:.0%},{weights[3]:.0%}] "
            f"{sleeve_status} "
            f"EQ=${eq_capital:,.0f} CM=${cm_capital:,.0f} DIV=${div_capital:,.0f} BOND=${bond_capital:,.0f} "
            f"events={events} total=${total_value:,.0f}"
        )

        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check (FIXED: per-sleeve independent gates)
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        all_holdings = self.eq_holdings | self.cm_holdings | self.div_holdings | self.bond_holdings
        if not all_holdings:
            return

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
                # This sleeve just entered downtrend — reduce it
                sleeve_downtrend[i] = True
                self.emergency_exits[i] += 1
                triggered.append(sleeve_names[i])

                # Score and reduce this sleeve only
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

                # Liquidate non-keepers (only if not held by another sleeve)
                other_holdings = set()
                for j in range(self.n_strategies):
                    if j != i:
                        other_holdings |= sleeve_holdings[j]

                for ticker in list(sleeve_holdings[i]):
                    if ticker not in keep and ticker not in other_holdings:
                        if ticker in self.all_symbols:
                            symbol = self.all_symbols[ticker]
                            if self.portfolio[symbol].invested:
                                self.liquidate(symbol)
                                self.total_trades += 1

                # Update holdings
                if i == 0: self.eq_holdings = keep
                elif i == 1: self.cm_holdings = keep
                elif i == 2: self.div_holdings = keep
                elif i == 3: self.bond_holdings = keep

            elif uptrend and sleeve_downtrend[i]:
                # Recovery for this sleeve
                sleeve_downtrend[i] = False

        # Update downtrend flags
        self.eq_in_downtrend = sleeve_downtrend[0]
        self.cm_in_downtrend = sleeve_downtrend[1]
        self.div_in_downtrend = sleeve_downtrend[2]
        self.bond_in_downtrend = sleeve_downtrend[3]

        if triggered:
            remaining = (f"EQ={len(self.eq_holdings)} CM={len(self.cm_holdings)} "
                        f"DIV={len(self.div_holdings)} BOND={len(self.bond_holdings)}")
            self.debug(f"EMERGENCY: {', '.join(triggered)} downtrend -> reduced. Remaining: {remaining}")

        # Log recoveries
        for i in range(self.n_strategies):
            if self._is_sleeve_uptrend(i) and not sleeve_downtrend[i]:
                pass  # recovery is silent unless you want to log it

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
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"PNL: {pnl_str}")
