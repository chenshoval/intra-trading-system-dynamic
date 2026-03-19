"""Combined Strategy v1 — v2 Equity (75%) + Commodity Momentum (25%)

Two uncorrelated strategies in one algorithm with fixed capital allocation.
Based on portfolio math showing:
- v2 alone:     Sharpe 0.850, MaxDD 32.4%
- 75/25 split:  Sharpe 1.37,  MaxDD ~27%  (theoretical)
- Correlation between strategies: 0.11

This backtest validates the theoretical numbers with actual execution.

How it works:
- 75% of capital → v2 equity rotator (50 stocks, top 15, 5 signals)
- 25% of capital → commodity momentum (15 ETFs, top 4, 4 signals)
- Both rebalance monthly on the same day
- Each strategy manages its own positions independently
- Wednesday emergency check applies to BOTH sleeves
- Monthly rebalance re-enforces the 75/25 split (drift correction)

Test periods (change dates per run):
- Run 1: 2016-2020 (standard gate)
- Run 2: 2020-2023 (commodity supercycle + bear market)
- Run 3: 2023-2025 (recent data)
Results: results_from_quant_connect/combinedv2commodity/{period}/
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class CombinedV2CommodityV1(QCAlgorithm):

    def initialize(self):
        # ── Test periods ──
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ══════════════════════════════════════════════════════════════
        # Capital allocation
        # ══════════════════════════════════════════════════════════════
        self.equity_weight = 0.75        # 75% to v2 equity rotator
        self.commodity_weight = 0.25     # 25% to commodity momentum

        # ══════════════════════════════════════════════════════════════
        # SLEEVE 1: v2 Equity Rotator parameters
        # ══════════════════════════════════════════════════════════════
        self.eq_top_n = 15
        self.eq_downtrend_top_n = 5
        self.trend_fast = 10
        self.trend_slow = 50

        # v2 signal weights
        self.eq_w_momentum = 0.35
        self.eq_w_trend = 0.25
        self.eq_w_recent = 0.20
        self.eq_w_volatility = 0.10
        self.eq_w_events = 0.10

        # ══════════════════════════════════════════════════════════════
        # SLEEVE 2: Commodity Momentum parameters
        # ══════════════════════════════════════════════════════════════
        self.cm_top_n = 4
        self.cm_downtrend_top_n = 3

        # Commodity signal weights (no events)
        self.cm_w_momentum = 0.40
        self.cm_w_trend = 0.25
        self.cm_w_recent = 0.20
        self.cm_w_volatility = 0.15

        # ══════════════════════════════════════════════════════════════
        # Shared momentum parameters
        # ══════════════════════════════════════════════════════════════
        self.mom_lookback = 126
        self.mom_skip = 21
        self.ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

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
            "GLD", "SLV", "PPLT",           # Precious metals
            "XLE", "USO", "UNG",             # Energy
            "DBA", "MOO", "WEAT", "CORN", "SOYB",  # Agriculture
            "CPER",                          # Base metals
            "DBC", "PDBC",                   # Broad commodity
            "TLT",                           # Safe haven bond
        ]

        # ── Event keywords (equity sleeve only) ──
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
        self.eq_symbols = {}
        self.cm_symbols = {}
        self.news_symbols = {}

        self.eq_holdings = set()
        self.cm_holdings = set()
        self.is_in_cash = False

        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        # ── Tracking ──
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
        # Add all securities
        # ══════════════════════════════════════════════════════════════

        # Equity sleeve + news
        for ticker in self.equity_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.eq_symbols[ticker] = equity.symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        # Commodity sleeve (avoid duplicates with equity sleeve)
        for ticker in self.commodity_tickers:
            if ticker in self.eq_symbols:
                # XLE, XOM etc. might overlap — reuse symbol
                self.cm_symbols[ticker] = self.eq_symbols[ticker]
            else:
                equity = self.add_equity(ticker, Resolution.DAILY)
                self.cm_symbols[ticker] = equity.symbol

        # SPY for trend gate
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
            f">>> COMBINED v2+COMMODITY: {len(self.equity_tickers)} stocks ({self.equity_weight:.0%}) + "
            f"{len(self.commodity_tickers)} ETFs ({self.commodity_weight:.0%})"
        )

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Event collection (equity sleeve only)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        if not self._initial_rebalance_done:
            self._initial_rebalance_done = True
            invested = sum(1 for t in self.equity_tickers if self.portfolio[self.eq_symbols[t]].invested)
            invested += sum(1 for t in self.commodity_tickers if self.portfolio[self.cm_symbols[t]].invested)
            if invested == 0:
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
    # Scoring: Equity sleeve (v2 — 5 signals)
    # ══════════════════════════════════════════════════════════════════════

    def _score_equities(self):
        scores = {}
        raw_data = {}

        for ticker in self.equity_tickers:
            symbol = self.eq_symbols[ticker]
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

            event_count = self.event_counts_this_month.get(ticker, 0)

            raw_data[ticker] = {
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol, "events": event_count,
            }

        if len(raw_data) < self.eq_top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        for signal in ["momentum", "recent"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_vol = sorted(tickers, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_events = sorted(tickers, key=lambda t: raw_data[t]["events"])
        for i, t in enumerate(ranked_events):
            raw_data[t]["events_rank"] = i / (n - 1) if n > 1 else 0.5

        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.eq_w_momentum * d.get("momentum_rank", 0.5)
                + self.eq_w_trend * d["trend"]
                + self.eq_w_recent * d.get("recent_rank", 0.5)
                + self.eq_w_volatility * d.get("vol_rank", 0.5)
                + self.eq_w_events * d.get("events_rank", 0.5)
            )
            scores[ticker] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Scoring: Commodity sleeve (4 signals)
    # ══════════════════════════════════════════════════════════════════════

    def _score_commodities(self):
        scores = {}
        raw_data = {}

        for ticker in self.commodity_tickers:
            symbol = self.cm_symbols[ticker]
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

            raw_data[ticker] = {
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol,
            }

        if len(raw_data) < self.cm_top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        for signal in ["momentum", "recent"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        ranked_vol = sorted(tickers, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.cm_w_momentum * d.get("momentum_rank", 0.5)
                + self.cm_w_trend * d["trend"]
                + self.cm_w_recent * d.get("recent_rank", 0.5)
                + self.cm_w_volatility * d.get("vol_rank", 0.5)
            )
            scores[ticker] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance — both sleeves
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

        # Calculate capital for each sleeve (enforces 75/25 split)
        total_value = self.portfolio.total_portfolio_value
        equity_capital = total_value * self.equity_weight
        commodity_capital = total_value * self.commodity_weight

        # ── Sleeve 1: Equity (v2) ──
        eq_scores = self._score_equities()
        eq_n_hold = self.eq_top_n if uptrend else self.eq_downtrend_top_n

        eq_targets = set()
        if eq_scores:
            eq_ranked = sorted(eq_scores.items(), key=lambda x: x[1], reverse=True)
            eq_targets = set(t for t, _ in eq_ranked[:eq_n_hold])

        # ── Sleeve 2: Commodity ──
        cm_scores = self._score_commodities()
        cm_n_hold = self.cm_top_n if uptrend else self.cm_downtrend_top_n

        cm_targets = set()
        if cm_scores:
            cm_ranked = sorted(cm_scores.items(), key=lambda x: x[1], reverse=True)
            cm_targets = set(t for t, _ in cm_ranked[:cm_n_hold])

        # ── Liquidate positions not in either target set ──
        for ticker in list(self.eq_holdings):
            if ticker not in eq_targets:
                symbol = self.eq_symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.eq_holdings.discard(ticker)

        for ticker in list(self.cm_holdings):
            if ticker not in cm_targets:
                symbol = self.cm_symbols[ticker]
                # Don't liquidate if it's also in equity targets (overlap like XLE)
                if ticker not in eq_targets and self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.cm_holdings.discard(ticker)

        # ── Buy equity sleeve ──
        if eq_targets and eq_n_hold > 0:
            eq_alloc = equity_capital / eq_n_hold
            for ticker in eq_targets:
                symbol = self.eq_symbols[ticker]
                price = self.securities[symbol].price
                if price <= 0:
                    continue
                target_qty = int(eq_alloc / price)
                if target_qty < 1:
                    continue
                current_qty = int(self.portfolio[symbol].quantity)
                # If ticker is also in commodity targets, add both allocations
                if ticker in cm_targets:
                    cm_alloc_per = commodity_capital / cm_n_hold
                    target_qty = int((eq_alloc + cm_alloc_per) / price)
                delta = target_qty - current_qty
                if abs(delta) > 0:
                    self.market_order(symbol, delta)
                    self.total_trades += 1
                self.eq_holdings.add(ticker)

        # ── Buy commodity sleeve ──
        if cm_targets and cm_n_hold > 0:
            cm_alloc = commodity_capital / cm_n_hold
            for ticker in cm_targets:
                # Skip if already handled in equity sleeve (overlap)
                if ticker in eq_targets:
                    self.cm_holdings.add(ticker)
                    continue
                symbol = self.cm_symbols[ticker]
                price = self.securities[symbol].price
                if price <= 0:
                    continue
                target_qty = int(cm_alloc / price)
                if target_qty < 1:
                    continue
                current_qty = int(self.portfolio[symbol].quantity)
                delta = target_qty - current_qty
                if abs(delta) > 0:
                    self.market_order(symbol, delta)
                    self.total_trades += 1
                self.cm_holdings.add(ticker)

        # ── Logging ──
        events_this_month = sum(self.event_counts_this_month.values())
        regime = "UP" if uptrend else "DOWN"

        eq_top3 = [(t, f"{s:.3f}") for t, s in eq_ranked[:3]] if eq_scores else []
        cm_top3 = [(t, f"{s:.3f}") for t, s in cm_ranked[:3]] if cm_scores else []

        self.debug(
            f"REBALANCE [{regime}]: EQ={eq_n_hold} stocks (${equity_capital:,.0f}), "
            f"CM={cm_n_hold} ETFs (${commodity_capital:,.0f}), "
            f"events={events_this_month}, eq_top3={eq_top3}, cm_top3={cm_top3}, "
            f"total=${total_value:,.0f}"
        )

        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check — both sleeves
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        if not self.eq_holdings and not self.cm_holdings:
            return

        uptrend = self._is_uptrend()

        if not uptrend and not self.is_in_cash:
            self.is_in_cash = True
            self.emergency_exits += 1

            total_value = self.portfolio.total_portfolio_value
            equity_capital = total_value * self.equity_weight
            commodity_capital = total_value * self.commodity_weight

            # ── Equity sleeve: reduce to downtrend_top_n ──
            eq_scores = self._score_equities()
            eq_keep = set()
            if eq_scores:
                eq_ranked = sorted(eq_scores.items(), key=lambda x: x[1], reverse=True)
                eq_keep = set(t for t, _ in eq_ranked[:self.eq_downtrend_top_n])

            for ticker in list(self.eq_holdings):
                if ticker not in eq_keep:
                    symbol = self.eq_symbols[ticker]
                    if self.portfolio[symbol].invested and ticker not in self.cm_holdings:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.eq_holdings.discard(ticker)

            if eq_keep:
                eq_alloc = equity_capital / len(eq_keep)
                for ticker in eq_keep:
                    symbol = self.eq_symbols[ticker]
                    price = self.securities[symbol].price
                    if price <= 0:
                        continue
                    target_qty = int(eq_alloc / price)
                    current_qty = int(self.portfolio[symbol].quantity)
                    delta = target_qty - current_qty
                    if abs(delta) > 0:
                        self.market_order(symbol, delta)
                        self.total_trades += 1

            # ── Commodity sleeve: reduce to downtrend_top_n ──
            cm_scores = self._score_commodities()
            cm_keep = set()
            if cm_scores:
                cm_ranked = sorted(cm_scores.items(), key=lambda x: x[1], reverse=True)
                cm_keep = set(t for t, _ in cm_ranked[:self.cm_downtrend_top_n])

            for ticker in list(self.cm_holdings):
                if ticker not in cm_keep:
                    symbol = self.cm_symbols[ticker]
                    if self.portfolio[symbol].invested and ticker not in self.eq_holdings:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.cm_holdings.discard(ticker)

            if cm_keep:
                cm_alloc = commodity_capital / len(cm_keep)
                for ticker in cm_keep:
                    if ticker in eq_keep:
                        continue  # already sized in equity sleeve
                    symbol = self.cm_symbols[ticker]
                    price = self.securities[symbol].price
                    if price <= 0:
                        continue
                    target_qty = int(cm_alloc / price)
                    current_qty = int(self.portfolio[symbol].quantity)
                    delta = target_qty - current_qty
                    if abs(delta) > 0:
                        self.market_order(symbol, delta)
                        self.total_trades += 1

            self.debug(
                f"EMERGENCY: downtrend, EQ→{len(self.eq_holdings)} stocks, "
                f"CM→{len(self.cm_holdings)} ETFs"
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
            f"EmergencyExits={self.emergency_exits} "
            f"Events={self.total_events_detected}"
        )
        self.debug(
            f"REGIME: Up={self.months_in_uptrend} Down={self.months_in_downtrend}"
        )
        self.debug(
            f"HOLDINGS: EQ={len(self.eq_holdings)} stocks, CM={len(self.cm_holdings)} ETFs"
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
