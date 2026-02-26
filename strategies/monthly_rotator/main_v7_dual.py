"""Monthly Rotator v7 — Dual Engine: v2 Stocks + v5 Sectors

Two uncorrelated streams optimized for different regimes:
- 70% capital: v2 stock rotation (50 stocks, event-boosted scoring)
  → Dominates in bull markets (43% CAR, Sharpe 1.45 on 2016-2020)
- 30% capital: v5 sector rotation (11 ETFs, long-short)
  → Dominates in bear markets (12% CAR, Sharpe 0.56 on 2022-2023)

Why this combination:
- v2 won 9/20 metric contests overall — best bull market strategy
- v5 won 5/20 — specifically the bear market periods (lowest DD, only
  strategy with positive Sharpe in 2022-2023 besides v2)
- They're uncorrelated: v2 picks individual stocks, v5 rotates sectors
- v5's short leg (bottom 3 sectors) provides natural bear hedge

What failed and why we're not using it:
- v3 (short bottom 5 of 50): 0 wins, shorting quality stocks doesn't work
- v4 (SPY hedge): hedge didn't add enough value, still lost in 2022
- v6 (trend overlay): SPY sat as dead weight for 2 years, whipsawed in 2022
- v3b (S&P 500 LS): high alpha but 49% drawdown — unacceptable
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV7DualEngine(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Capital split ──
        self.stock_pct = 0.70               # 70% to individual stock rotation
        self.sector_pct = 0.30              # 30% to sector long-short

        # ── Shared: SPY trend gate ──
        self.trend_fast = 10
        self.trend_slow = 50

        # ══════════════════════════════════════════════════════════════
        # ENGINE 1: Stock rotation (v2 logic)
        # ══════════════════════════════════════════════════════════════

        self.stock_top_n = 15
        self.stock_downtrend_n = 5

        self.stock_w_momentum = 0.35
        self.stock_w_trend = 0.25
        self.stock_w_recent = 0.20
        self.stock_w_vol = 0.10
        self.stock_w_events = 0.10

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
        # ENGINE 2: Sector rotation (v5 logic)
        # ══════════════════════════════════════════════════════════════

        self.sector_long_n = 4
        self.sector_short_n = 3
        self.sector_downtrend_long_n = 2
        self.sector_downtrend_short_n = 3   # keep shorts in downtrend

        self.sector_w_momentum = 0.45
        self.sector_w_trend = 0.30
        self.sector_w_recent = 0.15
        self.sector_w_vol = 0.10

        self.sector_tickers = [
            "XLK", "XLF", "XLE", "XLV", "XLI",
            "XLC", "XLY", "XLP", "XLB", "XLRE", "XLU",
        ]

        # ── Shared parameters ──
        self.mom_lookback = 126
        self.mom_skip = 21
        self.ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        # ── Data structures ──
        self.stock_symbols = {}
        self.news_symbols = {}
        self.sector_symbols = {}
        self.stock_holdings = set()
        self.sector_long_holdings = set()
        self.sector_short_holdings = set()
        self.last_rebalance = None
        self.is_in_cash = False
        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        # ── Counters ──
        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)

        # ── Add stocks + news ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.stock_symbols[ticker] = equity.symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        # ── Add sector ETFs ──
        for ticker in self.sector_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.sector_symbols[ticker] = equity.symbol

        # ── SPY ──
        self.add_equity("SPY", Resolution.DAILY)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

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

        self.debug(
            f">>> ROTATOR v7 DUAL: stocks={self.stock_pct:.0%} ({len(self.target_tickers)}) "
            f"sectors={self.sector_pct:.0%} ({len(self.sector_tickers)})"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Shared
    # ══════════════════════════════════════════════════════════════════════

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    def _get_history(self, symbol):
        """Get price history for scoring."""
        history = self.history(symbol, self.mom_lookback + self.mom_skip + 10, Resolution.DAILY)
        if history is None or history.empty:
            return None
        try:
            closes = history["close"].values
        except Exception:
            return None
        if len(closes) < self.mom_lookback + self.mom_skip:
            return None
        return closes

    def _compute_signals(self, closes):
        """Compute raw signal values from price array."""
        price_now = closes[-1]
        if price_now <= 0:
            return None

        price_6m = closes[0]
        price_1m = closes[-self.mom_skip]
        momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0

        if len(closes) >= self.ma_period:
            ma = np.mean(closes[-self.ma_period:])
            trend = 1.0 if price_now > ma else 0.0
        else:
            trend = 0.5

        if len(closes) >= self.recent_period:
            p = closes[-self.recent_period]
            recent = (price_now / p) - 1.0 if p > 0 else 0.0
        else:
            recent = 0.0

        if len(closes) >= self.vol_period:
            rets = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
            vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 1.0
        else:
            vol = 1.0

        return {"momentum": momentum, "trend": trend, "recent": recent, "vol": vol}

    def _rank_normalize(self, raw_data, keys_list):
        """Rank-normalize signals across a set of items."""
        n = len(keys_list)
        if n <= 1:
            return

        for signal in ["momentum", "recent"]:
            ranked = sorted(keys_list, key=lambda k: raw_data[k][signal])
            for i, k in enumerate(ranked):
                raw_data[k][f"{signal}_rank"] = i / (n - 1)

        ranked_vol = sorted(keys_list, key=lambda k: raw_data[k]["vol"], reverse=True)
        for i, k in enumerate(ranked_vol):
            raw_data[k]["vol_rank"] = i / (n - 1)

    # ══════════════════════════════════════════════════════════════════════
    # Event collection
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        for ticker in self.target_tickers:
            ns = self.news_symbols[ticker]
            if not data.contains_key(ns):
                continue
            article = data[ns]
            title = str(getattr(article, "title", "")).lower()
            desc = str(getattr(article, "description", "")).lower()
            text = f"{title} {desc}"
            for keywords in self.event_keywords.values():
                if any(kw in text for kw in keywords):
                    self.event_counts_this_month[ticker] += 1
                    self.total_events_detected += 1
                    break

    # ══════════════════════════════════════════════════════════════════════
    # Stock scoring (v2 logic)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks(self):
        scores = {}
        raw_data = {}

        for ticker in self.target_tickers:
            closes = self._get_history(self.stock_symbols[ticker])
            if closes is None:
                continue
            signals = self._compute_signals(closes)
            if signals is None:
                continue
            signals["events"] = self.event_counts_this_month.get(ticker, 0)
            raw_data[ticker] = signals

        if len(raw_data) < self.stock_top_n:
            return scores

        tickers = list(raw_data.keys())
        self._rank_normalize(raw_data, tickers)

        # Event rank
        n = len(tickers)
        ranked_events = sorted(tickers, key=lambda t: raw_data[t]["events"])
        for i, t in enumerate(ranked_events):
            raw_data[t]["events_rank"] = i / (n - 1) if n > 1 else 0.5

        for ticker in tickers:
            d = raw_data[ticker]
            scores[ticker] = (
                self.stock_w_momentum * d.get("momentum_rank", 0.5)
                + self.stock_w_trend * d["trend"]
                + self.stock_w_recent * d.get("recent_rank", 0.5)
                + self.stock_w_vol * d.get("vol_rank", 0.5)
                + self.stock_w_events * d.get("events_rank", 0.5)
            )
        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Sector scoring (v5 logic)
    # ══════════════════════════════════════════════════════════════════════

    def _score_sectors(self):
        scores = {}
        raw_data = {}

        for ticker in self.sector_tickers:
            closes = self._get_history(self.sector_symbols[ticker])
            if closes is None:
                continue
            signals = self._compute_signals(closes)
            if signals is None:
                continue
            raw_data[ticker] = signals

        if len(raw_data) < self.sector_long_n + self.sector_short_n:
            return scores

        tickers = list(raw_data.keys())
        self._rank_normalize(raw_data, tickers)

        for ticker in tickers:
            d = raw_data[ticker]
            scores[ticker] = (
                self.sector_w_momentum * d.get("momentum_rank", 0.5)
                + self.sector_w_trend * d["trend"]
                + self.sector_w_recent * d.get("recent_rank", 0.5)
                + self.sector_w_vol * d.get("vol_rank", 0.5)
            )
        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance — both engines
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

        total_value = self.portfolio.total_portfolio_value

        # ── ENGINE 1: Stock rotation ──
        stock_scores = self._score_stocks()
        if stock_scores:
            n_hold = self.stock_top_n if uptrend else self.stock_downtrend_n
            ranked = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            target = set(t for t, _ in ranked[:n_hold])

            for ticker in list(self.stock_holdings):
                if ticker not in target:
                    sym = self.stock_symbols[ticker]
                    if self.portfolio[sym].invested:
                        self.liquidate(sym)
                        self.total_trades += 1
                    self.stock_holdings.discard(ticker)

            stock_capital = total_value * self.stock_pct
            if stock_capital > 0 and n_hold > 0:
                alloc = stock_capital / n_hold
                for ticker in target:
                    sym = self.stock_symbols[ticker]
                    price = self.securities[sym].price
                    if price <= 0:
                        continue
                    tgt_qty = int(alloc / price)
                    if tgt_qty < 1:
                        continue
                    cur_qty = int(self.portfolio[sym].quantity)
                    delta = tgt_qty - cur_qty
                    if abs(delta) > 0:
                        self.market_order(sym, delta)
                        self.total_trades += 1
                    self.stock_holdings.add(ticker)

        # ── ENGINE 2: Sector rotation ──
        sector_scores = self._score_sectors()
        if sector_scores:
            ranked_s = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
            n_long = self.sector_long_n if uptrend else self.sector_downtrend_long_n
            n_short = self.sector_short_n if uptrend else self.sector_downtrend_short_n
            target_long = set(t for t, _ in ranked_s[:n_long])
            target_short = set(t for t, _ in ranked_s[-n_short:])

            # Close unwanted sector positions
            for ticker in list(self.sector_long_holdings | self.sector_short_holdings):
                if ticker not in target_long and ticker not in target_short:
                    sym = self.sector_symbols[ticker]
                    if self.portfolio[sym].invested:
                        self.liquidate(sym)
                        self.total_trades += 1
                    self.sector_long_holdings.discard(ticker)
                    self.sector_short_holdings.discard(ticker)

            sector_capital = total_value * self.sector_pct
            if sector_capital > 0:
                # Long sectors — 80% of sector capital
                long_alloc = sector_capital * 0.8 / n_long if n_long > 0 else 0
                for ticker in target_long:
                    sym = self.sector_symbols[ticker]
                    price = self.securities[sym].price
                    if price <= 0:
                        continue
                    tgt_qty = int(long_alloc / price)
                    if tgt_qty < 1:
                        continue
                    cur_qty = int(self.portfolio[sym].quantity)
                    delta = tgt_qty - cur_qty
                    if abs(delta) > 0:
                        self.market_order(sym, delta)
                        self.total_trades += 1
                    self.sector_long_holdings.add(ticker)

                # Short sectors — 20% of sector capital
                short_alloc = sector_capital * 0.2 / n_short if n_short > 0 else 0
                for ticker in target_short:
                    sym = self.sector_symbols[ticker]
                    price = self.securities[sym].price
                    if price <= 0:
                        continue
                    tgt_qty = -int(short_alloc / price)
                    if tgt_qty >= 0:
                        continue
                    cur_qty = int(self.portfolio[sym].quantity)
                    delta = tgt_qty - cur_qty
                    if abs(delta) > 0:
                        self.market_order(sym, delta)
                        self.total_trades += 1
                    self.sector_short_holdings.add(ticker)

        regime = "UP" if uptrend else "DOWN"
        self.debug(
            f"REBALANCE [{regime}]: stocks={len(self.stock_holdings)} "
            f"sec_long={len(self.sector_long_holdings)} sec_short={len(self.sector_short_holdings)} "
            f"eq=${total_value:,.0f}"
        )
        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        if not self.stock_holdings and not self.sector_long_holdings:
            return

        uptrend = self._is_uptrend()

        if not uptrend and not self.is_in_cash:
            self.is_in_cash = True
            self.emergency_exits += 1

            # Reduce stocks to top 5
            stock_scores = self._score_stocks()
            if stock_scores:
                ranked = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
                keep = set(t for t, _ in ranked[:self.stock_downtrend_n])
                for ticker in list(self.stock_holdings):
                    if ticker not in keep:
                        sym = self.stock_symbols[ticker]
                        if self.portfolio[sym].invested:
                            self.liquidate(sym)
                            self.total_trades += 1
                        self.stock_holdings.discard(ticker)

            # Reduce sector longs to 2, keep shorts (they profit in downtrend)
            sector_scores = self._score_sectors()
            if sector_scores:
                ranked_s = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
                keep_long = set(t for t, _ in ranked_s[:self.sector_downtrend_long_n])
                for ticker in list(self.sector_long_holdings):
                    if ticker not in keep_long:
                        sym = self.sector_symbols[ticker]
                        if self.portfolio[sym].invested:
                            self.liquidate(sym)
                            self.total_trades += 1
                        self.sector_long_holdings.discard(ticker)

            self.debug(
                f"EMERGENCY: stocks={len(self.stock_holdings)} "
                f"sec_long={len(self.sector_long_holdings)} sec_short={len(self.sector_short_holdings)}"
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
            f"EmergencyExits={self.emergency_exits} Events={self.total_events_detected}"
        )
        self.debug(
            f"REGIME: Up={self.months_in_uptrend} Down={self.months_in_downtrend}"
        )
        self.debug(
            f"HOLDINGS: stocks={len(self.stock_holdings)} "
            f"sec_long={len(self.sector_long_holdings)} sec_short={len(self.sector_short_holdings)}"
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
