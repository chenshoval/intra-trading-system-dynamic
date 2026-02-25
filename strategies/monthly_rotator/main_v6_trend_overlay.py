"""Monthly Rotator v6 — Trend Overlay (v2 + SPY Trend as 2nd Stream)

Two uncorrelated streams in one algorithm:
- 70% capital: v2 stock rotation (monthly, event-boosted scoring)
- 30% capital: SPY trend following (long above 200d MA, short below)

Based on Harvey/Man Group finding that trend-following is the most
robust strategy across ALL regimes (95 years, 100% hit rate on
bonds+commodities). The SPY trend stream should make money in bear
markets by shorting SPY, offsetting stock rotation losses.

Warsaw paper principle: two uncorrelated strategies combined >
either alone on risk-adjusted basis.
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV6TrendOverlay(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Capital split ──
        self.stock_pct = 0.70               # 70% to stock rotation
        self.trend_pct = 0.30               # 30% to SPY trend

        # ── Stock rotation parameters (same as v2) ──
        self.top_n = 15
        self.downtrend_top_n = 5
        self.trend_fast = 10                # for stock regime gate
        self.trend_slow = 50

        self.w_momentum = 0.35
        self.w_trend = 0.25
        self.w_recent_strength = 0.20
        self.w_volatility = 0.10
        self.w_events = 0.10

        self.mom_lookback = 126
        self.mom_skip = 21
        self.stock_ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        # ── SPY trend parameters ──
        self.spy_trend_ma_period = 200      # 200-day MA for trend signal

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

        self.symbols = {}
        self.news_symbols = {}
        self.current_holdings = set()
        self.last_rebalance = None
        self.is_in_cash = False
        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0
        self.spy_trend_position = 0         # 1=long, -1=short, 0=flat

        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.trend_flips = 0
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)

        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy_symbol = spy.symbol
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)
        self.spy_200_ma = self.sma("SPY", self.spy_trend_ma_period, Resolution.DAILY)

        self.set_benchmark("SPY")

        # Monthly rebalance for stocks
        self.schedule.on(
            self.date_rules.month_start("SPY", 0),
            self.time_rules.after_market_open("SPY", 30),
            self.monthly_rebalance,
        )
        # Weekly trend check for stocks + SPY trend update
        self.schedule.on(
            self.date_rules.every(DayOfWeek.WEDNESDAY),
            self.time_rules.after_market_open("SPY", 60),
            self.weekly_update,
        )

        self.debug(f">>> ROTATOR v6 TREND-OVERLAY: stock={self.stock_pct:.0%} trend={self.trend_pct:.0%}")

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    def _spy_above_200ma(self):
        if not self.spy_200_ma.is_ready:
            return True
        return self.securities[self.spy_symbol].price > self.spy_200_ma.current.value

    def on_data(self, data):
        for ticker in self.target_tickers:
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

    def _score_stocks(self):
        scores = {}
        raw_data = {}
        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
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
            event_count = self.event_counts_this_month.get(ticker, 0)
            raw_data[ticker] = {"momentum": momentum, "trend": trend_score, "recent": recent, "vol": vol, "events": event_count}

        if len(raw_data) < self.top_n:
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
                self.w_momentum * d.get("momentum_rank", 0.5)
                + self.w_trend * d["trend"]
                + self.w_recent_strength * d.get("recent_rank", 0.5)
                + self.w_volatility * d.get("vol_rank", 0.5)
                + self.w_events * d.get("events_rank", 0.5)
            )
            scores[ticker] = score
        return scores

    def _update_spy_trend(self):
        """Manage the SPY trend-following position (30% of capital)."""
        above_200 = self._spy_above_200ma()
        total_value = self.portfolio.total_portfolio_value
        trend_capital = total_value * self.trend_pct
        spy_price = self.securities[self.spy_symbol].price

        if spy_price <= 0 or trend_capital <= 0:
            return

        target_direction = 1 if above_200 else -1

        if target_direction != self.spy_trend_position:
            # Flip position
            target_qty = int(trend_capital / spy_price) * target_direction
            current_qty = int(self.portfolio[self.spy_symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(self.spy_symbol, delta)
                self.total_trades += 1
                self.trend_flips += 1
            self.spy_trend_position = target_direction
            direction_str = "LONG" if target_direction > 0 else "SHORT"
            self.debug(f"SPY TREND: {direction_str} {abs(target_qty)} shares")

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

        # Stock rotation (70% of capital)
        scores = self._score_stocks()
        if not scores:
            self.event_counts_this_month.clear()
            self._update_spy_trend()
            return

        n_hold = self.top_n if uptrend else self.downtrend_top_n
        self.is_in_cash = False
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = set(t for t, _ in ranked[:n_hold])

        for ticker in list(self.current_holdings):
            if ticker not in target_tickers:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.current_holdings.discard(ticker)

        stock_capital = self.portfolio.total_portfolio_value * self.stock_pct
        if stock_capital > 0 and n_hold > 0:
            target_alloc = stock_capital / n_hold
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

        # SPY trend (30% of capital)
        self._update_spy_trend()

        regime = "UP" if uptrend else "DOWN"
        spy_dir = "LONG" if self.spy_trend_position > 0 else "SHORT" if self.spy_trend_position < 0 else "FLAT"
        self.debug(f"REBALANCE [{regime}]: {n_hold} stocks, SPY={spy_dir}, eq=${current_equity:,.0f}")
        self.event_counts_this_month.clear()

    def weekly_update(self):
        """Weekly: check stock trend gate + update SPY trend position."""
        self._update_spy_trend()

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
                keep = set(t for t, _ in ranked[:self.downtrend_top_n])
            for ticker in list(self.current_holdings):
                if ticker not in keep:
                    symbol = self.symbols[ticker]
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.current_holdings.discard(ticker)
            self.debug(f"EMERGENCY: downtrend, stocks={len(self.current_holdings)}")
        elif uptrend and self.is_in_cash:
            self.is_in_cash = False

    def on_end_of_algorithm(self):
        current_equity = self.portfolio.total_portfolio_value
        if self.last_rebalance is not None:
            month_ret = (current_equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)
        ret_pct = self.portfolio.total_profit / 100_000
        spy_dir = "LONG" if self.spy_trend_position > 0 else "SHORT" if self.spy_trend_position < 0 else "FLAT"
        self.debug(
            f"RESULTS: Return={ret_pct:.2%} Final=${current_equity:,.0f} "
            f"Trades={self.total_trades} TrendFlips={self.trend_flips} "
            f"EmergExits={self.emergency_exits} FinalSPY={spy_dir}"
        )
        if self.monthly_returns:
            rets = np.array(self.monthly_returns)
            win_months = np.sum(rets > 0)
            loss_months = np.sum(rets <= 0)
            avg_ret = np.mean(rets)
            std = np.std(rets)
            self.debug(
                f"MONTHLY: Avg={avg_ret:.2%} Best={np.max(rets):.2%} Worst={np.min(rets):.2%} "
                f"Win={win_months} Loss={loss_months} WR={win_months/len(rets):.0%} Sharpe={avg_ret/std if std > 0 else 0:.2f}"
            )
