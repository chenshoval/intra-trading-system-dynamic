"""Monthly Rotator v3 — Long-Short

Same scoring as v2 but adds a SHORT leg: short the bottom 5 stocks.
Based on Kelly/Xiu finding that long-short decile spread achieves
Sharpe 1.72 OOS (Lewellen 2015). The short leg reduces beta and
should generate profits during bear markets.

Long top 15 (~6.7% each) + Short bottom 5 (~2% each)
Net exposure: ~85% long - ~10% short = ~75% net
In downtrend: long top 5, short bottom 5 (net ~20%)
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV3LongShort(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        self.long_n = 15
        self.short_n = 5
        self.short_size_pct = 0.02          # 2% per short position
        self.downtrend_long_n = 5
        self.downtrend_short_n = 5
        self.trend_fast = 10
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
        self.long_holdings = set()
        self.short_holdings = set()
        self.last_rebalance = None
        self.is_in_cash = False
        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
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

        self.debug(f">>> ROTATOR v3 LONG-SHORT: long {self.long_n}, short {self.short_n}")

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

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
            raw_data[ticker] = {
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol, "events": event_count,
            }

        if len(raw_data) < self.long_n + self.short_n:
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

        scores = self._score_stocks()
        if not scores:
            self.event_counts_this_month.clear()
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        n_long = self.long_n if uptrend else self.downtrend_long_n
        n_short = self.short_n if uptrend else self.downtrend_short_n
        self.is_in_cash = False

        target_longs = set(t for t, _ in ranked[:n_long])
        target_shorts = set(t for t, _ in ranked[-n_short:])

        # Close positions no longer in targets
        for ticker in list(self.long_holdings):
            if ticker not in target_longs:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.long_holdings.discard(ticker)

        for ticker in list(self.short_holdings):
            if ticker not in target_shorts:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.short_holdings.discard(ticker)

        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            self.event_counts_this_month.clear()
            return

        # Long positions — equal weight
        long_alloc = total_value / n_long if n_long > 0 else 0
        for ticker in target_longs:
            symbol = self.symbols[ticker]
            price = self.securities[symbol].price
            if price <= 0:
                continue
            target_qty = int(long_alloc / price)
            if target_qty < 1:
                continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1
            self.long_holdings.add(ticker)

        # Short positions — fixed % per position
        short_alloc = total_value * self.short_size_pct
        for ticker in target_shorts:
            symbol = self.symbols[ticker]
            price = self.securities[symbol].price
            if price <= 0:
                continue
            target_qty = -int(short_alloc / price)  # negative = short
            if target_qty >= 0:
                continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1
            self.short_holdings.add(ticker)

        regime = "UP" if uptrend else "DOWN"
        self.debug(
            f"REBALANCE [{regime}]: long={n_long} short={n_short}, eq=${total_value:,.0f}"
        )
        self.event_counts_this_month.clear()

    def midweek_trend_check(self):
        if not self.long_holdings and not self.short_holdings:
            return
        uptrend = self._is_uptrend()
        if not uptrend and not self.is_in_cash:
            self.is_in_cash = True
            self.emergency_exits += 1

            scores = self._score_stocks()
            if not scores:
                return

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            keep_long = set(t for t, _ in ranked[:self.downtrend_long_n])
            keep_short = set(t for t, _ in ranked[-self.downtrend_short_n:])

            for ticker in list(self.long_holdings):
                if ticker not in keep_long:
                    symbol = self.symbols[ticker]
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1
                    self.long_holdings.discard(ticker)

            # Keep shorts as-is during downtrend (they're making money)

            self.debug(f"EMERGENCY: downtrend, longs reduced to {len(self.long_holdings)}, shorts kept at {len(self.short_holdings)}")

        elif uptrend and self.is_in_cash:
            self.is_in_cash = False
            self.debug(f"RECOVERY: uptrend restored")

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
        self.debug(f"REGIME: Up={self.months_in_uptrend} Down={self.months_in_downtrend}")

        if self.monthly_returns:
            rets = np.array(self.monthly_returns)
            win_months = np.sum(rets > 0)
            loss_months = np.sum(rets <= 0)
            avg_ret = np.mean(rets)
            std = np.std(rets)
            monthly_sharpe = avg_ret / std if std > 0 else 0
            self.debug(
                f"MONTHLY: Avg={avg_ret:.2%} Best={np.max(rets):.2%} Worst={np.min(rets):.2%} "
                f"Win={win_months} Loss={loss_months} WR={win_months/len(rets):.0%} Sharpe={monthly_sharpe:.2f}"
            )
