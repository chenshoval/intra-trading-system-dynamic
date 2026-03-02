"""Monthly Rotator v8 — Fundamentals-Enhanced Scoring

Same architecture as v2 (the deployed winner) but adds 3 fundamental
signals from the academic literature:

New signals (from Kelly/Xiu, Lewellen, Harvey/Man Group):
  - Value: earnings yield (E/P ratio) — cheap stocks rank higher
  - Quality: return on equity (ROE) — profitable companies rank higher
  - Quality: debt-to-equity ratio — low leverage ranks higher

Lewellen (2015) used 15 variables including E/P and ROE to achieve
Sharpe 1.72 on long-short decile portfolios. Harvey/Man Group showed
quality factor returned +3% real during inflation with low correlation
to momentum. Adding these diversifies the scoring beyond pure
momentum/trend which can fail in regime changes.

Signal weights (8 signals, sum to 1.0):
  momentum  0.25  (was 0.35)
  trend     0.15  (was 0.25)
  recent    0.15  (was 0.20)
  vol       0.10  (same)
  events    0.05  (was 0.10)
  value     0.10  (NEW: earnings yield)
  quality   0.10  (NEW: ROE)
  leverage  0.10  (NEW: low debt-to-equity)
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV8Fundamentals(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.top_n = 15
        self.downtrend_top_n = 5
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Signal weights (8 signals, sum to 1.0) ──
        self.w_momentum = 0.25
        self.w_trend = 0.15
        self.w_recent_strength = 0.15
        self.w_volatility = 0.10
        self.w_events = 0.05
        self.w_value = 0.10                 # NEW: earnings yield (E/P)
        self.w_quality = 0.10               # NEW: return on equity
        self.w_leverage = 0.10              # NEW: low debt-to-equity

        # ── Momentum parameters ──
        self.mom_lookback = 126
        self.mom_skip = 21
        self.stock_ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

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

        # ── Event patterns ──
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

        # ── Data structures ──
        self.symbols = {}
        self.news_symbols = {}
        self.current_holdings = set()
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
        self.fundamentals_available = 0
        self.fundamentals_missing = 0

        # ── Add equities + news ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        # ── SPY for trend gate ──
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
            f">>> MONTHLY ROTATOR v8 (fundamentals): {len(self.target_tickers)} stocks, "
            f"8 signals, value={self.w_value} quality={self.w_quality} leverage={self.w_leverage}"
        )

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Event collection
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Fundamental data extraction
    # ══════════════════════════════════════════════════════════════════════

    def _get_fundamentals(self, ticker):
        """Extract fundamental data from QC's Fundamentals property."""
        symbol = self.symbols[ticker]
        security = self.securities[symbol]

        if not security.fundamentals:
            return None

        f = security.fundamentals

        try:
            # Earnings yield (E/P) — inverse of P/E, higher = cheaper
            pe_ratio = f.valuation_ratios.pe_ratio
            earnings_yield = 1.0 / pe_ratio if pe_ratio and pe_ratio > 0 else 0.0

            # Return on equity — higher = more profitable
            roe = f.operation_ratios.roe.value if f.operation_ratios.roe else 0.0

            # Debt-to-equity — lower = less leveraged (we'll invert for ranking)
            de_ratio = f.operation_ratios.total_debt_equity_ratio.value if f.operation_ratios.total_debt_equity_ratio else 1.0

            self.fundamentals_available += 1
            return {
                "earnings_yield": earnings_yield,
                "roe": roe,
                "de_ratio": de_ratio,
            }
        except Exception:
            self.fundamentals_missing += 1
            return None

    # ══════════════════════════════════════════════════════════════════════
    # Stock scoring (8 signals)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks(self):
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

            # Signal 1: Momentum
            price_6m = closes[0]
            price_1m = closes[-self.mom_skip]
            momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0

            # Signal 2: Trend
            if len(closes) >= self.stock_ma_period:
                ma_50 = np.mean(closes[-self.stock_ma_period:])
                trend_score = 1.0 if price_now > ma_50 else 0.0
            else:
                trend_score = 0.5

            # Signal 3: Recent strength
            if len(closes) >= self.recent_period:
                price_1m_ago = closes[-self.recent_period]
                recent = (price_now / price_1m_ago) - 1.0 if price_1m_ago > 0 else 0.0
            else:
                recent = 0.0

            # Signal 4: Volatility
            if len(closes) >= self.vol_period:
                returns = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1.0
            else:
                vol = 1.0

            # Signal 5: Events
            event_count = self.event_counts_this_month.get(ticker, 0)

            # Signals 6-8: Fundamentals
            fund = self._get_fundamentals(ticker)
            earnings_yield = fund["earnings_yield"] if fund else 0.0
            roe = fund["roe"] if fund else 0.0
            de_ratio = fund["de_ratio"] if fund else 1.0

            raw_data[ticker] = {
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol, "events": event_count,
                "earnings_yield": earnings_yield, "roe": roe, "de_ratio": de_ratio,
            }

        if len(raw_data) < self.top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        # Rank-normalize continuous signals (higher = better rank)
        for signal in ["momentum", "recent", "earnings_yield", "roe"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        # Volatility: lower = better
        ranked_vol = sorted(tickers, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        # Debt-to-equity: lower = better (rank same as vol — inverted)
        ranked_de = sorted(tickers, key=lambda t: raw_data[t]["de_ratio"], reverse=True)
        for i, t in enumerate(ranked_de):
            raw_data[t]["de_ratio_rank"] = i / (n - 1) if n > 1 else 0.5

        # Events: more = better
        ranked_events = sorted(tickers, key=lambda t: raw_data[t]["events"])
        for i, t in enumerate(ranked_events):
            raw_data[t]["events_rank"] = i / (n - 1) if n > 1 else 0.5

        # Composite score — 8 signals
        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.w_momentum * d.get("momentum_rank", 0.5)
                + self.w_trend * d["trend"]
                + self.w_recent_strength * d.get("recent_rank", 0.5)
                + self.w_volatility * d.get("vol_rank", 0.5)
                + self.w_events * d.get("events_rank", 0.5)
                + self.w_value * d.get("earnings_yield_rank", 0.5)
                + self.w_quality * d.get("roe_rank", 0.5)
                + self.w_leverage * d.get("de_ratio_rank", 0.5)
            )
            scores[ticker] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance (same as v2)
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

        scores = self._score_stocks()
        if not scores:
            self.event_counts_this_month.clear()
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

        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0 or n_hold <= 0:
            self.event_counts_this_month.clear()
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
        top3 = [(t, f"{s:.3f}") for t, s in ranked[:3]]
        self.debug(
            f"REBALANCE [{regime}]: {n_hold} stocks, "
            f"fund_avail={self.fundamentals_available} fund_miss={self.fundamentals_missing}, "
            f"top3={top3}, eq=${total_value:,.0f}"
        )
        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check (same as v2)
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
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

            self.debug(f"EMERGENCY: downtrend, reduced to {len(self.current_holdings)} positions")

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
            f"EmergencyExits={self.emergency_exits} Events={self.total_events_detected} "
            f"FundAvail={self.fundamentals_available} FundMiss={self.fundamentals_missing}"
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
