"""Monthly Rotator v13 — Regime-Aware Sector Rotation

The problem: v2 has 27-36% max drawdown in bear markets.
We tried 5 fixes (v3-v7): shorting, SPY hedge, sector long-short,
capital splits — all failed or sacrificed too much bull performance.

The fix: DON'T short, DON'T split capital — SWITCH what you hold.
- Bull regime → run v2 exactly (50 stocks, 5 signals, top 15)
- Bear regime → rotate into defensive sector ETFs (no shorts)

Research basis: RegimeFolio (Zhang 2025) — regime-aware sector rotation
achieved Sharpe 1.17 with 12% lower max DD on 2020-2024.

Our own data validates this:
- v2 = bull champion (43% CAR 2016-2020)
- v5 = bear champion (12% CAR, Sharpe 0.56 in 2022-2023)
- v13 = v2 in bull + defensive rotation in bear

Regime detection (dual gate):
- Bear if SPY MA(10) < MA(50)  OR  VIX > 25
- Bull if SPY MA(10) > MA(50)  AND  VIX <= 25
- Conservative: catches slow declines (MA) AND sudden crashes (VIX)

Defensive ETF universe (10 ETFs):
- Classic defensives: XLU, XLP, XLV
- Safe havens: TLT, GLD
- Commodity/crisis beneficiaries: XLE, MOO, DBA
- Moderate defensives: XLRE, XLI

The scoring engine adapts to each crisis type automatically:
- Commodity crisis (2022): XLE, MOO, DBA, GLD score highest
- COVID crash (2020): TLT, GLD, XLV score highest
- Rate hike bear: XLU, XLP score highest
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV13Regime(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ══════════════════════════════════════════════════════════════
        # Regime detection parameters
        # ══════════════════════════════════════════════════════════════
        self.trend_fast = 10
        self.trend_slow = 50
        self.vix_bear_threshold = 25
        self.spy_vol_period = 21          # fallback: realized vol proxy for VIX

        # ══════════════════════════════════════════════════════════════
        # Bull mode parameters (= v2 exactly)
        # ══════════════════════════════════════════════════════════════
        self.bull_top_n = 15
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

        # ══════════════════════════════════════════════════════════════
        # Bear mode parameters (defensive rotation)
        # ══════════════════════════════════════════════════════════════
        self.bear_top_n = 4
        self.bear_w_momentum = 0.40
        self.bear_w_trend = 0.30
        self.bear_w_volatility = 0.20
        self.bear_w_recent = 0.10

        # ══════════════════════════════════════════════════════════════
        # 50-stock universe (bull mode — same as v2)
        # ══════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════
        # Defensive ETF universe (bear mode)
        # ══════════════════════════════════════════════════════════════
        self.defensive_tickers = [
            # Classic defensives
            "XLU",    # Utilities
            "XLP",    # Consumer Staples
            "XLV",    # Healthcare
            # Safe havens
            "TLT",    # 20+ Year Treasury Bonds
            "GLD",    # Gold
            # Commodity/geopolitical crisis beneficiaries
            "XLE",    # Energy (2022: +60% while SPY -19%)
            "MOO",    # Agribusiness/Fertilizer (Nutrien, Mosaic, Deere)
            "DBA",    # Agriculture Commodities (wheat, corn, soybeans futures)
            # Moderate defensives
            "XLRE",   # Real Estate
            "XLI",    # Industrials (defense spending in geopolitical crises)
        ]

        # ── Event patterns (bull mode — same as v2) ──
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
        self.symbols = {}
        self.defensive_symbols = {}
        self.news_symbols = {}
        self.current_holdings = set()
        self.current_regime = "BULL"      # start assuming bull
        self.last_rebalance = None

        # Event tracking (bull mode)
        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        # ── Tracking ──
        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.regime_switches = 0
        self.months_in_bull = 0
        self.months_in_bear = 0
        self.bull_returns = []
        self.bear_returns = []
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)

        # ══════════════════════════════════════════════════════════════
        # Add equities
        # ══════════════════════════════════════════════════════════════

        # Bull mode: 50 stocks + Tiingo news
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol
            news = self.add_data(TiingoNews, ticker)
            self.news_symbols[ticker] = news.symbol

        # Bear mode: defensive ETFs
        for ticker in self.defensive_tickers:
            if ticker not in self.symbols:  # avoid duplicates (XLE is in both)
                equity = self.add_equity(ticker, Resolution.DAILY)
                self.defensive_symbols[ticker] = equity.symbol
            else:
                self.defensive_symbols[ticker] = self.symbols[ticker]

        # SPY for trend gate
        self.add_equity("SPY", Resolution.DAILY)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

        # VIX for volatility regime gate
        # QC provides CBOE VIX data
        self.vix_symbol = self.add_data(CBOE, "VIX").symbol

        self.set_benchmark("SPY")

        # ── Schedule ──
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
            f">>> ROTATOR v13 REGIME: {len(self.target_tickers)} stocks (bull) + "
            f"{len(self.defensive_tickers)} ETFs (bear), "
            f"VIX threshold={self.vix_bear_threshold}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Regime detection (dual gate)
    # ══════════════════════════════════════════════════════════════════════

    def _detect_regime(self):
        """Detect bull or bear regime using SPY MA + VIX.
        Bear if EITHER: SPY MA(10) < MA(50) OR VIX > 25.
        Bull requires BOTH: SPY MA(10) > MA(50) AND VIX <= 25.
        """
        # SPY trend gate
        if self.spy_fast_ma.is_ready and self.spy_slow_ma.is_ready:
            spy_downtrend = self.spy_fast_ma.current.value < self.spy_slow_ma.current.value
        else:
            spy_downtrend = False

        # VIX gate
        vix_high = False
        try:
            vix_history = self.history(self.vix_symbol, 1, Resolution.DAILY)
            if vix_history is not None and not vix_history.empty:
                vix_close = float(vix_history["close"].iloc[-1])
                vix_high = vix_close > self.vix_bear_threshold
        except Exception:
            # Fallback: compute realized SPY vol as VIX proxy
            vix_high = self._spy_vol_proxy_high()

        # Bear if EITHER trigger fires
        if spy_downtrend or vix_high:
            return "BEAR"
        return "BULL"

    def _spy_vol_proxy_high(self):
        """Fallback if VIX data unavailable: use SPY realized vol."""
        spy_history = self.history(
            self.symbol("SPY"), self.spy_vol_period + 1, Resolution.DAILY
        )
        if spy_history is None or spy_history.empty:
            return False
        try:
            closes = spy_history["close"].values
        except Exception:
            return False
        if len(closes) < self.spy_vol_period:
            return False
        returns = np.diff(closes) / closes[:-1]
        realized_vol = np.std(returns) * np.sqrt(252)
        # Realized vol > 25% annualized ≈ VIX > 25
        return realized_vol > 0.25

    # ══════════════════════════════════════════════════════════════════════
    # Event collection (bull mode — same as v2)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        # Deploy rebalance
        if not self._initial_rebalance_done:
            self._initial_rebalance_done = True
            invested = sum(
                1 for t in self.target_tickers
                if self.portfolio[self.symbols[t]].invested
            )
            if invested == 0:
                self.debug(f">>> DEPLOY REBALANCE: equity=${self.portfolio.total_portfolio_value:,.0f}")
                self.monthly_rebalance()
            return

        # Only collect events in bull mode
        if self.current_regime != "BULL":
            return

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
    # Bull mode scoring (= v2 exactly)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks_bull(self):
        """Score 50 stocks using v2's 5-signal engine."""
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

        if len(raw_data) < self.bull_top_n:
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

    # ══════════════════════════════════════════════════════════════════════
    # Bear mode scoring (defensive ETFs)
    # ══════════════════════════════════════════════════════════════════════

    def _score_defensive_etfs(self):
        """Score defensive ETFs using 4 signals: momentum, trend, vol, recent."""
        scores = {}
        raw_data = {}

        for ticker in self.defensive_tickers:
            symbol = self.defensive_symbols[ticker]
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

            # Momentum (6-month, skip last month)
            price_6m = closes[0]
            price_1m = closes[-self.mom_skip]
            momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0

            # Trend (above 50d MA)
            if len(closes) >= self.stock_ma_period:
                ma_50 = np.mean(closes[-self.stock_ma_period:])
                trend_score = 1.0 if price_now > ma_50 else 0.0
            else:
                trend_score = 0.5

            # Volatility (lower = better)
            if len(closes) >= self.vol_period:
                returns = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1.0
            else:
                vol = 1.0

            # Recent strength (21-day)
            if len(closes) >= self.recent_period:
                price_1m_ago = closes[-self.recent_period]
                recent = (price_now / price_1m_ago) - 1.0 if price_1m_ago > 0 else 0.0
            else:
                recent = 0.0

            raw_data[ticker] = {
                "momentum": momentum, "trend": trend_score,
                "vol": vol, "recent": recent,
            }

        if len(raw_data) < self.bear_top_n:
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
                self.bear_w_momentum * d.get("momentum_rank", 0.5)
                + self.bear_w_trend * d["trend"]
                + self.bear_w_volatility * d.get("vol_rank", 0.5)
                + self.bear_w_recent * d.get("recent_rank", 0.5)
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
            if self.current_regime == "BULL":
                self.bull_returns.append(month_ret)
            else:
                self.bear_returns.append(month_ret)
            key = f"{self.last_rebalance.year}-{self.last_rebalance.month:02d}"
            self.monthly_pnl[key] = current_equity - self.month_start_equity

        self.month_start_equity = current_equity
        self.last_rebalance = self.time

        # Detect regime
        new_regime = self._detect_regime()
        if new_regime != self.current_regime:
            old_regime = self.current_regime
            self.current_regime = new_regime
            self.regime_switches += 1
            self.debug(
                f"REGIME SWITCH: {old_regime} → {new_regime} "
                f"(switch #{self.regime_switches}), eq=${current_equity:,.0f}"
            )

        if self.current_regime == "BULL":
            self.months_in_bull += 1
            self._rebalance_bull()
        else:
            self.months_in_bear += 1
            self._rebalance_bear()

        # Reset event counts
        self.event_counts_this_month.clear()

    def _rebalance_bull(self):
        """Bull mode: score 50 stocks, hold top 15 (= v2)."""
        scores = self._score_stocks_bull()
        if not scores:
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = set(t for t, _ in ranked[:self.bull_top_n])

        # Liquidate anything not in target (including defensive ETFs from bear mode)
        self._liquidate_all_except(target_tickers, self.symbols)

        # Also liquidate any remaining defensive ETF positions
        for ticker in self.defensive_tickers:
            symbol = self.defensive_symbols[ticker]
            if self.portfolio[symbol].invested and ticker not in target_tickers:
                self.liquidate(symbol)
                self.total_trades += 1

        # Buy new positions (equal weight)
        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        target_alloc = total_value / self.bull_top_n

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

        events_this_month = sum(self.event_counts_this_month.values())
        top3 = [(t, f"{s:.3f}") for t, s in ranked[:3]]
        self.debug(
            f"REBALANCE [BULL]: {self.bull_top_n} stocks, events={events_this_month}, "
            f"top3={top3}, eq=${total_value:,.0f}"
        )

    def _rebalance_bear(self):
        """Bear mode: score defensive ETFs, hold top 4."""
        scores = self._score_defensive_etfs()
        if not scores:
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = set(t for t, _ in ranked[:self.bear_top_n])

        # Liquidate all stock positions from bull mode
        for ticker in list(self.current_holdings):
            if ticker in self.symbols:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
            self.current_holdings.discard(ticker)

        # Liquidate defensive ETFs not in target
        for ticker in self.defensive_tickers:
            if ticker not in target_tickers:
                symbol = self.defensive_symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1

        # Buy target defensive ETFs (equal weight)
        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        target_alloc = total_value / self.bear_top_n

        for ticker in target_tickers:
            symbol = self.defensive_symbols[ticker]
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

        top_all = [(t, f"{s:.3f}") for t, s in ranked[:5]]
        self.debug(
            f"REBALANCE [BEAR]: {self.bear_top_n} defensive ETFs, "
            f"selected={list(target_tickers)}, all_scores={top_all}, "
            f"eq=${total_value:,.0f}"
        )

    def _liquidate_all_except(self, keep_tickers, symbol_map):
        """Liquidate all positions not in keep_tickers."""
        for ticker in list(self.current_holdings):
            if ticker not in keep_tickers:
                if ticker in symbol_map:
                    symbol = symbol_map[ticker]
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.total_trades += 1
                self.current_holdings.discard(ticker)

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check
    # ══════════════════════════════════════════════════════════════════════

    def midweek_trend_check(self):
        """Wednesday check: only emergency bull→bear switch.
        Bear→bull transitions wait for monthly rebalance to prevent whipsaw."""
        if not self.current_holdings:
            return

        new_regime = self._detect_regime()

        # Emergency: bull → bear (rotate to defensives immediately)
        if self.current_regime == "BULL" and new_regime == "BEAR":
            self.current_regime = "BEAR"
            self.regime_switches += 1
            self.emergency_exits += 1
            self.debug(
                f"EMERGENCY REGIME SWITCH: BULL → BEAR (Wednesday), "
                f"rotating to defensive ETFs"
            )
            self._rebalance_bear()

        # Bear → bull: do NOT switch mid-week (wait for monthly)
        # This prevents whipsaw on temporary VIX dips

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        current_equity = self.portfolio.total_portfolio_value
        if self.last_rebalance is not None:
            month_ret = (current_equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)
            if self.current_regime == "BULL":
                self.bull_returns.append(month_ret)
            else:
                self.bear_returns.append(month_ret)

        ret_pct = self.portfolio.total_profit / 100_000

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} Final=${current_equity:,.0f} "
            f"Rebalances={self.total_rebalances} Trades={self.total_trades} "
            f"EmergencyExits={self.emergency_exits} "
            f"RegimeSwitches={self.regime_switches} "
            f"Events={self.total_events_detected}"
        )
        self.debug(
            f"REGIME: Bull={self.months_in_bull}mo Bear={self.months_in_bear}mo "
            f"Switches={self.regime_switches}"
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

        if self.bull_returns:
            bull_rets = np.array(self.bull_returns)
            self.debug(
                f"BULL MONTHS: Avg={np.mean(bull_rets):.2%} "
                f"Best={np.max(bull_rets):.2%} Worst={np.min(bull_rets):.2%} "
                f"Count={len(bull_rets)}"
            )

        if self.bear_returns:
            bear_rets = np.array(self.bear_returns)
            self.debug(
                f"BEAR MONTHS: Avg={np.mean(bear_rets):.2%} "
                f"Best={np.max(bear_rets):.2%} Worst={np.min(bear_rets):.2%} "
                f"Count={len(bear_rets)}"
            )

        if self.monthly_pnl:
            pnl_str = " | ".join(
                f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items())
            )
            self.debug(f"PNL: {pnl_str}")
