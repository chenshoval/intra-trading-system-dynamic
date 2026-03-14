"""Monthly Rotator v2 + Chart Signals — A/B Test

Exact copy of v2 ($100K, 15 stocks, same deploy logic, same rebalance logic)
but with 3 additional chart-reading signals in the scoring model.

Purpose: isolate whether volume, swing trend, and breakout signals improve
v2's stock selection at $100K. Compare directly to v2 results in
results_from_quant_connect/MonthlyRotatorV2/.

Only differences from v2:
  - 8 signals instead of 5 (3 new chart signals added)
  - Signal weights redistributed to accommodate new signals
  - _find_swing_highs, _find_swing_lows, _classify_swing_trend,
    _calc_volume_score, _calc_breakout_score methods added
  - history() also extracts high, low, volume data

Everything else — $100K, top 15, deploy logic, rebalance logic,
midweek check, universe, event keywords — is byte-for-byte v2.

v2 weights:   mom 0.35, trend 0.25, recent 0.20, vol 0.10, events 0.10
v2+chart:     mom 0.25, trend 0.15, recent 0.15, vol 0.05, events 0.05,
              volume 0.15, swing 0.10, breakout 0.10
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV2ChartTest(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters — SAME AS v2 ──
        self.top_n = 15
        self.downtrend_top_n = 5
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Signal weights (8 signals, sum to 1.0) ──
        self.w_momentum = 0.25
        self.w_trend = 0.15
        self.w_recent_strength = 0.15
        self.w_volatility = 0.05
        self.w_events = 0.05
        self.w_volume = 0.15                   # NEW: volume confirmation
        self.w_swing_trend = 0.10              # NEW: HH/HL/LH/LL structure
        self.w_breakout = 0.10                 # NEW: breakout detection

        # ── Momentum parameters — SAME AS v2 ──
        self.mom_lookback = 126
        self.mom_skip = 21
        self.stock_ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        # ── Chart-reading parameters (NEW) ──
        self.swing_lookback = 5
        self.volume_avg_period = 50
        self.volume_recent_period = 5
        self.breakout_lookback = 60
        self.breakout_skip = 5

        # ── 50-stock universe — SAME AS v2 ──
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

        # ── Event patterns — SAME AS v2 ──
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

        # ── Data structures — SAME AS v2 ──
        self.symbols = {}
        self.news_symbols = {}
        self.current_holdings = set()
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

        # ── Chart signal counters (NEW) ──
        self.total_swing_detections = 0
        self.total_breakout_signals = 0

        # ── Add equities + news — SAME AS v2 ──
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

        self._initial_rebalance_done = False

        self.debug(
            f">>> MONTHLY ROTATOR v2+CHART TEST: {len(self.target_tickers)} stocks, "
            f"top {self.top_n}, $100K, 8 signals "
            f"[mom={self.w_momentum} trend={self.w_trend} recent={self.w_recent_strength} "
            f"vol={self.w_volatility} events={self.w_events} "
            f"volume={self.w_volume} swing={self.w_swing_trend} breakout={self.w_breakout}]"
        )

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Chart-reading helper methods (NEW — from v12)
    # ══════════════════════════════════════════════════════════════════════

    def _find_swing_highs(self, highs, n=None):
        if n is None:
            n = self.swing_lookback
        swings = []
        for i in range(n, len(highs) - n):
            is_highest = True
            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_highest = False
                    break
            if is_highest:
                swings.append((i, highs[i]))
        return swings

    def _find_swing_lows(self, lows, n=None):
        if n is None:
            n = self.swing_lookback
        swings = []
        for i in range(n, len(lows) - n):
            is_lowest = True
            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_lowest = False
                    break
            if is_lowest:
                swings.append((i, lows[i]))
        return swings

    def _classify_swing_trend(self, highs, lows):
        sh = self._find_swing_highs(highs)
        sl = self._find_swing_lows(lows)

        if len(sh) < 2 and len(sl) < 2:
            return 0.5

        self.total_swing_detections += 1
        signals = 0
        bullish = 0

        if len(sh) >= 2:
            signals += 1
            if sh[-1][1] > sh[-2][1]:
                bullish += 1

        if len(sl) >= 2:
            signals += 1
            if sl[-1][1] > sl[-2][1]:
                bullish += 1

        if signals == 0:
            return 0.5

        if signals == 2:
            return bullish / 2.0
        else:
            return 0.75 if bullish == 1 else 0.25

    def _calc_volume_score(self, volumes):
        if len(volumes) < self.volume_avg_period:
            return 1.0

        avg_vol = np.mean(volumes[-self.volume_avg_period:])
        if avg_vol <= 0:
            return 1.0

        recent_vol = np.mean(volumes[-self.volume_recent_period:])
        volume_ratio = recent_vol / avg_vol

        if len(volumes) >= 20:
            vol_recent_10 = np.mean(volumes[-10:])
            vol_prior_10 = np.mean(volumes[-20:-10])
            vol_trend = vol_recent_10 / vol_prior_10 if vol_prior_10 > 0 else 1.0
        else:
            vol_trend = 1.0

        return (volume_ratio + vol_trend) / 2.0

    def _calc_breakout_score(self, closes, volumes):
        if len(closes) < self.breakout_lookback:
            return 0.0

        resistance = np.max(closes[-self.breakout_lookback:-self.breakout_skip])
        price_now = closes[-1]

        if resistance <= 0:
            return 0.0

        gap_pct = (resistance - price_now) / resistance
        if gap_pct <= 0:
            proximity = 1.0
        elif gap_pct <= 0.03:
            proximity = 1.0 - (gap_pct / 0.03)
        else:
            proximity = 0.0

        vol_confirm = 0.0
        if proximity > 0.5 and len(volumes) >= 30:
            recent_vol = np.mean(volumes[-3:])
            avg_vol = np.mean(volumes[-30:])
            if avg_vol > 0:
                vol_confirm = min(recent_vol / avg_vol, 2.0) / 2.0

        if proximity >= 0.9:
            self.total_breakout_signals += 1

        return proximity * 0.5 + vol_confirm * 0.5

    # ══════════════════════════════════════════════════════════════════════
    # Event collection — SAME AS v2
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        """Scan news and count events per stock. No trades here."""
        # v2's deploy logic — exact copy
        if not self.current_holdings and not self._initial_rebalance_done:
            self._initial_rebalance_done = True
            self.debug(f">>> DEPLOY REBALANCE: equity=${self.portfolio.total_portfolio_value:,.0f}")
            self.monthly_rebalance()
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
    # Stock scoring — v2 + 3 chart signals
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
                highs = history["high"].values
                lows = history["low"].values
                volumes = history["volume"].values
            except Exception:
                continue
            if len(closes) < self.mom_lookback + self.mom_skip:
                continue

            price_now = closes[-1]
            if price_now <= 0:
                continue

            # ── v2 signals (1-5) ──
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

            # ── NEW chart signals (6-8) ──
            volume_score = self._calc_volume_score(volumes)
            swing_trend = self._classify_swing_trend(highs, lows)
            breakout = self._calc_breakout_score(closes, volumes)

            raw_data[ticker] = {
                "momentum": momentum, "trend": trend_score,
                "recent": recent, "vol": vol, "events": event_count,
                "volume_score": volume_score, "swing_trend": swing_trend,
                "breakout": breakout,
            }

        if len(raw_data) < self.top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        for signal in ["momentum", "recent", "volume_score", "breakout"]:
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
                + self.w_volume * d.get("volume_score_rank", 0.5)
                + self.w_swing_trend * d["swing_trend"]
                + self.w_breakout * d.get("breakout_rank", 0.5)
            )
            scores[ticker] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance — SAME AS v2 (iterate target_tickers, no skip log)
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

        events_this_month = sum(self.event_counts_this_month.values())
        stocks_with_events = len([t for t, c in self.event_counts_this_month.items() if c > 0])

        swing_before = self.total_swing_detections
        breakout_before = self.total_breakout_signals

        scores = self._score_stocks()
        if not scores:
            return

        swings_detected = self.total_swing_detections - swing_before
        breakouts_detected = self.total_breakout_signals - breakout_before

        n_hold = self.top_n if uptrend else self.downtrend_top_n
        self.is_in_cash = False

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = set(t for t, _ in ranked[:n_hold])

        # Sell — SAME AS v2
        for ticker in list(self.current_holdings):
            if ticker not in target_tickers:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.current_holdings.discard(ticker)

        # Buy — SAME AS v2 (iterate target_tickers set, no ranked order)
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
            f"REBALANCE [{regime}]: {n_hold} stocks, events={events_this_month} "
            f"({stocks_with_events} stocks), swings={swings_detected} breakouts={breakouts_detected}, "
            f"top3={top3}, eq=${total_value:,.0f}"
        )

        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check — SAME AS v2
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
    # End of algorithm — SAME AS v2 + chart signal counts
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
            f"TotalEventsDetected={self.total_events_detected} "
            f"SwingDetections={self.total_swing_detections} "
            f"BreakoutSignals={self.total_breakout_signals}"
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
