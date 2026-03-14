"""Monthly Rotator v12 — Chart-Enhanced Scoring

Builds on v11 (medium account, 10 stocks) with 3 new chart-reading signals
learned from technical analysis study. Adds volume confirmation, swing trend
structure (HH/HL/LH/LL), and breakout detection to the scoring model.

8 signals total (sum to 1.0):
  momentum        0.25  — 6-month return, skip last month
  trend           0.15  — price above/below 50d MA
  recent          0.15  — 1-month return
  volatility      0.05  — annualized vol (lower = better)
  events          0.05  — news event count (earnings beats, upgrades)
  volume          0.15  — recent volume vs 50d average (NEW)
  swing_trend     0.10  — HH/HL/LH/LL structure analysis (NEW)
  breakout        0.10  — proximity to resistance + volume confirm (NEW)

Scaling ladder:
  v10: $500-$2K   → top 5   (main_v10_small.py)
  v11: $2K-$20K   → top 10  (main_v11_medium.py)
  v12: $2K-$20K   → top 10  (this file — chart-enhanced v11)
  v2:  $20K-$100K → top 15  (main_v2.py)

The 3 new signals address gaps identified in our chart reading analysis:
1. Volume: confirms whether price moves are backed by real buying/selling
2. Swing trend: captures trend STRUCTURE (not just a MA snapshot)
3. Breakout: catches stocks breaking through resistance with volume
"""

from AlgorithmImports import *
from QuantConnect.DataSource import *
from collections import defaultdict
import numpy as np


class MonthlyRotatorV12ChartEnhanced(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(5_00)

        # ── Portfolio parameters ──
        self.top_n = 10
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

        # ── Momentum parameters ──
        self.mom_lookback = 126
        self.mom_skip = 21
        self.stock_ma_period = 50
        self.recent_period = 21
        self.vol_period = 42

        # ── Chart-reading parameters ──
        self.swing_lookback = 5                # bars on each side for swing detection
        self.volume_avg_period = 50            # period for average volume
        self.volume_recent_period = 5          # recent volume window
        self.breakout_lookback = 60            # days to find resistance level
        self.breakout_skip = 5                 # skip last N days for resistance

        # ── 50-stock universe (same as v2/v11) ──
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

        # ── Event patterns (same keywords as v11) ──
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

        # ── Event tracking (reset each month) ──
        self.event_counts_this_month = defaultdict(int)
        self.total_events_detected = 0

        # ── Counters ──
        self.total_rebalances = 0
        self.total_trades = 0
        self.emergency_exits = 0
        self.months_in_uptrend = 0
        self.months_in_downtrend = 0
        self.monthly_returns = []

        # ── Per-month tracking ──
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)

        # ── Chart signal counters (for debugging) ──
        self.total_swing_detections = 0
        self.total_breakout_signals = 0

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

        # Track if initial deploy rebalance has been done
        self._initial_rebalance_done = False

        self.debug(
            f">>> MONTHLY ROTATOR v12 (chart-enhanced): {len(self.target_tickers)} stocks, "
            f"top {self.top_n}, 8 signals "
            f"[mom={self.w_momentum} trend={self.w_trend} recent={self.w_recent_strength} "
            f"vol={self.w_volatility} events={self.w_events} "
            f"volume={self.w_volume} swing={self.w_swing_trend} breakout={self.w_breakout}]"
        )

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # Chart-reading helper methods
    # ══════════════════════════════════════════════════════════════════════

    def _find_swing_highs(self, highs, n=None):
        """Find swing high points — local peaks in price data.

        A swing high at bar i means high[i] is greater than the N bars
        before AND after it. Think of it as a mountain top.

        Args:
            highs: array of high prices
            n: bars on each side to check (default: self.swing_lookback)

        Returns:
            list of (index, price) tuples
        """
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
        """Find swing low points — local valleys in price data.

        A swing low at bar i means low[i] is lower than the N bars
        before AND after it. Think of it as a valley bottom.

        Args:
            lows: array of low prices
            n: bars on each side to check (default: self.swing_lookback)

        Returns:
            list of (index, price) tuples
        """
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
        """Classify trend structure using swing highs and lows.

        Compares consecutive swing points to determine:
        - Higher Highs (HH) + Higher Lows (HL) = strong uptrend = 1.0
        - One of HH or HL = moderate uptrend = 0.75
        - Mixed signals = neutral = 0.5
        - One of LH or LL = moderate downtrend = 0.25
        - Lower Highs (LH) + Lower Lows (LL) = strong downtrend = 0.0

        Returns:
            float 0.0 to 1.0 representing trend quality
        """
        sh = self._find_swing_highs(highs)
        sl = self._find_swing_lows(lows)

        if len(sh) < 2 and len(sl) < 2:
            return 0.5  # not enough data

        self.total_swing_detections += 1

        score = 0.5  # start neutral
        signals = 0
        bullish = 0

        # Check last two swing highs
        if len(sh) >= 2:
            signals += 1
            if sh[-1][1] > sh[-2][1]:  # Higher High
                bullish += 1

        # Check last two swing lows
        if len(sl) >= 2:
            signals += 1
            if sl[-1][1] > sl[-2][1]:  # Higher Low
                bullish += 1

        if signals == 0:
            return 0.5

        # Map to 0.0-1.0 scale
        # 0 bullish out of 2 = 0.0 (LH + LL)
        # 1 bullish out of 2 = 0.5 (mixed)
        # 2 bullish out of 2 = 1.0 (HH + HL)
        if signals == 2:
            score = bullish / 2.0
        elif signals == 1:
            score = 0.75 if bullish == 1 else 0.25

        return score

    def _calc_volume_score(self, volumes):
        """Calculate volume confirmation score.

        Combines two measures:
        1. Volume ratio: recent 5-day avg vs 50-day avg
           High ratio = stock is getting attention = good
        2. Volume trend: last 10d avg vs prior 10d avg
           Rising volume = increasing conviction

        Returns:
            float — raw volume score (will be rank-normalized later)
        """
        if len(volumes) < self.volume_avg_period:
            return 1.0

        avg_vol = np.mean(volumes[-self.volume_avg_period:])
        if avg_vol <= 0:
            return 1.0

        recent_vol = np.mean(volumes[-self.volume_recent_period:])
        volume_ratio = recent_vol / avg_vol

        # Volume trend: are volumes increasing?
        if len(volumes) >= 20:
            vol_recent_10 = np.mean(volumes[-10:])
            vol_prior_10 = np.mean(volumes[-20:-10])
            vol_trend = vol_recent_10 / vol_prior_10 if vol_prior_10 > 0 else 1.0
        else:
            vol_trend = 1.0

        return (volume_ratio + vol_trend) / 2.0

    def _calc_breakout_score(self, closes, volumes):
        """Calculate breakout proximity and confirmation score.

        Identifies stocks near or breaking through recent resistance:
        1. Find resistance = highest close in last 60d (skip last 5d)
        2. Proximity = how close current price is to resistance
        3. Volume confirmation = is volume elevated at/near breakout?

        Returns:
            float 0.0 to 1.0 — breakout strength
        """
        if len(closes) < self.breakout_lookback:
            return 0.0

        # Find resistance level (highest close in lookback, skipping recent days)
        resistance = np.max(closes[-self.breakout_lookback:-self.breakout_skip])
        price_now = closes[-1]

        if resistance <= 0:
            return 0.0

        # Proximity: 0 if far below, 1 if at/above resistance
        # "within 3%" zone scales linearly
        gap_pct = (resistance - price_now) / resistance
        if gap_pct <= 0:
            # Already above resistance = breakout happened
            proximity = 1.0
        elif gap_pct <= 0.03:
            # Within 3% of resistance = approaching
            proximity = 1.0 - (gap_pct / 0.03)
        else:
            proximity = 0.0

        # Volume confirmation near breakout
        vol_confirm = 0.0
        if proximity > 0.5 and len(volumes) >= 30:
            recent_vol = np.mean(volumes[-3:])
            avg_vol = np.mean(volumes[-30:])
            if avg_vol > 0:
                # Cap at 2x average = score of 1.0
                vol_confirm = min(recent_vol / avg_vol, 2.0) / 2.0

        # Breakout with volume = strong signal
        if proximity >= 0.9:
            self.total_breakout_signals += 1

        return proximity * 0.5 + vol_confirm * 0.5

    # ══════════════════════════════════════════════════════════════════════
    # Event collection (no trades — just counting)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        """Scan news and count events per stock. No trades here."""
        if not self._initial_rebalance_done:
            self._initial_rebalance_done = True
            # Check IBKR's actual positions
            for ticker in self.target_tickers:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.current_holdings.add(ticker)
            invested = len(self.current_holdings)
            equity = self.portfolio.total_portfolio_value
            if invested == 0:
                self.debug(f">>> DEPLOY: No positions, equity=${equity:,.0f}, rebalancing")
                self.monthly_rebalance()
            else:
                self.debug(f">>> DEPLOY: Found {invested} positions, equity=${equity:,.0f}, synced")
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
    # Stock scoring (8 signals: 5 original + 3 chart-enhanced)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks(self):
        """Score stocks using 8 signals: momentum, trend, recent, vol,
        events, volume confirmation, swing trend, breakout detection."""
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

            # ── Signal 1: Momentum (6-month, skip last month) ──
            price_6m = closes[0]
            price_1m = closes[-self.mom_skip]
            momentum = (price_1m / price_6m) - 1.0 if price_6m > 0 and price_1m > 0 else 0.0

            # ── Signal 2: Trend (above 50d MA) ──
            if len(closes) >= self.stock_ma_period:
                ma_50 = np.mean(closes[-self.stock_ma_period:])
                trend_score = 1.0 if price_now > ma_50 else 0.0
            else:
                trend_score = 0.5

            # ── Signal 3: Recent strength (1-month return) ──
            if len(closes) >= self.recent_period:
                price_1m_ago = closes[-self.recent_period]
                recent = (price_now / price_1m_ago) - 1.0 if price_1m_ago > 0 else 0.0
            else:
                recent = 0.0

            # ── Signal 4: Volatility (lower = better) ──
            if len(closes) >= self.vol_period:
                returns = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1.0
            else:
                vol = 1.0

            # ── Signal 5: Event count this month ──
            event_count = self.event_counts_this_month.get(ticker, 0)

            # ── Signal 6: Volume confirmation (NEW) ──
            volume_score = self._calc_volume_score(volumes)

            # ── Signal 7: Swing trend structure (NEW) ──
            swing_trend = self._classify_swing_trend(highs, lows)

            # ── Signal 8: Breakout detection (NEW) ──
            breakout = self._calc_breakout_score(closes, volumes)

            raw_data[ticker] = {
                "momentum": momentum,
                "trend": trend_score,
                "recent": recent,
                "vol": vol,
                "events": event_count,
                "volume_score": volume_score,
                "swing_trend": swing_trend,
                "breakout": breakout,
            }

        if len(raw_data) < self.top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        # ── Rank-normalize continuous signals (higher raw = higher rank) ──
        for signal in ["momentum", "recent", "volume_score", "breakout"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        # Volatility: lower = better (reverse ranking)
        ranked_vol = sorted(tickers, key=lambda t: raw_data[t]["vol"], reverse=True)
        for i, t in enumerate(ranked_vol):
            raw_data[t]["vol_rank"] = i / (n - 1) if n > 1 else 0.5

        # Events: more = better
        ranked_events = sorted(tickers, key=lambda t: raw_data[t]["events"])
        for i, t in enumerate(ranked_events):
            raw_data[t]["events_rank"] = i / (n - 1) if n > 1 else 0.5

        # ── Composite score — 8 signals ──
        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.w_momentum * d.get("momentum_rank", 0.5)
                + self.w_trend * d["trend"]                        # already 0/1
                + self.w_recent_strength * d.get("recent_rank", 0.5)
                + self.w_volatility * d.get("vol_rank", 0.5)
                + self.w_events * d.get("events_rank", 0.5)
                + self.w_volume * d.get("volume_score_rank", 0.5)  # NEW
                + self.w_swing_trend * d["swing_trend"]            # already 0-1
                + self.w_breakout * d.get("breakout_rank", 0.5)    # NEW
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

        # Reset chart signal counters for this rebalance
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

        # Sell positions not in new top N
        for ticker in list(self.current_holdings):
            if ticker not in target_tickers:
                symbol = self.symbols[ticker]
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.total_trades += 1
                self.current_holdings.discard(ticker)

        # Buy new positions
        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0 or n_hold <= 0:
            self.event_counts_this_month.clear()
            return

        target_alloc = total_value / n_hold
        bought = 0
        skipped = []

        ranked_targets = [(t, s) for t, s in ranked[:n_hold]]

        for ticker, score in ranked_targets:
            symbol = self.symbols[ticker]
            price = self.securities[symbol].price
            if price <= 0:
                continue
            target_qty = int(target_alloc / price)
            if target_qty < 1:
                skipped.append(f"{ticker}(${price:.0f})")
                continue
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if abs(delta) > 0:
                self.market_order(symbol, delta)
                self.total_trades += 1
                bought += 1
            self.current_holdings.add(ticker)

        regime = "UP" if uptrend else "DOWN"
        top3 = [(t, f"{s:.3f}") for t, s in ranked[:3]]
        skipped_str = f", skipped=[{', '.join(skipped)}]" if skipped else ""
        self.debug(
            f"REBALANCE [{regime}]: {n_hold} targets, bought={bought}{skipped_str}, "
            f"events={events_this_month} ({stocks_with_events} stocks), "
            f"swings={swings_detected} breakouts={breakouts_detected}, "
            f"top3={top3}, eq=${total_value:,.0f}"
        )

        # Reset event counts for next month
        self.event_counts_this_month.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Mid-week trend check
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
