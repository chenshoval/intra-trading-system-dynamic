"""Overnight Hold v1b — Weekly Selective (High-Conviction Only)

FIXES from v1: v1 traded every single night (5,399 trades, $64K fees on $100K).
The overnight edge is real but tiny (~0.24% avg win) — fees destroyed it.

v1b fixes this by:
1. Only trading ONE night per week (Friday close → Monday open captures
   the WEEKEND premium, which is the strongest overnight effect)
2. Adding a confidence threshold — only trade when top score is strong
3. Holding just 2 stocks overnight (4 trades/week instead of 50/week)
4. Adding a volatility filter — skip when VIX/market vol is extreme

Research basis unchanged: Kakushadze 2014, Glasserman 2025, Knuteson 2020.
The overnight premium is real — v1 proved that with 52% win rate and
$17.5K gross profit. We just need fewer, higher-conviction trades.

Expected trades: ~2 stocks × 2 orders × 50 weeks = ~200 trades/year
vs v1's ~1,080 trades/year. Fees should be ~$2.4K/year vs $12.9K/year.
"""

from AlgorithmImports import *
import numpy as np


class OvernightHoldV1b(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.top_n = 2                   # only 2 stocks overnight (concentrated, high-conviction)
        self.downtrend_top_n = 0         # NO overnight in downtrend (cash is king)
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Confidence threshold ──
        # Only trade when the top-scored stock's composite > this threshold
        # (rank-normalized scores range 0-1, so 0.70 = top ~30% of possible scores)
        self.min_score_threshold = 0.65

        # ── Overnight factor weights (sum to 1.0) ──
        self.w_momentum = 0.35
        self.w_volatility = 0.25
        self.w_liquidity = 0.25
        self.w_size = 0.15

        # ── Factor lookback periods ──
        self.mom_period = 5
        self.vol_period = 21
        self.liq_period = 10

        # ── Volatility filter: skip when realized SPY vol is extreme ──
        self.spy_vol_period = 21
        self.max_spy_vol = 0.40          # annualized; skip if SPY vol > 40%

        # ── 50-stock universe (same as v2 Monthly Rotator) ──
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

        # ── Data structures ──
        self.symbols = {}
        self.overnight_holdings = set()

        # ── Tracking ──
        self.total_trades = 0
        self.total_overnights = 0
        self.winning_nights = 0
        self.losing_nights = 0
        self.overnight_returns = []
        self.daily_pnl = []
        self.equity_before_buy = 0
        self.skipped_nights = 0
        self.skipped_downtrend = 0
        self.skipped_low_score = 0
        self.skipped_high_vol = 0
        self.max_consecutive_losses = 0
        self._current_loss_streak = 0

        # ── Add equities at MINUTE resolution ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.MINUTE)
            self.symbols[ticker] = equity.symbol

        # ── SPY for trend gate + vol filter ──
        self.add_equity("SPY", Resolution.MINUTE)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

        self.set_benchmark("SPY")

        # ── Schedule: ONLY on Fridays (weekend overnight) ──
        # Friday close → Monday open captures the longest overnight window
        # and the strongest weekend premium effect
        self.schedule.on(
            self.date_rules.every(DayOfWeek.FRIDAY),
            self.time_rules.before_market_close("SPY", 15),
            self.score_and_buy,
        )
        self.schedule.on(
            self.date_rules.every(DayOfWeek.MONDAY),
            self.time_rules.after_market_open("SPY", 1),
            self.sell_overnight_positions,
        )

        # Also try Tuesday close → Wednesday open (mid-week)
        self.schedule.on(
            self.date_rules.every(DayOfWeek.TUESDAY),
            self.time_rules.before_market_close("SPY", 15),
            self.score_and_buy,
        )
        self.schedule.on(
            self.date_rules.every(DayOfWeek.WEDNESDAY),
            self.time_rules.after_market_open("SPY", 1),
            self.sell_overnight_positions,
        )

        self.debug(
            f">>> OVERNIGHT HOLD v1b (WEEKLY SELECTIVE): {len(self.target_tickers)} stocks, "
            f"top {self.top_n}, threshold={self.min_score_threshold}, "
            f"Tue+Fri nights only"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Filters
    # ══════════════════════════════════════════════════════════════════════

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    def _spy_vol_too_high(self):
        """Check if SPY realized vol is too high (crisis mode)."""
        spy_history = self.history(
            self.symbol("SPY"), self.spy_vol_period + 1, Resolution.DAILY,
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
        spy_vol = np.std(returns) * np.sqrt(252)
        return spy_vol > self.max_spy_vol

    def on_data(self, data):
        pass

    # ══════════════════════════════════════════════════════════════════════
    # Morning sell (Monday + Wednesday)
    # ══════════════════════════════════════════════════════════════════════

    def sell_overnight_positions(self):
        if not self.overnight_holdings:
            return

        current_equity = self.portfolio.total_portfolio_value
        if self.equity_before_buy > 0:
            overnight_ret = (current_equity - self.equity_before_buy) / self.equity_before_buy
            self.overnight_returns.append(overnight_ret)
            self.daily_pnl.append(current_equity - self.equity_before_buy)

            if overnight_ret > 0:
                self.winning_nights += 1
                self._current_loss_streak = 0
            else:
                self.losing_nights += 1
                self._current_loss_streak += 1
                self.max_consecutive_losses = max(
                    self.max_consecutive_losses, self._current_loss_streak
                )

            self.debug(
                f"OVERNIGHT SELL: ret={overnight_ret:.4%}, "
                f"pnl=${current_equity - self.equity_before_buy:,.0f}, "
                f"holdings={list(self.overnight_holdings)}"
            )

        for ticker in list(self.overnight_holdings):
            symbol = self.symbols[ticker]
            if self.portfolio[symbol].invested:
                self.liquidate(symbol)
                self.total_trades += 1
        self.overnight_holdings.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Afternoon buy (Tuesday + Friday)
    # ══════════════════════════════════════════════════════════════════════

    def score_and_buy(self):
        self.total_overnights += 1

        # Filter 1: No overnight in downtrend
        if not self._is_uptrend():
            self.skipped_downtrend += 1
            self.skipped_nights += 1
            return

        # Filter 2: No overnight when SPY vol is extreme
        if self._spy_vol_too_high():
            self.skipped_high_vol += 1
            self.skipped_nights += 1
            return

        scores = self._score_stocks_overnight()
        if not scores:
            self.skipped_nights += 1
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Filter 3: Confidence threshold — only trade when top score is strong
        top_score = ranked[0][1]
        if top_score < self.min_score_threshold:
            self.skipped_low_score += 1
            self.skipped_nights += 1
            return

        # Only take stocks above threshold
        target_tickers = [t for t, s in ranked[:self.top_n] if s >= self.min_score_threshold]
        if not target_tickers:
            self.skipped_low_score += 1
            self.skipped_nights += 1
            return

        n_hold = len(target_tickers)

        self.equity_before_buy = self.portfolio.total_portfolio_value
        total_value = self.equity_before_buy

        if total_value <= 0:
            self.skipped_nights += 1
            return

        target_alloc = total_value / n_hold

        for ticker in target_tickers:
            symbol = self.symbols[ticker]
            price = self.securities[symbol].price
            if price <= 0:
                continue
            qty = int(target_alloc / price)
            if qty < 1:
                continue
            self.market_order(symbol, qty)
            self.total_trades += 1
            self.overnight_holdings.add(ticker)

        top3 = [(t, f"{s:.3f}") for t, s in ranked[:3]]
        self.debug(
            f"OVERNIGHT BUY: {len(self.overnight_holdings)} stocks, "
            f"top3={top3}, eq=${total_value:,.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Overnight scoring engine (4 factors)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks_overnight(self):
        scores = {}
        raw_data = {}

        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            history = self.history(
                symbol, self.vol_period + 5, Resolution.DAILY,
            )
            if history is None or history.empty:
                continue
            try:
                closes = history["close"].values
                volumes = history["volume"].values
            except Exception:
                continue
            if len(closes) < self.vol_period:
                continue

            price_now = closes[-1]
            if price_now <= 0:
                continue

            # Factor 1: Short-term momentum (5-day return)
            if len(closes) >= self.mom_period + 1:
                price_5d_ago = closes[-(self.mom_period + 1)]
                momentum = (price_now / price_5d_ago) - 1.0 if price_5d_ago > 0 else 0.0
            else:
                momentum = 0.0

            # Factor 2: Realized volatility (21-day, annualized)
            if len(closes) >= self.vol_period:
                returns = np.diff(closes[-self.vol_period:]) / closes[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            else:
                vol = 0.0

            # Factor 3: Liquidity (10-day average volume)
            if len(volumes) >= self.liq_period:
                avg_volume = np.mean(volumes[-self.liq_period:])
            else:
                avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

            # Factor 4: Size (inverse price)
            size_inv = 1.0 / price_now

            raw_data[ticker] = {
                "momentum": momentum,
                "vol": vol,
                "liquidity": avg_volume,
                "size": size_inv,
            }

        if len(raw_data) < 2:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        for signal in ["momentum", "vol", "liquidity", "size"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        for ticker in tickers:
            d = raw_data[ticker]
            score = (
                self.w_momentum * d["momentum_rank"]
                + self.w_volatility * d["vol_rank"]
                + self.w_liquidity * d["liquidity_rank"]
                + self.w_size * d["size_rank"]
            )
            scores[ticker] = score

        return scores

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        current_equity = self.portfolio.total_portfolio_value
        initial = 100_000
        ret_pct = (current_equity - initial) / initial

        self.debug(
            f"RESULTS: Return={ret_pct:.2%} Final=${current_equity:,.0f} "
            f"Trades={self.total_trades} Overnights={self.total_overnights} "
            f"Skipped={self.skipped_nights}"
        )
        self.debug(
            f"SKIP REASONS: downtrend={self.skipped_downtrend} "
            f"low_score={self.skipped_low_score} high_vol={self.skipped_high_vol}"
        )

        total_nights = self.winning_nights + self.losing_nights
        if total_nights > 0:
            win_rate = self.winning_nights / total_nights
            self.debug(
                f"OVERNIGHT: Win={self.winning_nights} Loss={self.losing_nights} "
                f"WR={win_rate:.1%} MaxConsecLoss={self.max_consecutive_losses}"
            )

        if self.overnight_returns:
            rets = np.array(self.overnight_returns)
            avg_ret = np.mean(rets)
            median_ret = np.median(rets)
            std_ret = np.std(rets)
            best = np.max(rets)
            worst = np.min(rets)
            sharpe_daily = avg_ret / std_ret if std_ret > 0 else 0
            # Scale by sqrt(trades_per_year) not sqrt(252) since we don't trade daily
            trades_per_year = total_nights / 5  # 5-year backtest
            sharpe_annual = sharpe_daily * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

            self.debug(
                f"RETURNS: Avg={avg_ret:.4%} Med={median_ret:.4%} "
                f"Std={std_ret:.4%} Best={best:.2%} Worst={worst:.2%} "
                f"Sharpe(annual)={sharpe_annual:.2f} Nights/yr={trades_per_year:.0f}"
            )

        if self.daily_pnl:
            total_pnl = sum(self.daily_pnl)
            avg_pnl = np.mean(self.daily_pnl)
            total_fees_est = self.total_trades * 12  # approx $12/trade
            self.debug(
                f"PNL: Total=${total_pnl:,.0f} Avg/night=${avg_pnl:,.0f} "
                f"EstFees=${total_fees_est:,.0f}"
            )
