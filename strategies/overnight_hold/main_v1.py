"""Overnight Hold v1 — 50-Stock Universe

Captures the overnight return premium: buy near market close, sell at open.
Research basis: Kakushadze 2014 (4-factor overnight model),
               Glasserman 2025 (overnight news drives returns),
               Knuteson 2020 (overnight returns >> intraday returns).

How it works:
- Every day at 3:45 PM: score all 50 stocks using 4 overnight factors
- Buy top 5 (equal-weight) near market close
- Next day at 9:31 AM: sell all positions at open
- Sit in cash during the trading day
- SPY trend gate: reduce to top 2 in downtrend

The 4 overnight factors (from Kakushadze 2014):
1. Momentum (5-day): recent winners tend to gap up overnight
2. Volatility (21-day): higher vol = bigger overnight moves
3. Liquidity (10-day avg volume): institutional overnight flow
4. Size (inverse price): cheaper stocks have larger % gaps

All factors computed from OHLCV — no alternative data needed.
This is designed to be UNCORRELATED with v2 Monthly Rotator
(different timeframe, different factors, different hold period).
"""

from AlgorithmImports import *
import numpy as np


class OvernightHoldV1(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.top_n = 5                   # stocks to hold overnight (uptrend)
        self.downtrend_top_n = 2         # reduced overnight positions in downtrend
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Overnight factor weights (sum to 1.0) ──
        self.w_momentum = 0.35           # strongest predictor per Kakushadze
        self.w_volatility = 0.25         # high-vol stocks have bigger overnight gaps
        self.w_liquidity = 0.25          # institutional flow drives overnight
        self.w_size = 0.15               # smaller effect, controlled by universe

        # ── Factor lookback periods ──
        self.mom_period = 5              # 5-day momentum (short-term for overnight)
        self.vol_period = 21             # 21-day realized vol
        self.liq_period = 10             # 10-day average volume

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
        self.overnight_holdings = set()  # tickers currently held overnight

        # ── Tracking ──
        self.total_trades = 0
        self.total_overnights = 0        # number of overnight sessions
        self.winning_nights = 0
        self.losing_nights = 0
        self.overnight_returns = []      # daily overnight return (portfolio-level)
        self.daily_pnl = []
        self.equity_before_buy = 0       # snapshot before buying at close
        self.skipped_nights = 0          # nights skipped due to downtrend or no scores
        self.max_consecutive_losses = 0
        self._current_loss_streak = 0

        # ── Add equities at MINUTE resolution (needed for intraday scheduling) ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.MINUTE)
            self.symbols[ticker] = equity.symbol

        # ── SPY for trend gate ──
        self.add_equity("SPY", Resolution.MINUTE)
        self.spy_fast_ma = self.sma("SPY", self.trend_fast, Resolution.DAILY)
        self.spy_slow_ma = self.sma("SPY", self.trend_slow, Resolution.DAILY)

        self.set_benchmark("SPY")

        # ── Schedule: sell at open, buy near close ──
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 1),
            self.sell_overnight_positions,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 15),
            self.score_and_buy,
        )

        self.debug(
            f">>> OVERNIGHT HOLD v1: {len(self.target_tickers)} stocks, "
            f"top {self.top_n} overnight, weights=mom{self.w_momentum}/vol{self.w_volatility}/"
            f"liq{self.w_liquidity}/size{self.w_size}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Trend gate (same as v2)
    # ══════════════════════════════════════════════════════════════════════

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    # ══════════════════════════════════════════════════════════════════════
    # on_data — empty (all logic in scheduled events)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        pass

    # ══════════════════════════════════════════════════════════════════════
    # 9:31 AM — Sell all overnight positions
    # ══════════════════════════════════════════════════════════════════════

    def sell_overnight_positions(self):
        """Sell everything at market open. Record overnight P&L."""
        if not self.overnight_holdings:
            return

        # Calculate overnight return before selling
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

        # Liquidate all overnight positions
        for ticker in list(self.overnight_holdings):
            symbol = self.symbols[ticker]
            if self.portfolio[symbol].invested:
                self.liquidate(symbol)
                self.total_trades += 1
        self.overnight_holdings.clear()

    # ══════════════════════════════════════════════════════════════════════
    # 3:45 PM — Score stocks and buy top N
    # ══════════════════════════════════════════════════════════════════════

    def score_and_buy(self):
        """Score all stocks on 4 overnight factors, buy top N near close."""
        self.total_overnights += 1

        # Determine position count based on trend
        uptrend = self._is_uptrend()
        n_hold = self.top_n if uptrend else self.downtrend_top_n

        scores = self._score_stocks_overnight()
        if not scores:
            self.skipped_nights += 1
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = [t for t, _ in ranked[:n_hold]]

        # Snapshot equity before buying
        self.equity_before_buy = self.portfolio.total_portfolio_value
        total_value = self.equity_before_buy

        if total_value <= 0 or n_hold <= 0:
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

        regime = "UP" if uptrend else "DOWN"
        top3 = [(t, f"{s:.3f}") for t, s in ranked[:3]]
        self.debug(
            f"OVERNIGHT BUY [{regime}]: {len(self.overnight_holdings)} stocks, "
            f"top3={top3}, eq=${total_value:,.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Overnight scoring engine (4 factors)
    # ══════════════════════════════════════════════════════════════════════

    def _score_stocks_overnight(self):
        """Score stocks on 4 overnight factors from Kakushadze 2014."""
        scores = {}
        raw_data = {}

        for ticker in self.target_tickers:
            symbol = self.symbols[ticker]
            # Pull enough history for all factors
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

            # Factor 4: Size (inverse price — smaller = higher score)
            size_inv = 1.0 / price_now

            raw_data[ticker] = {
                "momentum": momentum,
                "vol": vol,
                "liquidity": avg_volume,
                "size": size_inv,
            }

        if len(raw_data) < self.top_n:
            return scores

        tickers = list(raw_data.keys())
        n = len(tickers)

        # Rank-normalize all 4 factors (higher rank = better for overnight)
        for signal in ["momentum", "vol", "liquidity", "size"]:
            ranked = sorted(tickers, key=lambda t: raw_data[t][signal])
            for i, t in enumerate(ranked):
                raw_data[t][f"{signal}_rank"] = i / (n - 1) if n > 1 else 0.5

        # Composite score
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
    # End of algorithm — summary stats
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
            sharpe_annual = sharpe_daily * np.sqrt(252)

            self.debug(
                f"RETURNS: Avg={avg_ret:.4%} Med={median_ret:.4%} "
                f"Std={std_ret:.4%} Best={best:.2%} Worst={worst:.2%} "
                f"DailySharpe={sharpe_daily:.3f} AnnualSharpe={sharpe_annual:.2f}"
            )

        if self.daily_pnl:
            total_pnl = sum(self.daily_pnl)
            avg_pnl = np.mean(self.daily_pnl)
            self.debug(
                f"PNL: Total=${total_pnl:,.0f} Avg/night=${avg_pnl:,.0f}"
            )
