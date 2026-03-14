"""Overnight Hold v1 Broad — ETF Universe

Same overnight strategy as v1 but using broad market and sector ETFs
instead of individual stocks. Fewer securities = lower transaction costs,
broader diversification, and simpler to manage.

Universe includes:
- Broad market: SPY, QQQ, IWM
- Sector ETFs: XLK, XLF, XLV, XLE, XLI, XLP, XLY, XLU
- Thematic: SOXX (semis), XBI (biotech), XHB (homebuilders)
- Alternative: GLD (gold), TLT (long bonds), HYG (high yield)

Alternative ETFs provide diversification since overnight returns
differ by asset class — gold and bonds have different overnight
patterns than equities.

All ETFs are highly liquid, tight spreads, and available on IBKR.
"""

from AlgorithmImports import *
import numpy as np


class OvernightHoldV1Broad(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        # ── Portfolio parameters ──
        self.top_n = 3                   # ETFs to hold overnight (uptrend)
        self.downtrend_top_n = 1         # reduced in downtrend
        self.trend_fast = 10
        self.trend_slow = 50

        # ── Overnight factor weights (sum to 1.0) ──
        self.w_momentum = 0.35
        self.w_volatility = 0.25
        self.w_liquidity = 0.25
        self.w_size = 0.15

        # ── Factor lookback periods ──
        self.mom_period = 5
        self.vol_period = 21
        self.liq_period = 10

        # ── ETF universe ──
        self.target_tickers = [
            # Broad market
            "SPY", "QQQ", "IWM",
            # Sector ETFs
            "XLK", "XLF", "XLV", "XLE",
            "XLI", "XLP", "XLY", "XLU",
            # Thematic
            "SOXX", "XBI", "XHB",
            # Alternative assets
            "GLD", "TLT", "HYG",
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
        self.max_consecutive_losses = 0
        self._current_loss_streak = 0

        # ── Add ETFs at MINUTE resolution ──
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.MINUTE)
            self.symbols[ticker] = equity.symbol

        # ── SPY for trend gate (already in universe) ──
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
            f">>> OVERNIGHT HOLD v1 BROAD: {len(self.target_tickers)} ETFs, "
            f"top {self.top_n} overnight"
        )

    def _is_uptrend(self):
        if not self.spy_fast_ma.is_ready or not self.spy_slow_ma.is_ready:
            return True
        return self.spy_fast_ma.current.value > self.spy_slow_ma.current.value

    def on_data(self, data):
        pass

    # ══════════════════════════════════════════════════════════════════════
    # 9:31 AM — Sell all overnight positions
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

        for ticker in list(self.overnight_holdings):
            symbol = self.symbols[ticker]
            if self.portfolio[symbol].invested:
                self.liquidate(symbol)
                self.total_trades += 1
        self.overnight_holdings.clear()

    # ══════════════════════════════════════════════════════════════════════
    # 3:45 PM — Score ETFs and buy top N
    # ══════════════════════════════════════════════════════════════════════

    def score_and_buy(self):
        self.total_overnights += 1

        uptrend = self._is_uptrend()
        n_hold = self.top_n if uptrend else self.downtrend_top_n

        scores = self._score_stocks_overnight()
        if not scores:
            self.skipped_nights += 1
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = [t for t, _ in ranked[:n_hold]]

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
            f"OVERNIGHT BUY [{regime}]: {len(self.overnight_holdings)} ETFs, "
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

        if len(raw_data) < self.top_n:
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
