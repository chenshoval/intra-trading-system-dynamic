"""FX Pairs Trading — Hypothesis 5c

Trades the SPREAD between cointegrated forex pairs, not individual pair direction.
The spread is stationary (mean-reverting) even when individual prices are not.

Key insight from research:
- 15-year static cointegration: 0 pairs found (relationship drifts)
- Rolling 6mo-2yr windows: 20-36 pairs found (temporary cointegration)
- Strategy must RE-EVALUATE pairs every formation period

Methodology (Gatev et al. 2006 adapted for FX):
- Formation period: 252 days (1 year) — test cointegration, compute hedge ratios
- Trading period: 126 days (6 months) — trade the spread on confirmed pairs
- Roll every 6 months: re-run formation, update pair list
- Entry: z-score > 2.0 or < -2.0
- Exit: z-score crosses 0 (mean reverted) or stop at z=3.5

Academic basis:
- Korniejczuk & Slepaczuk (2024): Graph clustering pairs, Kelly sizing
- Milstein et al. (2022): Neural Kalman for partial cointegration
- Gatev, Goetzmann & Rouwenhorst (2006): Original pairs trading methodology

Infrastructure reused from zone bounce (all IBKR bugs fixed):
- Quote currency conversion, per-currency min lots
- MOO error handling (skip hours 17/22/23/0)
- Orphan cleanup (1 attempt/pair/day)
- DD halt with peak reset, streak halt
- portfolio[symbol].invested entry guard
"""

from AlgorithmImports import *
import numpy as np
from collections import defaultdict


class ForexPairsTrading(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE)

        # ══════════════════════════════════════════════════════════════
        # Universe: 27 Forex Pairs (minus USDCHF which sometimes fails)
        # ══════════════════════════════════════════════════════════════
        self.all_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "NZDUSD", "USDCAD",
            "EURJPY", "EURGBP", "EURAUD", "EURNZD", "EURCHF",
            "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD",
            "AUDJPY", "AUDNZD", "AUDCHF",
            "NZDJPY", "NZDCHF", "NZDCAD",
            "CADJPY", "CADCHF", "CHFJPY",
            "EURCAD", "GBPCAD",
        ]

        # ══════════════════════════════════════════════════════════════
        # Pairs trading parameters
        # ══════════════════════════════════════════════════════════════
        self.formation_days = 252           # 1 year of daily data for cointegration test
        self.z_entry = 2.0                  # enter when |z-score| > 2
        self.z_exit = 0.5                   # exit when |z-score| < 0.5 (mean reverted)
        self.z_stop = 3.5                   # stop loss when |z-score| > 3.5 (diverging)
        self.spread_lookback = 60           # rolling window for z-score calculation
        self.max_spread_positions = 4       # max 4 spread trades (= 8 forex legs)
        self.max_hold_days = 15             # close after 15 days if not mean-reverted
        self.rebalance_months = 6           # re-evaluate pairs every 6 months
        self.min_coint_pvalue = 0.05        # cointegration significance threshold
        self.max_half_life = 25             # only trade pairs with HL < 25 days
        self.min_half_life = 2              # skip pairs that revert too fast (noise)

        # ══════════════════════════════════════════════════════════════
        # Position sizing
        # ══════════════════════════════════════════════════════════════
        self.risk_per_spread = 0.02         # 2% equity risk per spread trade
        self.max_units = 100_000            # hard cap per leg
        self.ibkr_min_by_currency = {
            "USD": 25_000, "EUR": 20_000, "GBP": 20_000,
            "JPY": 2_500_000, "CHF": 25_000, "AUD": 25_000,
            "NZD": 35_000, "CAD": 25_000,
        }

        # ══════════════════════════════════════════════════════════════
        # Risk management (from zone bounce, battle-tested)
        # ══════════════════════════════════════════════════════════════
        self.max_dd_pct = 0.15
        self.dd_cooldown_days = 20
        self.consecutive_loss_halt = 5
        self.consecutive_loss_cooldown = 10

        # ══════════════════════════════════════════════════════════════
        # Data structures
        # ══════════════════════════════════════════════════════════════
        self.symbols = {}

        # Active cointegrated pairs (updated every formation period)
        self.active_pairs = []              # list of {pair_a, pair_b, hedge_ratio, half_life}
        self.last_formation_date = None

        # Open spread positions
        self.open_spreads = {}              # key -> {pair_a, pair_b, direction, z_entry, entry_time, days_held}

        # Drawdown tracking
        self.peak_equity = 100_000
        self.trading_halted = False
        self.halt_until = None
        self.consecutive_losses = 0
        self.dd_halt_count = 0
        self.streak_halt_count = 0

        # Tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.formation_count = 0
        self.per_spread_pnl = defaultdict(float)
        self.per_spread_trades = defaultdict(int)
        self.per_spread_wins = defaultdict(int)
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)
        self.last_month = None

        # ══════════════════════════════════════════════════════════════
        # Add forex securities
        # ══════════════════════════════════════════════════════════════
        for pair in self.all_pairs:
            forex = self.add_forex(pair, Resolution.DAILY, Market.OANDA)
            self.symbols[pair] = forex.symbol

        # Daily check at 14:00 UTC (London afternoon, good liquidity)
        self.schedule.on(
            self.date_rules.every_day("EURUSD"),
            self.time_rules.at(14, 0),
            self.daily_check,
        )

        self.set_warm_up(timedelta(days=self.formation_days + 30))

        self.debug(
            f">>> FX PAIRS TRADING v1: {len(self.all_pairs)} pairs, "
            f"formation={self.formation_days}d, z_entry={self.z_entry}, "
            f"z_exit={self.z_exit}, z_stop={self.z_stop}, "
            f"max_spreads={self.max_spread_positions}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # IBKR helpers (from zone bounce, battle-tested)
    # ══════════════════════════════════════════════════════════════════════

    def _get_ibkr_min_units(self, pair):
        base = pair[:3]
        return self.ibkr_min_by_currency.get(base, 25_000)

    def _get_quote_currency_rate(self, pair):
        quote = pair[3:6]
        if quote == "USD":
            return 1.0
        direct = f"{quote}USD"
        if direct in self.symbols:
            price = self.securities[self.symbols[direct]].price
            if price > 0:
                return price
        inverse = f"USD{quote}"
        if inverse in self.symbols:
            price = self.securities[self.symbols[inverse]].price
            if price > 0:
                return 1.0 / price
        return 1.0

    # ══════════════════════════════════════════════════════════════════════
    # Formation: Find cointegrated pairs
    # ══════════════════════════════════════════════════════════════════════

    def _run_formation(self):
        """Test all pair combinations for cointegration using recent data."""
        self.formation_count += 1
        self.active_pairs = []

        # Get formation period data
        candidates = []
        for pair_a in self.all_pairs:
            for pair_b in self.all_pairs:
                if pair_a >= pair_b:  # avoid duplicates and self-pairs
                    continue

                hist_a = self.history(self.symbols[pair_a], self.formation_days, Resolution.DAILY)
                hist_b = self.history(self.symbols[pair_b], self.formation_days, Resolution.DAILY)

                if hist_a is None or hist_b is None or hist_a.empty or hist_b.empty:
                    continue

                try:
                    closes_a = hist_a["close"].values
                    closes_b = hist_b["close"].values
                except Exception:
                    continue

                min_len = min(len(closes_a), len(closes_b))
                if min_len < self.formation_days - 30:
                    continue

                closes_a = closes_a[-min_len:]
                closes_b = closes_b[-min_len:]

                if np.any(closes_a <= 0) or np.any(closes_b <= 0):
                    continue

                try:
                    log_a = np.log(closes_a)
                    log_b = np.log(closes_b)

                    # Engle-Granger cointegration test
                    from statsmodels.tsa.stattools import coint
                    score, pvalue, _ = coint(log_a, log_b)

                    if pvalue > self.min_coint_pvalue:
                        continue

                    # Hedge ratio via OLS
                    beta = np.cov(log_a, log_b)[0, 1] / np.var(log_b)

                    # Half-life
                    spread = log_a - beta * log_b
                    spread_lag = spread[:-1]
                    spread_diff = np.diff(spread)
                    if np.std(spread_lag) > 1e-10:
                        phi = np.corrcoef(spread_lag, spread_diff)[0, 1] * np.std(spread_diff) / np.std(spread_lag)
                        half_life = -np.log(2) / np.log(1 + phi) if phi < 0 else 999
                    else:
                        half_life = 999

                    if half_life < self.min_half_life or half_life > self.max_half_life:
                        continue

                    candidates.append({
                        "pair_a": pair_a,
                        "pair_b": pair_b,
                        "pvalue": pvalue,
                        "hedge_ratio": beta,
                        "half_life": half_life,
                        "spread_mean": np.mean(spread),
                        "spread_std": np.std(spread),
                    })

                except Exception:
                    continue

        # Sort by p-value, take top pairs
        candidates.sort(key=lambda x: x["pvalue"])
        self.active_pairs = candidates[:10]  # top 10 cointegrated pairs

        self.last_formation_date = self.time

        self.debug(
            f"FORMATION #{self.formation_count}: Found {len(candidates)} cointegrated pairs, "
            f"selected top {len(self.active_pairs)}"
        )
        for i, p in enumerate(self.active_pairs):
            self.debug(
                f"  {i+1}. {p['pair_a']}/{p['pair_b']} p={p['pvalue']:.4f} "
                f"HL={p['half_life']:.1f}d beta={p['hedge_ratio']:.4f}"
            )

    # ══════════════════════════════════════════════════════════════════════
    # Compute z-score for a pair spread
    # ══════════════════════════════════════════════════════════════════════

    def _compute_z_score(self, pair_info):
        """Compute current z-score of the spread for a cointegrated pair."""
        pair_a = pair_info["pair_a"]
        pair_b = pair_info["pair_b"]
        beta = pair_info["hedge_ratio"]

        hist_a = self.history(self.symbols[pair_a], self.spread_lookback, Resolution.DAILY)
        hist_b = self.history(self.symbols[pair_b], self.spread_lookback, Resolution.DAILY)

        if hist_a is None or hist_b is None or hist_a.empty or hist_b.empty:
            return None

        try:
            closes_a = hist_a["close"].values
            closes_b = hist_b["close"].values
        except Exception:
            return None

        min_len = min(len(closes_a), len(closes_b))
        if min_len < self.spread_lookback - 10:
            return None

        closes_a = closes_a[-min_len:]
        closes_b = closes_b[-min_len:]

        if np.any(closes_a <= 0) or np.any(closes_b <= 0):
            return None

        log_a = np.log(closes_a)
        log_b = np.log(closes_b)
        spread = log_a - beta * log_b

        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        if spread_std < 1e-10:
            return None

        current_spread = spread[-1]
        z_score = (current_spread - spread_mean) / spread_std

        return z_score

    # ══════════════════════════════════════════════════════════════════════
    # Daily check: formation, entries, exits
    # ══════════════════════════════════════════════════════════════════════

    def daily_check(self):
        if self.is_warming_up:
            return

        equity = self.portfolio.total_portfolio_value

        # Monthly tracking
        current_month = f"{self.time.year}-{self.time.month:02d}"
        if self.last_month is not None and current_month != self.last_month:
            month_ret = (equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)
            self.monthly_pnl[self.last_month] = equity - self.month_start_equity
            self.month_start_equity = equity
        self.last_month = current_month

        # ── Drawdown protection (from zone bounce) ──
        if equity > self.peak_equity:
            self.peak_equity = equity

        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Orphan cleanup (1 attempt/pair/day, safe hours only)
        if not hasattr(self, '_orphan_cleanup_dates'):
            self._orphan_cleanup_dates = {}
        today = self.time.date()
        hour = self.time.hour
        if hour not in (17, 22, 23, 0):
            for pair in self.all_pairs:
                symbol = self.symbols[pair]
                # Check if this pair is part of any open spread
                pair_in_spread = any(
                    pair in (s["pair_a"], s["pair_b"]) for s in self.open_spreads.values()
                )
                if not pair_in_spread and self.portfolio[symbol].invested:
                    if self._orphan_cleanup_dates.get(pair) != today:
                        self._orphan_cleanup_dates[pair] = today
                        qty = self.portfolio[symbol].quantity
                        self.market_order(symbol, -qty)
                        self.debug(f"ORPHAN CLEANUP: {pair} had {qty:,.0f} units")

        # DD halt check
        if self.trading_halted and self.halt_until is not None:
            if self.time >= self.halt_until:
                self.trading_halted = False
                self.halt_until = None
                self.consecutive_losses = 0
                self.peak_equity = equity
                self.debug(f"TRADING RESUMED: equity=${equity:,.0f}, peak reset")

        if current_dd > self.max_dd_pct and not self.trading_halted:
            self.trading_halted = True
            self.halt_until = self.time + timedelta(days=self.dd_cooldown_days)
            self.dd_halt_count += 1
            for key in list(self.open_spreads.keys()):
                self._exit_spread(key, "DD Halt")
            self.debug(
                f"DD HALT: {current_dd:.1%} > {self.max_dd_pct:.0%}. "
                f"Equity=${equity:,.0f} Peak=${self.peak_equity:,.0f}"
            )
            return

        # ── Run formation if needed ──
        if self.last_formation_date is None:
            self._run_formation()
        elif (self.time - self.last_formation_date).days >= self.rebalance_months * 30:
            self._run_formation()

        if not self.active_pairs:
            return

        if self.trading_halted:
            return

        # ── Check existing spread positions ──
        for key in list(self.open_spreads.keys()):
            spread = self.open_spreads[key]
            spread["days_held"] += 1

            # Find the pair_info for this spread
            pair_info = None
            for p in self.active_pairs:
                if p["pair_a"] == spread["pair_a"] and p["pair_b"] == spread["pair_b"]:
                    pair_info = p
                    break

            if pair_info is None:
                # Pair no longer in active list after re-formation — close it
                self._exit_spread(key, "Pair Dropped")
                continue

            z = self._compute_z_score(pair_info)
            if z is None:
                continue

            # Exit: mean reverted
            if abs(z) < self.z_exit:
                self._exit_spread(key, "Mean Reverted")
                continue

            # Stop: diverging further
            if (spread["direction"] == 1 and z > self.z_stop) or \
               (spread["direction"] == -1 and z < -self.z_stop):
                self._exit_spread(key, "Z-Stop")
                continue

            # Time stop
            if spread["days_held"] >= self.max_hold_days:
                self._exit_spread(key, "Time Stop")
                continue

        # ── Check for new entries ──
        if len(self.open_spreads) >= self.max_spread_positions:
            return

        for pair_info in self.active_pairs:
            if len(self.open_spreads) >= self.max_spread_positions:
                break

            key = f"{pair_info['pair_a']}/{pair_info['pair_b']}"
            if key in self.open_spreads:
                continue

            # Check portfolio — don't enter if either leg is already invested
            sym_a = self.symbols[pair_info["pair_a"]]
            sym_b = self.symbols[pair_info["pair_b"]]
            if self.portfolio[sym_a].invested or self.portfolio[sym_b].invested:
                continue

            z = self._compute_z_score(pair_info)
            if z is None:
                continue

            if z > self.z_entry:
                # Spread is too high — sell spread (short A, long B)
                self._enter_spread(pair_info, direction=-1, z_score=z)
            elif z < -self.z_entry:
                # Spread is too low — buy spread (long A, short B)
                self._enter_spread(pair_info, direction=1, z_score=z)

    # ══════════════════════════════════════════════════════════════════════
    # Enter a spread trade (two legs)
    # ══════════════════════════════════════════════════════════════════════

    def _enter_spread(self, pair_info, direction, z_score):
        pair_a = pair_info["pair_a"]
        pair_b = pair_info["pair_b"]
        beta = pair_info["hedge_ratio"]
        key = f"{pair_a}/{pair_b}"

        equity = self.portfolio.total_portfolio_value
        risk_amount = equity * self.risk_per_spread

        # Size leg A
        price_a = self.securities[self.symbols[pair_a]].price
        price_b = self.securities[self.symbols[pair_b]].price
        if price_a <= 0 or price_b <= 0:
            return

        quote_rate_a = self._get_quote_currency_rate(pair_a)
        quote_rate_b = self._get_quote_currency_rate(pair_b)

        # Target ~equal USD notional on each leg
        # Leg A notional = risk_amount, Leg B notional = risk_amount * |beta|
        notional_a = risk_amount
        notional_b = risk_amount * abs(beta) if abs(beta) > 0.1 else risk_amount

        units_a = int(notional_a / (price_a * quote_rate_a))
        units_b = int(notional_b / (price_b * quote_rate_b))

        # Apply IBKR minimums and caps
        min_a = self._get_ibkr_min_units(pair_a)
        min_b = self._get_ibkr_min_units(pair_b)

        units_a = max(min_a, min(self.max_units, (units_a // 1000) * 1000))
        units_b = max(min_b, min(self.max_units, (units_b // 1000) * 1000))

        if units_a < min_a or units_b < min_b:
            return

        # Direction: +1 = buy spread (long A, short B), -1 = sell spread (short A, long B)
        qty_a = units_a * direction
        qty_b = -units_b * direction  # opposite leg

        self.market_order(self.symbols[pair_a], qty_a)
        self.market_order(self.symbols[pair_b], qty_b)

        self.total_trades += 2  # two legs
        self.open_spreads[key] = {
            "pair_a": pair_a,
            "pair_b": pair_b,
            "direction": direction,
            "z_entry": z_score,
            "entry_time": self.time,
            "days_held": 0,
            "qty_a": qty_a,
            "qty_b": qty_b,
            "entry_price_a": price_a,
            "entry_price_b": price_b,
        }

        dir_str = "BUY" if direction == 1 else "SELL"
        self.debug(
            f"{dir_str} SPREAD {key} | z={z_score:.2f} | "
            f"A: {qty_a:,} @ {price_a:.5f} | B: {qty_b:,} @ {price_b:.5f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Exit a spread trade
    # ══════════════════════════════════════════════════════════════════════

    def _exit_spread(self, key, reason):
        if key not in self.open_spreads:
            return

        spread = self.open_spreads[key]
        sym_a = self.symbols[spread["pair_a"]]
        sym_b = self.symbols[spread["pair_b"]]

        # Calculate P&L
        price_a = self.securities[sym_a].price
        price_b = self.securities[sym_b].price
        quote_rate_a = self._get_quote_currency_rate(spread["pair_a"])
        quote_rate_b = self._get_quote_currency_rate(spread["pair_b"])

        pnl_a = (price_a - spread["entry_price_a"]) * spread["qty_a"] * quote_rate_a
        pnl_b = (price_b - spread["entry_price_b"]) * spread["qty_b"] * quote_rate_b
        total_pnl = pnl_a + pnl_b
        is_win = total_pnl > 0

        # Close both legs
        close_a = -spread["qty_a"]
        close_b = -spread["qty_b"]
        order_a = self.market_order(sym_a, close_a)
        order_b = self.market_order(sym_b, close_b)

        # Fallback if order fails
        if order_a is None:
            self.liquidate(sym_a)
        if order_b is None:
            self.liquidate(sym_b)

        # Record stats
        self.per_spread_pnl[key] += total_pnl
        self.per_spread_trades[key] += 1
        if is_win:
            self.winning_trades += 1
            self.per_spread_wins[key] += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.consecutive_loss_halt and not self.trading_halted:
                self.trading_halted = True
                self.halt_until = self.time + timedelta(days=self.consecutive_loss_cooldown)
                self.streak_halt_count += 1
                self.debug(f"STREAK HALT: {self.consecutive_losses} consecutive losses")

        self.debug(
            f"EXIT SPREAD {key} | {reason} | PnL: ${total_pnl:,.0f} | "
            f"Held: {spread['days_held']}d | z_entry={spread['z_entry']:.2f}"
        )

        del self.open_spreads[key]

    # ══════════════════════════════════════════════════════════════════════
    # OnData — not used (daily scheduling only)
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        pass

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        equity = self.portfolio.total_portfolio_value
        ret_pct = (equity - 100_000) / 100_000

        if self.last_month is not None:
            month_ret = (equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)

        total_closed = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_closed * 100 if total_closed > 0 else 0

        self.debug("=" * 70)
        self.debug("FX PAIRS TRADING — FINAL RESULTS")
        self.debug("=" * 70)
        self.debug(f"RETURNS: Total={ret_pct:.2%} Final=${equity:,.0f}")
        self.debug(
            f"TRADES: Total={self.total_trades} Spreads={total_closed} "
            f"W={self.winning_trades} L={self.losing_trades} WR={win_rate:.0f}% "
            f"Formations={self.formation_count} "
            f"DDHalts={self.dd_halt_count} StreakHalts={self.streak_halt_count}"
        )

        if self.monthly_returns:
            rets = np.array(self.monthly_returns)
            avg = np.mean(rets)
            std = np.std(rets)
            sharpe = avg / std if std > 0 else 0
            self.debug(
                f"MONTHLY: Avg={avg:.2%} Best={np.max(rets):.2%} "
                f"Worst={np.min(rets):.2%} Sharpe={sharpe:.2f} "
                f"Win={np.sum(rets > 0)} Loss={np.sum(rets <= 0)}"
            )

        if self.per_spread_pnl:
            self.debug("\nPER-SPREAD ANALYSIS:")
            for key, pnl in sorted(self.per_spread_pnl.items(), key=lambda x: x[1], reverse=True):
                trades = self.per_spread_trades.get(key, 0)
                wins = self.per_spread_wins.get(key, 0)
                wr = wins / trades * 100 if trades > 0 else 0
                self.debug(f"  {key:<20s}: PnL=${pnl:>+10,.0f} Trades={trades:>3d} WR={wr:>4.0f}%")

        if self.active_pairs:
            self.debug(f"\nFINAL ACTIVE PAIRS ({len(self.active_pairs)}):")
            for p in self.active_pairs:
                self.debug(f"  {p['pair_a']}/{p['pair_b']} p={p['pvalue']:.4f} HL={p['half_life']:.1f}d")

        if self.monthly_pnl:
            pnl_str = " | ".join(f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items()))
            self.debug(f"\nMONTHLY PNL: {pnl_str}")
