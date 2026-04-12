"""FX Carry Trade v2 — Hypothesis 5d

Long high-yield currencies, short low-yield currencies.
Earns the interest rate differential (swap/carry) daily.

v2 changes from v1 (-8.13% price, +$3.5K carry = -4.6% total, Sharpe -0.55):
- Force rebalance to target weights every month (v1 had 20% threshold = barely traded)
- Momentum overlay: only long if 3M return > 0, only short if 3M return < 0
  (Barroso 2015: carry + momentum combo outperforms either alone)
- Lower vol-scale cap: 1.5x (v1 was stuck at 2.0x for years in low-vol FX)
- VIX risk-off filter: flatten all positions when VIX > 25 (carry crashes in risk-off)
- Per-pair P&L tracking in end-of-algo report

v1 problem: only 4 trades / 14 orders in 5 years. Rankings almost never changed,
so positions were static = just directional FX exposure, not a real carry strategy.
v2 forces monthly resize + momentum filter causes more turnover.

IMPORTANT: QC does NOT simulate swap/carry income in backtests.
The equity curve shows PRICE-ONLY P&L. Carry income is tracked
separately in logs. True return = QC return + carry income.

Academic basis:
- Burnside, Eichenbaum & Rebelo (2011) "Carry Trades and Risk" — RFS
- Barroso & Santa-Clara (2015) "Beyond the Carry Trade" — JFE
- Asness, Moskowitz & Pedersen (2013) "Value and Momentum Everywhere" — JF
"""

from AlgorithmImports import *
import numpy as np
from datetime import date
from collections import defaultdict
from bisect import bisect_right


# ══════════════════════════════════════════════════════════════════════
# Central Bank Rate Table (hardcoded, publicly known dates)
# Format: sorted list of (effective_date, rate_pct) per currency
# Rate = policy/target rate as a percentage (e.g., 5.50 = 5.50%)
#
# Sources: Federal Reserve, ECB, BoJ, BoE, RBA, RBNZ, SNB, BoC
# official policy rate announcements
# ══════════════════════════════════════════════════════════════════════

CENTRAL_BANK_RATES = {
    "USD": [
        # Federal Funds Rate (upper bound of target range from Dec 2008)
        (date(2008, 1, 22), 3.50), (date(2008, 1, 30), 3.00),
        (date(2008, 3, 18), 2.25), (date(2008, 4, 30), 2.00),
        (date(2008, 10, 8), 1.50), (date(2008, 10, 29), 1.00),
        (date(2008, 12, 16), 0.25),  # ZIRP begins
        # Zero for 7 years
        (date(2015, 12, 17), 0.50),
        (date(2016, 12, 15), 0.75),
        (date(2017, 3, 16), 1.00), (date(2017, 6, 15), 1.25),
        (date(2017, 12, 14), 1.50),
        (date(2018, 3, 22), 1.75), (date(2018, 6, 14), 2.00),
        (date(2018, 9, 27), 2.25), (date(2018, 12, 20), 2.50),
        (date(2019, 8, 1), 2.25), (date(2019, 9, 19), 2.00),
        (date(2019, 10, 31), 1.75),
        (date(2020, 3, 3), 1.25), (date(2020, 3, 16), 0.25),  # COVID cut
        # Zero again for 2 years
        (date(2022, 3, 17), 0.50), (date(2022, 5, 5), 1.00),
        (date(2022, 6, 16), 1.75), (date(2022, 7, 28), 2.50),
        (date(2022, 9, 22), 3.25), (date(2022, 11, 3), 4.00),
        (date(2022, 12, 15), 4.50),
        (date(2023, 2, 2), 4.75), (date(2023, 3, 23), 5.00),
        (date(2023, 5, 4), 5.25), (date(2023, 7, 27), 5.50),
        (date(2024, 9, 19), 5.00), (date(2024, 11, 8), 4.75),
        (date(2024, 12, 19), 4.50),
    ],
    "EUR": [
        # ECB Main Refinancing Rate
        (date(2008, 7, 9), 4.25), (date(2008, 10, 8), 3.75),
        (date(2008, 11, 6), 3.25), (date(2008, 12, 4), 2.50),
        (date(2009, 1, 15), 2.00), (date(2009, 3, 5), 1.50),
        (date(2009, 4, 2), 1.25), (date(2009, 5, 7), 1.00),
        (date(2011, 4, 7), 1.25), (date(2011, 7, 7), 1.50),
        (date(2011, 11, 3), 1.25), (date(2011, 12, 8), 1.00),
        (date(2012, 7, 5), 0.75), (date(2013, 5, 2), 0.50),
        (date(2013, 11, 7), 0.25), (date(2014, 6, 5), 0.15),
        (date(2014, 9, 4), 0.05), (date(2016, 3, 10), 0.00),
        (date(2022, 7, 27), 0.50), (date(2022, 9, 14), 1.25),
        (date(2022, 11, 2), 2.00), (date(2022, 12, 21), 2.50),
        (date(2023, 2, 8), 3.00), (date(2023, 3, 22), 3.50),
        (date(2023, 5, 10), 3.75), (date(2023, 6, 21), 4.00),
        (date(2023, 8, 2), 4.25), (date(2023, 9, 20), 4.50),
        (date(2024, 6, 12), 4.25), (date(2024, 9, 18), 3.65),
        (date(2024, 10, 23), 3.40), (date(2024, 12, 18), 3.15),
        (date(2025, 1, 30), 2.90), (date(2025, 3, 6), 2.65),
    ],
    "JPY": [
        # Bank of Japan Policy Rate
        (date(2008, 10, 31), 0.30), (date(2008, 12, 19), 0.10),
        (date(2010, 10, 5), 0.00),   # ZIRP / QQE era
        (date(2016, 2, 16), -0.10),  # Negative rates
        (date(2024, 3, 19), 0.00),   # Exit negative rates
        (date(2024, 7, 31), 0.25),
        (date(2025, 1, 24), 0.50),
    ],
    "GBP": [
        # Bank of England Bank Rate
        (date(2008, 2, 7), 5.25), (date(2008, 4, 10), 5.00),
        (date(2008, 10, 8), 4.50), (date(2008, 11, 6), 3.00),
        (date(2008, 12, 4), 2.00), (date(2009, 1, 8), 1.50),
        (date(2009, 2, 5), 1.00), (date(2009, 3, 5), 0.50),
        (date(2016, 8, 4), 0.25), (date(2017, 11, 2), 0.50),
        (date(2018, 8, 2), 0.75),
        (date(2020, 3, 11), 0.25), (date(2020, 3, 19), 0.10),
        (date(2021, 12, 16), 0.25), (date(2022, 2, 3), 0.50),
        (date(2022, 3, 17), 0.75), (date(2022, 5, 5), 1.00),
        (date(2022, 6, 16), 1.25), (date(2022, 8, 4), 1.75),
        (date(2022, 9, 22), 2.25), (date(2022, 11, 3), 3.00),
        (date(2022, 12, 15), 3.50),
        (date(2023, 2, 2), 4.00), (date(2023, 3, 23), 4.25),
        (date(2023, 5, 11), 4.50), (date(2023, 6, 22), 5.00),
        (date(2023, 8, 3), 5.25),
        (date(2024, 8, 1), 5.00), (date(2024, 11, 7), 4.75),
        (date(2025, 2, 6), 4.50),
    ],
    "AUD": [
        # Reserve Bank of Australia Cash Rate
        (date(2008, 2, 5), 7.00), (date(2008, 3, 4), 7.25),
        (date(2008, 9, 2), 7.00), (date(2008, 10, 7), 6.00),
        (date(2008, 11, 4), 5.25), (date(2008, 12, 2), 4.25),
        (date(2009, 2, 3), 3.25), (date(2009, 4, 7), 3.00),
        (date(2009, 10, 6), 3.25), (date(2009, 11, 3), 3.50),
        (date(2009, 12, 1), 3.75), (date(2010, 3, 2), 4.00),
        (date(2010, 4, 6), 4.25), (date(2010, 5, 4), 4.50),
        (date(2010, 11, 2), 4.75),
        (date(2011, 11, 1), 4.50), (date(2011, 12, 6), 4.25),
        (date(2012, 5, 1), 3.75), (date(2012, 6, 5), 3.50),
        (date(2012, 10, 2), 3.25), (date(2012, 12, 4), 3.00),
        (date(2013, 5, 7), 2.75), (date(2013, 8, 6), 2.50),
        (date(2014, 8, 5), 2.50),
        (date(2015, 2, 3), 2.25), (date(2015, 5, 5), 2.00),
        (date(2016, 5, 3), 1.75), (date(2016, 8, 2), 1.50),
        (date(2019, 6, 4), 1.25), (date(2019, 7, 2), 1.00),
        (date(2019, 10, 1), 0.75),
        (date(2020, 3, 3), 0.50), (date(2020, 3, 19), 0.25),
        (date(2020, 11, 3), 0.10),
        (date(2022, 5, 3), 0.35), (date(2022, 6, 7), 0.85),
        (date(2022, 7, 5), 1.35), (date(2022, 8, 2), 1.85),
        (date(2022, 9, 6), 2.35), (date(2022, 10, 4), 2.60),
        (date(2022, 11, 1), 2.85), (date(2022, 12, 6), 3.10),
        (date(2023, 2, 7), 3.35), (date(2023, 3, 7), 3.60),
        (date(2023, 5, 2), 3.85), (date(2023, 6, 6), 4.10),
        (date(2023, 11, 7), 4.35),
        (date(2025, 2, 18), 4.10), (date(2025, 4, 1), 3.85),
    ],
    "NZD": [
        # Reserve Bank of New Zealand Official Cash Rate
        (date(2008, 6, 5), 8.25), (date(2008, 7, 24), 8.00),
        (date(2008, 9, 11), 7.50), (date(2008, 10, 23), 6.50),
        (date(2008, 12, 4), 5.00), (date(2009, 1, 29), 3.50),
        (date(2009, 3, 12), 3.00), (date(2009, 4, 30), 2.50),
        (date(2010, 6, 10), 2.75), (date(2010, 7, 29), 3.00),
        (date(2014, 3, 13), 2.75), (date(2014, 4, 24), 3.00),
        (date(2014, 6, 12), 3.25), (date(2014, 7, 24), 3.50),
        (date(2015, 6, 11), 3.25), (date(2015, 7, 23), 3.00),
        (date(2015, 9, 10), 2.75), (date(2015, 12, 10), 2.50),
        (date(2016, 3, 10), 2.25), (date(2016, 8, 11), 2.00),
        (date(2016, 11, 10), 1.75),
        (date(2019, 5, 8), 1.50), (date(2019, 8, 7), 1.00),
        (date(2020, 3, 16), 0.25),
        (date(2021, 10, 6), 0.50), (date(2021, 11, 24), 0.75),
        (date(2022, 2, 23), 1.00), (date(2022, 4, 13), 1.50),
        (date(2022, 5, 25), 2.00), (date(2022, 7, 13), 2.50),
        (date(2022, 8, 17), 3.00), (date(2022, 10, 5), 3.50),
        (date(2022, 11, 23), 4.25),
        (date(2023, 2, 22), 4.75), (date(2023, 4, 5), 5.25),
        (date(2023, 5, 24), 5.50),
        (date(2024, 8, 14), 5.25), (date(2024, 10, 9), 4.75),
        (date(2024, 11, 27), 4.25),
        (date(2025, 2, 19), 3.75), (date(2025, 4, 9), 3.50),
    ],
    "CHF": [
        # Swiss National Bank Policy Rate (target for 3M LIBOR until 2019, then policy rate)
        (date(2008, 10, 8), 2.50), (date(2008, 11, 6), 1.50),
        (date(2008, 12, 11), 0.50), (date(2009, 3, 12), 0.25),
        (date(2011, 8, 3), 0.00),
        (date(2015, 1, 15), -0.75),  # Negative rates + peg removal
        (date(2022, 6, 16), -0.25), (date(2022, 9, 22), 0.50),
        (date(2022, 12, 15), 1.00), (date(2023, 3, 23), 1.50),
        (date(2023, 6, 22), 1.75),
        (date(2024, 3, 21), 1.50), (date(2024, 6, 20), 1.25),
        (date(2024, 9, 26), 1.00), (date(2024, 12, 12), 0.50),
        (date(2025, 3, 20), 0.25),
    ],
    "CAD": [
        # Bank of Canada Overnight Rate Target
        (date(2008, 1, 22), 4.00), (date(2008, 3, 4), 3.50),
        (date(2008, 4, 22), 3.00), (date(2008, 10, 8), 2.50),
        (date(2008, 10, 21), 2.25), (date(2008, 12, 9), 1.50),
        (date(2009, 1, 20), 1.00), (date(2009, 3, 3), 0.50),
        (date(2009, 4, 21), 0.25),
        (date(2010, 6, 1), 0.50), (date(2010, 7, 20), 0.75),
        (date(2010, 9, 8), 1.00),
        (date(2015, 1, 21), 0.75), (date(2015, 7, 15), 0.50),
        (date(2017, 7, 12), 0.75), (date(2017, 9, 6), 1.00),
        (date(2018, 1, 17), 1.25), (date(2018, 7, 11), 1.50),
        (date(2018, 10, 24), 1.75),
        (date(2020, 3, 4), 1.25), (date(2020, 3, 13), 0.75),
        (date(2020, 3, 27), 0.25),
        (date(2022, 3, 2), 0.50), (date(2022, 4, 13), 1.00),
        (date(2022, 6, 1), 1.50), (date(2022, 7, 13), 2.50),
        (date(2022, 9, 7), 3.25), (date(2022, 10, 26), 3.75),
        (date(2022, 12, 7), 4.25),
        (date(2023, 1, 25), 4.50), (date(2023, 6, 7), 4.75),
        (date(2023, 7, 12), 5.00),
        (date(2024, 6, 5), 4.75), (date(2024, 7, 24), 4.50),
        (date(2024, 9, 4), 4.25), (date(2024, 10, 23), 3.75),
        (date(2024, 12, 11), 3.25),
        (date(2025, 1, 29), 3.00), (date(2025, 3, 12), 2.75),
    ],
}


class ForexCarryTrade(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)
        self.set_end_date(2020, 12, 31)
        self.set_cash(100_000)

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE)

        # ══════════════════════════════════════════════════════════════
        # Universe: 7 major USD pairs = 8 G10 currencies
        # ══════════════════════════════════════════════════════════════
        self.pair_map = {
            # pair -> (base_currency, quote_currency, is_usd_base)
            "EURUSD": ("EUR", "USD", False),
            "USDJPY": ("USD", "JPY", True),
            "GBPUSD": ("GBP", "USD", False),
            "AUDUSD": ("AUD", "USD", False),
            "NZDUSD": ("NZD", "USD", False),
            "USDCHF": ("USD", "CHF", True),
            "USDCAD": ("USD", "CAD", True),
        }
        self.all_pairs = list(self.pair_map.keys())
        self.currencies = ["USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CHF", "CAD"]

        # ══════════════════════════════════════════════════════════════
        # Carry parameters
        # ══════════════════════════════════════════════════════════════
        self.n_long = 3                     # long top 3 highest yield
        self.n_short = 3                    # short bottom 3 lowest yield
        self.momentum_lookback = 63         # v2: 3-month momentum filter (trading days)
        self.vix_threshold = 25.0           # v2: flatten carry when VIX > this (risk-off)

        # ══════════════════════════════════════════════════════════════
        # Barroso-Santa-Clara vol-scaling
        # ══════════════════════════════════════════════════════════════
        self.vol_target = 0.10              # 10% annualized target vol
        self.vol_lookback = 60              # 60-day realized vol window
        self.vol_scale_min = 0.25           # don't go below 25% of base size
        self.vol_scale_max = 1.5            # v2: was 2.0, too aggressive for FX

        # ══════════════════════════════════════════════════════════════
        # Position sizing (IBKR)
        # ══════════════════════════════════════════════════════════════
        self.base_risk_per_position = 1.0 / 6.0  # ~16.7% equity per position (6 active)
        self.max_units = 100_000
        self.ibkr_min_by_currency = {
            "USD": 25_000, "EUR": 20_000, "GBP": 20_000,
            "JPY": 2_500_000, "CHF": 25_000, "AUD": 25_000,
            "NZD": 35_000, "CAD": 25_000,
        }

        # ══════════════════════════════════════════════════════════════
        # Risk management
        # ══════════════════════════════════════════════════════════════
        self.max_dd_pct = 0.15
        self.dd_cooldown_days = 20

        # ══════════════════════════════════════════════════════════════
        # Data structures
        # ══════════════════════════════════════════════════════════════
        self.symbols = {}
        self.current_positions = {}         # pair -> direction (+1 long, -1 short)

        # Drawdown tracking
        self.peak_equity = 100_000
        self.trading_halted = False
        self.halt_until = None
        self.dd_halt_count = 0

        # Carry P&L tracking (separate from price P&L)
        self.cumulative_carry_pnl = 0.0
        self.daily_carry_log = []

        # v2: Per-pair P&L tracking
        self.per_pair_pnl = defaultdict(float)
        self.per_pair_trades = defaultdict(int)

        # Performance tracking
        self.total_rebalances = 0
        self.monthly_returns = []
        self.month_start_equity = 100_000
        self.monthly_pnl = defaultdict(float)
        self.last_month = None
        self.portfolio_daily_returns = []

        # Pre-process rate tables for binary search
        self._rate_dates = {}
        self._rate_values = {}
        for ccy, entries in CENTRAL_BANK_RATES.items():
            self._rate_dates[ccy] = [e[0] for e in entries]
            self._rate_values[ccy] = [e[1] for e in entries]

        # ══════════════════════════════════════════════════════════════
        # Add forex securities
        # ══════════════════════════════════════════════════════════════
        for pair in self.all_pairs:
            forex = self.add_forex(pair, Resolution.DAILY, Market.OANDA)
            self.symbols[pair] = forex.symbol

        # v2: VIX for risk-off filter (CBOE VIX via equity symbol)
        self.add_equity("SPY", Resolution.DAILY)  # needed for scheduling + VIX proxy
        self.vix_symbol = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol

        # Monthly rebalance on first trading day
        self.schedule.on(
            self.date_rules.month_start("EURUSD", 0),
            self.time_rules.at(14, 0),
            self.monthly_rebalance,
        )

        # Daily carry tracking
        self.schedule.on(
            self.date_rules.every_day("EURUSD"),
            self.time_rules.at(16, 0),
            self._track_daily,
        )

        self.set_warm_up(timedelta(days=90))

        self.debug(
            f">>> FX CARRY TRADE v2: {len(self.all_pairs)} pairs, "
            f"long={self.n_long} short={self.n_short}, "
            f"vol_target={self.vol_target:.0%}, vol_cap={self.vol_scale_max:.1f}x, "
            f"mom={self.momentum_lookback}d, vix_off={self.vix_threshold}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Rate lookup (binary search on hardcoded table)
    # ══════════════════════════════════════════════════════════════════════

    def _get_rate(self, currency, as_of_date):
        """Get the central bank rate in effect on a given date."""
        dates = self._rate_dates.get(currency)
        values = self._rate_values.get(currency)
        if not dates:
            return 0.0

        idx = bisect_right(dates, as_of_date) - 1
        if idx < 0:
            return values[0]  # before first entry, use earliest known
        return values[idx]

    def _rank_currencies(self):
        """Rank non-USD currencies by carry score with momentum filter.
        v2: Only include a currency in long/short if momentum confirms.
        - Long candidate: carry > 0 AND 3M FX return > 0 (trending up)
        - Short candidate: carry < 0 AND 3M FX return < 0 (trending down)
        - If momentum disagrees with carry -> FLAT (don't fight the trend)
        """
        today = self.time.date()
        usd_rate = self._get_rate("USD", today)

        scores = {}
        momentum = {}
        for ccy in self.currencies:
            if ccy == "USD":
                continue
            rate = self._get_rate(ccy, today)
            carry = rate - usd_rate
            scores[ccy] = carry

            # v2: Compute 3-month FX momentum for this currency
            # Find the pair that represents this currency vs USD
            pair_for_ccy = None
            is_usd_base = False
            for pair, (base, quote, usd_b) in self.pair_map.items():
                non_usd = base if quote == "USD" else quote
                if non_usd == ccy:
                    pair_for_ccy = pair
                    is_usd_base = usd_b
                    break

            mom_3m = 0.0
            if pair_for_ccy and pair_for_ccy in self.symbols:
                hist = self.history(self.symbols[pair_for_ccy], self.momentum_lookback + 5, Resolution.DAILY)
                if hist is not None and not hist.empty:
                    try:
                        closes = hist["close"].values
                        if len(closes) >= self.momentum_lookback:
                            ret = (closes[-1] / closes[-self.momentum_lookback]) - 1.0
                            # For USD-base pairs (USDJPY), positive return = USD strengthening = ccy weakening
                            mom_3m = -ret if is_usd_base else ret
                    except Exception:
                        pass
            momentum[ccy] = mom_3m

        # Sort by carry score (highest first)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # v2: Filter by momentum agreement
        long_candidates = []
        short_candidates = []
        for ccy, carry in ranked:
            mom = momentum.get(ccy, 0.0)
            if carry >= 0 and mom > 0:
                long_candidates.append(ccy)
            elif carry < 0 and mom < 0:
                short_candidates.append(ccy)
            # else: momentum disagrees with carry -> skip (FLAT)

        return ranked, usd_rate, scores, long_candidates, short_candidates, momentum

    # ══════════════════════════════════════════════════════════════════════
    # IBKR helpers (from zone bounce / pairs trading, battle-tested)
    # ══════════════════════════════════════════════════════════════════════

    def _is_safe_hour(self):
        """Prevent MOO errors on weekends and market transition hours."""
        dow = self.time.weekday()  # Monday=0, Sunday=6
        if dow in (5, 6):
            return False
        if self.time.hour in (17, 22, 23, 0):
            return False
        return True

    def _get_ibkr_min_units(self, pair):
        base = pair[:3]
        return self.ibkr_min_by_currency.get(base, 25_000)

    def _get_quote_currency_rate(self, pair):
        """Convert quote currency to USD for P&L calculation."""
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
    # Vol-scaling (Barroso-Santa-Clara 2015)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_vol_scale(self):
        """Compute position scale factor based on realized portfolio volatility."""
        if len(self.portfolio_daily_returns) < 30:
            return 1.0  # not enough history, use base size

        recent = self.portfolio_daily_returns[-self.vol_lookback:]
        realized_vol = np.std(recent) * np.sqrt(252)

        if realized_vol < 0.001:
            return self.vol_scale_max  # very low vol, scale up

        scale = self.vol_target / realized_vol
        return max(self.vol_scale_min, min(self.vol_scale_max, scale))

    # ══════════════════════════════════════════════════════════════════════
    # Daily tracking (portfolio returns for vol-scaling + carry P&L)
    # ══════════════════════════════════════════════════════════════════════

    def _track_daily(self):
        if self.is_warming_up:
            return

        equity = self.portfolio.total_portfolio_value

        # Track daily return for vol-scaling
        if hasattr(self, '_prev_equity') and self._prev_equity > 0:
            daily_ret = (equity - self._prev_equity) / self._prev_equity
            self.portfolio_daily_returns.append(daily_ret)
            # Keep only recent history
            if len(self.portfolio_daily_returns) > self.vol_lookback * 2:
                self.portfolio_daily_returns = self.portfolio_daily_returns[-self.vol_lookback * 2:]
        self._prev_equity = equity

        # Track carry income (not reflected in QC equity curve)
        today = self.time.date()
        usd_rate = self._get_rate("USD", today)
        daily_carry = 0.0

        for pair, direction in self.current_positions.items():
            if not self.portfolio[self.symbols[pair]].invested:
                continue

            base_ccy, quote_ccy, is_usd_base = self.pair_map[pair]
            notional = abs(self.portfolio[self.symbols[pair]].quantity)
            quote_rate = self._get_quote_currency_rate(pair)
            notional_usd = notional * self.securities[self.symbols[pair]].price * quote_rate

            # Determine which currency we're long and which we're short
            if direction == 1:  # long the pair
                long_ccy = base_ccy
                short_ccy = quote_ccy
            else:  # short the pair
                long_ccy = quote_ccy
                short_ccy = base_ccy

            long_rate = self._get_rate(long_ccy, today)
            short_rate = self._get_rate(short_ccy, today)
            carry = (long_rate - short_rate) / 100.0 / 365.0 * notional_usd
            daily_carry += carry

        self.cumulative_carry_pnl += daily_carry

        # Monthly tracking
        current_month = f"{self.time.year}-{self.time.month:02d}"
        if self.last_month is not None and current_month != self.last_month:
            month_ret = (equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)
            self.monthly_pnl[self.last_month] = equity - self.month_start_equity
            self.month_start_equity = equity
        self.last_month = current_month

    # ══════════════════════════════════════════════════════════════════════
    # Monthly rebalance
    # ══════════════════════════════════════════════════════════════════════

    def monthly_rebalance(self):
        if self.is_warming_up:
            return

        if not self._is_safe_hour():
            return

        equity = self.portfolio.total_portfolio_value

        # ── Drawdown protection ──
        if equity > self.peak_equity:
            self.peak_equity = equity

        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        if self.trading_halted and self.halt_until is not None:
            if self.time >= self.halt_until:
                self.trading_halted = False
                self.halt_until = None
                self.peak_equity = equity
                self.debug(f"TRADING RESUMED: equity=${equity:,.0f}, peak reset")

        if current_dd > self.max_dd_pct and not self.trading_halted:
            self.trading_halted = True
            self.halt_until = self.time + timedelta(days=self.dd_cooldown_days)
            self.dd_halt_count += 1
            for pair in list(self.current_positions.keys()):
                self.liquidate(self.symbols[pair])
            self.current_positions.clear()
            self.debug(
                f"DD HALT: {current_dd:.1%} drawdown. "
                f"Equity=${equity:,.0f} Peak=${self.peak_equity:,.0f}"
            )
            return

        if self.trading_halted:
            return

        self.total_rebalances += 1

        # ── v2: VIX risk-off filter ──
        vix_value = 0.0
        try:
            if self.vix_symbol in self.securities and self.securities[self.vix_symbol].price > 0:
                vix_value = self.securities[self.vix_symbol].price
        except Exception:
            pass

        if vix_value > self.vix_threshold:
            # Risk-off: flatten everything
            for pair in list(self.current_positions.keys()):
                self._record_pair_pnl(pair)
                self.liquidate(self.symbols[pair])
            self.current_positions.clear()
            self.debug(
                f"REBALANCE #{self.total_rebalances} | VIX={vix_value:.1f} > {self.vix_threshold} | "
                f"RISK-OFF: all positions closed | carry_pnl=${self.cumulative_carry_pnl:,.0f}"
            )
            return

        # ── Rank currencies by carry + momentum ──
        ranked, usd_rate, scores, long_candidates, short_candidates, momentum = self._rank_currencies()

        # Take top N from filtered candidates
        long_currencies = set(long_candidates[:self.n_long])
        short_currencies = set(short_candidates[:self.n_short])

        # ── Determine target positions ──
        target_positions = {}

        for pair, (base_ccy, quote_ccy, is_usd_base) in self.pair_map.items():
            non_usd = base_ccy if quote_ccy == "USD" else quote_ccy

            if non_usd in long_currencies:
                if is_usd_base:
                    target_positions[pair] = -1
                else:
                    target_positions[pair] = 1

            elif non_usd in short_currencies:
                if is_usd_base:
                    target_positions[pair] = 1
                else:
                    target_positions[pair] = -1

            # else: flat (filtered out by momentum or middle currency)

        # ── Vol-scaling ──
        vol_scale = self._compute_vol_scale()

        # ── v2: Force full rebalance — close everything, re-enter at target sizes ──
        # This ensures positions are always at correct weight (v1 barely traded)
        for pair in list(self.current_positions.keys()):
            target_dir = target_positions.get(pair, 0)
            current_dir = self.current_positions[pair]

            if target_dir == 0 or target_dir != current_dir:
                # Direction changed or no longer a target — close
                self._record_pair_pnl(pair)
                self.liquidate(self.symbols[pair])
                del self.current_positions[pair]
            else:
                # Same direction — force resize to current target weight
                self._force_resize(pair, target_dir, equity, vol_scale)

        # Open new positions
        for pair, direction in target_positions.items():
            if pair not in self.current_positions:
                self._enter_position(pair, direction, equity, vol_scale)

        # ── Log ──
        n_active = len(self.current_positions)
        self.debug(
            f"REBALANCE #{self.total_rebalances} | USD={usd_rate:.2f}% | VIX={vix_value:.1f} | "
            f"vol_scale={vol_scale:.2f} | positions={n_active} | carry_pnl=${self.cumulative_carry_pnl:,.0f}"
        )
        for ccy, score in ranked:
            mom = momentum.get(ccy, 0.0)
            position = "LONG" if ccy in long_currencies else "SHORT" if ccy in short_currencies else "FLAT"
            rate = self._get_rate(ccy, self.time.date())
            self.debug(f"  {ccy}: rate={rate:.2f}% carry={score:+.2f}% mom3m={mom:+.1%} -> {position}")

    def _record_pair_pnl(self, pair):
        """Record unrealized P&L before closing a position."""
        if pair in self.symbols and self.portfolio[self.symbols[pair]].invested:
            pnl = self.portfolio[self.symbols[pair]].unrealized_profit
            self.per_pair_pnl[pair] += pnl
            self.per_pair_trades[pair] += 1

    def _force_resize(self, pair, direction, equity, vol_scale):
        """v2: Force position to exact target size (replaces 20% threshold)."""
        current_qty = self.portfolio[self.symbols[pair]].quantity
        if current_qty == 0:
            self._enter_position(pair, direction, equity, vol_scale)
            return

        price = self.securities[self.symbols[pair]].price
        if price <= 0:
            return

        quote_rate = self._get_quote_currency_rate(pair)
        target_notional = equity * self.base_risk_per_position * vol_scale
        target_units = int(target_notional / (price * quote_rate))
        min_units = self._get_ibkr_min_units(pair)
        target_units = max(min_units, min(self.max_units, (target_units // 1000) * 1000))

        target_qty = target_units * direction
        diff = target_qty - current_qty

        # v2: Always resize if diff > min_units (no 20% threshold)
        if abs(diff) >= min_units:
            self.market_order(self.symbols[pair], diff)

    def _enter_position(self, pair, direction, equity, vol_scale):
        """Enter a new carry position."""
        price = self.securities[self.symbols[pair]].price
        if price <= 0:
            return

        quote_rate = self._get_quote_currency_rate(pair)
        target_notional = equity * self.base_risk_per_position * vol_scale
        units = int(target_notional / (price * quote_rate))

        min_units = self._get_ibkr_min_units(pair)
        units = max(min_units, min(self.max_units, (units // 1000) * 1000))

        if units < min_units:
            return

        qty = units * direction
        self.market_order(self.symbols[pair], qty)
        self.current_positions[pair] = direction

    # ══════════════════════════════════════════════════════════════════════
    # OnData
    # ══════════════════════════════════════════════════════════════════════

    def on_data(self, data):
        # Deploy rebalance on first data if no positions
        if not self.is_warming_up and not self.current_positions and self.total_rebalances == 0:
            self.monthly_rebalance()

    # ══════════════════════════════════════════════════════════════════════
    # End of algorithm
    # ══════════════════════════════════════════════════════════════════════

    def on_end_of_algorithm(self):
        equity = self.portfolio.total_portfolio_value
        ret_pct = (equity - 100_000) / 100_000

        # Record final P&L for open positions
        for pair in list(self.current_positions.keys()):
            self._record_pair_pnl(pair)

        if self.last_month is not None:
            month_ret = (equity - self.month_start_equity) / self.month_start_equity
            self.monthly_returns.append(month_ret)

        self.debug("=" * 70)
        self.debug("FX CARRY TRADE v2 - FINAL RESULTS")
        self.debug("=" * 70)
        self.debug(f"PRICE-ONLY RETURN: {ret_pct:.2%} (${equity:,.0f})")
        self.debug(f"CARRY INCOME: ${self.cumulative_carry_pnl:,.0f} ({self.cumulative_carry_pnl/1000:.1f}% of $100K)")
        self.debug(f"TOTAL RETURN (price+carry): {ret_pct + self.cumulative_carry_pnl/100_000:.2%}")
        self.debug(f"Rebalances: {self.total_rebalances} | DD Halts: {self.dd_halt_count}")

        if self.monthly_returns:
            rets = np.array(self.monthly_returns)
            avg = np.mean(rets)
            std = np.std(rets)
            sharpe = avg / std * np.sqrt(12) if std > 0 else 0
            self.debug(
                f"MONTHLY: Avg={avg:.2%} Std={std:.2%} Sharpe={sharpe:.2f} "
                f"Best={np.max(rets):.2%} Worst={np.min(rets):.2%} "
                f"Win={np.sum(rets > 0)} Loss={np.sum(rets <= 0)}"
            )

        # v2: Per-pair P&L breakdown
        if self.per_pair_pnl:
            self.debug("\nPER-PAIR P&L:")
            for pair, pnl in sorted(self.per_pair_pnl.items(), key=lambda x: x[1], reverse=True):
                trades = self.per_pair_trades.get(pair, 0)
                self.debug(f"  {pair:<10s}: PnL=${pnl:>+8,.0f} rotations={trades}")

        # Final rate snapshot
        today = self.time.date()
        self.debug(f"\nFINAL RATES (as of {today}):")
        for ccy in sorted(self.currencies):
            rate = self._get_rate(ccy, today)
            self.debug(f"  {ccy}: {rate:.2f}%")

        if self.monthly_pnl:
            pnl_str = " | ".join(f"{k}:${v:,.0f}" for k, v in sorted(self.monthly_pnl.items()))
            self.debug(f"\nMONTHLY PNL: {pnl_str}")
