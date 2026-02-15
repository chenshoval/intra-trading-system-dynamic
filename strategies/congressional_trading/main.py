"""Congressional Copy-Trading Strategy (Hypothesis 1)

QuantConnect algorithm that trades based on congressional stock disclosures.
Uses Quiver Quantitative dataset for STOCK Act filings.

Signal: When high-performing politicians disclose a BUY, buy on disclosure date +1.
Filters: Trade size > $50K, top performers only, buys only initially.
Hold periods to test: 7, 14, 30, 60 days.

This is the FASTEST hypothesis to validate — no ML model needed, pure event-driven.
"""

# ─── QuantConnect Algorithm ─────────────────────────────────
# This file runs inside QuantConnect's LEAN engine.
# Copy/paste to QuantConnect web IDE or use local LEAN CLI.
# ─────────────────────────────────────────────────────────────

from AlgorithmImports import *
from QuantConnect.DataSource import *


class CongressionalTradingAlgorithm(QCAlgorithm):
    """Buy stocks that top-performing congress members disclose buying."""

    def initialize(self):
        # ── Backtest Settings ──
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        # ── Parameters (tune these) ──
        self.holding_period = self.get_parameter("holding_period", 30)  # days
        self.min_trade_value = self.get_parameter("min_trade_value", 50_000)  # $
        self.max_positions = self.get_parameter("max_positions", 10)
        self.position_size_pct = self.get_parameter("position_size_pct", 0.05)  # 5% per position
        self.max_total_exposure = self.get_parameter("max_total_exposure", 0.80)  # 80% max

        # ── Top Performers (historically good track records) ──
        # These can be updated based on research in notebooks
        self.top_performers = {
            "Nancy Pelosi", "Dan Crenshaw", "Josh Gottheimer",
            "Michael McCaul", "Tommy Tuberville", "Ro Khanna",
            "Mark Green", "Virginia Foxx", "Marjorie Taylor Greene",
        }

        # ── Universe — start with liquid large caps ──
        self.universe_settings.resolution = Resolution.DAILY
        self.add_universe(self.coarse_selection)

        # ── Congressional Trading Data ──
        # Quiver Quantitative provides STOCK Act filing data
        self.quiver_congress = self.add_data(
            QuiverCongressTrading, "QuiverCongressTrading"
        ).symbol

        # ── Track positions and their exit dates ──
        self.position_exit_dates = {}  # symbol -> exit_date
        self.pending_buys = []  # list of (symbol, disclosure_date, politician, amount)

        # ── Benchmark ──
        self.set_benchmark("SPY")

        # ── Scheduled Events ──
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.execute_pending_trades,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 10),
            self.check_exits,
        )

        # ── Logging ──
        self.log(f"Congressional Trading initialized: "
                 f"hold={self.holding_period}d, "
                 f"min_value=${self.min_trade_value:,}, "
                 f"max_positions={self.max_positions}")

    def coarse_selection(self, coarse):
        """Universe selection — liquid stocks only."""
        return [
            c.symbol for c in coarse
            if c.has_fundamental_data
            and c.price > 10
            and c.dollar_volume > 10_000_000
        ]

    def on_data(self, data: Slice):
        """Process incoming data — check for congressional trades."""
        # Check for Quiver Congressional data
        if data.contains_key(self.quiver_congress):
            congress_data = data[self.quiver_congress]
            self.process_congressional_filings(congress_data)

    def process_congressional_filings(self, filings):
        """Filter and queue congressional buy signals."""
        for filing in filings:
            # ── Filters ──
            # 1. Buy transactions only
            if filing.transaction != "Purchase":
                continue

            # 2. Top performers only
            if filing.representative not in self.top_performers:
                continue

            # 3. Minimum trade size
            if filing.amount < self.min_trade_value:
                continue

            # 4. Not already in position
            ticker = filing.ticker
            symbol = self.add_equity(ticker, Resolution.DAILY).symbol
            if symbol in self.position_exit_dates:
                continue

            # 5. Position limit
            if len(self.position_exit_dates) >= self.max_positions:
                self.log(f"SKIP {ticker}: max positions ({self.max_positions}) reached")
                continue

            # ── Queue the trade ──
            self.pending_buys.append({
                "symbol": symbol,
                "ticker": ticker,
                "politician": filing.representative,
                "amount": filing.amount,
                "disclosure_date": self.time,
                "filing_date": filing.report_date,
            })

            self.log(
                f"SIGNAL: {filing.representative} bought {ticker} "
                f"(${filing.amount:,.0f}) — filed {filing.report_date}"
            )

    def execute_pending_trades(self):
        """Execute queued buy orders (next day after signal)."""
        if not self.pending_buys:
            return

        # Check total exposure
        total_value = self.portfolio.total_portfolio_value
        current_exposure = sum(
            abs(h.holdings_value) for h in self.portfolio.values()
            if h.invested
        ) / total_value if total_value > 0 else 0

        executed = []
        for trade in self.pending_buys:
            symbol = trade["symbol"]

            # Skip if we'd exceed max exposure
            if current_exposure + self.position_size_pct > self.max_total_exposure:
                self.log(f"SKIP {trade['ticker']}: exposure limit "
                        f"({current_exposure:.1%} + {self.position_size_pct:.1%} > "
                        f"{self.max_total_exposure:.1%})")
                continue

            # Calculate position size
            allocation = total_value * self.position_size_pct
            price = self.securities[symbol].price
            if price <= 0:
                continue

            quantity = int(allocation / price)
            if quantity <= 0:
                continue

            # Execute
            self.market_order(symbol, quantity)
            exit_date = self.time + timedelta(days=self.holding_period)
            self.position_exit_dates[symbol] = exit_date
            current_exposure += self.position_size_pct

            self.log(
                f"BUY {trade['ticker']}: {quantity} shares @ ${price:.2f} "
                f"(${allocation:,.0f}) — exit by {exit_date.strftime('%Y-%m-%d')} "
                f"— signal from {trade['politician']}"
            )
            executed.append(trade)

        # Remove executed trades from pending
        for t in executed:
            self.pending_buys.remove(t)

    def check_exits(self):
        """Liquidate positions that have reached their holding period."""
        to_remove = []
        for symbol, exit_date in self.position_exit_dates.items():
            if self.time >= exit_date:
                if self.portfolio[symbol].invested:
                    quantity = self.portfolio[symbol].quantity
                    price = self.securities[symbol].price
                    pnl = self.portfolio[symbol].unrealized_profit
                    pnl_pct = self.portfolio[symbol].unrealized_profit_percent

                    self.liquidate(symbol)
                    self.log(
                        f"EXIT {symbol.value}: {quantity} shares @ ${price:.2f} "
                        f"— P&L: ${pnl:,.2f} ({pnl_pct:.2%})"
                    )
                to_remove.append(symbol)

        for s in to_remove:
            del self.position_exit_dates[s]

    def on_end_of_algorithm(self):
        """Log final summary."""
        self.log(f"\n{'='*50}")
        self.log(f"FINAL RESULTS")
        self.log(f"{'='*50}")
        self.log(f"Total Return: {self.portfolio.total_profit / 100_000:.2%}")
        self.log(f"Final Value: ${self.portfolio.total_portfolio_value:,.2f}")
        self.log(f"{'='*50}\n")
