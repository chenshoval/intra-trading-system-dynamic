"""Congressional Copy-Trading Strategy — v2 (cleaned up)"""

from AlgorithmImports import *
from QuantConnect.DataSource import *


class CongressionalTradingAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        self.holding_period = 30
        self.min_trade_value = 1000
        self.max_positions = 15
        self.position_size_pct = 0.05
        self.max_total_exposure = 0.90
        self.max_signal_age = 3  # only act on signals from last 3 days

        self.top_performers = None  # None = all politicians

        self.add_universe(QuiverQuantCongressUniverse, "QuiverQuantCongressUniverse", self.congress_filter)

        self.position_exit_dates = {}
        self.pending_buys = []
        self.total_signals = 0
        self.total_trades = 0
        self.set_benchmark("SPY")

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

    def congress_filter(self, data):
        symbols = []
        for point in data:
            is_buy = False
            try:
                is_buy = (point.transaction == OrderDirection.BUY)
            except:
                pass
            if not is_buy:
                txn_str = str(point.transaction).lower()
                is_buy = txn_str in ["buy", "purchase", "0"]

            if not is_buy:
                continue

            if self.top_performers is not None and point.representative not in self.top_performers:
                continue

            if point.amount is None or point.amount < self.min_trade_value:
                continue

            self.total_signals += 1
            self.pending_buys.append({
                "symbol": point.symbol,
                "politician": point.representative,
                "amount": point.amount,
                "signal_date": self.time,
            })
            symbols.append(point.symbol)

        return symbols

    def execute_pending_trades(self):
        if not self.pending_buys:
            return

        # Remove stale signals (older than max_signal_age days)
        cutoff = self.time - timedelta(days=self.max_signal_age)
        self.pending_buys = [t for t in self.pending_buys if t["signal_date"] >= cutoff]

        if not self.pending_buys:
            return

        total_value = self.portfolio.total_portfolio_value
        if total_value <= 0:
            return

        current_exposure = sum(
            abs(h.holdings_value) for h in self.portfolio.values()
            if h.invested
        ) / total_value

        executed = []
        for trade in self.pending_buys:
            symbol = trade["symbol"]

            if symbol in self.position_exit_dates:
                executed.append(trade)
                continue
            if len(self.position_exit_dates) >= self.max_positions:
                continue
            if current_exposure + self.position_size_pct > self.max_total_exposure:
                continue
            if symbol not in self.securities or self.securities[symbol].price <= 0:
                continue

            allocation = total_value * self.position_size_pct
            price = self.securities[symbol].price
            quantity = int(allocation / price)
            if quantity <= 0:
                continue

            self.market_order(symbol, quantity)
            exit_date = self.time + timedelta(days=self.holding_period)
            self.position_exit_dates[symbol] = exit_date
            current_exposure += self.position_size_pct
            self.total_trades += 1
            executed.append(trade)

        for t in executed:
            if t in self.pending_buys:
                self.pending_buys.remove(t)

    def check_exits(self):
        to_remove = []
        for symbol, exit_date in self.position_exit_dates.items():
            if self.time >= exit_date:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                to_remove.append(symbol)
        for s in to_remove:
            del self.position_exit_dates[s]

    def on_data(self, data):
        pass

    def on_end_of_algorithm(self):
        self.debug(f"RESULTS: Return={self.portfolio.total_profit / 100_000:.2%} "
                   f"Signals={self.total_signals} Trades={self.total_trades} "
                   f"Final=${self.portfolio.total_portfolio_value:,.0f}")
