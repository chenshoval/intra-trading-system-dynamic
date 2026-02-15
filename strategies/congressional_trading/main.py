"""Congressional Copy-Trading Strategy — DEBUG VERSION"""

from AlgorithmImports import *
from QuantConnect.DataSource import *


class CongressionalTradingAlgorithm(QCAlgorithm):

    def initialize(self):
        self.debug(">>> INIT START")
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        self.holding_period = 30
        self.min_trade_value = 1000       # was 50000 — amounts are lower bounds of ranges
        self.max_positions = 15
        self.position_size_pct = 0.05     # 5% per position
        self.max_total_exposure = 0.90    # was 0.80

        # Set to None to accept ALL politicians, or keep a set to filter
        self.top_performers = None  # None = all politicians
        # Uncomment below to filter specific politicians:
        # self.top_performers = {
        #     "Nancy Pelosi", "Dan Crenshaw", "Josh Gottheimer",
        #     "Michael McCaul", "Tommy Tuberville", "Ro Khanna",
        #     "Mark Green", "Virginia Foxx", "Marjorie Taylor Greene",
        # }

        self.debug(">>> ADDING UNIVERSE")
        self.add_universe(QuiverQuantCongressUniverse, "QuiverQuantCongressUniverse", self.congress_filter)
        self.debug(">>> UNIVERSE ADDED OK")

        self.position_exit_dates = {}
        self.pending_buys = []
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

        self.debug(">>> INIT COMPLETE")

    def congress_filter(self, data):
        self.debug(f">>> FILTER CALLED type={type(data).__name__}")
        symbols = []

        for i, point in enumerate(data):
            if i < 5:
                self.debug(f">>> ROW {i}: sym={point.symbol} rep={point.representative} "
                           f"txn={point.transaction} amt={point.amount}")

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

            self.debug(f">>> BUY: {point.representative} -> {point.symbol} ${point.amount}")

            # Top performers filter (skip if None = accept all)
            if self.top_performers is not None and point.representative not in self.top_performers:
                continue

            if point.amount is None or point.amount < self.min_trade_value:
                continue

            self.pending_buys.append({
                "symbol": point.symbol,
                "politician": point.representative,
                "amount": point.amount,
            })
            self.debug(f">>> QUEUED: {point.representative} -> {point.symbol}")
            symbols.append(point.symbol)

        self.debug(f">>> FILTER DONE returning {len(symbols)}")
        return symbols

    def on_data(self, data):
        pass

    def execute_pending_trades(self):
        if not self.pending_buys:
            return
        self.debug(f">>> EXECUTING {len(self.pending_buys)} pending")

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
            self.debug(f">>> BOUGHT {symbol.value}: {quantity} @ ${price:.2f}")
            executed.append(trade)

        for t in executed:
            if t in self.pending_buys:
                self.pending_buys.remove(t)

    def check_exits(self):
        to_remove = []
        for symbol, exit_date in self.position_exit_dates.items():
            if self.time >= exit_date:
                if self.portfolio[symbol].invested:
                    pnl = self.portfolio[symbol].unrealized_profit
                    self.liquidate(symbol)
                    self.debug(f">>> EXIT {symbol.value} PnL=${pnl:,.2f}")
                to_remove.append(symbol)
        for s in to_remove:
            del self.position_exit_dates[s]

    def on_end_of_algorithm(self):
        self.debug(f">>> DONE Return={self.portfolio.total_profit / 100_000:.2%}")
