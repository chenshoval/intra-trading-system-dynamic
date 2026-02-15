"""Combined Signal Strategy (Hypothesis 3)

Combines congressional copy-trading signal with directional classifier.
When a congressional BUY aligns with a HIGH CONFIDENCE directional prediction,
the combined signal should have higher precision and lower drawdown.

This is a placeholder — implement after validating Hypotheses 1 and 2 individually.
"""

from AlgorithmImports import *


class CombinedSignalAlgorithm(QCAlgorithm):
    """Combined congressional + directional classifier signals.

    Logic:
    - Congressional BUY signal fires → check if directional model agrees
    - If both agree AND confidence > threshold → trade with larger position
    - If only one signal → trade with smaller position (or skip)

    TODO: Implement after validating H1 and H2 individually.
    """

    def initialize(self):
        self.set_start_date(2021, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)
        self.set_benchmark("SPY")

        self.log("Combined Signal Strategy — PLACEHOLDER")
        self.log("Implement after validating congressional trading and directional classifier individually.")
        self.log("Key questions to answer first:")
        self.log("  1. Does congressional signal have edge? (H1)")
        self.log("  2. Does directional classifier have edge? (H2)")
        self.log("  3. Are the signals uncorrelated?")
        self.log("  4. Does combining reduce drawdown?")

    def on_data(self, data: Slice):
        pass
