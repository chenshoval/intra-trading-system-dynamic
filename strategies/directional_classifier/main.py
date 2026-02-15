"""Global Directional Classifier Strategy (Hypothesis 2)

QuantConnect algorithm that uses a pre-trained LightGBM model to predict
stock direction (UP/DOWN) and trades based on per-stock confidence thresholds.

Signal: LightGBM binary classifier output.
Decision rule: Only trade when confidence > optimized threshold τ* per stock.
Features: Multi-timeframe returns, cross-stock features, technical indicators.

This strategy requires a pre-trained model. Train using:
- notebooks/03_model_training/lightgbm_directional.ipynb (local)
- training/train_lightgbm.py (Azure)

Upload the trained model to QC ObjectStore before running.
"""

from AlgorithmImports import *
import pickle
import numpy as np


class DirectionalClassifierAlgorithm(QCAlgorithm):
    """Trade based on LightGBM directional predictions with per-stock thresholds."""

    def initialize(self):
        # ── Backtest Settings ──
        self.set_start_date(2021, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        # ── Parameters ──
        self.lookback = self.get_parameter("lookback", 60)  # bars for feature computation
        self.default_threshold = self.get_parameter("default_threshold", 0.6)
        self.max_positions = self.get_parameter("max_positions", 10)
        self.position_size_pct = self.get_parameter("position_size_pct", 0.05)
        self.rebalance_frequency = self.get_parameter("rebalance_frequency", 5)  # days

        # ── Universe — diverse sectors ──
        # Start with 19-stock subset across sectors, expand later
        self.target_tickers = [
            # Tech
            "AAPL", "MSFT", "NVDA", "GOOGL",
            # Finance
            "JPM", "BAC", "GS",
            # Healthcare
            "JNJ", "UNH", "PFE",
            # Consumer
            "AMZN", "WMT", "KO",
            # Energy
            "XOM", "CVX",
            # Industrial
            "CAT", "BA",
            # Communication
            "META", "DIS",
        ]

        # Add equities
        self.symbols = {}
        for ticker in self.target_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            self.symbols[ticker] = equity.symbol

            # Set up indicators we need for features
            self._setup_indicators(ticker, equity.symbol)

        # ── Load Model ──
        self.model = None
        self.feature_names = None
        self._load_model()

        # ── Per-Stock Thresholds ──
        # These should be optimized via walk-forward validation
        # Default: use a single threshold, override per-stock as you tune
        self.thresholds = {ticker: self.default_threshold for ticker in self.target_tickers}

        # ── Rebalance Schedule ──
        self.days_since_rebalance = 0
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.rebalance,
        )

        # ── Benchmark ──
        self.set_benchmark("SPY")

        self.log(f"Directional Classifier initialized: "
                 f"{len(self.target_tickers)} stocks, "
                 f"threshold={self.default_threshold}, "
                 f"max_positions={self.max_positions}")

    def _setup_indicators(self, ticker: str, symbol):
        """Register technical indicators for feature computation."""
        # These are computed automatically by LEAN
        self._rsi = {}
        self._macd = {}
        self._bb = {}
        self._atr = {}

        self._rsi[ticker] = self.rsi(symbol, 14)
        self._macd[ticker] = self.macd(symbol, 12, 26, 9)
        self._bb[ticker] = self.bb(symbol, 20, 2)
        self._atr[ticker] = self.atr(symbol, 14)

    def _load_model(self):
        """Load pre-trained LightGBM model from ObjectStore."""
        model_key = "directional_classifier/model.pkl"

        if self.object_store.contains_key(model_key):
            model_bytes = self.object_store.read_bytes(model_key)
            self.model = pickle.loads(model_bytes)

            # Load feature names
            features_key = "directional_classifier/feature_names.json"
            if self.object_store.contains_key(features_key):
                import json
                self.feature_names = json.loads(
                    self.object_store.read(features_key)
                )

            self.log(f"Model loaded from ObjectStore: {type(self.model).__name__}")
        else:
            self.log("WARNING: No model found in ObjectStore. "
                     "Upload model first using QC ObjectStore API. "
                     "Running in signal-only mode (logging predictions without trading).")

    def _compute_features(self, ticker: str) -> dict | None:
        """Compute features for a single stock."""
        symbol = self.symbols[ticker]
        history = self.history(symbol, self.lookback, Resolution.DAILY)

        if history.empty or len(history) < self.lookback:
            return None

        close = history["close"].values
        high = history["high"].values
        low = history["low"].values
        volume = history["volume"].values

        features = {}

        # Multi-timeframe returns
        for period in [1, 5, 10, 21]:
            if len(close) > period:
                features[f"return_{period}"] = (close[-1] / close[-period - 1]) - 1
            else:
                features[f"return_{period}"] = 0

        # Technical indicators (from LEAN)
        if self._rsi[ticker].is_ready:
            features["rsi_14"] = self._rsi[ticker].current.value
        if self._macd[ticker].is_ready:
            features["macd_line"] = self._macd[ticker].current.value
            features["macd_signal"] = self._macd[ticker].signal.current.value
            features["macd_histogram"] = self._macd[ticker].histogram.current.value
        if self._bb[ticker].is_ready:
            features["bb_pct"] = (
                (close[-1] - self._bb[ticker].lower_band.current.value) /
                max(self._bb[ticker].upper_band.current.value -
                    self._bb[ticker].lower_band.current.value, 1e-8)
            )
        if self._atr[ticker].is_ready:
            features["atr_14"] = self._atr[ticker].current.value

        # Volume features
        if len(volume) >= 20:
            features["volume_ratio"] = volume[-1] / max(np.mean(volume[-20:]), 1)

        # Price structure
        if len(close) >= 20:
            sma20 = np.mean(close[-20:])
            features["close_to_sma20"] = close[-1] / max(sma20, 1e-8) - 1
        if len(close) >= 60:
            sma60 = np.mean(close[-60:])
            features["close_to_sma60"] = close[-1] / max(sma60, 1e-8) - 1

        # Candle features
        body = abs(close[-1] - history["open"].values[-1])
        full_range = max(high[-1] - low[-1], 1e-8)
        features["body_size"] = body / full_range

        return features

    def rebalance(self):
        """Periodic rebalance — predict directions and trade."""
        self.days_since_rebalance += 1
        if self.days_since_rebalance < self.rebalance_frequency:
            return
        self.days_since_rebalance = 0

        if self.model is None:
            return

        # ── Predict for all stocks ──
        signals = {}  # ticker -> (direction, confidence)

        for ticker in self.target_tickers:
            features = self._compute_features(ticker)
            if features is None:
                continue

            try:
                # Build feature vector in correct order
                if self.feature_names:
                    feature_vector = [features.get(f, 0) for f in self.feature_names]
                else:
                    feature_vector = list(features.values())

                X = np.array([feature_vector])
                prob = self.model.predict_proba(X)[0]
                confidence = max(prob)
                direction = 1 if prob[1] > prob[0] else -1

                signals[ticker] = (direction, confidence)

            except Exception as e:
                self.log(f"Prediction error for {ticker}: {e}")

        # ── Filter by confidence threshold ──
        qualified = {
            ticker: (direction, confidence)
            for ticker, (direction, confidence) in signals.items()
            if confidence >= self.thresholds.get(ticker, self.default_threshold)
        }

        # ── Sort by confidence (highest first) ──
        ranked = sorted(qualified.items(), key=lambda x: x[1][1], reverse=True)

        # ── Execute trades ──
        # First: close positions that no longer qualify
        for ticker in list(self.symbols.keys()):
            symbol = self.symbols[ticker]
            if self.portfolio[symbol].invested:
                if ticker not in qualified:
                    self.liquidate(symbol)
                    self.log(f"EXIT {ticker}: signal lost or below threshold")

        # Then: open new positions (up to max)
        current_positions = sum(
            1 for s in self.symbols.values() if self.portfolio[s].invested
        )

        for ticker, (direction, confidence) in ranked:
            if current_positions >= self.max_positions:
                break

            symbol = self.symbols[ticker]
            if self.portfolio[symbol].invested:
                continue

            # Only take long positions for now (direction == 1)
            if direction != 1:
                continue

            # Position size
            allocation = self.portfolio.total_portfolio_value * self.position_size_pct
            price = self.securities[symbol].price
            if price <= 0:
                continue

            quantity = int(allocation / price)
            if quantity <= 0:
                continue

            self.market_order(symbol, quantity)
            current_positions += 1

            self.log(
                f"BUY {ticker}: {quantity} shares @ ${price:.2f} "
                f"(confidence={confidence:.3f}, threshold={self.thresholds[ticker]:.3f})"
            )

    def on_end_of_algorithm(self):
        """Log final summary."""
        self.log(f"\n{'='*50}")
        self.log(f"DIRECTIONAL CLASSIFIER RESULTS")
        self.log(f"{'='*50}")
        self.log(f"Total Return: {self.portfolio.total_profit / 100_000:.2%}")
        self.log(f"Final Value: ${self.portfolio.total_portfolio_value:,.2f}")
        self.log(f"{'='*50}\n")
