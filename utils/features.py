"""Feature computation functions.

Used by:
- Jupyter notebooks for feature exploration
- Training scripts for model input preparation
- QuantConnect algorithms (via adapted versions)

Features are registered in config/features.yaml. This module provides
the computation functions referenced there.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════════
# Moving Averages
# ══════════════════════════════════════════════════════════════

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


# ══════════════════════════════════════════════════════════════
# Momentum
# ══════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD: returns DataFrame with macd_line, macd_signal, macd_histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    macd_histogram = macd_line - macd_signal
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
    })


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change (%)."""
    return series.pct_change(periods=period) * 100


# ══════════════════════════════════════════════════════════════
# Volatility
# ══════════════════════════════════════════════════════════════

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range. Expects columns: high, low, close."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, middle, lower, width, %B."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle
    pct_b = (series - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct": pct_b,
    })


# ══════════════════════════════════════════════════════════════
# Volume
# ══════════════════════════════════════════════════════════════

def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume / rolling average volume."""
    avg_vol = df["volume"].rolling(window=period).mean()
    return df["volume"] / avg_vol.replace(0, np.nan)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (intraday — resets daily)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


# ══════════════════════════════════════════════════════════════
# Price Action
# ══════════════════════════════════════════════════════════════

def returns(series: pd.Series, period: int = 1) -> pd.Series:
    """Percent returns over N periods."""
    return series.pct_change(periods=period)


def price_ratio(df: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Ratio of two price columns (e.g., close/open)."""
    return df[numerator] / df[denominator].replace(0, np.nan)


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candlestick features: body_size, upper_shadow, lower_shadow."""
    body = (df["close"] - df["open"]).abs()
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    upper = df["high"] - df[["close", "open"]].max(axis=1)
    lower = df[["close", "open"]].min(axis=1) - df["low"]
    return pd.DataFrame({
        "body_size": body / full_range,
        "upper_shadow": upper / full_range,
        "lower_shadow": lower / full_range,
    })


# ══════════════════════════════════════════════════════════════
# Structure
# ══════════════════════════════════════════════════════════════

def price_to_ma_ratio(series: pd.Series, period: int) -> pd.Series:
    """Price relative to its moving average."""
    ma = sma(series, period)
    return series / ma.replace(0, np.nan) - 1


def ma_crossover_ratio(series: pd.Series, fast: int, slow: int) -> pd.Series:
    """Fast MA / Slow MA — 1. Positive = fast above slow."""
    fast_ma = sma(series, fast)
    slow_ma = sma(series, slow)
    return fast_ma / slow_ma.replace(0, np.nan) - 1


# ══════════════════════════════════════════════════════════════
# Multi-timeframe Returns (for directional classifier)
# ══════════════════════════════════════════════════════════════

def multi_timeframe_returns(series: pd.Series, periods: list[int] = None) -> pd.DataFrame:
    """Returns over multiple lookback periods.
    Default: 1, 5, 10, 21 days (daily) or bars (intraday).
    """
    if periods is None:
        periods = [1, 5, 10, 21]
    result = {}
    for p in periods:
        result[f"return_{p}"] = series.pct_change(periods=p)
    return pd.DataFrame(result)


# ══════════════════════════════════════════════════════════════
# Cross-Stock Features (for global model)
# ══════════════════════════════════════════════════════════════

def cross_stock_returns(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
    close_col: str = "close",
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute returns of top N most-traded stocks as features for all stocks.
    Useful for the global directional classifier (cross-asset patterns).
    """
    pivot = df.pivot_table(index=date_col, columns=ticker_col, values=close_col)
    top_tickers = df.groupby(ticker_col)["volume"].sum().nlargest(top_n).index.tolist()
    cross_returns = pivot[top_tickers].pct_change()
    cross_returns.columns = [f"cross_{t}_return" for t in cross_returns.columns]
    return cross_returns


# ══════════════════════════════════════════════════════════════
# Target Variables
# ══════════════════════════════════════════════════════════════

def forward_return(series: pd.Series, periods: int = 12) -> pd.Series:
    """N-bar forward return (regression target)."""
    return series.shift(-periods) / series - 1


def forward_return_direction(series: pd.Series, periods: int = 12, threshold: float = 0.0) -> pd.Series:
    """Binary direction: 1 if forward return > threshold, 0 otherwise."""
    fwd = forward_return(series, periods)
    return (fwd > threshold).astype(int)


def triple_barrier_label(
    df: pd.DataFrame,
    periods: int = 12,
    take_profit_mult: float = 2.0,
    stop_loss_mult: float = 1.0,
    atr_period: int = 14,
) -> pd.Series:
    """Triple barrier labeling (de Prado).
    Returns: 1 (take profit), 0 (timeout), -1 (stop loss).
    """
    atr_val = atr(df, atr_period)
    labels = pd.Series(index=df.index, dtype=float)

    for i in range(len(df) - periods):
        entry_price = df["close"].iloc[i]
        current_atr = atr_val.iloc[i]
        if pd.isna(current_atr):
            labels.iloc[i] = np.nan
            continue

        tp = entry_price + take_profit_mult * current_atr
        sl = entry_price - stop_loss_mult * current_atr

        label = 0  # timeout (vertical barrier)
        for j in range(1, periods + 1):
            if i + j >= len(df):
                break
            high = df["high"].iloc[i + j]
            low = df["low"].iloc[i + j]
            if high >= tp:
                label = 1  # take profit
                break
            if low <= sl:
                label = -1  # stop loss
                break
        labels.iloc[i] = label

    return labels


# ══════════════════════════════════════════════════════════════
# Compute All Features (registry-driven)
# ══════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame, feature_config: Optional[dict] = None) -> pd.DataFrame:
    """Compute all features for a single-ticker DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        feature_config: Optional dict from features.yaml. If None, computes default set.

    Returns:
        DataFrame with all original columns + computed features.
    """
    result = df.copy()

    # Moving averages
    for period in [5, 10, 20, 60]:
        result[f"sma_{period}"] = sma(result["close"], period)
    for period in [12, 26]:
        result[f"ema_{period}"] = ema(result["close"], period)

    # Momentum
    result["rsi_14"] = rsi(result["close"], 14)
    result["rsi_7"] = rsi(result["close"], 7)
    macd_df = macd(result["close"])
    result = pd.concat([result, macd_df], axis=1)
    result["roc_5"] = roc(result["close"], 5)
    result["roc_10"] = roc(result["close"], 10)

    # Volatility
    result["atr_14"] = atr(result, 14)
    bb_df = bollinger_bands(result["close"], 20, 2.0)
    result = pd.concat([result, bb_df], axis=1)

    # Volume
    result["volume_sma_20"] = sma(result["volume"], 20)
    result["volume_ratio"] = volume_ratio(result, 20)

    # Price action
    result["daily_return"] = returns(result["close"], 1)
    result["close_to_open"] = price_ratio(result, "close", "open")
    result["high_low_range"] = price_ratio(result, "high", "low")
    candle_df = candle_features(result)
    result = pd.concat([result, candle_df], axis=1)

    # Structure
    result["close_to_sma20"] = price_to_ma_ratio(result["close"], 20)
    result["close_to_sma60"] = price_to_ma_ratio(result["close"], 60)
    result["sma5_to_sma20"] = ma_crossover_ratio(result["close"], 5, 20)

    # Multi-timeframe returns
    mtf = multi_timeframe_returns(result["close"])
    result = pd.concat([result, mtf], axis=1)

    return result
