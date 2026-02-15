"""Performance evaluation metrics for trading strategies.

Used by notebooks, training scripts, and QC post-analysis.
All functions work with pandas Series/DataFrames of returns or equity curves.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════════
# Return Metrics
# ══════════════════════════════════════════════════════════════

def total_return(equity_curve: pd.Series) -> float:
    """Total return from equity curve."""
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1


def annual_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return from equity curve."""
    total = total_return(equity_curve)
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total) ** (1 / years) - 1


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized volatility."""
    return returns.std() * np.sqrt(periods_per_year)


# ══════════════════════════════════════════════════════════════
# Risk-Adjusted Returns
# ══════════════════════════════════════════════════════════════

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe Ratio.

    Args:
        returns: Period returns (daily, hourly, etc.)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year (252 for daily, ~1638 for 5m)
    """
    if returns.std() == 0:
        return 0.0
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    return (excess / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino Ratio (penalizes only downside volatility)."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    return (excess / downside.std()) * np.sqrt(periods_per_year)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar Ratio: annualized return / max drawdown."""
    equity = (1 + returns).cumprod()
    ann_ret = annual_return(equity, periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("inf") if ann_ret > 0 else 0.0
    return ann_ret / abs(mdd)


# ══════════════════════════════════════════════════════════════
# Drawdown
# ══════════════════════════════════════════════════════════════

def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (negative value, e.g., -0.15 = 15% drawdown)."""
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown series."""
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    return equity / running_max - 1


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest drawdown duration in periods."""
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    underwater = equity < running_max

    if not underwater.any():
        return 0

    # Count consecutive underwater periods
    groups = (~underwater).cumsum()
    underwater_groups = groups[underwater]
    if len(underwater_groups) == 0:
        return 0
    return underwater_groups.value_counts().max()


# ══════════════════════════════════════════════════════════════
# Trade-Level Metrics
# ══════════════════════════════════════════════════════════════

def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of trades that are profitable."""
    if len(trade_returns) == 0:
        return 0.0
    return (trade_returns > 0).mean()


def profit_factor(trade_returns: pd.Series) -> float:
    """Gross profit / gross loss. >1.0 = profitable system."""
    gains = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def avg_win_loss_ratio(trade_returns: pd.Series) -> float:
    """Average winning trade / average losing trade."""
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return abs(wins.mean() / losses.mean())


def expectancy(trade_returns: pd.Series) -> float:
    """Expected value per trade."""
    return trade_returns.mean()


# ══════════════════════════════════════════════════════════════
# Risk Metrics
# ══════════════════════════════════════════════════════════════

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk at given confidence level."""
    return returns.quantile(1 - confidence)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


# ══════════════════════════════════════════════════════════════
# Benchmark Comparison
# ══════════════════════════════════════════════════════════════

def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Jensen's alpha and beta vs benchmark."""
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("Strategy and benchmark must have same length")

    cov = np.cov(strategy_returns.dropna(), benchmark_returns.dropna())
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
    alpha = (strategy_returns.mean() - beta * benchmark_returns.mean()) * periods_per_year
    return alpha, beta


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Information Ratio: active return / tracking error."""
    active = strategy_returns - benchmark_returns
    if active.std() == 0:
        return 0.0
    return (active.mean() / active.std()) * np.sqrt(periods_per_year)


def correlation_with_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Pearson correlation between strategy and benchmark returns."""
    return strategy_returns.corr(benchmark_returns)


# ══════════════════════════════════════════════════════════════
# Full Report
# ══════════════════════════════════════════════════════════════

def performance_report(
    returns: pd.Series,
    trade_returns: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Generate a comprehensive performance report.

    Args:
        returns: Period returns (e.g., daily)
        trade_returns: Optional per-trade returns for trade-level stats
        benchmark_returns: Optional benchmark for comparison
        periods_per_year: Periods per year for annualization

    Returns:
        Dict with all performance metrics.
    """
    equity = (1 + returns).cumprod()

    report = {
        "total_return": total_return(equity),
        "annual_return": annual_return(equity, periods_per_year),
        "annual_volatility": annual_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_duration": max_drawdown_duration(returns),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "n_periods": len(returns),
    }

    if trade_returns is not None and len(trade_returns) > 0:
        report.update({
            "n_trades": len(trade_returns),
            "win_rate": win_rate(trade_returns),
            "profit_factor": profit_factor(trade_returns),
            "avg_win_loss_ratio": avg_win_loss_ratio(trade_returns),
            "expectancy": expectancy(trade_returns),
        })

    if benchmark_returns is not None:
        alpha, beta = alpha_beta(returns, benchmark_returns, periods_per_year)
        report.update({
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio(returns, benchmark_returns, periods_per_year),
            "correlation": correlation_with_benchmark(returns, benchmark_returns),
        })

    return report


def print_report(report: dict) -> None:
    """Pretty-print a performance report."""
    print("\n" + "=" * 50)
    print("  PERFORMANCE REPORT")
    print("=" * 50)

    fmt = {
        "total_return": ("Total Return", "{:.2%}"),
        "annual_return": ("Annual Return", "{:.2%}"),
        "annual_volatility": ("Annual Volatility", "{:.2%}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.3f}"),
        "sortino_ratio": ("Sortino Ratio", "{:.3f}"),
        "calmar_ratio": ("Calmar Ratio", "{:.3f}"),
        "max_drawdown": ("Max Drawdown", "{:.2%}"),
        "max_drawdown_duration": ("Max DD Duration", "{:d} periods"),
        "var_95": ("VaR (95%)", "{:.4f}"),
        "cvar_95": ("CVaR (95%)", "{:.4f}"),
        "n_periods": ("Periods", "{:d}"),
        "n_trades": ("Trades", "{:d}"),
        "win_rate": ("Win Rate", "{:.2%}"),
        "profit_factor": ("Profit Factor", "{:.3f}"),
        "avg_win_loss_ratio": ("Avg Win/Loss", "{:.3f}"),
        "expectancy": ("Expectancy", "{:.4f}"),
        "alpha": ("Alpha", "{:.4f}"),
        "beta": ("Beta", "{:.3f}"),
        "information_ratio": ("Info Ratio", "{:.3f}"),
        "correlation": ("Correlation", "{:.3f}"),
    }

    for key, (label, f) in fmt.items():
        if key in report:
            val = report[key]
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                print(f"  {label:25s}: {'N/A':>12s}")
            else:
                print(f"  {label:25s}: {f.format(val):>12s}")

    print("=" * 50 + "\n")
