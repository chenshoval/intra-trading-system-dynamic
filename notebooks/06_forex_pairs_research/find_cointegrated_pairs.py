"""FX Pairs Cointegration Research — Phase 1 of Hypothesis 5c

Download daily data for 27 forex pairs, test all 351 possible pair combinations
for cointegration, and identify the best candidates for pairs trading.

Output:
- Cointegrated pairs ranked by p-value and half-life
- Hedge ratios for each pair
- Visual spread charts for top candidates
"""

import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. Define our 27 FX pairs (Yahoo Finance format: EURUSD=X)
# ══════════════════════════════════════════════════════════════

PAIRS = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    # EUR crosses
    "EURJPY", "EURGBP", "EURAUD", "EURNZD", "EURCHF",
    # GBP crosses
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD",
    # AUD crosses
    "AUDJPY", "AUDNZD", "AUDCHF",
    # Other crosses
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    "EURCAD", "GBPCAD",
]

YF_TICKERS = {pair: f"{pair[:3]}{pair[3:]}=X" for pair in PAIRS}

# ══════════════════════════════════════════════════════════════
# 2. Download data
# ══════════════════════════════════════════════════════════════

print(f"Downloading daily data for {len(PAIRS)} pairs (2010-2025)...")
data = {}
failed = []

for pair in PAIRS:
    ticker = YF_TICKERS[pair]
    try:
        df = yf.download(ticker, start="2010-01-01", end="2025-12-31", progress=False)
        if df is not None and len(df) > 100:
            # yfinance returns MultiIndex columns now: ('Close', 'TICKER=X')
            close_col = df["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]  # Take first column if DataFrame
            data[pair] = close_col.dropna()
            print(f"  {pair}: {len(data[pair])} days")
        else:
            failed.append(pair)
            print(f"  {pair}: FAILED (no data)")
    except Exception as e:
        failed.append(pair)
        print(f"  {pair}: FAILED ({e})")

print(f"\nLoaded: {len(data)} pairs, Failed: {len(failed)} ({failed})")

# Align all series to common dates
if len(data) < 2:
    print("ERROR: Need at least 2 pairs. Exiting.")
    exit(1)

# Align all series to common dates
if len(data) < 2:
    print("ERROR: Need at least 2 pairs. Exiting.")
    exit(1)

# Build price DataFrame directly
prices = pd.DataFrame(data)
prices = prices.dropna()
print(f"Clean price matrix: {prices.shape[0]} days × {prices.shape[1]} pairs")

# ══════════════════════════════════════════════════════════════
# 3. Test ALL pair combinations for cointegration
# ══════════════════════════════════════════════════════════════

available_pairs = list(prices.columns)
all_combos = list(combinations(available_pairs, 2))
print(f"\nTesting {len(all_combos)} pair combinations for cointegration...")

results = []

for i, (pair_a, pair_b) in enumerate(all_combos):
    try:
        series_a = np.log(prices[pair_a].values)
        series_b = np.log(prices[pair_b].values)

        # Engle-Granger cointegration test
        score, pvalue, _ = coint(series_a, series_b)

        # Compute hedge ratio via OLS
        from numpy.polynomial import polynomial as P
        # Simple OLS: series_a = alpha + beta * series_b + residual
        beta = np.cov(series_a, series_b)[0, 1] / np.var(series_b)
        alpha = np.mean(series_a) - beta * np.mean(series_b)

        # Compute spread and half-life
        spread = series_a - beta * series_b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        # Half-life via AR(1) regression on spread changes
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        if len(spread_lag) > 10 and np.std(spread_lag) > 1e-10:
            phi = np.corrcoef(spread_lag, spread_diff)[0, 1] * np.std(spread_diff) / np.std(spread_lag)
            if phi < 0:
                half_life = -np.log(2) / np.log(1 + phi)
            else:
                half_life = 999  # Not mean-reverting
        else:
            half_life = 999

        # ADF test on spread (should be stationary if cointegrated)
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread, maxlag=20)

        results.append({
            "pair_a": pair_a,
            "pair_b": pair_b,
            "coint_pvalue": pvalue,
            "coint_score": score,
            "hedge_ratio": beta,
            "half_life": half_life,
            "spread_std": spread_std,
            "adf_pvalue": adf_pvalue,
            "adf_stat": adf_stat,
        })

    except Exception as e:
        pass

    if (i + 1) % 50 == 0:
        print(f"  Tested {i+1}/{len(all_combos)}...")

df_results = pd.DataFrame(results)
print(f"\nTotal tested: {len(df_results)}")

# ══════════════════════════════════════════════════════════════
# 4. Filter and rank cointegrated pairs
# ══════════════════════════════════════════════════════════════

# Filter: cointegration p-value < 0.05 AND half-life < 30 days AND ADF p-value < 0.05
cointegrated = df_results[
    (df_results["coint_pvalue"] < 0.05) &
    (df_results["half_life"] < 30) &
    (df_results["half_life"] > 1) &
    (df_results["adf_pvalue"] < 0.05)
].sort_values("coint_pvalue")

print(f"\n{'='*80}")
print(f"COINTEGRATED PAIRS: {len(cointegrated)} out of {len(df_results)} tested")
print(f"{'='*80}")

if len(cointegrated) == 0:
    print("\nNO COINTEGRATED PAIRS FOUND. This means pairs trading won't work on FX.")
    print("The spread between FX pairs is NOT stationary — the relationship drifts.")
    print("\nThis is actually a valid research finding. Kill the hypothesis.")
else:
    print(f"\n{'Pair A':<10} {'Pair B':<10} {'Coint p':>10} {'Half-Life':>10} {'Hedge β':>10} {'ADF p':>10}")
    print("-" * 65)
    for _, row in cointegrated.head(20).iterrows():
        print(f"{row['pair_a']:<10} {row['pair_b']:<10} {row['coint_pvalue']:>10.4f} {row['half_life']:>10.1f} {row['hedge_ratio']:>10.4f} {row['adf_pvalue']:>10.4f}")

    print(f"\n{'='*80}")
    print("TOP 10 PAIRS (sorted by cointegration p-value):")
    print(f"{'='*80}")
    for i, (_, row) in enumerate(cointegrated.head(10).iterrows()):
        print(f"\n{i+1}. {row['pair_a']} vs {row['pair_b']}")
        print(f"   Cointegration p-value: {row['coint_pvalue']:.6f}")
        print(f"   Half-life: {row['half_life']:.1f} days")
        print(f"   Hedge ratio (β): {row['hedge_ratio']:.4f}")
        print(f"   ADF statistic: {row['adf_stat']:.4f} (p={row['adf_pvalue']:.4f})")

# ══════════════════════════════════════════════════════════════
# 5. Also check "obvious" pairs that should be cointegrated
# ══════════════════════════════════════════════════════════════

obvious_pairs = [
    ("EURUSD", "GBPUSD"),   # Both European vs USD
    ("AUDUSD", "NZDUSD"),   # Both commodity currencies
    ("EURJPY", "GBPJPY"),   # Both vs JPY
    ("EURCHF", "GBPCHF"),   # Both vs CHF
    ("USDCHF", "USDJPY"),   # Both USD-based
    ("AUDJPY", "NZDJPY"),   # Commodity vs JPY
    ("EURCAD", "GBPCAD"),   # Both vs CAD
    ("EURGBP", "EURCHF"),   # EUR base
]

print(f"\n{'='*80}")
print("'OBVIOUS' PAIRS CHECK:")
print(f"{'='*80}")
print(f"\n{'Pair A':<10} {'Pair B':<10} {'Coint p':>10} {'Half-Life':>10} {'Cointegrated?':>15}")
print("-" * 60)

for pair_a, pair_b in obvious_pairs:
    match = df_results[(df_results["pair_a"] == pair_a) & (df_results["pair_b"] == pair_b)]
    if match.empty:
        match = df_results[(df_results["pair_a"] == pair_b) & (df_results["pair_b"] == pair_a)]
    if not match.empty:
        row = match.iloc[0]
        is_coint = "YES" if row["coint_pvalue"] < 0.05 and row["half_life"] < 30 else "NO"
        print(f"{pair_a:<10} {pair_b:<10} {row['coint_pvalue']:>10.4f} {row['half_life']:>10.1f} {is_coint:>15}")
    else:
        print(f"{pair_a:<10} {pair_b:<10} {'N/A':>10} {'N/A':>10} {'MISSING':>15}")

# ══════════════════════════════════════════════════════════════
# 6. Save results
# ══════════════════════════════════════════════════════════════

output_path = "c:/ML/intra-trading-system-dynamic/notebooks/06_forex_pairs_research/cointegration_results.csv"
df_results.sort_values("coint_pvalue").to_csv(output_path, index=False)
print(f"\nFull results saved to: {output_path}")

if len(cointegrated) > 0:
    coint_path = "c:/ML/intra-trading-system-dynamic/notebooks/06_forex_pairs_research/cointegrated_pairs.csv"
    cointegrated.to_csv(coint_path, index=False)
    print(f"Cointegrated pairs saved to: {coint_path}")
