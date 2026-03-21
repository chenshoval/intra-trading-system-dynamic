"""Weight Learning — Train a model to learn optimal strategy weights.

This script:
1. Loads monthly PnL from all 4 standalone strategy backtests
2. For each month, computes what weight allocation would have maximized Sharpe
3. Builds features: recent returns, volatility, trend state per sleeve
4. Trains a DecisionTree (interpretable!) to predict optimal weights
5. Extracts the tree as if/else rules for pasting into QC algo

The key insight: we're NOT predicting returns. We're predicting WHICH SLEEVE
will be calmest and most profitable in the NEXT month, given what happened
in the LAST 3 months. The model learns patterns like:
- "When equity vol is high and bond returns are positive → overweight bonds"
- "When commodity momentum is strong and equity is flat → overweight commodity"

These are the same patterns the risk parity solver TRIES to capture, but
without the noisy covariance estimation that causes flip-flopping.
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 1. Load monthly PnL from all standalone strategy backtests
# ═══════════════════════════════════════════════════════════════

def get_monthly_pnl(json_file):
    """Extract monthly P&L from QC backtest JSON."""
    with open(json_file) as f:
        d = json.load(f)
    pnl = d.get('profitLoss', {})
    monthly = defaultdict(float)
    for ts, val in pnl.items():
        monthly[ts[:7]] += val
    return dict(monthly)

import os
# Run from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO_ROOT)

# All standalone strategy results
strategy_files = {
    'equity': [
        'results_from_quant_connect/MonthlyRotatorV2/experiment_19_2_2026_1645/2016-2020/Energetic Apricot Cobra.json',
        'results_from_quant_connect/MonthlyRotatorV2/experiment_19_2_2026_1645/2020-2023/Casual Orange Alpaca.json',
        'results_from_quant_connect/MonthlyRotatorV2/experiment_19_2_2026_1645/2023-2025/Square Green Rhinoceros.json',
    ],
    'commodity': [
        'results_from_quant_connect/commoditymomentumv1/2016-2020/Muscular Brown Gorilla.json',
        'results_from_quant_connect/commoditymomentumv1/2020-2023/Upgraded Brown Butterfly.json',
        'results_from_quant_connect/commoditymomentumv1/2023-2025/Virtual Apricot Owl.json',
    ],
    'dividend': [
        'results_from_quant_connect/dividendyieldv1/2016-2020/Hyper Active Yellow Green Sheep.json',
        'results_from_quant_connect/dividendyieldv1/2020-2023/Swimming Fluorescent Yellow Rhinoceros.json',
        'results_from_quant_connect/dividendyieldv1/2023-2025/Ugly Light Brown Pig.json',
    ],
    'bond': [
        'results_from_quant_connect/bondmomentumv1/2016-2020/Square Red Panda.json',
        'results_from_quant_connect/bondmomentumv1/2020-2023/Emotional Fluorescent Orange Eagle.json',
        'results_from_quant_connect/bondmomentumv1/2023-2025/Smooth Yellow Green Coyote.json',
    ],
}

# Load all monthly P&L
all_pnl = {}
for name, files in strategy_files.items():
    all_pnl[name] = {}
    for f in files:
        try:
            all_pnl[name].update(get_monthly_pnl(f))
        except FileNotFoundError:
            print(f"Warning: {f} not found, skipping")

# Get common months across all strategies
strategy_names = ['equity', 'commodity', 'dividend', 'bond']
common_months = sorted(set.intersection(*[set(all_pnl[n].keys()) for n in strategy_names]))
print(f"Common months: {len(common_months)}")
print(f"Date range: {common_months[0]} to {common_months[-1]}")

# ═══════════════════════════════════════════════════════════════
# 2. For each month, compute optimal weights (hindsight)
# ═══════════════════════════════════════════════════════════════

# Convert to arrays
returns = {}
for name in strategy_names:
    returns[name] = np.array([all_pnl[name][m] for m in common_months])

# For each month, what would have been the best allocation?
# We'll classify into 5 regimes based on which strategy performed best
labels = []
for i in range(len(common_months)):
    month_returns = [returns[name][i] for name in strategy_names]
    best_idx = np.argmax(month_returns)
    labels.append(best_idx)

labels = np.array(labels)
print(f"\nBest strategy distribution:")
for i, name in enumerate(strategy_names):
    count = np.sum(labels == i)
    print(f"  {name}: {count} months ({count/len(labels)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════
# 3. Build features: rolling metrics per sleeve
# ═══════════════════════════════════════════════════════════════

def build_features(returns_dict, months, lookback=3):
    """Build features for each month from the previous `lookback` months."""
    features = []
    valid_indices = []

    for i in range(lookback, len(months)):
        row = []
        for name in strategy_names:
            recent = returns_dict[name][i-lookback:i]
            # Feature 1: mean return over lookback
            row.append(np.mean(recent))
            # Feature 2: volatility over lookback
            row.append(np.std(recent) if len(recent) > 1 else 0)
            # Feature 3: last month return
            row.append(returns_dict[name][i-1])
            # Feature 4: trend (positive months / total)
            row.append(np.sum(recent > 0) / len(recent))
            # Feature 5: momentum (sum of returns)
            row.append(np.sum(recent))
        features.append(row)
        valid_indices.append(i)

    feature_names = []
    for name in strategy_names:
        feature_names.extend([
            f"{name}_mean_ret", f"{name}_vol", f"{name}_last_ret",
            f"{name}_trend_pct", f"{name}_momentum"
        ])

    return np.array(features), np.array(valid_indices), feature_names

X, valid_idx, feature_names = build_features(returns, common_months, lookback=3)
y = labels[valid_idx]

print(f"\nFeature matrix: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {feature_names}")

# ═══════════════════════════════════════════════════════════════
# 4. Train Decision Tree (interpretable)
# ═══════════════════════════════════════════════════════════════

# Shallow tree for interpretability — max depth 4 gives readable rules
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
dt.fit(X, y)

# Cross-validation score
cv_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
print(f"\nDecision Tree accuracy: {dt.score(X, y):.1%} (train)")
print(f"Cross-validation: {np.mean(cv_scores):.1%} +/- {np.std(cv_scores):.1%}")

# Compare to always picking equity (the naive baseline)
baseline = np.sum(y == 0) / len(y)
print(f"Baseline (always equity): {baseline:.1%}")

# ═══════════════════════════════════════════════════════════════
# 5. Extract tree as readable rules
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DECISION TREE RULES (paste into QC algo)")
print("="*60)
tree_rules = export_text(dt, feature_names=feature_names,
                          class_names=strategy_names)
print(tree_rules)

# Also extract as feature importances
print("\nFeature importances:")
for name, imp in sorted(zip(feature_names, dt.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    if imp > 0.01:
        print(f"  {name}: {imp:.3f}")

# ═══════════════════════════════════════════════════════════════
# 6. Convert to weight rules for QC
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("WEIGHT MAPPING")
print("="*60)
print("""
In the QC algo, use the tree prediction to set weights:
- If tree predicts 'equity'    -> weights = [0.50, 0.17, 0.17, 0.16]
- If tree predicts 'commodity' -> weights = [0.17, 0.50, 0.17, 0.16]
- If tree predicts 'dividend'  -> weights = [0.17, 0.17, 0.50, 0.16]
- If tree predicts 'bond'      -> weights = [0.16, 0.17, 0.17, 0.50]

The "winner" gets 50%, others split the remaining 50% equally.
This is the simplest translation: bet more on what the model says will win,
but keep all 4 sleeves active for diversification.

Alternative: use softer weights (40/20/20/20) for less aggressive betting.
""")

# ═══════════════════════════════════════════════════════════════
# 7. Simulate the strategy with learned weights
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("SIMULATED PERFORMANCE WITH LEARNED WEIGHTS")
print("="*60)

for winner_weight in [0.40, 0.50, 0.60]:
    other_weight = (1.0 - winner_weight) / 3

    portfolio_returns = []
    for i, idx in enumerate(valid_idx):
        prediction = dt.predict(X[i:i+1])[0]
        weights = [other_weight] * 4
        weights[prediction] = winner_weight

        month_ret = sum(weights[j] * returns[strategy_names[j]][idx]
                       for j in range(4))
        portfolio_returns.append(month_ret)

    port = np.array(portfolio_returns)
    total_ret = np.sum(port)
    avg_ret = np.mean(port)
    std_ret = np.std(port)
    sharpe = avg_ret / std_ret if std_ret > 0 else 0
    win_rate = np.sum(port > 0) / len(port)

    # Compare to equal weight
    equal_returns = []
    for idx in valid_idx:
        month_ret = sum(0.25 * returns[strategy_names[j]][idx] for j in range(4))
        equal_returns.append(month_ret)
    eq = np.array(equal_returns)

    print(f"\nWinner weight {winner_weight:.0%} / Others {other_weight:.0%}:")
    print(f"  AI weights:    Total=${np.sum(port):>10,.0f}  Avg=${avg_ret:>8,.0f}  Sharpe={sharpe:.3f}  WR={win_rate:.0%}")
    print(f"  Equal weight:  Total=${np.sum(eq):>10,.0f}  Avg=${np.mean(eq):>8,.0f}  Sharpe={np.mean(eq)/np.std(eq) if np.std(eq)>0 else 0:.3f}  WR={np.sum(eq>0)/len(eq):.0%}")
    print(f"  Improvement:   ${np.sum(port)-np.sum(eq):>+10,.0f}")

# ═══════════════════════════════════════════════════════════════
# 8. Save the decision tree rules as Python code
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PYTHON CODE FOR QC ALGO (copy-paste ready)")
print("="*60)

# Extract the tree structure
tree = dt.tree_
def tree_to_code(tree, feature_names, class_names, indent="    "):
    """Convert sklearn decision tree to Python if/else code."""
    lines = []
    def recurse(node, depth=0):
        prefix = indent * (depth + 1)
        if tree.feature[node] != -2:  # not a leaf
            fname = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            lines.append(f"{prefix}if {fname} <= {threshold:.6f}:")
            recurse(tree.children_left[node], depth + 1)
            lines.append(f"{prefix}else:")
            recurse(tree.children_right[node], depth + 1)
        else:  # leaf
            class_idx = np.argmax(tree.value[node])
            lines.append(f"{prefix}prediction = {class_idx}  # {class_names[class_idx]}")
    recurse(0)
    return "\n".join(lines)

code = tree_to_code(dt.tree_, feature_names, strategy_names)
print(f"""
def _predict_best_sleeve(self, features):
    \"\"\"AI-learned weight prediction from decision tree.
    features dict should contain:
      equity_mean_ret, equity_vol, equity_last_ret, equity_trend_pct, equity_momentum
      commodity_mean_ret, commodity_vol, commodity_last_ret, commodity_trend_pct, commodity_momentum
      dividend_mean_ret, dividend_vol, dividend_last_ret, dividend_trend_pct, dividend_momentum
      bond_mean_ret, bond_vol, bond_last_ret, bond_trend_pct, bond_momentum
    Returns: index 0-3 (equity, commodity, dividend, bond)
    \"\"\"
    # Unpack features
    {chr(10).join(f"    {fn} = features['{fn}']" for fn in feature_names)}

    # Decision tree rules (learned from backtest data)
{code}

    return prediction
""")
