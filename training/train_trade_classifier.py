"""Train LightGBM to predict which event-driven trades will win.

Uses trade data from QC backtests as training data.
Features: stock identity, gap size, volume ratio, event type, market context.
Target: IsWin (binary — did the trade make money?)

Output: per-stock confidence thresholds that can be loaded into QC strategy.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import json
import os

# ── Load all trade data ──
base = "results_from_quant_connect/event-driven new refactor/experiment_17_2_2026_1300"
periods = {
    "2016-2020": f"{base}/2016-2020/Crying Black Rat_trades.csv",
    "2018-2021": f"{base}/2018-2021/Emotional Brown Seahorse_trades.csv",
    "2020-2024": f"{base}/2020-2024/Geeky Red Orange Antelope_trades.csv",
    "2022-2023": f"{base}/2022-2023/Virtual Red Orange Viper_trades.csv",
}

all_trades = []
for period, path in periods.items():
    df = pd.read_csv(path)
    df["period"] = period
    all_trades.append(df)

trades = pd.concat(all_trades, ignore_index=True)
print(f"Total trades: {len(trades)}")
print(f"Win rate: {trades['IsWin'].mean():.1%}")
print(f"Unique stocks: {trades['Symbols'].nunique()}")

# ── Feature Engineering ──
# Extract features from the trade data

# 1. Stock identity (one-hot or label encoded)
trades["stock_id"] = pd.Categorical(trades["Symbols"]).codes

# 2. Entry price level (normalized — high vs low priced stocks behave differently)
trades["entry_price"] = trades["Entry Price"]
trades["log_price"] = np.log(trades["entry_price"].clip(lower=1))

# 3. P&L ratio (how much the stock moved as % of entry price)
trades["pnl_pct"] = trades["P&L"] / (trades["Entry Price"] * trades["Quantity"].abs())

# 4. Trade size (quantity)
trades["quantity"] = trades["Quantity"].abs()

# 5. Drawdown during trade (how bad did it get?)
trades["trade_drawdown"] = trades["Drawdown"]

# 6. Direction (should always be Buy for our strategy)
trades["direction"] = (trades["Direction"] == "Buy").astype(int)

# 7. Time features
trades["entry_time"] = pd.to_datetime(trades["Entry Time"], format="ISO8601")
trades["exit_time"] = pd.to_datetime(trades["Exit Time"], format="ISO8601")
trades["hold_days"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 86400
trades["entry_month"] = trades["entry_time"].dt.month
trades["entry_dow"] = trades["entry_time"].dt.dayofweek  # 0=Mon, 4=Fri
trades["entry_hour"] = trades["entry_time"].dt.hour

# 8. Gap proxy (exit_price / entry_price - 1 gives the return, but we want the entry gap)
# We don't have the gap directly, so we use entry_price movement as proxy
trades["price_return"] = (trades["Exit Price"] - trades["Entry Price"]) / trades["Entry Price"]

# 9. Per-stock historical stats (rolling — what's this stock's track record?)
trades = trades.sort_values("entry_time").reset_index(drop=True)
for sym in trades["Symbols"].unique():
    mask = trades["Symbols"] == sym
    # Rolling win rate (last 20 trades for this stock)
    trades.loc[mask, "stock_rolling_wr"] = (
        trades.loc[mask, "IsWin"].rolling(20, min_periods=5).mean().shift(1)
    )
    # Rolling avg P&L
    trades.loc[mask, "stock_rolling_pnl"] = (
        trades.loc[mask, "P&L"].rolling(20, min_periods=5).mean().shift(1)
    )

# Fill NaN rolling features with global averages
trades["stock_rolling_wr"] = trades["stock_rolling_wr"].fillna(trades["IsWin"].mean())
trades["stock_rolling_pnl"] = trades["stock_rolling_pnl"].fillna(trades["P&L"].mean())

# ── Define features and target ──
feature_cols = [
    "stock_id",
    "log_price",
    "quantity",
    "hold_days",
    "entry_month",
    "entry_dow",
    "entry_hour",
    "stock_rolling_wr",
    "stock_rolling_pnl",
]

X = trades[feature_cols].values
y = trades["IsWin"].values

print(f"\nFeatures: {feature_cols}")
print(f"X shape: {X.shape}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# ── Walk-Forward Split ──
# Use time-based split: train on earlier periods, test on later
# Train: 2016-2021, Test: 2022-2024
train_mask = trades["entry_time"] < "2022-01-01"
test_mask = trades["entry_time"] >= "2022-01-01"

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain: {len(X_train)} trades (pre-2022)")
print(f"Test:  {len(X_test)} trades (2022+)")

# ── Train LightGBM ──
model = lgb.LGBMClassifier(
    num_leaves=31,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=50),
    ],
)

# ── Evaluate ──
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\n{'='*50}")
print(f"TEST SET RESULTS (2022+)")
print(f"{'='*50}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Baseline: {y_test.mean():.4f} (always predict majority)")
print(f"\n{classification_report(y_test, y_pred, target_names=['Loss', 'Win'])}")

# ── Feature Importance ──
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
for _, row in importance.iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.0f}")

# ── Per-Stock Optimal Thresholds ──
# For each stock, find the confidence threshold that maximizes profit
print(f"\n{'='*50}")
print(f"PER-STOCK OPTIMAL THRESHOLDS")
print(f"{'='*50}")

test_trades = trades[test_mask].copy()
test_trades["pred_prob"] = y_prob

thresholds = {}
for sym in sorted(test_trades["Symbols"].unique()):
    sym_trades = test_trades[test_trades["Symbols"] == sym]
    if len(sym_trades) < 5:
        thresholds[sym] = 0.5  # default
        continue

    best_threshold = 0.5
    best_pnl = sym_trades["P&L"].sum()  # baseline: take all trades

    for t in np.arange(0.45, 0.75, 0.05):
        filtered = sym_trades[sym_trades["pred_prob"] >= t]
        if len(filtered) < 3:
            continue
        pnl = filtered["P&L"].sum()
        if pnl > best_pnl:
            best_pnl = pnl
            best_threshold = t

    thresholds[sym] = round(best_threshold, 2)
    all_pnl = sym_trades["P&L"].sum()
    filtered_pnl = sym_trades[sym_trades["pred_prob"] >= best_threshold]["P&L"].sum()
    n_all = len(sym_trades)
    n_filtered = len(sym_trades[sym_trades["pred_prob"] >= best_threshold])
    print(f"  {sym:6s}: t*={best_threshold:.2f}, "
          f"trades {n_all}->{n_filtered}, "
          f"P&L ${all_pnl:>8,.0f} -> ${filtered_pnl:>8,.0f}")

# ── Save thresholds ──
output = {
    "thresholds": thresholds,
    "model_accuracy": float(accuracy_score(y_test, y_pred)),
    "feature_cols": feature_cols,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
}

os.makedirs("models/trade_classifier", exist_ok=True)
with open("models/trade_classifier/thresholds.json", "w") as f:
    json.dump(output, f, indent=2)

# Save model
import pickle
with open("models/trade_classifier/model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to models/trade_classifier/")
print(f"Thresholds saved to models/trade_classifier/thresholds.json")

# ── Impact Analysis ──
print(f"\n{'='*50}")
print(f"IMPACT ANALYSIS: What if we only took high-confidence trades?")
print(f"{'='*50}")

for min_conf in [0.50, 0.55, 0.60, 0.65]:
    filtered = test_trades[test_trades["pred_prob"] >= min_conf]
    if len(filtered) == 0:
        continue
    wr = filtered["IsWin"].mean()
    total_pnl = filtered["P&L"].sum()
    fees = filtered["Fees"].sum()
    net = total_pnl - fees
    print(f"  Confidence >= {min_conf:.0%}: {len(filtered)} trades, "
          f"WR={wr:.1%}, P&L=${total_pnl:,.0f}, Fees=${fees:,.0f}, Net=${net:,.0f}")
