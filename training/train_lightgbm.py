"""Train LightGBM directional classifier.

Standalone training script — runs locally or in Azure (ACI/Azure ML).

Usage:
    # Local
    python training/train_lightgbm.py --data data/processed/features.csv

    # Azure (via container)
    docker build -t training training/
    az container create ... (see scripts/train-on-azure.sh)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.features import compute_features, forward_return_direction
from utils.evaluation import performance_report, print_report
from utils.model_store import ModelStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("training")


def load_and_prepare_data(data_path: str, target_periods: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    """Load data and compute features + target."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])

    # If multi-ticker, compute features per ticker
    if "ticker" in df.columns:
        dfs = []
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)
            featured = compute_features(group)
            featured["ticker"] = ticker
            dfs.append(featured)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)
        df = compute_features(df)

    # Compute target
    df["target"] = forward_return_direction(df["close"], periods=target_periods)

    # Drop rows with NaN
    df = df.dropna()

    # Separate features and target
    exclude_cols = ["date", "ticker", "target", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    logger.info(f"Data: {len(df)} rows, {len(feature_cols)} features")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

    return df, feature_cols


def train_lightgbm(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
) -> tuple:
    """Train LightGBM with purged walk-forward validation.

    Returns:
        (model, metrics_dict, feature_importance)
    """
    import lightgbm as lgb

    X = df[feature_cols].values
    y = df["target"].values

    # ── Purged Walk-Forward CV ──
    n_splits = config.get("cv_splits", 5)
    purge_gap = config.get("purge_gap", 12)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=purge_gap)

    cv_scores = []
    cv_auc = []
    best_model = None
    best_score = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            **config.get("hyperparameters", {}),
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=config.get("early_stopping_rounds", 50),
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )

        # Evaluate
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_prob[:, 1])
        except ValueError:
            auc = 0.5

        cv_scores.append(acc)
        cv_auc.append(auc)

        logger.info(f"  Fold {fold + 1}: accuracy={acc:.4f}, AUC={auc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model

    # ── Final metrics ──
    metrics = {
        "cv_accuracy_mean": np.mean(cv_scores),
        "cv_accuracy_std": np.std(cv_scores),
        "cv_auc_mean": np.mean(cv_auc),
        "cv_auc_std": np.std(cv_auc),
        "n_splits": n_splits,
        "purge_gap": purge_gap,
    }

    logger.info(f"\nCV Results: accuracy={metrics['cv_accuracy_mean']:.4f} "
                f"(±{metrics['cv_accuracy_std']:.4f}), "
                f"AUC={metrics['cv_auc_mean']:.4f} "
                f"(±{metrics['cv_auc_std']:.4f})")

    # ── Feature importance ──
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info(f"\nTop 15 features:")
    for _, row in importance.head(15).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.0f}")

    return best_model, metrics, importance


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM directional classifier")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--target-periods", type=int, default=12, help="Forward return periods")
    parser.add_argument("--model-name", default="lightgbm_directional", help="Model name for storage")
    parser.add_argument("--upload-azure", action="store_true", help="Upload model to Azure Blob")
    parser.add_argument("--config", default="config/models.yaml", help="Model config YAML")
    args = parser.parse_args()

    # ── Load config ──
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            all_config = yaml.safe_load(f)
        config = all_config.get("models", {}).get("lightgbm_v1", {})
        training_config = config.get("training", {})
        config["hyperparameters"] = config.get("hyperparameters", {})
        config.update(training_config)
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = {
            "hyperparameters": {
                "num_leaves": 31,
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "class_weight": "balanced",
                "random_state": 42,
                "verbose": -1,
            },
            "cv_splits": 5,
            "purge_gap": 12,
            "early_stopping_rounds": 50,
        }

    # ── Load data ──
    df, feature_cols = load_and_prepare_data(args.data, args.target_periods)

    # ── Train ──
    logger.info("\n" + "=" * 50)
    logger.info("Training LightGBM directional classifier")
    logger.info("=" * 50)

    model, metrics, importance = train_lightgbm(df, feature_cols, config)

    # ── Save ──
    store = ModelStore(
        local_dir="models",
        azure_container="models" if args.upload_azure else None,
    )

    metadata = {
        "data_path": args.data,
        "target_periods": args.target_periods,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        **metrics,
    }

    model_dir = store.save(
        model,
        args.model_name,
        metadata=metadata,
        upload_to_azure=args.upload_azure,
    )

    # Save feature names separately (needed by QC strategy)
    feature_names_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_names_path, "w") as f:
        json.dump(feature_cols, f)

    # Save feature importance
    importance.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)

    logger.info(f"\nModel saved to: {model_dir}")
    logger.info(f"Upload to Azure: {args.upload_azure}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
