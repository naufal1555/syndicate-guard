"""
Tabular Baseline Model (LightGBM) for FinTech Fraud Detection
Author: Principal Data Scientist & Software Engineer
Purpose:
  - Provides the rigorous industrial baseline for transaction-level fraud detection.
  - Evaluates standard velocity and entity features without graph structural embeddings.
  - Implements operational fraud metrics: PR-AUC, ROC-AUC, Recall@Top-K Alert Rate.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve


FEATURE_COLS = [
    "src_tx_count_1h",
    "src_amount_sum_1h",
    "src_tx_count_24h",
    "src_amount_sum_24h",
    "dst_tx_count_1h",
    "dst_amount_sum_1h",
    "dst_tx_count_24h",
    "dst_amount_sum_24h",
    "src_kyc",
    "dst_kyc",
    "src_age_days",
    "dst_age_days",
    "log_amount",
    "is_p2m",
    "src_burst_ratio_1h_to_24h",
    "dst_burst_ratio_1h_to_24h",
]


def evaluate_fraud_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, top_k_ratio: float = 0.01
) -> Dict[str, float]:
    """
    Computes industry-standard operational fraud metrics:
      1. PR-AUC (Average Precision)
      2. ROC-AUC
      3. Recall @ Top-K Alert Rate (capacity-constrained operational metric)
      4. Precision @ Top-K Alert Rate
      5. Max F1 across all thresholds
    """
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    # Capacity-constrained Alert Rate (e.g., Top 1% highest scored transactions reviewed by ops)
    n = len(y_true)
    k_cutoff = max(1, int(n * top_k_ratio))
    top_k_indices = np.argsort(y_prob)[::-1][:k_cutoff]

    tp_at_k = np.sum(y_true[top_k_indices] == 1)
    total_positives = np.sum(y_true == 1)

    recall_at_k = float(tp_at_k / total_positives) if total_positives > 0 else 0.0
    precision_at_k = float(tp_at_k / k_cutoff)

    # Optimal F1 search
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precisions + recalls
    f1_scores = np.divide(
        2 * (precisions * recalls), denom, out=np.zeros_like(denom), where=denom != 0
    )
    max_f1 = float(np.max(f1_scores))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        f"recall_at_top_{int(top_k_ratio*100)}pct": recall_at_k,
        f"precision_at_top_{int(top_k_ratio*100)}pct": precision_at_k,
        "max_f1": max_f1,
    }


def train_tabular_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_seed: int = 42,
) -> Tuple[xgb.XGBClassifier, Dict[str, float], np.ndarray]:
    """
    Trains an XGBoost gradient boosting classifier with early stopping on validation split.
    Evaluates on test split (Out-of-Time).
    """
    X_train, y_train = train_df[FEATURE_COLS], train_df["is_fraud"].values
    X_val, y_val = val_df[FEATURE_COLS], val_df["is_fraud"].values
    X_test, y_test = test_df[FEATURE_COLS], test_df["is_fraud"].values

    # Handle class imbalance ratio
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos = neg_count / max(1, pos_count)

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        scale_pos_weight=scale_pos,
        random_state=random_seed,
        eval_metric="logloss",
        early_stopping_rounds=30,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predict probabilities on test split
    test_probs = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_fraud_metrics(y_test, test_probs, top_k_ratio=0.01)

    return model, metrics, test_probs


if __name__ == "__main__":
    from generator import GeneratorConfig, generate_fintech_network
    from features import compute_temporal_velocity_features, temporal_split

    print("Generating network & features...")
    n_df, e_df = generate_fintech_network(GeneratorConfig())
    f_df = compute_temporal_velocity_features(e_df, n_df)
    train_d, val_d, test_d = temporal_split(f_df)

    print("Training Tabular XGBoost Baseline...")
    clf, results, preds = train_tabular_baseline(train_d, val_d, test_d)
    print("--- BASELINE XGBOOST RESULTS (TEST SET) ---")
    for k, v in results.items():
        print(f"{k:25s}: {v:.4f}")
