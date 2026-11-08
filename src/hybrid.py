"""
Hybrid Fraud Model (Tabular Velocity Features + GraphSAGE Structural Embeddings)
Author: Principal Data Scientist & Software Engineer
Purpose:
  - Enriches real-time transaction features with nearline 32-dimensional GraphSAGE node embeddings.
  - Computes edge-level graph affinity features (Cosine similarity, L2 distance, dot product).
  - Trains the Champion Hybrid Gradient Boosting Classifier.
  - Runs rigorous Out-Of-Time Head-to-Head Benchmark against Tabular Baseline.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from baseline import FEATURE_COLS, evaluate_fraud_metrics


def enrich_with_graph_embeddings(
    df: pd.DataFrame,
    node_embeddings: np.ndarray,
    node_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Enriches tabular transaction features with source & destination GraphSAGE embeddings,
    plus graph interaction metrics (dot product, L2 distance, cosine similarity).
    """
    n = len(df)
    emb_dim = node_embeddings.shape[1]

    src_embs = np.zeros((n, emb_dim), dtype=np.float32)
    dst_embs = np.zeros((n, emb_dim), dtype=np.float32)

    src_ids = df["src_id"].values
    dst_ids = df["dst_id"].values

    for i in range(n):
        s_id = src_ids[i]
        d_id = dst_ids[i]
        if s_id in node_to_idx:
            src_embs[i] = node_embeddings[node_to_idx[s_id]]
        if d_id in node_to_idx:
            dst_embs[i] = node_embeddings[node_to_idx[d_id]]

    # Interaction metrics in latent graph space
    dot_prod = np.sum(src_embs * dst_embs, axis=1, keepdims=True)
    l2_dist = np.linalg.norm(src_embs - dst_embs, axis=1, keepdims=True)

    src_norm = np.linalg.norm(src_embs, axis=1, keepdims=True) + 1e-8
    dst_norm = np.linalg.norm(dst_embs, axis=1, keepdims=True) + 1e-8
    cosine_sim = dot_prod / (src_norm * dst_norm)

    # Base tabular features
    X_base = df[FEATURE_COLS].values.astype(np.float32)

    # Concatenate: [Tabular, Dot_Product, L2_Distance, Cosine_Similarity, Src_Embs, Dst_Embs]
    X_hybrid = np.hstack([X_base, dot_prod, l2_dist, cosine_sim, src_embs, dst_embs])

    feature_names = (
        FEATURE_COLS
        + ["graph_dot_product", "graph_l2_dist", "graph_cosine_sim"]
        + [f"src_sage_emb_{j}" for j in range(emb_dim)]
        + [f"dst_sage_emb_{j}" for j in range(emb_dim)]
    )

    return X_hybrid, feature_names


def train_hybrid_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    node_embeddings: np.ndarray,
    node_to_idx: Dict[str, int],
    random_seed: int = 42,
) -> Tuple[xgb.XGBClassifier, Dict[str, float], np.ndarray, List[str]]:
    """
    Trains the Champion Hybrid model and evaluates on test split.
    """
    X_train, feat_names = enrich_with_graph_embeddings(train_df, node_embeddings, node_to_idx)
    X_val, _ = enrich_with_graph_embeddings(val_df, node_embeddings, node_to_idx)
    X_test, _ = enrich_with_graph_embeddings(test_df, node_embeddings, node_to_idx)

    y_train = train_df["is_fraud"].values
    y_val = val_df["is_fraud"].values
    y_test = test_df["is_fraud"].values

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

    test_probs = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_fraud_metrics(y_test, test_probs, top_k_ratio=0.01)

    return model, metrics, test_probs, feat_names


if __name__ == "__main__":
    from generator import GeneratorConfig, generate_fintech_network
    from features import compute_temporal_velocity_features, temporal_split
    from baseline import train_tabular_baseline
    from gnn import construct_graph_matrices, train_inductive_graphsage

    print("1. Generating FinTech Network & Temporal Features...")
    n_df, e_df = generate_fintech_network(GeneratorConfig())
    f_df = compute_temporal_velocity_features(e_df, n_df)
    train_d, val_d, test_d = temporal_split(f_df)

    print("2. Training Baseline XGBoost Model...")
    _, base_metrics, base_preds = train_tabular_baseline(train_d, val_d, test_d)

    print("3. Training GraphSAGE on Training Period Edges (Zero Leakage)...")
    t_max = int(train_d["timestamp"].max())
    X_g, A_norm, n2i = construct_graph_matrices(n_df, e_df, max_timestamp=t_max)
    y_nodes = n_df["is_mule"].values[:, None].astype(np.float32)
    _, node_embs, _ = train_inductive_graphsage(X_g, A_norm, y_nodes, epochs=120)

    print("4. Training Champion Hybrid Model (Tabular + GraphSAGE)...")
    hybrid_clf, hybrid_metrics, hybrid_preds, feat_cols = train_hybrid_model(
        train_d, val_d, test_d, node_embs, n2i
    )

    print("\n" + "=" * 65)
    print("      HEAD-TO-HEAD BENCHMARK: BASELINE vs CHAMPION HYBRID      ")
    print("=" * 65)
    print(f"{'Metric':<30s} | {'Tabular Baseline':<16s} | {'Champion Hybrid':<16s} | {'Delta':<10s}")
    print("-" * 65)
    for m in ["pr_auc", "roc_auc", "recall_at_top_1pct", "precision_at_top_1pct", "max_f1"]:
        b_val = base_metrics[m]
        h_val = hybrid_metrics[m]
        diff = h_val - b_val
        print(f"{m:<30s} | {b_val:<16.4f} | {h_val:<16.4f} | {diff:+10.4f}")
    print("=" * 65)
