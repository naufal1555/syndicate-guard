"""
Master Pipeline Runner - SyndicateGuard End-to-End Execution
Author: Principal Data Scientist & Software Engineer
Purpose:
  - Generates synthetic network.
  - Extracts non-leaked temporal velocity features.
  - Trains Tabular Baseline (XGBoost).
  - Trains Inductive GraphSAGE & extracts embeddings.
  - Trains Champion Hybrid Model.
  - Extracts and renders Pyvis interactive subgraph.
  - Runs Real-Time Serving Latency Profiling.
"""

import time
import os
import sys

# Ensure src is on python path
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from generator import GeneratorConfig, generate_fintech_network
from features import compute_temporal_velocity_features, temporal_split
from baseline import train_tabular_baseline
from gnn import construct_graph_matrices, train_inductive_graphsage
from hybrid import train_hybrid_model
from explainer import extract_syndicate_subgraph, export_interactive_graph_html
from api import ServingEngine, TransactionRequest
import numpy as np


def run_full_pipeline():
    print("=" * 70)
    print("   SYNDICATEGUARD: END-TO-END PIPELINE EXECUTION & BENCHMARK   ")
    print("=" * 70)

    # 1. Ingestion / Generation
    print("\n[Step 1/5] Generating FinTech Network & Injecting AML Topologies...")
    t0 = time.time()
    cfg = GeneratorConfig(random_seed=42)
    n_df, e_df = generate_fintech_network(cfg)
    print(f"  --> Accounts: {len(n_df)} | Transactions: {len(e_df)} | Fraud Ratio: {e_df['is_fraud'].mean():.2%}")
    print(f"  --> Completed in {time.time() - t0:.2f}s")

    # 2. Feature Engineering & Split
    print("\n[Step 2/5] Computing Non-Leaked Rolling Velocity Features & Temporal Split...")
    t0 = time.time()
    f_df = compute_temporal_velocity_features(e_df, n_df)
    train_d, val_d, test_d = temporal_split(f_df)
    print(f"  --> Train: {len(train_d)} rows | Val: {len(val_d)} rows | Test: {len(test_d)} rows (Strict OOT)")
    print(f"  --> Completed in {time.time() - t0:.2f}s")

    # 3. Tabular Baseline
    print("\n[Step 3/5] Training Tabular Baseline (XGBoost)...")
    t0 = time.time()
    base_clf, base_metrics, _ = train_tabular_baseline(train_d, val_d, test_d)
    print(f"  --> Baseline PR-AUC: {base_metrics['pr_auc']:.4f} | Recall@1%: {base_metrics['recall_at_top_1pct']:.4f}")
    print(f"  --> Completed in {time.time() - t0:.2f}s")

    # 4. GraphSAGE & Champion Hybrid
    print("\n[Step 4/5] Training Inductive GraphSAGE & Champion Hybrid Model...")
    t0 = time.time()
    t_max = int(train_d["timestamp"].max())
    X_g, A_norm, n2i = construct_graph_matrices(n_df, e_df, max_timestamp=t_max)
    y_nodes = n_df["is_mule"].values[:, None].astype(np.float32)
    sage_clf, node_embs, sage_metrics = train_inductive_graphsage(X_g, A_norm, y_nodes, epochs=120)

    hybrid_clf, hybrid_metrics, _, feat_names = train_hybrid_model(
        train_d, val_d, test_d, node_embs, n2i
    )
    print(f"  --> Hybrid PR-AUC: {hybrid_metrics['pr_auc']:.4f} | Recall@1%: {hybrid_metrics['recall_at_top_1pct']:.4f}")
    print(f"  --> Completed in {time.time() - t0:.2f}s")

    # 5. Serving Latency & Subgraph Export
    print("\n[Step 5/5] Profiling Two-Tier Serving Latency & Exporting Artifacts...")
    engine = ServingEngine()
    engine.load_artifacts(hybrid_clf, node_embs, n2i, feat_names, n_df)

    latencies = []
    test_req = TransactionRequest(src_account_id="USR_00010", dst_account_id="SYN_00001", amount=350000.0)
    for _ in range(100):
        res = engine.score_transaction(test_req)
        latencies.append(res.latency_ms)

    lat_arr = np.array(latencies)
    print(f"  --> Mean Latency: {np.mean(lat_arr):.3f} ms | p99 Latency: {np.percentile(lat_arr, 99):.3f} ms")

    # Export Subgraph
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    sample_mule = e_df[e_df["is_fraud"] == 1]["dst_id"].iloc[0]
    subgraph = extract_syndicate_subgraph(e_df, sample_mule, hops=2)
    html_out = os.path.join(artifacts_dir, "syndicate_subgraph.html")
    export_interactive_graph_html(subgraph, html_out, sample_mule, n_df)
    print(f"  --> Interactive Subgraph saved: {html_out}")

    # Final Benchmark Summary
    print("\n" + "=" * 65)
    print("      FINAL HEAD-TO-HEAD BENCHMARK (TEST SPLIT EVALUATION)     ")
    print("=" * 65)
    print(f"{'Metric':<30s} | {'Tabular Baseline':<16s} | {'Champion Hybrid':<16s} | {'Delta':<10s}")
    print("-" * 65)
    for m in ["pr_auc", "roc_auc", "recall_at_top_1pct", "precision_at_top_1pct", "max_f1"]:
        b_val = base_metrics[m]
        h_val = hybrid_metrics[m]
        diff = h_val - b_val
        print(f"{m:<30s} | {b_val:<16.4f} | {h_val:<16.4f} | {diff:+10.4f}")
    print("=" * 65)
    print("\n[SUCCESS] ALL PIPELINE MODULES VERIFIED & OPERATIONAL.")


if __name__ == "__main__":
    run_full_pipeline()
