"""
Explainability & Subgraph Visualizer (SyndicateGuard)
Author: Principal Data Scientist & Software Engineer
Purpose:
  - Regulatory compliance explainability for OJK/BI audit standards.
  - Generates Global Feature Importance via Gradient Boosting tree gain.
  - Extracts local k-hop Ego-Network around alerted accounts to visualize Mule Syndicate Rings.
  - Exports interactive HTML graph visualizations via Pyvis.
"""

from typing import Dict, Any, List, Tuple
import os
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
import xgboost as xgb


def get_global_feature_importance(
    model: xgb.XGBClassifier, feature_names: List[str], top_n: int = 15
) -> pd.DataFrame:
    """Extracts top-N feature importances by Gain."""
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type="gain")

    records = []
    for feat, gain in score_dict.items():
        # Feature names might be f0, f1... or original names
        idx = int(feat[1:]) if feat.startswith("f") and feat[1:].isdigit() else None
        name = feature_names[idx] if idx is not None and idx < len(feature_names) else feat
        records.append({"feature": name, "importance_gain": float(gain)})

    df_imp = pd.DataFrame(records).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return df_imp.head(top_n)


def extract_syndicate_subgraph(
    edges_df: pd.DataFrame,
    target_account: str,
    hops: int = 2,
    max_neighbors: int = 15,
) -> nx.DiGraph:
    """
    Extracts a local directed ego-network around target_account up to k-hops.
    Caps max neighbors per node to prevent super-node explosion.
    """
    G = nx.DiGraph()

    # Identify nodes in k-hop
    current_nodes = {target_account}
    visited_nodes = set(current_nodes)

    for _ in range(hops):
        next_nodes = set()
        for node in current_nodes:
            # Outgoing edges
            out_edges = edges_df[edges_df["src_id"] == node].head(max_neighbors)
            for _, r in out_edges.iterrows():
                G.add_edge(
                    r["src_id"],
                    r["dst_id"],
                    amount=float(r["amount"]),
                    timestamp=int(r["timestamp"]),
                    is_fraud=int(r["is_fraud"]),
                    pattern=str(r.get("pattern", "NORMAL")),
                )
                next_nodes.add(r["dst_id"])

            # Incoming edges
            in_edges = edges_df[edges_df["dst_id"] == node].head(max_neighbors)
            for _, r in in_edges.iterrows():
                G.add_edge(
                    r["src_id"],
                    r["dst_id"],
                    amount=float(r["amount"]),
                    timestamp=int(r["timestamp"]),
                    is_fraud=int(r["is_fraud"]),
                    pattern=str(r.get("pattern", "NORMAL")),
                )
                next_nodes.add(r["src_id"])

        current_nodes = next_nodes - visited_nodes
        visited_nodes.update(next_nodes)

    return G


def export_interactive_graph_html(
    G: nx.DiGraph,
    output_html_path: str,
    target_account: str,
    nodes_df: pd.DataFrame = None,
) -> str:
    """
    Renders the directed subgraph as an interactive HTML file with Pyvis.
    Color coding:
      - Target Alerted Account: Amber / Orange
      - Syndicate Mule: Red (#FF4136)
      - Normal Account: Emerald Green (#2ECC40)
      - Merchant: Blue (#0074D9)
    """
    net = Network(height="600px", width="100%", directed=True, bgcolor="#1E1E2E", font_color="white")
    net.force_atlas_2based()

    mule_lookup = {}
    if nodes_df is not None:
        mule_lookup = dict(zip(nodes_df["account_id"], nodes_df["is_mule"]))

    for node in G.nodes():
        is_target = node == target_account
        is_mule = mule_lookup.get(node, 0) == 1 or node.startswith("SYN_")
        is_merch = node.startswith("MER_")

        if is_target:
            color = "#FF851B"  # Orange
            size = 35
            title = f"TARGET ALERT: {node}"
        elif is_mule:
            color = "#FF4136"  # Red
            size = 28
            title = f"MULE SYNDICATE: {node}"
        elif is_merch:
            color = "#0074D9"  # Blue
            size = 24
            title = f"MERCHANT: {node}"
        else:
            color = "#2ECC40"  # Green
            size = 18
            title = f"USER: {node}"

        net.add_node(node, label=node, color=color, size=size, title=title)

    for u, v, data in G.edges(data=True):
        is_fraud = data.get("is_fraud", 0) == 1
        edge_color = "#FF4136" if is_fraud else "#888888"
        width = 3.0 if is_fraud else 1.0
        label_text = f"Rp {data.get('amount', 0):,.0f}"

        net.add_edge(u, v, color=edge_color, width=width, title=label_text)

    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    net.write_html(output_html_path)
    return output_html_path


if __name__ == "__main__":
    from generator import GeneratorConfig, generate_fintech_network

    print("Generating network for explainability demo...")
    n_df, e_df = generate_fintech_network(GeneratorConfig())
    mule_txs = e_df[e_df["is_fraud"] == 1]
    sample_mule = mule_txs.iloc[0]["dst_id"]

    print(f"Extracting 2-hop ego-network around syndicate mule {sample_mule}...")
    subgraph = extract_syndicate_subgraph(e_df, sample_mule, hops=2)
    print(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    out_file = r"d:\Data Science\syndicate_guard\artifacts\syndicate_subgraph.html"
    export_interactive_graph_html(subgraph, out_file, sample_mule, n_df)
    print(f"Interactive Pyvis graph saved to: {out_file}")
