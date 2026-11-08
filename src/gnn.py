"""
Vectorized GraphSAGE (Graph Sample and Aggregate) - First Principles Implementation
Author: Principal Data Scientist & Software Engineer
Reference: Hamilton, Ying, & Leskovec (NeurIPS 2017) "Inductive Representation Learning on Large Graphs"
Purpose:
  - Production-grade, zero-external-binary inductive GraphSAGE implementation in pure NumPy/SciPy.
  - Implements Mean Neighbor Aggregator: h_v^(k) = ReLU(W_self * h_v^(k-1) + W_neigh * Mean_{u in N(v)}(h_u^(k-1))).
  - Trains via Adam optimizer on class-weighted BCE loss for mule account identification.
  - Generates 32-dimensional inductive node embeddings for nearline caching and hybrid serving.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


class InductiveGraphSAGE:
    """
    2-Layer GraphSAGE with Mean Aggregation and skip-connections (W_self + W_neigh).
    Strictly inductive: Can compute embeddings for arbitrary subgraphs or unseen nodes given features and local edges.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 32,
        learning_rate: float = 0.02,
        random_seed: int = 42,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.rng = np.random.default_rng(random_seed)

        # Xavier/Glorot Initialization
        glorot1 = np.sqrt(2.0 / (in_dim + hidden_dim))
        self.W1_self = self.rng.normal(0, glorot1, (in_dim, hidden_dim)).astype(np.float32)
        self.W1_neigh = self.rng.normal(0, glorot1, (in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        glorot2 = np.sqrt(2.0 / (hidden_dim + embedding_dim))
        self.W2_self = self.rng.normal(0, glorot2, (hidden_dim, embedding_dim)).astype(np.float32)
        self.W2_neigh = self.rng.normal(0, glorot2, (hidden_dim, embedding_dim)).astype(np.float32)
        self.b2 = np.zeros(embedding_dim, dtype=np.float32)

        glorot_clf = np.sqrt(2.0 / (embedding_dim + 1))
        self.W_clf = self.rng.normal(0, glorot_clf, (embedding_dim, 1)).astype(np.float32)
        self.b_clf = np.zeros(1, dtype=np.float32)

        # Adam optimizer state variables
        self.m = {}
        self.v = {}
        for param_name in ["W1_self", "W1_neigh", "b1", "W2_self", "W2_neigh", "b2", "W_clf", "b_clf"]:
            p = getattr(self, param_name)
            self.m[param_name] = np.zeros_like(p)
            self.v[param_name] = np.zeros_like(p)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def forward(
        self, X: np.ndarray, A_norm: sp.csr_matrix
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through 2 GraphSAGE layers.
        Returns:
            probs: (N, 1) probability of being a syndicate mule account.
            embeddings: (N, embedding_dim) latent graph structural representation.
            cache: intermediate activations for backpropagation.
        """
        # Layer 1: Mean Neighbor Aggregation + Self Transformation
        X_neigh = A_norm.dot(X)
        z1 = X @ self.W1_self + X_neigh @ self.W1_neigh + self.b1
        h1 = self._relu(z1)

        # Layer 2: Second Hop Aggregation
        h1_neigh = A_norm.dot(h1)
        z2 = h1 @ self.W2_self + h1_neigh @ self.W2_neigh + self.b2
        embeddings = self._relu(z2)

        # Classifier Layer
        logits = embeddings @ self.W_clf + self.b_clf
        probs = self._sigmoid(logits)

        cache = {
            "X": X,
            "X_neigh": X_neigh,
            "z1": z1,
            "h1": h1,
            "h1_neigh": h1_neigh,
            "z2": z2,
            "embeddings": embeddings,
            "logits": logits,
            "probs": probs,
        }
        return probs, embeddings, cache

    def train_step(
        self,
        X: np.ndarray,
        A_norm: sp.csr_matrix,
        y: np.ndarray,
        pos_weight: float,
        step: int,
    ) -> float:
        """
        Performs one gradient descent step with class-weighted binary cross-entropy and Adam.
        """
        probs, embeddings, cache = self.forward(X, A_norm)
        N = len(y)

        # Gradient of class-weighted BCE with respect to logits:
        # L = - w * y * log(p) - (1-y) * log(1-p)
        # dL/d(logit) = (w * y * (p - 1) + (1-y) * p) / N
        d_logits = (pos_weight * y * (probs - 1.0) + (1.0 - y) * probs) / N

        # Backpropagation into Classifier
        dW_clf = embeddings.T @ d_logits
        db_clf = np.sum(d_logits, axis=0)

        # Backpropagation into Layer 2
        d_emb = d_logits @ self.W_clf.T
        dz2 = d_emb * (cache["z2"] > 0).astype(np.float32)

        dW2_self = cache["h1"].T @ dz2
        dW2_neigh = cache["h1_neigh"].T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Backpropagation into Layer 1
        dh1 = dz2 @ self.W2_self.T + A_norm.T.dot(dz2 @ self.W2_neigh.T)
        dz1 = dh1 * (cache["z1"] > 0).astype(np.float32)

        dW1_self = cache["X"].T @ dz1
        dW1_neigh = cache["X_neigh"].T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Adam Parameter Update
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grads = {
            "W1_self": dW1_self,
            "W1_neigh": dW1_neigh,
            "b1": db1,
            "W2_self": dW2_self,
            "W2_neigh": dW2_neigh,
            "b2": db2,
            "W_clf": dW_clf,
            "b_clf": db_clf,
        }

        for p_name, grad in grads.items():
            self.m[p_name] = beta1 * self.m[p_name] + (1 - beta1) * grad
            self.v[p_name] = beta2 * self.v[p_name] + (1 - beta2) * (grad ** 2)

            m_hat = self.m[p_name] / (1 - beta1 ** step)
            v_hat = self.v[p_name] / (1 - beta2 ** step)

            param = getattr(self, p_name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        # Compute Loss
        loss = -np.mean(
            pos_weight * y * np.log(np.clip(probs, 1e-12, 1.0))
            + (1.0 - y) * np.log(np.clip(1.0 - probs, 1e-12, 1.0))
        )
        return float(loss)


def construct_graph_matrices(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_timestamp: int,
) -> Tuple[np.ndarray, sp.csr_matrix, Dict[str, int]]:
    """
    Constructs normalized adjacency matrix A_norm and node feature matrix X
    strictly using transactions up to max_timestamp (ZERO LEAKAGE).
    """
    account_ids = nodes_df["account_id"].values
    node_to_idx = {acc: i for i, acc in enumerate(account_ids)}
    n = len(account_ids)

    # 1. Filter edges strictly <= max_timestamp
    valid_edges = edges_df[edges_df["timestamp"] <= max_timestamp].copy()
    src_idx = valid_edges["src_id"].map(node_to_idx).dropna().astype(int).values
    dst_idx = valid_edges["dst_id"].map(node_to_idx).dropna().astype(int).values

    # Undirected / bidirectional propagation for collaborative neighbor representation
    row = np.concatenate([src_idx, dst_idx])
    col = np.concatenate([dst_idx, src_idx])
    data = np.ones(len(row), dtype=np.float32)

    adj = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    # Add self loops
    adj = adj + sp.eye(n, dtype=np.float32)

    # Row-normalize (Mean aggregator)
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    d_inv = sp.diags(1.0 / row_sums)
    A_norm = (d_inv @ adj).tocsr()

    # 2. Node Features Matrix
    log_bal = np.log1p(nodes_df["initial_balance"].values)[:, None]
    kyc = nodes_df["kyc_level"].values[:, None]
    age_days = np.maximum(0.0, (max_timestamp - nodes_df["created_at"].values) / 86400.0)[:, None]

    # In-degree & Out-degree within valid window
    in_deg = np.zeros(n, dtype=np.float32)
    out_deg = np.zeros(n, dtype=np.float32)
    for acc, count in valid_edges["dst_id"].value_counts().items():
        if acc in node_to_idx:
            in_deg[node_to_idx[acc]] = count
    for acc, count in valid_edges["src_id"].value_counts().items():
        if acc in node_to_idx:
            out_deg[node_to_idx[acc]] = count

    in_deg_log = np.log1p(in_deg)[:, None]
    out_deg_log = np.log1p(out_deg)[:, None]

    X = np.hstack([log_bal, kyc, age_days, in_deg_log, out_deg_log]).astype(np.float32)
    # Standardize features
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-6
    X_scaled = (X - mean) / std

    return X_scaled, A_norm, node_to_idx


def train_inductive_graphsage(
    X: np.ndarray,
    A_norm: sp.csr_matrix,
    y: np.ndarray,
    epochs: int = 150,
    lr: float = 0.02,
    random_seed: int = 42,
) -> Tuple[InductiveGraphSAGE, np.ndarray, Dict[str, float]]:
    """
    Trains GraphSAGE on node-level mule detection and extracts 32-dim embeddings.
    """
    model = InductiveGraphSAGE(in_dim=X.shape[1], hidden_dim=32, embedding_dim=32, learning_rate=lr, random_seed=random_seed)
    
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    pos_weight = float(neg_count / max(1, pos_count))

    for epoch in range(1, epochs + 1):
        loss = model.train_step(X, A_norm, y, pos_weight, step=epoch)

    probs, embeddings, _ = model.forward(X, A_norm)
    node_pr_auc = float(average_precision_score(y, probs))
    node_roc_auc = float(roc_auc_score(y, probs))

    metrics = {
        "node_pr_auc": node_pr_auc,
        "node_roc_auc": node_roc_auc,
        "final_loss": loss,
    }
    return model, embeddings, metrics


if __name__ == "__main__":
    from generator import GeneratorConfig, generate_fintech_network
    from features import compute_temporal_velocity_features, temporal_split

    print("Generating network & features...")
    n_df, e_df = generate_fintech_network(GeneratorConfig())
    f_df = compute_temporal_velocity_features(e_df, n_df)
    train_d, val_d, test_d = temporal_split(f_df)

    t_max = int(train_d["timestamp"].max())
    print(f"Constructing Graph up to T_train={t_max} (Strictly No Leakage)...")
    X, A_norm, n2i = construct_graph_matrices(n_df, e_df, max_timestamp=t_max)
    y_nodes = n_df["is_mule"].values[:, None].astype(np.float32)

    print("Training Inductive GraphSAGE...")
    sage_model, node_embs, sage_metrics = train_inductive_graphsage(X, A_norm, y_nodes, epochs=120)
    print("--- GRAPHSAGE NODE-LEVEL RESULTS ---")
    for k, v in sage_metrics.items():
        print(f"{k:20s}: {v:.4f}")
    print(f"Extracted Node Embeddings Shape: {node_embs.shape}")
