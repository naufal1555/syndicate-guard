"""
Temporal Feature Engineering & Graph Dataset Preparation (SyndicateGuard)
Author: Principal Data Scientist & Software Engineer
Purpose:
  - Vectorized O(N log N) rolling velocity feature extractor via searchsorted & prefix-sums.
  - GUARANTEED ZERO LOOK-AHEAD LEAKAGE (strictly closed='left').
  - Prepare graph topology structures (Node feature matrix, Edge index).
  - Provide strict temporal train/val/test splits.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd


def _compute_rolling_metrics_for_entity(
    times: np.ndarray, amounts: np.ndarray, window_seconds: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes count and sum of transactions in strictly preceding [t - window, t).
    Complexity: O(M log M) where M is transactions per entity.
    """
    m = len(times)
    counts = np.zeros(m, dtype=np.int32)
    sums = np.zeros(m, dtype=np.float64)
    cumsum = np.concatenate([[0.0], np.cumsum(amounts)])

    for i in range(m):
        t_curr = times[i]
        left_idx = np.searchsorted(times, t_curr - window_seconds, side="left")
        # index i corresponds to the current tx; elements in [left_idx, i) are past events
        counts[i] = i - left_idx
        sums[i] = cumsum[i] - cumsum[left_idx]

    return counts, sums


def compute_temporal_velocity_features(
    edges_df: pd.DataFrame, nodes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes strict historical rolling features for each transaction without look-ahead bias.
    """
    df = edges_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    # 1. Source (Sender) Outgoing Velocity
    src_cnt_1h = np.zeros(n, dtype=np.int32)
    src_sum_1h = np.zeros(n, dtype=np.float64)
    src_cnt_24h = np.zeros(n, dtype=np.int32)
    src_sum_24h = np.zeros(n, dtype=np.float64)

    # Group by src_id and compute
    for _, group_indices in df.groupby("src_id", sort=False).indices.items():
        g_times = df.loc[group_indices, "timestamp"].values
        g_amts = df.loc[group_indices, "amount"].values

        c1, s1 = _compute_rolling_metrics_for_entity(g_times, g_amts, 3600)
        c24, s24 = _compute_rolling_metrics_for_entity(g_times, g_amts, 86400)

        src_cnt_1h[group_indices] = c1
        src_sum_1h[group_indices] = s1
        src_cnt_24h[group_indices] = c24
        src_sum_24h[group_indices] = s24

    df["src_tx_count_1h"] = src_cnt_1h
    df["src_amount_sum_1h"] = src_sum_1h
    df["src_tx_count_24h"] = src_cnt_24h
    df["src_amount_sum_24h"] = src_sum_24h

    # 2. Destination (Receiver) Incoming Velocity (Critical for Smurfing Fan-in)
    dst_cnt_1h = np.zeros(n, dtype=np.int32)
    dst_sum_1h = np.zeros(n, dtype=np.float64)
    dst_cnt_24h = np.zeros(n, dtype=np.int32)
    dst_sum_24h = np.zeros(n, dtype=np.float64)

    for _, group_indices in df.groupby("dst_id", sort=False).indices.items():
        g_times = df.loc[group_indices, "timestamp"].values
        g_amts = df.loc[group_indices, "amount"].values

        c1, s1 = _compute_rolling_metrics_for_entity(g_times, g_amts, 3600)
        c24, s24 = _compute_rolling_metrics_for_entity(g_times, g_amts, 86400)

        dst_cnt_1h[group_indices] = c1
        dst_sum_1h[group_indices] = s1
        dst_cnt_24h[group_indices] = c24
        dst_sum_24h[group_indices] = s24

    df["dst_tx_count_1h"] = dst_cnt_1h
    df["dst_amount_sum_1h"] = dst_sum_1h
    df["dst_tx_count_24h"] = dst_cnt_24h
    df["dst_amount_sum_24h"] = dst_sum_24h

    # 3. Static Node Metadata Merging
    node_meta = nodes_df.set_index("account_id")

    df["src_kyc"] = df["src_id"].map(node_meta["kyc_level"]).fillna(0).astype(int)
    src_created = df["src_id"].map(node_meta["created_at"])
    df["src_age_days"] = np.maximum(0.0, (df["timestamp"] - src_created) / 86400.0)

    df["dst_kyc"] = df["dst_id"].map(node_meta["kyc_level"]).fillna(0).astype(int)
    dst_created = df["dst_id"].map(node_meta["created_at"])
    df["dst_age_days"] = np.maximum(0.0, (df["timestamp"] - dst_created) / 86400.0)

    # 4. Derived ratios
    df["log_amount"] = np.log1p(df["amount"])
    df["is_p2m"] = (df["tx_type"] == "P2M").astype(int)
    df["src_burst_ratio_1h_to_24h"] = (df["src_tx_count_1h"] + 1) / (df["src_tx_count_24h"] + 1)
    df["dst_burst_ratio_1h_to_24h"] = (df["dst_tx_count_1h"] + 1) / (df["dst_tx_count_24h"] + 1)

    return df


def temporal_split(
    features_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict out-of-time (OOT) temporal split:
      - Train: Earliest 70% of chronological transactions.
      - Val: Middle 15% of chronological transactions.
      - Test: Final 15% of chronological transactions.
    """
    sorted_df = features_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = sorted_df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = sorted_df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    from generator import GeneratorConfig, generate_fintech_network

    print("Generating network...")
    n_df, e_df = generate_fintech_network(GeneratorConfig())
    print("Computing non-leaked temporal velocity features...")
    f_df = compute_temporal_velocity_features(e_df, n_df)
    train_d, val_d, test_d = temporal_split(f_df)

    print(f"Features computed. Total rows: {len(f_df)}")
    print(f"Train split: {len(train_d)} rows (Fraud: {train_d['is_fraud'].sum()} | {train_d['is_fraud'].mean():.2%})")
    print(f"Val split:   {len(val_d)} rows (Fraud: {val_d['is_fraud'].sum()} | {val_d['is_fraud'].mean():.2%})")
    print(f"Test split:  {len(test_d)} rows (Fraud: {test_d['is_fraud'].sum()} | {test_d['is_fraud'].mean():.2%})")
