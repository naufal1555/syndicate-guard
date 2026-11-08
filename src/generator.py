"""
Synthetic FinTech Transaction & Mule Account Generator (SyndicateGuard)
Author: Principal Data Scientist & Software Engineer
Purpose: Generates realistic FinTech e-Wallet network graphs with injected money laundering topologies.
Design Patterns:
  - Vectorized operations where possible.
  - Deterministic PRNG via numpy.random.default_rng.
  - Clear separation of normal vs syndicate behaviors.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratorConfig:
    num_normal_users: int = 5000
    num_merchants: int = 200
    num_syndicates: int = 35
    syndicate_size_range: Tuple[int, int] = (6, 12)
    start_timestamp: int = 1_700_000_000  # Unix timestamp
    duration_days: int = 30
    random_seed: int = 42


def generate_fintech_network(config: GeneratorConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic account nodes and transaction edges.
    Returns:
        nodes_df: pd.DataFrame with node features and target labels (is_mule).
        edges_df: pd.DataFrame with transaction edges, amounts, timestamps, and fraud labels.
    """
    rng = np.random.default_rng(config.random_seed)
    end_timestamp = config.start_timestamp + config.duration_days * 86400

    # -------------------------------------------------------------
    # 1. Generate Nodes (Accounts)
    # -------------------------------------------------------------
    # User accounts
    user_ids = [f"USR_{i:05d}" for i in range(config.num_normal_users)]
    # Merchant accounts
    merchant_ids = [f"MER_{i:04d}" for i in range(config.num_merchants)]
    
    # Syndicate accounts
    syndicate_accounts: Dict[int, List[str]] = {}
    syndicate_node_list: List[str] = []
    
    current_syn_idx = 0
    for syn_id in range(config.num_syndicates):
        size = rng.integers(config.syndicate_size_range[0], config.syndicate_size_range[1] + 1)
        members = [f"SYN_{current_syn_idx + j:05d}" for j in range(size)]
        current_syn_idx += size
        syndicate_accounts[syn_id] = members
        syndicate_node_list.extend(members)

    all_accounts = user_ids + merchant_ids + syndicate_node_list
    total_nodes = len(all_accounts)

    # Node attributes
    node_types = (
        ["USER"] * len(user_ids)
        + ["MERCHANT"] * len(merchant_ids)
        + ["SYNDICATE_MULE"] * len(syndicate_node_list)
    )
    is_mule = [0] * (len(user_ids) + len(merchant_ids)) + [1] * len(syndicate_node_list)
    
    # Account creation timestamps (older for legitimate, newer/clustered for mules)
    user_creations = rng.integers(
        config.start_timestamp - 365 * 86400,
        config.start_timestamp + (config.duration_days // 2) * 86400,
        size=len(user_ids)
    )
    merchant_creations = rng.integers(
        config.start_timestamp - 730 * 86400,
        config.start_timestamp,
        size=len(merchant_ids)
    )
    # Mules often register shortly before fraud campaign
    mule_creations = rng.integers(
        config.start_timestamp,
        end_timestamp - 5 * 86400,
        size=len(syndicate_node_list)
    )
    created_at = np.concatenate([user_creations, merchant_creations, mule_creations])

    # Initial Balances (in IDR)
    user_bal = rng.lognormal(mean=12.0, sigma=1.0, size=len(user_ids))  # Median ~ 160k
    merch_bal = rng.lognormal(mean=15.0, sigma=0.8, size=len(merchant_ids))  # Median ~ 3.2M
    mule_bal = rng.uniform(10_000, 50_000, size=len(syndicate_node_list))  # Low initial
    initial_balance = np.round(np.concatenate([user_bal, merch_bal, mule_bal]), -3)

    # KYC Levels: 0 = Basic (unverified), 1 = Verified (KTP)
    kyc_user = rng.choice([0, 1], p=[0.4, 0.6], size=len(user_ids))
    kyc_merch = np.ones(len(merchant_ids), dtype=int)  # Merchants are always verified
    kyc_mule = rng.choice([0, 1], p=[0.85, 0.15], size=len(syndicate_node_list))  # Mostly unverified
    kyc_level = np.concatenate([kyc_user, kyc_merch, kyc_mule])

    nodes_df = pd.DataFrame({
        "account_id": all_accounts,
        "account_type": node_types,
        "created_at": created_at,
        "initial_balance": initial_balance,
        "kyc_level": kyc_level,
        "is_mule": is_mule,
    })

    # -------------------------------------------------------------
    # 2. Generate Legitimate Transactions (P2P + P2M)
    # -------------------------------------------------------------
    num_legit_tx = 45000
    
    # 70% P2M (User to Merchant), 30% P2P (User to User)
    num_p2m = int(num_legit_tx * 0.70)
    num_p2p = num_legit_tx - num_p2m

    # P2M transactions
    p2m_src = rng.choice(user_ids, size=num_p2m)
    p2m_dst = rng.choice(merchant_ids, size=num_p2m)
    p2m_amount = np.round(rng.lognormal(mean=10.5, sigma=0.8, size=num_p2m), -3) + 5000  # 15k - 200k
    p2m_time = rng.integers(config.start_timestamp, end_timestamp, size=num_p2m)

    # P2P transactions
    p2p_src = rng.choice(user_ids, size=num_p2p)
    p2p_dst = rng.choice(user_ids, size=num_p2p)
    # Avoid self loops
    mask_self = p2p_src == p2p_dst
    p2p_dst[mask_self] = rng.choice(user_ids, size=np.sum(mask_self))
    p2p_amount = np.round(rng.lognormal(mean=11.5, sigma=1.0, size=num_p2p), -3) + 10000  # 30k - 500k
    p2p_time = rng.integers(config.start_timestamp, end_timestamp, size=num_p2p)

    legit_df = pd.DataFrame({
        "src_id": np.concatenate([p2m_src, p2p_src]),
        "dst_id": np.concatenate([p2m_dst, p2p_dst]),
        "amount": np.concatenate([p2m_amount, p2p_amount]),
        "timestamp": np.concatenate([p2m_time, p2p_time]),
        "tx_type": ["P2M"] * num_p2m + ["P2P"] * num_p2p,
        "is_fraud": 0,
        "syndicate_id": -1,
        "pattern": "NORMAL"
    })

    # -------------------------------------------------------------
    # 3. Inject Syndicate Money Laundering Topologies
    # -------------------------------------------------------------
    syndicate_tx_list: List[Dict[str, Any]] = []

    for syn_id, members in syndicate_accounts.items():
        # Campaign starts distributed across the full simulation period, allowing activity in train, val, and test splits
        campaign_start = rng.integers(config.start_timestamp + 86400, end_timestamp - 3600 * 6)
        
        # Designate roles within the syndicate:
        # First 40% are smurfs (feeders)
        # Middle 40% are layering mules (intermediaries)
        # Last 20% are cash-out mules (off-ramps)
        n_m = len(members)
        split_smurf = max(2, int(n_m * 0.45))
        split_layer = max(split_smurf + 1, int(n_m * 0.80))
        
        smurfs = members[:split_smurf]
        layer_mules = members[split_smurf:split_layer]
        cashouts = members[split_layer:]
        if not cashouts:
            cashouts = [members[-1]]
            layer_mules = members[split_smurf:-1]

        # Stage 1: Smurfing (Fan-In)
        # Multiple smurfs transfer fragmented amounts to the first layering mule in quick bursts
        target_mule_1 = layer_mules[0]
        t_current = campaign_start
        total_collected = 0.0

        for smurf in smurfs:
            burst_count = rng.integers(3, 7)
            for _ in range(burst_count):
                amt = float(np.round(rng.uniform(50_000, 450_000), -3))
                total_collected += amt
                t_current += rng.integers(30, 300)  # 30s to 5 mins
                syndicate_tx_list.append({
                    "src_id": smurf,
                    "dst_id": target_mule_1,
                    "amount": amt,
                    "timestamp": t_current,
                    "tx_type": "P2P",
                    "is_fraud": 1,
                    "syndicate_id": syn_id,
                    "pattern": "SMURFING_FAN_IN"
                })

        # Stage 2: Layering (Peeling Chain)
        # Funds are rapidly passed sequentially through intermediary mules with small retention
        active_balance = total_collected
        current_node = target_mule_1

        for next_mule in layer_mules[1:]:
            t_current += rng.integers(60, 600)  # Rapid hops (< 10 mins)
            peeled_fee = active_balance * rng.uniform(0.01, 0.03)  # 1-3% fee
            transfer_amt = float(np.round(active_balance - peeled_fee, -3))
            active_balance = transfer_amt
            
            syndicate_tx_list.append({
                "src_id": current_node,
                "dst_id": next_mule,
                "amount": transfer_amt,
                "timestamp": t_current,
                "tx_type": "P2P",
                "is_fraud": 1,
                "syndicate_id": syn_id,
                "pattern": "LAYERING_CHAIN"
            })
            current_node = next_mule

        # Stage 3: Cash-out (Fan-Out)
        # Final mule rapidly unloads to cashout nodes or external payment points
        for cashout_node in cashouts:
            t_current += rng.integers(120, 900)
            portion = active_balance / len(cashouts)
            cashout_amt = float(np.round(portion, -3))
            syndicate_tx_list.append({
                "src_id": current_node,
                "dst_id": cashout_node,
                "amount": cashout_amt,
                "timestamp": t_current,
                "tx_type": "P2P",
                "is_fraud": 1,
                "syndicate_id": syn_id,
                "pattern": "CASHOUT_FAN_OUT"
            })

    syn_df = pd.DataFrame(syndicate_tx_list)
    
    # -------------------------------------------------------------
    # 4. Merge, Sort by Time, and Ensure Integrity
    # -------------------------------------------------------------
    edges_df = pd.concat([legit_df, syn_df], ignore_index=True)
    edges_df = edges_df.sort_values("timestamp").reset_index(drop=True)
    
    # Add unique transaction ID
    edges_df.insert(0, "tx_id", [f"TX_{i:07d}" for i in range(len(edges_df))])

    return nodes_df, edges_df


if __name__ == "__main__":
    cfg = GeneratorConfig()
    print("Generating FinTech network...")
    n_df, e_df = generate_fintech_network(cfg)
    print(f"Generated Nodes: {len(n_df)} (Mules: {n_df['is_mule'].sum()})")
    print(f"Generated Edges: {len(e_df)} (Fraud TX: {e_df['is_fraud'].sum()})")
    print(f"Fraud Transaction Ratio: {e_df['is_fraud'].mean():.4%}")
