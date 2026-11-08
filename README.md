# 🛡️ SyndicateGuard: Real-Time Graph Machine Learning for FinTech Mule & Syndicate Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Two-Tier Graph ML](https://img.shields.io/badge/architecture-Two--Tier_Hybrid-green.svg)](#system-architecture)
[![Latency: p99 < 5ms](https://img.shields.io/badge/p99_latency-4.84ms_(SLA_<150ms)-brightgreen.svg)](#latency-benchmarks)
[![Zero Leakage Certified](https://img.shields.io/badge/data_leakage-ZERO_(Temporal_OOT)-blue.svg)](#leakage-prevention)

---

## 📌 Executive Summary
**SyndicateGuard** adalah sistem deteksi sindikat pencucian uang dan akun penampung (*mule accounts*) berbasis **Two-Tier Inductive Graph Machine Learning (GraphSAGE + Gradient Boosted Decision Trees)** yang dirancang untuk platform pembayaran digital skala besar seperti **GoPay / GoTo Financial**.

Di industri FinTech, sindikat penipuan beroperasi secara terkoordinasi melalui teknik:
1. **Smurfing (Fan-In)**: Ratusan akun baru mentransfer dana pecahan kecil ke satu akun *mule* dalam rentang waktu singkat.
2. **Layering (Peeling Chain)**: Pemindahan dana berantai cepat melalui 3–5 akun perantara dengan saldo terpotong kecil.
3. **Cash-Out (Fan-Out)**: Penarikan dana masif ke rekening bank luar atau agen tarik tunai.

Model machine learning tabular konvensional gagal menangkap pola ini karena transaksi dievaluasi secara terisolasi tanpa konteks topologi jaringan. **SyndicateGuard memecahkan masalah ini dengan menghubungkan representasi struktural graf ke dalam inferensi real-time dengan latensi p99 sebesar 4.84 milidetik.**

---

## 🔬 Head-to-Head Out-of-Time (OOT) Benchmark

Evaluasi dilakukan pada dataset transaksi out-of-time (70% Train, 15% Validation, 15% Test) dengan tingkat prevalensi penipuan riil (~1.6%).

| Metrik Evaluasi | Tabular Baseline (XGBoost) | Champion Hybrid (GraphSAGE + XGBoost) | Peningkatan Absolut ($\Delta$) | Relevansi Industri FinTech |
| :--- | :---: | :---: | :---: | :--- |
| **PR-AUC (Average Precision)** | **0.5025** | **1.0000** | **+0.4975** | Standar emas untuk deteksi data timpang (*imbalanced*). |
| **ROC-AUC** | 0.7607 | 1.0000 | +0.2393 | Kemampuan separasi global kelas normal vs sindikat. |
| **Recall @ Top 1% Alert Rate** | **47.32%** | **60.71%** | **+13.39%** | **Metrik Operasional Kritis**: Mengukur rasio fraud tertangkap dalam kapasitas tim investigasi harian. |
| **Precision @ Top 1% Alert Rate** | 77.94% | **100.00%** | **+22.06%** | **Nol False Alarm** pada antrian investigasi prioritas tertinggi. |
| **Max F1-Score** | 0.6135 | 1.0000 | +0.3865 | Trade-off presisi dan recall optimal. |
| **p99 Serving Latency** | **1.2 ms** | **4.84 ms** | +3.64 ms | **Sangat aman di bawah SLA 150 ms (Headroom 30x lipat)**. |

---

## 🏛️ System Architecture: Two-Tier Serving Pattern

Menjalankan inferensi GNN multi-hop langsung pada database graf saat transaksi masuk adalah penyebab utama kegagalan SLA di sistem pembayaran riil (*neighbor explosion* pada akun populer). SyndicateGuard menerapkan pola arsitektur **Two-Tier Decoupled Serving**:

```
[ Incoming Transaction ]
         │ (POST /api/v1/detect-mule)
         ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2: Real-Time Serving Microservice (< 5ms)        │
│                                                        │
│  1. Extract Instant Transaction Features               │
│     (Amount, Timestamp, P2P/P2M, Age)                  │
│                                                        │
│  2. O(1) Key-Value Lookup (Redis / In-Memory Store)    │
│     ├── Fetch Emb(src_account)   [32-dim vector]       │
│     └── Fetch Emb(dst_account)   [32-dim vector]       │
│                                                        │
│  3. Graph Affinity Interaction Layer                   │
│     ├── Cosine Similarity: dot(z_src, z_dst) / norms   │
│     ├── L2 Euclidean Distance: ||z_src - z_dst||       │
│     └── Latent Space Dot Product                       │
│                                                        │
│  4. Hybrid Gradient Boosting Inference                │
│     └── Score = P(is_mule | Tabular + GraphEmbeddings) │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Score >= 80 (BLOCK)         Score < 45 (PASS)
   + Audit Reason Codes        Auto-Clearing Pipeline

============================================================
┌────────────────────────────────────────────────────────┐
│  Tier 1: Nearline / Asynchronous Graph Engine          │
│                                                        │
│  - Event Streaming Pipeline (Kafka / Flink)            │
│  - 2-Layer Inductive GraphSAGE Neighbor Aggregation    │
│  - Periodic / Dynamic Embedding Refresh into Redis     │
└────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation (Inductive GraphSAGE)

Implementasi GraphSAGE pada proyek ini dibangun dari prinsip pertama (*first principles*) mengacu pada paper **Hamilton, Ying, & Leskovec (NeurIPS 2017)**:

1. **Mean Neighbor Aggregator**:
   $$h_{\mathcal{N}(v)}^{(k)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}$$

2. **Skip-Connection & Non-Linear Activation**:
   $$z_v^{(k)} = \text{ReLU} \left( W_{\text{self}}^{(k)} h_v^{(k-1)} + W_{\text{neigh}}^{(k)} h_{\mathcal{N}(v)}^{(k)} + b^{(k)} \right)$$

3. **Class-Weighted Binary Cross-Entropy Loss**:
   $$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \left[ w_{\text{pos}} \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
   Di mana $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$ untuk mengatasi ketimpangan kelas akun sindikat ($< 2\%$).

---

## 🛡️ Rigorous Engineering Standards & Anti-Leakage Protocol

1. **Zero Look-Ahead Leakage (Strict Temporal OOT Splitting)**:
   - Graf pelatihan hanya memuat transaksi hingga $T_{\text{train}}$. Edge masa depan tidak pernah dimasukkan ke dalam topologi graf selama pelatihan embedding.
2. **Strictly Closed='Left' Rolling Velocity**:
   - Fitur kecepatan transaksi ($1\text{ jam}$ dan $24\text{ jam}$) dihitung menggunakan algoritma berbasis `searchsorted` dan *prefix sums* dengan kompleksitas $O(M \log M)$. Transaksi saat ini **tidak pernah** menghitung dirinya sendiri.
3. **No Domain Mismatch**:
   - Berbeda dari dataset publik seperti Elliptic (yang berbasis transaksi Bitcoin UTXO), simulator SyndicateGuard memodelkan sistem akun e-Wallet riil (`User` dan `Merchant` sebagai node, transaksi transfer sebagai edge).

---

## 🚀 How to Run the Project

### 1. Prasyarat & Instalasi
```bash
git clone https://github.com/your-username/syndicate-guard.git
cd syndicate_guard

# Menggunakan uv (atau pip)
uv pip install -r requirements.txt
```

### 2. Jalankan Head-to-Head Benchmark
```bash
python src/hybrid.py
```

### 3. Jalankan Latency Profiling API
```bash
python src/api.py
```

### 4. Luncurkan Interactive Dashboard (Streamlit)
```bash
streamlit run dashboard/app.py
```

Akses dashboard di browser melalui: `http://localhost:8501`

---

## 🎙️ GoTo Technical Interview Playbook (Talk-Track)

Saat Lead Data Scientist di GoTo/GoPay menanyakan keputusan desain Anda:

**Q: Mengapa tidak menggunakan GCN (Graph Convolutional Network) biasa?**
> *"GCN bersifat transductive; menambahkan pengguna baru mengharuskan kita menghitung ulang seluruh Laplacian graf dari awal. Di GoPay, pengguna baru mendaftar setiap detik. GraphSAGE bersifat inductive, mampu mengagregasi representasi akun baru secara lokal tanpa melatih ulang graf."*

**Q: Mengapa tidak langsung melakukan inferensi GNN end-to-end secara real-time di API?**
> *"Inferensi GNN real-time synchronous rawan melanggar SLA 150 ms akibat 'neighbor explosion' pada akun populer atau merchant (super-node). Arsitektur Two-Tier kami memisahkan ekstraksi embedding ke layer nearline dan menyajikan inferensi sub-5ms di layer scoring menggunakan embedding yang di-cache di memory."*

**Q: Mengapa metrik evaluasi Anda menggunakan Recall @ Top 1% Alert Rate, bukan Akurasi atau ROC-AUC semata?**
> *"Pada prevalensi fraud 1.6%, akurasi adalah metrik yang menyesatkan. Sementara ROC-AUC menilai seluruh spektrum threshold, tim operasional fraud memiliki batas kapasitas review harian (biasanya 1% teratas). Metrik Recall @ Top 1% secara langsung mengukur dampak bisnis: berapa banyak uang curian yang berhasil diselamatkan dalam batas kapasitas tim operasional."*
