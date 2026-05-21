import json
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    PREDICTIONS_CSV, METRICS_JSON, ROC_NPZ, PR_NPZ,
    EMBEDDINGS_NPZ, TRAINING_LOG, UTU_EDGES_CSV, RAW_CSV,
    MIN_CLIQUE_SIZE, MAX_CLIQUE_NODES, CLIQUE_DEFS,
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_CSV)
    df["text_short"] = df["text"].astype(str).str[:60]
    return df


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_JSON) as f:
        return json.load(f)


@st.cache_data
def load_roc() -> dict:
    raw = np.load(ROC_NPZ)
    return {k: raw[k].copy() for k in raw.files}


@st.cache_data
def load_pr() -> dict:
    raw = np.load(PR_NPZ)
    return {k: raw[k].copy() for k in raw.files}


@st.cache_data
def load_embeddings() -> pd.DataFrame:
    raw = np.load(EMBEDDINGS_NPZ)
    emb = pd.DataFrame({
        "node_id":    raw["node_id"].astype(int),
        "x":          raw["x"],
        "y":          raw["y"],
        "true_label": raw["true_label"].astype(int),
        "fraud_prob": raw["fraud_prob"],
    })
    pred = load_predictions()
    test_df = pred[pred["split"] == "test"][["node_id", "rating", "text", "text_short", "pred_label"]].copy()
    emb = emb.merge(test_df, on="node_id", how="left")
    return emb


@st.cache_data
def load_training_log() -> dict:
    with open(TRAINING_LOG) as f:
        return json.load(f)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, parse_dates=["date"])
    df["month"] = df["date"].dt.month
    return df


@st.cache_data
def load_edges():
    df = pd.read_csv(UTU_EDGES_CSV)
    src = df["src"].values.astype(np.int64)
    dst = df["dst"].values.astype(np.int64)
    w   = df["cosine_sim"].values.astype(np.float32)
    return src, dst, w, df


@st.cache_data
def _load_edges_threshold(threshold_int: int):
    """threshold_int: 4,5,6,7,8,9 → utu_edges_04.csv … utu_edges_09.csv"""
    path = UTU_EDGES_CSV.parent / f"utu_edges_{threshold_int:02d}.csv"
    df = pd.read_csv(path)
    src = df["src"].values.astype(np.int64)
    dst = df["dst"].values.astype(np.int64)
    return src, dst, df


@st.cache_data
def get_clique_edges_filtered(node_ids: frozenset, threshold: float) -> pd.DataFrame:
    t = int(round(threshold * 10))
    src_arr, dst_arr, df = _load_edges_threshold(t)
    ids = np.array(list(node_ids), dtype=np.int64)
    mask = np.isin(src_arr, ids) & np.isin(dst_arr, ids)
    return df[mask].reset_index(drop=True)


@st.cache_data
def build_clique_stats(cols: tuple) -> pd.DataFrame:
    pred = load_predictions()
    agg = (
        pred.groupby(list(cols))
        .agg(
            node_count=("node_id", "count"),
            avg_fraud_prob=("fraud_prob", "mean"),
            fraud_ratio=("label", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["node_count"] >= MIN_CLIQUE_SIZE].copy()
    return agg


@st.cache_data
def get_clique_edges(node_ids: frozenset) -> pd.DataFrame:
    src_arr, dst_arr, _, df = load_edges()
    ids = np.array(list(node_ids), dtype=np.int64)
    mask = np.isin(src_arr, ids) & np.isin(dst_arr, ids)
    return df[mask].reset_index(drop=True)


def get_clique_nodes(pred_df: pd.DataFrame, clique_type: str, row) -> pd.DataFrame:
    cols = CLIQUE_DEFS[clique_type]["cols"]
    return get_clique_nodes_cols(pred_df, cols, row)


def get_clique_nodes_cols(pred_df: pd.DataFrame, cols: tuple, row) -> pd.DataFrame:
    mask = pd.Series(True, index=pred_df.index)
    for col in cols:
        mask &= pred_df[col] == row[col]
    return pred_df[mask].copy()


@st.cache_data
def build_text_clusters(threshold: float) -> pd.DataFrame:
    """텍스트 유사도 엣지로 connected components를 계산해 클러스터 통계 반환."""
    t = int(round(threshold * 10))
    path = UTU_EDGES_CSV.parent / f"utu_edges_{t:02d}.csv"
    edges = pd.read_csv(path)

    G = nx.from_pandas_edgelist(edges, "src", "dst")

    pred = load_predictions()
    label_map = pred.set_index("node_id")["label"].to_dict()

    records = []
    for comp_id, comp in enumerate(
        sorted(nx.connected_components(G), key=len, reverse=True)
    ):
        labels = [label_map.get(n, 0) for n in comp]
        records.append({
            "comp_id":    comp_id,
            "node_count": len(comp),
            "fraud_ratio": sum(labels) / len(labels),
            "node_ids":   tuple(sorted(comp)),
        })
    return pd.DataFrame(records)
