import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import OUTCOME_COLORS, OUTCOME_LABELS
from utils.data_loader import load_embeddings
from utils.style import inject_css, page_header, section_header, outcome_cards, C, PLOTLY_BASE, AXIS_STYLE

st.set_page_config(page_title="산점도", layout="wide", page_icon="🔵")
inject_css()
page_header("🔵", "t-SNE 산점도",
            "테스트 셋 9,310 노드 · 사기 확률 임베딩")

emb_df = load_embeddings()

# ─── Threshold 슬라이더 ────────────────────────────────────────────────────
threshold = st.slider("사기 임계값", 0.0, 1.0, 0.50, 0.01, key="scatter_thresh")

# Outcome 계산 (vectorized)
pred_bin = (emb_df["fraud_prob"].values >= threshold).astype(int)
true_lbl = emb_df["true_label"].values
outcome  = np.where(
    (pred_bin == 1) & (true_lbl == 1), "TP",
    np.where(
        (pred_bin == 1) & (true_lbl == 0), "FP",
        np.where((pred_bin == 0) & (true_lbl == 0), "TN", "FN"),
    ),
)
emb_df = emb_df.copy()
emb_df["outcome"] = outcome

# ─── TP/FP/TN/FN 컬러 카드 ────────────────────────────────────────────────
counts = emb_df["outcome"].value_counts()
outcome_cards({k: int(counts.get(k, 0)) for k in ("TP", "FP", "TN", "FN")})

# ─── 필터 토글 ────────────────────────────────────────────────────────────
filter_on = st.toggle("필터 활성화", value=False)
if filter_on:
    lo, hi = st.slider("사기 확률 범위", 0.0, 1.0, (0.0, 1.0), 0.01,
                        key="prob_range")
    plot_df = emb_df[
        (emb_df["fraud_prob"] >= lo) & (emb_df["fraud_prob"] <= hi)
    ].copy()
    st.caption(f"표시 노드: {len(plot_df):,} / {len(emb_df):,}")
else:
    plot_df = emb_df

# ─── t-SNE Scatter ────────────────────────────────────────────────────────
section_header("🗺️ t-SNE 임베딩")

fig = go.Figure()
for oc in ["TP", "FP", "TN", "FN"]:
    sub = plot_df[plot_df["outcome"] == oc]
    if sub.empty:
        continue
    fig.add_trace(go.Scattergl(
        x=sub["x"].values,
        y=sub["y"].values,
        mode="markers",
        name=OUTCOME_LABELS[oc],
        marker=dict(
            color=OUTCOME_COLORS[oc],
            size=5,
            opacity=0.78,
            line=dict(width=0),
        ),
        customdata=np.stack([
            sub["fraud_prob"].values,
            sub["true_label"].values,
            sub["pred_label"].fillna(-1).values.astype(int),
            sub["rating"].fillna(-1).values,
            sub["text"].fillna("").apply(lambda t: "<br>".join(t[i:i+60] for i in range(0, len(t), 60))).values,
        ], axis=-1),
        hovertemplate=(
            "%{customdata[4]}<br><br>"
            "사기 확률: %{customdata[0]:.4f}<br>"
            "실제 레이블: %{customdata[1]}<br>"
            "예측 레이블: %{customdata[2]}<br>"
            "별점: %{customdata[3]}<extra></extra>"
        ),
    ))

fig.update_layout(
    **PLOTLY_BASE,
    title="t-SNE 임베딩 — 테스트 셋 (9,310 노드)",
    xaxis=dict(title="t-SNE 차원 1", **AXIS_STYLE),
    yaxis=dict(title="t-SNE 차원 2", **AXIS_STYLE),
    legend_title="예측 결과",
    height=660,
    margin=dict(l=50, r=20, t=60, b=50),
)
st.plotly_chart(fig, use_container_width=True)
