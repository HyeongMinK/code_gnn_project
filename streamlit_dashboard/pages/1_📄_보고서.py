import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import (
    load_metrics, load_predictions, load_roc, load_pr, load_training_log,
)
from utils.style import inject_css, page_header, section_header, metric_cards, outcome_cards, C, PLOTLY_BASE, AXIS_STYLE, LEGEND_STYLE

st.set_page_config(page_title="보고서", layout="wide", page_icon="📄")
inject_css()
page_header("📄", "성능 보고서",
            "LA-SAGE-S · Bridge · 임계값 연동 혼동 행렬")

metrics  = load_metrics()
pred_df  = load_predictions()
roc_data = load_roc()
pr_data  = load_pr()
log_data = load_training_log()
test_df  = pred_df[pred_df["split"] == "test"].copy()

# ─── 섹션 1: 성능 지표 카드 ────────────────────────────────────────────────
section_header("📊 성능 지표")

METRIC_INFO = [
    ("AUC",          "auc_gnn",     C["blue"]),
    ("AP",           "ap_gnn",      C["green"]),
    ("GMean",        "gmean_gnn",   C["orange"]),
    ("F1-macro",     "f1_macro",    C["purple"]),
    ("F1 (사기 클래스)", "f1_binary_1", C["red"]),
    ("정밀도 (사기)",    "precision_1", C["blue"]),
    ("재현율 (사기)",    "recall_1",    C["green"]),
    ("재현율 (매크로)", "recall_macro",C["muted"]),
]

metric_cards([
    {
        "label": label,
        "value": f"{metrics['test'][key]:.4f}",
        "sub":   f"검증 {metrics['dev'][key]:.4f}",
        "color": color,
    }
    for label, key, color in METRIC_INFO
])

with st.expander("전체 지표 테이블 (검증 / 테스트 / Δ)", expanded=False):
    rows = []
    for label, key, _ in METRIC_INFO:
        dev_val  = metrics["dev"][key]
        test_val = metrics["test"][key]
        rows.append({
            "지표":          label,
            "검증":          f"{dev_val:.4f}",
            "테스트":        f"{test_val:.4f}",
            "Δ (테스트-검증)": f"{test_val - dev_val:+.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("지표"), use_container_width=True)

st.divider()

# ─── 섹션 2: ROC + PR 커브 ────────────────────────────────────────────────
section_header("📈 ROC / PR 커브 (테스트 셋)")

col_ctrl, _ = st.columns([3, 1])
with col_ctrl:
    threshold = st.slider("임계값", 0.0, 1.0, 0.50, 0.01, key="report_thresh")

# ROC dot (test only)
roc_thresh = roc_data["test_thresholds"]
finite     = np.isfinite(roc_thresh)
roc_fpr    = roc_data["test_fpr"][finite]
roc_tpr    = roc_data["test_tpr"][finite]
roc_t      = roc_thresh[finite]
idx_roc    = np.argmin(np.abs(roc_t - threshold))

# PR dot
pr_thresh  = pr_data["test_thresholds"]
pr_prec    = pr_data["test_precision"]
pr_rec     = pr_data["test_recall"]
idx_pr     = np.argmin(np.abs(pr_thresh - threshold))

# Confusion matrix (test)
pred_bin = (test_df["fraud_prob"] >= threshold).astype(int)
true_lbl = test_df["label"].values
TP = int(((pred_bin == 1) & (true_lbl == 1)).sum())
FP = int(((pred_bin == 1) & (true_lbl == 0)).sum())
TN = int(((pred_bin == 0) & (true_lbl == 0)).sum())
FN = int(((pred_bin == 0) & (true_lbl == 1)).sum())
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

col_roc, col_pr = st.columns(2)

with col_roc:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=roc_fpr, y=roc_tpr, mode="lines", name="ROC",
        line=dict(color=C["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.10)",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[roc_fpr[idx_roc]], y=[roc_tpr[idx_roc]], mode="markers",
        name=f"t={threshold:.2f}",
        marker=dict(color=C["red"], size=14, symbol="circle",
                    line=dict(width=2, color="white")),
    ))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dot", color=C["border"], width=1.5))
    fig_roc.update_layout(
        **PLOTLY_BASE,
        title=f"ROC 커브  AUC = {metrics['test']['auc_gnn']:.4f}",
        xaxis=dict(title="위양성률", **AXIS_STYLE),
        yaxis=dict(title="재현율", **AXIS_STYLE),
        height=390,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.58, y=0.08, **LEGEND_STYLE),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_pr:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=pr_rec, y=pr_prec, mode="lines", name="PR",
        line=dict(color=C["green"], width=2.5),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.10)",
    ))
    fig_pr.add_trace(go.Scatter(
        x=[pr_rec[idx_pr]], y=[pr_prec[idx_pr]], mode="markers",
        name=f"t={threshold:.2f}",
        marker=dict(color=C["red"], size=14, symbol="circle",
                    line=dict(width=2, color="white")),
    ))
    fig_pr.update_layout(
        **PLOTLY_BASE,
        title=f"PR 커브  AP = {metrics['test']['ap_gnn']:.4f}",
        xaxis=dict(title="재현율",   **AXIS_STYLE),
        yaxis=dict(title="정밀도",   **AXIS_STYLE),
        height=390,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.05, y=0.08, **LEGEND_STYLE),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

# ── 실시간 메트릭 + Confusion Matrix ──────────────────────────────────────
outcome_cards({"TP": TP, "FP": FP, "TN": TN, "FN": FN})

m1, m2, m3 = st.columns(3)
m1.metric("정밀도 (테스트)", f"{precision:.4f}")
m2.metric("재현율 (테스트)", f"{recall:.4f}")
m3.metric("F1 (테스트)",    f"{f1:.4f}")

total = TP + FP + TN + FN
z    = [[TN, FP], [FN, TP]]
text = [
    [f"TN\n{TN:,}\n({TN/total:.1%})", f"FP\n{FP:,}\n({FP/total:.1%})"],
    [f"FN\n{FN:,}\n({FN/total:.1%})", f"TP\n{TP:,}\n({TP/total:.1%})"],
]
fig_cm = go.Figure(go.Heatmap(
    z=z, text=text, texttemplate="%{text}",
    colorscale=[[0, C["card"]], [0.5, "#2d3748"], [1, C["red"]]],
    showscale=False,
    x=["예측 정상", "예측 사기"],
    y=["실제 정상", "실제 사기"],
))
fig_cm.update_layout(
    **PLOTLY_BASE,
    title=f"혼동 행렬  (임계값 = {threshold:.2f})",
    xaxis=dict(**AXIS_STYLE, side="bottom"),
    yaxis=dict(**AXIS_STYLE),
    height=320,
    margin=dict(l=90, r=20, t=50, b=60),
)
fig_cm.update_traces(textfont=dict(color=C["text"], size=13))
st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ─── 섹션 3: 학습 곡선 ────────────────────────────────────────────────────
section_header("📉 학습 곡선")

epochs  = log_data["epoch"]
tr_loss = log_data["train_loss"]
dev_auc = log_data["dev_auc"]

fig_train = make_subplots(specs=[[{"secondary_y": True}]])
fig_train.add_trace(
    go.Scatter(x=epochs, y=tr_loss, mode="lines+markers",
               name="학습 손실", marker=dict(size=5, color=C["red"]),
               line=dict(color=C["red"], width=2)),
    secondary_y=False,
)
fig_train.add_trace(
    go.Scatter(x=epochs, y=dev_auc, mode="lines+markers",
               name="검증 AUC", marker=dict(size=5, color=C["blue"]),
               line=dict(color=C["blue"], width=2)),
    secondary_y=True,
)
fig_train.update_xaxes(title_text="Epoch", **AXIS_STYLE)
fig_train.update_yaxes(title_text="학습 손실", secondary_y=False,
                       color=C["red"], **AXIS_STYLE)
fig_train.update_yaxes(title_text="검증 AUC",   secondary_y=True,
                       color=C["blue"], **AXIS_STYLE)
fig_train.update_layout(
    **PLOTLY_BASE,
    height=390,
    margin=dict(l=65, r=65, t=30, b=50),
)
st.plotly_chart(fig_train, use_container_width=True)
