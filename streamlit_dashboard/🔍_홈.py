import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import load_raw_data
from utils.style import inject_css, page_header, section_header, metric_cards, C, PLOTLY_BASE, AXIS_STYLE, LEGEND_STYLE

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

inject_css()
page_header("🔍", "GNN 사기 탐지 대시보드",
            "LA-SAGE-S · Bridge Dataset · 실시간 임계값 연동")

# ── 페이지 안내 카드 ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1.4rem">

  <a href="/보고서" style="text-decoration:none;flex:1;min-width:200px">
    <div style="background:{C['card']};border:1px solid {C['border']};
                border-top:3px solid {C['blue']};border-radius:12px;
                padding:1.2rem 1.4rem;transition:border-color .2s;cursor:pointer">
      <div style="font-size:1.6rem;margin-bottom:.5rem">📄</div>
      <div style="color:{C['blue']};font-weight:600;font-size:.95rem;
                  font-family:Inter,'Noto Sans KR',sans-serif">보고서</div>
      <div style="color:{C['muted']};font-size:.78rem;margin-top:.3rem;
                  font-family:Inter,'Noto Sans KR',sans-serif">성능 지표 · ROC/PR 커브 · 학습 곡선</div>
    </div>
  </a>

  <a href="/산점도" style="text-decoration:none;flex:1;min-width:200px">
    <div style="background:{C['card']};border:1px solid {C['border']};
                border-top:3px solid {C['green']};border-radius:12px;
                padding:1.2rem 1.4rem;transition:border-color .2s;cursor:pointer">
      <div style="font-size:1.6rem;margin-bottom:.5rem">🔵</div>
      <div style="color:{C['green']};font-weight:600;font-size:.95rem;
                  font-family:Inter,'Noto Sans KR',sans-serif">산점도</div>
      <div style="color:{C['muted']};font-size:.78rem;margin-top:.3rem;
                  font-family:Inter,'Noto Sans KR',sans-serif">t-SNE 임베딩 · 임계값 연동 TP/FP/TN/FN</div>
    </div>
  </a>

  <a href="/네트워크" style="text-decoration:none;flex:1;min-width:200px">
    <div style="background:{C['card']};border:1px solid {C['border']};
                border-top:3px solid {C['purple']};border-radius:12px;
                padding:1.2rem 1.4rem;transition:border-color .2s;cursor:pointer">
      <div style="font-size:1.6rem;margin-bottom:.5rem">🌐</div>
      <div style="color:{C['purple']};font-weight:600;font-size:.95rem;
                  font-family:Inter,'Noto Sans KR',sans-serif">네트워크</div>
      <div style="color:{C['muted']};font-size:.78rem;margin-top:.3rem;
                  font-family:Inter,'Noto Sans KR',sans-serif">식당·유저·텍스트 유사도 기준 클러스터 네트워크</div>
    </div>
  </a>

</div>
""", unsafe_allow_html=True)

# ── 모델 스펙 카드 ───────────────────────────────────────────────────────────
metric_cards([
    {"label": "모델",      "value": "LA-SAGE-S", "sub": "Yelp-Bridge",   "color": C["blue"]},
    {"label": "테스트 AUC","value": "0.9059",   "sub": "검증 0.9063",   "color": C["blue"]},
    {"label": "테스트 AP", "value": "0.6878",   "sub": "검증 0.6820",   "color": C["green"]},
    {"label": "GMean",    "value": "0.8352",    "sub": "검증 0.8357",   "color": C["orange"]},
    {"label": "F1-macro", "value": "0.8060",    "sub": "검증 0.8031",   "color": C["purple"]},
    {"label": "데이터셋",  "value": "Bridge",   "sub": "노드 46,551개", "color": C["muted"]},
])

st.markdown(
    f'<p style="color:{C["muted"]};font-size:.78rem;text-align:center;margin-top:.5rem;'
    f'font-family:Inter,"Noto Sans KR",sans-serif">왼쪽 사이드바에서 페이지를 선택하세요</p>',
    unsafe_allow_html=True,
)

st.divider()

# ── EDA 섹션 ──────────────────────────────────────────────────────────────────
raw = load_raw_data()

section_header("📊 데이터 개요  ·  2014년 Yelp 리뷰 · Bridge 데이터셋")

total_reviews  = len(raw)
fraud_reviews  = int(raw["label"].sum())
fraud_rate     = fraud_reviews / total_reviews
restaurant_cnt = raw["prod_id"].nunique()
reviewer_cnt   = raw["user_id"].nunique()

metric_cards([
    {"label": "총 리뷰 수",   "value": f"{total_reviews:,}건",     "sub": "2014년 전체",        "color": C["blue"]},
    {"label": "사기 리뷰 수", "value": f"{fraud_reviews:,}건",     "sub": f"비율 {fraud_rate:.1%}", "color": C["red"]},
    {"label": "사기 비율",    "value": f"{fraud_rate:.1%}",        "sub": "정상 87.1%",          "color": C["red"]},
    {"label": "식당 수",      "value": f"{restaurant_cnt:,}개",    "sub": "고유 prod_id",        "color": C["orange"]},
    {"label": "리뷰어 수",    "value": f"{reviewer_cnt:,}명",      "sub": "고유 user_id",        "color": C["purple"]},
])

# ── 월별 추이 + 별점 분포 ─────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    section_header("📅 월별 리뷰 추이")

    monthly = (
        raw.groupby(["month", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "정상", 1: "사기"})
        .reset_index()
    )
    monthly["사기율"] = (
        monthly["사기"] / (monthly["정상"] + monthly["사기"]) * 100
    )
    month_labels = ["1월","2월","3월","4월","5월","6월",
                    "7월","8월","9월","10월","11월","12월"]

    fig_month = make_subplots(specs=[[{"secondary_y": True}]])
    fig_month.add_trace(
        go.Bar(
            x=month_labels, y=monthly["정상"], name="정상",
            marker_color=C["blue"], opacity=0.85,
        ),
        secondary_y=False,
    )
    fig_month.add_trace(
        go.Bar(
            x=month_labels, y=monthly["사기"], name="사기",
            marker_color=C["red"], opacity=0.85,
        ),
        secondary_y=False,
    )
    fig_month.add_trace(
        go.Scatter(
            x=month_labels, y=monthly["사기율"], name="사기율 (%)",
            mode="lines+markers",
            line=dict(color=C["orange"], width=2.5),
            marker=dict(size=7, color=C["orange"]),
        ),
        secondary_y=True,
    )
    fig_month.update_layout(
        **PLOTLY_BASE,
        barmode="stack",
        height=340,
        margin=dict(l=55, r=55, t=30, b=40),
        legend=dict(x=0.01, y=0.99, **LEGEND_STYLE),
    )
    fig_month.update_xaxes(**AXIS_STYLE)
    fig_month.update_yaxes(title_text="리뷰 수",  secondary_y=False, **AXIS_STYLE)
    fig_month.update_yaxes(title_text="사기율 (%)", secondary_y=True,
                           color=C["orange"], **AXIS_STYLE)
    st.plotly_chart(fig_month, use_container_width=True)

with col_right:
    section_header("⭐ 별점별 사기/정상 분포")

    rating_grp = (
        raw.groupby(["rating", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "정상", 1: "사기"})
        .reset_index()
    )

    fig_rating = go.Figure()
    fig_rating.add_trace(go.Bar(
        x=rating_grp["rating"].astype(str), y=rating_grp["정상"],
        name="정상", marker_color=C["blue"], opacity=0.85,
    ))
    fig_rating.add_trace(go.Bar(
        x=rating_grp["rating"].astype(str), y=rating_grp["사기"],
        name="사기", marker_color=C["red"], opacity=0.85,
    ))
    fig_rating.update_layout(
        **PLOTLY_BASE,
        barmode="group",
        height=340,
        margin=dict(l=55, r=20, t=30, b=40),
        xaxis=dict(title="별점", **AXIS_STYLE),
        yaxis=dict(title="리뷰 수", **AXIS_STYLE),
        legend=dict(x=0.70, y=0.99, **LEGEND_STYLE),
    )
    st.plotly_chart(fig_rating, use_container_width=True)

# ── 상위 식당 사기 현황 ────────────────────────────────────────────────────────
section_header("🏪 리뷰 상위 식당 사기 현황  ·  리뷰 수 상위 15개")

rest_stats = (
    raw.groupby("prod_id")
    .agg(
        리뷰수=("label", "count"),
        사기수=("label", "sum"),
    )
    .assign(사기율=lambda d: d["사기수"] / d["리뷰수"])
    .nlargest(15, "리뷰수")
    .reset_index()
    .sort_values("사기율", ascending=True)
)

bar_colors = [
    C["red"] if r >= 0.25
    else C["orange"] if r >= 0.15
    else C["blue"]
    for r in rest_stats["사기율"]
]

fig_rest = go.Figure(go.Bar(
    x=rest_stats["사기율"],
    y=rest_stats["prod_id"].astype(str),
    orientation="h",
    marker_color=bar_colors,
    text=[f"{r:.1%}  ({n:,}건)" for r, n in zip(rest_stats["사기율"], rest_stats["리뷰수"])],
    textposition="outside",
    customdata=np.stack([rest_stats["리뷰수"], rest_stats["사기수"]], axis=1),
    hovertemplate=(
        "식당 %{y}<br>"
        "사기율: %{x:.1%}<br>"
        "리뷰 수: %{customdata[0]:,}건<br>"
        "사기 수: %{customdata[1]:,}건<extra></extra>"
    ),
))
fig_rest.update_layout(
    **PLOTLY_BASE,
    height=420,
    margin=dict(l=80, r=120, t=20, b=40),
    xaxis=dict(title="사기율", tickformat=".0%", **AXIS_STYLE),
    yaxis=dict(title="식당 ID", **AXIS_STYLE),
)
st.plotly_chart(fig_rest, use_container_width=True)
