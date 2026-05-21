import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import build_clique_stats, build_text_clusters, get_clique_edges, get_clique_edges_filtered, get_clique_nodes_cols, load_predictions
from utils.style import inject_css, page_header, section_header, C, PLOTLY_BASE

st.set_page_config(page_title="네트워크", layout="wide", page_icon="🌐")
inject_css()
page_header("🌐", "사기 리뷰어 네트워크",
            "식당·리뷰 월 / 유저 / 텍스트 유사도 기준 클러스터 시각화")

pred_df = load_predictions()

# ─── 클러스터 기준 토글 ───────────────────────────────────────────────────
view_mode = st.radio(
    "클러스터 기준",
    ["🏪 식당·리뷰 월", "👤 유저", "📝 텍스트 유사도"],
    horizontal=True,
    key="view_mode",
)

if view_mode == "🏪 식당·리뷰 월":
    ACTIVE_COLS = ("prod_id", "review_month")
    slider_min, slider_max, slider_default = 5, 300, 30
elif view_mode == "👤 유저":
    ACTIVE_COLS = ("user_id",)
    slider_min, slider_max, slider_default = 1, 50, 5
else:
    ACTIVE_COLS = None
    slider_min, slider_max, slider_default = 2, 100, 5

# 모드 전환 시 슬라이더 기본값 리셋 후 rerun
if st.session_state.get("_view_mode") != view_mode:
    st.session_state["_view_mode"] = view_mode
    st.session_state["min_nodes"] = slider_default
    st.rerun()

# ─── 사이드바 ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f'<p style="color:{C["blue"]};font-weight:600;font-size:.85rem;'
    f'text-transform:uppercase;letter-spacing:.06em;'
    f'font-family:Inter,"Noto Sans KR",sans-serif;margin-bottom:.4rem">🔢 노드 수</p>',
    unsafe_allow_html=True,
)
effective_min = st.sidebar.slider(
    "노드 수", slider_min, slider_max,
    key="min_nodes",
)

st.sidebar.markdown(
    f'<p style="color:{C["purple"]};font-weight:600;font-size:.85rem;'
    f'text-transform:uppercase;letter-spacing:.06em;'
    f'font-family:Inter,"Noto Sans KR",sans-serif;margin:.9rem 0 .4rem">'
    f'🔗 최소 텍스트 유사도</p>',
    unsafe_allow_html=True,
)
min_cosine_sim = st.sidebar.select_slider(
    "최소 텍스트 유사도",
    options=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    value=0.4,
    key="min_cosine_sim",
)

# ─── 통계 수집 ────────────────────────────────────────────────────────────
@st.cache_data
def _build_stats(cols: tuple) -> pd.DataFrame:
    return build_clique_stats(cols)

if view_mode == "📝 텍스트 유사도":
    all_stats = build_text_clusters(min_cosine_sim)
else:
    all_stats = _build_stats(ACTIVE_COLS)
all_stats = all_stats[all_stats["node_count"] >= effective_min].reset_index(drop=True)

# ─── session_state 초기화 ─────────────────────────────────────────────────
filter_fp = f"{view_mode}_{effective_min}_{min_cosine_sim:.2f}"
if st.session_state.get("_net_fp") != filter_fp:
    st.session_state["_net_fp"] = filter_fp
    for t in range(13):
        st.session_state[f"fraud_shown_t{t}"]  = 5
        st.session_state[f"normal_shown_t{t}"] = 5
for t in range(13):
    st.session_state.setdefault(f"fraud_shown_t{t}",  5)
    st.session_state.setdefault(f"normal_shown_t{t}", 5)

# ─── 별점별 색상 ──────────────────────────────────────────────────────────
_RATING_COLORS = {
    1: "#EF4444",
    2: "#F97316",
    3: "#EAB308",
    4: "#84CC16",
    5: "#22C55E",
}

# ─── 별점별 클러스터 레이아웃 ────────────────────────────────────────────
def _cluster_layout(G, node_rating: dict) -> dict:
    unique_ratings = sorted(r for r in set(node_rating.values()) if r > 0)
    n = len(unique_ratings)
    pos = {}

    for i, rating in enumerate(unique_ratings):
        sub_nodes = [nd for nd in G.nodes() if node_rating.get(nd) == rating]
        if not sub_nodes:
            continue
        SG = G.subgraph(sub_nodes)
        if len(sub_nodes) == 1:
            sub_pos = {sub_nodes[0]: (0.0, 0.0)}
        else:
            try:
                sub_pos = nx.kamada_kawai_layout(SG)
            except Exception:
                sub_pos = nx.spring_layout(SG, seed=42)

        xs = [v[0] for v in sub_pos.values()]
        ys = [v[1] for v in sub_pos.values()]
        x_span = max(xs) - min(xs) or 1e-6
        y_span = max(ys) - min(ys) or 1e-6
        scale = 0.45 / max(x_span, y_span)
        x_cx  = (max(xs) + min(xs)) / 2
        y_cx  = (max(ys) + min(ys)) / 2
        x_offset = (i - (n - 1) / 2) * 1.5

        for nd, (x, y) in sub_pos.items():
            pos[nd] = ((x - x_cx) * scale + x_offset, (y - y_cx) * scale)

    unknown = [nd for nd in G.nodes() if node_rating.get(nd, -1) <= 0]
    if unknown:
        x_off = ((n - 1) / 2 + 1) * 1.5 if n > 0 else 0
        for j, nd in enumerate(unknown):
            pos[nd] = (x_off, j * 0.3)

    return pos


def _dedupe_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        return pd.DataFrame(columns=["src", "dst", "cosine_sim"])

    edge_cols = ["src", "dst"] + (["cosine_sim"] if "cosine_sim" in edges_df.columns else [])
    deduped = edges_df[edge_cols].copy()
    deduped["src"] = deduped["src"].astype(int)
    deduped["dst"] = deduped["dst"].astype(int)
    deduped["a"] = deduped[["src", "dst"]].min(axis=1)
    deduped["b"] = deduped[["src", "dst"]].max(axis=1)

    if "cosine_sim" in deduped.columns:
        deduped["cosine_sim"] = pd.to_numeric(deduped["cosine_sim"], errors="coerce")
        deduped = (
            deduped.sort_values("cosine_sim", ascending=False)
            .drop_duplicates(["a", "b"], keep="first")
        )
        return pd.DataFrame({
            "src": deduped["a"].astype(int),
            "dst": deduped["b"].astype(int),
            "cosine_sim": deduped["cosine_sim"],
        }).reset_index(drop=True)

    deduped = deduped.drop_duplicates(["a", "b"])
    return pd.DataFrame({
        "src": deduped["a"].astype(int),
        "dst": deduped["b"].astype(int),
    }).reset_index(drop=True)


def _filter_edges_by_cosine(edges_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    deduped = _dedupe_edges(edges_df)
    if "cosine_sim" not in deduped.columns:
        return deduped
    return deduped[deduped["cosine_sim"] >= threshold].reset_index(drop=True)


# ─── 네트워크 그래프 생성 ─────────────────────────────────────────────────
def make_network_fig(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> go.Figure:
    G = nx.Graph()
    G.add_nodes_from(nodes_df["node_id"].tolist())
    for _, e in edges_df.iterrows():
        G.add_edge(int(e["src"]), int(e["dst"]))

    ndf = nodes_df.set_index("node_id")
    node_list = list(G.nodes())

    def _wrap(t: str, w: int = 60) -> str:
        return "<br>".join(t[i:i+w] for i in range(0, len(t), w))

    def _node_data(n):
        lbl = int(ndf.loc[n, "label"])       if n in ndf.index else -1
        rat = int(ndf.loc[n, "rating"])      if n in ndf.index else -1
        txt = _wrap(str(ndf.loc[n, "text"])) if (n in ndf.index and pd.notna(ndf.loc[n, "text"])) else ""
        return lbl, rat, txt

    node_data  = [_node_data(n) for n in node_list]
    node_rating = {node_list[i]: node_data[i][1] for i in range(len(node_list))}
    pos = _cluster_layout(G, node_rating)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.0, color="rgba(100,116,139,0.25)"),
        hoverinfo="none", showlegend=False,
    )

    _HTMPL = (
        "<b>노드 %{text}</b><br>"
        "레이블: %{customdata[0]}<br>"
        "별점: %{customdata[1]}<br>"
        "<i>%{customdata[2]}</i><extra></extra>"
    )

    traces = [edge_trace]
    for rating, color in _RATING_COLORS.items():
        idxs = [i for i, (_, rat, _) in enumerate(node_data) if rat == rating]
        if not idxs:
            continue
        traces.append(go.Scatter(
            x=[pos[node_list[i]][0] for i in idxs],
            y=[pos[node_list[i]][1] for i in idxs],
            mode="markers",
            name=f"{rating}점",
            marker=dict(
                size=13,
                color=color,
                symbol=["star" if node_data[i][0] == 1 else "circle" for i in idxs],
                line=dict(width=1, color="rgba(255,255,255,0.6)"),
            ),
            text=[str(node_list[i]) for i in idxs],
            customdata=[node_data[i] for i in idxs],
            hovertemplate=_HTMPL,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        **PLOTLY_BASE,
        height=380,
        margin=dict(l=0, r=60, t=10, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
        showlegend=False,
    )
    return fig


# ─── 네트워크 렌더링 ─────────────────────────────────────────────────────
def render_clique(row, tab_key: str, cosine_threshold: float, cols):
    fraud_pct = row["fraud_ratio"]
    bar_color = C["red"] if fraud_pct >= 0.5 else C["orange"] if fraud_pct >= 0.2 else C["blue"]

    if cols is None:
        title = f"텍스트 클러스터 #{int(row['comp_id']) + 1}  ·  노드 {int(row['node_count'])}개  ·  사기 리뷰 비율: {fraud_pct:.1%}"
        key_str_safe = f"{tab_key}_comp{int(row['comp_id'])}"
    elif cols == ("user_id",):
        title = f"리뷰어 Index: {int(row['user_id'])}번  ·  노드 {int(row['node_count'])}개  ·  사기 리뷰 비율: {fraud_pct:.1%}"
        key_str_safe = f"{tab_key}_u{int(row['user_id'])}"
    else:
        title = f"식당 Index: {int(row['prod_id'])}번  ·  {int(row['review_month'])}월  ·  노드 {int(row['node_count'])}개  ·  사기 리뷰 비율: {fraud_pct:.1%}"
        key_str_safe = f"{tab_key}_p{row['prod_id']}_m{int(row['review_month'])}"

    with st.expander(title, expanded=True):
        st.markdown(
            f'<div style="background:{C["border"]};border-radius:4px;height:5px;'
            f'margin-bottom:.9rem;overflow:hidden">'
            f'<div style="background:{bar_color};width:{fraud_pct*100:.1f}%;'
            f'height:100%;border-radius:4px"></div></div>',
            unsafe_allow_html=True,
        )

        if cols is None:
            nodes_df = pred_df[pred_df["node_id"].isin(row["node_ids"])].copy()
        else:
            nodes_df = get_clique_nodes_cols(pred_df, cols, row)
        if len(nodes_df) == 0:
            st.caption("노드 없음")
            return

        MAX_VIZ_NODES = 300
        is_sampled = len(nodes_df) > MAX_VIZ_NODES
        if is_sampled:
            fraud_df  = nodes_df[nodes_df["label"] == 1]
            normal_df = nodes_df[nodes_df["label"] == 0]
            fraud_ratio  = len(fraud_df) / len(nodes_df)
            n_fraud  = round(MAX_VIZ_NODES * fraud_ratio)
            n_normal = MAX_VIZ_NODES - n_fraud
            sampled = pd.concat([
                fraud_df.sample(min(n_fraud, len(fraud_df)),   random_state=42),
                normal_df.sample(min(n_normal, len(normal_df)), random_state=42),
            ]).reset_index(drop=True)
            nodes_df = sampled
            st.warning(
                f"⚠️ 노드가 {int(row['node_count']):,}개로 많아 사기/정상 비율을 유지하며 "
                f"{MAX_VIZ_NODES}개를 무작위 샘플링했습니다. 네트워크 생김새는 참고용으로만 활용하세요.",
                icon=None,
            )

        node_ids = frozenset(nodes_df["node_id"].tolist())
        base_edge_count = len(_dedupe_edges(get_clique_edges_filtered(node_ids, 0.4)))
        edges_df = _dedupe_edges(get_clique_edges_filtered(node_ids, cosine_threshold))
        filtered_edge_count = len(edges_df)

        connected_ids = set(edges_df["src"].tolist()) | set(edges_df["dst"].tolist())
        connected_df = nodes_df[nodes_df["node_id"].isin(connected_ids)]
        edge_fraud_ratio = connected_df["label"].mean() if len(connected_df) > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("노드 수", int(row["node_count"]))
        m2.metric("사기 비율", f"{fraud_pct:.1%}")
        delta = filtered_edge_count - base_edge_count
        m3.metric("엣지 수", f"{filtered_edge_count:,}", delta if delta != 0 else None)
        m4.metric("연결 노드 사기 비율", f"{edge_fraud_ratio:.1%}")

        fig = make_network_fig(nodes_df, edges_df)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"net_{key_str_safe}",
        )

        # 별점 색상 범례
        swatches = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
            f'display:inline-block;border:1px solid rgba(0,0,0,.1)"></span>'
            f'<span style="font-size:.78rem;color:#64748B;font-family:Inter,sans-serif">{r}점</span>'
            f'</span>'
            for r, color in _RATING_COLORS.items()
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:.2rem;margin-top:.3rem">'
            f'<span style="font-size:.78rem;color:#64748B;font-family:Inter,sans-serif;margin-right:.4rem">별점:</span>'
            f'{swatches}'
            f'<span style="font-size:.78rem;color:#94A3B8;margin-left:.4rem">| ★ = 사기 &nbsp; ● = 정상'
            f' &nbsp; | 텍스트 유사도 ≥ {cosine_threshold:.1f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        review_cols = [c for c in ["node_id", "label", "rating", "text"] if c in nodes_df.columns]
        if review_cols:
            review_df = nodes_df[review_cols].copy()
            review_df.columns = ["노드ID", "레이블", "별점", "리뷰"][:len(review_cols)]
            st.dataframe(review_df, use_container_width=True, hide_index=True)


# ─── 탭 구조 ─────────────────────────────────────────────────────────────
if view_mode == "🏪 식당·리뷰 월":
    tab_labels = ["전체"] + [f"{m}월" for m in range(1, 13)]
    tab_months = [None] + list(range(1, 13))
else:
    tab_labels = ["전체"]
    tab_months = [None]



tabs = st.tabs(tab_labels)

def _render_tab(tab_idx: int, stats: pd.DataFrame, tk: str):
    fraud_all  = stats[stats["fraud_ratio"] >= 0.3].sort_values("fraud_ratio", ascending=False).reset_index(drop=True)
    normal_all = stats[stats["fraud_ratio"] <  0.3].sort_values("fraud_ratio", ascending=True).reset_index(drop=True)

    fk = f"fraud_shown_t{tab_idx}"
    nk = f"normal_shown_t{tab_idx}"
    fraud_top  = fraud_all.iloc[:st.session_state[fk]]
    normal_top = normal_all.iloc[:st.session_state[nk]]

    col_fraud, col_normal = st.columns(2)

    with col_fraud:
        section_header("🔴 상위 사기 네트워크")
        if fraud_top.empty:
            st.info("네트워크 없음")
        else:
            for _, row in fraud_top.iterrows():
                render_clique(row, f"{tk}f", min_cosine_sim, ACTIVE_COLS)

    with col_normal:
        section_header("🔵 상위 정상 네트워크")
        if normal_top.empty:
            st.info("네트워크 없음")
        else:
            for _, row in normal_top.iterrows():
                render_clique(row, f"{tk}n", min_cosine_sim, ACTIVE_COLS)

    can_load_more = (st.session_state[fk] < len(fraud_all)) or (st.session_state[nk] < len(normal_all))
    if can_load_more:
        if st.button("더 불러오기", key=f"more_{tk}"):
            if st.session_state[fk] < len(fraud_all):
                st.session_state[fk] += 5
            if st.session_state[nk] < len(normal_all):
                st.session_state[nk] += 5
            st.rerun()
    else:
        st.caption("모든 결과를 불러왔습니다.")

for tab_idx, (tab, month) in enumerate(zip(tabs, tab_months)):
    with tab:
        if month is None or view_mode != "🏪 식당·리뷰 월":
            stats = all_stats
        else:
            stats = all_stats[all_stats["review_month"] == month].reset_index(drop=True)
        _render_tab(tab_idx, stats, f"t{tab_idx}")
