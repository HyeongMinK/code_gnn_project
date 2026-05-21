import streamlit as st

# ── 공통 Plotly 레이아웃 ────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#1E293B", size=12),
    title_font=dict(color="#64748B", size=13),
)

LEGEND_STYLE = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1)

AXIS_STYLE = dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#E2E8F0")

# ── 색상 팔레트 ────────────────────────────────────────────────────────────
C = dict(
    bg="#F8FAFC", card="#FFFFFF", border="#E2E8F0",
    text="#1E293B", muted="#64748B",
    blue="#3B82F6", red="#EF4444", green="#22C55E",
    orange="#F59E0B", purple="#8B5CF6",
)

OUTCOME_COLORS = {
    "TP": C["red"],
    "FP": C["orange"],
    "TN": C["blue"],
    "FN": C["purple"],
}
OUTCOME_LABELS = {
    "TP": "True Fraud",
    "FP": "False Alarm",
    "TN": "True Normal",
    "FN": "Missed Fraud",
}

# ── CSS 인젝션 ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
    <style>

    .stApp {{
        background-color: {C['bg']};
        color: {C['text']};
        font-family: 'Inter', 'Noto Sans KR', system-ui, sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {C['card']} !important;
        border-right: 1px solid {C['border']};
        box-shadow: 2px 0 8px rgba(0,0,0,0.06);
    }}
    [data-testid="stSidebarContent"] {{
        padding-top: 1.2rem;
    }}
    .block-container {{
        padding: 1.2rem 2.5rem 3rem;
        max-width: 1400px;
    }}
    h1, h2, h3, h4 {{
        color: {C['text']} !important;
        font-family: 'Inter', 'Noto Sans KR', system-ui, sans-serif !important;
    }}
    p, div, label {{
        font-family: 'Inter', 'Noto Sans KR', system-ui, sans-serif !important;
    }}
    hr {{
        border-color: {C['border']} !important;
        margin: 1.2rem 0;
    }}
    [data-testid="metric-container"] {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {C['muted']} !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    [data-testid="stMetricValue"] {{
        color: {C['text']} !important;
    }}
    [data-testid="stExpander"] {{
        background: {C['card']};
        border: 1px solid {C['border']} !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    [data-testid="stExpander"] summary {{
        color: {C['text']} !important;
        font-weight: 500;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: {C['blue']} !important;
    }}
    [data-testid="stSlider"] p {{
        color: {C['muted']} !important;
        font-size: 0.8rem !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
        background-color: {C['blue']} !important;
        border-color: {C['blue']} !important;
    }}
    [data-testid="stToggle"] p {{
        color: {C['muted']} !important;
    }}
    [data-testid="stCaptionContainer"] p {{
        color: {C['muted']} !important;
    }}
    [data-testid="stAlert"] {{
        border-radius: 10px;
        border: 1px solid {C['border']};
        background: {C['card']};
    }}
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {C['border']};
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    [data-testid="stRadio"] label {{
        color: {C['text']} !important;
    }}
    [data-testid="stCheckbox"] label p {{
        color: {C['text']} !important;
    }}
    .stSelectbox label, .stMultiselect label {{
        color: {C['muted']} !important;
        font-size: 0.8rem !important;
    }}
    [data-testid="stSidebarNavLink"] {{
        border-radius: 8px;
    }}
    [data-testid="stSidebarNavLink"]:hover {{
        background: rgba(59,130,246,0.08) !important;
    }}
    [data-testid="stSidebarNavLink"][aria-current="page"] {{
        background: rgba(59,130,246,0.12) !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── 페이지 헤더 배너 ──────────────────────────────────────────────────────
def page_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0EA5E9 0%,#6366F1 100%);
                border-radius:16px;
                padding:1.4rem 2rem; margin-bottom:1.5rem;
                display:flex; align-items:center; gap:1.2rem;
                box-shadow: 0 4px 20px rgba(99,102,241,0.22);">
        <span style="font-size:2.4rem;line-height:1">{icon}</span>
        <div>
            <h1 style="color:#FFFFFF;margin:0;font-size:1.7rem;
                       font-weight:700;font-family:Inter,'Noto Sans KR',sans-serif">{title}</h1>
            <p style="color:rgba(255,255,255,0.80);margin:.25rem 0 0;font-size:.85rem;
                      font-family:Inter,'Noto Sans KR',sans-serif">{subtitle}</p>
        </div>
    </div>""", unsafe_allow_html=True)


# ── 섹션 타이틀 ──────────────────────────────────────────────────────────
def section_header(title: str):
    st.markdown(
        f'<h3 style="color:{C["blue"]};border-bottom:2px solid {C["border"]};'
        f'padding-bottom:.35rem;margin:1.5rem 0 1rem;font-size:1rem;'
        f'font-weight:600;letter-spacing:.02em;font-family:Inter,"Noto Sans KR",sans-serif">{title}</h3>',
        unsafe_allow_html=True,
    )


# ── 메트릭 카드 그리드 ────────────────────────────────────────────────────
def metric_cards(items: list):
    cards_html = ""
    for it in items:
        cards_html += f"""
        <div style="background:{C['card']};border:1px solid {C['border']};
                    border-radius:12px;padding:1rem 1.2rem;flex:1;min-width:130px;
                    text-align:center;transition:box-shadow .2s;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06)">
            <div style="color:{C['muted']};font-size:.68rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:.08em;
                        margin-bottom:.45rem;font-family:Inter,'Noto Sans KR',sans-serif">{it['label']}</div>
            <div style="color:{it['color']};font-size:1.65rem;font-weight:700;
                        font-family:'SF Mono',ui-monospace,monospace;
                        line-height:1.2">{it['value']}</div>
            <div style="color:{C['muted']};font-size:.7rem;margin-top:.3rem;
                        font-family:Inter,'Noto Sans KR',sans-serif">{it.get('sub','')}</div>
        </div>"""
    st.markdown(
        f'<div style="display:flex;gap:.7rem;flex-wrap:wrap;margin-bottom:1.2rem">'
        f'{cards_html}</div>',
        unsafe_allow_html=True,
    )


# ── Outcome 카드 (TP/FP/TN/FN) ───────────────────────────────────────────
_OUTCOME_META = {
    "TP": (C["red"],    "TP · True Fraud"),
    "FP": (C["orange"], "FP · False Alarm"),
    "TN": (C["blue"],   "TN · True Normal"),
    "FN": (C["purple"], "FN · Missed Fraud"),
}

def outcome_cards(counts: dict):
    html = ""
    for key, (color, label) in _OUTCOME_META.items():
        n = counts.get(key, 0)
        html += f"""
        <div style="background:{C['card']};border:1px solid {C['border']};
                    border-left:4px solid {color};border-radius:10px;
                    padding:.85rem 1.2rem;flex:1;min-width:150px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06)">
            <div style="color:{color};font-size:.69rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:.07em;
                        font-family:Inter,'Noto Sans KR',sans-serif">{label}</div>
            <div style="color:{C['text']};font-size:2rem;font-weight:700;
                        margin-top:.15rem;font-family:'SF Mono',monospace">{n:,}</div>
        </div>"""
    st.markdown(
        f'<div style="display:flex;gap:.7rem;flex-wrap:wrap;margin:1rem 0 1.2rem">'
        f'{html}</div>',
        unsafe_allow_html=True,
    )
