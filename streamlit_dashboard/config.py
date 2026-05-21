from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "PMP-514" / "dashboard_data"

PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
METRICS_JSON    = DATA_DIR / "metrics.json"
ROC_NPZ         = DATA_DIR / "roc_curve.npz"
PR_NPZ          = DATA_DIR / "pr_curve.npz"
EMBEDDINGS_NPZ  = DATA_DIR / "embeddings_2d.npz"
TRAINING_LOG    = DATA_DIR / "training_log.json"
UTU_EDGES_CSV   = DATA_DIR / "utu_edges.csv"
RAW_CSV         = DATA_DIR.parent / "datasets" / "final_B_bridge_sample_with_text.csv"

MAX_CLIQUE_NODES = 50
MIN_CLIQUE_SIZE  = 2
DEFAULT_TOP_N    = 5

OUTCOME_COLORS = {
    "TP": "#EF4444",
    "FP": "#F59E0B",
    "TN": "#3B82F6",
    "FN": "#8B5CF6",
}
OUTCOME_LABELS = {
    "TP": "TP (True Fraud)",
    "FP": "FP (False Alarm)",
    "TN": "TN (True Normal)",
    "FN": "FN (Missed Fraud)",
}

CLIQUE_DEFS = {
    "net_uvu": {"label": "net_uvu  (같은 상품 + 리뷰 월)", "cols": ("prod_id", "review_month")},
    "net_usu": {"label": "net_usu  (같은 상품 + 별점)",    "cols": ("prod_id", "rating")},
}
