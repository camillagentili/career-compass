import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Career Compass",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# DATA CONFIG
# =========================
INDEX_COLS = [
    "income_index",
    "job_index",
    "cost_index",
    "health_env_index",
    "safety_index",
    "social_index",
    "mobility_index",
]

TREND_COLS = [f"trend_{c}" for c in INDEX_COLS]

# Indicatori con icone SVG
INDICATORS = [
    {
        "key": "income_index",
        "title": "Income & Economy",
        "desc": "Economic capacity and purchasing power",
        "icon": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M16 8h-6a2 2 0 1 0 0 4h4a2 2 0 1 1 0 4H8"/><path d="M12 18V6"/></svg>""",
        "color": "#3B82F6",
    },
    {
        "key": "job_index",
        "title": "Employment",
        "desc": "Job opportunities and market conditions",
        "icon": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="7" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>""",
        "color": "#10B981",
    },
    {
        "key": "cost_index",
        "title": "Cost of Living",
        "desc": "Housing affordability and living expenses",
        "icon": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>""",
        "color": "#F59E0B",
    },
    {
        "key": "health_env_index",
        "title": "Health & Environment",
        "desc": "Healthcare quality and environmental conditions",
        "icon": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>""",
        "color": "#EF4444",
    },
    {
        "key": "safety_index",
        "title": "Safety",
        "desc": "Security and crime rates",
        "icon": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/></svg>""",
        "color": "#8B5CF6",
    },
    {
        "key": "social_index",
        "title": "Social Life",
        "desc": "Life satisfaction and social support",
        "icon": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>""",
        "color": "#EC4899",
    },
    {
        "key": "mobility_index",
        "title": "Mobility",
        "desc": "Transportation and accessibility",
        "icon": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>""",
        "color": "#6366F1",
    },
]

IND_MAP = {d["key"]: d for d in INDICATORS}

# =========================
# STYLE
# =========================
st.markdown(
    """
<style>
/* Layout generale */
.stApp {
  background: linear-gradient(180deg, #F3F7FF 0%, #F7F5FF 50%, #FFFFFF 100%);
}

/* Header */
.cc-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  margin-top: 10px;
}
.cc-logo {
  width: 52px;
  height: 52px;
  border-radius: 16px;
  background: linear-gradient(135deg, #2F6BFF 0%, #7C3AED 100%);
  display: grid;
  place-items: center;
  color: white;
  box-shadow: 0 14px 35px rgba(46, 107, 255, 0.25);
}
.cc-logo svg {
  width: 28px;
  height: 28px;
}
.cc-title {
  font-size: 56px;
  line-height: 1;
  margin: 0;
  font-weight: 800;
  letter-spacing: -1px;
  background: linear-gradient(90deg, #2F6BFF 0%, #7C3AED 70%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.cc-subtitle {
  text-align: center;
  margin-top: 6px;
  font-size: 22px;
  color: #5B6475;
}

/* Card base */
.cc-card {
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(30,41,59,0.06);
  border-radius: 18px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
  padding: 18px 18px;
}

/* Sezione preferenze */
.pref-row {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 14px 8px;
  border-radius: 16px;
}
.pref-icon {
  width: 52px;
  height: 52px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  box-shadow: 0 16px 30px rgba(15,23,42,0.10);
}
.pref-icon svg {
  width: 24px;
  height: 24px;
  stroke: white;
}
.pref-title {
  font-size: 22px;
  font-weight: 800;
  margin: 0;
  color: #0B1220;
}
.pref-desc {
  margin: 2px 0 0 0;
  color: #6B7280;
  font-size: 16px;
}
.pref-value {
  margin-left: auto;
  font-size: 38px;
  font-weight: 800;
  color: #2F6BFF;
  line-height: 1.1;
}

/* Slider */
div[data-baseweb="slider"] {
  margin-top: -6px;
}
.small-muted {
  color: #8A94A6;
  font-size: 14px;
  margin-top: 2px;
}

/* Pulsante */
div.stButton > button, div.stFormSubmitButton > button {
  width: 100%;
  border: 0 !important;
  border-radius: 18px !important;
  padding: 14px 16px !important;
  background: linear-gradient(90deg, #2F6BFF 0%, #7C3AED 100%) !important;
  color: white !important;
  font-size: 22px !important;
  font-weight: 800 !important;
  box-shadow: 0 18px 45px rgba(46, 107, 255, 0.28) !important;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
  filter: brightness(1.05);
}

/* Card risultati */
.match-card {
  background: rgba(255,255,255,0.92);
  border-radius: 22px;
  padding: 22px 22px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
  border: 1px solid rgba(30,41,59,0.06);
  margin-bottom: 18px;
}
.rank-badge {
  width: 64px;
  height: 64px;
  border-radius: 999px;
  background: linear-gradient(135deg, #2F6BFF 0%, #7C3AED 100%);
  display: grid;
  place-items: center;
  color: white;
  font-size: 22px;
  font-weight: 900;
  box-shadow: 0 18px 40px rgba(124, 58, 237, 0.22);
}
.country-name {
  font-size: 40px;
  font-weight: 900;
  margin: 0;
  color: #0B1220;
}
.cluster-text {
  margin-top: -6px;
  color: #6B7280;
  font-size: 18px;
}
.score-wrap {
  text-align: right;
}
.score-number {
  font-size: 54px;
  font-weight: 900;
  margin: 0;
}
.score-label {
  margin-top: -8px;
  color: #6B7280;
  font-size: 18px;
}

/* Barre */
.section-title {
  font-size: 24px;
  font-weight: 900;
  margin: 12px 0 10px 0;
  color: #0B1220;
}
.metric-row {
  display: grid;
  grid-template-columns: 120px 1fr 70px;
  align-items: center;
  gap: 12px;
  margin: 12px 0;
}
.metric-label {
  color: #4B5563;
  font-size: 18px;
  font-weight: 600;
}
.metric-bar {
  height: 14px;
  background: #E5E7EB;
  border-radius: 999px;
  overflow: hidden;
}
.metric-fill {
  height: 100%;
  border-radius: 999px;
}
.metric-val {
  text-align: right;
  font-size: 20px;
  font-weight: 800;
  color: #0B1220;
}

/* Expander */
details summary {
  font-size: 18px !important;
  font-weight: 800 !important;
  color: #2F6BFF !important;
}

/* Tabs - unica barra blu sotto la tab selezionata */
button[data-baseweb="tab"] {
  color: #64748B !important;
  font-weight: 600;
  font-size: 18px !important;
  border-bottom: none !important;
  background: transparent !important;
  padding: 12px 24px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: #2F6BFF !important;
  font-weight: 800;
  border-bottom: 3px solid #2F6BFF !important;
  background: transparent !important;
}
button[data-baseweb="tab"]:hover {
  color: #7C3AED !important;
  background: transparent !important;
}
/* Container tabs senza bordo */
div[data-baseweb="tab-list"] {
  border-bottom: none !important;
  gap: 16px !important;
}
/* Rimuove highlight arancione */
div[data-baseweb="tab-highlight"] {
  display: none !important;
}
div[data-baseweb="tab-border"] {
  display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_levels_and_trends():
    """Carica i dati da CSV"""
    try:
        df_levels_latest = pd.read_csv("df_levels_latest.csv")
        df_trends = pd.read_csv("df_trends.csv")
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Make sure 'df_levels_latest.csv' and 'df_trends.csv' are in the same directory")
        st.stop()

    # Validazioni
    must_levels = ["Country"] + INDEX_COLS
    for c in must_levels:
        if c not in df_levels_latest.columns:
            raise ValueError(f"df_levels_latest: missing column '{c}'")

    must_trends = ["Country"] + TREND_COLS
    for c in must_trends:
        if c not in df_trends.columns:
            raise ValueError(f"df_trends: missing column '{c}'")

    # Seleziona solo colonne necessarie
    df_levels_latest = df_levels_latest[must_levels].copy()
    df_trends = df_trends[must_trends].copy()

    return df_levels_latest, df_trends


def choose_k_by_silhouette(X: np.ndarray, k_min=2, k_max=8, random_state=42):
    """Trova il numero ottimale di cluster"""
    best_k, best_score = None, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score


@st.cache_data
def add_clusters(df_levels_latest: pd.DataFrame, k: int | None = None, random_state: int = 42):
    """Aggiunge colonna cluster al dataframe"""
    X = df_levels_latest[INDEX_COLS].values
    if k is None:
        k, _ = choose_k_by_silhouette(X, k_min=2, k_max=8, random_state=random_state)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    out = df_levels_latest.copy()
    out["cluster"] = km.fit_predict(X)
    return out, k


# =========================
# RANKING ALGORITHM
# =========================
def normalize_weights(w: dict[str, int]) -> np.ndarray:
    """Normalizza i pesi utente"""
    arr = np.array([w[c] for c in INDEX_COLS], dtype=float)
    s = arr.sum()
    return arr / s if s > 0 else np.ones(len(INDEX_COLS)) / len(INDEX_COLS)


def compute_ranking(df_levels: pd.DataFrame, df_trends: pd.DataFrame, weights: dict):
    """
    Implementa il ranking cluster-aware completo come nel notebook
    """
    # 1. Normalizza pesi utente
    user_w_levels = normalize_weights(weights)
    user_w_trends = user_w_levels.copy()

    # 2. Weighted similarity sui livelli
    X_levels = df_levels[INDEX_COLS].values * user_w_levels
    u_levels = np.ones((1, len(INDEX_COLS))) * user_w_levels
    sim_levels = cosine_similarity(u_levels, X_levels).flatten()

    # 3. Weighted similarity sui trend
    X_trends_vals = df_trends[TREND_COLS].values * user_w_trends
    u_trends = np.ones((1, len(TREND_COLS))) * user_w_trends
    sim_trends = cosine_similarity(u_trends, X_trends_vals).flatten()

    # 4. Cluster-aware bonus
    df_merged = df_levels.copy()
    df_merged["sim_levels"] = sim_levels
    df_merged["sim_trends"] = sim_trends
    df_merged["trend_score"] = df_trends[TREND_COLS].mean(axis=1).values

    # Media trend per cluster
    df_merged["cluster_trend_mean"] = df_merged.groupby("cluster")["trend_score"].transform("mean")
    df_merged["relative_trend"] = df_merged["trend_score"] - df_merged["cluster_trend_mean"]

    # Normalizza relative_trend in [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_merged["relative_trend_norm"] = scaler.fit_transform(df_merged[["relative_trend"]])

    # 5. Score finale (pesi dal notebook)
    w_levels, w_trends, w_cluster = 0.6, 0.3, 0.1
    df_merged["final_score"] = (
        w_levels * df_merged["sim_levels"]
        + w_trends * df_merged["sim_trends"]
        + w_cluster * df_merged["relative_trend_norm"]
    )

    return df_merged.sort_values("final_score", ascending=False)


# =========================
# HELPERS
# =========================
def score_color(score_0_1: float) -> str:
    """Colore dinamico in base al punteggio"""
    if score_0_1 >= 0.85:
        return "#22C55E"  # verde
    if score_0_1 >= 0.72:
        return "#2563EB"  # blu
    return "#F59E0B"  # giallo


def progress_row(label: str, value_0_1: float, fill_color: str) -> str:
    """Genera HTML per una barra di progresso"""
    pct = int(round(float(value_0_1) * 100))
    pct = max(0, min(100, pct))
    return f"""
    <div class="metric-row">
      <div class="metric-label">{label}</div>
      <div class="metric-bar"><div class="metric-fill" style="width:{pct}%; background:{fill_color};"></div></div>
      <div class="metric-val">{pct}%</div>
    </div>
    """


def indicator_label(key: str) -> str:
    """Ottiene label breve per indicatore"""
    return IND_MAP[key]["title"].split(" & ")[0] if key in IND_MAP else key


def build_indicator_block(df_row: pd.Series, is_trend: bool):
    """Costruisce blocco con barre indicatori (livelli o trend)"""
    left_keys = ["income_index", "cost_index", "safety_index", "mobility_index"]
    right_keys = ["job_index", "health_env_index", "social_index"]

    if is_trend:
        keys_left = [f"trend_{k}" for k in left_keys]
        keys_right = [f"trend_{k}" for k in right_keys]
        title = "📈 Growth Trends"
        color = "#22C55E"
        label_map = {f"trend_{k}": indicator_label(k) for k in left_keys + right_keys}
    else:
        keys_left = left_keys
        keys_right = right_keys
        title = "📊 Current Levels"
        color = "#2563EB"
        label_map = {k: indicator_label(k) for k in left_keys + right_keys}

    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        html = ""
        for k in keys_left:
            if k in df_row.index and pd.notna(df_row[k]):
                html += progress_row(label_map.get(k, k), float(df_row[k]), color)
        st.markdown(html, unsafe_allow_html=True)

    with c2:
        html = ""
        for k in keys_right:
            if k in df_row.index and pd.notna(df_row[k]):
                html += progress_row(label_map.get(k, k), float(df_row[k]), color)
        st.markdown(html, unsafe_allow_html=True)


# =========================
# MAIN APP
# =========================
try:
    df_levels_latest, df_trends = load_levels_and_trends()
    df_levels_with_cluster, k_used = add_clusters(df_levels_latest, k=None)
    df_full = df_levels_with_cluster.merge(df_trends, on="Country", how="left")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# =========================
# HEADER
# =========================
st.markdown(
    """
<div class="cc-header">
  <div class="cc-logo">
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>
  </div>
  <div class="cc-title">Career Compass</div>
</div>
<div class="cc-subtitle">Find your ideal country based on what matters most to you</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# =========================
# SESSION STATE
# =========================
if "computed" not in st.session_state:
    st.session_state["computed"] = False
if "ranking" not in st.session_state:
    st.session_state["ranking"] = None
if "weights" not in st.session_state:
    st.session_state["weights"] = {c: 3 for c in INDEX_COLS}

# =========================
# TABS
# =========================
tab_prefs, tab_results = st.tabs(["⚙️  Set Preferences", "📊  Results"])

# =========================
# TAB 1: PREFERENCES
# =========================
with tab_prefs:
    st.markdown('<div class="cc-card">', unsafe_allow_html=True)
    st.markdown(
        """
<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
  <div style="width:34px; height:34px; border-radius:10px; background:#EEF2FF; display:grid; place-items:center;">
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2F6BFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
  </div>
  <div style="font-size:22px; font-weight:800; color:#0B1220;">
    Rate each factor from 1 (not important) to 5 (very important)
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("prefs_form", clear_on_submit=False):
        weights = dict(st.session_state["weights"])

        for d in INDICATORS:
            k = d["key"]
            # Crea un container per ogni indicatore
            current_value = st.session_state.get(f"slider_{k}", weights.get(k, 3))

            st.markdown(
                f"""
<div class="pref-row">
  <div class="pref-icon" style="background:{d["color"]};">
    {d["icon"]}
  </div>
  <div>
    <p class="pref-title">{d["title"]}</p>
    <p class="pref-desc">{d["desc"]}</p>
  </div>
  <div class="pref-value">{current_value}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            weights[k] = st.slider(
                label="",
                min_value=1,
                max_value=5,
                value=int(weights.get(k, 3)),
                step=1,
                key=f"slider_{k}",
                label_visibility="collapsed",
            )
            st.markdown(
                """
<div style="display:flex; justify-content:space-between; margin-top:-6px; margin-bottom:8px;">
  <div class="small-muted">Not Important</div>
  <div class="small-muted">Very Important</div>
</div>
""",
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button("🔍  Find My Best Matches")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        st.session_state["weights"] = weights

        # Calcola ranking con algoritmo cluster-aware completo
        ranking_full = compute_ranking(df_levels_with_cluster, df_trends, weights)

        # Salva top 10
        top10 = ranking_full[["Country", "cluster", "final_score"]].head(10).copy()
        top10.columns = ["Country", "cluster", "score"]
        top10["rank"] = np.arange(1, 11)

        st.session_state["ranking"] = top10
        st.session_state["computed"] = True

        st.success("✅ Matches updated! Open the **Results** tab to view your recommendations.")

# =========================
# TAB 2: RESULTS
# =========================
with tab_results:
    if not st.session_state["computed"] or st.session_state["ranking"] is None:
        st.info("👈 Set your preferences first, then click **Find My Best Matches**")
    else:
        ranking = st.session_state["ranking"].copy()

        for _, r in ranking.iterrows():
            country = r["Country"]
            cluster = int(r["cluster"])
            score = float(r["score"])
            score_pct = score * 100.0
            sc_color = score_color(score)

            # Card header
            st.markdown(
                f"""
<div class="match-card">
  <div style="display:flex; align-items:center; gap:18px;">
    <div class="rank-badge">{int(r["rank"])}</div>
    <div style="flex:1;">
      <p class="country-name">{country}</p>
      <div class="cluster-text">Cluster {cluster + 1}</div>
    </div>
    <div class="score-wrap">
      <p class="score-number" style="color:{sc_color};">{score_pct:.1f}%</p>
      <div class="score-label">Match Score</div>
    </div>
  </div>
""",
                unsafe_allow_html=True,
            )

            # Dettagli espandibili
            with st.expander("Show Details"):
                row = df_full.loc[df_full["Country"] == country]
                if row.empty:
                    st.warning("No details available for this country.")
                else:
                    row = row.iloc[0]
                    build_indicator_block(row, is_trend=False)
                    build_indicator_block(row, is_trend=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown(
            """
<div style="text-align:center; color:#7C8496; padding: 16px 0 6px 0;">
  Career Compass • Data Science Lab Project<br/>
  Helping young professionals find their ideal destination
</div>
""",
            unsafe_allow_html=True,
        )
