# =============================================================================
# ⚽ PREDITOR DE RISCO DE LESÃO — Copa do Mundo 2022
# Streamlit App | Elasticsearch + XGBoost
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- Page config (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="⚽ Injury Risk Predictor · Copa 2022",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — Dark football-themed aesthetic
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;600&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
}
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1729 60%, #0d1a0f 100%); }

/* ---- Headers ---- */
h1 { font-family: 'Bebas Neue', cursive !important; letter-spacing: 3px; font-size: 3rem !important; color: #f5e642 !important; }
h2 { font-family: 'Bebas Neue', cursive !important; letter-spacing: 2px; color: #c8e6c9 !important; }
h3 { font-family: 'IBM Plex Mono', monospace !important; color: #81d4fa !important; font-size: 1rem !important; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a0f 0%, #0a1628 100%);
    border-right: 1px solid #1e3a1e;
}
[data-testid="stSidebar"] .stMarkdown { color: #a5d6a7; }

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #78909c !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { font-family: 'Bebas Neue', cursive; font-size: 2rem; }

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] { border: 1px solid #1e3a2e; border-radius: 8px; }

/* ---- Risk badge ---- */
.risk-high   { background: #b71c1c; color: #ffcdd2; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; display: inline-block; }
.risk-medium { background: #f57f17; color: #fff8e1; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; display: inline-block; }
.risk-low    { background: #1b5e20; color: #c8e6c9; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; display: inline-block; }

/* ---- Player card ---- */
.player-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(245,230,66,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.player-name { font-family: 'Bebas Neue', cursive; font-size: 1.8rem; color: #f5e642; letter-spacing: 2px; }
.team-name   { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #78909c; text-transform: uppercase; }

/* ---- Progress bar overrides ---- */
.stProgress > div > div { background-color: #f5e642; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { color: #f5e642 !important; border-bottom-color: #f5e642 !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: #f5e642;
    color: #0a0e1a;
    font-family: 'Bebas Neue', cursive;
    font-size: 1.1rem;
    letter-spacing: 2px;
    border: none;
    border-radius: 6px;
    padding: 10px 32px;
    width: 100%;
    transition: transform 0.1s, box-shadow 0.1s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(245,230,66,0.4); }

/* ---- Selectbox ---- */
[data-baseweb="select"] { background: rgba(255,255,255,0.05) !important; border-color: #1e3a2e !important; }

/* ---- Divider ---- */
hr { border-color: rgba(255,255,255,0.08); }

/* ---- Info boxes ---- */
.stInfo { background: rgba(129,212,250,0.1); border-left-color: #81d4fa; }
.stWarning { background: rgba(245,127,23,0.15); border-left-color: #f57f17; }
.stSuccess { background: rgba(27,94,32,0.25); border-left-color: #4caf50; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONNECTION & DATA LOADING
# =============================================================================
ES_HOST    = st.secrets["ES_HOST"]
ES_API_KEY = st.secrets["ES_API_KEY"]
INDEX      = "copa"

STAGE_ORDER = {
    "Group Stage": 1, "Round of 16": 2, "Quarter-finals": 3,
    "Semi-finals": 4, "3rd Place Final": 5, "Final": 6
}

@st.cache_resource(show_spinner="🔗 Conectando ao Elasticsearch...")
def get_es_client():
    from elasticsearch import Elasticsearch
    return Elasticsearch(ES_HOST, api_key=ES_API_KEY, verify_certs=True)

def run_esql(es, query: str) -> pd.DataFrame:
    resp = es.esql.query(body={"query": query})
    cols = [c["name"] for c in resp["columns"]]
    return pd.DataFrame(resp["values"], columns=cols)

@st.cache_data(show_spinner="📡 Carregando dados do índice 'copa'...")
def load_data():
    es = get_es_client()

    ESQL_FEATURES = """
FROM copa
| STATS
    injury_stoppages      = COUNT(*) WHERE type == "Injury Stoppage" AND injury_stoppage_in_chain IS NOT NULL,
    player_offs           = COUNT(*) WHERE type == "Player Off",
    duels                 = COUNT(*) WHERE type == "Duel",
    aerial_duels          = COUNT(*) WHERE type == "Duel" AND duel_type == "Aerial Lost",
    tackle_duels          = COUNT(*) WHERE type == "Duel" AND duel_type == "Tackle",
    fouls_won             = COUNT(*) WHERE type == "Foul Won",
    fouls_committed       = COUNT(*) WHERE type == "Foul Committed",
    dangerous_fouls       = COUNT(*) WHERE type == "Foul Committed" AND foul_committed_type.keyword == "Dangerous Play",
    under_pressure_events = COUNT(*) WHERE under_pressure == "True",
    carries               = COUNT(*) WHERE type == "Carry",
    passes                = COUNT(*) WHERE type == "Pass",
    shots                 = COUNT(*) WHERE type == "Shot",
    pressures             = COUNT(*) WHERE type == "Pressure",
    dribbles              = COUNT(*) WHERE type == "Dribble",
    clearances            = COUNT(*) WHERE type == "Clearance",
    blocks                = COUNT(*) WHERE type == "Block",
    interceptions         = COUNT(*) WHERE type == "Interception",
    injury_clearance      = COUNT(*) WHERE type == "Pass" AND pass_outcome == "Injury Clearance",
    injury_subs           = COUNT(*) WHERE type == "Substitution" AND substitution_outcome_id.keyword == "102.0",
    total_events          = COUNT(*),
    max_minute            = MAX(minute)
    BY player, match_id, team
| WHERE player IS NOT NULL
| SORT match_id ASC, player ASC
| LIMIT 10000
"""

    ESQL_MATCHES = """
FROM copa
| STATS
    match_date        = MAX(match_date),
    home_team         = MAX(home_team),
    away_team         = MAX(away_team),
    match_week        = MAX(match_week),
    competition_stage = MAX(competition_stage)
    BY match_id
| SORT match_date ASC, match_id ASC
| LIMIT 200
"""

    df_features = run_esql(es, ESQL_FEATURES)
    df_matches  = run_esql(es, ESQL_MATCHES)
    return df_features, df_matches

@st.cache_data(show_spinner="⚙️ Engenharia de features...")
def build_features(df_features, df_matches):
    df = df_features.merge(
        df_matches[["match_id", "match_date", "match_week", "competition_stage"]],
        on="match_id", how="left"
    )
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["player", "match_date"]).reset_index(drop=True)

    df["injury_event"] = (
        (df["injury_stoppages"] > 0) |
        (df["player_offs"]      > 0) |
        (df["injury_subs"]      > 0)
    ).astype(int)

    df["minutes_played"] = df["max_minute"].clip(lower=1)
    for col in ["duels", "fouls_committed", "fouls_won", "under_pressure_events",
                "carries", "pressures", "clearances", "aerial_duels", "tackle_duels"]:
        df[f"{col}_per_min"] = df[col] / df["minutes_played"]

    df["heuristic_risk"] = (
        df["injury_stoppages"]      * 40 +
        df["player_offs"]           * 30 +
        df["duels"]                 *  3 +
        df["fouls_committed"]       *  4 +
        df["fouls_won"]             *  2 +
        df["under_pressure_events"] *  1 +
        df["dangerous_fouls"]       * 10 +
        df["aerial_duels"]          *  2
    )

    lag_cols = ["injury_stoppages", "player_offs", "duels", "fouls_committed",
                "fouls_won", "under_pressure_events", "heuristic_risk",
                "aerial_duels", "dangerous_fouls", "injury_clearance"]
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby("player")[col].shift(1)

    for col in ["heuristic_risk", "duels", "under_pressure_events", "fouls_committed"]:
        df[f"{col}_roll2"] = (
            df.groupby("player")[col]
              .transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        )

    df["stage_ordinal"]  = df["competition_stage"].map(STAGE_ORDER).fillna(1)
    df["matches_played"] = df.groupby("player").cumcount()
    return df

@st.cache_resource(show_spinner="🏋️ Treinando modelo XGBoost...")
def train_model(df):
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    FEATURE_COLS = [
        "injury_stoppages_lag1", "player_offs_lag1", "duels_lag1",
        "fouls_committed_lag1", "fouls_won_lag1", "under_pressure_events_lag1",
        "aerial_duels_lag1", "dangerous_fouls_lag1", "injury_clearance_lag1",
        "heuristic_risk_lag1",
        "heuristic_risk_roll2", "duels_roll2",
        "under_pressure_events_roll2", "fouls_committed_roll2",
        "duels_per_min", "fouls_committed_per_min",
        "under_pressure_events_per_min", "carries_per_min",
        "stage_ordinal", "matches_played", "minutes_played", "total_events",
    ]

    df_model = df.dropna(subset=["injury_stoppages_lag1"]).copy()
    X = df_model[FEATURE_COLS].fillna(0)
    y = df_model["injury_event"]

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    model.fit(X, y)

    return model, FEATURE_COLS, X, y, auc_scores


# =============================================================================
# PREDICTION HELPERS
# =============================================================================
def predict_player(df, model, FEATURE_COLS, player_name, next_stage):
    player_data = df[df["player"] == player_name].sort_values("match_date")
    if player_data.empty:
        return None

    last_match = player_data.iloc[-1]
    last_2     = player_data.tail(2)

    features = {
        "injury_stoppages_lag1":         last_match["injury_stoppages"],
        "player_offs_lag1":              last_match["player_offs"],
        "duels_lag1":                    last_match["duels"],
        "fouls_committed_lag1":          last_match["fouls_committed"],
        "fouls_won_lag1":                last_match["fouls_won"],
        "under_pressure_events_lag1":    last_match["under_pressure_events"],
        "aerial_duels_lag1":             last_match["aerial_duels"],
        "dangerous_fouls_lag1":          last_match["dangerous_fouls"],
        "injury_clearance_lag1":         last_match["injury_clearance"],
        "heuristic_risk_lag1":           last_match["heuristic_risk"],
        "heuristic_risk_roll2":          last_2["heuristic_risk"].mean(),
        "duels_roll2":                   last_2["duels"].mean(),
        "under_pressure_events_roll2":   last_2["under_pressure_events"].mean(),
        "fouls_committed_roll2":         last_2["fouls_committed"].mean(),
        "duels_per_min":                 last_match["duels_per_min"],
        "fouls_committed_per_min":       last_match["fouls_committed_per_min"],
        "under_pressure_events_per_min": last_match["under_pressure_events_per_min"],
        "carries_per_min":               last_match["carries_per_min"],
        "stage_ordinal":                 STAGE_ORDER.get(next_stage, 1),
        "matches_played":                len(player_data),
        "minutes_played":                last_match["minutes_played"],
        "total_events":                  last_match["total_events"],
    }

    X_pred = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
    prob   = float(model.predict_proba(X_pred)[0][1])

    return {
        "player":           player_name,
        "team":             last_match["team"],
        "last_match_id":    int(last_match["match_id"]),
        "last_stage":       last_match.get("competition_stage", "—"),
        "probability":      round(prob, 4),
        "heuristic_score":  int(last_match["heuristic_risk"]),
        "injury_stoppages": int(last_match["injury_stoppages"]),
        "player_offs":      int(last_match["player_offs"]),
        "duels":            int(last_match["duels"]),
        "fouls_committed":  int(last_match["fouls_committed"]),
        "fouls_won":        int(last_match["fouls_won"]),
        "under_pressure":   int(last_match["under_pressure_events"]),
        "features":         features,
    }

def risk_badge(prob):
    if prob >= 0.55:
        return '<span class="risk-high">🔴 ALTO</span>'
    elif prob >= 0.35:
        return '<span class="risk-medium">🟡 MÉDIO</span>'
    else:
        return '<span class="risk-low">🟢 BAIXO</span>'

def risk_color(prob):
    if prob >= 0.55: return "#ef5350"
    elif prob >= 0.35: return "#ffa726"
    else: return "#66bb6a"


# =============================================================================
# APP LAYOUT
# =============================================================================
st.markdown('<h1>⚽ PREDITOR DE RISCO DE LESÃO</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;color:#546e7a;font-size:0.85rem;margin-top:-12px;letter-spacing:1px;">'
    'COPA DO MUNDO 2022 · STATSBOMB DATA · XGBOOST MODEL</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# --- Load & train ---
try:
    with st.spinner(""):
        df_features, df_matches = load_data()
        df = build_features(df_features, df_matches)
        model, FEATURE_COLS, X_train, y_train, auc_scores = train_model(df)
    data_ok = True
except Exception as e:
    st.error(f"❌ Erro ao conectar ou carregar dados: `{e}`")
    data_ok = False

if data_ok:
    all_players = sorted(df["player"].dropna().unique().tolist())
    df_model    = df.dropna(subset=["injury_stoppages_lag1"])

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        st.markdown("---")

        next_stage = st.selectbox(
            "🏆 Próxima fase",
            list(STAGE_ORDER.keys()),
            index=2
        )
        st.markdown("---")

        st.markdown("### 📊 Resumo do Modelo")
        st.metric("ROC-AUC médio", f"{auc_scores.mean():.3f}")
        st.metric("Desvio padrão", f"± {auc_scores.std():.3f}")
        st.metric("Amostras treino", len(X_train))
        positives = int(y_train.sum())
        st.metric("Eventos de lesão", f"{positives} ({y_train.mean()*100:.1f}%)")

        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#37474f;">'
            'Dados: StatsBomb · Elasticsearch<br>'
            'Índice: <b style="color:#78909c;">copa</b></p>',
            unsafe_allow_html=True
        )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Jogador Individual", "🏆 Ranking Geral", "📊 Feature Importance"])

    # ─── TAB 1: Individual ───────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("### Selecione o Jogador")
            selected_player = st.selectbox(
                "Jogador", all_players,
                label_visibility="collapsed",
                key="player_select"
            )
            run_btn = st.button("🔮 ANALISAR RISCO", key="run_pred")

        if run_btn or True:  # Always show last selected
            result = predict_player(df, model, FEATURE_COLS, selected_player, next_stage)

            if result:
                prob = result["probability"]
                with col_right:
                    st.markdown(
                        f'<div class="player-card">'
                        f'<div class="player-name">{result["player"]}</div>'
                        f'<div class="team-name">{result["team"]} · {result["last_stage"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🎯 Probabilidade", f"{prob*100:.1f}%")
                c2.metric("🔥 Score Heurístico", result["heuristic_score"])
                c3.metric("⚔️ Duelos", result["duels"])
                c4.metric("😤 Sob Pressão", result["under_pressure"])

                st.markdown("")
                st.markdown(
                    f"**Nível de Risco:** {risk_badge(prob)}",
                    unsafe_allow_html=True
                )
                st.progress(min(prob, 1.0))

                st.markdown("---")
                st.markdown("### 📋 Detalhes da Última Partida")
                detail_cols = st.columns(3)
                stats = [
                    ("Injury Stoppages", result["injury_stoppages"], "🚑"),
                    ("Player Offs",       result["player_offs"],      "🚶"),
                    ("Faltas Cometidas",  result["fouls_committed"],  "🟨"),
                    ("Faltas Sofridas",   result["fouls_won"],        "🤕"),
                    ("Duelos Aéreos",     result["features"].get("aerial_duels_lag1", 0), "🦅"),
                    ("Faltas Perigosas",  result["features"].get("dangerous_fouls_lag1", 0), "⚠️"),
                ]
                for i, (label, val, icon) in enumerate(stats):
                    detail_cols[i % 3].metric(f"{icon} {label}", int(val))

    # ─── TAB 2: Ranking ──────────────────────────────────────────────────────
    with tab2:
        st.markdown(f"### 🏆 Ranking de Risco — Próxima Fase: **{next_stage}**")

        top_n = st.slider("Exibir top N jogadores", 5, 50, 20, key="topn")

        with st.spinner("Calculando risco para todos os jogadores..."):
            results = []
            progress = st.progress(0)
            players_list = df["player"].dropna().unique().tolist()
            for i, p in enumerate(players_list):
                r = predict_player(df, model, FEATURE_COLS, p, next_stage)
                if r:
                    results.append({
                        "Jogador":       r["player"],
                        "Time":          r["team"],
                        "Prob (%)":      round(r["probability"] * 100, 1),
                        "Risco":         ("🔴 ALTO" if r["probability"] >= 0.55 else
                                          "🟡 MÉDIO" if r["probability"] >= 0.35 else "🟢 BAIXO"),
                        "Score Heur.":   r["heuristic_score"],
                        "Duelos":        r["duels"],
                        "Sob Pressão":   r["under_pressure"],
                    })
                progress.progress((i + 1) / len(players_list))
            progress.empty()

        ranking_df = (
            pd.DataFrame(results)
              .sort_values("Prob (%)", ascending=False)
              .reset_index(drop=True)
        )
        ranking_df.index += 1

        # Colour-coded display
        def highlight_risk(row):
            p = row["Prob (%)"]
            if p >= 55:   color = "rgba(183,28,28,0.25)"
            elif p >= 35: color = "rgba(245,127,23,0.20)"
            else:         color = "rgba(27,94,32,0.20)"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            ranking_df.head(top_n).style.apply(highlight_risk, axis=1),
            use_container_width=True,
            height=600
        )

        # Summary
        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("🔴 Alto Risco",  len(ranking_df[ranking_df["Prob (%)"] >= 55]))
        r2.metric("🟡 Médio Risco", len(ranking_df[(ranking_df["Prob (%)"] >= 35) & (ranking_df["Prob (%)"] < 55)]))
        r3.metric("🟢 Baixo Risco", len(ranking_df[ranking_df["Prob (%)"] < 35]))

    # ─── TAB 3: Feature Importance ───────────────────────────────────────────
    with tab3:
        st.markdown("### 📊 Importância das Features")

        feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        feat_imp = feat_imp.sort_values(ascending=True)

        # Build horizontal bar chart with Altair
        try:
            import altair as alt
            fi_df = feat_imp.reset_index()
            fi_df.columns = ["Feature", "Importance"]
            fi_df["Color"] = fi_df["Importance"].apply(
                lambda x: "#f5e642" if x >= feat_imp.quantile(0.75) else
                          "#4db6ac" if x >= feat_imp.quantile(0.5) else "#546e7a"
            )
            chart = (
                alt.Chart(fi_df)
                   .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                   .encode(
                       x=alt.X("Importance:Q", title="Importância"),
                       y=alt.Y("Feature:N", sort="-x", title=""),
                       color=alt.Color("Color:N", scale=None, legend=None),
                       tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")]
                   )
                   .properties(height=550, background="transparent")
                   .configure_axis(
                       labelColor="#78909c", titleColor="#546e7a",
                       gridColor="rgba(255,255,255,0.05)", labelFontSize=12
                   )
                   .configure_view(stroke="transparent")
            )
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            # Fallback: text table
            fi_display = feat_imp.sort_values(ascending=False).reset_index()
            fi_display.columns = ["Feature", "Importance"]
            fi_display["Bar"] = fi_display["Importance"].apply(
                lambda x: "█" * max(1, int(x * 80))
            )
            st.dataframe(fi_display, use_container_width=True)

        st.markdown("---")
        st.markdown(
            """
**Grupos de features:**
- **Lag 1** — dados da partida anterior (preditor mais direto)
- **Rolling 2** — média das 2 últimas partidas (carga acumulada)
- **Por minuto** — intensidade física normalizada
- **Contexto** — fase da competição, partidas acumuladas, minutos jogados
            """
        )
