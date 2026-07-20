"""
My Spotify Content-Based Recommendation Engine
================================================
A  Streamlit dashboard built on top of a content-based
(cosine similarity) recommendation engine trained on the user's own
exported Spotify listening history.

Sections in this file:
    1. Page config & theming (CSS)
    2. Header / branding
    3.Load saved model artifacts -> load_models()
    4. Recommendation engine                     -> RecommendationEngine
    5. Sidebar filters
    6. Seed-song selection + recommend action
    7. Results: KPIs, table, charts, explanations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import joblib
import sys
from pathlib import Path

# Streamlit Cloud runs this file as Dashboard/app.py, which only puts the
# Dashboard/ folder on sys.path — not the repo root. src/ lives next to
# Dashboard/, not inside it, so it has to be added manually or the import
# below fails with "ModuleNotFoundError: No module named 'src'".
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.Recommender import ContentBasedRecommender

# ============================================================
# 1. PAGE CONFIG & THEMING
# ============================================================

st.set_page_config(
    page_title="My Spotify Content Based Recommendation Engine",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

PINK = "#FFB6D9"
MINT = "#A8F0D1"
LILAC = "#CBB8F5"
CYAN = "#7BF1FF"
NAVY = "#12122b"
NAVY_2 = "#1a1a3d"
CARD = "#1e1e46"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&family=Poppins:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&family=Nunito:wght@400;600;700&display=swap');

/* ---------- App background ---------- */
.stApp {{
    background:
        radial-gradient(circle at 15% 10%, rgba(203,184,245,0.12) 0%, transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(168,240,209,0.10) 0%, transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(255,182,217,0.10) 0%, transparent 45%),
        {NAVY};
    color: #EAEAF7;
    font-family: 'Inter', sans-serif;
}}

section[data-testid="stSidebar"] {{
    background: {NAVY_2};
    border-right: 1px solid rgba(203,184,245,0.25);
}}
section[data-testid="stSidebar"] * {{
    font-family: 'Inter', sans-serif !important;
}}

/* ---------- Headings ---------- */
h1, h2, h3, .cute-title {{
    font-family: 'Baloo 2', 'Poppins', cursive !important;
    letter-spacing: 0.3px;
}}

h1 {{ color: {PINK} !important; }}
h2 {{ color: {MINT} !important; }}
h3 {{ color: {LILAC} !important; }}

/* ---------- KPI cards ---------- */
.kpi-card {{
    background: linear-gradient(145deg, {CARD}, #232357);
    border: 1px solid rgba(123,241,255,0.25);
    border-radius: 18px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 0 18px rgba(203,184,245,0.08);
}}
.kpi-label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem;
    color: #B9B9E0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: {CYAN};
}}

/* ---------- Explanation chips ---------- */
.reason-chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(255,182,217,0.15);
    color: {PINK};
    border: 1px solid rgba(255,182,217,0.4);
}}

/* ---------- Header banner ---------- */
.header-wrap {{
    text-align: center;
    padding: 10px 0 6px 0;
}}
.header-sub {{
    font-family: 'Nunito', sans-serif !important;
    color: #B9B9E0;
    font-size: 0.95rem;
    margin-top: -6px;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    background: linear-gradient(90deg, {PINK}, {LILAC});
    color: {NAVY};
    font-family: 'Baloo 2', cursive;
    font-weight: 700;
    border: none;
    border-radius: 14px;
    padding: 0.5em 1.4em;
}}
.stButton > button:hover {{
    background: linear-gradient(90deg, {MINT}, {CYAN});
    color: {NAVY};
}}

/* ---------- Dataframe tweak ---------- */
[data-testid="stDataFrame"] {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(123,241,255,0.2);
}}

hr {{ border-color: rgba(203,184,245,0.2); }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# 2. HEADER / BRANDING
# ============================================================
# Note: the real Spotify logo is a trademarked brand asset, so instead of
# reproducing it we use a playful, original headphone/note mark in the
# same pastel-on-navy palette to keep the "techy album-art" feel.

LOGO_SVG = f"""
<div style="display:flex; justify-content:center; margin-top:6px;">
<svg width="86" height="86" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="46" fill="{CARD}" stroke="{MINT}" stroke-width="2.5"/>
  <circle cx="50" cy="50" r="34" fill="none" stroke="{PINK}" stroke-width="2.5" stroke-dasharray="4 6"/>
  <path d="M30 55 Q50 35 70 55" stroke="{CYAN}" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M32 55 Q50 42 68 55" stroke="{LILAC}" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.7"/>
  <circle cx="30" cy="58" r="6" fill="{PINK}"/>
  <circle cx="70" cy="58" r="6" fill="{MINT}"/>
</svg>
</div>
"""

st.markdown(LOGO_SVG, unsafe_allow_html=True)
st.markdown(
    """
    <div class="header-wrap">
        <h1 style="margin-bottom:0;">My Spotify Content Based Recommendation Engine</h1>
        <div class="header-sub">pick a few songs you love ✨ get a pastel-coded, cosine-similarity powered playlist</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# 3. LOAD SAVED MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_models():
    try:
        tracks_df = joblib.load("models/tracks_df.pkl")
        similarity_matrix = joblib.load("models/similarity_matrix.pkl")
        track_lookup = joblib.load("models/track_lookup.pkl")

        # Guard against rows with missing values (pandas stores these as
        # NaN, a float) — mixing NaN with strings breaks sorted(),
        # slicing, and other string ops used throughout the dashboard.
        text_cols = ["artist_name", "track_name", "track_key", "playlists"]
        for col in text_cols:
            if col in tracks_df.columns:
                tracks_df[col] = tracks_df[col].fillna("Unknown").astype(str)

        numeric_cols = [
            "preference_score", "play_count", "total_minutes_played",
            "playlist_count", "days_since_last_play",
        ]
        for col in numeric_cols:
            if col in tracks_df.columns:
                tracks_df[col] = pd.to_numeric(tracks_df[col], errors="coerce").fillna(0)

        if "in_library" in tracks_df.columns:
            tracks_df["in_library"] = tracks_df["in_library"].fillna(False).astype(bool)

        return tracks_df, similarity_matrix, track_lookup

    except FileNotFoundError:
        st.error(
            "Model files were not found. Please generate them from the notebook first."
        )
        st.stop()

with st.spinner("Loading recommendation model..."):
    tracks_df, similarity_matrix, track_lookup = load_models()
# ============================================================
# 4. RECOMMENDATION ENGINE
# ============================================================
# The engine itself lives in Recommender.py (ContentBasedRecommender) —
# fixed to include track_key in its outputs, support multiple seed
# tracks at once, and accept a pre-filtered candidate pool from the
# sidebar filters below.

engine = ContentBasedRecommender(tracks_df, similarity_matrix, track_lookup)

# ============================================================
# 5. SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("### 🎛️ Filters")

all_artists = sorted(tracks_df["artist_name"].unique().tolist())
selected_artists = st.sidebar.multiselect(
    "Artist",
    options=all_artists,
    help="Narrow recommendations down to these artists only.",
)

all_playlists = sorted(tracks_df["playlists"].unique().tolist())
selected_playlists = st.sidebar.multiselect(
    "Playlist (used as a genre/mood proxy — this dataset has no genre column)",
    options=all_playlists,
)

max_days = int(tracks_df["days_since_last_play"].max())
recency_cutoff = st.sidebar.slider(
    "Only recommend tracks played in the last N days",
    min_value=1,
    max_value=max_days if max_days > 1 else 2,
    value=max_days if max_days > 1 else 2,
    help="Lower this to bias toward songs you've been playing recently.",
)

library_only = st.sidebar.checkbox("Only tracks already in my library", value=False)

top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=25, value=10)

st.sidebar.markdown("---")
st.sidebar.caption("Built on your exported Spotify history 💾 · content-based cosine similarity engine")

# Build the eligible candidate pool from filters
candidate_mask = pd.Series(True, index=tracks_df.index)
if selected_artists:
    candidate_mask &= tracks_df["artist_name"].isin(selected_artists)
if selected_playlists:
    candidate_mask &= tracks_df["playlists"].isin(selected_playlists)
candidate_mask &= tracks_df["days_since_last_play"] <= recency_cutoff
if library_only:
    candidate_mask &= tracks_df["in_library"] == True  # noqa: E712

candidate_indices = tracks_df.index[candidate_mask].tolist()
candidate_df_filtered = tracks_df.loc[candidate_mask]

# ============================================================
# 6. SEED SONG SELECTION
# ============================================================

st.markdown("### 🌱 Pick your seed song(s)")
seed_pool = tracks_df["track_key"].tolist()
seed_songs = st.multiselect(
    "Search and select one or more songs you love",
    options=sorted(seed_pool),
    max_selections=8,
)
st.caption(f"🎵 {len(tracks_df):,} tracks available")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    go_btn = st.button("✨ Get recommendations")

# ============================================================
# 7. RESULTS
# ============================================================

if go_btn:
    if not seed_songs:
        st.warning("Pick at least one seed song first 🎵")
    elif len(candidate_indices) == 0:
        st.warning("No tracks match your current filters — try loosening them in the sidebar.")
    else:
        try:
            recs = engine.recommend(
                method="content",
                track_key=seed_songs,
                top_n=top_n,
                candidate_indices=candidate_indices,
            )
        except ValueError as e:
            st.error(str(e))
            recs = None

        if recs is not None and len(recs) > 0:
            # ---------------- KPIs ----------------
            k1, k2, k3, k4 = st.columns(4)
            kpis = [
                (k1, "Recommendations", f"{len(recs)}"),
                (k2, "Avg. similarity", f"{recs['similarity_score'].mean():.2f}"),
                (k3, "Avg. preference score", f"{recs['preference_score'].mean():.2f}"),
                (k4, "Top artist", recs.iloc[0]["artist_name"][:14]),
            ]
            for col, label, value in kpis:
                with col:
                    st.markdown(
                        f"""<div class="kpi-card">
                                <div class="kpi-label">{label}</div>
                                <div class="kpi-value">{value}</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------- Table ----------------
            st.markdown("### 🎯 Recommended tracks")
            display_df = recs[
                ["Rank", "track_name", "artist_name", "preference_score",
                 "similarity_score", "recommendation_reason"]
            ].rename(columns={

                "track_name": "Track",
                "artist_name": "Artist",
                "preference_score": "Preference Score",
                "similarity_score": "Similarity",
                "recommendation_reason": "Why this track?",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ---------------- Bar chart ----------------
            st.markdown("### 📊 Similarity by track")
            bar_fig = go.Figure(
                data=[
                    go.Bar(
                        x=recs["similarity_score"],
                        y=recs["track_name"] + " — " + recs["artist_name"],
                        orientation="h",
                        marker=dict(
                            color=recs["similarity_score"],
                            colorscale=[[0, LILAC], [0.5, PINK], [1, MINT]],
                        ),
                    )
                ]
            )
            bar_fig.update_layout(
                yaxis=dict(autorange="reversed"),
                plot_bgcolor=NAVY,
                paper_bgcolor=NAVY,
                font=dict(color="#EAEAF7", family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=max(320, 34 * len(recs)),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # ---------------- Similarity heatmap (seeds + recs) ----------------
            st.markdown("### 🔥 Similarity heatmap (seeds × recommendations)")
            seed_idx = [track_lookup[s] for s in seed_songs]
            rec_idx = [track_lookup[k] for k in recs["track_key"]]

            heat_matrix = similarity_matrix[np.ix_(seed_idx, rec_idx)]
            heat_fig = px.imshow(
                heat_matrix,
                labels=dict(x="Recommended track", y="Seed track", color="Similarity"),
                x=[tracks_df.loc[i, "track_name"][:20] for i in rec_idx],
                y=[tracks_df.loc[i, "track_name"][:20] for i in seed_idx],
                color_continuous_scale=[[0, NAVY_2], [0.5, LILAC], [1, CYAN]],
                aspect="auto",
            )
            heat_fig.update_layout(
                plot_bgcolor=NAVY,
                paper_bgcolor=NAVY,
                font=dict(color="#EAEAF7", family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(heat_fig, use_container_width=True)

            # ---------------- Explanations ----------------
            st.markdown("### 💬 Why these songs?")
            for _, row in recs.iterrows():
                st.markdown(
                    f"**{row['track_name']}** — *{row['artist_name']}*  "
                    f"<span class='reason-chip'>{row['recommendation_reason']}</span>",
                    unsafe_allow_html=True,
                )
else:
    st.info("Select seed songs on the left/above, adjust filters in the sidebar, then hit **Get recommendations**.")