# Outlet Intelligence Web App — Streamlit
# Browse predictions, model overview, geographic view, budget allocation, and outlet details with XAI

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.configs.config import config

log = logging.getLogger("streamlit.app")

DIST_PROVINCE = {
    "DIST_W_01": "Western", "DIST_W_02": "Western", "DIST_W_03": "Western",
    "DIST_C_01": "Central", "DIST_C_02": "Central", "DIST_C_03": "Central",
    "DIST_NW_01": "North-Western", "DIST_NW_02": "North-Western",
    "DIST_S_01": "Southern", "DIST_S_02": "Southern",
}

CONFIDENCE_COLORS = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}
MAP_COLORS = {"high": (44, 160, 44, 180), "medium": (255, 127, 14, 180), "low": (214, 39, 40, 180)}
FUNDED_COLOR = (31, 119, 180, 200)
UNFUNDED_COLOR = (160, 160, 160, 120)
ACCENT_COLOR = "#1f77b4"

PAGE_TITLE = "Outlet Intelligence Dashboard"

BUDGET_LKR = 5_000_000
COST_PER_LITER = 50

WESTERN_DISTRIBUTORS = ["DIST_W_01", "DIST_W_02", "DIST_W_03"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_predictions():
    path = Path("notebooks/predictions_jan2026.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_fact_table():
    path = config.GOLD_PATH / "fact_table" / "data.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_explanations():
    path = config.REPORTS_DIR / "teamname_outlet_explanations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_budget():
    path = config.REPORTS_DIR / "teamname_budget_allocations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Data merging
# ---------------------------------------------------------------------------

def merge_data():
    preds = load_predictions()
    fact = load_fact_table()
    budget = load_budget()

    if preds.empty:
        return pd.DataFrame()

    outlet_info = fact.groupby("Outlet_ID").first().reset_index()
    drop_cols = [c for c in outlet_info.columns if c.startswith("_")]
    outlet_info = outlet_info.drop(columns=drop_cols)

    merged = preds.merge(outlet_info, on="Outlet_ID", how="left", suffixes=("", "_drop"))
    merged = merged.loc[:, ~merged.columns.str.endswith("_drop")]

    merged["Province"] = merged["Distributor_ID"].map(DIST_PROVINCE).fillna("Unknown")

    merged["funded"] = False
    if not budget.empty:
        budget_cols = budget[["Outlet_ID", "Trade_Spend_LKR", "historical_mean", "incremental_volume"]]
        merged = merged.merge(budget_cols, on="Outlet_ID", how="left")
        merged["funded"] = merged["Trade_Spend_LKR"].notna() & (merged["Trade_Spend_LKR"] > 0)
        merged["potential"] = (
            merged["predicted_volume"] - merged["historical_mean"]
        ).clip(lower=0)

    merged["map_radius"] = merged["predicted_volume"] * 60
    clr = merged["confidence_label"]
    merged["map_color"] = [list(MAP_COLORS.get(c, (128, 128, 128, 128))) for c in clr]
    merged["map_color_funded"] = [
        list(FUNDED_COLOR) if f else list(UNFUNDED_COLOR) for f in merged["funded"]
    ]

    return merged


from src.xai.outlet_explainer import OutletExplainer

@st.cache_data
def get_dynamic_explanation(outlet_id: str, row_dict: dict) -> str:
    explainer = OutletExplainer()
    res = explainer.explain_outlet(pd.Series(row_dict))
    return res.get("narrative", "Could not generate explanation.")


# ---------------------------------------------------------------------------
# Chart builders — model overview
# ---------------------------------------------------------------------------

def chart_volume_distribution(df):
    fig = px.histogram(
        df, x="predicted_volume", nbins=60,
        color="confidence_label",
        color_discrete_map=CONFIDENCE_COLORS,
        title="Predicted Volume Distribution",
        labels={"predicted_volume": "Volume (Liters)", "count": "Number of Outlets"},
        barmode="overlay",
        opacity=0.7,
        marginal="box",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    return fig


def chart_competition_vs_volume(df):
    fig = px.scatter(
        df.sample(min(5000, len(df))),
        x="competition_density", y="predicted_volume",
        color="confidence_label",
        color_discrete_map=CONFIDENCE_COLORS,
        title="Competition Density vs Predicted Volume",
        labels={
            "competition_density": "Competition Density (outlets within 5km)",
            "predicted_volume": "Predicted Volume (Liters)",
        },
        hover_data=["Outlet_ID", "Outlet_Type", "Distributor_ID"],
        opacity=0.6,
        trendline="lowess" if len(df) >= 50 else None,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    return fig


def chart_volume_by_outlet_type(df):
    grouped = df.groupby("Outlet_Type", observed=True)["predicted_volume"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["Outlet_Type", "Mean Volume", "Outlet Count"]
    fig = px.bar(
        grouped, x="Outlet_Type", y="Mean Volume",
        color="Mean Volume",
        color_continuous_scale="Blues",
        title="Mean Predicted Volume by Outlet Type",
        text_auto=".0f",
        hover_data={"Outlet Count": True},
        labels={"Mean Volume": "Mean Volume (Liters)"},
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_traces(
        textfont_size=12,
        textposition="outside",
        hovertemplate="%{x}<br>Mean: %{y:.1f} L<br>Outlets: %{customdata[0]:,}",
    )
    return fig


def chart_top_outlets(df, n=20):
    top = df.nlargest(n, "predicted_volume")
    fig = px.bar(
        top, y="Outlet_ID", x="predicted_volume",
        color="confidence_label",
        color_discrete_map=CONFIDENCE_COLORS,
        title=f"Top {n} Outlets by Predicted Volume",
        labels={"predicted_volume": "Volume (Liters)", "Outlet_ID": ""},
        orientation="h",
        hover_data=["Distributor_ID", "Outlet_Type"],
        text="predicted_volume",
    )
    fig.update_traces(texttemplate="%{text:.0f} L", textposition="outside")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), font=dict(size=12))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    return fig


# ---------------------------------------------------------------------------
# Chart builders — budget allocation
# ---------------------------------------------------------------------------

def chart_top_funded(df, n=20):
    top = df.nlargest(n, "Trade_Spend_LKR")
    fig = px.bar(
        top, y="Outlet_ID", x="Trade_Spend_LKR",
        color="incremental_volume",
        color_continuous_scale="Blues",
        title=f"Top {n} Funded Outlets by Trade Spend",
        labels={"Trade_Spend_LKR": "Trade Spend (LKR)", "Outlet_ID": ""},
        orientation="h",
        hover_data=["Outlet_Type", "Distributor_ID", "incremental_volume"],
        text="Trade_Spend_LKR",
    )
    fig.update_traces(texttemplate="LKR %{text:,.0f}", textposition="outside")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), font=dict(size=12))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    return fig


def chart_spend_by_type(df):
    grouped = df.groupby("Outlet_Type", observed=True)["Trade_Spend_LKR"].sum().reset_index()
    fig = px.bar(
        grouped, x="Outlet_Type", y="Trade_Spend_LKR",
        color="Trade_Spend_LKR",
        color_continuous_scale="Blues",
        title="Total Trade Spend by Outlet Type",
        text_auto=".0s",
        labels={"Trade_Spend_LKR": "Total Spend (LKR)"},
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>LKR %{y:,.0f}")
    return fig


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_map_layer(df, color_by_funded=False):
    view_state = pdk.ViewState(latitude=7.2, longitude=80.6, zoom=7, pitch=0)

    if color_by_funded:
        color_field = "map_color_funded"
        tooltip_html = (
            "<b>{Outlet_ID}</b><br>"
            "Volume: <b>{predicted_volume}</b> L<br>"
            "Funded: <b>{funded}</b><br>"
            "Type: {Outlet_Type}<br>"
            "Distributor: {Distributor_ID}"
        )
    else:
        color_field = "map_color"
        tooltip_html = (
            "<b>{Outlet_ID}</b><br>"
            "Volume: <b>{predicted_volume}</b> L<br>"
            "Confidence: {confidence_label}<br>"
            "Type: {Outlet_Type}<br>"
            "Distributor: {Distributor_ID}<br>"
            "Competition: {competition_density} outlets"
        )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["Longitude", "Latitude"],
        get_radius="map_radius",
        get_fill_color=color_field,
        get_line_color=[255, 255, 255, 60],
        get_line_width=1,
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=2,
        radius_max_pixels=80,
    )

    tooltip = {
        "html": tooltip_html,
        "style": {"backgroundColor": "#1a1a2e", "color": "white"},
    }

    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)


# ---------------------------------------------------------------------------
# Funding rationale
# ---------------------------------------------------------------------------

def funding_rationale(row):
    funded = row.get("funded", False)
    trade_spend = row.get("Trade_Spend_LKR", 0)
    incr_vol = row.get("incremental_volume", 0)
    hist_mean = row.get("historical_mean", 0)
    pred_vol = row.get("predicted_volume", 0)
    oid = row.get("Outlet_ID", "")

    if not funded:
        if hist_mean > 0 and pred_vol > 0:
            gap = pred_vol - hist_mean
            if gap <= 0:
                return (
                    f"**{oid} is not funded.** "
                    f"Its predicted volume ({pred_vol:.0f} L) does not exceed "
                    f"its historical mean ({hist_mean:.0f} L). The model found "
                    f"no incremental potential to invest in."
                )
            return (
                f"**{oid} is not funded.** "
                f"Although it has {gap:.0f} L of incremental potential, "
                f"the optimizer prioritized other outlets in the Western Province "
                f"that yield higher volume per LKR spent under the LKR 5M budget."
            )
        return f"**{oid} is not funded.** No historical data available for comparison."

    gap = pred_vol - hist_mean
    return (
        f"**{oid} is funded — LKR {trade_spend:,.0f} allocated.**\n\n"
        f"This outlet has a predicted monthly demand of **{pred_vol:,.0f} L** "
        f"versus a historical mean of **{hist_mean:,.0f} L**, "
        f"leaving **{gap:,.0f} L of incremental potential**. "
        f"The PuLP optimizer allocated LKR {trade_spend:,.0f} to capture "
        f"**{incr_vol:,.0f} L** of that gap, at the standard rate of "
        f"**LKR {COST_PER_LITER:,}/L**.\n\n"
        f"**Future goal**: This investment targets {incr_vol:,.0f} L of additional "
        f"monthly volume. Across all funded Western outlets, "
        f"the total target is **100,000 L/month** at a total trade spend of "
        f"**LKR {BUDGET_LKR:,}**, with every liter costing **LKR {COST_PER_LITER:,}**."
    )


# ---------------------------------------------------------------------------
# Chat assistant
# ---------------------------------------------------------------------------

DATASET_CONTEXT = None

def build_dataset_context(data: pd.DataFrame) -> str:
    n_funded = data["funded"].sum()
    total_spend = data["Trade_Spend_LKR"].sum() if "Trade_Spend_LKR" in data.columns else 0
    total_vol = data["predicted_volume"].sum()
    n_high = (data["confidence_label"] == "high").sum()
    n_medium = (data["confidence_label"] == "medium").sum()
    n_low = (data["confidence_label"] == "low").sum()
    
    funded_details = ""
    if n_funded > 0:
        funded_df = data[data["funded"]].sort_values("Trade_Spend_LKR", ascending=False)
        funded_details = "\n\nFUNDED OUTLETS (All 286 outlets that received budget):\n"
        # To avoid massive token usage, format compactly
        for _, r in funded_df.iterrows():
            funded_details += f"{r['Outlet_ID']}: LKR {r['Trade_Spend_LKR']:.0f} (Vol: {r['predicted_volume']:.0f}L), "
        funded_details = funded_details.strip(", ")

    return f"""
You are a business analyst assistant for a retail outlet intelligence system. You have access to the following dataset:

DATASET OVERVIEW:
- Total outlets: {len(data):,}
- Total predicted volume: {total_vol:,.0f} L/month
- Mean volume per outlet: {data['predicted_volume'].mean():.1f} L
- Volume range: {data['predicted_volume'].min():.0f} - {data['predicted_volume'].max():,.0f} L
- Confidence distribution: high={n_high:,}, medium={n_medium:,}, low={n_low:,}
- Number of distributors: {data['Distributor_ID'].nunique()}
- Provinces: {', '.join(sorted(data['Province'].unique()))}
- Outlet types: {', '.join(sorted(data['Outlet_Type'].dropna().unique()))}
- Funded outlets: {n_funded:,}
- Total trade spend: LKR {total_spend:,.0f}{funded_details}

You can answer questions about predicted volumes, confidence levels, competition density, outlet characteristics, budget allocation, and funding decisions. Keep answers concise and data-driven. Do not make up numbers not present in the data.
"""

def _load_chat_api_key() -> Optional[str]:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("GROQ_API_KEY")
        if key:
            return key
    except ImportError:
        pass
    return None


from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def ask_llm(user_message: str, dataset_context: str, history: list[dict]) -> str:
    key = _load_chat_api_key()
    if not key:
        return "The Groq API key is not configured. Set GROQ_API_KEY in your environment or .env file to enable the chat assistant."

    try:
        from groq import Groq
        
        messages = [
            {"role": "system", "content": dataset_context}
        ]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        client = Groq(api_key=key)

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True
        )
        def _do_call():
            return client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
            )
            
        resp = _do_call()
        return resp.choices[0].message.content.strip() or "I could not generate a response."
    except Exception as exc:
        err = str(exc)
        if "429" in err or "quota" in err.lower():
            return "API quota exceeded. Please try again later or check your billing plan."
        return f"An error occurred: {err}"


def init_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "dataset_context" not in st.session_state:
        st.session_state.dataset_context = ""


def render_chat_tab(data: pd.DataFrame):
    init_chat_state()

    if not st.session_state.dataset_context:
        st.session_state.dataset_context = build_dataset_context(data)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about outlets, predictions, or funding..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                import re
                nums = re.findall(r"\d+", prompt)
                dynamic_context = st.session_state.dataset_context
                
                if nums:
                    matched_rows = []
                    for num in set(nums):
                        pad1 = str(num).zfill(4)
                        pad2 = str(num).zfill(5)
                        mask = data["Outlet_ID"].astype(str).str.contains(f"({num}|{pad1}|{pad2})$", regex=True, na=False)
                        if mask.any():
                            matched_rows.append(data[mask])
                    
                    if matched_rows:
                        matched_df = pd.concat(matched_rows).drop_duplicates(subset=["Outlet_ID"]).head(5)
                        dynamic_context += "\n\nSpecific Outlet Context for this query:\n"
                        for _, row in matched_df.iterrows():
                            dynamic_context += f"- {row['Outlet_ID']}: Vol={row.get('predicted_volume', 0):.1f}L, Funded={row.get('funded', False)}, Type={row.get('Outlet_Type', 'N/A')}\n"
                            dynamic_context += f"  Rationale: {funding_rationale(row)}\n"

                response = ask_llm(
                    prompt,
                    dynamic_context,
                    st.session_state.chat_messages[:-1],
                )
            st.markdown(response)
        st.session_state.chat_messages.append({"role": "model", "content": response})
        st.rerun()


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: #ffffff; }}

    /* ── Tab container: pill-style button bar ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: #f0f2f6;
        padding: 6px 8px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #e0e0e0;
    }}

    /* ── Individual tab (inactive) ── */
    .stTabs [data-baseweb="tab"] {{
        height: 42px;
        padding: 0 20px;
        font-size: 14px;
        font-weight: 500;
        color: #555;
        background-color: transparent;
        border-radius: 8px;
        border: none;
        transition: all 0.15s ease;
    }}

    /* ── Hover ── */
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: #e0e0e0;
        color: #222;
    }}

    /* ── Active tab ── */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {ACCENT_COLOR};
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }}

    /* ── Hide the default underline ── */
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 26px !important;
        font-weight: 700 !important;
        color: {ACCENT_COLOR} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 13px !important;
        font-weight: 400 !important;
        color: #888 !important;
    }}
    h1, h2, h3, h4 {{
        color: #333 !important;
    }}
    .st-emotion-cache-1r4qj8v {{ border: none; }}
</style>
"""


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title(PAGE_TITLE)
st.markdown("**Latent Demand Estimation for January 2026** — Data Storm v7.0")

data = merge_data()

if data.empty:
    st.warning("No prediction data found. Run the pipeline first.")
    st.stop()

# ─── Sidebar Filters ─────────────────────────────────────────────

st.sidebar.header("Filters")

all_provinces = ["All"] + sorted(data["Province"].unique())
province = st.sidebar.selectbox("Province", all_provinces)

filtered = data.copy()
if province != "All":
    filtered = filtered[filtered["Province"] == province]

all_dists = ["All"] + sorted(filtered["Distributor_ID"].unique())
distributor = st.sidebar.selectbox("Distributor", all_dists)
if distributor != "All":
    filtered = filtered[filtered["Distributor_ID"] == distributor]

confidence_opts = ["All"] + ["high", "medium", "low"]
conf_label = st.sidebar.selectbox("Confidence", confidence_opts)
if conf_label != "All":
    filtered = filtered[filtered["confidence_label"] == conf_label]

all_types = sorted(filtered["Outlet_Type"].dropna().unique())
selected_types = st.sidebar.multiselect("Outlet Type", all_types, default=all_types)
if selected_types:
    filtered = filtered[filtered["Outlet_Type"].isin(selected_types)]

all_sizes = sorted(filtered["Outlet_Size"].dropna().unique())
selected_sizes = st.sidebar.multiselect("Outlet Size", all_sizes, default=all_sizes)
if selected_sizes:
    filtered = filtered[filtered["Outlet_Size"].isin(selected_sizes)]

min_vol, max_vol = (
    float(filtered["predicted_volume"].min()),
    float(filtered["predicted_volume"].max()),
)
vol_range = st.sidebar.slider(
    "Predicted Volume Range",
    min_value=min_vol, max_value=max_vol,
    value=(min_vol, max_vol),
)
filtered = filtered[
    (filtered["predicted_volume"] >= vol_range[0])
    & (filtered["predicted_volume"] <= vol_range[1])
]

funded_only = st.sidebar.checkbox("Show funded outlets only", value=False)
if funded_only:
    filtered = filtered[filtered["funded"]]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(filtered):,} of {len(data):,} outlets")

# Budget utilization in sidebar
if "Trade_Spend_LKR" in data.columns:
    st.sidebar.markdown("---")
    total_spend_all = data["Trade_Spend_LKR"].sum()
    utilization = total_spend_all / BUDGET_LKR
    st.sidebar.markdown("### Budget Utilization")
    st.sidebar.progress(min(utilization, 1.0))
    st.sidebar.caption(
        f"LKR {total_spend_all:,.0f} / LKR {BUDGET_LKR:,} "
        f"({utilization:.1%}) | {(data['funded']).sum():,} outlets funded"
    )

# ─── Tabs ────────────────────────────────────────────────────────

tab_browse, tab_model, tab_map, tab_funding, tab_detail, tab_chat = st.tabs(
    ["Browse", "Model Overview", "Geographic View", "Budget Allocation", "Outlet Details", "Assistant"]
)

# ── Tab 1: Browse ────────────────────────────────────────────────

with tab_browse:
    display_cols = [
        "Outlet_ID", "funded", "predicted_volume", "confidence_label",
        "Province", "Distributor_ID", "Outlet_Type", "Outlet_Size",
        "Cooler_Count", "competition_density", "Trade_Spend_LKR",
        "incremental_volume",
    ]
    available = [c for c in display_cols if c in filtered.columns]

    df_display = filtered[available].sort_values("predicted_volume", ascending=False)
    df_display["funded_status"] = df_display["funded"].apply(lambda f: "Yes" if f else "No")

    st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
        column_config={
            "predicted_volume": st.column_config.NumberColumn("Volume (L)", format="%.1f"),
            "Trade_Spend_LKR": st.column_config.NumberColumn("Spend (LKR)", format="LKR %.0f"),
            "incremental_volume": st.column_config.NumberColumn("Incr. Vol.", format="%.1f"),
            "competition_density": st.column_config.NumberColumn("Competition", format="%.0f"),
            "Cooler_Count": st.column_config.NumberColumn("Coolers", format="%d"),
            "funded": st.column_config.TextColumn("Funded"),
            "funded_status": st.column_config.TextColumn("Funded"),
        },
        column_order=["Outlet_ID", "funded_status", "predicted_volume", "confidence_label",
                       "Province", "Distributor_ID", "Outlet_Type", "Outlet_Size",
                       "Cooler_Count", "competition_density", "Trade_Spend_LKR",
                       "incremental_volume"],
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predicted Volume", f"{filtered['predicted_volume'].sum():,.0f} L")
    with col2:
        st.metric("Mean per Outlet", f"{filtered['predicted_volume'].mean():.1f} L")
    with col3:
        st.metric("Median per Outlet", f"{filtered['predicted_volume'].median():.1f} L")
    with col4:
        high_pct = (filtered["confidence_label"] == "high").mean() * 100
        st.metric("High Confidence", f"{high_pct:.0f}%")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Total Outlets", f"{len(filtered):,}")
    with col6:
        if "Outlet_Type" in filtered.columns:
            n_types = filtered["Outlet_Type"].nunique()
            st.metric("Outlet Types", n_types)
    with col7:
        if "competition_density" in filtered.columns:
            avg_comp = filtered["competition_density"].mean()
            st.metric("Mean Competition Density", f"{avg_comp:.0f}")
    with col8:
        if "Trade_Spend_LKR" in filtered.columns:
            total_spend = filtered["Trade_Spend_LKR"].sum()
            st.metric("Total Spend", f"LKR {total_spend:,.0f}")

# ── Tab 2: Model Overview ────────────────────────────────────────

with tab_model:
    st.subheader("Prediction Summary")
    n_high = (filtered["confidence_label"] == "high").sum()
    n_medium = (filtered["confidence_label"] == "medium").sum()
    n_low = (filtered["confidence_label"] == "low").sum()
    total = len(filtered)

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("High Confidence", f"{n_high:,} ({n_high/total*100:.0f}%)")
    col_m2.metric("Medium Confidence", f"{n_medium:,} ({n_medium/total*100:.0f}%)")
    col_m3.metric("Low Confidence", f"{n_low:,} ({n_low/total*100:.0f}%)")
    col_m4.metric("Mean Censoring Score", f"{filtered['censoring_score'].mean():.3f}")
    col_m5.metric("Volume Range", f"{filtered['predicted_volume'].min():.0f} - {filtered['predicted_volume'].max():,.0f} L")

    with st.expander("Model Explanation (XAI)"):
        st.markdown(
            """
            **Model Architecture**
            - A neural network (PyTorch) trained on historical sales data from Data Storm v7.0
            - Input features include: outlet type, size, cooler count, competition density,
              POI proximity scores (school, hospital, bus stop, tourist), seasonality index,
              and holiday calendar effects

            **Prediction Output**
            - Each outlet receives a predicted monthly volume in liters
            - A confidence label (high/medium/low) is assigned based on the prediction interval width
            - The censoring score (0 to 1) indicates potential supply constraints that may
              cause true demand to be higher than recorded sales

            **Feature Engineering**
            - Competition density computed via 5km buffer-based spatial joins
            - Geographic coordinates corrected for swapped lat/lon entries
            - Missing outlet attributes imputed during the silver-layer cleaning stage
            """
        )

    st.subheader("Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_volume_distribution(filtered), width="stretch")
    with c2:
        st.plotly_chart(chart_competition_vs_volume(filtered), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_volume_by_outlet_type(filtered), width="stretch")
    with c4:
        st.plotly_chart(chart_top_outlets(filtered), width="stretch")

# ── Tab 3: Geographic View ───────────────────────────────────────

with tab_map:
    color_by_funded = st.checkbox("Color by funded status", value=False)

    map_data = filtered.dropna(subset=["Latitude", "Longitude"])
    if map_data.empty:
        st.info("No geo-coordinates available for the current filter selection.")
    else:
        st.pydeck_chart(build_map_layer(map_data, color_by_funded=color_by_funded))
        if color_by_funded:
            st.caption(
                f"Showing {len(map_data):,} outlets. "
                "Bubble size proportional to predicted volume. "
                "Color: blue = funded, gray = unfunded."
            )
        else:
            st.caption(
                f"Showing {len(map_data):,} outlets. "
                "Bubble size proportional to predicted volume. "
                "Color: green = high confidence, orange = medium, red = low."
            )

# ── Tab 4: Budget Allocation ─────────────────────────────────────

with tab_funding:
    funded_df = data[data["funded"]].copy()

    if funded_df.empty:
        st.info("No funded outlets found. Run the budget optimizer pipeline first.")
    else:
        total_spend = funded_df["Trade_Spend_LKR"].sum()
        total_incr = funded_df["incremental_volume"].sum()

        st.subheader("Investment Overview")

        col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
        col_k1.metric("Total Budget", f"LKR {BUDGET_LKR:,}")
        col_k2.metric("Total Spend", f"LKR {total_spend:,.0f}")
        col_k3.metric("Utilization", f"{total_spend / BUDGET_LKR:.1%}")
        col_k4.metric("Outlets Funded", f"{len(funded_df):,}")
        col_k5.metric("Incremental Target", f"{total_incr:,.0f} L")

        col_k6, col_k7, col_k8, col_k9, col_k10 = st.columns(5)
        col_k6.metric("Avg Spend per Outlet", f"LKR {total_spend/len(funded_df):,.0f}")
        col_k7.metric("Avg Incremental Volume", f"{total_incr/len(funded_df):,.0f} L")
        col_k8.metric("Cost per Liter", f"LKR {COST_PER_LITER:,}")
        col_k9.metric("Avg Predicted Volume", f"{funded_df['predicted_volume'].mean():,.0f} L")
        col_k10.metric("Avg Historical Mean", f"{funded_df['historical_mean'].mean():,.1f} L")

        with st.expander("Budget Optimization Method"):
            st.markdown(
                f"""
                The budget optimizer uses **PuLP (Linear Programming)** to allocate a fixed
                **LKR {BUDGET_LKR:,}** promotional budget across **{len(funded_df):,} Western Province outlets**
                to maximize total incremental volume.

                **Rules:**
                - Only Western Province outlets (DIST_W_01, DIST_W_02, DIST_W_03) are eligible
                - Only outlets where **predicted volume exceeds historical mean** can receive funding
                - Each incremental liter costs **LKR {COST_PER_LITER:,}** in trade spend
                - The solver maximizes total incremental volume across all eligible outlets

                **Selection Criteria:** The LP solver allocates budget to outlets with the
                highest untapped potential, ensuring every LKR spent delivers maximum return.
                """
            )

        st.subheader("Funded Outlets Detail")
        funded_display = funded_df[
            ["Outlet_ID", "Trade_Spend_LKR", "predicted_volume", "historical_mean",
             "incremental_volume", "potential", "Outlet_Type", "Outlet_Size",
             "Distributor_ID"]
        ].sort_values("Trade_Spend_LKR", ascending=False)
        funded_display["Efficiency"] = funded_display["incremental_volume"] / (funded_display["Trade_Spend_LKR"] / COST_PER_LITER)

        st.dataframe(
            funded_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Trade_Spend_LKR": st.column_config.NumberColumn("Trade Spend (LKR)", format="LKR %.0f"),
                "predicted_volume": st.column_config.NumberColumn("Predicted Vol.", format="%.1f"),
                "historical_mean": st.column_config.NumberColumn("Hist. Mean", format="%.1f"),
                "incremental_volume": st.column_config.NumberColumn("Incremental Vol.", format="%.1f"),
                "potential": st.column_config.NumberColumn("Potential (L)", format="%.1f"),
                "Efficiency": st.column_config.NumberColumn("Efficiency", format="%.0f%%"),
            },
        )

        st.subheader("Budget Allocation Charts")
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            st.plotly_chart(chart_top_funded(funded_df), width="stretch")
        with c_f2:
            st.plotly_chart(chart_spend_by_type(funded_df), width="stretch")

# ── Tab 5: Outlet Details ────────────────────────────────────────

with tab_detail:
    st.subheader("Outlet Details")

    outlet_ids = sorted(filtered["Outlet_ID"].unique())
    selected = st.selectbox("Select Outlet", outlet_ids)

    if selected:
        row = data[data["Outlet_ID"] == selected].iloc[0]

        is_funded = row.get("funded", False)
        if is_funded:
            st.success(funding_rationale(row))
        else:
            st.info(funding_rationale(row))

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Volume", f"{row['predicted_volume']:,.1f} L")
        conf = row["confidence_label"]
        conf_color = CONFIDENCE_COLORS.get(conf, "#666")
        c2.markdown(
            f"<div style='text-align:center'>"
            f"<span style='font-size:13px;color:#888'>Confidence</span><br>"
            f"<span style='font-size:26px;font-weight:700;color:{conf_color}'>{conf.title()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        c3.metric("Censoring Score", f"{row.get('censoring_score', 'N/A')}")

        st.write("**Outlet Profile**")
        profile_cols = [
            ("Distributor ID", "Distributor_ID"),
            ("Province", "Province"),
            ("Outlet Type", "Outlet_Type"),
            ("Outlet Size", "Outlet_Size"),
            ("Cooler Count", "Cooler_Count"),
        ]
        if "competition_density" in row and not pd.isna(row.get("competition_density")):
            profile_cols.append(("Competition Density (5km)", "competition_density"))

        profile_data = {label: row.get(col, "N/A") for label, col in profile_cols}
        st.json(profile_data)

        st.write("**Predicted vs Historical**")
        if "incremental_volume" in row and "historical_mean" in row:
            hist_mean = row.get("historical_mean", 0)
            incr = row.get("incremental_volume", 0)
            c_hist, c_incr = st.columns(2)
            c_hist.metric("Historical Mean", f"{hist_mean:,.1f} L")
            c_incr.metric("Incremental Potential", f"{incr:,.1f} L")

        if is_funded:
            st.write("**Funding Details**")
            c_spend, c_incr2, c_pot = st.columns(3)
            c_spend.metric("Trade Spend", f"LKR {row.get('Trade_Spend_LKR', 0):,.0f}")
            c_incr2.metric("Incremental Target", f"{row.get('incremental_volume', 0):,.1f} L")
            potential = row.get("potential", 0)
            c_pot.metric("Untapped Potential", f"{potential:,.1f} L")

        st.write("**Model Explanation (XAI)**")
        if st.button("Generate Explanation"):
            with st.spinner("Generating dynamic explanation..."):
                narrative = get_dynamic_explanation(selected, row.to_dict())
                st.info(narrative)

# ── Tab 6: Assistant ────────────────────────────────────────────

with tab_chat:
    st.subheader("Ask about the data")
    st.caption("Ask questions about outlets, predictions, competition, funding, or any trends in the dataset.")
    render_chat_tab(data)
