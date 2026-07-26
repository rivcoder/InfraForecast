"""
InfraForecast - Streamlit Dashboard
=====================================
Premium dark-mode 4-page dashboard visualizing MoSPI Flash Report data.

Pages:
  1. Overview        – KPI cards + snapshot summary
  2. Sector Trends   – COR% and TOR trend lines across 5 report snapshots
  3. Chronic Offenders – Projects in >= 3 reports with escalating overrun
  4. Forecast Sandbox  – Predict delay + COR% for a new project

Run:  streamlit run src/app.py   (from InfraForecast root)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src import database as db
from src import analysis
from src.model import predict, get_feature_importance, get_known_sectors, get_known_states

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfraForecast · Indian Infrastructure Overrun Analytics",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS – dark glassmorphic theme ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
    color: #c9d1d9;
}

/* Main background */
.main { background: #0d1117; }
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stHeader"] { background: transparent; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.9) 0%, rgba(33,38,45,0.9) 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.kpi-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
.kpi-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #f0f6fc; line-height: 1.1; }
.kpi-unit  { font-size: 0.8rem; color: #58a6ff; margin-top: 4px; }
.kpi-red   { border-left: 3px solid #f85149; }
.kpi-amber { border-left: 3px solid #d29922; }
.kpi-blue  { border-left: 3px solid #58a6ff; }
.kpi-green { border-left: 3px solid #3fb950; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0f6fc;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Alert callout */
.insight-box {
    background: rgba(56, 139, 253, 0.1);
    border: 1px solid rgba(56, 139, 253, 0.3);
    border-radius: 8px;
    padding: 14px 18px;
    color: #cdd9e5;
    font-size: 0.88rem;
    margin: 12px 0;
}
.insight-box.red-tint {
    background: rgba(248,81,73,0.1);
    border-color: rgba(248,81,73,0.3);
}

/* Headings */
h1 { color: #f0f6fc !important; }
h2, h3 { color: #c9d1d9 !important; }
p, li { color: #8b949e; }

/* DataFrame styling */
[data-testid="stDataFrame"] { border-radius: 8px; }

/* Streamlit select/slider */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #c9d1d9 !important;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#8b949e",
    "axes.titlecolor": "#f0f6fc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "text.color": "#c9d1d9",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

SECTOR_COLORS = [
    "#58a6ff", "#3fb950", "#f78166", "#d29922",
    "#bc8cff", "#79c0ff", "#56d364", "#ffa657",
    "#ff7b72", "#6e7681", "#a5d6ff", "#ffdf5d",
]


# ── Data loaders with caching ────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_stats():
    return analysis.summary_stats()

@st.cache_data(ttl=300)
def load_sector_trends():
    return analysis.sector_trends()

@st.cache_data(ttl=300)
def load_state_summary():
    return analysis.state_overrun_summary()

@st.cache_data(ttl=300)
def load_chronic():
    return analysis.chronic_offenders(min_snapshots=2)  # 2+ snapshots for better coverage

@st.cache_data(ttl=300)
def load_db_stats():
    return db.get_db_stats()

@st.cache_data(ttl=300)
def load_delayed():
    return db.load_table("delayed_projects")

def _check_db():
    stats = load_db_stats()
    return any(v > 0 for v in stats.values())


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
      <div style="font-size:1.5rem; font-weight:700; color:#f0f6fc;">🏗️ InfraForecast</div>
      <div style="font-size:0.78rem; color:#8b949e; margin-top:4px;">
        Central Sector Projects · MoSPI Data<br>Apr – Sep 2024
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Overview", "📈 Sector Trends", "🚨 Chronic Offenders", "🔮 Forecast Sandbox"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#6e7681; line-height:1.6;">
      Data source:<br>
      <a href="https://mospi.gov.in" style="color:#58a6ff;">MoSPI Flash Reports</a><br>
      Rs. 150 crore+ projects<br>
      Real government data only.
    </div>
    """, unsafe_allow_html=True)

    if not _check_db():
        st.warning("⚠️ No data in DB. Run: `python src/pipeline.py` first.")


# ── Helper: kpi_card ─────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, unit: str = "", color_class: str = "kpi-blue"):
    return f"""<div class="kpi-card {color_class}">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-unit">{unit}</div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.markdown("<h1>Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e; font-size:0.9rem;'>"
        "Real-data summary of Central Sector Infrastructure Projects "
        "(₹150 crore+) from 5 MoSPI Flash Reports — Apr, May, Jul, Aug, Sep 2024.</p>",
        unsafe_allow_html=True,
    )

    stats = load_stats()

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "Delayed Project Records",
            f"{stats.get('total_delayed_projects', 0):,}",
            "across 5 report snapshots",
            "kpi-red",
        ), unsafe_allow_html=True)
    with c2:
        cor_cr = stats.get("total_anticipated_cost_cr", 0) - stats.get("total_original_cost_cr", 0)
        st.markdown(kpi_card(
            "Total Cost Overrun (₹ crore)",
            f"₹{cor_cr:,.0f}",
            "anticipated − original across all records",
            "kpi-amber",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(
            "Mean Delay",
            f"{stats.get('mean_delay_months', 0):.1f}",
            "months per delayed project",
            "kpi-blue",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(
            "Worst Sector (COR%)",
            stats.get("worst_sector", "N/A"),
            f"{stats.get('worst_sector_cor', 0):.1f}% mean cost overrun",
            "kpi-red",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DB health ─────────────────────────────────────────────────────────────
    db_stats = load_db_stats()
    st.markdown('<div class="section-header">📁 Database Contents</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(kpi_card("Delayed Projects (T16)", f"{db_stats.get('delayed_projects', 0):,}", "rows in DB", "kpi-blue"), unsafe_allow_html=True)
    with sc2:
        st.markdown(kpi_card("Sector-Focused (Ann-1)", f"{db_stats.get('sector_focused', 0):,}", "rows in DB", "kpi-green"), unsafe_allow_html=True)
    with sc3:
        st.markdown(kpi_card("State Summary (Ann-2)", f"{db_stats.get('state_summary', 0):,}", "rows in DB", "kpi-amber"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── State-level cost overrun bar chart ────────────────────────────────────
    state_df = load_state_summary()
    if not state_df.empty:
        st.markdown('<div class="section-header">🗺️ Top 15 States by Mean Cost Overrun %</div>', unsafe_allow_html=True)

        top15 = state_df.head(15).copy()
        top15["mean_cor_pct"] = top15["mean_cor_pct"].fillna(0)

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#f85149" if v > 30 else "#d29922" if v > 15 else "#58a6ff"
                  for v in top15["mean_cor_pct"]]
        bars = ax.barh(top15["state"][::-1], top15["mean_cor_pct"][::-1], color=colors[::-1],
                       height=0.65, edgecolor="none")
        for bar, val in zip(bars, top15["mean_cor_pct"][::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left", fontsize=8.5, color="#c9d1d9")
        ax.set_xlabel("Mean Cost Overrun % (anticipated vs. original)", labelpad=8)
        ax.set_title("State-Wise Infrastructure Cost Overrun", pad=12, fontsize=12, fontweight="600")
        ax.axvline(top15["mean_cor_pct"].mean(), color="#58a6ff", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(top15["mean_cor_pct"].mean() + 0.3, 0, f"  avg {top15['mean_cor_pct'].mean():.1f}%",
                color="#58a6ff", fontsize=8, va="bottom")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Scatter: original cost vs delay ──────────────────────────────────────
    dp = load_delayed()
    if not dp.empty:
        dp["original_cost"] = pd.to_numeric(dp["original_cost"], errors="coerce")
        dp["delay_months"] = pd.to_numeric(dp["delay_months"], errors="coerce")
        dp_plot = dp.dropna(subset=["original_cost", "delay_months"])
        dp_plot = dp_plot[(dp_plot["original_cost"] > 0) & (dp_plot["delay_months"] > 0)]

        if len(dp_plot) > 5:
            st.markdown('<div class="section-header">💠 Delay Severity Scatter: Budget vs. Delay (months)</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(11, 5))
            sc = ax2.scatter(
                dp_plot["original_cost"],
                dp_plot["delay_months"],
                c=dp_plot["delay_months"],
                cmap="YlOrRd",
                alpha=0.55,
                s=40,
                edgecolors="none",
            )
            cbar = fig2.colorbar(sc, ax=ax2, pad=0.01)
            cbar.ax.yaxis.set_tick_params(color="#8b949e")
            cbar.set_label("Delay (months)", color="#8b949e")
            ax2.set_xscale("log")
            ax2.set_xlabel("Original Cost (₹ crore, log scale)", labelpad=8)
            ax2.set_ylabel("Delay (months)", labelpad=8)
            ax2.set_title("Delay Severity: Original Budget vs. Delay Duration", pad=10, fontsize=11, fontweight="600")
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    st.markdown('<div class="insight-box">💡 <b>Reading this chart:</b> Each point is one delayed project record. Points farther right are larger budgets; higher up = longer delays. Costlier projects don\'t necessarily have longer delays — see the spread at the right edge.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Sector Trends
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Sector Trends":
    st.markdown("<h1>Sector Trends</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:0.9rem;'>"
        "Mean Cost Overrun % and Time Overrun (months) per sector across 5 report snapshots, "
        "sourced from Annexure 1 (Sector-Wise Focused Attention list).</p>",
        unsafe_allow_html=True,
    )

    trends = load_sector_trends()

    if trends.empty:
        st.warning("No sector trend data available. Run the pipeline first.")
    else:
        sectors = sorted(trends["sector"].unique())
        snapshots = sorted(trends["snapshot"].unique())

        # ── Sector selector ───────────────────────────────────────────────────
        sel_sectors = st.multiselect(
            "Select sectors to display",
            options=sectors,
            default=sectors[:min(8, len(sectors))],
        )

        if not sel_sectors:
            st.info("Select at least one sector above.")
        else:
            filtered = trends[trends["sector"].isin(sel_sectors)]

            tab1, tab2 = st.tabs(["💰 Cost Overrun %", "⏱️ Time Overrun (months)"])

            def _trend_chart(y_col: str, y_label: str, title: str, snapshot_labels: list):
                fig, ax = plt.subplots(figsize=(12, 5))
                for idx, sector in enumerate(sel_sectors):
                    sub = filtered[filtered["sector"] == sector].sort_values("snapshot")
                    if sub.empty:
                        continue
                    color = SECTOR_COLORS[idx % len(SECTOR_COLORS)]
                    vals = sub[y_col].fillna(0).values
                    snaps = sub["snapshot"].values
                    ax.plot(snaps, vals, "o-", color=color, linewidth=2, markersize=6,
                            label=sector, markerfacecolor=color, markeredgecolor="#0d1117",
                            markeredgewidth=1.5)
                    # label last point
                    if len(vals) > 0:
                        ax.annotate(
                            f" {vals[-1]:.0f}",
                            xy=(snaps[-1], vals[-1]),
                            fontsize=7.5, color=color, va="center",
                        )
                ax.set_xticks(range(len(snapshots)))
                ax.set_xticklabels([s.replace("-", "\n") for s in snapshots], fontsize=9)
                ax.set_ylabel(y_label, labelpad=8)
                ax.set_title(title, pad=12, fontsize=12, fontweight="600")
                ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9,
                          fontsize=8, borderpad=0.8)
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                return fig

            with tab1:
                fig = _trend_chart(
                    "mean_cor_pct", "Mean Cost Overrun %",
                    "Sector-Wise Cost Overrun Trend (Apr–Sep 2024)",
                    snapshots
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with tab2:
                fig = _trend_chart(
                    "mean_tor_months", "Mean Time Overrun (months)",
                    "Sector-Wise Time Overrun Trend (Apr–Sep 2024)",
                    snapshots
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Heatmap: sectors × snapshots ─────────────────────────────────────
        st.markdown('<div class="section-header">🔥 COR% Heatmap: Sector × Snapshot</div>', unsafe_allow_html=True)

        pivot = trends.pivot_table(
            index="sector", columns="snapshot", values="mean_cor_pct", aggfunc="mean"
        ).fillna(0)

        if not pivot.empty:
            fig3, ax3 = plt.subplots(figsize=(10, max(4, len(pivot) * 0.45)))
            sns.heatmap(
                pivot, ax=ax3, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.5, linecolor="#0d1117",
                cbar_kws={"label": "Mean COR %", "shrink": 0.8},
                annot_kws={"size": 8},
            )
            ax3.set_title("Cost Overrun % by Sector and Snapshot", pad=12, fontsize=11, fontweight="600")
            ax3.set_xlabel("Report Snapshot", labelpad=8)
            ax3.set_ylabel("")
            ax3.tick_params(axis="x", labelsize=9, rotation=0)
            ax3.tick_params(axis="y", labelsize=8, rotation=0)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        st.markdown('<div class="insight-box red-tint">⚠️ Higher values (darker red) indicate sectors where anticipated costs consistently exceed original sanctioned costs. This is calculated from actual Annexure 1 data in the MoSPI Flash Reports.</div>', unsafe_allow_html=True)

        # ── Raw table ─────────────────────────────────────────────────────────
        with st.expander("View raw sector trend data"):
            st.dataframe(
                trends.rename(columns={
                    "mean_cor_pct": "Mean COR%",
                    "mean_tor_months": "Mean TOR (months)",
                    "project_count": "# Projects",
                }),
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Chronic Offenders
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Chronic Offenders":
    st.markdown("<h1>Chronic Offenders</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:0.9rem;'>"
        "Real named projects appearing in multiple MoSPI Flash Reports, ranked by a "
        "severity score combining mean Cost Overrun % and Time Overrun months.</p>",
        unsafe_allow_html=True,
    )

    chronic = load_chronic()

    if chronic.empty:
        st.warning("No chronic offenders found. Ensure the pipeline has run with multiple report snapshots.")
    else:
        # ── Metric strip ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(kpi_card(
                "Repeat Offenders",
                f"{len(chronic)}",
                "projects in 2+ snapshots",
                "kpi-red",
            ), unsafe_allow_html=True)
        with c2:
            worst = chronic.iloc[0]
            st.markdown(kpi_card(
                "Worst Project",
                worst["project_name"][:32] + ("…" if len(worst["project_name"]) > 32 else ""),
                f"{worst['sector']} — {worst['mean_cor_pct']:.1f}% COR",
                "kpi-red",
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card(
                "Max Time Overrun",
                f"{chronic['max_tor_months'].max():.0f}",
                "months — worst offender",
                "kpi-amber",
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Horizontal bar: severity score ────────────────────────────────────
        st.markdown('<div class="section-header">📊 Top 20 Projects by Severity Score</div>', unsafe_allow_html=True)

        top20 = chronic.head(20).copy()
        fig4, ax4 = plt.subplots(figsize=(12, 6))

        labels = [n[:45] + ("…" if len(n) > 45 else "") for n in top20["project_name"][::-1]]
        scores = top20["severity_score"][::-1].values
        cors = top20["mean_cor_pct"][::-1].values

        colors4 = ["#f85149" if s > 100 else "#d29922" if s > 50 else "#58a6ff" for s in scores]
        bars4 = ax4.barh(labels, scores, color=colors4, height=0.65, edgecolor="none")
        for bar, cor in zip(bars4, cors):
            ax4.text(
                bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"COR {cor:.0f}%",
                va="center", fontsize=7.5, color="#8b949e",
            )
        ax4.set_xlabel("Severity Score (0.6 × COR% + 0.4 × TOR months)", labelpad=8)
        ax4.set_title("Chronic Offenders — Project Severity Ranking", pad=12, fontsize=12, fontweight="600")
        ax4.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        # ── Scatter: COR vs TOR ───────────────────────────────────────────────
        st.markdown('<div class="section-header">💠 Cost Overrun vs. Time Overrun</div>', unsafe_allow_html=True)

        fig5, ax5 = plt.subplots(figsize=(10, 5))
        unique_sectors = chronic["sector"].unique()
        sector_color_map = {s: SECTOR_COLORS[i % len(SECTOR_COLORS)]
                            for i, s in enumerate(sorted(unique_sectors))}
        for _, row in chronic.head(40).iterrows():
            ax5.scatter(
                row["mean_cor_pct"], row["mean_tor_months"],
                color=sector_color_map.get(row["sector"], "#58a6ff"),
                s=max(40, row["snapshot_count"] * 30),
                alpha=0.75, edgecolors="none",
                label=row["sector"],
            )
        # De-dupe legend
        handles, labels_l = ax5.get_legend_handles_labels()
        by_label = dict(zip(labels_l, handles))
        ax5.legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
                   title="Sector", title_fontsize=8)
        ax5.set_xlabel("Mean Cost Overrun %", labelpad=8)
        ax5.set_ylabel("Mean Time Overrun (months)", labelpad=8)
        ax5.set_title("COR% vs. TOR — bubble size = appearances in reports", pad=10, fontsize=11, fontweight="600")
        ax5.grid(alpha=0.3)
        # Quadrant labels
        xlim, ylim = ax5.get_xlim(), ax5.get_ylim()
        ax5.axhline(chronic["mean_tor_months"].median(), color="#30363d", linestyle=":", linewidth=1)
        ax5.axvline(chronic["mean_cor_pct"].median(), color="#30363d", linestyle=":", linewidth=1)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

        # ── Full table ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Full Chronic Offenders Table</div>', unsafe_allow_html=True)
        display_cols = {
            "project_name": "Project Name",
            "sector": "Sector",
            "snapshot_count": "Appearances",
            "mean_cor_pct": "Mean COR%",
            "max_cor_pct": "Max COR%",
            "mean_tor_months": "Mean TOR (months)",
            "max_tor_months": "Max TOR (months)",
            "original_cost": "Original Cost (₹ cr)",
            "anticipated_cost": "Anticipated Cost (₹ cr)",
            "severity_score": "Severity Score",
        }
        col_map = {k: v for k, v in display_cols.items() if k in chronic.columns}
        st.dataframe(
            chronic.rename(columns=col_map)[list(col_map.values())]
            .style.format({
                "Mean COR%": "{:.1f}",
                "Max COR%": "{:.1f}",
                "Mean TOR (months)": "{:.1f}",
                "Max TOR (months)": "{:.0f}",
                "Original Cost (₹ cr)": "{:,.1f}",
                "Anticipated Cost (₹ cr)": "{:,.1f}",
                "Severity Score": "{:.1f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Source: MoSPI Flash Reports, Annexure 1 (Sector-Wise Focused Attention Projects), Apr–Sep 2024")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Forecast Sandbox
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Forecast Sandbox":
    st.markdown("<h1>Forecast Sandbox</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:0.9rem;'>"
        "Ridge regression model trained on real MoSPI data. "
        "Input a sector + budget to predict expected cost overrun % and delay months.</p>",
        unsafe_allow_html=True,
    )

    sectors = get_known_sectors()

    if not sectors:
        st.warning("No trained models found. Run `python src/pipeline.py` to train models.")
    else:
        col_input, col_results = st.columns([1, 1.2], gap="large")

        with col_input:
            st.markdown('<div class="section-header">⚙️ Project Parameters</div>', unsafe_allow_html=True)
            sector_choice = st.selectbox("Sector", options=sectors)
            state_choice = st.selectbox("State", options=get_known_states())
            original_cost = st.number_input(
                "Original Sanctioned Cost (₹ crore)",
                min_value=150.0, max_value=200000.0, value=1000.0, step=100.0,
                help="MoSPI monitors projects ≥ ₹150 crore"
            )
            predict_btn = st.button("🔮 Predict Overrun", use_container_width=True)

        with col_results:
            if predict_btn:
                with st.spinner("Running model…"):
                    result = predict(sector_choice, state_choice, original_cost)

                cor = result.get("cost_overrun_pct")
                delay = result.get("delay_months")

                st.markdown('<div class="section-header">📊 Model Predictions</div>', unsafe_allow_html=True)
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(kpi_card(
                        "Predicted Cost Overrun",
                        f"{cor:.1f}%" if cor is not None else "N/A",
                        f"≈ ₹{original_cost * cor / 100:,.0f} crore extra" if cor is not None else "",
                        "kpi-red" if cor and cor > 30 else "kpi-amber" if cor and cor > 10 else "kpi-green",
                    ), unsafe_allow_html=True)
                with r2:
                    st.markdown(kpi_card(
                        "Predicted Delay",
                        f"{delay:.1f} months" if delay is not None else "N/A",
                        f"≈ {delay/12:.1f} years" if delay and delay > 12 else "< 1 year",
                        "kpi-red" if delay and delay > 24 else "kpi-amber" if delay and delay > 12 else "kpi-blue",
                    ), unsafe_allow_html=True)

                if cor is not None and delay is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    # Simple gauge bar
                    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(10, 3))
                    for ax, val, maxv, label, color in [
                        (ax6a, cor, 200, "Cost Overrun %", "#f85149"),
                        (ax6b, delay, 200, "Delay (months)", "#d29922"),
                    ]:
                        pct = min(val / maxv, 1.0)
                        ax.barh([""], [maxv], color="#21262d", height=0.35, edgecolor="none")
                        ax.barh([""], [val], color=color, height=0.35, edgecolor="none", alpha=0.85)
                        ax.set_xlim(0, maxv)
                        ax.set_title(label, fontsize=10, fontweight="600", color="#f0f6fc")
                        ax.text(val + maxv * 0.02, 0, f"{val:.1f}", va="center", fontsize=11,
                                fontweight="600", color="#f0f6fc")
                        ax.axis("off")
                    fig6.patch.set_facecolor("#0d1117")
                    plt.tight_layout(pad=1.5)
                    st.pyplot(fig6, use_container_width=True)
                    plt.close(fig6)

        # ── Feature importance ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔍 Model Feature Importances</div>', unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        for col_f, label, title in [
            (f1, "cor", "Cost Overrun % — Feature Coefficients"),
            (f2, "delay", "Delay (months) — Feature Coefficients"),
        ]:
            with col_f:
                imp = get_feature_importance(label)
                if not imp.empty:
                    top_imp = imp.head(12)
                    fig7, ax7 = plt.subplots(figsize=(5.5, 4))
                    colors7 = ["#f85149" if v > 0 else "#58a6ff" for v in top_imp["coefficient"]]
                    ax7.barh(
                        top_imp["feature"][::-1],
                        top_imp["coefficient"][::-1],
                        color=colors7[::-1], height=0.6, edgecolor="none"
                    )
                    ax7.axvline(0, color="#30363d", linewidth=0.8)
                    ax7.set_title(title, fontsize=9, fontweight="600", color="#f0f6fc")
                    ax7.tick_params(labelsize=7.5)
                    plt.tight_layout()
                    st.pyplot(fig7, use_container_width=True)
                    plt.close(fig7)
                else:
                    st.info(f"No importance data for {label} model.")

        st.markdown("""
        <div class="insight-box">
        💡 <b>Interpretation:</b> Ridge regression coefficients show the additive effect of each feature on the
        prediction. Red = positive contribution (increases overrun), blue = negative. The log-scale cost feature
        captures the non-linear relationship between project size and cost overrun behaviour.
        </div>
        """, unsafe_allow_html=True)
