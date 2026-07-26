"""
InfraForecast - Analysis Module
================================
Computes derived analytics from the SQLite database:
  - Sector-wise COR% and TOR trend across 5 snapshots
  - State-wise cost overrun ratio
  - Chronic offenders: projects in >= 3 snapshots with escalating overrun
"""

import logging
import numpy as np
import pandas as pd
from . import database as db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector trends
# ---------------------------------------------------------------------------

def sector_trends() -> pd.DataFrame:
    """
    Returns a DataFrame of mean cost_overrun_pct and time_overrun_months
    per sector per snapshot from the sector_focused table.
    """
    df = db.load_table("sector_focused")
    if df.empty:
        logger.warning("sector_focused table is empty")
        return pd.DataFrame()

    df["cost_overrun_pct"] = pd.to_numeric(df["cost_overrun_pct"], errors="coerce")
    df["time_overrun_months"] = pd.to_numeric(df["time_overrun_months"], errors="coerce")

    grouped = (
        df.groupby(["sector", "snapshot"], as_index=False)
        .agg(
            mean_cor_pct=("cost_overrun_pct", "mean"),
            mean_tor_months=("time_overrun_months", "mean"),
            project_count=("project_name", "count"),
        )
        .sort_values(["sector", "snapshot"])
    )
    return grouped


def worst_sectors_by_snapshot() -> pd.DataFrame:
    """Return the worst sector by mean COR% per snapshot."""
    trends = sector_trends()
    if trends.empty:
        return pd.DataFrame()
    idx = trends.groupby("snapshot")["mean_cor_pct"].idxmax()
    return trends.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# State-wise analysis
# ---------------------------------------------------------------------------

def state_overrun_summary() -> pd.DataFrame:
    """
    Aggregate state-level cost overrun across all snapshots.
    Returns mean COR% per state, sorted descending.
    """
    df = db.load_table("state_summary")
    if df.empty:
        return pd.DataFrame()

    df["cost_overrun_pct"] = pd.to_numeric(df["cost_overrun_pct"], errors="coerce")
    df["original_cost"] = pd.to_numeric(df["original_cost"], errors="coerce")
    df["anticipated_cost"] = pd.to_numeric(df["anticipated_cost"], errors="coerce")

    agg = (
        df.groupby("state", as_index=False)
        .agg(
            snapshots=("snapshot", "count"),
            mean_cor_pct=("cost_overrun_pct", "mean"),
            total_original_cost=("original_cost", "mean"),
            total_anticipated_cost=("anticipated_cost", "mean"),
        )
        .sort_values("mean_cor_pct", ascending=False)
        .reset_index(drop=True)
    )
    return agg


# ---------------------------------------------------------------------------
# Chronic offenders
# ---------------------------------------------------------------------------

def chronic_offenders(min_snapshots: int = 3) -> pd.DataFrame:
    """
    Projects appearing in >= min_snapshots reports in sector_focused,
    sorted by descending mean COR%.

    These are the "worst of the worst" – real named projects that have
    persistently failed to meet original cost or schedule targets.
    """
    df = db.load_table("sector_focused")
    if df.empty:
        return pd.DataFrame()

    df["cost_overrun_pct"] = pd.to_numeric(df["cost_overrun_pct"], errors="coerce")
    df["time_overrun_months"] = pd.to_numeric(df["time_overrun_months"], errors="coerce")
    df["original_cost"] = pd.to_numeric(df["original_cost"], errors="coerce")
    df["anticipated_cost"] = pd.to_numeric(df["anticipated_cost"], errors="coerce")

    agg = (
        df.groupby(["project_name", "sector"], as_index=False)
        .agg(
            snapshot_count=("snapshot", "nunique"),
            mean_cor_pct=("cost_overrun_pct", "mean"),
            max_cor_pct=("cost_overrun_pct", "max"),
            mean_tor_months=("time_overrun_months", "mean"),
            max_tor_months=("time_overrun_months", "max"),
            original_cost=("original_cost", "first"),
            anticipated_cost=("anticipated_cost", "last"),
        )
    )

    chronic = agg[agg["snapshot_count"] >= min_snapshots].copy()
    chronic["severity_score"] = (
        chronic["mean_cor_pct"].fillna(0) * 0.6 +
        chronic["mean_tor_months"].fillna(0) * 0.4
    )
    chronic = chronic.sort_values("severity_score", ascending=False).reset_index(drop=True)
    return chronic


# ---------------------------------------------------------------------------
# Delayed projects aggregates (for regression model data)
# ---------------------------------------------------------------------------

def delayed_projects_clean() -> pd.DataFrame:
    """
    Load and clean the delayed_projects table for modelling.
    Infers sector from sector_focused crosswalk where possible.
    """
    dp = db.load_table("delayed_projects")
    sf = db.load_table("sector_focused")

    if dp.empty:
        return pd.DataFrame()

    for col in ["original_cost", "anticipated_cost", "cost_overrun_pct", "delay_months"]:
        dp[col] = pd.to_numeric(dp[col], errors="coerce")

    # Build a name -> sector lookup from Annexure 1
    sector_map = {}
    if not sf.empty:
        for _, row in sf.iterrows():
            name = str(row.get("project_name", "")).strip().upper()[:60]
            sector = str(row.get("sector", "")).strip()
            if name and sector:
                sector_map[name] = sector

    dp["sector"] = dp["project_name"].apply(
        lambda n: sector_map.get(str(n).strip().upper()[:60], "Unknown")
    )
    
    # Fill NaN states (e.g. from Format A) with 'Unknown'
    if "state" in dp.columns:
        dp["state"] = dp["state"].fillna("Unknown").apply(lambda s: str(s).strip() if str(s).strip() else "Unknown")
    else:
        dp["state"] = "Unknown"

    # Filter out rows with missing key fields
    dp = dp.dropna(subset=["original_cost", "delay_months"])
    dp = dp[dp["original_cost"] > 0]
    dp = dp[dp["delay_months"] >= 0]

    return dp


def summary_stats() -> dict:
    """Return top-level summary metrics for the Overview dashboard page."""
    dp = db.load_table("delayed_projects")
    sf = db.load_table("sector_focused")
    ss = db.load_table("state_summary")

    for col in ["original_cost", "anticipated_cost", "cost_overrun_pct", "delay_months"]:
        if col in dp.columns:
            dp[col] = pd.to_numeric(dp[col], errors="coerce")

    stats = {
        "total_delayed_projects": len(dp),
        "snapshots": sorted(dp["snapshot"].unique().tolist()) if not dp.empty else [],
        "total_original_cost_cr": dp["original_cost"].sum() if not dp.empty else 0,
        "total_anticipated_cost_cr": dp["anticipated_cost"].sum() if not dp.empty else 0,
        "mean_delay_months": round(dp["delay_months"].mean(), 1) if not dp.empty else 0,
        "max_delay_months": dp["delay_months"].max() if not dp.empty else 0,
    }

    if not sf.empty:
        for col in ["cost_overrun_pct", "time_overrun_months"]:
            if col in sf.columns:
                sf[col] = pd.to_numeric(sf[col], errors="coerce")
        worst_sector_row = sf.groupby("sector")["cost_overrun_pct"].mean()
        stats["worst_sector"] = worst_sector_row.idxmax() if not worst_sector_row.empty else "N/A"
        stats["worst_sector_cor"] = round(worst_sector_row.max(), 1) if not worst_sector_row.empty else 0

    return stats
