"""
InfraForecast - SQLite Database Layer
=====================================
Creates and populates three tables in infraforecast.db from parsed data.
"""

import os
import sqlite3
import logging
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "infraforecast.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS delayed_projects (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name        TEXT    NOT NULL,
            original_cost       REAL,
            anticipated_cost    REAL,
            cost_overrun_pct    REAL,
            original_doc        TEXT,
            last_doc            TEXT,
            this_doc            TEXT,
            delay_months        REAL,
            snapshot            TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sector_focused (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name        TEXT    NOT NULL,
            sector              TEXT,
            doc_original        TEXT,
            doc_anticipated     TEXT,
            original_cost       REAL,
            anticipated_cost    REAL,
            cost_overrun_pct    REAL,
            time_overrun_months REAL,
            snapshot            TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS state_summary (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            state                   TEXT    NOT NULL,
            total_projects          INTEGER,
            original_cost           REAL,
            anticipated_cost        REAL,
            cumulative_expenditure  REAL,
            cost_overrun_pct        REAL,
            snapshot                TEXT    NOT NULL
        );
    """)
    conn.commit()
    logger.info("Tables created / verified.")


def insert_delayed_projects(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    # Derive cost_overrun_pct if both cost columns present
    if "original_cost" in df.columns and "anticipated_cost" in df.columns:
        df = df.copy()
        df["cost_overrun_pct"] = df.apply(
            lambda r: round(
                (r["anticipated_cost"] - r["original_cost"]) / r["original_cost"] * 100, 2
            )
            if r["original_cost"] and r["original_cost"] > 0 else None,
            axis=1,
        )
    cols = [
        "project_name", "original_cost", "anticipated_cost", "cost_overrun_pct",
        "original_doc", "last_doc", "this_doc", "delay_months", "snapshot"
    ]
    # Keep only cols that exist in df
    cols = [c for c in cols if c in df.columns]
    # Delete old snapshot rows first to support re-runs
    snapshots = df["snapshot"].unique().tolist()
    conn.execute(
        f"DELETE FROM delayed_projects WHERE snapshot IN ({','.join(['?']*len(snapshots))})",
        snapshots
    )
    df[cols].to_sql("delayed_projects", conn, if_exists="append", index=False)
    conn.commit()
    logger.info("Inserted %d rows into delayed_projects", len(df))


def insert_sector_focused(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    rename = {
        "cor_pct": "cost_overrun_pct",
        "tor_months": "time_overrun_months",
    }
    df.rename(columns=rename, inplace=True)
    cols = [
        "project_name", "sector", "doc_original", "doc_anticipated",
        "original_cost", "anticipated_cost", "cost_overrun_pct",
        "time_overrun_months", "snapshot"
    ]
    cols = [c for c in cols if c in df.columns]
    snapshots = df["snapshot"].unique().tolist()
    conn.execute(
        f"DELETE FROM sector_focused WHERE snapshot IN ({','.join(['?']*len(snapshots))})",
        snapshots
    )
    df[cols].to_sql("sector_focused", conn, if_exists="append", index=False)
    conn.commit()
    logger.info("Inserted %d rows into sector_focused", len(df))


def insert_state_summary(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    # Derive cost_overrun_pct at state aggregate level
    if "original_cost" in df.columns and "anticipated_cost" in df.columns:
        df["cost_overrun_pct"] = df.apply(
            lambda r: round(
                (r["anticipated_cost"] - r["original_cost"]) / r["original_cost"] * 100, 2
            )
            if r["original_cost"] and r["original_cost"] > 0 else None,
            axis=1,
        )
    cols = [
        "state", "total_projects", "original_cost", "anticipated_cost",
        "cumulative_expenditure", "cost_overrun_pct", "snapshot"
    ]
    cols = [c for c in cols if c in df.columns]
    snapshots = df["snapshot"].unique().tolist()
    conn.execute(
        f"DELETE FROM state_summary WHERE snapshot IN ({','.join(['?']*len(snapshots))})",
        snapshots
    )
    df[cols].to_sql("state_summary", conn, if_exists="append", index=False)
    conn.commit()
    logger.info("Inserted %d rows into state_summary", len(df))


def load_table(table: str) -> pd.DataFrame:
    """Convenience: load a full table into a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def get_db_stats() -> dict:
    """Return row counts for each table."""
    with get_connection() as conn:
        stats = {}
        for t in ["delayed_projects", "sector_focused", "state_summary"]:
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
                stats[t] = cur.fetchone()[0]
            except Exception:
                stats[t] = 0
    return stats
