"""
InfraForecast - Pipeline Orchestrator
======================================
Run this once to bootstrap the entire database and models:
  python -m src.pipeline   (from InfraForecast root)
  OR
  python src/pipeline.py

Steps:
  1. Download 5 quarterly MoSPI Flash Report PDFs
  2. Parse 3 target tables per report
  3. Insert into SQLite (infraforecast.db)
  4. Train Ridge regression models
"""

import logging
import sys
import os

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(DATA_DIR, "pipeline.log"),
            mode="a", encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("pipeline")


from src.pdf_parser import parse_all_reports
from src import database as db
from src.analysis import delayed_projects_clean
from src.model import train_models


def run():
    logger.info("=" * 60)
    logger.info("InfraForecast Pipeline Starting")
    logger.info("=" * 60)

    # ── Step 1 & 2: Download + Parse ───────────────────────────────
    logger.info("Step 1/3: Downloading and parsing MoSPI Flash Reports...")
    report_data = parse_all_reports()

    # ── Step 3: Insert into DB ─────────────────────────────────────
    logger.info("Step 2/3: Inserting parsed data into SQLite...")
    conn = db.get_connection()
    db.create_tables(conn)

    total_t16 = total_ann1 = total_ann2 = 0
    for snapshot, tables in report_data.items():
        t16 = tables.get("t16")
        ann1 = tables.get("ann1")
        ann2 = tables.get("ann2")

        if t16 is not None and not t16.empty:
            db.insert_delayed_projects(conn, t16)
            total_t16 += len(t16)
        else:
            logger.warning("[%s] T16 (delayed projects) is empty", snapshot)

        if ann1 is not None and not ann1.empty:
            db.insert_sector_focused(conn, ann1)
            total_ann1 += len(ann1)
        else:
            logger.warning("[%s] Ann1 (sector focused) is empty", snapshot)

        if ann2 is not None and not ann2.empty:
            db.insert_state_summary(conn, ann2)
            total_ann2 += len(ann2)
        else:
            logger.warning("[%s] Ann2 (state summary) is empty", snapshot)

    conn.close()

    stats = db.get_db_stats()
    logger.info("DB stats -> delayed_projects: %d, sector_focused: %d, state_summary: %d",
                stats.get("delayed_projects", 0),
                stats.get("sector_focused", 0),
                stats.get("state_summary", 0))

    # ── Step 4: Train models ────────────────────────────────────────
    logger.info("Step 3/3: Training Ridge regression models...")
    df = delayed_projects_clean()
    if not df.empty:
        results = train_models(df)
        for label, info in results.items():
            logger.info(
                "  [%s] R²=%.3f (±%.3f) on %d samples",
                label, info["r2_mean"], info["r2_std"], info["n_samples"]
            )
    else:
        logger.warning("No training data available — models not trained.")

    logger.info("=" * 60)
    logger.info("Pipeline complete! Run: streamlit run src/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
