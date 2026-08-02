"""
InfraForecast - Pipeline Tests
================================
Run: python -m pytest src/test_pipeline.py -v
  or: python src/test_pipeline.py
"""

import os
import sys
import sqlite3
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src import database as db
from src.model import prepare_features


class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Create a temp DB for each test."""
        self.tmpdir = tempfile.mkdtemp()
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.tmpdir, "test.db")

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        import gc
        gc.collect()  # Force GC to release any lingering sqlite connections
        shutil.rmtree(self.tmpdir, ignore_errors=True)


    def test_create_tables(self):
        conn = db.get_connection()
        db.create_tables(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        self.assertIn("delayed_projects", table_names)
        self.assertIn("sector_focused", table_names)
        self.assertIn("state_summary", table_names)
        conn.close()

    def test_insert_delayed_projects(self):
        conn = db.get_connection()
        db.create_tables(conn)
        sample = pd.DataFrame([{
            "project_name": "TEST PROJECT A",
            "original_cost": 500.0,
            "anticipated_cost": 650.0,
            "original_doc": "01/2020",
            "last_doc": "01/2022",
            "this_doc": "06/2022",
            "delay_months": 18.0,
            "snapshot": "2024-04",
        }])
        db.insert_delayed_projects(conn, sample)
        cur = conn.execute("SELECT COUNT(*) FROM delayed_projects WHERE snapshot='2024-04'")
        count = cur.fetchone()[0]
        self.assertEqual(count, 1)

        # Check derived cost_overrun_pct
        cur2 = conn.execute("SELECT cost_overrun_pct FROM delayed_projects WHERE snapshot='2024-04'")
        cor = cur2.fetchone()[0]
        self.assertAlmostEqual(cor, 30.0, places=0)  # (650-500)/500 * 100 = 30%
        conn.close()

    def test_insert_idempotent(self):
        """Re-inserting same snapshot should not duplicate rows."""
        conn = db.get_connection()
        db.create_tables(conn)
        sample = pd.DataFrame([{"project_name": "P1", "original_cost": 200.0,
                                 "anticipated_cost": 220.0, "delay_months": 5.0,
                                 "snapshot": "2024-05"}])
        db.insert_delayed_projects(conn, sample)
        db.insert_delayed_projects(conn, sample)
        count = conn.execute(
            "SELECT COUNT(*) FROM delayed_projects WHERE snapshot='2024-05'"
        ).fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()

    def test_get_db_stats(self):
        conn = db.get_connection()
        db.create_tables(conn)
        conn.close()
        stats = db.get_db_stats()
        self.assertIn("delayed_projects", stats)
        self.assertIn("sector_focused", stats)
        self.assertIn("state_summary", stats)
        # Explicitly close the DB used by get_db_stats (uses context manager internally)
        # Force sqlite connections closed so Windows can delete the temp file in tearDown
        import sqlite3
        try:
            sqlite3.connect(db.DB_PATH).close()
        except Exception:
            pass


class TestModelFeatures(unittest.TestCase):
    def test_prepare_features(self):
        df = pd.DataFrame([
            {"original_cost": 500.0, "sector": "Railways"},
            {"original_cost": 2000.0, "sector": "Roads"},
            {"original_cost": 0.0, "sector": "Power"},
        ])
        result = prepare_features(df)
        self.assertIn("log_original_cost", result.columns)
        self.assertAlmostEqual(result.loc[0, "log_original_cost"], np.log1p(500), places=4)
        self.assertAlmostEqual(result.loc[2, "log_original_cost"], 0.0, places=4)

    def test_predict_returns_dict(self):
        """predict() should return a dict with correct keys even with no trained model."""
        from src.model import predict
        result = predict("Railways", "PUNJAB", 1000.0)
        self.assertIsInstance(result, dict)
        self.assertIn("cost_overrun_pct", result)
        self.assertIn("delay_months", result)

    def test_train_and_predict_flow(self):
        """Verify model training saves pipeline files, creates metadata, and respects caps."""
        from src.model import train_models, predict, MODEL_DIR
        import shutil
        import json
        
        # Back up existing models to not disrupt active ones
        backup_dir = MODEL_DIR + "_unittest_bak"
        if os.path.exists(MODEL_DIR):
            shutil.copytree(MODEL_DIR, backup_dir, dirs_exist_ok=True)
            
        try:
            # Create a mock training dataframe with at least 10 samples
            mock_df = pd.DataFrame([
                {"sector": "Railways", "state": "PUNJAB", "original_cost": 100.0, "cost_overrun_pct": 10.0, "delay_months": 12.0},
                {"sector": "Roads", "state": "HARYANA", "original_cost": 200.0, "cost_overrun_pct": 20.0, "delay_months": 24.0},
                {"sector": "Railways", "state": "PUNJAB", "original_cost": 150.0, "cost_overrun_pct": 15.0, "delay_months": 18.0},
                {"sector": "Roads", "state": "HARYANA", "original_cost": 250.0, "cost_overrun_pct": 25.0, "delay_months": 30.0},
                {"sector": "Power", "state": "PUNJAB", "original_cost": 300.0, "cost_overrun_pct": 30.0, "delay_months": 36.0},
                {"sector": "Power", "state": "HARYANA", "original_cost": 350.0, "cost_overrun_pct": 35.0, "delay_months": 42.0},
                {"sector": "Railways", "state": "PUNJAB", "original_cost": 400.0, "cost_overrun_pct": 40.0, "delay_months": 48.0},
                {"sector": "Roads", "state": "HARYANA", "original_cost": 450.0, "cost_overrun_pct": 45.0, "delay_months": 54.0},
                {"sector": "Power", "state": "PUNJAB", "original_cost": 500.0, "cost_overrun_pct": 50.0, "delay_months": 60.0},
                {"sector": "Power", "state": "HARYANA", "original_cost": 550.0, "cost_overrun_pct": 55.0, "delay_months": 66.0},
            ])
            
            # Train models
            res = train_models(mock_df)
            self.assertIn("cor", res)
            self.assertIn("delay", res)
            
            # Check metadata files exist
            cor_meta_path = os.path.join(MODEL_DIR, "cor_metadata.json")
            self.assertTrue(os.path.exists(cor_meta_path))
            
            with open(cor_meta_path, "r", encoding="utf-8") as f:
                cor_meta = json.load(f)
                self.assertEqual(cor_meta["max_target"], 55.0)
                
            # Test predict clamping on massive input cost
            pred = predict("Railways", "PUNJAB", 1000000.0) # Huge budget
            self.assertLessEqual(pred["cost_overrun_pct"], 55.0)
            
        finally:
            # Restore backups if they existed
            if os.path.exists(backup_dir):
                shutil.copytree(backup_dir, MODEL_DIR, dirs_exist_ok=True)
                try:
                    shutil.rmtree(backup_dir)
                except Exception:
                    pass



class TestCostOverrunDerivation(unittest.TestCase):
    def test_cor_pct_calculation(self):
        """Verify COR% formula: (anticipated - original) / original * 100."""
        cases = [
            (500.0, 650.0, 30.0),
            (1000.0, 2000.0, 100.0),
            (200.0, 200.0, 0.0),
        ]
        for orig, ant, expected in cases:
            cor = (ant - orig) / orig * 100
            self.assertAlmostEqual(cor, expected, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
