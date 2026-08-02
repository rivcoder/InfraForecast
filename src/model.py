"""
InfraForecast - Ridge Regression Model
=======================================
Trains two Ridge regression models on real parsed data:
  - Model A: predict cost_overrun_pct  (target: how much over budget?)
  - Model B: predict delay_months      (target: how many months late?)

Features:
  - sector (one-hot encoded)
  - log(original_cost + 1)  <- log-transform to handle wide budget ranges

Saves trained models + feature importance to data/models/.
"""

import os
import logging
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")


def _build_pipeline() -> Pipeline:
    """Build the sklearn pipeline: one-hot sector & state + scaled log-cost -> Ridge."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("sector_enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
            ("state_enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["state"]),
            ("cost_scaler", StandardScaler(), ["log_original_cost"]),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("ridge", Ridge(alpha=1.0)),
    ])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed cost feature."""
    df = df.copy()
    df["log_original_cost"] = np.log1p(df["original_cost"].fillna(0))
    return df


def train_models(df: pd.DataFrame) -> dict:
    """
    Train both models. Returns dict with models + feature importance DataFrames.
    df must contain: sector, state, original_cost, cost_overrun_pct, delay_months
    """
    if df.empty or len(df) < 10:
        logger.warning("Insufficient data for training (%d rows). Need >= 10.", len(df))
        return {}

    df = prepare_features(df)
    feature_cols = ["sector", "state", "log_original_cost"]

    results = {}

    for target, label in [("cost_overrun_pct", "cor"), ("delay_months", "delay")]:
        valid = df.dropna(subset=[target] + feature_cols).copy()
        # Filter target to be finite and positive/zero for log transform
        valid = valid[np.isfinite(valid[target])]
        valid = valid[valid[target] >= 0]

        if len(valid) < 10:
            logger.warning("Not enough valid rows for %s model (%d)", target, len(valid))
            continue

        X = valid[feature_cols]
        # Log transform target variable to handle extreme skew/outliers
        y = np.log1p(valid[target])

        pipe = _build_pipeline()
        pipe.fit(X, y)

        # Cross-validation R² on log scale target
        cv_scores = cross_val_score(pipe, X, y, cv=min(5, len(valid) // 2), scoring="r2")
        logger.info("[%s] Ridge Log-R² = %.3f ± %.3f (n=%d)", label, cv_scores.mean(), cv_scores.std(), len(valid))

        # Feature importance (Ridge coefficients after preprocessing)
        preprocessor = pipe.named_steps["preprocessor"]
        sector_enc = preprocessor.named_transformers_["sector_enc"]
        state_enc = preprocessor.named_transformers_["state_enc"]
        
        sector_names = [f"sector={c}" for c in sector_enc.categories_[0]]
        state_names = [f"state={c}" for c in state_enc.categories_[0]]
        feature_names = sector_names + state_names + ["log_original_cost"]
        
        coefs = pipe.named_steps["ridge"].coef_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{label}_model.joblib")
        importance_path = os.path.join(MODEL_DIR, f"{label}_importance.csv")
        metadata_path = os.path.join(MODEL_DIR, f"{label}_metadata.json")

        joblib.dump(pipe, model_path)
        importance_df.to_csv(importance_path, index=False)
        
        # Save training metadata to clamp predictions during sandbox testing
        metadata = {
            "max_target": float(valid[target].max()) if len(valid) > 0 else 500.0,
            "mean_target": float(valid[target].mean()) if len(valid) > 0 else 0.0,
            "n_samples": int(len(valid))
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info("Saved model -> %s and metadata -> %s", model_path, metadata_path)

        results[label] = {
            "model": pipe,
            "r2_mean": cv_scores.mean(),
            "r2_std": cv_scores.std(),
            "n_samples": len(valid),
            "importance": importance_df,
        }

    return results


def load_model(label: str):
    """Load a saved model by label ('cor' or 'delay')."""
    path = os.path.join(MODEL_DIR, f"{label}_model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict(sector: str, state: str, original_cost_cr: float) -> dict:
    """
    Single-point prediction for Streamlit sandbox.
    Returns {'cost_overrun_pct': float, 'delay_months': float}
    """
    input_df = pd.DataFrame([{
        "sector": sector,
        "state": state,
        "log_original_cost": np.log1p(max(0, original_cost_cr)),
    }])

    output = {}
    for label, key in [("cor", "cost_overrun_pct"), ("delay", "delay_months")]:
        model = load_model(label)
        if model is not None:
            try:
                pred_log = model.predict(input_df)[0]
                # Inverse transform of log1p: expm1
                pred_orig = np.expm1(pred_log)
                
                # Check for target clamping bounds from metadata to avoid extreme extrapolation
                max_cap = 1000.0 if label == "cor" else 480.0
                meta_path = os.path.join(MODEL_DIR, f"{label}_metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            max_cap = meta.get("max_target", max_cap)
                    except Exception:
                        pass
                
                final_val = min(float(pred_orig), max_cap)
                output[key] = max(0.0, round(final_val, 2))
            except Exception as e:
                logger.warning("Prediction failed for %s: %s", label, e)
                output[key] = None
        else:
            output[key] = None

    return output


def get_feature_importance(label: str) -> pd.DataFrame:
    """Load feature importance CSV for a model."""
    path = os.path.join(MODEL_DIR, f"{label}_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def get_known_sectors() -> list[str]:
    """Return list of sectors seen during training (from saved OHE categories)."""
    model = load_model("cor")
    if model is None:
        return []
    try:
        enc = model.named_steps["preprocessor"].named_transformers_["sector_enc"]
        return list(enc.categories_[0])
    except Exception:
        return []


def get_known_states() -> list[str]:
    """Return list of states seen during training (from saved OHE categories)."""
    model = load_model("cor")
    if model is None:
        return []
    try:
        enc = model.named_steps["preprocessor"].named_transformers_["state_enc"]
        return list(enc.categories_[0])
    except Exception:
        return []
