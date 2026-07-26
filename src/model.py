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
    """Build the sklearn pipeline: one-hot sector + scaled log-cost -> Ridge."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("sector_enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
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
    df must contain: sector, original_cost, cost_overrun_pct, delay_months
    """
    if df.empty or len(df) < 10:
        logger.warning("Insufficient data for training (%d rows). Need >= 10.", len(df))
        return {}

    df = prepare_features(df)
    feature_cols = ["sector", "log_original_cost"]

    results = {}

    for target, label in [("cost_overrun_pct", "cor"), ("delay_months", "delay")]:
        valid = df.dropna(subset=[target] + feature_cols).copy()
        valid = valid[np.isfinite(valid[target])]

        if len(valid) < 10:
            logger.warning("Not enough valid rows for %s model (%d)", target, len(valid))
            continue

        X = valid[feature_cols]
        y = valid[target]

        pipe = _build_pipeline()
        pipe.fit(X, y)

        # Cross-validation R²
        cv_scores = cross_val_score(pipe, X, y, cv=min(5, len(valid) // 2), scoring="r2")
        logger.info("[%s] Ridge R² = %.3f ± %.3f (n=%d)", label, cv_scores.mean(), cv_scores.std(), len(valid))

        # Feature importance (Ridge coefficients after preprocessing)
        enc: OneHotEncoder = pipe.named_steps["preprocessor"].named_transformers_["sector_enc"]
        sector_names = [f"sector={c}" for c in enc.categories_[0]]
        feature_names = sector_names + ["log_original_cost"]
        coefs = pipe.named_steps["ridge"].coef_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{label}_model.joblib")
        importance_path = os.path.join(MODEL_DIR, f"{label}_importance.csv")

        joblib.dump(pipe, model_path)
        importance_df.to_csv(importance_path, index=False)
        logger.info("Saved model -> %s", model_path)

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


def predict(sector: str, original_cost_cr: float) -> dict:
    """
    Single-point prediction for Streamlit sandbox.
    Returns {'cost_overrun_pct': float, 'delay_months': float}
    """
    input_df = pd.DataFrame([{
        "sector": sector,
        "log_original_cost": np.log1p(max(0, original_cost_cr)),
    }])

    output = {}
    for label, key in [("cor", "cost_overrun_pct"), ("delay", "delay_months")]:
        model = load_model(label)
        if model is not None:
            try:
                pred = model.predict(input_df)[0]
                output[key] = max(0.0, round(float(pred), 2))
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
