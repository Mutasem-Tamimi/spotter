"""Shared cleaning, split verification, and reporting for alternative models.

The original CatBoost module and artifacts remain unchanged. These helpers fit
imputation, encoding, and optional scaling only on the rows passed to fit().
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from freight_model import (
    CATEGORICAL_FEATURES, COORDINATES, FreightPreprocessor,
    load_development, regression_metrics, split_development,
)

NUMERIC_FEATURES = [
    "distance", "weight", "weight_is_missing", "weight_was_negative",
    *COORDINATES, "days_since_start", "day_of_week", "day_of_month",
]


def make_preprocessor(scale=False):
    """Return an unfitted raw-row -> numeric-matrix preprocessing pipeline.

    Weight signs and missing flags are handled before imputation. Medians are
    placeholders, not recovered measurements. One-hot categories are learned on
    fitting rows; unseen categories become an all-zero block at inference.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    encode = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                       dtype=np.float32), CATEGORICAL_FEATURES),
    ], remainder="drop", sparse_threshold=0)
    return Pipeline([("features", FreightPreprocessor()), ("encode", encode)])


def regression_report(actual, predicted):
    """Dollar errors plus explicitly defined relative-error percentages."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or not len(actual):
        raise ValueError("Actual and predicted prices must have matching nonempty shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("All measured prices must be finite.")
    if not (actual > 0).all() or not (predicted > 0).all():
        raise ValueError("Relative-error reporting expects positive actual and predicted prices.")
    relative_error = np.abs(predicted - actual) / actual
    return {
        **regression_metrics(actual, predicted),
        "MAPE_percent": float(100 * relative_error.mean()),
        "within_5_percent": float(100 * (relative_error <= 0.05).mean()),
        "within_10_percent": float(100 * (relative_error <= 0.10).mean()),
        "within_20_percent": float(100 * (relative_error <= 0.20).mean()),
    }


def verified_splits(root):
    """Require the exact row assignments from the saved CatBoost comparison."""
    root = Path(root)
    splits = split_development(load_development(root))
    expected = pd.read_csv(root / "outputs" / "catboost" / "split_assignments.csv",
                           parse_dates=["date"])
    observed = pd.concat([part[["load_id", "date"]].assign(split=name)
                          for name, part in splits.items()], ignore_index=True)
    columns = ["load_id", "date", "split"]
    pd.testing.assert_frame_equal(
        expected[columns].sort_values("load_id").reset_index(drop=True),
        observed[columns].sort_values("load_id").reset_index(drop=True),
    )
    return splits
