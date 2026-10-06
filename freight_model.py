"""Chronological CatBoost training and inference for the freight-rate project.

Run from the project root: .venv/Scripts/python.exe freight_model.py
The same functions are used by notebooks/02_catboost_pipeline.ipynb.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


SEED = 42
TRAIN_END = pd.Timestamp("2025-07-01")
DEV_END = pd.Timestamp("2025-09-01")
MIN_RATE = 0.01
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
CORE_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
SOURCE_FILES = ["train-test.csv", "validation.csv", "december-chart-inputs.csv",
                "validation-predictions-template.csv"]
CANDIDATES = [
    {"name": "CatBoost depth 6 MAE", "depth": 6, "loss_function": "MAE"},
    {"name": "CatBoost depth 8 MAE", "depth": 8, "loss_function": "MAE"},
    {"name": "CatBoost depth 6 RMSE", "depth": 6, "loss_function": "RMSE"},
]


def source_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def load_development(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "train-test.csv", parse_dates=["date"])
    if frame["load_id"].isna().any() or not frame["load_id"].is_unique:
        raise ValueError("Development load IDs must be present and unique.")
    target = pd.to_numeric(frame["posted_rate"], errors="raise")
    if not np.isfinite(target).all() or not target.gt(0).all():
        raise ValueError("Training prices must be finite, positive, and observed.")
    return frame.sort_values(["date", "load_id"], kind="stable").reset_index(drop=True)


def split_development(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split raw rows before fitting any feature preparation or model."""
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("Dates cannot be missing when splitting chronologically.")
    splits = {
        "train": frame.loc[dates.lt(TRAIN_END)].copy(),
        "dev": frame.loc[dates.ge(TRAIN_END) & dates.lt(DEV_END)].copy(),
        "test": frame.loc[dates.ge(DEV_END)].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError("The fixed chronological boundaries produced an empty split.")
    ids = [set(part["load_id"]) for part in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, splits.values())) == len(frame)
    assert splits["train"]["date"].max() < splits["dev"]["date"].min()
    assert splits["dev"]["date"].max() < splits["test"]["date"].min()
    return splits


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = sum(map(len, splits.values()))
    return pd.DataFrame([
        {"split": name, "rows": len(part), "percent": 100 * len(part) / total,
         "first_date": part["date"].min().date().isoformat(),
         "last_date": part["date"].max().date().isoformat(),
         "missing_weight": int(part["weight"].isna().sum()),
         "negative_weight": int(part["weight"].lt(0).sum())}
        for name, part in splits.items()
    ])


class FreightPreprocessor(TransformerMixin, BaseEstimator):
    """Copy raw inputs, correct weight signs, and use a training-only city lookup.

    Numerical NaNs remain NaNs. No target encoding, scaling, or median imputation
    is computed outside CatBoost. Extra columns (including targets) are ignored.
    """

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Supply a pandas DataFrame with named raw columns.")
        missing = set(CORE_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Required input columns are missing: {sorted(missing)}")

    def fit(self, X, y=None):
        self._validate(X)
        dates = pd.to_datetime(X["date"], errors="raise")
        if dates.isna().any():
            raise ValueError("Dates must be present.")
        self.date_origin_ = dates.min().normalize()
        locations = []
        for end in ["pickup", "delivery"]:
            columns = [end, f"{end}_lat", f"{end}_lon"]
            if all(column in X for column in columns):
                locations.append(X[columns].rename(columns={
                    end: "city", f"{end}_lat": "lat", f"{end}_lon": "lon"}))
        self.city_lookup_ = {}
        if locations:
            cities = pd.concat(locations, ignore_index=True).dropna(subset=["city"])
            cities["city"] = cities["city"].astype(str)
            for coordinate in ["lat", "lon"]:
                cities[coordinate] = pd.to_numeric(cities[coordinate], errors="raise")
            counts = cities.groupby("city")[["lat", "lon"]].nunique()
            if counts.gt(1).any().any():
                raise ValueError("A city has conflicting coordinates; resolve the lookup first.")
            self.city_lookup_ = cities.groupby("city")[["lat", "lon"]].first().to_dict("index")
        self.feature_names_out_ = np.array([
            *CATEGORICAL_FEATURES, "distance", "weight", "weight_is_missing",
            "weight_was_negative", *COORDINATES, "days_since_start",
            "day_of_week", "day_of_month",
        ], dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        self._validate(X)
        out = pd.DataFrame(index=X.index)
        for column in CATEGORICAL_FEATURES:
            out[column] = X[column].astype("string").fillna("__MISSING__").astype(object)
        out["distance"] = pd.to_numeric(X["distance"], errors="raise").astype(float)
        if not np.isfinite(out["distance"]).all() or not out["distance"].gt(0).all():
            raise ValueError("Distance must be observed, finite, and positive.")
        weight = pd.to_numeric(X["weight"], errors="raise").astype(float)
        out["weight"] = weight.abs()
        out["weight_is_missing"] = weight.isna().astype(int)
        negative = weight.lt(0)
        if "weight_was_negative" in X:
            negative = negative | X["weight_was_negative"].fillna(False).astype(bool)
        out["weight_was_negative"] = negative.astype(int)
        for end in ["pickup", "delivery"]:
            for coordinate, limit in [("lat", 90), ("lon", 180)]:
                column = f"{end}_{coordinate}"
                supplied = (pd.to_numeric(X[column], errors="raise").astype(float)
                            if column in X else pd.Series(np.nan, index=X.index))
                lookup = {city: value[coordinate] for city, value in self.city_lookup_.items()}
                out[column] = supplied.fillna(out[end].map(lookup)).astype(float)
                if out[column].abs().gt(limit).any():
                    raise ValueError(f"{column} is outside its geographic range.")
        dates = pd.to_datetime(X["date"], errors="raise")
        if dates.isna().any():
            raise ValueError("Dates must be present.")
        out["days_since_start"] = (dates - self.date_origin_).dt.days
        out["day_of_week"] = dates.dt.dayofweek
        out["day_of_month"] = dates.dt.day
        numeric = out.drop(columns=CATEGORICAL_FEATURES)
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError("Infinite feature values are unsupported.")
        return out.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class FreightRatePipeline(Pipeline):
    """Complete raw-row inference, including the fixed positive-output rule."""

    def predict(self, X, **params):
        prediction = np.asarray(super().predict(X, **params), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("The model produced a nonfinite prediction.")
        return np.maximum(prediction, MIN_RATE)


class EquipmentRateBaseline:
    """Training median price/mile for each equipment type, times load distance."""

    def fit(self, X, y):
        rates = pd.DataFrame({"equipment": X["equipment"].to_numpy(),
                              "rate": np.asarray(y) / X["distance"].to_numpy()})
        self.equipment_rates_ = rates.groupby("equipment")["rate"].median().to_dict()
        self.fallback_ = float(rates["rate"].median())
        return self

    def predict(self, X):
        per_mile = X["equipment"].map(self.equipment_rates_).fillna(self.fallback_)
        return np.maximum(per_mile.to_numpy() * X["distance"].to_numpy(), MIN_RATE)


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
            "bias": float(np.mean(predicted - actual)),
            "median_absolute_error": float(np.median(np.abs(predicted - actual)))}


def model_parameters(candidate, iterations):
    return dict(iterations=int(iterations), depth=int(candidate["depth"]),
                loss_function=candidate["loss_function"], eval_metric="MAE",
                custom_metric=["MAE"], learning_rate=0.05, l2_leaf_reg=3,
                random_seed=SEED, thread_count=min(4, os.cpu_count() or 1),
                has_time=True, nan_mode="Min", allow_writing_files=False,
                cat_features=CATEGORICAL_FEATURES, verbose=False)


def select_on_dev(splits, max_iterations=1200, patience=100):
    """Use train and dev ONLY; test is neither read nor used for early stopping."""
    train, dev = splits["train"], splits["dev"]
    baseline = EquipmentRateBaseline().fit(train, train["posted_rate"])
    rows = [{"model": "Equipment median baseline", "trees": 0, "seconds": 0.0,
             **regression_metrics(dev["posted_rate"], baseline.predict(dev))}]
    pipelines, histories = {}, {}
    for candidate in CANDIDATES:
        start = perf_counter()
        features = FreightPreprocessor().fit(train)
        X_train, X_dev = features.transform(train), features.transform(dev)
        model = CatBoostRegressor(**model_parameters(candidate, max_iterations))
        model.fit(X_train, train["posted_rate"], eval_set=(X_dev, dev["posted_rate"]),
                  early_stopping_rounds=patience, use_best_model=True)
        pipeline = FreightRatePipeline([("features", features), ("model", model)])
        prediction = pipeline.predict(dev)
        rows.append({"model": candidate["name"], "trees": model.tree_count_,
                     "seconds": perf_counter() - start,
                     **regression_metrics(dev["posted_rate"], prediction)})
        pipelines[candidate["name"]] = pipeline
        histories[candidate["name"]] = model.get_evals_result()
        print(f'{candidate["name"]}: dev MAE=${rows[-1]["MAE"]:,.2f}, '
              f'{model.tree_count_} trees, {rows[-1]["seconds"]:.1f}s', flush=True)
    leaderboard = pd.DataFrame(rows).sort_values("MAE", kind="stable").reset_index(drop=True)
    best_row = leaderboard.loc[leaderboard["model"].isin(pipelines)].iloc[0]
    best_name = best_row["model"]
    return {"leaderboard": leaderboard, "pipelines": pipelines, "histories": histories,
            "best_name": best_name, "best_trees": int(best_row["trees"]),
            "best_candidate": next(c.copy() for c in CANDIDATES if c["name"] == best_name),
            "baseline": baseline, "max_iterations": max_iterations, "patience": patience}


def prediction_details(frame, prediction, baseline_prediction, fitted_rows):
    columns = ["load_id", "date", "pickup", "delivery", "equipment", "distance", "posted_rate"]
    result = frame[columns].copy()
    result["predicted_rate"] = prediction
    result["baseline_prediction"] = baseline_prediction
    result["error"] = result["predicted_rate"] - result["posted_rate"]
    result["absolute_error"] = result["error"].abs()
    result["weight_is_missing"] = frame["weight"].isna()
    result["weight_was_negative"] = frame["weight"].lt(0)
    cities = set(fitted_rows["pickup"]) | set(fitted_rows["delivery"])
    lanes = set(zip(fitted_rows["pickup"], fitted_rows["delivery"]))
    result["unseen_city"] = ~frame["pickup"].isin(cities) | ~frame["delivery"].isin(cities)
    result["unseen_lane"] = [pair not in lanes for pair in zip(frame["pickup"], frame["delivery"])]
    return result


def evaluate_slices(details):
    slices = {
        "month": pd.to_datetime(details["date"]).dt.strftime("%Y-%m"),
        "equipment": details["equipment"],
        "distance_miles": pd.cut(details["distance"], [0, 250, 750, 1500, np.inf],
                                 labels=["0-250", "250-750", "750-1500", "1500+"], right=False),
        "missing_weight": details["weight_is_missing"].map({True: "missing", False: "observed"}),
        "negative_weight": details["weight_was_negative"].map({True: "corrected", False: "other"}),
        "city_status": details["unseen_city"].map({True: "unseen", False: "seen"}),
        "lane_status": details["unseen_lane"].map({True: "unseen", False: "seen"}),
    }
    rows = []
    for dimension, groups in slices.items():
        for label, part in details.groupby(groups, observed=True, sort=True):
            model_scores = regression_metrics(part["posted_rate"], part["predicted_rate"])
            baseline_mae = mean_absolute_error(part["posted_rate"], part["baseline_prediction"])
            rows.append({"dimension": dimension, "group": str(label), "rows": len(part),
                         **model_scores, "baseline_MAE": float(baseline_mae),
                         "MAE_improvement_percent": 100 * (1 - model_scores["MAE"] / baseline_mae)})
    return pd.DataFrame(rows)


def refit_and_evaluate(splits, selection):
    """Freeze selection, fit on Jan-Aug, and evaluate Sep-Oct without an eval_set."""
    fit_rows = pd.concat([splits["train"], splits["dev"]]).sort_values(["date", "load_id"])
    test = splits["test"]
    model = CatBoostRegressor(**model_parameters(selection["best_candidate"], selection["best_trees"]))
    pipeline = FreightRatePipeline([("features", FreightPreprocessor()), ("model", model)])
    pipeline.fit(fit_rows, fit_rows["posted_rate"])
    baseline = EquipmentRateBaseline().fit(fit_rows, fit_rows["posted_rate"])
    prediction, baseline_prediction = pipeline.predict(test), baseline.predict(test)
    scores = pd.DataFrame([
        {"model": "Equipment median baseline", **regression_metrics(test["posted_rate"], baseline_prediction)},
        {"model": selection["best_name"], **regression_metrics(test["posted_rate"], prediction)},
    ])
    baseline_mae = float(scores.iloc[0]["MAE"])
    scores["MAE_improvement_percent"] = 100 * (1 - scores["MAE"] / baseline_mae)
    details = prediction_details(test, prediction, baseline_prediction, fit_rows)
    importance = pd.DataFrame({"feature": pipeline.named_steps["features"].get_feature_names_out(),
                               "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    return {"pipeline": pipeline, "baseline": baseline, "test_metrics": scores,
            "test_predictions": details, "slice_metrics": evaluate_slices(details),
            "feature_importance": importance, "fit_rows": len(fit_rows),
            "fit_end": fit_rows["date"].max().date().isoformat(),
            "positive_floor_count": int((model.predict(pipeline.named_steps["features"].transform(test)) < MIN_RATE).sum())}


def save_results(root, output_dir, splits, selection, evaluation, original_hashes):
    """Persist the measured model, its exact split, and reproducible evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_hashes(root) != original_hashes:
        raise AssertionError("A source CSV changed during this run.")
    split_summary(splits).to_csv(output_dir / "split_summary.csv", index=False)
    pd.concat([part[["load_id", "date"]].assign(split=name)
               for name, part in splits.items()]).to_csv(output_dir / "split_assignments.csv", index=False)
    selection["leaderboard"].to_csv(output_dir / "dev_metrics.csv", index=False)
    for key in ["test_metrics", "test_predictions", "slice_metrics", "feature_importance"]:
        evaluation[key].to_csv(output_dir / f"{key}.csv", index=False)
    selected_dev = selection["pipelines"][selection["best_name"]]
    dev_details = prediction_details(splits["dev"], selected_dev.predict(splits["dev"]),
                                    selection["baseline"].predict(splits["dev"]), splits["train"])
    dev_details.to_csv(output_dir / "dev_predictions.csv", index=False)
    (output_dir / "learning_curves.json").write_text(json.dumps(selection["histories"], indent=2), encoding="utf-8")
    artifact = output_dir / "catboost_pipeline.joblib"
    joblib.dump(evaluation["pipeline"], artifact)
    restored = joblib.load(artifact)
    np.testing.assert_allclose(restored.predict(splits["test"]),
                               evaluation["test_predictions"]["predicted_rate"].to_numpy(), rtol=0, atol=1e-10)
    metadata = {
        "selected_model": selection["best_name"], "selected_trees": selection["best_trees"],
        "selection_metric": "Development MAE in total dollars", "candidates": CANDIDATES,
        "max_iterations": selection["max_iterations"], "early_stopping_patience": selection["patience"],
        "parameters": model_parameters(selection["best_candidate"], selection["best_trees"]),
        "train_end_exclusive": TRAIN_END.date().isoformat(),
        "dev_end_exclusive": DEV_END.date().isoformat(),
        "saved_model_fit_rows": evaluation["fit_rows"], "saved_model_fit_end": evaluation["fit_end"],
        "features": evaluation["pipeline"].named_steps["features"].get_feature_names_out().tolist(),
        "excluded_columns": ["load_id", "posted_rate", "predicted_rate", "quote_signal", "market_index"],
        "negative_weights": "absolute value in memory; source CSVs unchanged",
        "missing_numerical_values": "NaN retained; explicit weight missing indicator",
        "prediction_floor_dollars": MIN_RATE, "test_predictions_floored": evaluation["positive_floor_count"],
        "source_sha256": original_hashes,
        "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
        "limitations": [
            "September-October was held out from model selection, but was previously seen during EDA.",
            "No unseen cities occur in the local dev/test periods; unseen-city accuracy is unmeasured.",
            "Ordinal date trees cannot extrapolate trends beyond the fitted time range.",
            "November-December targets are hidden; no final-file accuracy can be reported.",
            "The saved model was refitted on January-August, not on all labeled rows.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_training(root, output_dir=None):
    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root / "outputs" / "catboost"
    hashes = source_hashes(root)
    splits = split_development(load_development(root))
    print(split_summary(splits).to_string(index=False), flush=True)
    selection = select_on_dev(splits)
    evaluation = refit_and_evaluate(splits, selection)
    save_results(root, output_dir, splits, selection, evaluation, hashes)
    print(evaluation["test_metrics"].round(3).to_string(index=False), flush=True)
    return splits, selection, evaluation


if __name__ == "__main__":
    # Import by module name so serialized classes remain loadable from notebooks.
    from freight_model import run_training as run_imported
    run_imported(Path(__file__).resolve().parent)
