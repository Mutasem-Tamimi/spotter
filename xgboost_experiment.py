"""A fixed, chronological XGBoost comparison against the saved CatBoost cycle.

Run from the project root: .venv/Scripts/python.exe xgboost_experiment.py
Source CSVs and the original CatBoost artifacts are read-only inputs.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBRegressor

from comparison_common import make_preprocessor, regression_report, verified_splits
from freight_model import (
    DEV_END, MIN_RATE, SEED, TRAIN_END, EquipmentRateBaseline,
    FreightRatePipeline, evaluate_slices, prediction_details, source_hashes,
    split_summary,
)


# Declare the search before reading any test outcomes. Do not expand it after
# inspecting September-October; that period has already been used in cycle 1.
CANDIDATES = [
    {"name": "XGBoost depth 4 MAE", "max_depth": 4, "objective": "reg:absoluteerror"},
    {"name": "XGBoost depth 6 MAE", "max_depth": 6, "objective": "reg:absoluteerror"},
    {"name": "XGBoost depth 4 squared error", "max_depth": 4, "objective": "reg:squarederror"},
]
MAX_ROUNDS = 1200
PATIENCE = 100


def model_parameters(candidate, rounds, *, early_stopping=False):
    parameters = {
        "n_estimators": int(rounds),
        "max_depth": int(candidate["max_depth"]),
        "objective": candidate["objective"],
        "learning_rate": 0.05,
        "eval_metric": "mae",
        "tree_method": "hist",
        "n_jobs": 2,
        "random_state": SEED,
        "verbosity": 0,
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = PATIENCE
    return parameters


def select_on_dev(train, dev):
    """Train-only preprocessing, then early stopping on the external dev set."""
    baseline = EquipmentRateBaseline().fit(train, train["posted_rate"])
    baseline_prediction = baseline.predict(dev)
    rows = [{
        "model": "Equipment median baseline", "rounds": 0, "seconds": 0.0,
        **regression_report(dev["posted_rate"], baseline_prediction),
    }]
    fitted, histories = {}, {}
    for candidate in CANDIDATES:
        started = perf_counter()
        preprocessor = make_preprocessor(scale=False)
        X_train = preprocessor.fit_transform(train, train["posted_rate"])
        X_dev = preprocessor.transform(dev)
        if not np.isfinite(X_train).all() or not np.isfinite(X_dev).all():
            raise AssertionError("Cleaned XGBoost input must contain finite numbers.")
        model = XGBRegressor(**model_parameters(candidate, MAX_ROUNDS, early_stopping=True))
        # The last eval_set entry determines early stopping. Test data never
        # enters this function or the estimator's evaluation list.
        model.fit(X_train, train["posted_rate"],
                  eval_set=[(X_train, train["posted_rate"]), (X_dev, dev["posted_rate"])],
                  verbose=False)
        pipeline = FreightRatePipeline([("features", preprocessor), ("model", model)])
        rounds = int(model.best_iteration) + 1
        prediction = pipeline.predict(dev)
        rows.append({
            "model": candidate["name"], "rounds": rounds,
            "seconds": perf_counter() - started,
            **regression_report(dev["posted_rate"], prediction),
        })
        fitted[candidate["name"]] = pipeline
        histories[candidate["name"]] = model.evals_result()
        print(f'{candidate["name"]}: dev MAE=${rows[-1]["MAE"]:,.2f}, '
              f'{rounds} rounds, {rows[-1]["seconds"]:.1f}s', flush=True)
    leaderboard = pd.DataFrame(rows).sort_values("MAE", kind="stable").reset_index(drop=True)
    best_row = leaderboard.loc[leaderboard["model"].isin(fitted)].iloc[0]
    best_name = best_row["model"]
    return {
        "leaderboard": leaderboard, "histories": histories,
        "best_name": best_name, "best_rounds": int(best_row["rounds"]),
        "best_candidate": next(c.copy() for c in CANDIDATES if c["name"] == best_name),
        "dev_pipeline": fitted[best_name], "baseline": baseline,
    }


def refit_and_evaluate(splits, selection):
    """Freeze dev selection, refit Jan-Aug, then evaluate Sep-Oct once."""
    fit_rows = pd.concat([splits["train"], splits["dev"]]).sort_values(["date", "load_id"])
    model = XGBRegressor(**model_parameters(selection["best_candidate"], selection["best_rounds"]))
    pipeline = FreightRatePipeline([
        ("features", make_preprocessor(scale=False)), ("model", model),
    ])
    started = perf_counter()
    pipeline.fit(fit_rows, fit_rows["posted_rate"])
    refit_seconds = perf_counter() - started
    if model.get_booster().num_boosted_rounds() != selection["best_rounds"]:
        raise AssertionError("The final estimator must use the dev-selected number of rounds.")
    baseline = EquipmentRateBaseline().fit(fit_rows, fit_rows["posted_rate"])
    test = splits["test"]
    prediction = pipeline.predict(test)
    baseline_prediction = baseline.predict(test)
    baseline_metrics = regression_report(test["posted_rate"], baseline_prediction)
    metrics = regression_report(test["posted_rate"], prediction)
    metrics["MAE_improvement_percent"] = 100 * (1 - metrics["MAE"] / baseline_metrics["MAE"])
    details = prediction_details(test, prediction, baseline_prediction, fit_rows)
    raw_prediction = model.predict(pipeline.named_steps["features"].transform(test))
    return {
        "pipeline": pipeline, "baseline_metrics": baseline_metrics,
        "test_metrics": pd.DataFrame([{"model": "XGBoost", **metrics}]),
        "test_predictions": details, "slice_metrics": evaluate_slices(details),
        "fit_rows": len(fit_rows), "fit_end": fit_rows["date"].max().date().isoformat(),
        "refit_seconds": refit_seconds,
        "positive_floor_count": int((raw_prediction < MIN_RATE).sum()),
    }


def run_experiment(root):
    root = Path(root).resolve()
    output = root / "outputs" / "xgboost"
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(root)
    splits = verified_splits(root)
    plan = {
        "candidates": CANDIDATES, "max_rounds": MAX_ROUNDS,
        "early_stopping_patience": PATIENCE,
        "selection_metric": "Development MAE in total dollars",
        "train_end_exclusive": TRAIN_END.date().isoformat(),
        "dev_end_exclusive": DEV_END.date().isoformat(),
        "seed": SEED,
        "cleaning": "In-memory absolute weight; training-only numerical medians; dense one-hot categories",
        "test_policy": "September-October evaluated after locking candidate and rounds; no post-test tuning",
    }
    (output / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(split_summary(splits).to_string(index=False), flush=True)
    selection = select_on_dev(splits["train"], splits["dev"])
    # Persist the selection before accessing test labels for any model metric.
    selection["leaderboard"].to_csv(output / "dev_metrics.csv", index=False)
    locked_selection = {
        "selected_model": selection["best_name"],
        "selected_rounds": selection["best_rounds"],
        "parameters": model_parameters(selection["best_candidate"], selection["best_rounds"]),
    }
    (output / "locked_selection.json").write_text(json.dumps(locked_selection, indent=2), encoding="utf-8")
    evaluation = refit_and_evaluate(splits, selection)

    for key in ["test_metrics", "test_predictions", "slice_metrics"]:
        evaluation[key].to_csv(output / f"{key}.csv", index=False)
    dev = splits["dev"]
    dev_details = prediction_details(
        dev, selection["dev_pipeline"].predict(dev),
        selection["baseline"].predict(dev), splits["train"],
    )
    dev_details.to_csv(output / "dev_predictions.csv", index=False)
    (output / "learning_curves.json").write_text(json.dumps(selection["histories"], indent=2), encoding="utf-8")
    split_summary(splits).to_csv(output / "split_summary.csv", index=False)

    pipeline = evaluation["pipeline"]
    joblib.dump(pipeline, output / "pipeline.joblib")
    restored = joblib.load(output / "pipeline.joblib")
    np.testing.assert_allclose(
        restored.predict(splits["test"]), evaluation["test_predictions"]["predicted_rate"],
        rtol=0, atol=1e-10,
    )
    december = pd.read_csv(root / "december-chart-inputs.csv")
    december_before = december.copy(deep=True)
    december_prediction = restored.predict(december)
    assert december_prediction.shape == (len(december),)
    assert np.isfinite(december_prediction).all() and (december_prediction > 0).all()
    pd.testing.assert_frame_equal(december, december_before)
    # Exercise unseen-category behavior without evaluating the hidden targets.
    unknown = december.head(1).copy()
    unknown["pickup"] = "__UNSEEN_CITY_INFERENCE_CHECK__"
    unknown["equipment"] = "__UNSEEN_EQUIPMENT_INFERENCE_CHECK__"
    assert np.isfinite(restored.predict(unknown)).all()
    if source_hashes(root) != hashes:
        raise AssertionError("A source CSV changed during the XGBoost run.")

    preprocessor = pipeline.named_steps["features"]
    transformed_names = preprocessor.get_feature_names_out().tolist()
    raw_features = preprocessor.steps[0][1].get_feature_names_out().tolist()
    importance = pd.DataFrame({
        "feature": transformed_names,
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    metadata = {
        **plan, **locked_selection,
        "saved_model_fit_rows": evaluation["fit_rows"],
        "saved_model_fit_end": evaluation["fit_end"],
        "refit_seconds": evaluation["refit_seconds"],
        "features": raw_features, "encoded_feature_count": len(transformed_names),
        "excluded_columns": ["load_id", "posted_rate", "predicted_rate", "quote_signal", "market_index"],
        "prediction_floor_dollars": MIN_RATE,
        "test_predictions_floored": evaluation["positive_floor_count"],
        "baseline_test_metrics": evaluation["baseline_metrics"],
        "source_sha256": hashes,
        "versions": {
            "python": platform.python_version(), "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "joblib": joblib.__version__,
        },
        "checks": {
            "split_manifest_matches_catboost": True,
            "saved_pipeline_reload_predictions_match": True,
            "december_reduced_schema_predicts_finite_positive_rates": True,
            "unseen_categories_predict_without_error": True,
            "inference_does_not_mutate_input": True,
            "source_csv_hashes_unchanged": True,
        },
        "limitations": [
            "September-October was already inspected during the CatBoost cycle; it is a reused benchmark, not a fresh untouched holdout.",
            "No test metric was used for candidate selection, early stopping, or post-test tuning in this run.",
            "One-hot encoding maps unseen categories to all-zero blocks; their remaining numerical features still contribute.",
            "No unseen cities occur in local dev/test, so unseen-city prediction accuracy is unmeasured.",
            "November-December targets are hidden; final-file accuracy is unknown.",
            "Saved pipeline is fitted through August only; no submission or final-data refit was performed.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(evaluation["test_metrics"].round(4).to_string(index=False), flush=True)
    print(f"Saved XGBoost experiment to {output}", flush=True)
    return {"output_dir": output, "metadata": metadata,
            "dev_metrics": selection["leaderboard"], "test_metrics": evaluation["test_metrics"]}


if __name__ == "__main__":
    from xgboost_experiment import run_experiment as run_imported
    run_imported(Path(__file__).resolve().parent)
