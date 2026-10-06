"""Refit the frozen CatBoost configuration and fill the validation template.

Run: .venv/Scripts/python.exe predict_validation.py
The earlier evaluation model and the supplied CSVs remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from freight_model import FreightPreprocessor, FreightRatePipeline, load_development, source_hashes
from score import validate_predictions


def main():
    root = Path(__file__).resolve().parent
    original_hashes = source_hashes(root)
    evaluation_dir = root / "outputs" / "catboost"
    evaluation_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evaluation_dir.iterdir() if path.is_file()
    }
    configuration = json.loads((evaluation_dir / "metadata.json").read_text(encoding="utf-8"))
    labeled = load_development(root)
    inputs = pd.read_csv(root / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(root / "validation-predictions-template.csv")
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Unexpected validation template columns.")
    for name, frame in [("validation inputs", inputs), ("template", template)]:
        if frame["load_id"].isna().any() or not frame["load_id"].is_unique:
            raise ValueError(f"Missing or duplicated load_id in {name}.")
    if set(inputs["load_id"]) != set(template["load_id"]):
        raise ValueError("Validation input IDs and template IDs do not match.")
    if inputs["date"].isna().any() or labeled["date"].max() >= inputs["date"].min():
        raise ValueError("Expected final inputs to come after all labeled training dates.")

    # No new tuning or early stopping: reuse the 94-tree configuration as chosen.
    pipeline = FreightRatePipeline([
        ("features", FreightPreprocessor()),
        ("model", CatBoostRegressor(**configuration["parameters"])),
    ])
    print(f"Refitting {configuration['selected_model']} on {len(labeled):,} labeled loads...", flush=True)
    pipeline.fit(labeled, labeled["posted_rate"])
    predictions = pipeline.predict(inputs)
    by_id = pd.Series(predictions, index=inputs["load_id"])
    filled = template.copy()
    filled["predicted_rate"] = filled["load_id"].map(by_id)
    validate_predictions(filled)

    artifact_dir = root / "outputs" / "final_catboost"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / "pipeline.joblib"
    joblib.dump(pipeline, artifact)
    restored = joblib.load(artifact)
    np.testing.assert_allclose(restored.predict(inputs), predictions, rtol=0, atol=1e-10)

    destination = root / "validation-predictions.csv"
    filled.to_csv(destination, index=False, float_format="%.2f")
    saved = pd.read_csv(destination)
    validate_predictions(saved)
    pd.testing.assert_series_equal(saved["load_id"], template["load_id"])
    np.testing.assert_allclose(saved["predicted_rate"], filled["predicted_rate"], rtol=0, atol=0.005000001)
    if source_hashes(root) != original_hashes:
        raise AssertionError("A supplied CSV changed during prediction.")
    for name, digest in evaluation_hashes.items():
        assert hashlib.sha256((evaluation_dir / name).read_bytes()).hexdigest() == digest, name

    metadata = {
        "model": configuration["selected_model"],
        "parameters": configuration["parameters"],
        "training_rows": len(labeled),
        "training_start": labeled["date"].min().date().isoformat(),
        "training_end": labeled["date"].max().date().isoformat(),
        "prediction_rows": len(saved),
        "output": destination.name,
        "output_columns": list(saved.columns),
        "alignment": "Matched by load_id; original template order preserved",
        "price_precision": "USD, two decimal places",
        "source_sha256": original_hashes,
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "catboost_version": catboost.__version__,
        "checks": {"schema_and_ids_valid": True, "all_predictions_finite_positive": True,
                   "model_reload_matches": True, "source_csvs_unchanged": True,
                   "original_evaluation_artifacts_unchanged": True},
        "evaluation_note": "Refitted on all January-October labels. Earlier test scores describe the January-August evaluation model; November-December true prices are unavailable.",
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(saved):,} validated predictions to {destination}", flush=True)
    print(f"Prediction range: ${saved.predicted_rate.min():,.2f} to ${saved.predicted_rate.max():,.2f}", flush=True)
    print("Template order preserved; source CSVs and evaluation artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
