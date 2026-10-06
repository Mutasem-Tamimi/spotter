"""Generate submission CSVs and the official December chart without retraining.

Run after notebook 04 or predict_validation.py: python complete_submission.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from freight_model import source_hashes
from score import validate_december, validate_predictions


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_predictions(root):
    """Use the same final model for both input schemas; preserve supplied files."""
    root = Path(root).resolve()
    original_hashes = source_hashes(root)
    protected_readme = root / "readme-spotter.md"
    readme_hash = sha256(protected_readme) if protected_readme.exists() else None
    artifact = root / "outputs/final_catboost/pipeline.joblib"
    metadata = json.loads((artifact.parent / "metadata.json").read_text(encoding="utf-8"))
    if metadata["source_sha256"] != original_hashes:
        raise ValueError("Inputs differ from those used to fit the final model.")
    if metadata["training_rows"] != 48_000 or metadata["training_end"] != "2025-10-31":
        raise ValueError("Expected the final model fitted on all January-October labels.")
    # This is our own trusted artifact; loading a joblib file executes Python code.
    pipeline = joblib.load(artifact)
    validation = pd.read_csv(root / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(root / "validation-predictions-template.csv")
    if not validation.load_id.is_unique or not template.load_id.is_unique:
        raise ValueError("Expected unique load IDs.")
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("Template IDs differ from validation inputs.")
    values = pd.Series(pipeline.predict(validation), index=validation.load_id)
    filled = template.copy()
    filled["predicted_rate"] = filled.load_id.map(values)
    validate_predictions(filled)
    validation_path = root / "validation_predictions.csv"
    filled.to_csv(validation_path, index=False, float_format="%.2f")
    saved_validation = pd.read_csv(validation_path)
    validate_predictions(saved_validation)
    pd.testing.assert_series_equal(saved_validation.load_id, template.load_id)
    np.testing.assert_allclose(saved_validation.predicted_rate, filled.predicted_rate,
                               rtol=0, atol=.005000001)
    previous = root / "validation-predictions.csv"
    if previous.exists() and sha256(previous) != sha256(validation_path):
        raise AssertionError("Submission copy differs from the earlier final predictions.")

    raw_december = pd.read_csv(root / "december-chart-inputs.csv")
    completed_december = raw_december.copy()
    completed_december["predicted_rate"] = pipeline.predict(raw_december)
    validate_december(completed_december)
    december_path = root / "december_predictions.csv"
    completed_december.to_csv(december_path, index=False, float_format="%.2f")
    saved_december = pd.read_csv(december_path)
    validate_december(saved_december)
    pd.testing.assert_frame_equal(saved_december.drop(columns="predicted_rate"),
                                  raw_december.drop(columns="predicted_rate"),
                                  check_dtype=False)
    np.testing.assert_allclose(saved_december.predicted_rate, completed_december.predicted_rate,
                               rtol=0, atol=.005000001)
    prepared = pipeline.named_steps["features"].transform(raw_december)
    coordinate_columns = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    if prepared[coordinate_columns].isna().any().any():
        raise AssertionError("The fixed December cities should be in the fitted coordinate lookup.")
    assert source_hashes(root) == original_hashes
    if readme_hash is not None:
        assert sha256(protected_readme) == readme_hash
    result = {
        "model": "CatBoost depth 6, MAE loss, 94 trees, seed 42",
        "training_rows": metadata["training_rows"],
        "training_end": metadata["training_end"],
        "model_sha256": sha256(artifact),
        "source_sha256": original_hashes,
        "readme_spotter_sha256": readme_hash,
        "validation_rows": len(saved_validation),
        "december_rows": len(saved_december),
        "december_min": float(saved_december.predicted_rate.min()),
        "december_max": float(saved_december.predicted_rate.max()),
        "december_unique_prices": int(saved_december.predicted_rate.nunique()),
        "december_coordinates": prepared[coordinate_columns].iloc[0].to_dict(),
        "output_sha256": {p.name: sha256(p) for p in [validation_path, december_path]},
        "checks": {"schema_ids_and_values_valid": True, "template_order_preserved": True,
                   "december_input_values_preserved": True, "original_csvs_unchanged": True,
                   "readme_spotter_unchanged": True},
        "evaluation_note": "Hidden final labels are unavailable; the scorer validates format and draws a chart, not accuracy.",
    }
    return saved_validation, saved_december, result


def run_scorer(root, checks):
    """Run the unmodified provided scorer and retain its stdout as evidence."""
    root = Path(root).resolve()
    command = [sys.executable, "score.py", "--predictions", "validation_predictions.csv",
               "--december-predictions", "december_predictions.csv"]
    result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    output = root / "scorer_results"
    (output / "scorer_output.txt").write_text(result.stdout, encoding="utf-8")
    chart = output / "candidate_december.png"
    if not chart.is_file() or chart.stat().st_size == 0:
        raise AssertionError("The scorer did not produce the required chart.")
    checks["checks"]["official_scorer_passed"] = True
    checks["output_sha256"]["scorer_results/candidate_december.png"] = sha256(chart)
    artifact_dir = root / "outputs/submission"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "submission_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return result.stdout


def main():
    root = Path(__file__).resolve().parent
    _, december, checks = prepare_predictions(root)
    print(run_scorer(root, checks), end="")
    print(f"December range: ${december.predicted_rate.min():,.2f} to ${december.predicted_rate.max():,.2f}")
    print("Source CSVs, previous predictions, final model, and readme-spotter.md were preserved.")


if __name__ == "__main__":
    main()
