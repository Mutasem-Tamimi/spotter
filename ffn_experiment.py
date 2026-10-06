"""Chronological feed-forward neural-network comparison for freight prices.

Run with ``.venv/Scripts/python.exe ffn_experiment.py`` from the project root.
The MLP is a genuine dense ReLU network, optimized with Adam. Its features and
target scaling are learned exclusively from the current training rows.
"""
from __future__ import annotations

from copy import deepcopy
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from comparison_common import make_preprocessor, regression_report, verified_splits
from freight_model import (
    DEV_END, MIN_RATE, SEED, TRAIN_END, EquipmentRateBaseline,
    FreightRatePipeline, evaluate_slices, prediction_details, source_hashes,
    split_summary,
)


# Locked before inspecting this experiment's September–October predictions.
CANDIDATES = [
    {"name": "FFN 64-32 alpha .001", "hidden_layer_sizes": (64, 32), "alpha": .001},
    {"name": "FFN 128-64 alpha .001", "hidden_layer_sizes": (128, 64), "alpha": .001},
    {"name": "FFN 64-32 alpha .01", "hidden_layer_sizes": (64, 32), "alpha": .01},
]
MAX_EPOCHS = 200
PATIENCE = 25


class ScaledFFNRegressor(RegressorMixin, BaseEstimator):
    """Dense neural network whose public predictions are unscaled dollars.

    ``fit`` performs a fixed number of epochs, suitable for refitting after
    selection. ``fit_with_dev`` explicitly uses the chronological development
    rows for stopping; sklearn's random internal validation split is disabled.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=.001, epochs=100,
                 learning_rate_init=.001, batch_size=256, random_state=SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.epochs = epochs
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.random_state = random_state

    def _initialize(self, X, y):
        target = np.asarray(y, dtype=float).reshape(-1, 1)
        if not np.isfinite(target).all():
            raise ValueError("FFN targets must be finite.")
        if int(self.epochs) < 1:
            raise ValueError("The FFN needs at least one training epoch.")
        self.n_features_in_ = X.shape[1]
        self.target_scaler_ = StandardScaler().fit(target)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu", solver="adam", loss="squared_error",
            alpha=self.alpha, batch_size=min(self.batch_size, len(target)),
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state, shuffle=True, early_stopping=False,
            max_iter=1, tol=0.0, n_iter_no_change=int(self.epochs) + 1,
        )
        return self.target_scaler_.transform(target).ravel()

    def fit(self, X, y):
        target_scaled = self._initialize(X, y)
        self.training_loss_ = []
        for epoch in range(1, int(self.epochs) + 1):
            self.model_.partial_fit(X, target_scaled)
            self.training_loss_.append(float(self.model_.loss_))
        self.best_epoch_ = int(self.epochs)
        self.epochs_run_ = int(self.epochs)
        return self

    def fit_with_dev(self, X, y, X_dev, y_dev, patience=PATIENCE, label="FFN"):
        if patience < 1:
            raise ValueError("Patience must be positive.")
        target_scaled = self._initialize(X, y)
        best_mae, best_model, best_epoch = np.inf, None, 0
        self.history_ = []
        start = perf_counter()
        for epoch in range(1, int(self.epochs) + 1):
            self.model_.partial_fit(X, target_scaled)
            prediction = np.maximum(self.predict(X_dev), MIN_RATE)
            dev_mae = float(np.mean(np.abs(np.asarray(y_dev) - prediction)))
            self.history_.append({"epoch": epoch, "dev_MAE": dev_mae,
                                  "training_scaled_loss": float(self.model_.loss_)})
            if dev_mae < best_mae:
                best_mae, best_epoch, best_model = dev_mae, epoch, deepcopy(self.model_)
            if epoch == 1 or epoch % 10 == 0:
                print(f"{label}: epoch {epoch}, dev MAE=${dev_mae:,.2f}, "
                      f"best=${best_mae:,.2f} at {best_epoch}, "
                      f"elapsed={perf_counter()-start:.1f}s", flush=True)
            if epoch - best_epoch >= patience:
                break
        self.model_ = best_model
        self.best_epoch_ = best_epoch
        self.epochs_run_ = epoch
        self.best_dev_mae_ = best_mae
        return self

    def predict(self, X):
        check_is_fitted(self, ["target_scaler_", "model_"])
        scaled = self.model_.predict(X)
        return self.target_scaler_.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()


def select_on_dev(splits):
    train, dev = splits["train"], splits["dev"]
    preprocessing = make_preprocessor(scale=True)
    X_train = preprocessing.fit_transform(train, train["posted_rate"])
    X_dev = preprocessing.transform(dev)
    rows, models, histories = [], {}, {}
    for candidate in CANDIDATES:
        start = perf_counter()
        regressor = ScaledFFNRegressor(
            hidden_layer_sizes=candidate["hidden_layer_sizes"],
            alpha=candidate["alpha"], epochs=MAX_EPOCHS,
        ).fit_with_dev(X_train, train["posted_rate"], X_dev, dev["posted_rate"],
                       patience=PATIENCE, label=candidate["name"])
        pipeline = FreightRatePipeline([("preprocessing", preprocessing), ("model", regressor)])
        rows.append({"model": candidate["name"], "epochs": regressor.best_epoch_,
                     "epochs_run": regressor.epochs_run_,
                     "seconds": perf_counter() - start,
                     **regression_report(dev["posted_rate"], pipeline.predict(dev))})
        models[candidate["name"]] = pipeline
        histories[candidate["name"]] = regressor.history_
        print(f"Selected checkpoint for {candidate['name']}: epoch {regressor.best_epoch_}, "
              f"dev MAE=${rows[-1]['MAE']:,.2f}", flush=True)
    leaderboard = pd.DataFrame(rows).sort_values("MAE", kind="stable").reset_index(drop=True)
    selected = leaderboard.iloc[0]
    selected_name = selected["model"]
    return {
        "leaderboard": leaderboard, "histories": histories,
        "best_name": selected_name, "best_epochs": int(selected["epochs"]),
        "best_candidate": next(c.copy() for c in CANDIDATES if c["name"] == selected_name),
        "selected_pipeline": models[selected_name],
        "encoded_features": int(X_train.shape[1]),
    }


def run_experiment(root):
    root = Path(root)
    output_dir = root / "outputs" / "ffn"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_hashes = source_hashes(root)
    splits = verified_splits(root)
    print(split_summary(splits).to_string(index=False), flush=True)
    (output_dir / "experiment_plan.json").write_text(json.dumps({
        "candidates": CANDIDATES, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "selection_metric": "July-August MAE in total dollars",
        "target_transform": "Training-only StandardScaler; invert to total dollars",
        "optimizer": "Adam", "loss": "squared_error", "activation": "relu",
        "learning_rate": .001, "batch_size": 256, "seed": SEED,
        "test_use": "September-October evaluated after model and epoch count are locked",
    }, indent=2), encoding="utf-8")
    started = perf_counter()
    with threadpool_limits(limits=2):
        selection = select_on_dev(splits)
        selection["leaderboard"].to_csv(output_dir / "dev_metrics.csv", index=False)
        (output_dir / "learning_curves.json").write_text(
            json.dumps(selection["histories"], indent=2), encoding="utf-8")
        locked = {"selected_candidate": selection["best_candidate"],
                  "selected_epochs": selection["best_epochs"],
                  "selection_metric": "Development MAE", "test_seen_in_this_run": False}
        (output_dir / "selection_locked.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
        train, dev, test = splits["train"], splits["dev"], splits["test"]
        dev_baseline = EquipmentRateBaseline().fit(train, train["posted_rate"])
        dev_details = prediction_details(dev, selection["selected_pipeline"].predict(dev),
                                         dev_baseline.predict(dev), train)
        dev_details.to_csv(output_dir / "dev_predictions.csv", index=False)
        fit_rows = pd.concat([train, dev]).sort_values(["date", "load_id"], kind="stable")
        candidate = selection["best_candidate"]
        final_model = ScaledFFNRegressor(
            hidden_layer_sizes=candidate["hidden_layer_sizes"], alpha=candidate["alpha"],
            epochs=selection["best_epochs"],
        )
        pipeline = FreightRatePipeline([
            ("preprocessing", make_preprocessor(scale=True)), ("model", final_model),
        ])
        print(f"Selection locked: {selection['best_name']}, {selection['best_epochs']} epochs. "
              "Refitting from scratch on January-August.", flush=True)
        pipeline.fit(fit_rows, fit_rows["posted_rate"])
        baseline = EquipmentRateBaseline().fit(fit_rows, fit_rows["posted_rate"])
        prediction = pipeline.predict(test)
        test_metrics = pd.DataFrame([{"model": "FFN", **regression_report(test["posted_rate"], prediction)}])
        details = prediction_details(test, prediction, baseline.predict(test), fit_rows)
        test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
        details.to_csv(output_dir / "test_predictions.csv", index=False)
        evaluate_slices(details).to_csv(output_dir / "slice_metrics.csv", index=False)
        split_summary(splits).to_csv(output_dir / "split_summary.csv", index=False)
        artifact = output_dir / "pipeline.joblib"
        joblib.dump(pipeline, artifact)
        restored = joblib.load(artifact)
        np.testing.assert_allclose(restored.predict(test), prediction, rtol=0, atol=1e-10)
        december = pd.read_csv(root / "december-chart-inputs.csv")
        december_prediction = restored.predict(december)
        assert len(december_prediction) == len(december)
        assert np.isfinite(december_prediction).all() and (december_prediction > 0).all()
        assert source_hashes(root) == original_hashes, "A source CSV changed."
        unbounded = pipeline.named_steps["model"].predict(
            pipeline.named_steps["preprocessing"].transform(test))
        features = pipeline.named_steps["preprocessing"].named_steps["features"].get_feature_names_out().tolist()
        metadata = {
            "selected_model": selection["best_name"], "selected_architecture": candidate["hidden_layer_sizes"],
            "selected_alpha": candidate["alpha"], "selected_epochs": selection["best_epochs"],
            "selection_metric": "July-August MAE in total dollars", "candidates": CANDIDATES,
            "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
            "optimizer": "Adam", "loss": "squared_error", "activation": "relu",
            "learning_rate_init": .001, "batch_size": 256, "seed": SEED,
            "train_end_exclusive": TRAIN_END.date().isoformat(),
            "dev_end_exclusive": DEV_END.date().isoformat(),
            "saved_model_fit_rows": len(fit_rows),
            "saved_model_fit_end": fit_rows["date"].max().date().isoformat(),
            "features": features,
            "encoded_feature_count": int(pipeline.named_steps["preprocessing"].transform(fit_rows.iloc[:1]).shape[1]),
            "excluded_columns": ["load_id", "posted_rate", "predicted_rate", "quote_signal", "market_index"],
            "preprocessing": {
                "negative_weight": "Absolute value in memory, original CSV unchanged; sign flag retained",
                "numerical": "Training-only median imputation then StandardScaler; weight missing flag retained",
                "categorical": "Training-only dense one-hot encoding; unseen categories ignored",
                "coordinates": "Kept separate; training-only city lookup supplies omitted coordinates; an unknown city with absent coordinates receives training-median coordinates, which do not recover its true location",
                "target": "Training-only StandardScaler; inverse transformed to total dollars",
                "target_center": float(final_model.target_scaler_.mean_[0]),
                "target_scale": float(final_model.target_scaler_.scale_[0]),
            },
            "prediction_floor_dollars": MIN_RATE,
            "test_predictions_floored": int(np.sum(unbounded < MIN_RATE)),
            "source_sha256": original_hashes, "elapsed_seconds": perf_counter() - started,
            "checks": {"source_csvs_unchanged": True, "save_reload_prediction_match": True,
                       "december_reduced_schema_rows": len(december_prediction),
                       "december_predictions_finite_positive": True},
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
            "limitations": [
                "September-October had previously been inspected during EDA and CatBoost evaluation; it is a reused local comparison period, not a fresh holdout.",
                "No FFN hyperparameters or epoch counts were selected using September-October targets.",
                "No unseen cities occur in local dev/test; unseen-city accuracy remains unmeasured.",
                "The neural network may extrapolate time features unpredictably beyond training dates.",
                "November-December targets are hidden and no final-file accuracy can be measured.",
                "The saved model was refitted on January-August, not on all labeled rows.",
                "One fixed seed and three bounded candidates were compared; this is not an exhaustive neural-network search.",
            ],
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(test_metrics.round(4).to_string(index=False), flush=True)
    return {"pipeline": pipeline, "selection": selection, "test_metrics": test_metrics,
            "test_predictions": details, "metadata": metadata}


if __name__ == "__main__":
    # Importable module names make the saved custom estimator reloadable later.
    from ffn_experiment import run_experiment as run_imported
    run_imported(Path(__file__).resolve().parent)
