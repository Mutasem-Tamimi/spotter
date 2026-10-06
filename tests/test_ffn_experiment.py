"""Focused checks for neural-network target scaling and temporal stopping."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from comparison_common import make_preprocessor
from ffn_experiment import ScaledFFNRegressor
from freight_model import FreightRatePipeline


class FFNTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(32, 4))
        self.y = 1500 + 100 * self.X[:, 0]

    def test_dev_targets_do_not_change_training_target_scaler(self):
        model = ScaledFFNRegressor(hidden_layer_sizes=(8,), epochs=3, batch_size=16)
        with threadpool_limits(limits=1):
            model.fit_with_dev(self.X, self.y, self.X[:8], np.full(8, 99999.), patience=2)
        self.assertAlmostEqual(model.target_scaler_.mean_[0], self.y.mean())
        self.assertAlmostEqual(model.target_scaler_.scale_[0], self.y.std())
        self.assertFalse(model.model_.early_stopping)

    def test_best_chronological_checkpoint_is_restored(self):
        model = ScaledFFNRegressor(hidden_layer_sizes=(8,), epochs=6, batch_size=16)
        with threadpool_limits(limits=1):
            model.fit_with_dev(self.X, self.y, self.X[:8], self.y[:8], patience=2)
            measured = np.mean(np.abs(np.maximum(model.predict(self.X[:8]), .01) - self.y[:8]))
        best = min(model.history_, key=lambda row: row["dev_MAE"])
        self.assertEqual(model.best_epoch_, best["epoch"])
        self.assertAlmostEqual(measured, best["dev_MAE"])

    def test_raw_row_pipeline_serialization_and_reduced_schema(self):
        rows = pd.DataFrame({
            "pickup": ["A", "B"] * 16, "delivery": ["B", "A"] * 16,
            "equipment": ["Dry Van"] * 32, "distance": np.arange(32) + 100.,
            "weight": [None, -30000., 31000., 32000.] * 8,
            "date": pd.date_range("2025-01-01", periods=32),
            "pickup_lat": [30., 40.] * 16, "pickup_lon": [-80., -90.] * 16,
            "delivery_lat": [40., 30.] * 16, "delivery_lon": [-90., -80.] * 16,
        })
        original = rows.copy(deep=True)
        pipeline = FreightRatePipeline([
            ("preprocessing", make_preprocessor(scale=True)),
            ("model", ScaledFFNRegressor(hidden_layer_sizes=(8,), epochs=2, batch_size=16)),
        ])
        reduced = rows.drop(columns=["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as directory:
            pipeline.fit(rows, self.y)
            before = pipeline.predict(reduced)
            path = Path(directory) / "model.joblib"
            joblib.dump(pipeline, path)
            after = joblib.load(path).predict(reduced)
        np.testing.assert_allclose(before, after, rtol=0, atol=0)
        self.assertTrue(np.isfinite(after).all())
        self.assertTrue((after > 0).all())
        pd.testing.assert_frame_equal(rows, original)


if __name__ == "__main__":
    unittest.main()
