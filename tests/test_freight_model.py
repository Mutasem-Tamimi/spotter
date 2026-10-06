"""Focused checks for leakage-safe freight preprocessing and saved inference.

All fixtures are synthetic: these tests do not inspect the real holdout targets.
Run from the project root: python -m unittest discover -s tests -v
"""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pandas.testing import assert_frame_equal

from freight_model import (
    CATEGORICAL_FEATURES,
    COORDINATES,
    CORE_COLUMNS,
    EquipmentRateBaseline,
    FreightPreprocessor,
    FreightRatePipeline,
    MIN_RATE,
    model_parameters,
    split_development,
)


def synthetic_rows():
    dates = pd.to_datetime([
        "2025-01-01", "2025-02-10", "2025-03-10", "2025-04-10",
        "2025-05-10", "2025-06-30", "2025-07-01", "2025-07-20",
        "2025-08-10", "2025-08-31", "2025-09-01", "2025-10-31",
    ])
    count = len(dates)
    return pd.DataFrame({
        "load_id": [f"SYN-{i:03d}" for i in range(count)],
        "pickup": ["Alpha", "Beta"] * 6,
        "delivery": ["Beta", "Alpha"] * 6,
        "pickup_lat": [40.0, 42.0] * 6,
        "pickup_lon": [-80.0, -82.0] * 6,
        "delivery_lat": [42.0, 40.0] * 6,
        "delivery_lon": [-82.0, -80.0] * 6,
        "distance": np.arange(count) * 30.0 + 100.0,
        "equipment": ["Dry Van", "Reefer", "Flatbed"] * 4,
        "weight": [-20000.0, np.nan, 31000.0, 25000.0] * 3,
        "date": dates,
        "market_index": np.linspace(0.9, 1.1, count),
        "quote_signal": np.linspace(2.0, 3.0, count),
        "posted_rate": np.arange(count) * 75.0 + 300.0,
        "predicted_rate": np.nan,
    })


class SplitTests(unittest.TestCase):
    def test_exact_boundaries_and_complete_disjoint_coverage(self):
        source = synthetic_rows().sample(frac=1, random_state=7)
        before = source.copy(deep=True)
        splits = split_development(source)
        self.assertEqual([len(splits[name]) for name in ("train", "dev", "test")], [6, 4, 2])
        self.assertEqual(set(splits["train"].load_id), {f"SYN-{i:03d}" for i in range(6)})
        self.assertEqual(set(splits["dev"].load_id), {f"SYN-{i:03d}" for i in range(6, 10)})
        self.assertEqual(set(splits["test"].load_id), {"SYN-010", "SYN-011"})
        all_ids = pd.concat(splits.values()).load_id
        self.assertTrue(all_ids.is_unique)
        self.assertEqual(set(all_ids), set(source.load_id))
        assert_frame_equal(source, before)

    def test_missing_date_and_empty_period_are_rejected(self):
        source = synthetic_rows()
        source.loc[0, "date"] = pd.NaT
        with self.assertRaises(ValueError):
            split_development(source)
        with self.assertRaises(ValueError):
            split_development(synthetic_rows().iloc[:6])


class PreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.train = synthetic_rows().iloc[:6].copy()
        self.features = FreightPreprocessor().fit(self.train)

    def test_weight_correction_retains_nulls_and_does_not_mutate_raw_rows(self):
        before = self.train.copy(deep=True)
        transformed = self.features.transform(self.train)
        self.assertEqual(transformed.loc[0, "weight"], 20000.0)
        self.assertEqual(transformed.loc[0, "weight_was_negative"], 1)
        self.assertTrue(pd.isna(transformed.loc[1, "weight"]))
        self.assertEqual(transformed.loc[1, "weight_is_missing"], 1)
        np.testing.assert_array_equal(transformed.weight.isna(), self.train.weight.isna())
        assert_frame_equal(self.train, before)
        already_cleaned = self.train.copy()
        already_cleaned["weight_was_negative"] = already_cleaned.weight.lt(0)
        already_cleaned["weight"] = already_cleaned.weight.abs()
        assert_frame_equal(transformed, self.features.transform(already_cleaned))

    def test_targets_ids_and_excluded_signals_cannot_change_features(self):
        original = self.features.transform(self.train)
        modified = self.train.copy()
        modified["posted_rate"] = -999999.0
        modified["predicted_rate"] = 999999.0
        modified["quote_signal"] = 999999.0
        modified["market_index"] = 999999.0
        modified["load_id"] = "CHANGED"
        assert_frame_equal(original, self.features.transform(modified))
        differently_fitted = FreightPreprocessor().fit(modified, np.zeros(len(modified)))
        assert_frame_equal(original, differently_fitted.transform(self.train))
        self.assertEqual(self.features.city_lookup_, differently_fitted.city_lookup_)
        self.assertTrue(set(original.columns).isdisjoint({
            "load_id", "posted_rate", "predicted_rate", "quote_signal", "market_index",
        }))

    def test_missing_coordinate_columns_use_training_lookup_in_both_roles(self):
        raw = self.train.loc[:, CORE_COLUMNS].iloc[[0, 1]].copy()
        transformed = self.features.transform(raw)
        np.testing.assert_array_equal(transformed.pickup_lat, [40.0, 42.0])
        np.testing.assert_array_equal(transformed.pickup_lon, [-80.0, -82.0])
        np.testing.assert_array_equal(transformed.delivery_lat, [42.0, 40.0])
        np.testing.assert_array_equal(transformed.delivery_lon, [-82.0, -80.0])

    def test_lookup_fills_missing_cells_and_preserves_provided_coordinates(self):
        raw = self.train.iloc[[0]].copy()
        raw.loc[0, "pickup_lat"] = 41.25
        raw.loc[0, "pickup_lon"] = np.nan
        raw.loc[0, "delivery_lat"] = np.nan
        transformed = self.features.transform(raw)
        self.assertEqual(transformed.loc[0, "pickup_lat"], 41.25)
        self.assertEqual(transformed.loc[0, "pickup_lon"], -80.0)
        self.assertEqual(transformed.loc[0, "delivery_lat"], 42.0)
        self.assertEqual(transformed.loc[0, "delivery_lon"], -82.0)

    def test_unknown_cities_remain_unknown_without_learning_from_inference(self):
        raw = self.train.loc[:, CORE_COLUMNS].iloc[[0]].copy()
        raw.loc[0, "pickup"] = "Unseen"
        transformed = self.features.transform(raw)
        self.assertTrue(pd.isna(transformed.loc[0, "pickup_lat"]))
        self.assertTrue(pd.isna(transformed.loc[0, "pickup_lon"]))
        self.assertEqual(transformed.loc[0, "delivery_lat"], 42.0)
        supplied = raw.assign(pickup_lat=45.0, pickup_lon=-85.0)
        provided_result = self.features.transform(supplied)
        self.assertEqual(provided_result.loc[0, "pickup_lat"], 45.0)
        self.assertEqual(provided_result.loc[0, "pickup_lon"], -85.0)
        self.assertNotIn("Unseen", self.features.city_lookup_)
        self.assertTrue(self.features.transform(raw)[["pickup_lat", "pickup_lon"]].isna().all().all())

    def test_calendar_origin_is_fitted_only_on_training_rows(self):
        future = synthetic_rows().iloc[-1:].copy()
        origin = self.features.date_origin_
        transformed = self.features.transform(future)
        self.assertEqual(origin, pd.Timestamp("2025-01-01"))
        self.assertEqual(transformed.iloc[0].days_since_start, 303)
        self.assertEqual(self.features.date_origin_, origin)

    def test_conflicting_training_coordinates_are_rejected(self):
        inconsistent = self.train.copy()
        inconsistent.loc[0, "pickup_lat"] = 39.0
        with self.assertRaises(ValueError):
            FreightPreprocessor().fit(inconsistent)


class PersistenceTests(unittest.TestCase):
    def test_tiny_model_round_trip_preserves_full_and_reduced_schema_predictions(self):
        source = synthetic_rows()
        train = source.iloc[:10].copy()
        before = train.copy(deep=True)
        parameters = model_parameters({"depth": 2, "loss_function": "RMSE"}, iterations=12)
        parameters["thread_count"] = 1
        pipeline = FreightRatePipeline([
            ("features", FreightPreprocessor()),
            ("model", CatBoostRegressor(**parameters)),
        ]).fit(train, train.posted_rate)
        full = source.iloc[10:].copy()
        reduced = full.drop(columns=COORDINATES)
        expected = pipeline.predict(full)
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue((expected >= MIN_RATE).all())
        np.testing.assert_allclose(pipeline.predict(reduced), expected, rtol=0, atol=1e-10)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            joblib.dump(pipeline, artifact)
            restored = joblib.load(artifact)
            np.testing.assert_allclose(restored.predict(full), expected, rtol=0, atol=1e-10)
            np.testing.assert_allclose(restored.predict(reduced), expected, rtol=0, atol=1e-10)
        assert_frame_equal(train, before)
        self.assertEqual(
            pipeline.named_steps["model"].get_cat_feature_indices(),
            list(range(len(CATEGORICAL_FEATURES))),
        )

    def test_equipment_baseline_handles_unknown_equipment_with_training_median(self):
        train = synthetic_rows().iloc[:6]
        baseline = EquipmentRateBaseline().fit(train, train.posted_rate)
        future = train.iloc[[0]].copy()
        future["equipment"] = "Unseen equipment"
        expected = (train.posted_rate / train.distance).median() * future.distance.iloc[0]
        np.testing.assert_allclose(baseline.predict(future), [expected])


if __name__ == "__main__":
    unittest.main()
