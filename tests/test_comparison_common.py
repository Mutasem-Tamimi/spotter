"""Small behavioral checks for the alternative-model data preparation."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from comparison_common import NUMERIC_FEATURES, make_preprocessor, regression_report


def rows():
    return pd.DataFrame({
        "pickup": ["A", "B", "A"], "delivery": ["B", "A", "B"],
        "equipment": ["Dry Van", "Reefer", "Dry Van"],
        "pickup_lat": [40., 42., 40.], "pickup_lon": [-80., -82., -80.],
        "delivery_lat": [42., 40., 42.], "delivery_lon": [-82., -80., -82.],
        "distance": [100., 300., 500.], "weight": [-100., 500., np.nan],
        "date": pd.date_range("2025-01-01", periods=3),
    })


class ComparisonPreprocessingTests(unittest.TestCase):
    def test_abs_correction_precedes_training_median_and_keeps_missing_flag(self):
        original = rows()
        before = original.copy(deep=True)
        prep = make_preprocessor(scale=False)
        matrix = prep.fit_transform(original)
        w = NUMERIC_FEATURES.index("weight")
        flag = NUMERIC_FEATURES.index("weight_is_missing")
        np.testing.assert_allclose(matrix[:, w], [100., 500., 300.])
        np.testing.assert_array_equal(matrix[:, flag], [0, 0, 1])
        assert_frame_equal(original, before)

    def test_evaluation_values_cannot_change_fitted_imputation_or_scaling(self):
        prep = make_preprocessor(scale=True).fit(rows())
        numeric = prep.named_steps["encode"].named_transformers_["numeric"]
        medians = numeric.named_steps["impute"].statistics_.copy()
        means = numeric.named_steps["scale"].mean_.copy()
        future = rows().iloc[[2]].copy()
        future["distance"] = 100000.
        future["date"] = pd.Timestamp("2025-12-31")
        future["posted_rate"] = 9999999.
        self.assertTrue(np.isfinite(prep.transform(future)).all())
        np.testing.assert_array_equal(numeric.named_steps["impute"].statistics_, medians)
        np.testing.assert_array_equal(numeric.named_steps["scale"].mean_, means)

    def test_unknown_category_and_absent_coordinates_keep_shape_and_are_finite(self):
        prep = make_preprocessor(scale=False).fit(rows())
        future = rows().iloc[[0]].drop(columns=["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
        future["pickup"] = "UNSEEN_CITY"
        output = prep.transform(future)
        self.assertEqual(output.shape[1], prep.transform(rows()).shape[1])
        self.assertTrue(np.isfinite(output).all())
        encoder = prep.named_steps["encode"].named_transformers_["categorical"]
        self.assertNotIn("UNSEEN_CITY", encoder.categories_[0])

    def test_percentage_reports_have_explicit_actual_price_denominator(self):
        metrics = regression_report([100., 100., 100., 100.], [100., 105., 110., 120.])
        self.assertAlmostEqual(metrics["MAPE_percent"], 8.75)
        self.assertEqual(metrics["within_5_percent"], 50.)
        self.assertEqual(metrics["within_10_percent"], 75.)
        self.assertEqual(metrics["within_20_percent"], 100.)
        with self.assertRaises(ValueError):
            regression_report([0.], [1.])


if __name__ == "__main__":
    unittest.main()
