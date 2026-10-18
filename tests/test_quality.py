import unittest

import pandas as pd

from src import data_validator as dv
from src import utilities as util


class QualityTests(unittest.TestCase):
    def test_iqr_outliers_lower_dqi_component(self):
        frame = pd.DataFrame({"amount": list(range(20)) + [10000]})
        report = util.detect_outliers_iqr(frame)
        self.assertEqual(report.iloc[0]["Выбросов"], 1)
        _, details = dv.compute_dqi(frame, report)
        self.assertAlmostEqual(details["no_outliers"], 95.24)

    def test_unique_identifiers_and_binary_flags_are_not_cardinality_errors(self):
        frame = pd.DataFrame({"key": [f"user-{i}" for i in range(100)], "flag": [0, 1] * 50})
        _, details = dv.compute_dqi(frame)
        self.assertEqual(details["cardinality"], 100)

    def test_date_ratio_uses_checked_sample_size(self):
        frame = pd.DataFrame({"date": ["2026-09-22"] * 10000})
        dates = dv.detect_date_columns(frame)
        self.assertEqual(dates.iloc[0]["Доля дат"], "100.0%")
        self.assertEqual(dates.iloc[0]["Проверено значений"], 2000)

    def test_text_only_and_empty_data(self):
        self.assertTrue(util.detect_outliers_iqr(pd.DataFrame({"name": ["Alice", "Bob"]})).empty)
        score, details = dv.compute_dqi(pd.DataFrame())
        self.assertEqual(score, 0)
        self.assertIn("error", details)

    def test_isolation_forest_excludes_nonfinite_rows(self):
        frame = pd.DataFrame({"a": [1, 2, float("inf"), 4], "b": [2, 3, 4, None]})
        self.assertEqual(util.complete_numeric_rows(frame).index.tolist(), [0, 1])
        outliers, model = util.detect_outliers_isolation_forest(frame)
        self.assertIsNotNone(model)
        self.assertTrue(set(outliers.index) <= {0, 1})
