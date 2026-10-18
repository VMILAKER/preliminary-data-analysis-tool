import unittest
from unittest.mock import patch

import pandas as pd

from src import analysis


class AnalysisCacheTests(unittest.TestCase):
    def tearDown(self):
        analysis.summarize.clear()
        analysis.isolation_outliers.clear()

    def test_isolation_cache_tracks_data_and_settings(self):
        analysis.isolation_outliers.clear()
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with patch.object(analysis.util, "detect_outliers_isolation_forest",
                          return_value=(frame.iloc[:1], None)) as detect:
            analysis.isolation_outliers(frame, 0.05)
            analysis.isolation_outliers(frame.copy(), 0.05)
            self.assertEqual(detect.call_count, 1)
            analysis.isolation_outliers(frame, 0.1)
            analysis.isolation_outliers(frame.iloc[:1], 0.1)
            self.assertEqual(detect.call_count, 3)

    def test_summary_reuses_profile_and_invalidates_after_cleaning(self):
        analysis.summarize.clear()
        frame = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        with patch.object(analysis.util, "profile_dataframe",
                          wraps=analysis.util.profile_dataframe) as profile:
            analysis.summarize(frame)
            analysis.summarize(frame.copy())
            self.assertEqual(profile.call_count, 1)
            result = analysis.summarize(frame.drop_duplicates())
            self.assertEqual(profile.call_count, 2)
            self.assertEqual(result["profile"].iloc[0]["Уникальных"], 2)
