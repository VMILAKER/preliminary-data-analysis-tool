import unittest
from io import BytesIO

import pandas as pd

from src.utilities import load_dataframe, normalize_nested_values, profile_dataframe


class JsonLoadingTests(unittest.TestCase):
    def test_nested_values_beyond_preview_are_safe(self):
        frame = pd.DataFrame({"value": list(range(12)) + [{"city": "Москва"}, [1, 2]]})
        result = normalize_nested_values(frame)
        self.assertEqual(profile_dataframe(result).iloc[0]["Уникальных"], 14)
        self.assertEqual(result.duplicated().sum(), 0)
        self.assertIn("Москва", result.iloc[12, 0])

    def test_flatten_keeps_rows_lists_and_nulls(self):
        frame = pd.DataFrame({"user": [{"city": "Москва"}, None], "tags": [[1, 2], []]})
        result = normalize_nested_values(frame, "flatten")
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "user.city"], "Москва")
        self.assertEqual(result.loc[0, "tags"], "[1, 2]")
        self.assertTrue(pd.isna(result.loc[1, "user.city"]))

    def test_flatten_rejects_colliding_paths(self):
        with self.assertRaises(ValueError):
            normalize_nested_values(pd.DataFrame({"a": [{"b": 1}], "a.b": [2]}), "flatten")

    def test_loading_rewinds_stream_and_normalizes_numeric_columns(self):
        source = BytesIO(b'[[1, {"x": 2}], [3, [4]]]')
        source.name = "data.json"
        source.seek(5)
        frame, info = load_dataframe(source)
        self.assertIsNotNone(frame, info)
        self.assertEqual(frame.columns.tolist(), ["col_0", "col_1"])
        self.assertEqual(frame["col_1"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
