import unittest

import pandas as pd

from src import utilities as util


class ExportTests(unittest.TestCase):
    def test_export_preserves_displayed_rows_outliers_and_settings(self):
        df = pd.DataFrame({"a": [1, 1, 9], "b": [2, 2, 15]}, index=[0, 1, 7])
        outliers = df.loc[[7]].assign(anomaly_score=-0.25)
        empty = pd.DataFrame()
        report = util.export_excel_report(
            df, util.profile_dataframe(df), util.data_quality_report(df), empty,
            empty, outliers, util.get_df_describe(df), {"file_name": "test.csv"},
            analysis_settings={"contamination": 0.2},
        )
        sheets = pd.read_excel(report, sheet_name=None)
        self.assertEqual(len(sheets["Данные"]), 3)
        self.assertEqual(sheets["Данные"]["Индекс строки"].tolist(), [0, 1, 7])
        self.assertEqual(sheets["Выбросы_IF"]["Индекс строки"].tolist(), [7])
        self.assertAlmostEqual(sheets["Выбросы_IF"]["anomaly_score"].iloc[0], -0.25)
        self.assertEqual(sheets["Настройки"]["Значение"].iloc[0], 0.2)
