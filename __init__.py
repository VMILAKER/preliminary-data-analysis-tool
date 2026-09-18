import json

import data_validator as dv
import utilities as util


class PreAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def gain_report(self):
        """Main function that compile entire code

        Args:
            filepath (str): destination to file (CSV, JSON, XLSX, XLSs)
        """
        df = util.load_df(self.filepath)

        prof_df = util.profile_dataframe(df)
        quality_df = util.data_quality_report(df)
        miss_df = util.missing_values_statistics(df)
        desc_df = util.get_df_describe(df)
        iqr_df = util.detect_outliers_iqr(df)
        ifo_outliers, _ = util.detect_outliers_isolation_forest(df)
        mixed_df = dv.detect_mixed_types(df)
        date_df = dv.detect_date_columns(df)
        patterns_df = dv.detect_special_patterns(df)

        report_data = util.export_json_report(
            df, prof_df, quality_df, miss_df, iqr_df, ifo_outliers, desc_df, self.filepath,
            mixed_types_df=mixed_df, date_columns_df=date_df,
            special_patterns_df=patterns_df,
        )

        json_name = self.filepath.rsplit(".", 1)[0] + "_report.json"

        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
