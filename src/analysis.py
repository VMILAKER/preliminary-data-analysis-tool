"""Shared, bounded caches for the screen and report."""

import streamlit as st

from src import data_validator as dv
from src import utilities as util


@st.cache_data(show_spinner=False, max_entries=8, ttl=600)
def summarize(df):
    iqr = util.detect_outliers_iqr(df)
    return {
        "profile": util.profile_dataframe(df),
        "quality": util.data_quality_report(df),
        "missing": util.missing_values_statistics(df),
        "describe": util.get_df_describe(df),
        "info": util.get_df_info(df),
        "iqr": iqr,
        "validation": dv.full_validation_report(df, iqr),
    }


@st.cache_data(show_spinner=False, max_entries=8, ttl=600)
def isolation_outliers(df, contamination):
    outliers, _ = util.detect_outliers_isolation_forest(df, contamination=contamination)
    return outliers
