import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def load_df(filepath: str) -> pd.DataFrame:
    """Dataframe load

    Args:
        filepath (str): path to file with data
    """
    file_extension = os.path.splitext(filepath)[1]

    if file_extension == '.json':
        df = pd.read_json(filepath, encoding='utf-8')
    elif file_extension == '.csv':
        df = pd.read_csv(filepath)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise TypeError('Not supported file type')
    df = df.astype(object).replace({np.nan: None})
    return df


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe profile manipulation

    Args:
        df (pd.DataFrame): initial dataframe

    Returns:
        pd.DataFrame: profile dataframe
    """
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        total = len(df)
        pct = round(missing / total * 100, 2) if total > 0 else 0.0
        rows.append(
            {
                "Колонка": col,
                "Тип": str(df[col].dtype),
                "Пропуски (NaN)": missing,
                "% пропусков": pct,
                "Уникальных": df[col].nunique(),
            }
        )
    return pd.DataFrame(rows)


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """IQR detection

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: table with outliers information
    """
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if 'id' not in col:
            series = df[col].dropna()
            if series.nunique() < 5:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            rows.append(
                {
                    "Колонка": col,
                    "Тип данных": str(df[col].dtype),
                    "Всего значений": len(series),
                    "Выбросов": len(outliers),
                    "% выбросов": round(len(outliers) / len(series) * 100, 2),
                    "Нижняя граница": round(lower, 4),
                    "Верхняя граница": round(upper, 4),
                }
            )
    return pd.DataFrame(rows)


def detect_outliers_isolation_forest(df: pd.DataFrame, contamination=0.05, random_state=42):
    """Detection outliers by Isolation Forest

    Args:
        df (pd.DataFrame): initial dataframe
        contamination (float, optional): percentage of contamination in df. Defaults to 0.05.
        random_state (int, optional): Defaults to 42.

    Returns:
        _type_: dataframe with outliers and IsolationForest model
    """
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[0] == 0 or num_df.shape[1] < 2:
        return pd.DataFrame(), None

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    preds = model.fit_predict(num_df)
    scores = model.decision_function(num_df)

    mask_outliers = preds == -1
    outlier_idx = num_df.index[mask_outliers]
    outliers_df = df.loc[outlier_idx].copy()
    outliers_df["anomaly_score"] = scores[mask_outliers]
    return outliers_df, model


def get_df_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Return describe dataframe with percentiles

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: dataframe with percentiles
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return pd.DataFrame({"Сообщение": ["Нет числовых колонок"]})
    return numeric.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a unified report on data quality

    Args:
        df (pd.DataFrame): initial dataframe

    Returns:
        pd.DataFrame: dataframe with quality report 
    """
    total = len(df)
    report_rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0
        nunique = df[col].nunique()
        dtype = str(df[col].dtype)
        # Определяем тип данных
        if pd.api.types.is_numeric_dtype(df[col]):
            data_type = "Числовая"
            problems = []
            if missing_pct > 50:
                problems.append(">50% пропусков")
            if nunique == 1:
                problems.append("Константа")
            if nunique < 5 and len(df[col].dropna()) > 0:
                problems.append("Малая кардинальность")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            data_type = "Дата/время"
            problems = []
        else:
            data_type = "Категориальная/Текст"
            problems = []
            if nunique == total and total > 100:
                problems.append(
                    "Высокая кардинальность (уникальный идентификатор)")
            elif nunique == 1:
                problems.append("Константа")
        report_rows.append({
            "Колонка": col,
            "Тип данных": dtype,
            "Категория": data_type,
            "Непустых": total - missing,
            "Пропуски": missing,
            "% пропусков": missing_pct,
            "Уникальных": nunique,
            "Проблемы": "; ".join(problems) if problems else "—"
        })
    return pd.DataFrame(report_rows)


def missing_values_statistics(df: pd.DataFrame):
    """Return statistics on missing values with recommended filling strategies

    Args:
        df (pd.DataFrame): initial dataframe

    Returns:
        pd.DataFrame: dataframe with missing values
    """
    total = len(df)
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0
        if missing == 0:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            if missing_pct < 5:
                strategy = "median"
            elif missing_pct < 30:
                strategy = "mean"
            else:
                strategy = "Удалить колонку (>30% пропусков)"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            strategy = "ffill (заполнить предыдущим)"
        else:
            if missing_pct < 5:
                strategy = "mode (мода)"
            elif missing_pct < 30:
                strategy = "'Unknown' / 'Неизвестно'"
            else:
                strategy = "Удалить колонку (>30% пропусков)"
        rows.append({
            "Колонка": col,
            "Тип": str(dtype),
            "Пропусков": missing,
            "% пропусков": missing_pct,
            "Рекомендуемая стратегия": strategy
        })
    return pd.DataFrame(rows)


def export_json_report(df: pd.DataFrame, prof_df: pd.DataFrame, quality_df: pd.DataFrame, miss_df: pd.DataFrame, iqr_df: pd.DataFrame, ifo_df: pd.DataFrame, desc_df: pd.DataFrame, filepath: str,
                       mixed_types_df=None, date_columns_df=None,
                       special_patterns_df=None) -> list:
    """Export full report in Excel format

    Args:
        df (pd.DataFrame): _description_
        prof_df (pd.DataFrame): _description_
        quality_df (pd.DataFrame): _description_
        miss_df (pd.DataFrame): _description_
        iqr_df (pd.DataFrame): _description_
        ifo_df (pd.DataFrame): _description_
        desc_df (pd.DataFrame): _description_
        filepath (str): _description_
        mixed_types_df (_type_, optional): _description_. Defaults to None.
        date_columns_df (_type_, optional): _description_. Defaults to None.
        special_patterns_df (_type_, optional): _description_. Defaults to None.

    Returns:
        list: list with preanalysis dict
    """
    output = []

    # Sheet 1: clean dataframe
    df_clean = df.drop_duplicates(keep="first")

    output.append(df_clean.to_dict())

    # Sheet 2: file info
    info_df = pd.DataFrame([
        {"Параметр": "Имя файла",
            "Значение": filepath},
        # {"Параметр": "Формат", "Значение": info_dict.get("format", "—")},
        {"Параметр": "Строк", "Значение": df.shape[0]},
        {"Параметр": "Столбцов", "Значение": df.shape[1]},
        # {"Параметр": "Разделитель",
        #     "Значение": info_dict.get("separator", "—")},
        # {"Параметр": "Кодировка",
        #     "Значение": info_dict.get("encoding", "—")},
        # {"Параметр": "Заголовок сгенерирован",
        #     "Значение": "Да" if info_dict.get("header_generated") else "Нет"},
    ])
    output.append(info_df.to_dict())

    # Sheet 3: column profile
    output.append(prof_df.to_dict())

    # Sheet 4: data quality
    output.append(quality_df.to_dict())

    # Sheet 5: describe
    if desc_df is not None and not desc_df.empty:
        output.append(desc_df.to_dict())

    # Sheet 6: gaps
    if miss_df is not None and not miss_df.empty:
        output.append(miss_df.to_dict())
    else:
        output.append(pd.DataFrame(
            {"Сообщение": ["Пропуски отсутствуют"]}).to_dict())

    # Sheet 7: IQR outliers
    if iqr_df is not None and not iqr_df.empty:
        output.append(iqr_df.to_dict())
    else:
        output.append(pd.DataFrame(
            {"Сообщение": ["Выбросы IQR не обнаружены"]}).to_dict())

    # Sheet 8: Isolation Forest Outliers
    if ifo_df is not None and not ifo_df.empty:
        output.append(ifo_df.to_dict())
    else:
        output.append(pd.DataFrame(
            {"Сообщение": ["Выбросы Isolation Forest не обнаружены"]}).to_dict())

    if mixed_types_df is not None and not mixed_types_df.empty:
        output.append(mixed_types_df.to_dict())

    if date_columns_df is not None and not date_columns_df.empty:
        output.append(date_columns_df.to_dict())

    if special_patterns_df is not None and not special_patterns_df.empty:
        output.append(special_patterns_df.to_dict())

    return output
