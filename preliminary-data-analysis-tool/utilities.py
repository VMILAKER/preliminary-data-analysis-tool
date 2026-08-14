import csv
from io import BytesIO

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Table
from sklearn.ensemble import IsolationForest

import config


def guess_csv_params(file_bytes):
    detected = chardet.detect(file_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    if encoding.lower() in ("iso-8859-1", "windows-1252"):
        encoding = "cp1251"

    try:
        text_sample = file_bytes[:10000].decode(encoding, errors="replace")
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text_sample)
        sep = dialect.delimiter
    except Exception:
        for s in [",", ";", "\t", "|"]:
            if s in file_bytes[:2000].decode(encoding, errors="replace"):
                sep = s
                break
        else:
            sep = ","
    return {"sep": sep, "encoding": encoding}


def load_dataframe(uploaded_file):
    file_name = uploaded_file.name
    file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    file_bytes = uploaded_file.getvalue()

    if not file_bytes or len(file_bytes) == 0:
        return None, "Файл пуст."

    info = {"file_name": file_name, "format": file_ext.upper()}

    try:
        if file_ext in ("csv",):
            params = guess_csv_params(file_bytes)
            info["separator"] = params["sep"]
            info["encoding"] = params["encoding"]
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=params["sep"],
                    encoding=params["encoding"],
                    engine="python",
                )
            except Exception:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(
                        uploaded_file,
                        sep=params["sep"],
                        encoding=params["encoding"],
                        engine="c",
                    )
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, engine="python")

            if _needs_header_fix(df):
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    sep=params["sep"],
                    encoding=params["encoding"],
                    header=None,
                    engine="python",
                )
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
                info["header_generated"] = True
            else:
                info["header_generated"] = False

        elif file_ext in ("xlsx", "xls"):
            info["separator"] = "—"
            info["encoding"] = "—"
            uploaded_file.seek(0)
            try:
                df = pd.read_excel(
                    uploaded_file,
                    engine="openpyxl" if file_ext == "xlsx" else "xlrd",
                )
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
            if df.columns.tolist() == list(range(df.shape[1])):
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
                info["header_generated"] = True
            else:
                info["header_generated"] = False
        else:
            return (
                None,
                f"Неподдерживаемый формат: .{file_ext}. Допустимы CSV, XLSX, XLS.",
            )

    except Exception as e:
        return None, f"Ошибка при чтении файла: {e}"

    return df, info


def _needs_header_fix(df):
    numeric_col_count = 0
    for col in df.columns:
        try:
            pd.to_numeric(df[col].iloc[:5], errors="raise")
            numeric_col_count += 1
        except (ValueError, TypeError):
            pass
    if numeric_col_count > len(df.columns) / 2:
        str_cols = sum(
            1
            for c in df.columns
            if isinstance(c, str) and not c.replace(".", "", 1).isdigit()
        )
        if str_cols < len(df.columns) / 2:
            return True
    return False


def profile_dataframe(df):
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


def plot_missing_values(df):
    prof = df.sort_values("% пропусков", ascending=True)
    colors = ["red" if v > 50 else "#636efa" for v in prof["% пропусков"]]

    fig = go.Figure(
        go.Bar(
            x=prof["% пропусков"],
            y=prof["Колонка"],
            orientation="h",
            marker_color=colors,
            text=prof["% пропусков"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Доля пропусков по колонкам",
        xaxis_title="% пропусков",
        yaxis_title="Колонка",
        height=250 + 25 * len(prof),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.add_vline(x=50, line_dash="dash",
                  line_color="red", annotation_text="50%")
    return fig


def plot_correlation_heatmap(df, colormap):
    numeric_df = df.select_dtypes(
        include=[np.number]).dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=colormap, ax=ax)
    return fig


def detect_outliers_iqr(df):
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


def create_image_reportlab(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    x, y = fig.get_size_inches()
    plt.close(fig)
    buffer.seek(0)

    return Image(buffer, x * inch, y * inch)


def detect_outliers_isolation_forest(df, contamination=0.05, random_state=42):
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


def get_df_info(df):
    """Возвращает строку с df.info()"""
    from io import StringIO
    buf = StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def get_df_describe(df):
    """Возвращает describe с процентилями"""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return pd.DataFrame({"Сообщение": ["Нет числовых колонок"]})
    return numeric.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])


def plot_histograms(df):
    """Строит гистограммы распределения для всех числовых колонок"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    for i, col in enumerate(numeric_cols):
        series = df[col].dropna()
        if series.nunique() < 2:
            axes[i].text(0.5, 0.5, f"Колонка '{col}' содержит менее 2 уникальных значений",
                         ha='center', va='center', transform=axes[i].transAxes, fontsize=10)
            continue
        axes[i].hist(series, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Частота')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def data_quality_report(df):
    """Единый отчёт о качестве данных"""
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
                problems.append("Высокая кардинальность (уникальный идентификатор)")
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


def plot_missing_matrix(df):
    """Тепловая карта пропусков (бинарная матрица: 1 — пропуск, 0 — значение)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_cols = df.columns
    # Для больших данных сэмплируем
    sample_df = df if len(df) < 5000 else df.sample(5000, random_state=42)
    missing_matrix = sample_df[all_cols].isna().astype(int)
    if missing_matrix.sum().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(all_cols) * 0.4), max(4, len(sample_df) * 0.002)))
    sns.heatmap(missing_matrix.T, cbar=False, cmap=["#f0f0f0", "#d32f2f"],
                ax=ax, yticklabels=True, xticklabels=False)
    ax.set_title("Матрица пропусков (красный — пропуск, серый — значение)", fontsize=11)
    ax.set_xlabel("Строки (выборка до 5000)" if len(df) >= 5000 else "Строки")
    ax.set_ylabel("Колонки")
    plt.tight_layout()
    return fig


def missing_values_statistics(df):
    """Статистика по пропускам + рекомендуемая стратегия заполнения"""
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


def detect_outliers_isolation_forest(df, contamination=0.05, random_state=42):
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


def export_csv(dataframe):
    return dataframe.to_csv(index=False)


def export_excel_report(df, prof_df, quality_df, miss_df, iqr_df, ifo_df, desc_df, info_dict,
                        mixed_types_df=None, date_columns_df=None,
                        special_patterns_df=None):
    """Экспорт полного отчёта в Excel с несколькими листами"""
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Лист 1: Данные (очищенные от дубликатов)
        df_clean = df.drop_duplicates(keep="first")
        df_clean.to_excel(writer, sheet_name='Данные', index=False)

        # Лист 2: Информация о файле
        info_df = pd.DataFrame([
            {"Параметр": "Имя файла", "Значение": info_dict.get("file_name", "—")},
            {"Параметр": "Формат", "Значение": info_dict.get("format", "—")},
            {"Параметр": "Строк", "Значение": df.shape[0]},
            {"Параметр": "Столбцов", "Значение": df.shape[1]},
            {"Параметр": "Разделитель", "Значение": info_dict.get("separator", "—")},
            {"Параметр": "Кодировка", "Значение": info_dict.get("encoding", "—")},
            {"Параметр": "Заголовок сгенерирован", "Значение": "Да" if info_dict.get("header_generated") else "Нет"},
        ])
        info_df.to_excel(writer, sheet_name='Инфо', index=False)

        # Лист 3: Профиль колонок
        prof_df.to_excel(writer, sheet_name='Профиль', index=False)

        # Лист 4: Качество данных
        quality_df.to_excel(writer, sheet_name='Качество', index=False)

        # Лист 5: Статистика (describe)
        if desc_df is not None and not desc_df.empty:
            desc_df.to_excel(writer, sheet_name='Статистика')

        # Лист 6: Пропуски
        if miss_df is not None and not miss_df.empty:
            miss_df.to_excel(writer, sheet_name='Пропуски', index=False)
        else:
            pd.DataFrame({"Сообщение": ["Пропуски отсутствуют"]}).to_excel(
                writer, sheet_name='Пропуски', index=False)

        # Лист 7: Выбросы IQR
        if iqr_df is not None and not iqr_df.empty:
            iqr_df.to_excel(writer, sheet_name='Выбросы_IQR', index=False)
        else:
            pd.DataFrame({"Сообщение": ["Выбросы IQR не обнаружены"]}).to_excel(
                writer, sheet_name='Выбросы_IQR', index=False)

        # Лист 8: Выбросы Isolation Forest
        if ifo_df is not None and not ifo_df.empty:
            ifo_df.to_excel(writer, sheet_name='Выбросы_IF', index=False)
        else:
            pd.DataFrame({"Сообщение": ["Выбросы Isolation Forest не обнаружены"]}).to_excel(
                reader, sheet_name='Выбросы_IF', index=False)

        if mixed_types_df is not None and not mixed_types_df.empty:
            mixed_types_df.to_excel(writer, sheet_name='Структурные_ошибки', index=False)

        if date_columns_df is not None and not date_columns_df.empty:
            date_columns_df.to_excel(writer, sheet_name='Колонки_с_датами', index=False)

        if special_patterns_df is not None and not special_patterns_df.empty:
            special_patterns_df.to_excel(writer, sheet_name='Паттерны_тел_email', index=False)

    output.seek(0)
    return output


def create_list_of_lists(*args):
    if args:
        for item in args:
            if isinstance(item, dict):
                for title, df in item.items():
                    yield {'title': title, 'df': [df.columns.values.tolist()] + df.values.tolist()}
            else:
                return 'Unsupported type'
    else:
        return 'No data provided'


def set_head_for_df_table(df: list, length: int):
    if len(df) > length:
        return Table(df[:length])
    else:
        return Table(df)


def create_title_with_df(data_dict: dict):
    text_styles = config.TextStyle('DejaVuSerif', 'DejaVuSerif.ttf')
    style = config.get_table_style()

    p = Paragraph(str(data_dict['title']).replace(
        "\n", "<br />"), text_styles.left_style(12))
    df_table = set_head_for_df_table(data_dict['df'], 7)
    df_table.setStyle(style)
    return [p, df_table]


def draw_boxplot(df: pd.DataFrame, column: str):
    plt.figure(figsize=(4, 3))
    sns.boxplot(y=df[column])
    return plt
