import re
from datetime import datetime

import pandas as pd

_DATE_FORMATS = [
    "%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S", "%Y%m%d", "%d.%m.%y",
    "%Y-%m-%d %H:%M", "%b %d %Y", "%B %d, %Y",
]

_PHONE_CLEAN_RE = re.compile(r"[\s\-\(\)\.]")
_PHONE_PATTERNS = [
    re.compile(r"^\+?7\d{10}$"), re.compile(r"^8\d{10}$"),
    re.compile(r"^\+?1\d{10}$"), re.compile(r"^\d{10,15}$"),
]
_EMAIL_RE = re.compile(r"^[\w\.\+\-]+@[\w\-]+(\.[\w\-]+)+$")
_URL_RE = re.compile(r"^https?://[^\s/$.?#][^\s]*$", re.IGNORECASE)


def _try_parse_date(string):
    if not string or not isinstance(string, str):
        return None
    string = string.strip()
    if len(string) > 25:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(string, fmt)
        except ValueError:
            continue
    return None


def _classify_value(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return "empty"
    if re.match(r"^[+-]?\d+$", s):
        return "integer"
    if re.match(r"^[+-]?\d+\.\d*$", s) or re.match(r"^[+-]?\d*\.\d+$", s):
        return "float"
    if _try_parse_date(s):
        return "date"
    if _EMAIL_RE.match(s):
        return "email"
    cleaned = _PHONE_CLEAN_RE.sub("", s)
    for pat in _PHONE_PATTERNS:
        if pat.match(cleaned):
            return "phone"
    if _URL_RE.match(s):
        return "url"
    return "text"


def detect_mixed_types(df, sample_size=2000, min_ratio=0.05):
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        sample = series if len(series) <= sample_size else series.sample(
            sample_size, random_state=42)

        type_counts = {}
        examples = {}
        for val in sample:
            cat = _classify_value(val)
            if cat and cat != "empty":
                type_counts[cat] = type_counts.get(cat, 0) + 1
                if cat not in examples:
                    examples[cat] = str(val)[:80]

        if not type_counts:
            continue

        total_cls = sum(type_counts.values())
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        main_type = sorted_types[0][0]
        main_ratio = sorted_types[0][1] / total_cls

        other = []
        for t, c in sorted_types[1:]:
            if (c / total_cls) >= min_ratio:
                other.append(f"{t} ({c/total_cls*100:.1f}%)")

        if other:
            ex_list = []
            for t, _ in sorted_types:
                if t in examples:
                    ex_list.append(f"[{t}] {examples[t]}")
            results.append({
                "Колонка": col,
                "Тип данных": str(df[col].dtype),
                "Основной тип": main_type,
                "Другие типы": ", ".join(other),
                "Доля основного": f"{main_ratio*100:.1f}%",
                "Примеры": "; ".join(ex_list[:4]),
            })
    return pd.DataFrame(results)


def compute_dqi(df, iqr_outliers_df=None):
    if df.empty:
        return 0.0, {"error": "Датасет пуст"}

    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    details = {}

    total_nan = int(df.isna().sum().sum())
    completeness = (1 - total_nan / total_cells) * \
        100 if total_cells > 0 else 100
    details["completeness"] = round(completeness, 2)
    details["completeness_desc"] = (
        f"Заполнено ячеек: {total_cells - total_nan} из {total_cells}"
        f" ({completeness:.1f}%)"
    )

    dup_count = df.duplicated().sum()
    uniqueness = (1 - dup_count / total_rows) * 100 if total_rows > 0 else 100
    details["uniqueness"] = round(uniqueness, 2)
    details["uniqueness_desc"] = (
        f"Уникальных строк: {total_rows - dup_count} из {total_rows}"
        f" ({uniqueness:.1f}%)"
    )

    constant_cols = sum(
        1 for c in df.columns if df[c].nunique(dropna=False) <= 1)
    non_constant = total_cols - constant_cols
    no_constants = (non_constant / total_cols) * 100 if total_cols > 0 else 100
    details["no_constants"] = round(no_constants, 2)
    details["no_constants_desc"] = (
        f"Колонок-констант: {constant_cols} из {total_cols}"
    )

    try:
        mixed_df = detect_mixed_types(df, min_ratio=0.02)
        mixed_count = len(mixed_df)
    except Exception:
        mixed_count = 0
    type_consistency = ((total_cols - mixed_count) / total_cols) * 100
    details["type_consistency"] = round(type_consistency, 2)
    details["type_consistency_desc"] = (
        f"Колонок со смешанными типами: {mixed_count} из {total_cols}"
    )

    outlier_score = 100.0
    if iqr_outliers_df is not None and not iqr_outliers_df.empty:
        outlier_pcts = []
        for _, row in iqr_outliers_df.iterrows():
            total_vals = row.get("Всего значений", 0)
            outlier_count = row.get("Кол-во выбросов", 0)
            if total_vals and total_vals > 0:
                outlier_pcts.append((1 - outlier_count / total_vals) * 100)
        if outlier_pcts:
            outlier_score = sum(outlier_pcts) / len(outlier_pcts)
    details["no_outliers"] = round(outlier_score, 2)
    details["no_outliers_desc"] = (
        f"Средний процент не-выбросов: {outlier_score:.1f}%"
    )

    bad_card = 0
    for col in df.columns:
        nu = df[col].nunique()
        if nu <= 1:
            bad_card += 1
        elif pd.api.types.is_numeric_dtype(df[col]):
            if nu < 3 and nu > 0:
                bad_card += 1
        else:
            if nu == len(df[col].dropna()) and nu > 50:
                bad_card += 1
    card_score = ((total_cols - bad_card) / total_cols) * \
        100 if total_cols > 0 else 100
    details["cardinality"] = round(card_score, 2)
    details["cardinality_desc"] = (
        f"Колонок с проблемной кардинальностью: {bad_card} из {total_cols}"
    )

    rows_nan = df.isna().any(axis=1).sum()
    row_comp = (1 - rows_nan / total_rows) * 100 if total_rows > 0 else 100
    details["row_completeness"] = round(row_comp, 2)
    details["row_completeness_desc"] = (
        f"Строк без пропусков: {total_rows - rows_nan} из {total_rows}"
        f" ({row_comp:.1f}%)"
    )

    weights = {
        "completeness": 0.30, "uniqueness": 0.15,
        "no_constants": 0.10, "type_consistency": 0.15,
        "no_outliers": 0.10, "cardinality": 0.10,
        "row_completeness": 0.10,
    }
    score = sum(details[k] * weights[k] for k in weights)
    score = max(0.0, min(100.0, score))
    details["dqi_score"] = round(score, 2)

    if score >= 90:
        details["dqi_grade"] = "Отличное"
    elif score >= 70:
        details["dqi_grade"] = "Хорошее"
    elif score >= 50:
        details["dqi_grade"] = "Среднее"
    else:
        details["dqi_grade"] = "Плохое"

    return round(score, 2), details


def detect_date_columns(df, threshold=0.3):
    results = []
    for col in df.select_dtypes(include=["datetime64"]).columns:
        results.append({
            "Колонка": col,
            "Тип данных": "datetime64",
            "Похоже на дату": len(df[col].dropna()),
            "Доля дат": "100.0%",
            "Образец формата": "datetime64",
            "Рекомендация": "Уже в формате datetime",
        })

    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str)
        if len(series) < 5:
            continue
        date_count = 0
        formats_found = set()
        samples = []
        for val in series.head(2000):
            dt = _try_parse_date(val)
            if dt is not None:
                date_count += 1
                if len(samples) < 3:
                    samples.append(val)
                for fmt in _DATE_FORMATS:
                    try:
                        datetime.strptime(val.strip(), fmt)
                        formats_found.add(fmt)
                        break
                    except ValueError:
                        continue

        ratio = date_count / len(series)
        if ratio >= threshold:
            fmt_str = ", ".join(sorted(formats_found)[
                                :3]) if formats_found else "разные"
            results.append({
                "Колонка": col,
                "Тип данных": str(df[col].dtype),
                "Похоже на дату": date_count,
                "Доля дат": f"{ratio*100:.1f}%",
                "Образец формата": fmt_str,
                "Рекомендация": (
                    "Стандартизировать в datetime" if ratio > 0.8
                    else "Смешанный тип — ручная проверка"
                ),
            })
    return pd.DataFrame(results)


def _detect_phones_in_series(series):
    results = []
    fmt_map = {0: "РФ (+7)", 1: "РФ (8)", 2: "US (+1)", 3: "Generic"}
    for idx, val in series.dropna().items():
        s = str(val).strip()
        cleaned = _PHONE_CLEAN_RE.sub("", s)
        for i, pat in enumerate(_PHONE_PATTERNS):
            if pat.match(cleaned):
                results.append((idx, s, fmt_map.get(i, "unknown")))
                break
    return results


def _detect_emails_in_series(series):
    results = []
    for idx, val in series.dropna().items():
        s = str(val).strip()
        if _EMAIL_RE.match(s):
            results.append((idx, s))
    return results


def detect_special_patterns(df):
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        phones = _detect_phones_in_series(series)
        emails = _detect_emails_in_series(series)
        urls = []
        dates_found = []
        examples = []

        for val in series.head(5000):
            s = val.strip()
            if _URL_RE.match(s):
                urls.append(s)
            dt = _try_parse_date(s)
            if dt is not None:
                dates_found.append(s)

        if not (phones or emails or urls or dates_found):
            continue

        if phones:
            examples.append(f"Тел: {phones[0][1]}")
        if emails:
            examples.append(f"Email: {emails[0][1]}")
        if urls:
            examples.append(f"URL: {urls[0]}")
        if dates_found:
            examples.append(f"Дата: {dates_found[0]}")

        results.append({
            "Колонка": col,
            "Телефоны": len(phones),
            "Email": len(emails),
            "URL": len(urls),
            "Даты (текст)": len(dates_found),
            "Примеры": " | ".join(examples[:4]),
        })
    return pd.DataFrame(results)


def missing_pattern_report(df):
    result = {}
    total_rows = len(df)
    if total_rows == 0:
        return {"empty": True}

    nan_per_row = df.isna().sum(axis=1)

    groups = {"0": 0, "1-2": 0, "3-5": 0, ">5": 0}
    for k, v in nan_per_row.value_counts().items():
        if k == 0:
            groups["0"] += v
        elif k <= 2:
            groups["1-2"] += v
        elif k <= 5:
            groups["3-5"] += v
        else:
            groups[">5"] += v

    result["row_summary"] = pd.DataFrame([
        {"Группа (NaN в строке)": g, "Строк": c,
         "Доля": f"{c/total_rows*100:.2f}%"}
        for g, c in groups.items() if c > 0
    ])

    nan_mat = df.isna().astype(int)
    cols_nan = [c for c in df.columns if nan_mat[c].sum() > 0]

    if len(cols_nan) >= 2:
        corr = nan_mat[cols_nan].corr()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                v = corr.iloc[i, j]
                if abs(v) > 0.3:
                    pairs.append({
                        "Колонка 1": corr.columns[i],
                        "Колонка 2": corr.columns[j],
                        "Корреляция NaN": round(v, 3),
                        "Совместных NaN": int((nan_mat[corr.columns[i]] & nan_mat[corr.columns[j]]).sum()),
                    })
        pairs.sort(key=lambda x: -abs(x["Корреляция NaN"]))
        result["pair_corr"] = pd.DataFrame(pairs[:15])

    top5 = nan_per_row.sort_values(ascending=False).head(5)
    rows_detail = []
    for idx, cnt in top5.items():
        missing_cols = df.columns[df.loc[idx].isna()].tolist()
        rows_detail.append({
            "Индекс строки": str(idx),
            "Пропусков": cnt,
            "Колонки с пропусками": ", ".join(missing_cols[:10]),
        })
    result["top_missing_rows"] = pd.DataFrame(rows_detail)

    return result


def standardize_dates(df, columns=None, target_format="%Y-%m-%d"):
    df_out = df.copy()
    report_rows = []

    if columns is None:
        dc = detect_date_columns(df)
        if dc.empty:
            return df_out, pd.DataFrame({"Сообщение": ["Нет колонок с датами"]})
        columns = dc["Колонка"].tolist()

    for col in columns:
        if col not in df_out.columns:
            continue
        converted = 0
        failed = 0
        parsed = []

        for idx, val in df_out[col].dropna().items():
            dt = _try_parse_date(str(val))
            if dt is not None:
                parsed.append((idx, dt))
                converted += 1
            else:
                failed += 1

        if converted > 0:
            df_out[col] = df_out[col].astype(object)
            for idx, dt in parsed:
                df_out.at[idx, col] = dt.strftime(target_format)
            try:
                df_out[col] = pd.to_datetime(df_out[col], errors="coerce")
            except Exception as e:
                print(f'Error: {e}')
                raise
            report_rows.append({
                "Колонка": col,
                "Конвертировано": converted,
                "Ошибок": failed,
                "Формат": target_format,
            })

    if not report_rows:
        return df_out, pd.DataFrame({"Сообщение": ["Нет сконвертированных колонок"]})
    return df_out, pd.DataFrame(report_rows)


def full_validation_report(df, iqr_outliers_df=None):
    dqi_score, dqi_details = compute_dqi(df, iqr_outliers_df)
    return {
        "dqi": (dqi_score, dqi_details),
        "mixed_types": detect_mixed_types(df),
        "date_columns": detect_date_columns(df),
        "special_patterns": detect_special_patterns(df),
        "missing_patterns": missing_pattern_report(df),
    }
