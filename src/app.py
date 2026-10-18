import numpy as np
import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate

import src.data_validator as dv
import src.utilities as util
from src import config
from src import analysis
from src.session import apply_change


@st.cache_resource
def get_pygwalker_renderer(data):
    return StreamlitRenderer(data, spec_io_mode="rw")


def use_pygwalker(df):
    with st.expander("Интерактивное исследование данных (PyGWalker)", expanded=True):
        try:
            renderer = get_pygwalker_renderer(df)
            renderer.explorer()
        except Exception as e:
            st.error(f"Ошибка при загрузке PyGWalker: {e}")


def count_values_by_column(df):
    with st.expander("Подсчет уникальных значений по столбцам"):
        column_choice = st.radio("Столбец", list(df.columns), horizontal=True)
        st.session_state.column_value_count = column_choice
        st.dataframe(
            df[st.session_state.column_value_count].value_counts(),
            use_container_width=True,
        )


def drop_visualization(df):
    with st.expander("Визуализация пропусков", expanded=False):
        if df["Пропуски (NaN)"].sum() > 0:
            fig_missing = util.plot_missing_values(df)
            st.plotly_chart(fig_missing)
        else:
            st.success("Пропуски отсутствуют.")


def corr_visaulization(df):
    with st.expander("Корреляция числовых колонок", expanded=False):
        column_choice = st.radio(
            "Colormap", ["crest", "coolwarm", "PiYG"], horizontal=True
        )
        fig_corr = util.plot_correlation_heatmap(df, column_choice)
        if fig_corr is None:
            st.info(
                "Недостаточно числовых колонок для построения корреляции (нужно минимум 2)."
            )
        else:
            st.write(fig_corr)


def basic_overview(results):
    with st.expander("Базовая информация (df.info)", expanded=False):
        info_text = results["info"]
        st.text(info_text)

    with st.expander("Статистическое описание (df.describe)", expanded=False):
        desc_df = results["describe"]
        st.dataframe(desc_df, use_container_width=True)


def histogram_distributions(df):
    with st.expander("Гистограммы распределения числовых колонок", expanded=False):
        fig = util.plot_histograms(df)
        if fig is None:
            st.info("Нет числовых колонок для построения гистограмм.")
        else:
            st.pyplot(fig)


def data_quality_section(quality_df):
    st.subheader("Отчёт о качестве данных")
    st.dataframe(quality_df, use_container_width=True)

    total_problems = quality_df[quality_df["Проблемы"] != "—"].shape[0]
    total_cols = len(quality_df)
    if total_problems > 0:
        st.warning(
            f"Обнаружено **{total_problems}** колонок с потенциальными проблемами из **{total_cols}**.")
    else:
        st.success(f"Все **{total_cols}** колонок не имеют явных проблем.")


def missing_analysis(miss_df):
    with st.expander("Статистика и стратегия заполнения пропусков", expanded=False):
        if miss_df.empty:
            st.success("Пропуски отсутствуют.")
        else:
            st.caption("План обработки сохраняется в Excel; данные автоматически не изменяются. "
                       "Медиана устойчива к выбросам, среднее подходит для симметричных распределений. "
                       "ffill имеет смысл только при осмысленном порядке строк; удаление требует проверки потери информации.")
            return st.data_editor(
                miss_df, key="missing_plan_" + str(st.session_state.get("source_id")),
                hide_index=True, use_container_width=True,
                disabled=[col for col in miss_df.columns if col != "Рекомендуемая стратегия"],
                column_config={"Рекомендуемая стратегия": st.column_config.SelectboxColumn(
                    "Выбранная стратегия", options=["Проверить вручную", "median", "mean", "mode (мода)",
                    "ffill (заполнить предыдущим)", "'Unknown' / 'Неизвестно'", "Удалить колонку"],
                    required=True,
                )},
            )
    return miss_df


def duplicate_search(df):
    st.subheader("Поиск дубликатов")
    dup_mask = df.duplicated(keep=False)
    total_dups = dup_mask.sum()
    st.write(
        f"**Полных дубликатов (повторяющиеся строки):** {total_dups} "
        f"(лишних строк для удаления: {df.duplicated(keep='first').sum()})"
    )

    if total_dups > 0:
        st.dataframe(df[dup_mask].head())
        if st.button(f"Удалить дубликаты: {int(df.duplicated().sum())} строк"):
            apply_change(st.session_state, df.drop_duplicates(keep="first"), "Удаление дубликатов")
            st.rerun()
    else:
        st.success("Дубликаты не найдены.")


def outliers_iqr_search(df, outliers_df, show_charts):
    st.subheader("Обнаружение выбросов (IQR)")
    st.write(
        "IQR рассчитываются по формуле IQR = Q3 − Q1, где Q3- третий квартиль, Q1- первый квартиль")
    if outliers_df.empty:
        st.info("Нет колонок, удовлетворяющих условиям IQR.")
    else:
        st.dataframe(outliers_df)
        if show_charts:
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if df[c].nunique() >= 5
            ]
            if not numeric_cols:
                st.info(
                    "Нет числовых колонок с >=5 уникальными значениями для анализа выбросов."
                )
            else:
                choice = st.radio(
                    "Столбец", numeric_cols, horizontal=True
                )
                boxplot = util.draw_boxplot(df, choice)
                st.pyplot(boxplot)


def outliers_isolation_forest_search(df):
    st.subheader("Многомерные выбросы (Isolation Forest)")
    num_full = df.select_dtypes(include=[np.number]).dropna()
    if num_full.shape[0] == 0 or num_full.shape[1] < 2:
        st.info(
            "Недостаточно числовых данных для Isolation Forest (нужно минимум 2 числовые колонки без пропусков).")
        return pd.DataFrame()

    rows_before = len(num_full)
    rows_total = len(df)
    st.caption(f"Строк с полными числовыми данными: {rows_before} из {rows_total} "
               f"(строк с пропусками: {rows_total - rows_before})")

    contamination = st.slider(
        "Доля выбросов (contamination)",
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        key="if_contamination",
    )

    ifo_outliers = analysis.isolation_outliers(
        df, contamination=contamination
    )

    if ifo_outliers.empty:
        st.success("Isolation Forest не обнаружил выбросов.")
    else:
        pct = round(len(ifo_outliers) / rows_before * 100, 2)
        st.write(
            f"**Найдено выбросов:** {len(ifo_outliers)} из {rows_before} ({pct}%)")
        st.dataframe(ifo_outliers.head(100))

        if st.button(f"Удалить найденные выбросы: {len(ifo_outliers)} строк"):
            apply_change(st.session_state, df.drop(index=ifo_outliers.index), "Удаление выбросов Isolation Forest")
            st.rerun()
    return ifo_outliers


def validation_section(validation):
    st.subheader("Валидация данных")

    with st.expander("Индекс качества данных (DQI)", expanded=True):
        st.caption("DQI — эвристическая оценка по семи метрикам с весами ниже. "
                   "IQR-выбросы требуют проверки по смыслу данных. Уникальные идентификаторы "
                   "и бинарные признаки сами по себе не снижают оценку кардинальности. "
                   "Константы влияют и на метрику констант, и на кардинальность.")
        dqi_score, dqi_details = validation["dqi"]
        col1, col2, col3 = st.columns([1, 2, 2])
        score_color = "green" if dqi_score >= 70 else (
            "orange" if dqi_score >= 50 else "red")
        col1.markdown(
            f"<h2 style='color:{score_color}; font-size:48px; margin:0'>{dqi_score:.1f}%</h2>"
            f"<p style='font-size:16px;'><b>{dqi_details['dqi_grade']}</b></p>",
            unsafe_allow_html=True,
        )
        with col2:
            st.markdown(f"**{dqi_details['completeness_desc']}**")
            st.markdown(f"**{dqi_details['uniqueness_desc']}**")
            st.markdown(f"**{dqi_details['no_constants_desc']}**")
            st.markdown(f"**{dqi_details['type_consistency_desc']}**")
        with col3:
            st.markdown(f"**{dqi_details['no_outliers_desc']}**")
            st.markdown(f"**{dqi_details['cardinality_desc']}**")
            st.markdown(f"**{dqi_details['row_completeness_desc']}**")

        weights_df = pd.DataFrame([
            {"Метрика": "Заполненность", "Вес": "30%", "Оценка": f"{dqi_details['completeness']:.1f}%",
             "Вклад": f"{dqi_details['completeness'] * 0.30:.2f}%"},
            {"Метрика": "Уникальность (нет дубликатов)", "Вес": "15%", "Оценка": f"{dqi_details['uniqueness']:.1f}%",
             "Вклад": f"{dqi_details['uniqueness'] * 0.15:.2f}%"},
            {"Метрика": "Нет констант", "Вес": "10%", "Оценка": f"{dqi_details['no_constants']:.1f}%",
             "Вклад": f"{dqi_details['no_constants'] * 0.10:.2f}%"},
            {"Метрика": "Целостность типов", "Вес": "15%", "Оценка": f"{dqi_details['type_consistency']:.1f}%",
             "Вклад": f"{dqi_details['type_consistency'] * 0.15:.2f}%"},
            {"Метрика": "Нет выбросов", "Вес": "10%", "Оценка": f"{dqi_details['no_outliers']:.1f}%",
             "Вклад": f"{dqi_details['no_outliers'] * 0.10:.2f}%"},
            {"Метрика": "Кардинальность", "Вес": "10%", "Оценка": f"{dqi_details['cardinality']:.1f}%",
             "Вклад": f"{dqi_details['cardinality'] * 0.10:.2f}%"},
            {"Метрика": "Строки без пропусков", "Вес": "10%", "Оценка": f"{dqi_details['row_completeness']:.1f}%",
             "Вклад": f"{dqi_details['row_completeness'] * 0.10:.2f}%"},
        ])
        st.dataframe(weights_df, use_container_width=True)

    mixed_df = validation["mixed_types"]
    with st.expander("Структурные ошибки (смешанные типы)", expanded=not mixed_df.empty):
        if not mixed_df.empty:
            st.warning(
                f"Обнаружено **{len(mixed_df)}** колонок со смешанными типами.")
            st.dataframe(mixed_df, use_container_width=True)
        else:
            st.success("Смешанных типов не обнаружено.")

    date_df = validation["date_columns"]
    with st.expander("Колонки с датами", expanded=not date_df.empty):
        if not date_df.empty:
            st.dataframe(date_df, use_container_width=True)
        else:
            st.info("Колонок с датами не обнаружено.")

    patterns_df = validation["special_patterns"]
    with st.expander("Специальные паттерны (телефоны, email, URL, даты)", expanded=not patterns_df.empty):
        if not patterns_df.empty:
            st.dataframe(patterns_df, use_container_width=True)
            total_phones = patterns_df["Телефоны"].sum()
            total_emails = patterns_df["Email"].sum()
            total_urls = patterns_df["URL"].sum()
            st.caption(
                f"Всего: телефонов {total_phones}, email {total_emails}, URL {total_urls}")
        else:
            st.info("Специальных паттернов не обнаружено.")

    miss_patterns = validation["missing_patterns"]
    with st.expander("Паттерны пропусков", expanded=False):
        if "row_summary" in miss_patterns and not miss_patterns["row_summary"].empty:
            st.write("**Распределение пропусков по строкам:**")
            st.dataframe(miss_patterns["row_summary"],
                         use_container_width=True)
        if "pair_corr" in miss_patterns and not miss_patterns["pair_corr"].empty:
            st.write("**Попарная корреляция NaN (только |r| > 0.3):**")
            st.dataframe(miss_patterns["pair_corr"], use_container_width=True)
        if "top_missing_rows" in miss_patterns and not miss_patterns["top_missing_rows"].empty:
            st.write("**Строки с наибольшим числом пропусков:**")
            st.dataframe(
                miss_patterns["top_missing_rows"], use_container_width=True)

    return mixed_df, date_df, patterns_df, miss_patterns


def create_pdf_review(df_list: list, filename: str):
    paragraph_list = []
    text_styles = config.TextStyle('TNR', 'TNR.ttf')

    doc = SimpleDocTemplate(filename, pagesize=A4)
    p1 = Paragraph(f'Аналитический отчет по файлу {filename}'.replace(
        "\n", "<br />"), text_styles.centered_style(16))
    paragraph_list.append(p1)

    for i in df_list:
        paragraphs = util.create_title_with_df(i)
        paragraph_list.extend(paragraphs)

    doc.build(paragraph_list,)


def main(df, info):
    st.subheader("Информация о файле")
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Имя файла", info.get("file_name", "—"))
    col2.metric("Формат", info.get("format", "—"))
    col3.metric("Строк", df.shape[0])
    col4.metric("Столбцов", df.shape[1])

    if info.get("separator"):
        st.caption(
            f"Разделитель: `{info['separator']}` | Кодировка: `{info.get('encoding', '—')}` | "
            f"Заголовок сгенерирован: {'Да' if info.get('header_generated') else 'Нет'}"
        )

    with st.spinner("Расчёт статистики и проверок..."):
        results = analysis.summarize(df)
    prof_df = results["profile"]
    show_charts = st.checkbox("Показать графики", value=False)

    st.subheader("Просмотр первых строк")
    st.dataframe(df.head(5))

    basic_overview(results)
    count_values_by_column(df)
    if show_charts:
        drop_visualization(prof_df)
        histogram_distributions(df)
        corr_visaulization(df)
        with st.expander("Матрица пропусков (тепловая карта)"):
            fig = util.plot_missing_matrix(df)
            if fig is None:
                st.success("Пропуски отсутствуют.")
            else:
                st.pyplot(fig)
    data_quality_section(results["quality"])
    missing_plan = missing_analysis(results["missing"])
    validation_section(results["validation"])
    outliers_iqr_search(df, results["iqr"], show_charts)
    ifo_outliers = outliers_isolation_forest_search(df)
    duplicate_search(df)

    st.subheader("Экспорт")
    base_name = info.get("file_name", "data")
    excel_name = base_name.rsplit(".", 1)[0] + "_report.xlsx"

    if not st.button("Подготовить Excel-отчёт"):
        return
    quality_df = results["quality"]
    miss_df = missing_plan
    desc_df = results["describe"]
    iqr_df = results["iqr"]
    mixed_df = results["validation"]["mixed_types"]
    date_df = results["validation"]["date_columns"]
    patterns_df = results["validation"]["special_patterns"]

    excel_buffer = util.export_excel_report(
        df, prof_df, quality_df, miss_df, iqr_df, ifo_outliers, desc_df, info,
        mixed_types_df=mixed_df, date_columns_df=date_df,
        special_patterns_df=patterns_df,
        analysis_settings={
            "Доля выбросов Isolation Forest": st.session_state.get("if_contamination", 0.05),
            "Строк с полными числовыми данными": len(df.select_dtypes(include=[np.number]).dropna()),
            "История очистки": "; ".join(st.session_state.get("changes", [])) or "Нет действий",
        },
    )
    st.download_button(
        label="Скачать Excel-отчёт",
        data=excel_buffer,
        file_name=excel_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        on_click="ignore",
    )
