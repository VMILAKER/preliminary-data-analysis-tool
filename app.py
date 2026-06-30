import numpy as np
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate

import config
import utilities as util


def use_pygwalker(df):
    with st.expander("Интерактивное исследование данных (PyGWalker)", expanded=True):

        @st.cache_resource
        def get_pygwalker_renderer(data):
            return StreamlitRenderer(data, spec_io_mode="rw")

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


def basic_overview(df):
    with st.expander("Базовая информация (df.info)", expanded=False):
        info_text = util.get_df_info(df)
        st.text(info_text)

    with st.expander("Статистическое описание (df.describe)", expanded=False):
        desc_df = util.get_df_describe(df)
        st.dataframe(desc_df, use_container_width=True)


def histogram_distributions(df):
    with st.expander("Гистограммы распределения числовых колонок", expanded=False):
        fig = util.plot_histograms(df)
        if fig is None:
            st.info("Нет числовых колонок для построения гистограмм.")
        else:
            st.pyplot(fig)


def data_quality_section(df):
    st.subheader("Отчёт о качестве данных")
    quality_df = util.data_quality_report(df)
    st.dataframe(quality_df, use_container_width=False)

    total_problems = quality_df[quality_df["Проблемы"] != "—"].shape[0]
    total_cols = len(quality_df)
    if total_problems > 0:
        st.warning(
            f"Обнаружено **{total_problems}** колонок с потенциальными проблемами из **{total_cols}**.")
    else:
        st.success(f"Все **{total_cols}** колонок не имеют явных проблем.")


def missing_analysis(df):
    with st.expander("Матрица пропусков (тепловая карта)", expanded=False):
        fig = util.plot_missing_matrix(df)
        if fig is None:
            st.success("Пропуски отсутствуют — матрица пуста.")
        else:
            st.pyplot(fig)

    with st.expander("Статистика и стратегия заполнения пропусков", expanded=False):
        miss_df = util.missing_values_statistics(df)
        if miss_df.empty:
            st.success("Пропуски отсутствуют.")
        else:
            st.dataframe(miss_df, use_container_width=True)


def duplicate_search(df):
    st.subheader("Поиск дубликатов")
    dup_mask = df.duplicated(keep=False)
    total_dups = dup_mask.sum()
    st.write(
        f"**Полных дубликатов (повторяющиеся строки):** {total_dups} "
        f"(из них уникальных групп: {df.duplicated(keep='first').sum()})"
    )

    if total_dups > 0:
        st.dataframe(df[dup_mask].head())
        if st.button("Удалить все дубликаты (оставить первые вхождения)"):
            st.session_state.df.drop_duplicates(keep="first", inplace=True)
            st.rerun()
    else:
        st.success("Дубликаты не найдены.")


def outliers_iqr_search(df):
    st.subheader("Обнаружение выбросов (IQR)")
    st.write(
        "IQR рассчитываются по формуле IQR = Q3 − Q1, где Q3- третий квартиль, Q1- первый квартиль")
    outliers_df = util.detect_outliers_iqr(df)
    if outliers_df.empty:
        st.info("Нет колонок, удовлетворяющих условиям IQR.")
    else:
        st.dataframe(outliers_df)
        with st.expander("Обнаружение выбросов с помощью Boxplot", expanded=False):
            numeric_cols = [
                column for column in df.select_dtypes(include=[np.number]).columns if df[column].nunique() >= 5
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
        return

    rows_before = len(num_full)
    rows_total = len(df)
    st.caption(f"Строк с полными числовыми данными: {rows_before} из {rows_total} "
               f"(строк с пропусками: {rows_total - rows_before})")

    contamination = st.slider(
        "Доля выбросов (contamination)",
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        key="if_contamination",
    )

    ifo_outliers = util.detect_outliers_isolation_forest(
        df, contamination=contamination
    )

    if ifo_outliers.empty:
        st.success("Isolation Forest не обнаружил выбросов.")
    else:
        pct = round(len(ifo_outliers) / rows_before * 100, 2)
        st.write(
            f"**Найдено выбросов:** {len(ifo_outliers)} из {rows_before} ({pct}%)")
        st.dataframe(ifo_outliers.head(100), use_container_width=True)

        if st.button("Удалить найденные выбросы (Isolation Forest)"):
            st.session_state.df.drop(
                index=ifo_outliers.index, inplace=True)
            if st.session_state.df.empty:
                st.warning("Все строки удалены. Загрузите новый файл.")
            st.rerun()


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

    prof_df = util.profile_dataframe(df)

    st.subheader("Просмотр первых строк")
    st.dataframe(df.head(5))

    basic_overview(df)
    count_values_by_column(df)
    drop_visualization(prof_df)
    histogram_distributions(df)
    data_quality_section(df)
    missing_analysis(df)
    corr_visaulization(df)
    outliers_iqr_search(df)
    outliers_isolation_forest_search(df)
    duplicate_search(df)

    st.subheader("Экспорт")
    base_name = info.get("file_name", "data")
    excel_name = base_name.rsplit(".", 1)[0] + "_report.xlsx"

    # Собираем все датафреймы для экспорта
    quality_df = util.data_quality_report(df)
    miss_df = util.missing_values_statistics(df)
    desc_df = util.get_df_describe(df)
    iqr_df = util.detect_outliers_iqr(df)
    ifo_outliers = util.detect_outliers_isolation_forest(df)

    excel_buffer = util.export_excel_report(
        df, prof_df, quality_df, miss_df, iqr_df, ifo_outliers, desc_df, info
    )
    st.download_button(
        label="Скачать полный Excel-отчёт (8 листов)",
        data=excel_buffer,
        file_name=excel_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
