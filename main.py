from hashlib import sha256
from io import BytesIO

import pandas as pd
import streamlit as st

import src.utilities as util
from src import app
from src.session import clear_dataset, start_dataset, undo_change, restore_original


@st.cache_data(show_spinner=False, max_entries=4, ttl=600)
def read_upload(name, contents, options):
    source = BytesIO(contents)
    source.name = name
    return util.load_dataframe(source, **options)


@st.cache_data(show_spinner=False, max_entries=4, ttl=600)
def excel_sheets(contents, extension):
    with pd.ExcelFile(BytesIO(contents), engine="openpyxl" if extension == "xlsx" else "xlrd") as book:
        return book.sheet_names


def configure_streamlit():
    st.set_page_config(
        page_title="Инструмент для предварительного анализа данных",
        page_icon=None,
        layout="wide",
    )

    st.title("Инструмент для предварительного анализа данных")
    st.markdown(
        "Загрузите CSV, Excel-файл или JSON для автоматического профилирования.")

    with st.sidebar:
        st.header("Загрузка файла")
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=["csv", "xlsx", "xls", "json"],
        )
        options = {}
        if uploaded_file is not None and uploaded_file.name.lower().endswith(".json"):
            options["json_mode"] = st.radio(
                "Вложенные поля JSON", ["text", "flatten"],
                format_func=lambda mode: "Преобразовать в текст" if mode == "text" else "Раскрыть в столбцы",
            )
        if uploaded_file is not None:
            extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
            if extension in ("csv", "xlsx", "xls"):
                options["header"] = st.selectbox(
                    "Заголовок", ["auto", "present", "absent"],
                    format_func=lambda value: {"auto": "Определить автоматически", "present": "Первая строка — заголовок", "absent": "Заголовка нет"}[value],
                )
            if extension == "csv":
                separator = st.selectbox("Разделитель", ["Авто", ",", ";", "Табуляция", "|"])
                options["separator"] = None if separator == "Авто" else "\t" if separator == "Табуляция" else separator
                encoding = st.selectbox("Кодировка", ["Авто", "utf-8-sig", "utf-8", "cp1251", "windows-1252", "latin1", "utf-16"])
                options["encoding"] = None if encoding == "Авто" else encoding
            if extension in ("xlsx", "xls"):
                try:
                    sheets = excel_sheets(uploaded_file.getvalue(), extension)
                    options["sheet_name"] = st.selectbox("Лист Excel", sheets)
                except Exception as exc:
                    st.error(f"Не удалось открыть книгу Excel: {exc}")
                    return

    # Access to DataFrame
    if uploaded_file is None:
        st.info("Загрузите файл через боковую панель, чтобы начать анализ.")
        clear_dataset(st.session_state)
        return

    source_id = (uploaded_file.name, sha256(uploaded_file.getvalue()).hexdigest(), tuple(sorted(options.items())))
    if (
        "df" not in st.session_state
        or st.session_state.get("source_id") != source_id
    ):
        with st.spinner("Загрузка и анализ файла..."):
            result = read_upload(uploaded_file.name, uploaded_file.getvalue(), options)

        if result is None or (isinstance(result, tuple) and result[0] is None):
            error_msg = (
                result[1] if isinstance(
                    result, tuple) else "Неизвестная ошибка."
            )
            st.error(error_msg)
            return

        df, info = result

        if df is None or df.empty:
            st.error("Файл не содержит данных.")
            return

        st.subheader("Предпросмотр загрузки")
        st.caption(f"Строк: {len(df)}, столбцов: {len(df.columns)}. Проверьте настройки чтения в боковой панели.")
        st.dataframe(df.head(5), width="stretch")
        if not st.button("Начать анализ"):
            return
        start_dataset(st.session_state, df, info, source_id)

    with st.sidebar:
        st.subheader("История очистки")
        for change in st.session_state.changes:
            st.caption(change)
        if st.button("Отменить последнее действие", disabled=not st.session_state.history):
            undo_change(st.session_state)
            st.rerun()
        if st.button("Вернуть исходные данные", disabled=not st.session_state.changes):
            restore_original(st.session_state)
            st.rerun()

    df = st.session_state.df
    info = st.session_state.info
    if df.empty:
        st.warning("После очистки строк не осталось. Отмените действие или верните исходные данные.")
        return

    tab1, tab2 = st.tabs(["Кастомный вариант", "PyGWalker"])
    with tab1:
        app.main(df, info)
    with tab2:
        if st.checkbox("Открыть интерактивное исследование PyGWalker"):
            app.use_pygwalker(df)

    st.caption("© Copyright Ilya Myshakov, 2026")


if __name__ == "__main__":
    configure_streamlit()
