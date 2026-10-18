from hashlib import sha256

import streamlit as st

import src.utilities as util
from src import app
from src.session import clear_dataset, start_dataset, undo_change, restore_original


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
        json_mode = "text"
        if uploaded_file is not None and uploaded_file.name.lower().endswith(".json"):
            json_mode = st.radio(
                "Вложенные поля JSON", ["text", "flatten"],
                format_func=lambda mode: "Преобразовать в текст" if mode == "text" else "Раскрыть в столбцы",
            )

    # Access to DataFrame
    if uploaded_file is None:
        st.info("Загрузите файл через боковую панель, чтобы начать анализ.")
        clear_dataset(st.session_state)
        return

    source_id = (uploaded_file.name, sha256(uploaded_file.getvalue()).hexdigest(), json_mode)
    if (
        "df" not in st.session_state
        or st.session_state.get("source_id") != source_id
    ):
        with st.spinner("Загрузка и анализ файла..."):
            result = util.load_dataframe(uploaded_file, json_mode=json_mode)

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
        app.use_pygwalker(df)

    st.caption("© Copyright Ilya Myshakov, 2026")


if __name__ == "__main__":
    configure_streamlit()
