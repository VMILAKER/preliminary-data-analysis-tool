import streamlit as st

import src.utilities as util
from src import app


def configure_streamlit():
    df = ""
    st.set_page_config(
        page_title="Инструмент для предварительного анализа данных",
        page_icon=None,
        layout="wide",
    )

    st.title("Инструмент для предварительного анализа данных")
    st.markdown(
        "Загрузите CSV или Excel-файл для автоматического профилирования.")

    with st.sidebar:
        st.header("Загрузка файла")
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=["csv", "xlsx", "xls"],
        )

    # Access to DataFrame
    if uploaded_file is None:
        st.info("Загрузите файл через боковую панель, чтобы начать анализ.")
        if "df" in st.session_state:
            del st.session_state.df
        return

    if (
        "df" not in st.session_state
        or st.session_state.get("uploaded_name") != uploaded_file.name
    ):
        with st.spinner("Загрузка и анализ файла..."):
            result = util.load_dataframe(uploaded_file)

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

        st.session_state.df = df
        st.session_state.info = info
        st.session_state.uploaded_name = uploaded_file.name

    df = st.session_state.df
    info = st.session_state.info
    if "state" not in st.session_state:
        st.session_state.state = False

    tab1, tab2 = st.tabs(["Кастомный вариант", "PyGWalker"])
    with tab1:
        app.main(df, info)
    with tab2:
        app.use_pygwalker(df)

    st.caption("© Copyright Ilya Myshakov, 2026")


if __name__ == "__main__":
    configure_streamlit()
