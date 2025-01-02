""" Файл с кодом создания страницы с EDA"""

import streamlit as st
import requests
import plotly.graph_objs as go
import numpy as np


# Почему-то pylit ругается на импорт этого класса - закомментил.
#from streamlit.runtime.uploaded_file_manager import  UploadedFile

def send_to_fastapi(file, llm_name: str) -> requests.Response:
    """Функция отправки запроса"""
    url = "http://127.0.0.1:8000/llm-process"
    files = {"file": (file.name, file, "text/csv")}
    data = {"llm_name": llm_name}
    response = requests.post(url, files=files, data=data)
    return response


def eda_page() -> None:
    """Одна из страниц в streamlit приложении - эта с EDA"""
    st.title("Streamlit EDA")

    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    llm_name = st.text_input(
        "Введите название LLM (например, 'sentence-transformers/all-MiniLM-L6-v2')"
    )

    # Инициализация состояния
    if "result" not in st.session_state:
        st.session_state.result = None

    if st.button("Отправить на обработку"):
        if uploaded_file is not None and llm_name:
            try:
                with st.spinner("Отправка файла..."):
                    response = send_to_fastapi(uploaded_file, llm_name)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Файл обработан!")
                        #st.json(result)
                        st.session_state.result = (
                            result  # Сохраняем результат в session_state
                        )
                    else:
                        st.error(f"Ошибка {response.status_code}: {response.content}")
            except Exception as error:
                st.error(f"Произошла ошибка: {str(error)}")
        else:
            st.error("Пожалуйста, загрузите файл и введите название LLM")

    # Проверяем, есть ли результат в session_state
    if st.session_state.result is not None:
        result = st.session_state.result
        bins = st.slider(
            "Выберите количество бинов", min_value=1, max_value=100, value=30
        )

        hist = go.Histogram(
            x=result["data_sample"], nbinsx=bins, name="Данные", opacity=0.75
        )
        hist_vals = np.histogram(result["data_sample"], bins=bins)
        max_y = max(hist_vals[0])
        # Создание вертикальной линии
        line = go.Scatter(
            x=[result["max_len"], result["max_len"]],
            y=[0, max_y],
            mode="lines",
            line={"color":"red", "width":2},
            name="Вертикальная линия",
        )

        # Создание фигуры
        fig = go.Figure(data=[hist, line])

        # Настройка оформления
        fig.update_layout(
            xaxis_title="Значение", yaxis_title="Частота", showlegend=True
        )

        # Отображение графика в Streamlit
        st.plotly_chart(fig)
