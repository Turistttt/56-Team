""" Файл с кодом создания страницы с RAG"""

import streamlit as st
import requests


def rag_page() -> None:
    """ Страница streamlit приложения с реализацией RAG"""
    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False

    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>🤖 Чат с LLM на основе вашего CSV файла</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("## 📥 Загрузите CSV файл")

    uploaded_file = st.file_uploader("", type="csv")

    if not st.session_state["file_uploaded"] and uploaded_file is not None:
        files = {"file": uploaded_file}
        url = "http://127.0.0.1:8000/load/"  # Адрес вашего FastAPI сервера
        # Отправка POST запроса с файлом
        response = requests.post(url, files=files).json()  # .get('message')
        print(response)
        st.success("✅ Файл успешно загружен и обработан!")
        st.session_state["file_uploaded"] = True

    if st.session_state["file_uploaded"]:
        if st.button("🔄 Загрузить новый файл"):
            st.session_state["file_uploaded"] = False
            st.session_state["history"] = []
            st.experimental_rerun()

    # Инициализация истории сообщений
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Отображение чата
    if True:
        st.markdown("---")
        st.markdown("## 💬 Чат")

        # Отображение истории сообщений
        for _, chat in enumerate(st.session_state["history"]):
            if chat["role"] == "user":
                with st.chat_message("user", avatar="bot_icon.png"):
                    st.write(chat["content"])
            else:
                with st.chat_message("bot", avatar="user_icon.png"):
                    st.write(chat["content"])

        # Поле для ввода вопроса
        question = st.text_input("Задайте вопрос LLM:")

        if st.button("Отправить") and question:
            # Добавление вопроса пользователя в историю
            st.session_state["history"].append({"role": "user", "content": question})

            # Получение ответа от LLM
            url = "http://127.0.0.1:8000/llm_query/"

            response = (
                requests.post(url, json={"user_message": question})
                .json()
                .get("message")
            )
            # Добавление ответа бота в историю

            st.session_state["history"].append(
                {"role": "assistant", "content": response}
            )

            # Обновление страницы для отображения новых сообщений
            st.rerun()

    else:
        st.info("Пожалуйста, загрузите CSV файл для начала.")

    # Стилизация с помощью CSS
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color:white;
        }
        .stTextInput>div>div>input {
            background-color: #f1f1f1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
