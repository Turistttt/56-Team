import streamlit as st
from pages.eda_page import eda_page
from pages.rag_page import rag_page

# Конфигурация страницы Streamlit
st.sidebar.title("Навигация")
page = st.sidebar.selectbox("Выберите страницу", ("Реализация RAG", "EDA"))

if page == "Реализация RAG":
    rag_page()
elif page == "EDA":
    eda_page()
