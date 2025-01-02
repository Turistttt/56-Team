# Используем официальный минимальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt ./requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn plotly  # Устанавливаем необходимые пакеты
RUN pip install transformers
# Копируем весь код приложения в контейнер
COPY . .

# Открываем необходимые порты
EXPOSE 8000 
EXPOSE 8501  

# Определяем команду запуска по умолчанию для совместимости 
CMD ["bash"]