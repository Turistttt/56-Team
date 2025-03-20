FROM python:3.10-slim

ENV HF_TOKEN="TOKEN"

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY app.py .
COPY ragpipe.py .

# Пробрасываем порт (замени на порт, который использует твой FastAPI сервер, обычно 8000)
EXPOSE 8000

# Запускаем приложение с помощью Uvicorn
CMD ["uvicorn", "fastapy:app", "--host", "0.0.0.0", "--port", "8000"]
