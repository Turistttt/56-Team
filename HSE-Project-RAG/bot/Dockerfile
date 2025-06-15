# Используем базовый образ с Python 3.10
FROM python:3.10-slim

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Пробрасываем порт
EXPOSE 8001

# Устанавливаем переменные окружения, если необходимо
# ENV VARIABLE_NAME=value

# Запускаем бота
CMD ["python", "bot.py"]