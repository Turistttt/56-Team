import telebot
import numpy as np
import httpx
from dotenv import load_dotenv

import os

load_dotenv()

TOKEN = os.getenv('TOKEN')
FASTAPI_URL = os.getenv('FASTAPI_URL')

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello, how can I help you?")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    # Переходим от асинхронного кода к синхронному для `httpx`
    bot.send_message(message.chat.id, np.random.choice(["Thinking ...", "Looking for an answer..."]))

    # Используем синхронный клиент
    try:
        payload = {
            "message": message.text,
            "num_results": 2
        }

        # Делаем обычный HTTP-запрос без async/await
        with httpx.Client() as client:
            response = client.post(FASTAPI_URL, json=payload,timeout = 120  )
            response.raise_for_status()
            data = response.json()
            generated_text = data.get('generated_text', 'No response received')

            bot.reply_to(message, generated_text)

    except httpx.HTTPError as exc:
        bot.send_message(message.chat.id, f"An error occurred: {exc}")


if __name__ == '__main__':
    print("Bot starting...")
    bot.infinity_polling()
