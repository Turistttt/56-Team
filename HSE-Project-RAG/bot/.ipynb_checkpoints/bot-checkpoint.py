import os

import httpx
import numpy as np
import telebot
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
TOKEN = os.getenv('TOKEN')
FASTAPI_URL = os.getenv('FASTAPI_URL')

# Initialize the Telegram bot
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    """
    Handles the /start command. Sends a welcome message to the user.

    Args:
        message: The message object from the user.
    """
    bot.reply_to(message, "Hello, how can I help you?")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    """
    Handles all incoming messages. Sends a random "thinking" message, makes a request to the FastAPI server,
    and replies with the generated text.

    Args:
        message: The message object from the user.
    """
    # Send a random "thinking" message
    bot.send_message(message.chat.id, np.random.choice(["Thinking ...", "Looking for an answer..."]))

    try:
        # Prepare the payload for the FastAPI request
        payload = {
            "message": message.text,
            "num_results": 2
        }

        # Make a synchronous HTTP POST request to the FastAPI server
        with httpx.Client() as client:
            response = client.post(FASTAPI_URL, json=payload, timeout=120)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            generated_text = data.get('generated_text', 'No response received')

            # Reply to the user with the generated text
            bot.reply_to(message, generated_text)

    except httpx.HTTPError as exc:
        # Handle any HTTP errors and notify the user
        bot.send_message(message.chat.id, f"An error occurred: {exc}")


if __name__ == '__main__':
    print("Bot starting...")
    # Start the bot and keep it running
    bot.infinity_polling()