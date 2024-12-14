
from RagClass import RagProject
import telebot

from extra import TOKEN
import numpy as np


bot = telebot.TeleBot(TOKEN)


rag_system = RagProject("meta-llama/Llama-3.2-3B-Instruct",
                        'sentence-transformers/all-MiniLM-L6-v2')

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello, how can I help you?")



@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.send_message(message.chat.id, np.random.choice(["Thinking ...","Looking for an answer..."]))
    bot.reply_to(message, rag_system.query(message.text,2)[0]['generated_text'])

if __name__ == '__main__':
    print("Bot starting...")
    bot.infinity_polling()