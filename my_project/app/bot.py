
from project_functions import RagProject
import telebot

from qdrant_client import QdrantClient
from extra import TOKEN, model, tokenizer, encoder, pipe, generation_params
import numpy as np


client = QdrantClient(path="./")
bot = telebot.TeleBot(TOKEN)


rag_system = RagProject(model=model,
                        tokenizer=tokenizer,
                        client=client,
                        encoder = encoder,
                        pipeline=pipe,
                        generation_params=generation_params)


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