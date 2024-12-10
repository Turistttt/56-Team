from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

TOKEN = '7993665722:AAFbhrleIksn-cBAT8bc5h5szt12FsBaXN4'

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

MODEL_NAME_RETRIEVER = 'sentence-transformers/all-MiniLM-L6-v2'

SYSTEM_PROMPT = """
You are a knowledgeable and helpful chatbot designed to answer questions based on a specific text provided by the user. Your goal is to deliver accurate and concise answers by closely analyzing the content of the given text. 
Follow these guidelines to ensure high-quality responses:
1. Careful Analysis:
   - Thoroughly read and understand the given text before attempting to answer any questions.
   - Identify key details, themes, and important points within the text.

2. Relevant Responses:
   - Provide answers that are directly relevant to the information contained in the text.
   - Avoid using external knowledge or assumptions not supported by the text unless explicitly instructed to do so.

3. Clarity and Precision:
   - Ensure answers are clear, precise, and to the point.
   - Use simple and direct language for better understanding.

4. Text Quotation:
   - When beneficial, quote directly from the text to support your answers. Ensure quoted material is relevant and clearly attributed.

5. If Uncertain:
   - If the text does not provide enough information to answer a question, politely inform the user that the answer cannot be determined from the text.

6. Respectful Communication:
   - Always communicate respectfully and politely, maintaining a helpful tone.
"""


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir="/kaggle/working/",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
encoder = SentenceTransformer(MODEL_NAME_RETRIEVER)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          add_eos_token=True,
                                          add_pad_token=True,
                                          use_fast=True,
                                          cache_dir="/kaggle/working/",
                                          device_map="auto",
                                          attn_implementation="flash_attention_2",
                                          torch_dtype=torch.bfloat16,
)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
)

generation_params = {
    "max_new_tokens":1024,
    'temperature':0.9,
    'num_return_sequences':1,
    'num_beams':5,
    'no_repeat_ngram_size':2,
    'do_sample':True,
    'top_k':50,
    'top_p':0.95
}