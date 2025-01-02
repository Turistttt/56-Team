"""
Файл с функциями, которые возвращают генеративные/ретривер модели + клиент векторной бд
"""
from typing import Union, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
from utils.config import Config

# Из-за аннотации типов pylint выдает ошибку какую-то
def create_llm(
    config: Config = Config,
    only_tokenizer: bool = False
) -> Union[PreTrainedTokenizer, Tuple[PreTrainedModel, PreTrainedTokenizer]]:
    """
    Функция возвращает генеративную модель и токенайзер к ней.
    В зависимости от флага only_tokenizer, функция может вернуть только токенайзер.
    (Например, для EDA части приложения, где проводится аналитика над токенайзером)
    """
    if only_tokenizer:
        return AutoTokenizer.from_pretrained(
            config.LLM_MODEL.MODEL,
            add_eos_token=True,
            add_pad_token=True,
            use_fast=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL.MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=config.LLM_MODEL.CACHE_DIR,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL.MODEL,
        add_eos_token=True,
        add_pad_token=True,
        use_fast=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    return model, tokenizer


def create_retriever(config: Config) -> SentenceTransformer:
    """
    Функция возвращает модель-encoder.
    """
    encoder = SentenceTransformer(config.EMBEDDINGS.MODEL)

    return encoder


def create_client(config: Config) -> QdrantClient:
    """
    Функция возвращает клиент векторной БД.
    """
    return QdrantClient(path=config.QDRANT.CLIENT_PATH)
