from textwrap import dedent

import pandas as pd
from transformers import pipeline
from utils.config import Config
from utils.create_functions import create_llm, create_retriever, create_client
from qdrant_client import models

# Из-за аннотации типов pylint выдает ошибку какую-то
class RagProject:
    """
    Основной класс проекта,
    Принимает вход модели, осуществяет поиск в векторной БД,
    Создает промпт и выдает с помощью найденных пассажей ответ.
    """

    def __init__(self, config: type[Config] = Config):
        self.config = config
        self.model, self.tokenizer = create_llm(self.config)

        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )
        self.encoder = create_retriever(self.config)
        self.client = create_client(self.config)

    def recreate_collection(self, dataframe: pd.DataFrame) -> None:
        self.client.recreate_collection(
            collection_name="passages",
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )

        self.client.upload_records(
            collection_name="passages",
            records=[
                models.Record(
                    id=idx,
                    vector=self.encoder.encode(doc, device="cuda").tolist(),
                    payload={"idx": idx, "doc": doc},
                )
                for idx, doc in enumerate(dataframe["context"])
            ],
        )

    def retrieve_context(self, query: str, n_passages: int) -> str:
        """
        Метод осуществяющий поиск в векторной БД
        :param query:
        :param n_passages:
        :return:
        """
        hits = self.client.search(
            collection_name="passages",
            query_vector=self.encoder.encode(
                query
            ).tolist(),  # =self.encoder.embed(query).tolist()
            limit=n_passages,
        )

        return "\n---\n".join([f"{hit.payload['doc']}\n" for hit in hits])

    def create_user_prompt(self, query: str, n_passages: int) -> str:
        """
        Метод создающий промпт
        :param query: запрос user'a
        :param n_passages: количество пассажей для поиска в векторной БД
        :return:
        """
        return dedent(
            f"""
            Use the following information:

            ```
            {self.retrieve_context(query,
                                   n_passages)}
            ```

            to answer the question, if you are not able to find any relevant
             information try to answer without it, by yourself:
            {query}
                """
        )

    def query(self, query: str, n_passages: int = 2) -> str:
        """
        Метод возвращающий ответ LLM на основе найденных пассажей
        :param query: запрос user'a
        :param n_passages: количество пассажей для поиска в векторной БД
        :return: Ответ LLM
        """
        prompt = self.create_user_prompt(query, n_passages)
        messages = [
            {
                "role": "system",
                "content": self.config.LLM_MODEL.SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]
        answer = self.pipeline(
            text_inputs=messages, **self.config.LLM_MODEL.GENERATION_PARAMS
        )

        return answer
