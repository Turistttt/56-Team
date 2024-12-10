from textwrap import dedent
from extra import SYSTEM_PROMPT

class RagProject:
    """
    Основной класс проекта,
    Принимает вход модели, осуществяет поиск в векторной БД,
    Создает промпт и выдает с помощью найденных пассажей ответ.
    """
    def __init__(self, model,
                 tokenizer,
                 client,
                 encoder,
                 pipeline,
                 generation_params,):

        self.model = model
        self.tokenizer = tokenizer
        self.client = client
        self.encoder = encoder
        self.pipeline = pipeline
        self.generation_params = generation_params

    def retrieve_context(self,
                         query: str,
                         n_passages: int) -> str:
        """
        Метод осуществяющий поиск в векторной БД
        :param query:
        :param n_passages:
        :return:
        """
        hits = self.client.search(
            collection_name="passages",
            query_vector=self.encoder.encode(query).tolist(),
            limit=n_passages
        )

        return "\n---\n".join([f"{hit.payload['doc']}\n" for hit in hits])


    def create_user_prompt(self,query: str, n_passages:int) -> str:
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
        
            to answer the question, if you are not able to find any relevant information try to answer without it, by yourself:
            {query}
                """
                )

    def query(self,
              query: str,
              n_passages: int = 2) -> str:
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
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
        answer = self.pipeline(text_inputs = messages, **self.generation_params)
        return answer

