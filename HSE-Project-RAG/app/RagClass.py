from textwrap import dedent
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from extra import SYSTEM_PROMPT, quadrant_client_path
#from fastembed import SparseTextEmbedding


class RagProject:
    """
    Основной класс проекта,
    Принимает вход модели, осуществяет поиск в векторной БД,
    Создает промпт и выдает с помощью найденных пассажей ответ.
    """
    def __init__(self,
                 llm_model_name : str =  "meta-llama/Llama-3.2-3B-Instruct",
                 encoder_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 ):

        self.model =  AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name,
                                                  add_eos_token=True,
                                                  add_pad_token=True,
                                                  use_fast=True,
                                                  device_map="auto",
                                                  attn_implementation="flash_attention_2",
                                                  torch_dtype=torch.bfloat16,
                                                  )
        self.pipeline  = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )
        self.encoder = SentenceTransformer(encoder_model_name)
        #self.encoder = SparseTextEmbedding(model_name="Qdrant/bm25")

        self.client = QdrantClient(path=quadrant_client_path)
        self.generation_params =  {
            "max_new_tokens":1024,
            'temperature':0.9,
            'num_return_sequences':1,
            'num_beams':5,
            'no_repeat_ngram_size':2,
            'do_sample':True,
            'top_k':50,
            'top_p':0.95
        }

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
            query_vector=self.encoder.encode(query).tolist(),#=self.encoder.embed(query).tolist()
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

