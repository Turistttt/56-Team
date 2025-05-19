import os
from textwrap import dedent

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()


class RAGPipeline:
    """
    A class for implementing a Retrieval-Augmented Generation (RAG) pipeline.
    This class integrates a language model for text generation and a vector database
    for retrieving relevant passages to enhance the model's responses.
    """

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initializes the RAGPipeline with specified language and encoder models.

        Args:
            llm_model_name (str): Name of the language model to use for text generation.
            encoder_model_name (str): Name of the encoder model to use for embedding passages.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            add_eos_token=True,
            add_pad_token=True,
            use_fast=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )
        self.encoder = SentenceTransformer(encoder_model_name)

        self.client = QdrantClient(path=os.getenv("quadrant_client_path"))
        self.generation_params = {
            "max_new_tokens": 1024,
            "temperature": 0.9,
            "num_return_sequences": 1,
            "num_beams": 5,
            "no_repeat_ngram_size": 2,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
        }

    def retrieve_context(self, query: str, n_passages: int) -> str:
        """
        Retrieves relevant passages from the vector database based on the query.

        Args:
            query (str): The query to search for in the vector database.
            n_passages (int): The number of passages to retrieve.

        Returns:
            str: A string containing the retrieved passages separated by a delimiter.
        """
        hits = self.client.search(
            collection_name="passages",
            query_vector=self.encoder.encode(query).tolist(),
            limit=n_passages,
        )

        return "\n---\n".join([f"{hit.payload['doc']}\n" for hit in hits])

    def create_user_prompt(self, query: str, n_passages: int) -> str:
        """
        Constructs a prompt for the language model using the retrieved passages.

        Args:
            query (str): The user's query.
            n_passages (int): The number of passages to retrieve and include in the prompt.

        Returns:
            str: A formatted prompt containing the retrieved passages and the user's query.
        """
        return dedent(
            f"""
            Use the following information:

            ```
            {self.retrieve_context(query, n_passages)}
            ```

            to answer the question, if you are not able to find any relevant information try to answer without it, by yourself:
            {query}
            """
        )

    def query(self, query: str, n_passages: int = 2) -> str:
        """
        Generates a response to the user's query using the language model and retrieved passages.

        Args:
            query (str): The user's query.
            n_passages (int): The number of passages to retrieve and use for generating the response.

        Returns:
            str: The generated response from the language model.
        """
        prompt = self.create_user_prompt(query, n_passages)
        messages = [
            {
                "role": "system",
                "content": os.getenv("SYSTEM_PROMPT"),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        answer = self.pipeline(text_inputs=messages, **self.generation_params)
        return answer