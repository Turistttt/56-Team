"""
Main RAG pipeline implementation.
This module provides the main RAGPipeline class that orchestrates the entire RAG system.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.core.agents import AgentManager
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStore
from app.config.settings import (
    DEFAULT_LLM_MODEL,
    DEFAULT_ENCODER_MODEL,
)

load_dotenv()


class RAGPipeline:
    """
    A class for implementing a Retrieval-Augmented Generation (RAG) pipeline using LangGraph.
    This class integrates a language model for text generation and a vector database
    for retrieving relevant passages to enhance the model's responses.
    """

    def __init__(
        self,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    ) -> None:
        """
        Initialize the RAGPipeline with specified language and encoder models.

        Args:
            llm_model_name: Name of the language model to use for text generation.
            encoder_model_name: Name of the encoder model to use for embedding passages.
        """
        # Initialize components
        self.embedding_manager = EmbeddingManager(encoder_model_name)
        self.vector_store = VectorStore(os.getenv("quadrant_client_path"))
        self.vector_store.embedding_manager = self.embedding_manager
        
        # Initialize agent system
        self.agent_manager = AgentManager(
            llm_model_name=llm_model_name,
        )
        self.workflow = self.agent_manager.setup_workflow(self.vector_store)

    def query(self, query: str, n_passages: int = 2) -> str:
        """
        Generates a response to the user's query using the agent system.

        Args:
            query: The user's query.
            n_passages: The number of passages to retrieve and use for generating the response.

        Returns:
            The generated response from the language model.
        """
        # Create the initial message
        messages = [HumanMessage(content=query)]
        
        # Run the workflow
        result = self.workflow.invoke({"messages": messages})
        
        # Return the final answer
        return result["messages"][-1].content