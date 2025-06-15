"""
Embedding models and utilities for the RAG system.
This module provides functionality for generating both dense and sparse embeddings.
"""

from typing import List, Tuple, Any
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from app.config.settings import (
    DEFAULT_ENCODER_MODEL,
    DEFAULT_SPARSE_MODEL,
)


class EmbeddingManager:
    """
    Manages both dense and sparse embedding models for the RAG system.
    
    This class handles the initialization and usage of embedding models,
    providing a unified interface for generating both dense and sparse embeddings.
    """

    def __init__(
        self,
        dense_model_name: str = DEFAULT_ENCODER_MODEL,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
    ) -> None:
        """
        Initialize the embedding manager with specified models.

        Args:
            dense_model_name: Name of the dense embedding model to use.
            sparse_model_name: Name of the sparse embedding model to use.
        """
        self.dense_model = SentenceTransformer(dense_model_name)
        self.sparse_model = SparseTextEmbedding(sparse_model_name)

    def get_embeddings(self, text: str) -> Tuple[List[float], Any]:
        """
        Generate both dense and sparse embeddings for the given text.

        Args:
            text: The input text to generate embeddings for.

        Returns:
            A tuple containing:
            - List of floats representing the dense embedding
            - Sparse embedding object
        """
        dense_embedding = list(self.dense_model.encode(text))
        sparse_embedding = next(iter(self.sparse_model.embed(text)))
        
        return dense_embedding, sparse_embedding 