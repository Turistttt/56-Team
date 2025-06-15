"""
Vector store management for the RAG system.
This module provides functionality for interacting with the Qdrant vector store.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models

from app.config.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_PREFETCH_LIMIT,
)


class VectorStore:
    """
    Manages interactions with the Qdrant vector store.
    
    This class provides methods for querying the vector store using both
    dense and sparse embeddings, with support for hybrid search.
    """

    def __init__(self, client_path: str) -> None:
        """
        Initialize the vector store client.

        Args:
            client_path: Path to the Qdrant client storage.
        """
        self.client = QdrantClient(path=client_path)
        self.collection_name = DEFAULT_COLLECTION_NAME

    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        limit: int = DEFAULT_SEARCH_LIMIT,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search using both dense and sparse embeddings.

        Args:
            dense_vector: The dense embedding vector.
            sparse_vector: The sparse embedding vector.
            limit: Maximum number of results to return.
            prefetch_limit: Maximum number of results to prefetch for each embedding type.

        Returns:
            List of search results with their payloads.
        """
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="sentence-transformers/all-MiniLM-L6-v2",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vector.as_object()),
                using="Qdrant/bm25",
                limit=prefetch_limit,
            ),
        ]

        results = self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=True,
            limit=limit,
        )

        return [point.payload for point in results.points] 