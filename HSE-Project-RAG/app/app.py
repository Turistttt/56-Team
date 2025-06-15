"""
FastAPI application for the RAG system.
This module provides the API endpoints for interacting with the RAG system.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from asyncio import get_running_loop
from functools import partial

from ragpipe import RAGPipeline
from app.config.settings import DEFAULT_LLM_MODEL, DEFAULT_ENCODER_MODEL


# Initialize the FastAPI application
app = FastAPI(
    title="RAG API",
    description="API for the Retrieval-Augmented Generation system",
    version="1.0.0",
)

# Initialize the RAG system
rag_system = RAGPipeline(
    llm_model_name=DEFAULT_LLM_MODEL,
    encoder_model_name=DEFAULT_ENCODER_MODEL,
)


class QueryRequest(BaseModel):
    """
    Request model for the /query endpoint.

    Attributes:
        message: The user's query message.
        num_results: The number of results to retrieve. Defaults to 2.
    """
    message: str
    num_results: int = 2


class QueryResponse(BaseModel):
    """
    Response model for the /query endpoint.

    Attributes:
        generated_text: The generated text response from the RAG system.
    """
    generated_text: str


@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest) -> QueryResponse:
    """
    Endpoint to query the RAG system and retrieve a generated response.

    Args:
        request: The request object containing the user's query and number of results.

    Returns:
        The response object containing the generated text.
    """
    # Run the synchronous RAG query in an executor to avoid blocking the event loop
    result = await get_running_loop().run_in_executor(
        None,  # Use the default executor
        partial(rag_system.query, request.message, request.num_results)
    )

    return QueryResponse(generated_text=result)