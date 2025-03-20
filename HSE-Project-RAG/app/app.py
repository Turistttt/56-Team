from fastapi import FastAPI
from pydantic import BaseModel
from asyncio import get_running_loop
from functools import partial

from ragpipe import RAGPipeline

# Initialize the FastAPI application
app = FastAPI()

# Initialize the RAG system
rag_system = RAGPipeline(
    llm_model_name="meta-llama/Llama-3.2-3B-Instruct",
    encoder_model_name="sentence-transformers/all-MiniLM-L6-v2",
)


# Define request and response models
class QueryRequest(BaseModel):
    """
    Request model for the /query endpoint.

    Attributes:
        message (str): The user's query message.
        num_results (int): The number of results to retrieve. Defaults to 2.
    """
    message: str
    num_results: int = 2


class QueryResponse(BaseModel):
    """
    Response model for the /query endpoint.

    Attributes:
        generated_text (str): The generated text response from the RAG system.
    """
    generated_text: str


@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    """
    Endpoint to query the RAG system and retrieve a generated response.

    Args:
        request (QueryRequest): The request object containing the user's query and number of results.

    Returns:
        QueryResponse: The response object containing the generated text.
    """
    # Run the synchronous RAG query in an executor to avoid blocking the event loop
    result = await get_running_loop().run_in_executor(
        None,  # Use the default executor
        partial(rag_system.query, request.message, request.num_results)
    )

    # Extract the generated text from the result
    generated_text = result[0]['generated_text']
    return QueryResponse(generated_text=generated_text)