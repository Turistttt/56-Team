from fastapi import FastAPI

from pydantic import BaseModel
from RagClass import RagProject

app = FastAPI()

# Initialize the rag_system
rag_system = RagProject("meta-llama/Llama-3.2-3B-Instruct",
                        'sentence-transformers/all-MiniLM-L6-v2')

# Request and Response models
class QueryRequest(BaseModel):
    message: str
    num_results: int = 2

class QueryResponse(BaseModel):
    generated_text: str

@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    from asyncio import get_running_loop
    from functools import partial

    loop = get_running_loop()
    result = await loop.run_in_executor(None, partial(
        rag_system.query, request.message, request.num_results))

    generated_text = result[0]['generated_text']
    return QueryResponse(generated_text=generated_text)
