import io
from typing import Annotated, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form
import pandas as pd
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from RagClass import RagProject


# Из-за аннотации типов pylint выдает ошибку какую-то
app = FastAPI()
rag_system = RagProject()


class Message(BaseModel):
    user_message: str  # Аннотация для сообщения пользователя


@app.post("/load/")
async def load_csv(file: Annotated[UploadFile, File(...)]) -> JSONResponse:
    """Загрузка и обработка CSV файла"""
    try:
        # Читаем CSV файл из запроса
        contents: bytes = await file.read()
        df: pd.DataFrame = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        rag_system.recreate_collection(df)

        return JSONResponse(content=[{"message": ""}])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)  # Если ошибка, возвращаем JSON с ошибкой


@app.post("/llm_query/")
async def fit_model(message: Message) -> JSONResponse:
    """Обработка запроса к LLM"""
    message_text: str = message.user_message

    llm_answer: str = rag_system.query(message_text, 2)[0]["generated_text"]
    print(llm_answer)
    return JSONResponse(content={"message": llm_answer,
                                 "user_message": message_text})


@app.post("/llm-process")
async def llm_process(
    file: Annotated[UploadFile, File(...)],
    llm_name: Annotated[str, Form(...)]
) -> Dict[str, Any]:
    """Обработка файла с использование LLM"""
    try:
        contents: bytes = await file.read()
        df: pd.DataFrame = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Используем llm_name как нужно
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        passage_lengths: list = list(
            df["context"]
            .apply(lambda x: len(tokenizer(x, truncation=False)["input_ids"]))
            .to_dict()
            .values()
        )
        result: Dict[str, Any] = {
            "llm_name": llm_name,
            "data_sample": passage_lengths,
            "max_len": tokenizer.model_max_length,
        }

        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
