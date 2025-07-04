#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from typing import List
from FlagEmbedding import FlagAutoModel

app = FastAPI(title="FlagEmbedding API")

MODEL_NAMES = [
    "BAAI/bge-m3",
    "BAAI/bge-large-zh-v1.5"
]

models = {}

@app.on_event("startup")
def load_models():
    for name in MODEL_NAMES:
        models[name] = FlagAutoModel.from_finetuned(name, 
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices=['cuda:0'])

class EmbedRequest(BaseModel):
    model: str
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if req.model not in models:
        raise HTTPException(status_code=400, detail=f'Model "{req.model}" not loaded.')
    if not req.texts:
        raise HTTPException(status_code=400, detail='"texts" cannot be empty.')

    try:
        embedding_model = models[req.model]
        embeddings = embedding_model.encode(req.texts)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    return {"loaded_models": list(models.keys())}

if __name__ == "__main__":
    api_port = os.getenv("API_PORT", 8080)
    api_host = os.getenv("API_HOST", "127.0.0.1")
    import uvicorn
    uvicorn.run(app, host=api_host, port=int(api_port))