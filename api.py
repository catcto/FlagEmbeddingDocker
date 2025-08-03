#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from FlagEmbedding import FlagAutoModel
import hdbscan
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="FlagEmbedding API")

MODEL_NAMES = [
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

class WeightedText(BaseModel):
    text: str
    weight: int

class ClusterRequest(BaseModel):
    model: str
    texts: List[str]
    weights: Optional[List[int]] = None
    min_cluster_size: Optional[int] = 5
    min_samples: Optional[int] = None
    metric: Optional[str] = "euclidean"
    cluster_selection_epsilon: Optional[float] = 0.0
    alpha: Optional[float] = 1.0

class ClusterGroup(BaseModel):
    cluster_id: int
    texts: List[str]
    indices: List[int]
    weights: List[int]
    size: int
    total_weight: int
    max_weight: int
    avg_weight: float

class ClusterResponse(BaseModel):
    clusters: List[ClusterGroup]
    noise_texts: List[str]
    noise_indices: List[int]
    noise_weights: List[int]
    total_clusters: int
    clustered_count: int
    noise_count: int

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

@app.post("/cluster", response_model=ClusterResponse)
def cluster_texts(req: ClusterRequest):
    if req.model not in models:
        raise HTTPException(status_code=400, detail=f'Model "{req.model}" not loaded.')
    if not req.texts:
        raise HTTPException(status_code=400, detail='"texts" cannot be empty.')
    if len(req.texts) < req.min_cluster_size:
        raise HTTPException(status_code=400, detail=f'Number of texts ({len(req.texts)}) must be >= min_cluster_size ({req.min_cluster_size}).')
    
    if req.weights is None:
        weights = [1] * len(req.texts)
    else:
        if len(req.weights) != len(req.texts):
            raise HTTPException(status_code=400, detail="Length of weights must match length of texts.")
        weights = req.weights
    
    try:
        embedding_model = models[req.model]
        embeddings = embedding_model.encode(req.texts)
        embeddings = np.array(embeddings)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            metric=req.metric,
            cluster_selection_epsilon=req.cluster_selection_epsilon,
            alpha=req.alpha
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        clusters = {}
        noise_texts = []
        noise_indices = []
        noise_weights = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:
                noise_texts.append(req.texts[i])
                noise_indices.append(i)
                noise_weights.append(weights[i])
            else:
                if label not in clusters:
                    clusters[label] = {
                        'texts': [],
                        'indices': [],
                        'weights': []
                    }
                clusters[label]['texts'].append(req.texts[i])
                clusters[label]['indices'].append(i)
                clusters[label]['weights'].append(weights[i])
        
        cluster_groups = []
        for cluster_id, cluster_data in clusters.items():
            cluster_weights = cluster_data['weights']
            total_weight = sum(cluster_weights)
            max_weight = max(cluster_weights)
            avg_weight = total_weight / len(cluster_weights)
            
            cluster_groups.append(ClusterGroup(
                cluster_id=int(cluster_id),
                texts=cluster_data['texts'],
                indices=cluster_data['indices'],
                weights=cluster_weights,
                size=len(cluster_data['texts']),
                total_weight=total_weight,
                max_weight=max_weight,
                avg_weight=avg_weight
            ))
        
        cluster_groups.sort(key=lambda x: (x.total_weight, x.avg_weight), reverse=True)
        
        return ClusterResponse(
            clusters=cluster_groups,
            noise_texts=noise_texts,
            noise_indices=noise_indices,
            noise_weights=noise_weights,
            total_clusters=len(clusters),
            clustered_count=len(req.texts) - len(noise_texts),
            noise_count=len(noise_texts)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")

@app.get("/models")
def list_models():
    return {"loaded_models": list(models.keys())}

@app.get("/health")
def health_check():
    return {"status": "healthy", "loaded_models_count": len(models)}

if __name__ == "__main__":
    api_port = os.getenv("API_PORT", 8080)
    api_host = os.getenv("API_HOST", "127.0.0.1")
    import uvicorn
    uvicorn.run(app, host=api_host, port=int(api_port))