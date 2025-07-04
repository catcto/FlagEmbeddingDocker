#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from FlagEmbedding import FlagAutoModel

MODEL_NAMES = [
    "BAAI/bge-large-zh-v1.5"
]

def download_all_models():
    for name in MODEL_NAMES:
        model = FlagAutoModel.from_finetuned(name, 
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices='cpu')
        _ = model.encode(["你好BGE"])
        print("Model loaded successfully!")

if __name__ == "__main__":
    download_all_models()