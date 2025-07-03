from FlagEmbedding import FlagAutoModel

models_to_download = [
    "BAAI/bge-m3",
    "BAAI/bge-large-zh-v1.5"
]

def download_all_models():
    for name in models_to_download:
        model = FlagAutoModel.from_finetuned(
            model_name_or_path=name,
            normalize_embeddings=True,
            use_fp16=True,
            devices="cuda:0",
            model_class="encoder-only-m3",
        )
        _ = model.encode(["你好BGE"])
        print("Model loaded successfully!")

if __name__ == "__main__":
    download_all_models()