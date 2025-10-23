from dataclasses import dataclass
import os


@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "emailclassification")
    embedding_model_name: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    classifier_model_name: str = os.getenv(
    "CLASSIFIER_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    seed: int = int(os.getenv("SEED", 42))


SETTINGS = Settings()