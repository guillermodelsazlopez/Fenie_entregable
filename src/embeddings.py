from sentence_transformers import SentenceTransformer
from src.config import SETTINGS

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(SETTINGS.embedding_model_name)
    return _model


def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(list(texts), normalize_embeddings=True)