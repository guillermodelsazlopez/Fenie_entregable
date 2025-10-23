import argparse
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
from src.config import SETTINGS
from src.embeddings import get_embedding_model, embed_texts


def ensure_collection(client: QdrantClient, dim: int):
    try:
        client.get_collection(SETTINGS.qdrant_collection)
    except Exception:
        client.recreate_collection(
            collection_name=SETTINGS.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    dim = get_embedding_model().get_sentence_embedding_dimension()

    client = QdrantClient(url=SETTINGS.qdrant_url)
    ensure_collection(client, dim)

    vectors = embed_texts(df["texto"].fillna("").tolist())
    points = [
        PointStruct(id=row.id, vector=vectors[i], payload=row._asdict())
        for i, row in enumerate(df.itertuples(index=False))
    ]

    client.upsert(collection_name=SETTINGS.qdrant_collection, points=points)
    print(f"Ingestados {len(points)} puntos en {SETTINGS.qdrant_collection}")

if __name__ == "__main__":
    main()