import argparse
import hashlib
import os
import pandas as pd
import torch
from transformers import pipeline
from dotenv import load_dotenv
from src.config import SETTINGS
from src.embeddings import get_embedding_model, embed_texts

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except Exception:
    QdrantClient = None

CANDIDATE_LABELS_ES = [
    "Queja", "Petición de servicio", "Sugerencia de mejora"
]
HYPOTHESIS_TEMPLATE_ES = "El propósito de este texto es expresar una {}."


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def classify_series(texts):
    # Determinismo
    torch.manual_seed(SETTINGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SETTINGS.seed)
    # Pipeline ZSL (encoder NLI)
    clf = pipeline(
        task="zero-shot-classification",
        model=SETTINGS.classifier_model_name,
        device=0 if torch.cuda.is_available() else -1,
    )
    # Sin dropout
    clf.model.eval()

    preds = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            preds.append({"label": None, "score": 0.0, "scores": {}, "labels": CANDIDATE_LABELS_ES})
            continue
        out = clf(
            sequences=t,
            candidate_labels=CANDIDATE_LABELS_ES,
            hypothesis_template=HYPOTHESIS_TEMPLATE_ES,
            multi_label=False,
        )
        # Normaliza a probas
        labels = out["labels"]
        scores = out["scores"]
        top_label = labels[0]
        top_score = float(scores[0])
        preds.append({
            "label": top_label,
            "score": top_score,
            "scores": dict(zip(labels, map(float, scores))),
            "labels": labels,
        })
    return preds


def upsert_qdrant(df: pd.DataFrame):
    if QdrantClient is None:
        print("qdrant-client no instalado; omito upsert.")
        return
    client = QdrantClient(url=SETTINGS.qdrant_url)
    # Asegura colecciÃ³n
    vector_dim = get_embedding_model().get_sentence_embedding_dimension()
    try:
        client.get_collection(SETTINGS.qdrant_collection)
    except Exception:
        client.recreate_collection(
            collection_name=SETTINGS.qdrant_collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
    # Embeddings
    vectors = embed_texts(df["texto"].fillna("").tolist())
    # Upsert
    points = []
    for i, row in df.iterrows():
        pid = row["id"]
        payload = {
            "fecha": row.get("fecha"),
            "remitente": row.get("remitente"),
            "texto": row.get("texto"),
            "etiqueta_predicha": row.get("etiqueta_predicha"),
            "confianza": row.get("confianza"),
        }
        points.append(PointStruct(id=pid, vector=vectors[i], payload=payload))
    client.upsert(collection_name=SETTINGS.qdrant_collection, points=points)
    print(
        f"Upserted {len(points)} puntos en {SETTINGS.qdrANT_collection if hasattr(SETTINGS, 'qdrANT_collection') else SETTINGS.qdrant_collection}.")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV con columnas: fecha, remitente, texto")
    ap.add_argument("--output", required=True, help="CSV de salida con predicciones")
    ap.add_argument("--to-qdrant", action="store_true", help="Inserta/actualiza en Qdrant")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=";")
    df = df.rename(columns={
        "Fecha": "fecha",
        "Email": "remitente",
        "Descripción": "texto"
    })[["fecha", "remitente", "texto"]]

    preds = classify_series(df["texto"].tolist())
    df["etiqueta_predicha"] = [p["label"] for p in preds]
    df["confianza"] = [p["score"] for p in preds]
    # id determinista
    df["id"] = [
        int(hashlib.sha256(f"{r.remitente}|{r.fecha}|{(r.texto or '')[:512]}".encode("utf-8")).hexdigest(), 16) % (
                10 ** 12)
        for r in df.itertuples()
    ]

    df.to_csv(args.output, index=False)
    print(f"Escrito {args.output} ({len(df)} filas)")
    # Se deja la subida a la vectorial como opcional

    if args.to_qdrant:
        upsert_qdrant(df)


if __name__ == "__main__":
    main()
