# src/rag_ollama.py
"""
RAG module using Ollama for local LLM inference.
Retrieves semantically similar emails from Qdrant and generates an answer.
"""

from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from src.config import SETTINGS
from src.embeddings import embed_texts
import requests
import json

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"  # You can change to mistral, phi3, gemma, etc.


def retrieve(qdrant: QdrantClient, query: str, top_k: int = 5) -> List[dict]:
    """Retrieve top_k similar documents from Qdrant."""
    qvec = embed_texts([query])[0]
    res = qdrant.search(
        collection_name=SETTINGS.qdrant_collection,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=None,
    )
    hits = []
    for p in res:
        pl = p.payload or {}
        pl["_score"] = float(p.score)
        hits.append(pl)
    return hits


def ollama_chat(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Generate text completion using local Ollama server."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt},
            stream=False,
            timeout=120
        )
        response.raise_for_status()
        text = ""
        for line in response.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if "response" in obj:
                text += obj["response"]
        return text.strip()
    except Exception as e:
        return f"(⚠️ Error al conectar con Ollama: {e})"


def generate_answer(question: str, contexts: List[dict]) -> str:
    """Compose the context and generate a final answer using Ollama."""
    if not contexts:
        return "No se encontraron correos relevantes en la base de datos."

    context_str = "\n\n".join(
        [
            f"[Doc {i+1}] Remitente: {c.get('remitente')} | Fecha: {c.get('fecha')}\n{c.get('texto','')[:800]}"
            for i, c in enumerate(contexts)
        ]
    )

    prompt = f"""
Eres un asistente que ayuda a revisar correos de clientes de una empresa de telecomunicaciones.
Responde de forma breve, precisa y profesional usando solo la información del CONTEXTO.
Cita los [Doc X] de los que saques la información.

PREGUNTA: {question}

CONTEXTO:
{context_str}
    """

    return ollama_chat(prompt)


def ask(qdrant: QdrantClient, question: str, top_k: int = 5) -> Tuple[str, List[dict]]:
    """Main function: retrieve and generate final answer."""
    docs = retrieve(qdrant, question, top_k=top_k)
    answer = generate_answer(question, docs)
    return answer, docs
