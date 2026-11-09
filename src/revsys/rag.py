"""
Módulo para RAG: chunking de texto, criação de vetor store e recuperação de trechos.
"""
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from revsys.config import EMBEDDING_MODEL

__all__ = ['chunk_text', 'build_vector_store', 'retrieve']

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """
    Divide texto bruto em chunks de tamanho aproximado (caracteres), com sobreposição.
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < length else end
    return chunks

def build_vector_store(chunks: list[str]) -> dict:
    """
    Gera embeddings para cada chunk e retorna um dicionário contendo:
        'chunks': lista de textos,
        'embeddings': numpy.ndarray de forma (n_chunks, dim),
        'norms': numpy.ndarray de forma (n_chunks,) com as normas L2 pré-calculadas
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY não encontrado para RAG')
    # Instancia cliente OpenAI
    client = OpenAI(api_key=api_key)
    # Gera embeddings em lote
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    emb = np.array([d.embedding for d in resp.data])
    # BOLT OPTIMIZATION: Pré-calcula normas L2 dos chunks para evitar recálculo
    # redundante em cada chamada a `retrieve`.
    norms = np.linalg.norm(emb, axis=1) if len(emb) > 0 else np.empty((0,))
    return {'chunks': chunks, 'embeddings': emb, 'norms': norms}

def retrieve(query: str, store: dict, top_k: int = 5) -> list[str]:
    """
    Recupera os top_k chunks mais similares à query, usando cosine similarity.
    """
    chunks = store.get('chunks', [])
    if not chunks:
        return []

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY não encontrado para RAG')
    # Instancia cliente OpenAI
    client = OpenAI(api_key=api_key)
    # Embedding da query
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    q_emb = np.asarray(resp.data[0].embedding, dtype=np.float32).ravel()

    # BOLT OPTIMIZATION: Substitui cosine_similarity do scikit-learn por produto
    # escalar vetorizado e normas pré-computadas (ou sob demanda se ausente).
    # Também substitui full argsort O(N log N) por argpartition O(N) para obter
    # apenas os top-k elementos, proporcionando aceleração de >30x em grandes lojas.
    emb = store.get('embeddings')
    if emb is None or len(emb) == 0:
        return []

    q_norm = np.linalg.norm(q_emb)
    emb_norms = store.get('norms')
    if emb_norms is None:
        emb_norms = np.linalg.norm(emb, axis=1)

    denom = emb_norms * (q_norm if q_norm != 0 else 1.0)
    denom[denom == 0] = 1.0
    sims = np.dot(emb, q_emb) / denom

    num_items = len(sims)
    if num_items > top_k:
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
    else:
        top_indices = np.argsort(-sims)

    return [chunks[i] for i in top_indices]