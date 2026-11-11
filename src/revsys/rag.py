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
        'embeddings': numpy.ndarray de forma (n_chunks, dim)
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY não encontrado para RAG')
    # Instancia cliente OpenAI
    client = OpenAI(api_key=api_key)
    # Gera embeddings em lote
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    emb = np.array([d.embedding for d in resp.data])
    return {'chunks': chunks, 'embeddings': emb}

def retrieve(query: str, store: dict, top_k: int = 5) -> list[str]:
    """
    Recupera os top_k chunks mais similares à query, usando cosine similarity.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY não encontrado para RAG')
    # Instancia cliente OpenAI
    client = OpenAI(api_key=api_key)
    # Embedding da query
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    q_vec = np.array(resp.data[0].embedding)

    chunks = store.get('chunks', [])
    emb = store.get('embeddings')

    if emb is None or len(emb) == 0 or len(chunks) == 0:
        return []

    # Optimization: Compute cosine similarity directly via numpy dot product and norms (~3x faster than cosine_similarity)
    # Avoids sklearn overhead, array copies, and redundant matrix conversions
    norm_q = np.linalg.norm(q_vec)
    norm_emb = np.linalg.norm(emb, axis=1)
    denom = norm_emb * norm_q
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = np.where(denom > 0, np.dot(emb, q_vec) / denom, 0.0)

    # Optimization: Use np.argpartition to find top-k in O(N) instead of sorting all N elements with np.argsort in O(N log N)
    if len(sims) <= top_k:
        idxs = np.argsort(sims)[::-1]
    else:
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        idxs = top_indices[np.argsort(sims[top_indices])[::-1]]

    return [chunks[i] for i in idxs]