import numpy as np
import pytest
from revsys.rag import retrieve

class DummyEmbedding:
    def __init__(self, emb):
        self.embedding = emb

class DummyResp:
    def __init__(self, emb):
        self.data = [DummyEmbedding(emb)]

class DummyEmbeddingsClient:
    def __init__(self, query_emb):
        self.query_emb = query_emb

    def create(self, model, input):
        return DummyResp(self.query_emb)

class DummyOpenAI:
    def __init__(self, query_emb):
        self.embeddings = DummyEmbeddingsClient(query_emb)

def test_retrieve_ranking(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    # Query embedding pointing in direction [1.0, 0.0]
    query_emb = [1.0, 0.0]
    monkeypatch.setattr("revsys.rag.OpenAI", lambda api_key: DummyOpenAI(query_emb))

    chunks = ["chunk_0", "chunk_1", "chunk_2"]
    # chunk_1 is most aligned [1.0, 0.0], chunk_2 is partially aligned [0.7, 0.7], chunk_0 is orthogonal [0.0, 1.0]
    embeddings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.7, 0.7]
    ])
    store = {"chunks": chunks, "embeddings": embeddings}

    results = retrieve("test query", store, top_k=2)
    assert results == ["chunk_1", "chunk_2"]

def test_retrieve_top_k_larger_than_store(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    query_emb = [1.0, 0.0]
    monkeypatch.setattr("revsys.rag.OpenAI", lambda api_key: DummyOpenAI(query_emb))

    chunks = ["chunk_0", "chunk_1"]
    embeddings = np.array([
        [0.5, 0.5],
        [1.0, 0.0]
    ])
    store = {"chunks": chunks, "embeddings": embeddings}

    results = retrieve("test query", store, top_k=5)
    assert len(results) == 2
    assert results == ["chunk_1", "chunk_0"]

def test_retrieve_empty_store(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    query_emb = [1.0, 0.0]
    monkeypatch.setattr("revsys.rag.OpenAI", lambda api_key: DummyOpenAI(query_emb))

    assert retrieve("test query", {"chunks": [], "embeddings": np.array([])}, top_k=5) == []
    assert retrieve("test query", {}, top_k=5) == []
