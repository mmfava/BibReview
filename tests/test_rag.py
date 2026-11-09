import time
import numpy as np
import pytest

from revsys.rag import chunk_text, build_vector_store, retrieve


class DummyEmbed:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingsAPI:
    def __init__(self, mock_map=None):
        self.mock_map = mock_map or {}

    def create(self, model, input):
        data = []
        for text in input:
            if text in self.mock_map:
                data.append(DummyEmbed(self.mock_map[text]))
            else:
                # Deterministic pseudo-embedding based on string hash
                rng = np.random.RandomState(abs(hash(text)) % (2**31))
                vec = rng.randn(64).tolist()
                data.append(DummyEmbed(vec))

        class Response:
            pass

        resp = Response()
        resp.data = data
        return resp


class FakeOpenAI:
    def __init__(self, api_key=None, mock_map=None):
        self.api_key = api_key
        self.embeddings = FakeEmbeddingsAPI(mock_map)


def test_chunk_text_basic():
    text = "A" * 4500
    chunks = chunk_text(text, chunk_size=2000, overlap=200)
    assert len(chunks) == 3
    assert len(chunks[0]) == 2000
    assert len(chunks[1]) == 2000
    assert len(chunks[2]) == 900


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text(None) == []


def test_build_vector_store_and_retrieve(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_vectors = {
        "query": [1.0, 0.0, 0.0],
        "chunk1": [0.9, 0.1, 0.0],  # high similarity to query
        "chunk2": [0.0, 1.0, 0.0],  # orthogonal to query
        "chunk3": [0.5, 0.5, 0.0],  # moderate similarity to query
    }

    fake_client = FakeOpenAI(mock_map=mock_vectors)
    monkeypatch.setattr("revsys.rag.OpenAI", lambda api_key: fake_client)

    chunks = ["chunk1", "chunk2", "chunk3"]
    store = build_vector_store(chunks)

    assert store["chunks"] == chunks
    assert "embeddings" in store
    assert "norms" in store
    assert len(store["norms"]) == 3

    # Retrieve top 2 chunks
    retrieved = retrieve("query", store, top_k=2)
    assert retrieved == ["chunk1", "chunk3"]


def test_retrieve_fewer_than_top_k(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_vectors = {
        "query": [1.0, 0.0],
        "chunkA": [0.2, 0.8],
        "chunkB": [0.9, 0.1],
    }

    fake_client = FakeOpenAI(mock_map=mock_vectors)
    monkeypatch.setattr("revsys.rag.OpenAI", lambda api_key: fake_client)

    chunks = ["chunkA", "chunkB"]
    store = {
        "chunks": chunks,
        "embeddings": np.array([mock_vectors["chunkA"], mock_vectors["chunkB"]]),
        # Deliberately test fallback when 'norms' is absent from store
    }

    retrieved = retrieve("query", store, top_k=5)
    assert retrieved == ["chunkB", "chunkA"]


def test_retrieve_empty_store():
    assert retrieve("query", {"chunks": [], "embeddings": np.empty((0, 3))}) == []
    assert retrieve("query", {}) == []


def test_retrieve_performance_benchmark(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    n_chunks = 1000
    dim = 256
    rng = np.random.RandomState(42)
    embeddings = rng.randn(n_chunks, dim).astype(np.float32)
    query_vec = rng.randn(dim).astype(np.float32).tolist()

    fake_client = FakeOpenAI(mock_map={"search query": query_vec})
    monkeypatch.setattr("revsys.rag.OpenAI", lambda api_key: fake_client)

    chunks = [f"Text chunk {i}" for i in range(n_chunks)]
    store = {
        "chunks": chunks,
        "embeddings": embeddings,
        "norms": np.linalg.norm(embeddings, axis=1),
    }

    # Verify retrieval completes rapidly across 100 queries
    start_time = time.perf_counter()
    for _ in range(100):
        res = retrieve("search query", store, top_k=5)
        assert len(res) == 5
    elapsed = time.perf_counter() - start_time

    # 100 queries over 1000 chunks should complete well within 0.5s with optimized numpy
    assert elapsed < 0.5, f"Retrieval was slower than expected: {elapsed:.3f}s"
