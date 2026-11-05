import pytest
from revsys.clients.openalex import PyAlexFetcher


def test_convert_inverted_index_empty():
    assert PyAlexFetcher._convert_inverted_index({}) == ""
    assert PyAlexFetcher._convert_inverted_index(None) == ""


def test_convert_inverted_index_standard():
    inverted_index = {
        "A": [0],
        "systematic": [1, 5],
        "review": [2, 6],
        "of": [3],
        "evidence": [4],
    }
    expected = "A systematic review of evidence systematic review"
    result = PyAlexFetcher._convert_inverted_index(inverted_index)
    assert result == expected


def test_convert_inverted_index_single_word():
    inverted_index = {"Hello": [0]}
    assert PyAlexFetcher._convert_inverted_index(inverted_index) == "Hello"


def test_convert_inverted_index_out_of_order():
    # Test dictionary with keys and positions in reverse/arbitrary order
    inverted_index = {
        "world": [1],
        "hello": [0],
    }
    assert PyAlexFetcher._convert_inverted_index(inverted_index) == "hello world"


def test_convert_inverted_index_benchmark():
    # Benchmark to demonstrate performance efficiency of O(N) direct indexing
    words = "Machine learning models have demonstrated significant accuracy improvements in medical imaging analysis.".split()
    inverted_index = {}
    for idx, w in enumerate(words):
        inverted_index.setdefault(w, []).append(idx)

    # Perform multiple conversions
    for _ in range(100):
        reconstructed = PyAlexFetcher._convert_inverted_index(inverted_index)
        assert reconstructed == " ".join(words)
