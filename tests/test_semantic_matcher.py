import pytest
from modules.semantic_matcher import chunk_similarity, similarity_score_to_percent

def test_chunk_similarity():
    # Orthogonal/simple vectors make this test independent of model downloads.
    score = chunk_similarity([[1.0, 0.0], [0.0, 1.0]], [[0.9, 0.1]])
    assert 0.8 < score < 1.0

def test_empty_embeddings_raise():
    with pytest.raises(ValueError):
        chunk_similarity([], [[1.0, 0.0]])

def test_percent_is_clamped():
    assert similarity_score_to_percent(1.2) == 100.0
    assert similarity_score_to_percent(-0.2) == 0.0
