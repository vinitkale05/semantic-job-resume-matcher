"""Embedding-based semantic matching."""
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from modules.text_preprocessor import split_into_chunks

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once; first use requires a model download."""
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(text: str):
    chunks = split_into_chunks(text)
    if not chunks:
        raise ValueError("Cannot create embeddings from empty text.")
    model = get_model()
    return model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)


def chunk_similarity(resume_embeddings, job_embeddings) -> float:
    """Compute mean of each job chunk's best resume-chunk similarity."""
    resume_arr = np.atleast_2d(np.asarray(resume_embeddings))
    job_arr = np.atleast_2d(np.asarray(job_embeddings))
    if resume_arr.size == 0 or job_arr.size == 0:
        raise ValueError("Embeddings cannot be empty.")
    matrix = cosine_similarity(job_arr, resume_arr)
    # Average best supporting resume chunk for each job-description chunk.
    return float(np.mean(np.max(matrix, axis=1)))


def similarity_score_to_percent(score: float) -> float:
    """Map cosine similarity to 0-100 for display; not a hiring probability."""
    # Sentence embedding cosine scores can be negative. Clip for a readable scale.
    return max(0.0, min(100.0, float(score) * 100.0))


def explain_chunk_matches(resume_text: str, job_text: str, top_n: int = 5) -> list[dict]:
    resume_chunks = split_into_chunks(resume_text)
    job_chunks = split_into_chunks(job_text)
    model = get_model()
    r_emb = model.encode(resume_chunks, normalize_embeddings=True, convert_to_numpy=True)
    j_emb = model.encode(job_chunks, normalize_embeddings=True, convert_to_numpy=True)
    matrix = cosine_similarity(j_emb, r_emb)
    results = []
    for idx, job_chunk in enumerate(job_chunks):
        best_idx = int(np.argmax(matrix[idx]))
        results.append({
            "job_requirement_excerpt": job_chunk,
            "resume_evidence_excerpt": resume_chunks[best_idx],
            "similarity": float(matrix[idx, best_idx]),
        })
    return sorted(results, key=lambda item: item["similarity"], reverse=True)[:top_n]
