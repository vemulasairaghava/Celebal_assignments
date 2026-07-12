"""
retriever.py — Similarity search against the FAISS vector store.

Returns the top-K most relevant chunks with their similarity scores
and filters out results below a configurable threshold.
"""

import math
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 4
# L2 distance threshold — with normalised embeddings, L2 ranges from 0 to 2.
# Cosine similarity = 1 - (L2^2 / 2).
# A max L2 distance of 1.5 ≈ cosine similarity of -0.125 (very lenient).
DEFAULT_MAX_L2_DISTANCE = 1.5


def _l2_to_cosine(l2_distance: float) -> float:
    """Convert L2 distance (normalised embeddings) to cosine similarity [−1, 1]."""
    return 1.0 - (l2_distance ** 2) / 2.0


def retrieve(
    query: str,
    vectorstore: FAISS,
    k: int = DEFAULT_TOP_K,
    max_l2_distance: float = DEFAULT_MAX_L2_DISTANCE,
) -> List[Tuple[Document, float]]:
    """
    Perform similarity search and return ranked, filtered results.

    Uses ``similarity_search_with_score`` which returns raw FAISS L2
    distances (lower = more similar).  Distances are converted to
    cosine similarity for display purposes.

    Parameters
    ----------
    query : str
        The user's question.
    vectorstore : FAISS
        The FAISS vector store to search.
    k : int
        Number of top results to retrieve.
    max_l2_distance : float
        Maximum L2 distance to keep (results farther away are discarded).

    Returns
    -------
    List[Tuple[Document, float]]
        ``(document, cosine_similarity)`` tuples sorted by descending
        similarity.
    """
    if not query or not query.strip():
        logger.warning("Empty query received — skipping retrieval.")
        return []

    logger.info("Retrieving top-%d chunks for query: '%.80s…'", k, query)

    try:
        # similarity_search_with_score returns (Document, L2_distance)
        # Lower distance = more similar
        raw_results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise RuntimeError(f"Vector search failed: {exc}") from exc

    # Log raw scores for debugging
    for doc, dist in raw_results:
        logger.info(
            "  RAW — L2=%.4f  cosine=%.4f  file=%s",
            dist,
            _l2_to_cosine(dist),
            doc.metadata.get("filename", "?"),
        )

    # Filter by max L2 distance and convert to cosine similarity
    filtered: List[Tuple[Document, float]] = []
    for doc, l2_dist in raw_results:
        if l2_dist <= max_l2_distance:
            cosine_sim = _l2_to_cosine(l2_dist)
            # Clamp to [0, 1] for display
            cosine_sim = max(0.0, min(1.0, cosine_sim))
            filtered.append((doc, cosine_sim))

    # Sort by similarity descending (highest first)
    filtered.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        "Retrieved %d result(s), %d within distance threshold (%.2f).",
        len(raw_results),
        len(filtered),
        max_l2_distance,
    )

    return filtered
