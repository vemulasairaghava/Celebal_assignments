"""
embeddings.py — Embedding model initialisation and caching.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` via LangChain's
HuggingFace integration. The model is loaded once and reused.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singleton (populated on first call)
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Return a singleton ``HuggingFaceEmbeddings`` instance.

    The model is downloaded and cached locally on first invocation.
    Subsequent calls return the same object.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.

    Returns
    -------
    HuggingFaceEmbeddings
    """
    global _embedding_model  # noqa: PLW0603

    if _embedding_model is not None:
        return _embedding_model

    logger.info("Loading embedding model: %s", model_name)
    try:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        logger.info("Embedding model loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        raise RuntimeError(f"Could not load embedding model '{model_name}': {exc}") from exc

    return _embedding_model


def get_model_name() -> str:
    """Return the display name of the current embedding model."""
    return DEFAULT_MODEL_NAME
