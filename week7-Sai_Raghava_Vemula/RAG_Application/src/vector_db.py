"""
vector_db.py — FAISS vector store management.

Provides functions to create, save, load, delete, and update a FAISS
index backed by LangChain's ``FAISS`` wrapper.
"""

import os
import shutil
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.logger import get_logger
from src.utils import get_project_root

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default persistence path
# ---------------------------------------------------------------------------
DEFAULT_PERSIST_DIR = os.path.join(get_project_root(), "vectorstore")


def create_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """
    Build a new FAISS vector store from a list of chunked documents.

    Parameters
    ----------
    documents : List[Document]
        Chunked documents with metadata.
    embeddings : Embeddings
        Embedding model instance.

    Returns
    -------
    FAISS
        The populated vector store.
    """
    if not documents:
        raise ValueError("Cannot create a vector store from an empty document list.")

    logger.info("Creating FAISS vector store from %d chunk(s)…", len(documents))
    vectorstore = FAISS.from_documents(documents, embeddings)
    logger.info("FAISS vector store created successfully.")
    return vectorstore


def save(vectorstore: FAISS, persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
    """Persist the FAISS index and docstore to disk."""
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    logger.info("Vector store saved to '%s'.", persist_dir)


def load(
    embeddings: Embeddings,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> Optional[FAISS]:
    """
    Load a previously saved FAISS vector store.

    Returns ``None`` if the directory does not exist or is empty.
    """
    index_file = os.path.join(persist_dir, "index.faiss")
    if not os.path.isfile(index_file):
        logger.info("No existing vector store found at '%s'.", persist_dir)
        return None

    logger.info("Loading vector store from '%s'…", persist_dir)
    vectorstore = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("Vector store loaded (%d vectors).", vectorstore.index.ntotal)
    return vectorstore


def delete(persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
    """Delete the persisted vector store directory."""
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
        logger.info("Vector store deleted from '%s'.", persist_dir)
    else:
        logger.info("No vector store to delete at '%s'.", persist_dir)


def update(
    vectorstore: FAISS,
    new_documents: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """
    Add new documents to an existing FAISS vector store.

    Parameters
    ----------
    vectorstore : FAISS
        Existing vector store.
    new_documents : List[Document]
        Additional chunked documents.
    embeddings : Embeddings
        Embedding model (must match the one used to create the store).

    Returns
    -------
    FAISS
        Updated vector store.
    """
    if not new_documents:
        logger.warning("No new documents provided for update.")
        return vectorstore

    new_vs = FAISS.from_documents(new_documents, embeddings)
    vectorstore.merge_from(new_vs)
    logger.info("Merged %d new chunk(s) into the vector store.", len(new_documents))
    return vectorstore


def get_stats(persist_dir: str = DEFAULT_PERSIST_DIR) -> dict:
    """
    Return basic statistics about the persisted vector store.

    Returns
    -------
    dict
        ``{"exists": bool, "num_vectors": int}``
    """
    index_file = os.path.join(persist_dir, "index.faiss")
    if not os.path.isfile(index_file):
        return {"exists": False, "num_vectors": 0}

    # Read the number of vectors without loading the full store
    try:
        import faiss as _faiss
        index = _faiss.read_index(index_file)
        count = index.ntotal
    except Exception:
        count = -1  # Unknown

    return {"exists": True, "num_vectors": count}
