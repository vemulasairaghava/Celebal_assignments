"""
text_splitter.py — Split documents into chunks for embedding.

Uses LangChain's ``RecursiveCharacterTextSplitter`` with configurable
chunk size and overlap. Enriches each chunk's metadata with unique
identifiers for traceability.
"""

import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default chunking parameters
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of documents into smaller chunks.

    Each chunk receives enriched metadata:
    * ``chunk_id``    — a unique UUID for the chunk.
    * ``document_id`` — a UUID derived from the source filename.
    * ``chunk_index`` — zero-based index within this splitting run.
    * (inherits ``filename``, ``page_number``, ``source`` from the parent doc.)

    Parameters
    ----------
    documents : List[Document]
        Raw documents (one per page / section).
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    List[Document]
        Chunked documents with enriched metadata.
    """
    if not documents:
        logger.warning("No documents provided to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # --- Enrich metadata ---
    doc_id_cache: dict[str, str] = {}
    for idx, chunk in enumerate(chunks):
        filename = chunk.metadata.get("filename", "unknown")

        # Stable document_id per filename
        if filename not in doc_id_cache:
            doc_id_cache[filename] = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))

        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["document_id"] = doc_id_cache[filename]
        chunk.metadata["chunk_index"] = idx

    logger.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
