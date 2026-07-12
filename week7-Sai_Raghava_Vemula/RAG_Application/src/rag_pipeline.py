"""
rag_pipeline.py — End-to-end RAG orchestration.

Ties together retrieval, prompt building, and LLM invocation.
Returns a structured result dict that the Streamlit UI renders.
"""

import time
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.logger import get_logger
from src.prompts import SYSTEM_PROMPT, build_prompt
from src.retriever import retrieve

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fallback message (no LLM call)
# ---------------------------------------------------------------------------
FALLBACK_ANSWER = "I could not find this information in the uploaded documents."


def answer_question(
    question: str,
    vectorstore: Any,
    llm: Any,
    k: int = 4,
) -> Dict[str, Any]:
    """
    Execute the full RAG pipeline for a single question.

    Workflow
    --------
    1. Retrieve top-K chunks from FAISS.
    2. Filter by L2 distance threshold (handled by retriever).
    3. If no relevant chunks → return fallback (skip LLM).
    4. Build grounded prompt with context.
    5. Call LLM with temperature=0.
    6. Return structured result.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    vectorstore : FAISS
        Populated vector store.
    llm : BaseChatModel
        Initialised LangChain chat model.
    k : int
        Number of chunks to retrieve.

    Returns
    -------
    dict
        Keys: ``answer``, ``sources``, ``chunks``, ``scores``,
        ``processing_time``, ``num_chunks_retrieved``.
    """
    start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Retrieve (filtering by L2 distance is handled internally)
    # ------------------------------------------------------------------
    logger.info("RAG pipeline started for question: '%.80s…'", question)
    results: List[Tuple[Document, float]] = retrieve(
        query=question,
        vectorstore=vectorstore,
        k=k,
    )

    # ------------------------------------------------------------------
    # Step 2 — Guard: no relevant context → skip LLM
    # ------------------------------------------------------------------
    if not results:
        elapsed = time.perf_counter() - start
        logger.info("No relevant chunks found. Returning fallback answer.")
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "chunks": [],
            "scores": [],
            "processing_time": round(elapsed, 2),
            "num_chunks_retrieved": 0,
        }

    # ------------------------------------------------------------------
    # Step 3 — Build prompt
    # ------------------------------------------------------------------
    user_prompt = build_prompt(question, results)

    # ------------------------------------------------------------------
    # Step 4 — Call LLM
    # ------------------------------------------------------------------
    logger.info("Sending prompt to LLM (%d context chunks)…", len(results))
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        answer = response.content.strip()
    except Exception as exc:
        logger.error("LLM invocation failed: %s", exc)
        answer = f"An error occurred while generating the answer: {exc}"

    # ------------------------------------------------------------------
    # Step 5 — Package results
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start

    sources = list({doc.metadata.get("filename", "unknown") for doc, _ in results})
    chunks_data = [
        {
            "content": doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "filename": doc.metadata.get("filename", "unknown"),
            "page_number": doc.metadata.get("page_number", "?"),
            "score": round(score, 4),
        }
        for doc, score in results
    ]
    scores = [round(s, 4) for _, s in results]

    logger.info(
        "RAG pipeline completed in %.2f s — %d source(s), %d chunk(s).",
        elapsed,
        len(sources),
        len(results),
    )

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks_data,
        "scores": scores,
        "processing_time": round(elapsed, 2),
        "num_chunks_retrieved": len(results),
    }
