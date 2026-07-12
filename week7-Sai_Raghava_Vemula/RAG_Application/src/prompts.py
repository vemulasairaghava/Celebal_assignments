"""
prompts.py — Prompt templates for the RAG pipeline.

Contains the system prompt that enforces grounded, hallucination-free
answers and a helper to build the full prompt with retrieved context.
"""

# ---------------------------------------------------------------------------
# System prompt — strict grounding rules
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a **Document Question Answering Assistant**.

### Rules you MUST follow:
1. Answer the user's question using **ONLY** the provided context below.
2. **Never** use outside knowledge, personal opinions, or information not present in the context.
3. **Never** guess or fabricate information (hallucinate).
4. If the context does **not** contain enough information to answer the question, respond with:
   "I could not find this information in the uploaded documents."
5. Always **cite the source filename(s)** from which you derived the answer.
6. Be concise and factual.
7. Use bullet points when listing multiple items.
8. If the question is ambiguous, state what you can answer based on the context and note the ambiguity.
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """### Context (retrieved from uploaded documents):
{context}

### Question:
{question}

### Instructions:
- Answer the question using ONLY the context above.
- Cite the source filename(s) for every claim.
- If the answer is not in the context, say: "I could not find this information in the uploaded documents."
"""


def build_prompt(question: str, context_chunks: list) -> str:
    """
    Build the user-side prompt by injecting retrieved context.

    Parameters
    ----------
    question : str
        The user's question.
    context_chunks : list
        List of ``(Document, score)`` tuples from the retriever.

    Returns
    -------
    str
        Fully formatted prompt string.
    """
    if not context_chunks:
        context_text = "(No relevant context was found.)"
    else:
        sections = []
        for idx, (doc, score) in enumerate(context_chunks, start=1):
            filename = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page_number", "?")
            sections.append(
                f"**[Source {idx}]** File: *{filename}* | Page: {page} | "
                f"Relevance: {score:.2f}\n{doc.page_content}"
            )
        context_text = "\n\n---\n\n".join(sections)

    return USER_PROMPT_TEMPLATE.format(context=context_text, question=question)
