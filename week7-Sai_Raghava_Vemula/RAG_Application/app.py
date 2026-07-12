"""
app.py — Streamlit UI for the Document Question Answering RAG System.

Run with:
    streamlit run app.py
"""

import os
import pickle
import time
import streamlit as st

from src.logger import get_logger
from src.utils import ensure_directories, get_project_root
from src.document_loader import load_document
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model, get_model_name
from src.vector_db import (
    create_vectorstore,
    save as save_vectorstore,
    load as load_vectorstore,
    delete as delete_vectorstore,
    get_stats,
    DEFAULT_PERSIST_DIR,
)
from src.llm import get_llm, get_provider_name
from src.rag_pipeline import answer_question

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocQA — RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark professional theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Cards ---------- */
.card {
    background: linear-gradient(135deg, rgba(30, 33, 40, 0.95), rgba(22, 25, 31, 0.98));
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.18);
}

/* ---------- Answer card ---------- */
.answer-card {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(30, 33, 40, 0.95));
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1rem 0;
}

/* ---------- Source badge ---------- */
.source-badge {
    display: inline-block;
    background: rgba(108, 99, 255, 0.15);
    color: #A8A3FF;
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ---------- Score bar ---------- */
.score-bar-bg {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin: 0.3rem 0;
}
.score-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6C63FF, #A78BFA);
    transition: width 0.6s ease;
}

/* ---------- Metric card ---------- */
.metric-card {
    background: linear-gradient(135deg, rgba(30, 33, 40, 0.9), rgba(22, 25, 31, 0.95));
    border: 1px solid rgba(108, 99, 255, 0.12);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.8rem;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #A78BFA;
    margin: 0;
}
.metric-card .metric-label {
    font-size: 0.75rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
}

/* ---------- Chunk card ---------- */
.chunk-card {
    background: rgba(30, 33, 40, 0.8);
    border: 1px solid rgba(108, 99, 255, 0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ---------- Chat message ---------- */
.chat-user {
    background: rgba(108, 99, 255, 0.12);
    border-left: 3px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
}
.chat-assistant {
    background: rgba(30, 33, 40, 0.8);
    border-left: 3px solid #A78BFA;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
}

/* ---------- Header ---------- */
.header-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #A78BFA, #C4B5FD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.header-sub {
    font-size: 1rem;
    color: #9CA3AF;
    margin-bottom: 1.5rem;
}

/* ---------- Status indicator ---------- */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.status-active { background: #34D399; }
.status-inactive { background: #EF4444; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ---------- Inspect table row ---------- */
.inspect-row {
    background: rgba(30, 33, 40, 0.6);
    border: 1px solid rgba(108, 99, 255, 0.08);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s ease;
}
.inspect-row:hover {
    border-color: rgba(108, 99, 255, 0.35);
}
.inspect-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.inspect-file {
    font-weight: 600;
    color: #A78BFA;
}
.inspect-meta {
    color: #9CA3AF;
    font-size: 0.75rem;
}
.inspect-preview {
    color: #D1D5DB;
    font-size: 0.82rem;
    margin-top: 0.3rem;
    line-height: 1.45;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108, 99, 255, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(108, 99, 255, 0.5); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialise directories
# ---------------------------------------------------------------------------
ensure_directories()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "vectorstore": None,
    "chat_history": [],
    "processed_files": [],
    "total_chunks": 0,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Helper: load chunks from the FAISS pickle for inspection
# ---------------------------------------------------------------------------
def _load_all_chunks() -> list[dict]:
    """Read every chunk from the persisted FAISS docstore for display."""
    pkl_path = os.path.join(DEFAULT_PERSIST_DIR, "index.pkl")
    if not os.path.isfile(pkl_path):
        return []

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    docstore, id_map = None, None
    for item in data:
        if type(item).__name__ == "InMemoryDocstore":
            docstore = item
        elif isinstance(item, dict):
            id_map = item

    if docstore is None or id_map is None:
        return []

    chunks = []
    for idx, doc_id in sorted(id_map.items()):
        doc = docstore.search(doc_id)
        meta = doc.metadata
        chunks.append({
            "chunk_num": idx + 1,
            "chunk_id": meta.get("chunk_id", "N/A"),
            "filename": meta.get("filename", "unknown"),
            "page": meta.get("page_number", "?"),
            "chunk_index": meta.get("chunk_index", "?"),
            "chars": len(doc.page_content),
            "content": doc.page_content,
        })
    return chunks


# ===================================================================
#  SIDEBAR
# ===================================================================
with st.sidebar:
    # --- Branding ---
    st.markdown('<p style="font-size:1.6rem;font-weight:700;color:#A78BFA;">📄 DocQA</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9CA3AF;font-size:0.85rem;margin-top:-0.8rem;">RAG Document Assistant</p>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Page navigation ---
    st.markdown("##### 🧭 Navigation")
    active_page = st.radio(
        "Navigate",
        ["💬 Ask Questions", "🗂 Inspect Vector Database"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # --- File upload ---
    st.markdown("##### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show uploaded file names
    if uploaded_files:
        st.markdown("**Uploaded files:**")
        for uf in uploaded_files:
            icon = {"pdf": "📕", "docx": "📘", "txt": "📝"}.get(uf.name.rsplit(".", 1)[-1], "📄")
            st.markdown(f"{icon} `{uf.name}`")

    st.markdown("---")

    # --- Process button ---
    process_btn = st.button("⚡  Process Documents", use_container_width=True, type="primary")

    if process_btn and uploaded_files:
        upload_dir = os.path.join(get_project_root(), "data", "uploads")
        saved_paths = []

        # Save uploaded files to disk
        for uf in uploaded_files:
            dest = os.path.join(upload_dir, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(dest)

        with st.status("Processing documents…", expanded=True) as status:
            # 1 — Load
            st.write("📖 Loading documents…")
            all_docs = []
            for path in saved_paths:
                try:
                    docs = load_document(path)
                    all_docs.extend(docs)
                except Exception as exc:
                    st.warning(f"⚠️ Could not load `{os.path.basename(path)}`: {exc}")
                    logger.error("Load error for '%s': %s", path, exc)

            if not all_docs:
                st.error("No text could be extracted from the uploaded files.")
                st.stop()

            # 2 — Split
            st.write(f"✂️ Splitting {len(all_docs)} page(s) into chunks…")
            chunks = split_documents(all_docs)

            # 3 — Embed & index
            st.write(f"🧠 Embedding {len(chunks)} chunk(s)…")
            embeddings = get_embedding_model()
            vectorstore = create_vectorstore(chunks, embeddings)

            # 4 — Save
            st.write("💾 Saving vector store…")
            save_vectorstore(vectorstore)

            # Update state
            st.session_state["vectorstore"] = vectorstore
            st.session_state["processed_files"] = [uf.name for uf in uploaded_files]
            st.session_state["total_chunks"] = len(chunks)

            status.update(label="✅ Processing complete!", state="complete")
            logger.info("Documents processed: %d chunks from %d file(s).", len(chunks), len(uploaded_files))

    elif process_btn and not uploaded_files:
        st.warning("Please upload at least one document first.")

    st.markdown("---")

    # --- Clear vector DB ---
    if st.button("🗑️  Clear Vector Database", use_container_width=True):
        delete_vectorstore()
        st.session_state["vectorstore"] = None
        st.session_state["processed_files"] = []
        st.session_state["total_chunks"] = 0
        st.success("Vector database cleared.")
        logger.info("Vector database cleared by user.")

    st.markdown("---")

    # --- Sidebar metrics ---
    st.markdown("##### 📊 System Metrics")

    stats = get_stats()
    status_class = "status-active" if stats["exists"] else "status-inactive"
    status_label = "Ready" if stats["exists"] else "No Index"

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{st.session_state["total_chunks"]}</p>
        <p class="metric-label">Total Chunks</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="font-size:0.9rem;">{get_model_name().split("/")[-1]}</p>
        <p class="metric-label">Embedding Model</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="font-size:1rem;">
            <span class="status-dot {status_class}"></span>{status_label}
        </p>
        <p class="metric-label">Vector DB Status</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="font-size:0.85rem;">{get_provider_name()}</p>
        <p class="metric-label">LLM Provider</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Try to load existing vectorstore if not in session
# ---------------------------------------------------------------------------
if st.session_state["vectorstore"] is None:
    try:
        embeddings = get_embedding_model()
        vs = load_vectorstore(embeddings)
        if vs is not None:
            st.session_state["vectorstore"] = vs
            st.session_state["total_chunks"] = vs.index.ntotal
    except Exception:
        pass


# ===================================================================
#  PAGE: 💬 Ask Questions
# ===================================================================
if active_page == "💬 Ask Questions":

    # --- Header ---
    st.markdown('<p class="header-title">Document Question Answering</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="header-sub">'
        'Upload your documents, ask questions, and get answers grounded entirely in your data. '
        'No hallucinations — only facts from your files.'
        '</p>',
        unsafe_allow_html=True,
    )

    # --- Question input ---
    st.markdown("##### 💬 Ask a Question")
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "Type your question here…",
            placeholder="e.g. What are the candidate's key skills?",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_btn = st.button("Ask ➜", type="primary", use_container_width=True)

    # --- Handle question ---
    if ask_btn and question:
        if st.session_state["vectorstore"] is None:
            st.warning("⚠️ Please upload and process documents first.")
        else:
            with st.spinner("🔍 Searching documents and generating answer…"):
                try:
                    llm = get_llm()
                    result = answer_question(
                        question=question,
                        vectorstore=st.session_state["vectorstore"],
                        llm=llm,
                    )
                except EnvironmentError as env_err:
                    st.error(f"🔑 {env_err}")
                    st.stop()
                except Exception as exc:
                    st.error(f"An error occurred: {exc}")
                    logger.error("Pipeline error: %s", exc)
                    st.stop()

            # ---- Save to chat history ----
            st.session_state["chat_history"].append({
                "question": question,
                "result": result,
            })

            # ---- Display answer ----
            st.markdown("---")
            st.markdown("##### ✨ Answer")
            st.markdown(f"""
            <div class="answer-card">
                {result["answer"]}
            </div>
            """, unsafe_allow_html=True)

            # ---- Metrics row ----
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{result["num_chunks_retrieved"]}</p>
                    <p class="metric-label">Chunks Retrieved</p>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                avg_score = (
                    sum(result["scores"]) / len(result["scores"])
                    if result["scores"]
                    else 0
                )
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{avg_score:.2%}</p>
                    <p class="metric-label">Avg Relevance</p>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{result["processing_time"]}s</p>
                    <p class="metric-label">Processing Time</p>
                </div>
                """, unsafe_allow_html=True)

            # ---- Sources ----
            if result["sources"]:
                st.markdown("##### 📎 Sources")
                src_html = " ".join(
                    f'<span class="source-badge">📄 {s}</span>' for s in result["sources"]
                )
                st.markdown(src_html, unsafe_allow_html=True)

            # ---- Retrieved context ----
            if result["chunks"]:
                with st.expander("📚 Retrieved Context", expanded=False):
                    for chunk in result["chunks"]:
                        pct = int(chunk["score"] * 100)
                        st.markdown(f"""
                        <div class="chunk-card">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                                <span style="font-weight:600;color:#A78BFA;">
                                    📄 {chunk["filename"]} — Page {chunk["page_number"]}
                                </span>
                                <span style="color:#9CA3AF;font-size:0.75rem;">
                                    Chunk: {chunk["chunk_id"][:8]}…
                                </span>
                            </div>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:{pct}%;"></div>
                            </div>
                            <p style="color:#9CA3AF;font-size:0.7rem;margin:0.2rem 0 0.6rem;">
                                Similarity: {chunk["score"]:.4f}
                            </p>
                            <p style="margin:0;">{chunk["content"]}</p>
                        </div>
                        """, unsafe_allow_html=True)

    elif ask_btn and not question:
        st.warning("Please enter a question.")

    # --- Conversation history ---
    st.markdown("---")
    st.markdown("##### 🕑 Conversation History")

    if st.session_state["chat_history"]:
        col_clear, _ = st.columns([1, 5])
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()

        for entry in reversed(st.session_state["chat_history"]):
            st.markdown(f"""
            <div class="chat-user">
                <strong>🧑 You:</strong> {entry["question"]}
            </div>
            <div class="chat-assistant">
                <strong>🤖 Assistant:</strong> {entry["result"]["answer"]}
                <br><span style="color:#9CA3AF;font-size:0.75rem;">
                    Sources: {", ".join(entry["result"]["sources"]) if entry["result"]["sources"] else "None"}
                    &nbsp;|&nbsp; ⏱ {entry["result"]["processing_time"]}s
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color:#6B7280;text-align:center;padding:2rem;">'
            "No conversations yet. Upload documents and ask a question to get started!"
            "</p>",
            unsafe_allow_html=True,
        )


# ===================================================================
#  PAGE: 🗂 Inspect Vector Database
# ===================================================================
elif active_page == "🗂 Inspect Vector Database":

    st.markdown('<p class="header-title">Inspect Vector Database</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="header-sub">'
        'Browse every chunk stored in the FAISS index. Click a row to expand and view the full text.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Load chunks from the persisted store
    all_chunks = _load_all_chunks()

    if not all_chunks:
        st.info("🗄️ The vector database is empty. Upload and process documents first.")
    else:
        # --- Summary metrics ---
        unique_files = list({c["filename"] for c in all_chunks})
        total_chars = sum(c["chars"] for c in all_chunks)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{len(all_chunks)}</p>
                <p class="metric-label">Total Chunks</p>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{len(unique_files)}</p>
                <p class="metric-label">Documents</p>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            avg_chars = total_chars // len(all_chunks) if all_chunks else 0
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{avg_chars}</p>
                <p class="metric-label">Avg Chunk Size</p>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{total_chars:,}</p>
                <p class="metric-label">Total Characters</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- Filter controls ---
        filter_col1, filter_col2 = st.columns([2, 4])
        with filter_col1:
            file_filter = st.selectbox(
                "Filter by file",
                ["All files"] + sorted(unique_files),
            )
        with filter_col2:
            search_text = st.text_input(
                "Search chunk content",
                placeholder="Type to search within chunks…",
            )

        # Apply filters
        display_chunks = all_chunks
        if file_filter != "All files":
            display_chunks = [c for c in display_chunks if c["filename"] == file_filter]
        if search_text:
            search_lower = search_text.lower()
            display_chunks = [c for c in display_chunks if search_lower in c["content"].lower()]

        st.markdown(f"**Showing {len(display_chunks)} of {len(all_chunks)} chunks**")
        st.markdown("")

        # --- Chunk table header ---
        header_cols = st.columns([0.5, 2.5, 0.7, 0.8, 4])
        with header_cols[0]:
            st.markdown("**#**")
        with header_cols[1]:
            st.markdown("**File**")
        with header_cols[2]:
            st.markdown("**Page**")
        with header_cols[3]:
            st.markdown("**Chars**")
        with header_cols[4]:
            st.markdown("**Preview**")

        st.markdown('<hr style="margin:0.3rem 0;border-color:rgba(108,99,255,0.2);">', unsafe_allow_html=True)

        # --- Chunk rows (expandable) ---
        for chunk in display_chunks:
            preview = chunk["content"][:100].replace("\n", " ")
            if len(chunk["content"]) > 100:
                preview += "…"

            with st.expander(
                f"📄  **Chunk {chunk['chunk_num']}**  —  `{chunk['filename']}`  |  "
                f"Page {chunk['page']}  |  {chunk['chars']} chars  |  "
                f"_{preview}_"
            ):
                # Chunk metadata
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value" style="font-size:0.85rem;">{chunk['filename']}</p>
                        <p class="metric-label">Source File</p>
                    </div>
                    """, unsafe_allow_html=True)
                with meta_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{chunk['page']}</p>
                        <p class="metric-label">Page Number</p>
                    </div>
                    """, unsafe_allow_html=True)
                with meta_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{chunk['chars']}</p>
                        <p class="metric-label">Characters</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Chunk ID
                st.markdown(
                    f'<p style="color:#9CA3AF;font-size:0.72rem;">Chunk ID: <code>{chunk["chunk_id"]}</code></p>',
                    unsafe_allow_html=True,
                )

                # Full content
                st.markdown("**Full Content:**")
                st.markdown(
                    f'<div class="chunk-card" style="white-space:pre-wrap;">{chunk["content"]}</div>',
                    unsafe_allow_html=True,
                )


# ===================================================================
#  FOOTER
# ===================================================================
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#4B5563;font-size:0.75rem;">'
    "DocQA RAG System • Built with Streamlit, LangChain & FAISS • "
    "Answers grounded in your documents only"
    "</p>",
    unsafe_allow_html=True,
)
