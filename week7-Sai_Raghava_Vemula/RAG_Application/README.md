# 📄 DocQA — Document Question Answering (RAG) System

A production-ready **Retrieval-Augmented Generation** application that answers questions exclusively from your uploaded documents. Built with Python, Streamlit, LangChain, FAISS, and Google Gemini 2.5 Flash.

> **Zero hallucination guarantee** — every answer is grounded in your documents. If the information isn't there, the system says so.

---

## 🏗️ Architecture

```
                    ┌──────────────┐
                    │   Upload     │
                    │  PDF/DOCX/TXT│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Document    │
                    │  Loader      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Text        │
                    │  Splitter    │
                    │  (1000/200)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Embeddings  │
                    │  MiniLM-L6   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  FAISS       │
                    │  Vector DB   │
                    └──────┬───────┘
                           │
          User Question ───┤
                           │
                    ┌──────▼───────┐
                    │  Retriever   │
                    │  (Top-K = 4) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Score       │
                    │  Threshold   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LLM         │
                    │  Gemini 2.5  │
                    │  Flash       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Answer +    │
                    │  Sources     │
                    └──────────────┘
```

---

## 📁 Folder Structure

```
RAG_Application/
│
├── app.py                  # Streamlit UI
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # API keys (not committed)
├── .env.example            # Template for .env
├── .gitignore
│
├── .streamlit/
│   └── config.toml         # Theme & server config
│
├── data/
│   ├── uploads/            # Uploaded documents
│   └── processed/          # Processing artifacts
│
├── vectorstore/            # FAISS index (persisted)
│
├── logs/
│   └── app.log             # Application logs
│
├── sample_docs/
│   ├── sample_resume.txt   # Test resume document
│   └── ai_notes.txt        # Test AI/ML notes
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # PDF, DOCX, TXT extraction
│   ├── text_splitter.py    # Recursive text chunking
│   ├── embeddings.py       # HuggingFace embeddings
│   ├── vector_db.py        # FAISS management
│   ├── retriever.py        # Similarity search
│   ├── llm.py              # LLM provider abstraction
│   ├── rag_pipeline.py     # End-to-end RAG orchestration
│   ├── prompts.py          # Prompt templates
│   ├── utils.py            # Shared utilities
│   └── logger.py           # Logging configuration
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG_Application
```

### 2. Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and set at least one:
```
GOOGLE_API_KEY=your-google-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

- **Google Gemini** (default): Get a key from [Google AI Studio](https://aistudio.google.com/apikey)
- **OpenAI** (fallback): Get a key from [OpenAI Platform](https://platform.openai.com/api-keys)

### 5. Run the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 💡 Usage

1. **Upload documents** — Use the sidebar to upload PDF, DOCX, or TXT files.
2. **Process** — Click "⚡ Process Documents" to index them.
3. **Ask questions** — Type your question and click "Ask ➜".
4. **Review sources** — Expand "Retrieved Context" to see exactly which chunks were used.

---

## 📝 Example Queries

Using the included sample documents:

| Query | Expected Source |
|-------|----------------|
| "What are Alex Johnson's technical skills?" | sample_resume.txt |
| "Where did Alex Johnson work before TechNova?" | sample_resume.txt |
| "What is Retrieval-Augmented Generation?" | ai_notes.txt |
| "Explain the types of machine learning" | ai_notes.txt |
| "What is the capital of France?" | ❌ Not in documents |

---

## 🛡️ Hallucination Prevention

| Layer | Mechanism |
|-------|-----------|
| Retrieval | Score threshold filter (< 0.3 = discarded) |
| Pre-LLM Gate | Empty context → fallback message, LLM not called |
| Prompt | System prompt strictly forbids outside knowledge |
| LLM Config | `temperature=0` for deterministic output |
| UI | Retrieved chunks + scores shown alongside answer |

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Google Gemini 2.5 Flash / OpenAI GPT-4.1 |
| PDF Parsing | pypdf |
| DOCX Parsing | python-docx |
| Config | python-dotenv |

---

## 🔮 Future Improvements

- [ ] Support for CSV, XLSX, and Markdown files
- [ ] Multi-language document support
- [ ] Streaming LLM responses
- [ ] User authentication and multi-tenant support
- [ ] Cloud deployment (GCP Cloud Run / AWS ECS)
- [ ] Persistent conversation memory across sessions
- [ ] Fine-tuned embedding models for domain-specific use
- [ ] OCR support for scanned PDFs
- [ ] Automated testing suite

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
