# DocMind: RAG Document Q&A with FastAPI

> **Ask questions about any document using Retrieval-Augmented Generation (RAG)**
> Built with FastAPI · FAISS · Sentence Transformers · Claude AI · Streamlit

---

## 📸 Overview

DocMind is a full-stack **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF or TXT documents and ask natural-language questions about their content. It retrieves the most relevant passages using semantic vector search and generates accurate, context-grounded answers using Anthropic's Claude AI.

```
User uploads PDF/TXT
        │
        ▼
  Text Extraction
  (pdfplumber / utf-8)
        │
        ▼
  Chunking (600 chars, 100 overlap)
        │
        ▼
  Embedding (all-MiniLM-L6-v2)
        │
        ▼
  FAISS Vector Index
        │
        ▼
  User asks a question
        │
        ▼
  Query Embedding → FAISS similarity search → Top-K chunks retrieved
        │
        ▼
  Context + Question → Claude claude-sonnet-4-20250514 → Answer
        │
        ▼
  Answer + Source citations shown in Streamlit UI
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Document Upload** | PDF and TXT files via drag-and-drop UI |
| **Semantic Search** | FAISS flat L2 index with `all-MiniLM-L6-v2` embeddings |
| **RAG Pipeline** | Chunk → Embed → Retrieve → Generate |
| **Claude Integration** | Uses `claude-sonnet-4-20250514` for generation |
| **Source Citations** | Every answer shows which chunks were used + relevance score |
| **Multi-doc support** | Index multiple documents and query across all of them |
| **REST API** | Full FastAPI backend with Swagger UI at `/docs` |
| **Chat UI** | Streamlit chat interface with message history |

---

## 🏗️ Project Structure

```
rag-doc-qa/
├── backend/
│   ├── main.py            # FastAPI application & routes
│   ├── rag_engine.py      # Core RAG pipeline (chunking, embedding, retrieval, generation)
│   └── requirements.txt   # Backend Python dependencies
│
├── frontend/
│   ├── app.py             # Streamlit UI
│   └── requirements.txt   # Frontend Python dependencies
│
├── data/
│   └── sample_docs/       # Sample documents to get started quickly
│       ├── intro_to_ai.txt
│       └── python_overview.txt
│
├── scripts/
│   └── start.sh           # One-command startup script (starts both services)
│
├── .env.example           # Environment variable template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-doc-qa.git
cd rag-doc-qa
```

### 2. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

> Get your key at [https://console.anthropic.com](https://console.anthropic.com)

### 3. Run with the startup script (recommended)

```bash
bash scripts/start.sh
```

This installs all dependencies and launches both services automatically.

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Swagger API docs | http://localhost:8000/docs |

---

## 🔧 Manual Setup (Alternative)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

---

## 📖 How to Use

1. **Open** http://localhost:8501 in your browser
2. **Upload** a PDF or TXT file using the sidebar uploader
3. **Click** "Index Documents" — the file is chunked, embedded, and stored in FAISS
4. **Type** a question in the chat input at the bottom
5. **Get** a grounded answer with source citations and relevance scores

### Try with the sample documents

The `data/sample_docs/` folder includes two documents you can upload right away:
- `intro_to_ai.txt` — covers AI history, LLMs, RAG, ethics
- `python_overview.txt` — covers Python basics, FastAPI, virtual environments

**Example questions to try:**
- *"What is Retrieval-Augmented Generation?"*
- *"Who invented Python and when?"*
- *"What are the ethical concerns around AI?"*
- *"How do I install FastAPI?"*

---

## 🔌 API Reference

The FastAPI backend exposes a full REST API documented at `/docs`.

### Endpoints

#### `GET /health`
Returns backend status and index statistics.
```json
{
  "status": "healthy",
  "documents_indexed": 2,
  "chunks_indexed": 47
}
```

#### `POST /upload`
Upload and index a document.
- **Body**: `multipart/form-data` with field `file` (PDF or TXT)
- **Response**:
```json
{
  "message": "Document 'intro_to_ai.txt' indexed successfully",
  "chunks_created": 23,
  "filename": "intro_to_ai.txt"
}
```

#### `POST /query`
Ask a question against the indexed documents.
```json
// Request
{
  "question": "What is RAG?",
  "top_k": 4
}

// Response
{
  "question": "What is RAG?",
  "answer": "RAG (Retrieval-Augmented Generation) is an AI framework...",
  "sources": [
    {
      "source": "intro_to_ai.txt",
      "chunk_id": 12,
      "relevance_score": 0.8731,
      "excerpt": "RAG is an AI framework that combines information retrieval..."
    }
  ]
}
```

#### `GET /documents`
List all indexed documents.

#### `DELETE /documents`
Clear all indexed documents and reset the FAISS index.

---

## 🧱 Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Anthropic Claude (`claude-sonnet-4-20250514`) | Answer generation |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Text → vectors |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Fast ANN retrieval |
| **PDF Parsing** | pdfplumber | Extract text from PDFs |
| **API** | FastAPI + Uvicorn | High-performance REST backend |
| **UI** | Streamlit | Rapid, reactive chat interface |
| **Validation** | Pydantic v2 | Request/response schemas |

---

## ⚙️ Configuration

All key parameters are set in `backend/rag_engine.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `embedding_model` | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| `top_k` (query) | `4` | Number of chunks to retrieve |
| `max_tokens` (LLM) | `1024` | Claude response token limit |

You can swap `IndexFlatL2` for `IndexIVFFlat` (with training) for large-scale deployments.

---

## 🔮 Potential Extensions

- **Persistent storage**: Replace in-memory FAISS with Chroma, Pinecone, or Qdrant
- **Multi-modal**: Add image understanding with Claude's vision capabilities
- **Authentication**: Add API key auth to the FastAPI backend
- **Docker**: Containerise with `docker-compose` for easy deployment
- **Streaming**: Stream Claude's response tokens for faster perceived latency
- **Evaluation**: Add RAGAS metrics (faithfulness, answer relevancy, context recall)
- **Hybrid search**: Combine BM25 keyword search with semantic retrieval

---

## 🤝 Skills Demonstrated

This project showcases:

- ✅ **RAG architecture** — full pipeline from ingestion to generation
- ✅ **Vector databases** — FAISS indexing and similarity search
- ✅ **Embeddings** — sentence-transformer models for semantic similarity
- ✅ **LLM integration** — Anthropic Python SDK with prompt engineering
- ✅ **REST API design** — FastAPI with Pydantic validation, OpenAPI docs
- ✅ **Full-stack development** — decoupled backend + Streamlit frontend
- ✅ **Production patterns** — error handling, chunking strategy, source attribution

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Anthropic Claude, FastAPI, FAISS, and Streamlit*

