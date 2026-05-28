"""
RAG Engine - Core Retrieval-Augmented Generation Logic
Uses FAISS for vector search + Anthropic Claude for generation
"""

import os
import re
import json
import hashlib
from typing import Optional
import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGEngine:
    """
    A complete RAG pipeline:
      1. Ingest documents → chunk → embed → store in FAISS
      2. Query → embed query → retrieve top-k chunks → generate answer via Claude
    """

    CHUNK_SIZE = 600          # characters per chunk
    CHUNK_OVERLAP = 100       # overlap between chunks

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: list[dict] = []          # [{text, source, chunk_id}]
        self.documents: dict[str, dict] = {}  # {filename: metadata}
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_document(self, content: bytes, filename: str, ext: str) -> dict:
        """Parse, chunk, embed and index a document."""
        text = self._extract_text(content, ext, filename)
        chunks = self._chunk_text(text, filename)

        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        embeddings = self._embed(chunks)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.chunks.extend(chunks)

        doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        self.documents[filename] = {
            "id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "characters": len(text),
        }

        return {"chunks": len(chunks)}

    def _extract_text(self, content: bytes, ext: str, filename: str) -> str:
        if ext == ".txt":
            return content.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            try:
                import pdfplumber
                import io
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n\n".join(pages)
            except ImportError:
                raise RuntimeError("pdfplumber is required for PDF parsing. Run: pip install pdfplumber")
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def _chunk_text(self, text: str, source: str) -> list[dict]:
        """Split text into overlapping chunks."""
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": chunk_id,
                    "start_char": start,
                })
                chunk_id += 1
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    def _embed(self, chunks: list[dict]) -> list[list[float]]:
        texts = [c["text"] for c in chunks]
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    # ── Querying ───────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 4) -> dict:
        """Embed question, retrieve chunks, generate answer via Claude."""
        q_emb = self.embedder.encode([question], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype=np.float32)

        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))

        retrieved = []
        seen_texts = set()
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if chunk["text"] in seen_texts:
                continue
            seen_texts.add(chunk["text"])
            retrieved.append({
                **chunk,
                "relevance_score": float(1 / (1 + dist)),  # normalised
            })

        context = self._build_context(retrieved)
        answer = self._generate_answer(question, context)

        sources = [
            {
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "relevance_score": round(c["relevance_score"], 4),
                "excerpt": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            }
            for c in retrieved
        ]

        return {"answer": answer, "sources": sources}

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {c['source']} | chunk {c['chunk_id']}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        system_prompt = (
            "You are a helpful document assistant. Answer the user's question using ONLY "
            "the context provided below. If the answer cannot be found in the context, "
            "say so clearly. Be concise, accurate, and cite sources where relevant.\n\n"
            f"CONTEXT:\n{context}"
        )
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return message.content[0].text

    # ── Utility ────────────────────────────────────────────────────────────

    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_document_count(self) -> int:
        return len(self.documents)

    def list_documents(self) -> list[dict]:
        return list(self.documents.values())

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.documents = {}
