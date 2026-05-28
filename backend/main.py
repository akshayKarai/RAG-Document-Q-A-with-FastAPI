"""
RAG Document Q&A - FastAPI Backend
Retrieval-Augmented Generation with FAISS + Claude API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from rag_engine import RAGEngine

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload documents and ask questions using Retrieval-Augmented Generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine instance
rag_engine = RAGEngine()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    question: str


@app.get("/")
def root():
    return {"message": "RAG Document Q&A API is running 🚀", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": rag_engine.get_document_count(),
        "chunks_indexed": rag_engine.get_chunk_count(),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF or TXT)."""
    allowed_types = ["application/pdf", "text/plain"]
    allowed_extensions = [".pdf", ".txt"]

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}"
        )

    content = await file.read()
    try:
        result = rag_engine.ingest_document(content, file.filename, ext)
        return {
            "message": f"Document '{file.filename}' indexed successfully",
            "chunks_created": result["chunks"],
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Ask a question against the indexed documents."""
    if rag_engine.get_chunk_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload documents first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_engine.query(request.question, top_k=request.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=request.question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.delete("/documents")
def clear_documents():
    """Clear all indexed documents."""
    rag_engine.clear()
    return {"message": "All documents cleared from index."}


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    return {"documents": rag_engine.list_documents()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
