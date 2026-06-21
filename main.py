import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from core.rag_engine import RAGEngine
from services.claude_service import ClaudeService

app = FastAPI(title="AI Customer Support Agent")

rag = RAGEngine()
claude = ClaudeService()

SAMPLE_DOCS_PATH = Path(__file__).parent / "data" / "sample_docs.txt"

@app.on_event("startup")
def load_default_docs():
    if SAMPLE_DOCS_PATH.exists():
        rag.load_documents(SAMPLE_DOCS_PATH.read_text())
        print(f"Loaded {len(rag.chunks)} chunks")

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    question: str
    free_chat: Optional[bool] = False

class LoadDocsRequest(BaseModel):
    content: str
    source_name: str = "Custom Document"

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())

@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.free_chat:
        # General conversation — no RAG, just Claude
        answer = claude.free_chat(req.question)
        return {"answer": answer, "retrieved_chunks": []}

    if not rag.is_loaded():
        raise HTTPException(status_code=400, detail="No documents loaded.")

    retrieved = rag.retrieve(req.question, top_k=3)
    answer = claude.answer(req.question, retrieved)
    return {
        "answer": answer,
        "retrieved_chunks": [
            {"text": chunk[:200] + "...", "score": round(score, 3)}
            for chunk, score in retrieved
        ]
    }

@app.post("/api/load-docs")
def load_docs(req: LoadDocsRequest):
    if len(req.content.strip()) < 10:
        raise HTTPException(status_code=400, detail="Content too short.")
    rag.load_documents(req.content)
    return {"status": "loaded", "chunks_created": len(rag.chunks)}

@app.get("/api/status")
def status():
    return {
        "status": "online",
        "knowledge_base_loaded": rag.is_loaded(),
        "chunk_count": len(rag.chunks)
    }