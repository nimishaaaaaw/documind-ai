from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

import os
import shutil
import uuid

from app.file_utils import extract_text, map_chunks_to_pages
from app.rag_pipeline import (
    chunk_text,
    create_embeddings,
    build_faiss_index,
    build_bm25_index,
    generate_rag_answer
)

app = FastAPI(title="DocuMind AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ── In-memory session store ──
# Keyed by session_id so multiple users don't share state
sessions = {}


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "text_chunks": [],
            "chunk_sources": [],
            "faiss_index": None,
            "bm25_index": None,
            "conversation_history": [],
            "document_name": None,
        }
    return sessions[session_id]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/session")
async def create_session():
    """Create a new session and return session_id"""
    session_id = str(uuid.uuid4())
    get_session(session_id)
    return {"session_id": session_id}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    session = get_session(session_id)
    upload_path = f"temp_{session_id}_{file.filename}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text with page metadata
        full_text, pages = extract_text(upload_path)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="File contains no readable text.")

        # Semantic chunking
        chunks, chunk_metadata = chunk_text(full_text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract chunks from document.")

        # Map chunks to source pages
        chunk_sources = map_chunks_to_pages(chunks, pages)

        # Build both indexes
        embeddings = create_embeddings(chunks)
        faiss_index = build_faiss_index(embeddings)
        bm25_index = build_bm25_index(chunks)

        # Store in session
        session["text_chunks"] = chunks
        session["chunk_sources"] = chunk_sources
        session["faiss_index"] = faiss_index
        session["bm25_index"] = bm25_index
        session["conversation_history"] = []  # reset on new doc
        session["document_name"] = file.filename

        return {
            "message": f"'{file.filename}' processed successfully.",
            "chunks_created": len(chunks),
            "pages_detected": len(pages)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


@app.post("/query")
async def query_document(
    question: str = Form(...),
    session_id: str = Form(...)
):
    session = get_session(session_id)

    if session["faiss_index"] is None:
        return {"answer": "Please upload a document first.", "confidence": None, "sources": []}

    if not question.strip():
        return {"answer": "Question cannot be empty.", "confidence": None, "sources": []}

    try:
        result = generate_rag_answer(
            query=question,
            faiss_index=session["faiss_index"],
            bm25_index=session["bm25_index"],
            chunks=session["text_chunks"],
            chunk_sources=session["chunk_sources"],
            conversation_history=session["conversation_history"]
        )

        # Save to conversation history (keep last 5)
        session["conversation_history"].append({
            "question": question,
            "answer": result["answer"]
        })
        if len(session["conversation_history"]) > 5:
            session["conversation_history"].pop(0)

        return result

    except Exception as e:
        return {
            "answer": "An error occurred while processing your query.",
            "confidence": None,
            "sources": [],
            "error": str(e)
        }


@app.post("/reset")
async def reset_session(session_id: str = Form(...)):
    """Clear document and conversation history for a session"""
    if session_id in sessions:
        sessions[session_id] = {
            "text_chunks": [],
            "chunk_sources": [],
            "faiss_index": None,
            "bm25_index": None,
            "conversation_history": [],
            "document_name": None,
        }
    return {"message": "Session reset successfully."}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}