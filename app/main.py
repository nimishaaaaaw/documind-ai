from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import os
import shutil

from app.file_utils import extract_text
from app.rag_pipeline import (
    chunk_text,
    create_embeddings,
    build_faiss_index,
    generate_rag_answer
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# In-memory storage
text_chunks = []
faiss_index = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global text_chunks, faiss_index

    try:
        upload_path = f"temp_{file.filename}"

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text(upload_path)

        if not text.strip():
            os.remove(upload_path)
            raise HTTPException(status_code=400, detail="File contains no readable text.")

        text_chunks = chunk_text(text)
        embeddings = create_embeddings(text_chunks)
        faiss_index = build_faiss_index(embeddings)

        os.remove(upload_path)

        return {"message": "File processed successfully."}

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")


@app.post("/query")
async def query_document(question: str = Form(...)):
    global text_chunks, faiss_index

    if faiss_index is None:
        return {"answer": "Please upload a document first."}

    if not question.strip():
        return {"answer": "Question cannot be empty."}

    try:
        answer = generate_rag_answer(question, faiss_index, text_chunks)
        return {"answer": answer}

    except Exception:
        return {"answer": "An error occurred while processing your query."}