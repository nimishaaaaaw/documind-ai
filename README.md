[![Live](https://img.shields.io/badge/Live-Render-green?style=for-the-badge)](https://documind-ai-vozp.onrender.com/)

# 🚀 DocuMind AI

A production-ready Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask context-aware questions using semantic search + Groq LLM.

🔗 **Live Demo:** https://documind-ai-vozp.onrender.com/

---

## 🧠 What It Does

- Upload `.txt`, `.pdf`, `.docx` documents
- Extracts and chunks text
- Generates embeddings using Sentence Transformers
- Builds FAISS vector index
- Retrieves top-k relevant chunks
- Sends contextual prompt to Groq LLM
- Returns structured, context-aware answers

---

## 🏗️ Architecture

User Upload  
→ Text Extraction  
→ Chunking  
→ Embedding Generation  
→ FAISS Index  
→ Similarity Search  
→ Groq LLM  
→ Final Answer  

---

## 🛠 Tech Stack

- **Backend:** FastAPI
- **LLM:** Groq (LLaMA 3.1)
- **Embeddings:** Sentence Transformers (MiniLM)
- **Vector Store:** FAISS
- **Frontend:** Custom HTML, CSS (Glass UI)
- **Containerization:** Docker
- **Deployment:** Render

---

## ✨ Features

- Retrieval-Augmented Generation pipeline
- Context-aware semantic search
- Custom pastel glassmorphism UI
- Dark mode toggle
- Drag & drop upload
- Animated answer typing
- Dockerized deployment
- Production-ready error handling

---

## 🐳 Run Locally

```bash
docker build -t rag-groq-app .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key rag-groq-app
```

Open:
```
http://localhost:8000
```

---

## 🌍 Deployment

This project is deployed using **Render (Docker runtime)**.

Note: Free tier instances may experience cold starts (~30-50s).

---

## 📌 Future Improvements

- Persistent vector database
- Multi-document support
- Streaming LLM responses
- Authentication
- Production scaling

---

## 👩‍💻 Author

**Nimisha Majgawali**  
