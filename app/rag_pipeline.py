import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.llm import generate_answer as llm_generate_answer

class Solution:
  def makeLargestSpecial(self, s: str) -> str:
    specials = []
    count = 0

    i = 0
    for j, c in enumerate(s):
      count += 1 if c == '1' else -1
      if count == 0:
        specials.append(
            '1' + self.makeLargestSpecial(s[i + 1:j]) + '0')
        i = j + 1

    return ''.join(sorted(specials)[::-1])

# Load embedding model once (global, efficient)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# 1️⃣ Text Chunking
# -------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# -------------------------
# 2️⃣ Create Embeddings
# -------------------------

def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")


# -------------------------
# 3️⃣ Build FAISS Index
# -------------------------

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# -------------------------
# 4️⃣ Retrieve Relevant Chunks
# -------------------------

def retrieve(query: str, index, chunks, top_k: int = 3):
    query_embedding = embedding_model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = [chunks[i] for i in indices[0]]
    return results


# -------------------------
# 5️⃣ Full RAG Pipeline
# -------------------------

def generate_rag_answer(query: str, index, chunks, top_k: int = 3):
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve(query, index, chunks, top_k)

    # Step 2: Create context
    context = "\n\n".join(retrieved_chunks)

    # Step 3: Build prompt
    prompt = f"""
You are an intelligent assistant.
Answer the question ONLY using the context below.
If the answer is not present in the context, say "I don't know."

Context:
{context}

Question:
{query}
"""

    # Step 4: Send to LLM
    answer = llm_generate_answer(prompt)

    return answer

