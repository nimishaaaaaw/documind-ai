from app.file_utils import extract_text
from app.rag_pipeline import chunk_text, create_embeddings, build_faiss_index, generate_rag_answer

# Step 1: Extract text
text = extract_text("sample2.pdf")

# Step 2: Chunk
chunks = chunk_text(text)

# Step 3: Embed
embeddings = create_embeddings(chunks)

# Step 4: Index
index = build_faiss_index(embeddings)

# Step 5: Ask question
query = "What is this document about?"

answer = generate_rag_answer(query, index, chunks)

print("\nFinal Answer:\n")
print(answer)