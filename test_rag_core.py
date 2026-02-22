from app.file_utils import extract_text
from app.rag_pipeline import chunk_text, create_embeddings, build_faiss_index, retrieve

# Step 1: Extract
text = extract_text("sample2.pdf")

# Step 2: Chunk
chunks = chunk_text(text)

# Step 3: Embed
embeddings = create_embeddings(chunks)

# Step 4: Index
index = build_faiss_index(embeddings)

# Step 5: Query
query = "What is this document about?"
retrieved = retrieve(query, index, chunks)

print("\nRetrieved Context:\n")
for r in retrieved:
    print("-" * 50)
    print(r)