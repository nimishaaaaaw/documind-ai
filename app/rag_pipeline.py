import numpy as np
import faiss
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from app.llm import generate_answer as llm_generate_answer

# Download nltk data silently on first run
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Models loaded once globally for efficiency ──
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ─────────────────────────────────────────────
# 1. SEMANTIC CHUNKING
# Splits on sentence boundaries, not characters.
# Groups sentences into ~max_tokens sized chunks
# with overlap for context continuity.
# ─────────────────────────────────────────────

def chunk_text(text: str, max_tokens: int = 400, overlap_sentences: int = 2):
    sentences = sent_tokenize(text)
    chunks = []
    chunk_metadata = []  # tracks which sentences are in each chunk

    current_chunk = []
    current_len = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        word_count = len(sentence.split())

        if current_len + word_count > max_tokens and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)
            chunk_metadata.append({"sentence_start": i - len(current_chunk), "sentence_end": i - 1})

            # Overlap: keep last N sentences for continuity
            current_chunk = current_chunk[-overlap_sentences:]
            current_len = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_len += word_count
            i += 1

    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        chunk_metadata.append({"sentence_start": i - len(current_chunk), "sentence_end": i - 1})

    return chunks, chunk_metadata


# ─────────────────────────────────────────────
# 2. EMBEDDINGS & FAISS INDEX
# ─────────────────────────────────────────────

def create_embeddings(chunks: list[str]) -> np.ndarray:
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    # IndexFlatIP = inner product (cosine sim when vectors normalized)
    index = faiss.IndexFlatIP(dimension)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


# ─────────────────────────────────────────────
# 3. BM25 INDEX (keyword search)
# ─────────────────────────────────────────────

def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


# ─────────────────────────────────────────────
# 4. HYBRID SEARCH
# Combines FAISS vector score (60%) + BM25 keyword
# score (40%) using reciprocal rank fusion.
# ─────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    faiss_index,
    bm25_index: BM25Okapi,
    chunks: list[str],
    top_k: int = 10
) -> list[tuple[str, float, int]]:

    # ── Vector search ──
    query_embedding = embedding_model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = faiss_index.search(query_vector, min(top_k, len(chunks)))
    vector_results = {int(idx): float(score) for idx, score in zip(indices[0], scores[0]) if idx != -1}

    # Normalize vector scores to [0, 1]
    if vector_results:
        max_v = max(vector_results.values())
        min_v = min(vector_results.values())
        rng = max_v - min_v if max_v != min_v else 1
        vector_results = {k: (v - min_v) / rng for k, v in vector_results.items()}

    # ── BM25 keyword search ──
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # Get top-k BM25 indices
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = {int(idx): float(bm25_scores[idx]) for idx in bm25_top_indices}

    # Normalize BM25 scores to [0, 1]
    if bm25_results:
        max_b = max(bm25_results.values())
        min_b = min(bm25_results.values())
        rng = max_b - min_b if max_b != min_b else 1
        bm25_results = {k: (v - min_b) / rng for k, v in bm25_results.items()}

    # ── Fuse scores (60% vector, 40% BM25) ──
    all_indices = set(vector_results.keys()) | set(bm25_results.keys())
    fused = {}
    for idx in all_indices:
        v_score = vector_results.get(idx, 0.0)
        b_score = bm25_results.get(idx, 0.0)
        fused[idx] = 0.6 * v_score + 0.4 * b_score

    # Sort by fused score, return top_k
    sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunks[idx], score, idx) for idx, score in sorted_results]


# ─────────────────────────────────────────────
# 5. QUERY DECOMPOSITION
# For complex questions, breaks into sub-queries
# using the LLM, retrieves for each, merges.
# ─────────────────────────────────────────────

def decompose_query(query: str) -> list[str]:
    prompt = f"""You are a query decomposition expert.
Break the following question into 2-3 simpler sub-questions that together answer the original.
If the question is already simple, return just the original question.

Return ONLY a numbered list. No explanation. No preamble.

Question: {query}

Sub-questions:"""

    response = llm_generate_answer(prompt)

    # Parse numbered list from response
    sub_queries = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1.", "1)", "- "
        for prefix in ["1.", "2.", "3.", "1)", "2)", "3)", "- ", "• "]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line:
            sub_queries.append(line)

    # Fallback: if parsing fails, use original query
    return sub_queries if sub_queries else [query]


# ─────────────────────────────────────────────
# 6. CROSS-ENCODER RE-RANKING
# After hybrid retrieval, re-ranks using a
# cross-encoder for higher precision top results.
# ─────────────────────────────────────────────

def rerank(query: str, candidates: list[tuple[str, float, int]], top_k: int = 4) -> list[tuple[str, int]]:
    if not candidates:
        return []

    pairs = [(query, chunk) for chunk, _, _ in candidates]
    scores = reranker_model.predict(pairs)

    # Sort by reranker score (highest first)
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [(chunk, chunk_idx) for _, (chunk, _, chunk_idx) in ranked]


# ─────────────────────────────────────────────
# 7. FULL RAG PIPELINE
# Orchestrates: decompose → hybrid retrieve →
# rerank → build context → LLM answer
# with source citations + confidence score
# ─────────────────────────────────────────────

def generate_rag_answer(
    query: str,
    faiss_index,
    bm25_index: BM25Okapi,
    chunks: list[str],
    chunk_sources: list[dict],  # [{"page": int, "chunk_index": int}, ...]
    conversation_history: list[dict] = None,
    top_k: int = 10
) -> dict:

    # ── Step 1: Query decomposition ──
    sub_queries = decompose_query(query)

    # ── Step 2: Hybrid retrieval for each sub-query ──
    all_candidates = {}
    for sub_q in sub_queries:
        results = hybrid_retrieve(sub_q, faiss_index, bm25_index, chunks, top_k=top_k)
        for chunk_text, score, chunk_idx in results:
            # Deduplicate: keep highest score per chunk
            if chunk_idx not in all_candidates or all_candidates[chunk_idx][1] < score:
                all_candidates[chunk_idx] = (chunk_text, score, chunk_idx)

    candidates = list(all_candidates.values())

    # ── Step 3: Re-rank ──
    top_chunks = rerank(query, candidates, top_k=4)

    # ── Step 4: Build context with source labels ──
    context_parts = []
    sources_used = []

    for chunk_text, chunk_idx in top_chunks:
        source_info = chunk_sources[chunk_idx] if chunk_idx < len(chunk_sources) else {"page": "?"}
        page = source_info.get("page", "?")
        context_parts.append(f"[Source: Page {page}]\n{chunk_text}")
        sources_used.append({"page": page, "snippet": chunk_text[:120] + "..."})

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 5: Build conversation history string ──
    history_str = ""
    if conversation_history:
        recent = conversation_history[-3:]  # last 3 exchanges
        for exchange in recent:
            history_str += f"User: {exchange['question']}\nAssistant: {exchange['answer']}\n\n"

    # ── Step 6: Build final prompt ──
    prompt = f"""You are an intelligent document analysis assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the document to answer this."

After your answer, on a new line write:
CONFIDENCE: [a number from 1-10 indicating how well the context supports your answer]

{f"Previous conversation:{chr(10)}{history_str}" if history_str else ""}

Context:
{context}

Question: {query}

Answer:"""

    # ── Step 7: Generate answer ──
    raw_response = llm_generate_answer(prompt)

    # ── Step 8: Parse confidence score ──
    confidence = None
    answer_text = raw_response

    if "CONFIDENCE:" in raw_response:
        parts = raw_response.split("CONFIDENCE:")
        answer_text = parts[0].strip()
        try:
            confidence = int(parts[1].strip().split()[0])
            confidence = max(1, min(10, confidence))  # clamp to 1-10
        except (ValueError, IndexError):
            confidence = None

    return {
        "answer": answer_text,
        "confidence": confidence,
        "sources": sources_used,
        "sub_queries": sub_queries if len(sub_queries) > 1 else None
    }
