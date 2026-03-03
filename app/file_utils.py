import os
from pypdf import PdfReader
from docx import Document


def extract_text_from_txt(file_path: str) -> tuple[str, list[dict]]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # TXT has no pages — treat whole file as page 1
    return text, [{"page": 1, "text": text}]


def extract_text_from_pdf(file_path: str) -> tuple[str, list[dict]]:
    reader = PdfReader(file_path)
    full_text = ""
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        full_text += page_text + "\n"
        pages.append({"page": page_num, "text": page_text})

    return full_text, pages


def extract_text_from_docx(file_path: str) -> tuple[str, list[dict]]:
    doc = Document(file_path)
    # DOCX has no real pages — group every 10 paragraphs as a logical "page"
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)

    pages = []
    page_size = 10
    for i in range(0, len(paragraphs), page_size):
        group = paragraphs[i:i + page_size]
        pages.append({
            "page": (i // page_size) + 1,
            "text": "\n".join(group)
        })

    return full_text, pages if pages else [{"page": 1, "text": full_text}]


def extract_text(file_path: str) -> tuple[str, list[dict]]:
    """
    Returns:
        full_text: complete document text
        pages: list of {"page": int, "text": str} dicts
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        return extract_text_from_txt(file_path)
    elif extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only .txt, .pdf, .docx allowed.")


def map_chunks_to_pages(chunks: list[str], pages: list[dict]) -> list[dict]:
    """
    Maps each chunk back to the most likely source page
    by finding which page contains the most text overlap.
    Returns list of {"page": int, "chunk_index": int}
    """
    chunk_sources = []

    for chunk_idx, chunk in enumerate(chunks):
        best_page = 1
        best_overlap = 0

        # Use first 100 chars of chunk as fingerprint
        chunk_fingerprint = chunk[:100].lower()

        for page_info in pages:
            page_text_lower = page_info["text"].lower()
            # Count character overlap
            overlap = sum(1 for c in chunk_fingerprint if c in page_text_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page_info["page"]

        chunk_sources.append({"page": best_page, "chunk_index": chunk_idx})

    return chunk_sources