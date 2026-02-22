from app.file_utils import extract_text

text = extract_text("sample3.docx")
print(text[:500])