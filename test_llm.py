from app.llm import generate_answer

prompt = "Explain in one sentence what a vector database is."

answer = generate_answer(prompt)

print(answer)