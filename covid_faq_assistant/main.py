# main.py
from fastapi import FastAPI
from vector_db import create_and_populate_index
from api_utils import get_answer  # Check if api_utils is reachable like this

app = FastAPI()
db = create_and_populate_index()

@app.post("/ask/")
async def ask_question(question: str):
    embedding = db.model.encode(question)
    matched_questions = db.search(embedding)
    answer = get_answer(matched_questions, question)
    return {"question": question, "answer": answer}