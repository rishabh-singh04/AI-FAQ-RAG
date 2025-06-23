from fastapi import FastAPI, HTTPException
import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(title="COVID-19 FAQ Assistant", version="1.0")

# --- Load API Credentials Securely ---
DIAL_API_KEY = os.getenv("DIAL_API_KEY")
DIAL_API_VERSION = "2024-02-01"
DIAL_ENDPOINT = "https://ai-proxy.lab.epam.com"
DIAL_DEPLOYMENT = "gpt-35-turbo"

# Validate API Key
if not DIAL_API_KEY:
    raise ValueError("DIAL_API_KEY is missing. Please set it in the .env file.")

# Initialize the DIAL API client
client = AzureOpenAI(
    api_key=DIAL_API_KEY,
    api_version=DIAL_API_VERSION,
    azure_endpoint=DIAL_ENDPOINT
)

# --- Load FAISS Index and FAQ Data ---
INDEX_PATH = "data/faiss_index.bin"
FAQ_DATA_PATH = "data/faq_data.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_faiss_index():
    """Load FAISS index and FAQ dataset."""
    with open(FAQ_DATA_PATH, "rb") as f:
        faq_data = pickle.load(f)
    faiss_index = faiss.read_index(INDEX_PATH)
    return faiss_index, faq_data

faiss_index, faq_data = load_faiss_index()  # Load at startup
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# --- Request Models ---
class QueryRequest(BaseModel):
    query: str
    history: list  # List of previous questions and answers

# --- FAISS Search Function ---
def search_faq(query_text, top_k=1):
    """Retrieve the most relevant FAQ using FAISS."""
    query_embedding = embedding_model.encode([query_text])
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)

    best_faq_index = indices[0][0]

    # Debugging prints
    print("Query:", query_text)
    print("Best FAQ Index (from FAISS):", best_faq_index)
    print("FAISS Indices Returned:", indices)
    print("FAQ DataFrame Length:", len(faq_data))

    if best_faq_index == -1:
        return None, None

    # Debugging check for valid index
    if best_faq_index < 0 or best_faq_index >= len(faq_data):
        print(f"❌ Invalid FAQ Index: {best_faq_index}. Total FAQ entries: {len(faq_data)}")
        return None, None

    best_faq = faq_data.iloc[best_faq_index]  # Retrieve from DataFrame
    return best_faq['question'], best_faq['answer']

# --- DIAL API Call for Response Enhancement ---
def enhance_response_with_dial(faq_question, faq_answer, history):
    """Enhance the retrieved FAQ answer using DIAL API."""
    try:
        # Combine history into a single string
        history_str = "\n".join([f"Q: {q['question']}\nA: {q['answer']}" for q in history])

        response = client.chat.completions.create(
            model=DIAL_DEPLOYMENT,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an AI assistant providing concise and accurate COVID-19 information."},
                {"role": "user", "content": f"Previous Conversation:\n{history_str}"},
                {"role": "user", "content": f"New Question: {faq_question}\nAnswer: {faq_answer}"},
                {"role": "user", "content": "Can you improve and summarize this response to directly address the question?"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error calling DIAL API: {e}")
        return faq_answer  # Fallback to original answer

# --- API Endpoint ---
@app.post("/get_faq_response")
def get_faq_response(request: QueryRequest):
    """API endpoint to retrieve FAQ and enhance response using DIAL API."""
    faq_question, faq_answer = search_faq(request.query)

    if faq_question is None:
        raise HTTPException(status_code=404, detail="No relevant FAQ found.")

    enhanced_response = enhance_response_with_dial(faq_question, faq_answer, request.history)

    return {
        "query": request.query,
        "enhanced_response": enhanced_response
    }

