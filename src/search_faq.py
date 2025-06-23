import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# --- Load Environment Variables ---
load_dotenv()
DIAL_API_KEY = os.getenv("DIAL_API_KEY")
DIAL_API_VERSION = "2024-02-01"
DIAL_ENDPOINT = "https://ai-proxy.lab.epam.com"
DIAL_DEPLOYMENT = "gpt-35-turbo"

if not DIAL_API_KEY:
    raise ValueError("⚠️ DIAL_API_KEY not found in .env file!")

# --- Initialize EPAM Dial API Client ---
client = AzureOpenAI(
    api_key=DIAL_API_KEY,
    api_version=DIAL_API_VERSION,
    azure_endpoint=DIAL_ENDPOINT
)

# --- Load FAISS Index & FAQ Data ---
def load_faiss_index(index_path, faq_data_path):
    """Load FAISS index and FAQ data."""
    with open(faq_data_path, "rb") as f:
        faq_data = pickle.load(f)

    faiss_index = faiss.read_index(index_path)
    return faiss_index, faq_data

# --- FAISS Search Function ---
def search_faq(query, model, faiss_index, faq_data, top_k=1):
    """Retrieve the most relevant FAQ using FAISS."""
    query_embedding = model.encode([query])

    # Search FAISS for closest match
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    best_faq_index = indices[0][0]

    if distances < 0:
        return None, None  # No relevant FAQ found

    best_faq = faq_data.iloc[best_faq_index]  # Retrieve from DataFrame
    return best_faq["question"], best_faq["answer"]  # Extract Q&A

# --- DIAL API Call for Response Enhancement ---
def enhance_response_with_dial(faq_question, faq_answer):
    """Enhance the FAQ answer using EPAM Dial API."""
    try:
        response = client.chat.completions.create(
            model=DIAL_DEPLOYMENT,
            temperature=0.5,  # Reduced randomness to prevent repetition
            messages=[
                {"role": "system", "content": "You are an AI assistant providing COVID-19 information."},
                {"role": "system", "content": "If the response is irrelivant to the given data, just respond I don't know the answer."},
                {"role": "user", "content": f"Question: {faq_question} \nAnswer: {faq_answer}"},
                {"role": "user", "content": "Can you improve and elaborate on this response?"}
            ]
        )

        enhanced_response = response.choices[0].message.content.strip()

        # 🔍 Debugging: Log API response
        print(f"\n🔍 Raw DIAL API Response:\n{enhanced_response}\n")

        #  Remove repetitive sentences
        sentences = enhanced_response.split(". ")
        unique_sentences = list(dict.fromkeys(sentences))  # Preserve order, remove duplicates
        enhanced_response = ". ".join(unique_sentences)

        return enhanced_response

    except Exception as e:
        print(f"⚠️ Error calling DIAL API: {e}")
        return faq_answer  # Fallback to original answer

# --- Main Function: Retrieve FAQ & Enhance Response ---
def get_faq_response(query):
    """Retrieve FAQ and enhance response using DIAL API."""
    # Load FAISS & FAQ Data
    faiss_index, faq_data = load_faiss_index("data/faiss_index.bin", "data/faq_data.pkl")

    # Load Embedding Model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Search FAISS
    faq_question, faq_answer = search_faq(query, model, faiss_index, faq_data)

    if faq_question is None:
        return "⚠️ Sorry, I couldn't find relevant information."

    # Enhance the answer using the DIAL API
    enhanced_response = enhance_response_with_dial(faq_question, faq_answer)

    return enhanced_response

# --- Example Usage ---
if __name__ == "__main__":
    user_query = input("Ask a COVID-19 question: ")
    response = get_faq_response(user_query)
        
    print("\n💡 AI-Enhanced Response:")
    print(response)
