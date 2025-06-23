import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle  


faq_df = pd.read_csv("data/FAQ_Bank.csv")


faq_texts = faq_df["question"].astype(str) + " " + faq_df["answer"].astype(str)

model = SentenceTransformer("all-MiniLM-L6-v2")

faq_embeddings = model.encode(faq_texts.tolist(), show_progress_bar=True)

dimension = faq_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(faq_embeddings))

faiss.write_index(faiss_index, "data/faiss_index.bin")

# Save FAQ Data Mapping
with open("data/faq_data.pkl", "wb") as f:
    pickle.dump(faq_df, f)

print("✅ Embeddings generated and stored in FAISS!")