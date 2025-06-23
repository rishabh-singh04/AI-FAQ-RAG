import faiss
import numpy as np
import pandas as pd
from embedding_model import EmbeddingModel


class VectorDB:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric
        self.questions = []
    
    def add_to_index(self, embeddings, questions):
        self.index.add(np.array(embeddings))
        self.questions.extend(questions)
    
    def search(self, query_embedding, k=5):
        D, I = self.index.search(np.array([query_embedding]), k)
        return [self.questions[i] for i in I[0]]

# Helper functions to preprocess and add data to the DB
def create_and_populate_index():
    df = pd.read_csv('data/faq.csv')
    model = EmbeddingModel()
    db = VectorDB(768)  # Assuming BERT-base model
    # Sanitize data: Coerce NaN to an empty string and ensure all data are strings
    questions = df.question.dropna().astype(str).tolist()
    embeddings = [model.encode(question) for question in questions]
    db.add_to_index(embeddings, questions)
    return db