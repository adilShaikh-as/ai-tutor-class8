from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(texts):
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings