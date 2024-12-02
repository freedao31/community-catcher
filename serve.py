from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Configuration
VECTOR_DB_FILE = 'vector_db.pkl'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Load Vector Database
class VectorDatabase:
    def __init__(self):
        self.index = None
        self.metadata = []

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        db = VectorDatabase()
        db.index = data['index']
        db.metadata = data['metadata']
        return db

vector_db = VectorDatabase.load(VECTOR_DB_FILE)
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Flask App
app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    max_results = data.get('max_results', 10)
    
    # Embed query
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    
    # Search in vector database
    D, I = vector_db.index.search(query_embedding, max_results)
    results = [
        {**vector_db.metadata[i], "similarity": float(1 - D[0][j])} 
        for j, i in enumerate(I[0]) if i != -1
    ]
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
