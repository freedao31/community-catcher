import faiss
import pickle

# Vector DB
class VectorDatabase:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)  # 384 dimensions for MiniLM
        self.metadata = []

    def add(self, embeddings, meta):
        self.index.add(embeddings)
        self.metadata.extend(meta)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'index': self.index, 'metadata': self.metadata}, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        db = VectorDatabase()
        db.index = data['index']
        db.metadata = data['metadata']
        return db
