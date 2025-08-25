# app/qdrant_utils.py
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "gurumitra_docs")
EMBED_DIMS = 1536  # 3072 if you use text-embedding-3-large

def get_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL else QdrantClient(host="localhost", port=6333)

def ensure_collection(client: QdrantClient):
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIMS, distance=Distance.COSINE),
        )
    return COLLECTION
