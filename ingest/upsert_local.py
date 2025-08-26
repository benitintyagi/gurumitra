# ingest/upsert_local.py
import os, json, pathlib
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from app.chunking import build_payloads
from app.vector_store import get_store

load_dotenv()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DATA_DIR = pathlib.Path("processed_clean")  # use your cleaned corpus

client = OpenAI()
store = get_store()

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

def main():
    for p in DATA_DIR.glob("*.json"):
        doc = json.loads(p.read_text(encoding="utf-8"))
        batch_texts, ids, metas = [], [], []
        for pl in build_payloads(doc, max_tokens=450):
            batch_texts.append(pl["text"])
            ids.append(pl["id"])
            metas.append(pl["metadata"])
            if len(batch_texts) >= 64:
                vectors = embed_texts(batch_texts)
                store.upsert(ids, batch_texts, metas, vectors)
                batch_texts, ids, metas = [], [], []
        if batch_texts:
            vectors = embed_texts(batch_texts)
            store.upsert(ids, batch_texts, metas, vectors)
        print("Upserted", p.name)

if __name__ == "__main__":
    main()
