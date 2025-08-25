# ingest/upsert_qdrant.py
import os, json, pathlib
from typing import List
from dotenv import load_dotenv
from app.chunking import build_payloads
from app.qdrant_utils import get_client, ensure_collection, COLLECTION
from qdrant_client.models import PointStruct
from openai import OpenAI

load_dotenv()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    qd = get_client()
    ensure_collection(qd)
    processed = pathlib.Path("processed_clean")
    for p in processed.glob("*.json"):
        doc = json.loads(p.read_text())
        batch_texts, ids, metas = [], [], []
        for pl in build_payloads(doc, max_tokens=450):
            batch_texts.append(pl["text"]); ids.append(pl["id"]); metas.append(pl["metadata"])
            if len(batch_texts) >= 64:
                vectors = embed_texts(batch_texts)
                qd.upsert(collection_name=COLLECTION, points=[
                    PointStruct(id=ids[i], vector=vectors[i], payload={"text": batch_texts[i], **metas[i]})
                    for i in range(len(batch_texts))
                ])
                batch_texts, ids, metas = [], [], []
        if batch_texts:
            vectors = embed_texts(batch_texts)
            qd.upsert(collection_name=COLLECTION, points=[
                PointStruct(id=ids[i], vector=vectors[i], payload={"text": batch_texts[i], **metas[i]})
                for i in range(len(batch_texts))
            ])
        print("Upserted", p.name)

if __name__ == "__main__":
    main()
