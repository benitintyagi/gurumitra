# app/vector_store.py
import os, json, pathlib
from typing import List, Dict, Any

VECTOR_DB = os.getenv("VECTOR_DB", "chroma").lower()
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1536"))

# ------- Common interface -------
class VectorStore:
    def upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict], vectors: List[List[float]]): ...
    def search(self, query_vector: List[float], top_k: int = 6) -> List[Dict[str, Any]]: ...

# ------- Chroma implementation -------
class ChromaStore(VectorStore):
    def __init__(self, path="chromadb"):
        import chromadb
        path = pathlib.Path(path); path.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.coll = self.client.get_or_create_collection(name="gurumitra")

    def upsert(self, ids, texts, metadatas, vectors):
        self.coll.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)

    def search(self, query_vector, top_k=6):
        q = self.coll.query(query_embeddings=[query_vector], n_results=top_k, include=["documents","metadatas","distances"])
        out = []
        docs = q["documents"][0] if q["documents"] else []
        metas = q["metadatas"][0] if q["metadatas"] else []
        dists = q["distances"][0] if q["distances"] else []
        for i in range(len(docs)):
            out.append({
                "text": docs[i],
                "score": float(dists[i]),          # smaller is closer in Chroma (L2 by default)
                "source": metas[i].get("identifier"),
                "chunk": metas[i].get("chunk"),
                "meta": metas[i],
            })
        return out

# ------- FAISS implementation -------
class FaissStore(VectorStore):
    def __init__(self, path="faiss_index"):
        import faiss, numpy as np
        self.faiss = faiss
        self.np = np
        self.path = pathlib.Path(path); self.path.mkdir(exist_ok=True)
        self.index_file = self.path / "index.bin"
        self.payload_file = self.path / "payload.jsonl"
        self.d = EMBED_DIMS

        if self.index_file.exists() and self.payload_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            self.payloads = [json.loads(l) for l in self.payload_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            # Cosine via inner product on normalized vectors
            self.index = faiss.IndexFlatIP(self.d)
            self.payloads = []

    def _normalize(self, arr):
        # arr: (n, d)
        n = (arr**2).sum(axis=1, keepdims=True)**0.5
        n[n==0] = 1.0
        return arr / n

    def upsert(self, ids, texts, metadatas, vectors):
        import numpy as np
        vec = np.array(vectors, dtype="float32")
        vec = self._normalize(vec)
        # append
        self.index.add(vec)
        for i in range(len(texts)):
            self.payloads.append({"id": ids[i], "text": texts[i], "meta": metadatas[i]})
        # persist
        self.faiss.write_index(self.index, str(self.index_file))
        with self.payload_file.open("w", encoding="utf-8") as f:
            for p in self.payloads:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def search(self, query_vector, top_k=6):
        import numpy as np
        if self.index.ntotal == 0:
            return []
        q = np.array([query_vector], dtype="float32")
        q = self._normalize(q)
        sims, idxs = self.index.search(q, top_k)  # inner product ~ cosine
        out = []
        for rank, (score, idx) in enumerate(zip(sims[0], idxs[0])):
            if idx == -1: continue
            p = self.payloads[idx]
            m = p["meta"]
            out.append({
                "text": p["text"],
                "score": float(score),             # higher is closer for IP/cosine
                "source": m.get("identifier"),
                "chunk": m.get("chunk"),
                "meta": m
            })
        return out

# ------- Factory -------
def get_store() -> VectorStore:
    if VECTOR_DB == "faiss":
        return FaissStore()
    else:
        return ChromaStore()  # default
