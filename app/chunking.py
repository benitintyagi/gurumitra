# app/chunking.py
from typing import List, Dict
import tiktoken

def token_chunks(text: str, max_tokens: int = 400, overlap: int = 40, model: str = "gpt-4o-mini") -> List[str]:
    enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text)
    chunks = []
    i = 0
    while i < len(ids):
        chunk_ids = ids[i:i+max_tokens]
        chunks.append(enc.decode(chunk_ids))
        i += max_tokens - overlap
    return chunks

def build_payloads(doc: Dict, max_tokens=400, model="gpt-4o-mini"):
    content = doc["content"]; meta = doc["meta"]
    chunks = token_chunks(content, max_tokens=max_tokens, overlap=40, model=model)
    for idx, ch in enumerate(chunks):
        yield {
            "id": f'{meta["identifier"]}:{idx}',
            "text": ch,
            "metadata": {**meta, "chunk": idx}
        }
