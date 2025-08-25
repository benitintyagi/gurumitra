# app/rag.py
import os
from typing import List, Tuple
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from app.qdrant_utils import get_client, COLLECTION
from app.prompts import SYSTEM, USER_TEMPLATE
from dotenv import load_dotenv

load_dotenv()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
client = OpenAI()

def retrieve(q: str, top_k=6, lang=None) -> List[dict]:
    qd: QdrantClient = get_client()
    # Hybrid: pure vector search via Qdrant's "search" with remote embedding
    # We'll embed query locally using the same embedding model for better results.
    emb = client.embeddings.create(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"), input=q).data[0].embedding
    hits = qd.search(collection_name=COLLECTION, query_vector=emb, limit=top_k, with_payload=True)
    return [{"text": h.payload["text"], "score": h.score, "source": h.payload.get("identifier"), "chunk": h.payload.get("chunk")} for h in hits]

def build_context(snippets: List[dict]) -> str:
    lines = []
    for s in snippets:
        tag = f'{s.get("source","?")}#{s.get("chunk","?")}'
        lines.append(f"- ({tag}) {s['text'][:800]}")
    return "\n".join(lines)

def generate_answer(question: str, snippets: List[dict]) -> str:
    context = build_context(snippets)
    user = USER_TEMPLATE.format(question=question, context=context)
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user}
        ]
    )
    return resp.choices[0].message.content
