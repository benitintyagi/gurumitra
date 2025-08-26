# app/rag.py
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from app.prompts import SYSTEM, USER_TEMPLATE
from app.vector_store import get_store

load_dotenv()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()
store = get_store()

def retrieve(q: str, top_k=6) -> List[dict]:
    emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
    hits = store.search(emb, top_k=top_k)
    return hits

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
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content
