# app/main.py
import os, json, asyncio
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from dotenv import load_dotenv
from app.rag import retrieve, generate_answer
from openai import OpenAI

load_dotenv()

app = FastAPI(title="GuruMitra API", version="1.0.0")
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/query")
def query(payload=Body(...)):
    q = payload.get("question", "").strip()
    if not q:
        return JSONResponse({"error":"question required"}, status_code=400)
    passages = retrieve(q, top_k=6)
    answer = generate_answer(q, passages)
    return {"answer": answer, "sources": passages}

# Optional: SSE streaming using chunked generation (emulates stream)
# For full streaming, adapt to OpenAI Responses API and yield deltas.
@app.post("/stream")
async def stream(payload=Body(...)):
    q = payload.get("question","").strip()
    if not q: return StreamingResponse(iter([b"data: {\"error\":\"question required\"}\n\n"]), media_type="text/event-stream")
    passages = retrieve(q, top_k=6)

    async def event_gen():
        # send context first
        yield f"data: {json.dumps({'event':'context','sources':passages})}\n\n"
        # simple non-token streaming: chunk the final answer by sentences
        full = generate_answer(q, passages)
        for line in full.split(". "):
            yield f"data: {json.dumps({'event':'delta','text': line + '. '})}\n\n"
            await asyncio.sleep(0.02)
        yield "data: {\"event\":\"done\"}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/")
def landing():
    # tiny demo page (optional)
    html = (Path(__file__).parent / "static" / "chat.html").read_text()
    return HTMLResponse(html)
