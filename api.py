"""
Web UI API for the Memory Bank landing page.

Endpoints:
  GET  /api/library      - list all indexed entries (with search)
  POST /api/ingest      - add a new entry (paste / url / file)
  POST /api/search      - (optional) delegate to search for consistency
"""
import json
import os
import time
import base64
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# We reuse the core logic
from ingest import (
    load_cases, save_cases, chunk_cases, build_index,
    create_chunk_from_text, extract_text
)
from retrieve import get_pipeline, search as retrieve_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(BASE_DIR, "data", "cases.json")

app = FastAPI(title="Memory Bank Web API")

# ---------------------------------------------------------------------------
# Startup: build index from the current cases.json
# ---------------------------------------------------------------------------
index, model, cases, chunks = get_pipeline()

@app.on_event("startup")
def startup():
    global index, model, cases, chunks
    index, model, cases, chunks = get_pipeline()

@app.get("/api/library")
async def library(q: str = ""):
    """
    Return all library entries, optionally filtered by a search query.
    For MVP we do a lightweight text match so users can browse the corpus.
    """
    q = (q or "").strip().lower()
    if not q:
        return {"entries": cases, "total": len(cases)}

    results = []
    for c in cases:
        haystack = " ".join([
            c.get("title", ""),
            c.get("context", ""),
            c.get("lessons", ""),
            c.get("tags", [])
        ]).lower()
        if q in haystack:
            results.append(c)
    return {"entries": results, "total": len(results)}

@app.post("/api/ingest")
async def ingest(request: Request):
    """
    Ingest new content into the Memory Bank.
    Body (form-encoded or JSON):
      kind: "paste" | "url" | "file"
      content: text for paste
      url: URL for url
      file_base64 / filename: multipart files
    Returns the new entry + updated library count.
    """
    body = await request.form()
    kind = body.get("kind", "paste")
    content = body.get("content", "")
    url = body.get("url", "")
    file_base64 = body.get("file_base64", "")
    filename = body.get("filename", "")

    try:
        text = extract_text(kind, content, url, file_base64, filename)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if not text.strip():
        return JSONResponse({"error": "Empty content. Nothing to index."}, status_code=400)

    chunk = create_chunk_from_text(text, source=kind, title=body.get("title", ""))
    if not chunk:
        return JSONResponse({"error": "Could not create entry from empty text."}, status_code=400)

    # Append to corpus
    cases.append(chunk)
    save_cases(cases)

    # Rebuild the index in memory (fine for MVP scale)
    global index, model, chunks
    chunks = chunk_cases(cases)
    index, model, chunks = build_index(chunks)

    return JSONResponse({
        "entry": chunk,
        "total_entries": len(cases),
        "note": "Reindexing completed. Your new entry is now searchable."
    })

@app.get("/api/search")
async def web_search(q: str = ""):
    """
    Simple search endpoint that returns real matches against the full corpus.
    """
    q = (q or "").strip()
    if not q:
        return JSONResponse({
            "summary": "Enter a query to search your memory bank.",
            "confidence": "none",
            "recommendations": [],
            "cases": []
        })

    # Use the core search
    results = retrieve_search(q, index, model, cases, chunks, top_k=5)
    return {
        "query": q,
        "total_matches": len(results),
        "cases": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
