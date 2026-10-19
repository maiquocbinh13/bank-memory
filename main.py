"""
Memory Bank — FastAPI entry point.

Serves:
  GET  /                    -> the landing page (ui/index.html)
  GET  /api/library         -> list all indexed entries
  POST /api/ingest          -> add a new entry (paste/url/file)
  GET  /api/search          -> real search against the full corpus

Built on fastapi + sentence-transformers + faiss.
Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import json
import os
import time
import uuid
import re
import urllib.request
import urllib.error
import base64
import mimetypes
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from ingest import (
    load_cases, save_cases, chunk_cases, build_index,
    create_chunk_from_text, extract_text
)
from retrieve import get_pipeline, search as retrieve_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(BASE_DIR, "data", "cases.json")
UI_PATH = os.path.join(BASE_DIR, "ui", "index.html")

app = FastAPI(title="Memory Bank API")

# ---------------------------------------------------------------------------
# Startup: build index from the current cases.json
# ---------------------------------------------------------------------------
index, model, cases, chunks = get_pipeline()

@app.on_event("startup")
def startup():
    global index, model, cases, chunks
    index, model, cases, chunks = get_pipeline()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return open(UI_PATH).read()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
@app.get("/api/library")
async def library(q: str = ""):
    q = (q or "").strip().lower()
    if not q:
        return {"entries": cases, "total": len(cases)}

    results = []
    for c in cases:
        meta = c.get("metadata", {}) if isinstance(c, dict) else {}
        # Case can be flat (original 10) or nested under metadata (newly ingested)
        title = c.get("title", "") or meta.get("title", "")
        context = c.get("context", "") or meta.get("context", "")
        lessons = c.get("lessons", "") or meta.get("lessons", "")
        tags = c.get("tags", []) or meta.get("tags", [])
        haystack = " ".join([title or "", context or "", lessons or "", " ".join(tags) if isinstance(tags, list) else tags]).lower()
        if q in haystack:
            results.append(c)
    return {"entries": results, "total": len(results)}

# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
@app.post("/api/ingest")
async def ingest(request: Request):
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

    cases.append(chunk)
    save_cases(cases)

    global index, model, chunks
    chunks = chunk_cases(cases)
    index, model, chunks = build_index(chunks)

    return JSONResponse({
        "entry": chunk,
        "total_entries": len(cases),
        "note": "Reindexing completed. Your new entry is now searchable."
    })

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@app.get("/api/search")
async def web_search(q: str = ""):
    q = (q or "").strip()
    if not q:
        return JSONResponse({
            "summary": "Enter a query to search your memory bank.",
            "confidence": "none",
            "recommendations": [],
            "cases": []
        })
    results = retrieve_search(q, index, model, cases, chunks, top_k=5)
    return {
        "query": q,
        "total_matches": len(results),
        "cases": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
