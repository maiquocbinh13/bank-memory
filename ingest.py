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

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(BASE_DIR, "data", "cases.json")

# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def load_cases(path=None):
    if path is None:
        path = CASES_PATH
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cases(cases, path=None):
    if path is None:
        path = CASES_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_cases(cases):
    chunks = []
    for case in cases:
        meta = case.get("metadata", {}) if isinstance(case, dict) else {}
        # Flat structure (original 10): fields are direct children
        # Nested structure (newly ingested): fields live under metadata
        ctx = case.get("context", "") or meta.get("context", "") or case.get("text", "") or ""
        out = case.get("outcome", "") or meta.get("outcome", "") or "New entry from user"
        les = case.get("lessons", "") or meta.get("lessons", "") or (case.get("text", "") if isinstance(case, dict) else "")
        title = case.get("title", "") or meta.get("title", "") or "Untitled"
        cid = case.get("id") or meta.get("id") or f"user-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        ctype = case.get("type") or meta.get("type") or "general"
        tags = case.get("tags", []) or meta.get("tags") or []
        text = f"{title}. {ctx} {out} {les}"
        chunks.append({
            "id": cid,
            "type": ctype,
            "tags": tags,
            "text": text,
            "metadata": case
        })
    return chunks

# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(chunks):
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    return index, model, chunks

# ---------------------------------------------------------------------------
# Creating new chunks from user content
# ---------------------------------------------------------------------------

def create_chunk_from_text(text, source="user-paste", title=None):
    """
    Turn a user's pasted/URL/file text into a minimal chunk entry.
    We keep it minimal: the text is what gets embedded, and we store the
    raw text as well so the UI can display it.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return None

    # Derive a short title: first heading if present, else the first sentence
    # (not the raw blob), capped at ~90 chars so cards stay readable.
    title = text[:90]
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if line.startswith("#") or len(line) < 90:
            title = line if line else title
            break
    chunk = {
        "id": f"user-{int(time.time())}-{uuid.uuid4().hex[:6]}",
        "title": title[:120],
        "type": "general",
        "tags": [],
        "text": text,
        "metadata": {
            "title": title[:120],
            "context": text,
            "outcome": "New entry from user",
            "lessons": text,
            "tags": [],
            "type": "general",
            "source": source,
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    }
    return chunk

# ---------------------------------------------------------------------------
# Text extraction from different sources
# ---------------------------------------------------------------------------

def _fetch_url(url):
    """Fetch text from a URL. Supports plain text, HTML, markdown."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120 Safari/537.36"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "text/markdown" in ct or "text/plain" in ct:
                return raw.decode("utf-8", errors="replace")
            if "text/html" in ct:
                return _strip_html(raw.decode("utf-8", errors="replace"))
            return raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise ValueError(f"Could not fetch URL: {e}")

def _strip_html(html_text):
    """Very small HTML to text converter (title + paragraphs + headings + lists)."""
    import re
    import html as h
    text = h.unescape(html_text)
    text = re.sub(r"<(p|h[1-6]|li|div|br)\b[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<(strong|b).*?</\1>", " ", text, flags=re.I)
    text = re.sub(r"<(em|i).*?</\1>", " ", text, flags=re.I)
    text = re.sub(r"<(code|pre).*?</\1>", lambda m: "\n" + m.group(0) + "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = h.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

def extract_text(kind, content=None, url=None, file_base64=None, filename=None):
    """
    kind: "paste" | "url" | "file"
    Returns normalized text, or raises ValueError.
    """
    if kind == "paste":
        return content or ""
    elif kind == "url":
        if not url:
            raise ValueError("Missing URL")
        return _fetch_url(url)
    elif kind == "file":
        if not file_base64 or not filename:
            raise ValueError("Missing file data or filename")
        try:
            raw = base64.b64decode(file_base64)
        except Exception as e:
            raise ValueError(f"Could not decode file: {e}")

        ext = os.path.splitext(filename)[1].lower()
        mime = mimetypes.guess_type(filename)[0]

        if mime and "text" in mime:
            try:
                return raw.decode("utf-8", errors="replace")
            except Exception:
                pass
            return raw.decode("latin-1", errors="replace")

        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return raw.decode("latin-1", errors="replace")
    else:
        raise ValueError(f"Unknown kind: {kind}")

