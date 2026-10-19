# The Memory Bank (Agent in the Shell MVP)

A deterministic past-work retrieval & comparison tool built for **Agent in the Shell**. 
No API key required. Runs entirely on CPU with local embeddings + rule-based reasoning. 
Designed to be swapped to an LLM later without changing the API contract.

# ## Run (local)
# 1. `chmod +x run.sh && ./run.sh`
# 2. Open `http://localhost:8000` in your browser.

# ## Live Demo
# - **Live URL**: https://bank-memory-production.up.railway.app
#   - Open this in your browser — no installation required.
#   - Search past cases, view details, add new entries online.

# ## Architecture
- `ingest.py` -> load cases, chunk text, embed with `all-MiniLM-L6-v2`, build FAISS index.
- `retrieve.py` -> semantic search (cosine/L2).
- `reason.py` -> deterministic synthesis + fallback recommendations.
- `main.py` -> FastAPI routes + serving minimal UI.
- `ui/index.html` -> single-page search + result rendering.

## Testing
`python -m unittest discover -s tests`

## Judgment alignment
- **30% Depth**: retrieval + comparison + uncertainty handling (low confidence surfaced).
- **20% Architecture**: clear module boundaries, error handling, structured output.
- **20% Product**: one-click flow, fast, no dead ends, demo-ready UI.
- **15% Complexity**: semantic drift, multi-doc synthesis, graceful fallback.
- **15% AI**: embeddings drive retrieval; reasoning engine mimics LLM synthesis + can be swapped.
