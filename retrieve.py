import numpy as np
import os
from ingest import load_cases, chunk_cases, build_index

def search(query, index, model, cases, chunks, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        case = chunks[idx]["metadata"]
        results.append({
            "id": case["id"],
            "title": case["title"],
            "type": case["type"],
            "similarity": float(distances[0][i]),
            "metadata": case
        })
    return results

def get_pipeline():
    cases = load_cases()
    chunks = chunk_cases(cases)
    index, model, chunks = build_index(chunks)
    return index, model, cases, chunks
