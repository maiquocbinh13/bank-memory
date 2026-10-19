def generate_answer(query, results):
    if not results:
        return {
            "summary": "No similar past cases found. Refine your query or provide more context.",
            "confidence": "low",
            "recommendations": ["Add domain-specific keywords or upload a richer case description."],
            "cases": []
        }

    top = results[0]["metadata"]
    out_cases = []
    for r in results:
        c = r["metadata"]
        out_cases.append({
            "id": c["id"],
            "title": c["title"],
            "context": c["context"],
            "outcome": c["outcome"],
            "lessons": c["lessons"],
            "similarity": r["similarity"],
            "tags": c["tags"],
            "type": c["type"]
        })

    query_lower = query.lower()
    query_tags = [t for t in query_lower.split() if len(t) > 3]
    recommended = []

    matched_tags = [r["metadata"]["tags"] for r in results]
    flat_tags = [t for sublist in matched_tags for t in sublist]

    if any("compliance" in t for t in flat_tags):
        recommended.append("Prioritize structured compliance checklists and external verification.")
    if any("deadline" in t or "escalation" in t for t in flat_tags):
        recommended.append("Map task dependencies early and flag blockers before they ripple.")
    if any("estimation" in t or "quote" in t for t in flat_tags):
        recommended.append("Break the request into components and track estimate-vs-actual accuracy.")
    if any("documentation" in t or "handoff" in t for t in flat_tags):
        recommended.append("Create explicit handoff templates with ownership and next steps.")
    if not recommended:
        recommended.append("Apply best practices from the top matched case; document decisions explicitly.")

    return {
        "summary": f"Found {len(results)} similar past cases. Top match: {top['title']}.",
        "confidence": "medium" if len(results) >= 2 else "low",
        "recommendations": recommended,
        "cases": out_cases
    }
