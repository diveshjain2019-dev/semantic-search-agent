"""
Phase 5: hybrid search + fusion

Implements:
- Hybrid retrieval: Chroma semantic search + BM25 keyword scoring
- Fusion: Reciprocal Rank Fusion (RRF)
- Answer synthesis: Groq (Llama) over the top fused results

Semantic query embeddings are generated with sentence-transformers:
`SentenceTransformer("all-MiniLM-L6-v2")`.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ENRICHED_PATH = os.path.join(DATA_DIR, "enriched.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "movies"

RRF_K = 60
DEFAULT_TOP_K = 10
SEMANTIC_TOP_K = 50
BM25_TOP_K = 50

# Initialize once at the top (requested).
model = SentenceTransformer("all-MiniLM-L6-v2")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _make_where(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert planner filters to ChromaDB `where` syntax.
    Planner schema example:
      {
        "year_gte": int,
        "year_lte": int,
        "genre": str,
        "vote_gte": float
      }
    """
    where: Dict[str, Any] = {}

    year_gte = filters.get("year_gte")
    year_lte = filters.get("year_lte")
    if year_gte is not None or year_lte is not None:
        year_cond: Dict[str, Any] = {}
        if year_gte is not None:
            year_cond["$gte"] = year_gte
        if year_lte is not None:
            year_cond["$lte"] = year_lte
        where["year"] = year_cond

    genre = filters.get("genre")
    if genre:
        where["genre"] = genre

    vote_gte = filters.get("vote_gte")
    if vote_gte is not None:
        where["vote_average"] = {"$gte": vote_gte}

    return where


def _build_bm25(records: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[str]]:
    corpus_tokens: List[List[str]] = []
    doc_texts: List[str] = []
    for r in records:
        # Build a text field BM25 can use for keyword fallback.
        themes = r.get("themes") or []
        themes_str = " ".join(themes) if isinstance(themes, list) else str(themes)
        doc_text = " ".join(
            filter(
                None,
                [
                    str(r.get("title") or ""),
                    str(r.get("overview") or ""),
                    str(r.get("genre") or ""),
                    themes_str,
                    str(r.get("tone") or ""),
                    str(r.get("pacing") or ""),
                ],
            )
        )
        doc_texts.append(doc_text)
        corpus_tokens.append(_tokenize(doc_text))

    bm25 = BM25Okapi(corpus_tokens)
    return bm25, doc_texts


def _semantic_rank(
    collection: chromadb.Collection,
    plan: Dict[str, Any],
    embedding: List[float],
    n_results: int,
) -> List[str]:
    filters = plan.get("filters") or {}
    where = _make_where(filters)

    # Chroma can error on `{}` where clauses, so only pass `where` when needed.
    # If a filtered query fails for any reason, fall back to an unfiltered query.
    try:
        if where:
            query_res = collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where,
            )
        else:
            query_res = collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
            )
    except Exception:
        query_res = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
    ids = query_res.get("ids", [[]])[0] or []
    return [str(x) for x in ids if x is not None]


def _bm25_rank(
    bm25: BM25Okapi,
    records: List[Dict[str, Any]],
    plan: Dict[str, Any],
    n_results: int,
) -> List[str]:
    keywords = plan.get("keywords") or []
    query_text = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        # No keywords => no meaningful BM25 rank.
        return []

    scores = bm25.get_scores(query_tokens)
    # Get top N indices sorted by score.
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    ranked_ids: List[str] = []
    for i in top_idx:
        ranked_ids.append(str(records[i].get("id")))
    return ranked_ids


def _rrf_fuse(
    semantic_ids: List[str],
    bm25_ids: List[str],
    rrf_k: int = RRF_K,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Fuse two ranked lists using RRF.
    """
    rank_sem = {doc_id: rank for rank, doc_id in enumerate(semantic_ids, start=1)}
    rank_bm25 = {doc_id: rank for rank, doc_id in enumerate(bm25_ids, start=1)}
    all_ids = set(rank_sem.keys()) | set(rank_bm25.keys())

    fused: List[Tuple[str, float]] = []
    for doc_id in all_ids:
        score = 0.0
        if doc_id in rank_sem:
            score += 1.0 / (rrf_k + rank_sem[doc_id])
        if doc_id in rank_bm25:
            score += 1.0 / (rrf_k + rank_bm25[doc_id])
        fused.append((doc_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_k]

    results: List[Dict[str, Any]] = []
    for doc_id, score in fused:
        results.append(
            {
                "id": doc_id,
                "fused_score": score,
                "semantic_rank": rank_sem.get(doc_id),
                "bm25_rank": rank_bm25.get(doc_id),
            }
        )
    return results


def hybrid_search(plan: Dict[str, Any], top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Runs hybrid retrieval and RRF fusion.

    Returns fused ranked list of enriched records with ranks and scores.
    """
    if not os.path.exists(ENRICHED_PATH):
        raise FileNotFoundError(f"Missing {ENRICHED_PATH} — run ingest.py and embeddings.py first.")

    with open(ENRICHED_PATH, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    # Build BM25 index over the enriched records.
    bm25, _ = _build_bm25(records)
    id_to_record = {str(r.get("id")): r for r in records}

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    query_embedding = model.encode(plan["semantic_query"]).tolist()

    semantic_ids = _semantic_rank(
        collection=collection,
        plan=plan,
        embedding=query_embedding,
        n_results=SEMANTIC_TOP_K,
    )
    bm25_ids = _bm25_rank(
        bm25=bm25,
        records=records,
        plan=plan,
        n_results=BM25_TOP_K,
    )

    fused = _rrf_fuse(
        semantic_ids=semantic_ids,
        bm25_ids=bm25_ids,
        rrf_k=RRF_K,
        top_k=top_k,
    )

    # Attach full record fields for downstream UI.
    out: List[Dict[str, Any]] = []
    for item in fused:
        r = id_to_record.get(item["id"], {})
        out.append({**r, **item})
    return out


def synthesize_answer(
    user_query: str,
    plan: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> str:
    """
    Use Groq to synthesize a short answer referencing the top results.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = "llama-3.3-70b-versatile"

    top = results[:10]
    context = [
        {
            "id": r.get("id"),
            "title": r.get("title"),
            "year": r.get("year"),
            "genre": r.get("genre"),
            "vote_average": r.get("vote_average"),
            "budget": r.get("budget"),
            "tone": r.get("tone"),
            "pacing": r.get("pacing"),
            "overview": (r.get("overview") or "")[:600],
        }
        for r in top
    ]

    prompt = f"""You are a helpful movie recommendation assistant.

User query:
{user_query}

Planned filters + reasoning:
{json.dumps(plan, ensure_ascii=False)}

Candidate movies (ranked by hybrid retrieval):
{json.dumps(context, ensure_ascii=False)}

Task:
1. Recommend 3-5 movies from the candidates.
2. For each recommendation, briefly explain why it matches the user's intent (use tone/themes/pacing and genre where relevant).
3. If candidates are weak matches, say so and recommend the closest alternatives.

Constraints:
- Keep it concise (under ~250 words).
- Do not mention internal scoring algorithms like BM25 or RRF.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def search_and_synthesize(user_query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper: hybrid_search + Claude synthesis.
    """
    results = hybrid_search(plan)
    answer = synthesize_answer(user_query=user_query, plan=plan, results=results)
    return {"results": results, "answer": answer}

