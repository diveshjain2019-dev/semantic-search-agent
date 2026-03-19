# Semantic Search Agent

## What this is
A natural language search layer over the TMDB movie dataset.
Users type free-text queries; the agent decomposes them into
structured filters + semantic search + keyword search, fuses
results, and explains the reasoning.

## Stack
- Claude API (claude-sonnet-4-6) for query planning and answer synthesis
- ChromaDB for vector storage and filtered search
- rank_bm25 for keyword search
- Streamlit for the UI
- Pandas for data loading

## Key files
- src/ingest.py    — loads TMDB CSVs, enriches metadata with Claude, saves enriched JSON
- src/embeddings.py — embeds each record, upserts into ChromaDB
- src/planner.py   — Claude query planner: decomposes query into filters + semantic + keywords
- src/search.py    — runs hybrid search, RRF fusion, returns ranked results
- src/app.py       — Streamlit UI: search bar, results panel, reasoning display

## Data model (enriched record)
{
  "id": str,
  "title": str,
  "overview": str,
  "year": int,
  "genre": str,           # primary genre only
  "vote_average": float,
  "budget": int,
  "tone": str,            # Claude-inferred: dark / lighthearted / tense / comedic
  "themes": [str],        # Claude-inferred: 3-5 keywords
  "pacing": str,          # Claude-inferred: slow-burn / moderate / fast-paced
  "embed_text": str       # combined field used for embedding
}

## Search plan schema (output of planner.py)
{
  "filters": {            # passed directly to ChromaDB .where()
    "year_gte": int,
    "year_lte": int,
    "genre": str,
    "vote_gte": float
  },
  "semantic_query": str,  # rewritten for embedding similarity
  "keywords": [str],      # for BM25 fallback
  "reasoning": str        # one sentence explaining decomposition
}