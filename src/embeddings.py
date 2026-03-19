"""
Phase 3: embed + store

Loads `data/enriched.json`, generates embeddings for `embed_text` using
sentence-transformers, and upserts records into a persistent ChromaDB
collection (`movies`) with metadata fields used by filtering.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ENRICHED_PATH = DATA_DIR / "enriched.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "movies"

# SentenceTransformer embedding model requested by the user.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _sanitize_metadata_value(v: Any) -> Any:
    """Chroma metadata should be JSON-serializable. Convert NaN -> None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main() -> None:
    if not ENRICHED_PATH.exists():
        raise SystemExit(f"Missing {ENRICHED_PATH} — run ingest.py first.")

    with open(ENRICHED_PATH, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    # Only embed the first N records (useful for iterative development).
    records = records[:500]

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Skip records already stored.
    existing = collection.get()
    existing_ids = set(existing.get("ids", []) or [])

    to_embed = [r for r in records if str(r.get("id")) not in existing_ids]
    print(f"Chroma: {len(existing_ids)} existing IDs, {len(to_embed)} to embed.")

    embedded_count = 0
    model = SentenceTransformer(EMBEDDING_MODEL)
    for idx, row in enumerate(tqdm(to_embed, desc="Embedding", unit="movie")):
        movie_id = str(row.get("id"))
        embed_text: str = str(row.get("embed_text") or "")
        if not embed_text.strip():
            # Nothing to embed; skip rather than error.
            continue

        try:
            embedding = model.encode(row["embed_text"]).tolist()
        except Exception as e:
            err = str(e)
            print(
                f"\nSentenceTransformer embedding failed: {err}",
                file=sys.stderr,
            )
            print(
                "Aborting embedding run so you don't get a partial/incorrect index.",
                file=sys.stderr,
            )
            return

        metadata = {
            "year": _sanitize_metadata_value(row.get("year")),
            "genre": row.get("genre"),
            "vote_average": _sanitize_metadata_value(row.get("vote_average")),
            "budget": _sanitize_metadata_value(row.get("budget")),
            "tone": row.get("tone"),
            "pacing": row.get("pacing"),
        }

        collection.upsert(
            ids=[movie_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        embedded_count += 1
        if embedded_count % 100 == 0:
            print(f"Upserted {embedded_count} / {len(to_embed)} embeddings...")

    print(f"Done. Upserted {embedded_count} new movies into ChromaDB.")


if __name__ == "__main__":
    main()
