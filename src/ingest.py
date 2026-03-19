"""
Phase 2: Load TMDB CSV, clean, enrich with OpenAI (tone/themes/pacing), save enriched JSON.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "tmdb_5000_movies.csv"
OUT_PATH = DATA_DIR / "enriched.json"
BATCH_SIZE = 50
KEEP_COLUMNS = ["id", "title", "overview", "year", "genre", "vote_average", "budget"]


def first_genre(genres_str: str) -> str:
    """Parse JSON genres string and return the first genre name, or empty string."""
    if pd.isna(genres_str) or not str(genres_str).strip():
        return ""
    try:
        arr = json.loads(genres_str)
        if arr and isinstance(arr, list):
            first = arr[0]
            if isinstance(first, dict) and "name" in first:
                return first["name"]
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def year_from_release(release_date: str) -> Optional[int]:
    """Parse year from release_date (YYYY-MM-DD or similar)."""
    if pd.isna(release_date):
        return None
    s = str(release_date).strip()
    match = re.match(r"(\d{4})", s)
    return int(match.group(1)) if match else None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows, parse year/genre, keep only required columns."""
    df = df.dropna(subset=["overview", "title"])
    df = df[df["overview"].astype(str).str.strip() != ""]
    df = df[df["title"].astype(str).str.strip() != ""]

    df["year"] = df["release_date"].map(year_from_release)
    df["genre"] = df["genres"].astype(str).map(first_genre)

    df = df.rename(columns={c: c for c in KEEP_COLUMNS if c in df.columns})
    return df[KEEP_COLUMNS]


def placeholder_metadata(row: dict) -> dict:
    """Fallback when API is unavailable: derive tone/themes/pacing from genre and overview."""
    genre = (row.get("genre") or "").lower()
    overview = (row.get("overview") or "")[:200]
    tone = "neutral"
    if any(s in genre for s in ("horror", "thriller")):
        tone = "tense"
    elif any(s in genre for s in ("comedy", "animation")):
        tone = "lighthearted"
    elif any(s in genre for s in ("drama", "war")):
        tone = "dark"
    themes = [g.strip() for g in genre.split(",") if g.strip()][:5] or ["drama"]
    pacing = "moderate"
    return {"tone": tone, "themes": themes, "pacing": pacing}


def infer_metadata(client: Optional[OpenAI], row: dict) -> dict:
    """Call OpenAI to infer tone, themes, pacing; on failure or no client, use placeholders."""
    if client is None:
        return placeholder_metadata(row)
    title = row.get("title", "")
    overview = row.get("overview", "")
    prompt = f"""Given this movie, respond with exactly a JSON object (no markdown, no extra text) with keys: "tone", "themes", "pacing".

Movie: {title}
Overview: {overview}

Rules:
- tone: one of: dark, lighthearted, tense, comedic, neutral, mixed (one word or short phrase).
- themes: array of 3-5 short theme keywords (e.g. ["redemption", "family", "war"]).
- pacing: one of: slow-burn, moderate, fast-paced.

JSON only:"""

    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        err = str(e).lower()
        if "credit" in err or "balance" in err or "billing" in err or "rate" in err or "quota" in err:
            return placeholder_metadata(row)
        raise


def enrich_row(client: Optional[OpenAI], row: dict) -> dict:
    """Add tone, themes, pacing and embed_text to a row."""
    inferred = infer_metadata(client, row)
    row["tone"] = inferred.get("tone", "")
    row["themes"] = inferred.get("themes", [])
    row["pacing"] = inferred.get("pacing", "")
    themes_str = " ".join(row["themes"]) if isinstance(row["themes"], list) else str(row["themes"])
    row["embed_text"] = " ".join(
        filter(None, [str(row["title"]), str(row["overview"]), themes_str, str(row["tone"])])
    )
    return row


def main() -> None:
    skip_api = os.getenv("SKIP_CLAUDE_ENRICH", "").strip().lower() in ("1", "true", "yes")
    api_key = os.getenv("OPENAI_API_KEY")
    client = None
    if not skip_api and api_key:
        client = OpenAI(api_key=api_key)
    elif skip_api:
        print("SKIP_CLAUDE_ENRICH=1: using placeholder tone/themes/pacing (no API calls).")
    else:
        print("OPENAI_API_KEY not set; using placeholder tone/themes/pacing (no API calls).")

    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH} — add tmdb_5000_movies.csv to data/")

    df = pd.read_csv(CSV_PATH)
    df = clean_df(df)
    records = df.to_dict("records")

    enriched = []
    used_placeholder = False
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Batches", unit="batch"):
        batch = records[i : i + BATCH_SIZE]
        for row in batch:
            try:
                enriched.append(enrich_row(client, row.copy()))
            except Exception as e:
                err = str(e).lower()
                if "credit" in err or "balance" in err or "billing" in err or "rate" in err or "quota" in err:
                    if not used_placeholder:
                        print(
                            "\nOpenAI API: rate limit or insufficient credits. Continuing with placeholder tone/themes/pacing for this run.",
                            file=sys.stderr,
                        )
                        used_placeholder = True
                    client = None
                    enriched.append(enrich_row(None, row.copy()))
                else:
                    raise
        if client and i + BATCH_SIZE < len(records):
            time.sleep(1)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(enriched)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
