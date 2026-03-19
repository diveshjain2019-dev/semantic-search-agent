"""
Phase 4: Claude query planner

This file decomposes user queries into:
- filters (for ChromaDB .where())
- semantic_query (for embedding similarity)
- keywords (for BM25)
- reasoning (one sentence)

It uses the Groq API for the chat completion call.
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GROQ_MODEL = "llama-3.3-70b-versatile"


def _json_only_system_prompt() -> str:
    return (
        "You are Claude acting as a query planning assistant for a movie semantic search system.\n"
        "Return ONLY a single valid JSON object (no markdown, no backticks, no preamble, no trailing text).\n"
        "The JSON must be parseable with json.loads().\n"
        "Either return a search plan with keys: filters, semantic_query, keywords, reasoning\n"
        "OR return a clarification request with keys: needs_clarification (true) and clarification_question.\n"
    )


def _build_plan_request(user_query: str) -> str:
    return f"""Decompose the following user query for hybrid movie retrieval.

Rules:
1. filters is an object. Only include filter fields that the query actually mentions.
   Allowed filter fields:
   - year_gte (int)
   - year_lte (int)
   - genre (string, primary genre)
   - vote_gte (float)
   If a value is not explicitly mentioned, do not include that field.
2. semantic_query is a rewritten query optimized for embedding similarity.
3. keywords is a list of important terms optimized for BM25 keyword matching.
4. reasoning is exactly one sentence explaining the decomposition.

Vagueness rule:
- If the query is too vague to produce meaningful filters/keywords, return:
  {{
    "needs_clarification": true,
    "clarification_question": "..."
  }}

User query:
{user_query}
"""


def _build_stricter_retry_request(user_query: str) -> str:
    return f"""Return ONLY the required JSON object and nothing else.

User query:
{user_query}

If you are unsure or can't produce a meaningful plan, return:
{{"needs_clarification": true, "clarification_question": "..."}}.

Otherwise return:
{{
  "filters": {{"year_gte": 2020, "year_lte": 2021, "genre": "Drama", "vote_gte": 7.5}},
  "semantic_query": "rewritten embedding query",
  "keywords": ["keyword1", "keyword2"],
  "reasoning": "one sentence"
}}
"""


def _parse_or_raise(text: str) -> Dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Response JSON must be an object.")
    return parsed


def plan_query(user_query: str) -> dict:
    """
    Plan the query for retrieval.

    Returns either:
    - search plan dict: {filters, semantic_query, keywords, reasoning}
    - clarification dict: {needs_clarification: true, clarification_question: str}
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set in .env")

    client = Groq(api_key=api_key)

    messages_base = [
        {"role": "system", "content": _json_only_system_prompt()},
        {"role": "user", "content": _build_plan_request(user_query)},
    ]

    def attempt(messages):
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=messages,
        )
        content = (resp.choices[0].message.content or "").strip()
        return _parse_or_raise(content)

    try:
        return attempt(messages_base)
    except (json.JSONDecodeError, ValueError):
        # Retry once with a stricter instruction.
        messages_retry = [
            {"role": "system", "content": _json_only_system_prompt()},
            {"role": "user", "content": _build_stricter_retry_request(user_query)},
        ]
        return attempt(messages_retry)
