from html import escape
from pathlib import Path

from dotenv import load_dotenv

# Load env vars before importing modules that depend on them.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import streamlit as st

from src.planner import plan_query
from src.search import hybrid_search, synthesize_answer


# Global styling (Netflix / cinema theme)
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

      /* Page + typography */
      body {
        background-color: #0a0a0a;
        color: #e6e6e6;
        font-family: 'Inter', sans-serif;
      }
      h1, h2, h3, h4, p, label, div, span {
        color: #f5f5f5;
      }

      /* Remove Streamlit default padding/max-width */
      .block-container {
        padding-top: 0rem !important;
        padding-right: 1rem !important;
        padding-left: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
      }

      /* Hide hamburger menu + footer */
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }

      /* Input styling */
      div[data-baseweb="input"] input,
      div[data-baseweb="textarea"] textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
      }
      div[data-baseweb="input"] input::placeholder,
      div[data-baseweb="textarea"] textarea::placeholder {
        color: #7a7a7a !important;
      }
      div[data-baseweb="input"] input:focus,
      div[data-baseweb="textarea"] textarea:focus {
        border-color: #e50914 !important;
        box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.20) !important;
      }

      /* Button styling */
      div.stButton > button {
        background-color: #e50914 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.15rem !important;
        font-weight: 700 !important;
      }
      div.stButton > button:hover {
        background-color: #b20710 !important;
      }
      div.stButton > button:disabled {
        opacity: 0.45 !important;
        cursor: not-allowed !important;
        background-color: #e50914 !important;
      }

      /* Header banner */
      .cineseek-banner {
        width: 100%;
        background: #0a0a0a;
        padding: 2.5rem 1rem 1.5rem 1rem;
        border-bottom: 1px solid #2a2a2a;
      }
      .cineseek-title {
        text-align: center;
        font-size: 3.1rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin: 0;
        color: #ffffff;
      }
      .cineseek-title-underline {
        display: block;
        width: 260px;
        height: 4px;
        background: #e50914;
        margin: 1rem auto 1rem auto;
      }
      .cineseek-subtitle {
        text-align: center;
        color: #9a9a9a;
        font-size: 1.05rem;
        margin-top: 0;
      }
      .cineseek-divider {
        height: 1px;
        background: #e50914;
        width: 100%;
      }

      /* Reasoning pill */
      .reason-pill {
        display: inline-block;
        background: #2a2a2a;
        color: #bdbdbd;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin: 0.25rem 0 0.5rem 0;
        max-width: 100%;
        word-wrap: break-word;
      }

      /* Clarification box */
      .clarification-box {
        border: 1px solid #f5c518;
        background: rgba(245, 197, 24, 0.10);
        color: #f5c518;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0 1rem 0;
        font-weight: 600;
      }

      /* Answer card */
      .ai-summary-card {
        background: #141414;
        border-left: 4px solid #e50914;
        padding: 1rem 1rem 1rem 1rem;
        border-radius: 12px;
        margin: 0.75rem 0 1.25rem 0;
      }
      .ai-summary-label {
        color: #e50914;
        font-weight: 800;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
        letter-spacing: 0.04em;
      }
      .ai-summary-text {
        color: #ffffff;
        font-style: italic;
        line-height: 1.4;
      }

      /* Movie cards */
      .movie-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 20px;
        box-sizing: border-box;
        margin-bottom: 1rem;
        border-bottom: 3px solid #e50914;
      }
      .movie-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
      }
      .pill-row {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 0.65rem;
      }
      .pill-gray {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: #2a2a2a;
        color: #bdbdbd;
        font-size: 0.78rem;
        white-space: nowrap;
      }
      .genre-vote-line {
        color: #bdbdbd;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
      }
      .stars {
        font-size: 1.0rem;
        margin: 0.2rem 0 0.6rem 0;
        letter-spacing: 0.02em;
      }
      .star.filled { color: #f5c518; }
      .star.empty { color: #2a2a2a; }
      .overview-snippet {
        color: #aaaaaa;
        font-style: italic;
        line-height: 1.35;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="cineseek-banner">
      <h1 class="cineseek-title">CINESEEK<span style="color:#e50914;">.</span></h1>
      <span class="cineseek-title-underline"></span>
      <div class="cineseek-subtitle">Discover movies through natural language</div>
      <div class="cineseek-divider"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Maintain existing logic / flow via session state
if "needs_clarification" not in st.session_state:
    st.session_state.needs_clarification = False
if "clarification_question" not in st.session_state:
    st.session_state.clarification_question = ""
if "plan" not in st.session_state:
    st.session_state.plan = None
if "refined_query" not in st.session_state:
    st.session_state.refined_query = ""


def _run_search(user_query: str) -> None:
    plan = plan_query(user_query)
    st.session_state.plan = plan

    if plan.get("needs_clarification") is True:
        st.session_state.needs_clarification = True
        st.session_state.clarification_question = plan.get("clarification_question", "")
    else:
        st.session_state.needs_clarification = False
        st.session_state.clarification_question = ""


search_col_left, search_col_mid, search_col_right = st.columns([1, 6, 1])

with search_col_mid:
    st.write("")  # spacer for the banner area
    query_row = st.columns([5, 1])
    with query_row[0]:
        query = st.text_input(
            "",
            key="query_input",
            label_visibility="collapsed",
            placeholder="Try: 'mind-bending sci-fi with a twist ending'...",
        )
    with query_row[1]:
        search_clicked = st.button("Search", disabled=not bool(query.strip()))

    if search_clicked:
        _run_search(query.strip())


if st.session_state.needs_clarification:
    st.markdown(
        f'<div class="clarification-box">{escape(st.session_state.clarification_question or "Please clarify your request.")}</div>',
        unsafe_allow_html=True,
    )

    refine_col_left, refine_col_mid, refine_col_right = st.columns([1, 6, 1])
    with refine_col_mid:
        refine_row = st.columns([5, 1])
        with refine_row[0]:
            refined = st.text_input(
                "",
                key="refined_input",
                label_visibility="collapsed",
                placeholder="Refine your query...",
            )
        with refine_row[1]:
            refine_clicked = st.button("Submit", disabled=not bool(refined.strip()))

        if refine_clicked:
            _run_search(refined.strip())


plan = st.session_state.plan

if plan and not plan.get("needs_clarification"):
    user_query = query.strip() if query and query.strip() else st.session_state.get("refined_input", "").strip()

    with st.spinner("Searching the cinematic universe..."):
        results = hybrid_search(plan)
        answer = synthesize_answer(user_query=user_query, plan=plan, results=results)

    reasoning = plan.get("reasoning", "")
    st.markdown(f"<div class='reason-pill'>{escape(str(reasoning))}</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ai-summary-card">
          <div class="ai-summary-label">AI Summary</div>
          <div class="ai-summary-text">{escape(str(answer))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_results = (results or [])[:5]
    grid = st.columns(2)

    for i, r in enumerate(top_results):
        col = grid[i % 2]
        title = r.get("title", "")
        year = r.get("year", "")
        genre = r.get("genre", "")
        vote_average = r.get("vote_average", "")
        overview = r.get("overview") or ""
        overview_snippet = overview[:180] + ("…" if len(overview) > 180 else "")

        try:
            score = float(vote_average)
        except Exception:
            score = 0.0
        score = max(0.0, min(10.0, score))
        filled = int(round(score))
        filled = max(0, min(10, filled))
        empty = 10 - filled

        star_html = "".join(['<span class="star filled">★</span>' for _ in range(filled)]) + "".join(
            ['<span class="star empty">☆</span>' for _ in range(empty)]
        )

        with col:
            st.markdown(
                f"""
                <div class="movie-card">
                  <div class="movie-title">{escape(str(title))}</div>
                  <div class="pill-row">
                    <span class="pill-gray">{escape(str(year))}</span>
                    <span class="pill-gray">{escape(str(genre))}</span>
                  </div>
                  <div class="genre-vote-line">{escape(str(genre))} • {escape(str(vote_average))}</div>
                  <div class="stars">{star_html}</div>
                  <div class="overview-snippet">{escape(str(overview_snippet))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
