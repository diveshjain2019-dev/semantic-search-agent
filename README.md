# CINESEEK — Semantic Movie Search Agent

An AI-powered search engine that understands what you mean, not just what you type. Built on a four-layer architecture spanning metadata enrichment, semantic vectors, agentic query planning, and a knowledge graph.

**Live demo:** [movieagent.streamlit.app](https://movieagent.streamlit.app)

---

## What it does

Traditional search matches keywords. CINESEEK matches meaning.

You can search for something like *"a dark film where nobody can be trusted"* or *"something lighthearted for a Friday night"* and get genuinely relevant results — even if those exact words don't appear anywhere in the dataset.

---

## Architecture — four layers

### 1. Metadata layer
The raw TMDB dataset only provides basic fields like title, genre, year, and rating. An LLM reads every movie's plot description and infers structured attributes that were never explicitly in the data — tone (dark, lighthearted, tense), themes (identity, revenge, survival), and pacing (slow-burn, moderate, fast-paced). This enrichment is what makes everything downstream possible.

### 2. Semantic layer
Every enriched movie record is converted into a 384-dimensional meaning vector using `sentence-transformers` and stored in ChromaDB. Similar meanings cluster together in vector space, so concept-based queries find the right results even when the exact words don't match.

### 3. Context layer
When a query arrives it doesn't go straight to the database. A Groq LLaMA-3 model acts as a query planner — deconstructing your intent into structured filters (year, genre, rating), a rewritten semantic query, and keywords for exact matching. If your query is too vague, it asks a clarifying question before searching. That back-and-forth is what makes it feel like a conversation rather than a search box.

### 4. Knowledge graph
Movies, genres, tones, themes, and pacing exist as connected nodes and edges built with NetworkX. At retrieval time three strategies run in parallel — ChromaDB vector search, BM25 keyword search, and metadata filtering — and are fused using Reciprocal Rank Fusion (RRF) into a single ranked result. Groq LLaMA-3 then synthesizes a natural language explanation of why those results matched your query.

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web UI and cloud deployment |
| Pandas | Data loading and cleaning |
| ChromaDB | Vector storage and filtered search |
| sentence-transformers (`all-MiniLM-L6-v2`) | Local text embeddings |
| Groq LLaMA-3.3-70b | Query planning and answer synthesis |
| rank_bm25 | Keyword retrieval |
| NetworkX | Knowledge graph construction |
| pyvis | Interactive graph visualisation |
| Streamlit Community Cloud | Free public hosting |

---

## Project structure

```
semantic-search-agent/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── planner.py          # Groq query planner + clarification logic
│   ├── search.py           # Hybrid search + RRF fusion + answer synthesis
│   ├── embeddings.py       # Embed records and upsert into ChromaDB
│   ├── ingest.py           # Load TMDB CSV + LLM metadata enrichment
│   ├── graph_builder.py    # Build NetworkX knowledge graph
│   └── graph_viz.py        # Export interactive graph as HTML
├── data/
│   ├── enriched.json       # LLM-enriched movie records
│   └── chroma_db/          # Persisted vector embeddings
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── requirements.txt
├── packages.txt
├── project.md
└── .gitignore
```

---

## Getting started locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/semantic-search-agent.git
cd semantic-search-agent
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

---

## Example queries to try

| Query | What it tests |
|---|---|
| `top rated action movies from the 1990s` | Metadata filters (year + genre) |
| `films with a dark twisted ending you don't see coming` | Pure semantic search |
| `movies about a man seeking revenge for his family` | Hybrid search |
| `something fun for the weekend` | Clarification flow |
| `a film like Blade Runner but more hopeful` | Semantic similarity |

---

## How the search pipeline works

```
User query
    ↓
Groq LLaMA-3 query planner
    ↓ (if vague → asks clarifying question)
Structured search plan
    ↓
┌──────────────────────────────────────────┐
│  Vector search  │  BM25  │  Metadata filter  │
└──────────────────────────────────────────┘
    ↓
Reciprocal Rank Fusion
    ↓
Ranked results + Groq-synthesised explanation
```

---

## Dataset

Built on the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle. The dataset covers older films and does not include recent releases. The focus of this project is the search architecture, not data freshness.

---

## Inspiration

This project was inspired by conversations at the Gartner conference around the future of enterprise search — the idea that search should understand intent rather than match keywords. Movies were used as the playground, but the same four-layer architecture applies directly to enterprise use cases like internal knowledge bases, document retrieval, product catalogs, and customer support systems.

---

## Enterprise applicability

The architecture demonstrated here maps directly to real enterprise search problems:

- **Metadata enrichment** → enrich support tickets, contracts, or product listings with LLM-inferred tags
- **Semantic layer** → enable concept-based search over internal documents regardless of terminology
- **Context layer** → agentic query planning for enterprise knowledge bases
- **Knowledge graph** → connect entities across departments, projects, or product lines

---

## Built with

Developed using [Cursor](https://cursor.sh) as an AI coding assistant alongside Claude. Deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## License

MIT
