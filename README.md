# Finance Policy Assistant

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that answers employee questions grounded in official finance policy documents.

---

## Folder Structure

```
Assignment/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # API keys (not committed)
├── docs/                   # Place your PDF policy documents here
│   ├── Financial Policy.pdf
│   ├── HR-Policy.pdf
│   └── jp-morgan-code-of-conduct.pdf
└── .streamlit/
    └── config.toml         # Disables file watcher for faster startup
```

---

## Setup

**1. Create and activate a virtual environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key to `.env`**
```
OPENAI_API_KEY=sk-...
```

**4. Place PDF documents in the `docs/` folder**

**5. Run the app**
```bash
streamlit run app.py
```

---

## How It Works

### Retrieval Pipeline

```
Query
  │
  ├── BM25 keyword search   → top 20 chunks
  ├── FAISS vector search   → top 20 chunks
  │
  ├── RRF Fusion            → merges both lists → 20 unique chunks
  │
  ├── Cross-Encoder Rerank  → scores all 20 → picks best 5
  │
  └── GPT-4o-mini           → generates grounded answer from 5 chunks
```

| Stage | What it does |
|-------|-------------|
| **BM25** | Keyword-based search using term frequency (good for exact terms) |
| **FAISS** | Semantic vector search using embeddings (good for meaning/context) |
| **RRF Fusion** | Combines BM25 + FAISS rankings using Reciprocal Rank Fusion formula |
| **Cross-Encoder** | Re-scores fused chunks using a deep neural model for precise relevance |
| **GPT-4o-mini** | Generates a concise, cited answer using only the retrieved context |

### Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `CANDIDATES` | 20 | Number of chunks fetched by each retriever and passed after RRF |
| `FINAL_K` | 5 | Number of top chunks sent to GPT-4o-mini after reranking |
| `EMBED_MODEL` | text-embedding-3-small | OpenAI model used to embed chunks and queries |
| `LLM_MODEL` | gpt-4o-mini | OpenAI model used to generate the final answer |
| `chunk_size` | 500 | Max characters per chunk when splitting PDFs |
| `chunk_overlap` | 100 | Overlap between consecutive chunks to preserve context |

---

## Models Used

| Model | Purpose |
|-------|---------|
| `text-embedding-3-small` | Converts text to vectors for semantic search |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranks retrieved chunks by relevance |
| `gpt-4o-mini` | Generates the final answer from context |

---

## UI Features

- **Sidebar** — Lists all loaded documents, total chunks indexed, retrieval stats
- **Demo Queries** — 8 clickable sample questions to try instantly
- **Answer Panel** — GPT-generated answer with source citations
- **Retrieved Chunks** — Toggle to view exact chunks used, with source file, page number, and cross-encoder score

---

## Notes

- The index is built once on first load and cached (`@st.cache_resource`). Restart the app if you add new PDFs.
- The `.streamlit/config.toml` sets `fileWatcherType = "none"` to avoid slow startup caused by Streamlit scanning the large `transformers` library.
- The assistant will only answer from document context — it will not fabricate policy details.
