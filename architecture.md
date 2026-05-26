# Finance Policy Assistant — Architecture Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT INGESTION                           │
│                                                                     │
│   docs/                                                             │
│   ├── Financial Policy.pdf                                          │
│   ├── HR-Policy.pdf          ──► PyPDF (PdfReader)                  │
│   └── jp-morgan-code-of-conduct.pdf                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Raw page text (per page)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CHUNKING                                   │
│                                                                     │
│   RecursiveCharacterTextSplitter                                    │
│   ├── chunk_size    = 500 characters                                │
│   ├── chunk_overlap = 100 characters                                │
│   └── separators   = ["\n\n", "\n", ". ", " ", ""]                 │
│                                                                     │
│   Output: List of chunks  {id, source, page, text}                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Chunks
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌───────────────────────┐       ┌─────────────────────────────────────┐
│    BM25 INDEX         │       │         FAISS INDEX                 │
│                       │       │                                     │
│  BM25Okapi            │       │  text-embedding-3-small             │
│  (keyword frequency)  │       │  (OpenAI Embeddings API)            │
│                       │       │  → 1536-dim float32 vectors         │
│  Tokenized corpus     │       │  → L2 normalized (cosine sim)       │
│  stored in memory     │       │  → IndexFlatIP (inner product)      │
└───────────────────────┘       └─────────────────────────────────────┘
                │  (built once, cached via @st.cache_resource)        │
                └──────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         QUERY TIME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Employee Question (natural language)
                               │
                               ▼
              ┌────────────────────────────────┐
              │         STREAMLIT UI           │
              │   Text input / Demo buttons    │
              └────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌───────────────────────┐       ┌─────────────────────────────────────┐
│   BM25 SEARCH         │       │        VECTOR SEARCH                │
│                       │       │                                     │
│  Tokenize query       │       │  Embed query via                    │
│  BM25Okapi.get_scores │       │  text-embedding-3-small             │
│  Top 20 chunks        │       │  FAISS.search(query_vec, k=20)      │
│  ranked by BM25 score │       │  Top 20 chunks by cosine similarity │
└──────────┬────────────┘       └──────────────┬──────────────────────┘
           │   BM25 hits (20)                  │  Vector hits (20)
           └──────────────┬────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RRF FUSION                                        │
│                                                                     │
│  Reciprocal Rank Fusion formula:                                    │
│  score(d) = Σ  1 / (60 + rank(d, list))                            │
│                                                                     │
│  Merges BM25 + Vector ranked lists without score normalization      │
│  Output: Top 20 unique chunks ranked by combined RRF score          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  20 fused candidates
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   CROSS-ENCODER RERANKING                           │
│                                                                     │
│  Model: cross-encoder/ms-marco-MiniLM-L-6-v2                        │
│  Input: (query, chunk_text) pairs — all 20 candidates               │
│  Output: Relevance score per pair                                   │
│  Picks: Top 5 chunks by cross-encoder score                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  5 best chunks with source + page
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LLM ANSWER GENERATION                             │
│                                                                     │
│  Model: GPT-4o-mini                                                 │
│  System Prompt: Finance Policy Assistant rules                      │
│  Context: 5 retrieved chunks with [source, page] citations          │
│  Temperature: 0.1 (low, for factual accuracy)                       │
│                                                                     │
│  Rules enforced:                                                    │
│  - Answer ONLY from provided context                                │
│  - Cite source [filename, page X] after each point                  │
│  - Say "I don't know" if answer not in context                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │         STREAMLIT UI           │
              │  ┌─────────────────────────┐  │
              │  │  Answer (st.info)        │  │
              │  │  with source citations  │  │
              │  └─────────────────────────┘  │
              │  ┌─────────────────────────┐  │
              │  │  Retrieved Chunks       │  │
              │  │  (expandable)           │  │
              │  │  source | page | score  │  │
              │  └─────────────────────────┘  │
              └────────────────────────────────┘
```

---

## Component Summary

| Component | Technology | Role |
|-----------|-----------|------|
| PDF Loading | PyPDF (PdfReader) | Extracts text from policy documents page by page |
| Chunking | LangChain RecursiveCharacterTextSplitter | Splits pages into 500-char overlapping chunks |
| Keyword Index | BM25Okapi (rank_bm25) | Exact keyword matching and term frequency scoring |
| Vector Index | FAISS IndexFlatIP | Semantic similarity search using cosine distance |
| Embeddings | text-embedding-3-small (OpenAI) | 1536-dim dense vectors for semantic representation |
| Fusion | Reciprocal Rank Fusion (RRF) | Merges BM25 and vector results without score normalization |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 | Fine-grained relevance scoring of fused candidates |
| LLM | GPT-4o-mini (OpenAI) | Grounded answer generation from retrieved context |
| UI | Streamlit | Web interface with sidebar, demo queries, source traceability |
| Caching | @st.cache_resource | Builds indexes once, reuses across all queries |
