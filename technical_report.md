# Technical Report — Finance Policy Assistant

## 1. System Overview

The Finance Policy Assistant is an AI-powered Retrieval-Augmented Generation (RAG) system that allows employees to ask natural language questions about internal finance policy documents and receive accurate, grounded answers with source citations.

The system is implemented as a single-file Streamlit web application (`app.py`) backed by a hybrid retrieval pipeline combining keyword search, semantic vector search, rank fusion, and neural reranking.

---

## 2. System Design Decisions

### 2.1 Hybrid Retrieval over Pure Semantic Search

A pure vector search approach was considered but rejected in favour of a hybrid BM25 + FAISS pipeline.

**Reason:** Finance policy documents contain specific terminology — policy names, monetary limits, percentage values, procedure names — where exact keyword matching is critical. Pure semantic search may return thematically similar chunks that miss the specific term being queried.

By combining BM25 (keyword) and FAISS (semantic), the system achieves both:
- Precision on exact policy terms (via BM25)
- Recall on paraphrased or conceptually similar queries (via FAISS)

### 2.2 Reciprocal Rank Fusion (RRF) for Merging Results

RRF was chosen over score-based fusion because BM25 and cosine similarity scores operate on different scales and cannot be directly compared or added. RRF uses only rank positions, making it scale-independent and robust across retrieval methods.

**Formula:** `score(d) = Σ 1 / (60 + rank(d, list))`

The constant `60` dampens the effect of high-ranked results dominating and gives mid-ranked results a fair contribution.

### 2.3 Cross-Encoder Reranking as Final Filter

After RRF fusion produces 20 candidates, a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores each (query, chunk) pair jointly. Unlike bi-encoders (used in FAISS), cross-encoders attend to both query and passage simultaneously, producing more accurate relevance scores.

This two-stage approach (fast retrieval → precise reranking) is standard in production RAG systems and significantly improves final answer quality over a single-stage retrieval.

### 2.4 Low Temperature for LLM (0.1)

GPT-4o-mini is configured with `temperature=0.1` to minimize randomness and creative embellishment. For policy Q&A, factual accuracy is paramount over variety of expression.

### 2.5 Streamlit with Index Caching

`@st.cache_resource` ensures the entire index (BM25, FAISS, cross-encoder, chunk list) is built only once per session. Without caching, every user interaction would re-embed all document chunks — an expensive and slow operation.

---

## 3. Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk Size | 500 characters | Large enough to contain a complete policy point; small enough to stay focused |
| Chunk Overlap | 100 characters | Preserves context across chunk boundaries for continuous policies |
| Splitter | RecursiveCharacterTextSplitter | Tries natural separators first (`\n\n`, `\n`, `. `) before hard-cutting |

**Why RecursiveCharacterTextSplitter?**

This splitter respects document structure by preferring to split at paragraph or sentence boundaries rather than mid-sentence. This is important for policy documents where a policy rule often spans multiple lines but belongs to one logical unit.

**Chunking Flow:**
```
PDF Page Text
    → Split at \n\n (paragraph breaks) first
    → If still too large, split at \n (line breaks)
    → If still too large, split at ". " (sentence boundaries)
    → Last resort: split at word or character boundary
```

Each chunk stores: `{id, source filename, page number, text}`

---

## 4. Retrieval Approach

### Stage 1: BM25 Keyword Search
- Tokenizes query and all chunks using regex `[A-Za-z0-9]+`
- Scores all chunks using BM25Okapi (term frequency + inverse document frequency)
- Returns top 20 chunks ranked by BM25 score

### Stage 2: FAISS Vector Search
- Embeds query using `text-embedding-3-small` (1536 dimensions)
- L2-normalizes both query and chunk vectors → inner product = cosine similarity
- FAISS IndexFlatIP performs exact cosine search
- Returns top 20 chunks ranked by cosine similarity

### Stage 3: RRF Fusion
- Merges BM25 and FAISS ranked lists using Reciprocal Rank Fusion
- Produces a unified ranked list of 20 unique chunks

### Stage 4: Cross-Encoder Reranking
- Scores each of the 20 fused (query, chunk) pairs using `ms-marco-MiniLM-L-6-v2`
- Selects top 5 chunks by cross-encoder score

### Stage 5: Answer Generation
- Passes top 5 chunks as context to GPT-4o-mini
- System prompt enforces grounded, cited, policy-accurate responses
- Temperature set to 0.1 for factual consistency

---

## 5. Scalability Considerations

| Concern | Current Approach | Scalable Extension |
|---------|-----------------|-------------------|
| Document volume | All PDFs loaded into memory | Replace FAISS with ChromaDB / Pinecone for persistent disk-based storage |
| Embedding re-computation | Re-embedded on every cold start | Pre-compute and persist embeddings to disk |
| Multiple users | Streamlit single-threaded | Deploy with Streamlit Cloud or behind a FastAPI server |
| New documents | Requires app restart to re-index | Implement incremental indexing with document change detection |
| Chunk count | Works up to ~50K chunks in memory | FAISS IVF index or approximate search for 100K+ chunks |

---

## 6. Limitations

1. **Static Index** — Adding new documents requires restarting the application to rebuild the index. No live document ingestion.

2. **English Only** — The system is not designed for multilingual documents. BM25 tokenization and the embedding model are optimized for English.

3. **Text-Only Retrieval** — Tables, charts, and images in PDFs are not processed. Policy information embedded in non-text formats may be missed.

4. **PDF Quality Dependent** — Scanned PDFs or PDFs with poor text extraction will produce low-quality chunks and retrieval failures.

5. **No Conversation Memory** — Each query is independent. The assistant does not maintain conversation context across multiple turns.

6. **Chunk Boundary Cuts** — Despite overlap, a policy rule that spans more than two chunks may be partially retrieved, potentially giving an incomplete answer.

7. **Cross-Encoder Speed** — Running the cross-encoder on 20 candidates adds latency (~1-2 seconds). For very large candidate sets this could slow response time.

---

## 7. Future Improvements

1. **Persistent Vector Store** — Use ChromaDB or Pinecone to persist embeddings across restarts and support incremental document addition.

2. **Conversation History** — Add multi-turn conversation support so employees can ask follow-up questions in context.

3. **Table Extraction** — Use a PDF parser like `pdfplumber` or `camelot` to extract structured table data (e.g., reimbursement limit tables) as separate indexed entries.

4. **Query Expansion** — Use LLM to rewrite or expand the query before retrieval to improve recall for vague or incomplete questions.

5. **Confidence Scoring** — Surface a confidence indicator when no highly-relevant chunk is found, so the system gracefully signals uncertainty rather than attempting a low-confidence answer.

6. **Feedback Loop** — Allow users to rate answers (thumbs up/down) and log low-rated answers for periodic review and retrieval improvement.

7. **Authentication** — Add enterprise SSO or role-based access to restrict sensitive policy documents to authorized departments.

8. **Monitoring Dashboard** — Track query patterns, no-answer rates, and retrieval latency to identify gaps in the document corpus.

---

## 8. Technology Stack Summary

| Component | Library / Model | Version |
|-----------|----------------|---------|
| PDF Loading | PyPDF | latest |
| Chunking | LangChain Text Splitters | latest |
| Keyword Search | rank-bm25 (BM25Okapi) | latest |
| Vector Search | FAISS (faiss-cpu) | latest |
| Embeddings | text-embedding-3-small | OpenAI API |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | sentence-transformers |
| LLM | gpt-4o-mini | OpenAI API |
| UI | Streamlit | latest |
| Environment | python-dotenv | latest |

---

*Report prepared as part of the Finance Policy Assistant assignment submission.*
