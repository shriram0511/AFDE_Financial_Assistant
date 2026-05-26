import os
import re
import glob
import numpy as np
import faiss
import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

DOCS_FOLDER = "docs"
EMBED_MODEL  = "text-embedding-3-small"
EMBED_DIM    = 1536
LLM_MODEL    = "gpt-4o-mini"
CANDIDATES   = 20
FINAL_K      = 5

client = OpenAI()

SYSTEM_PROMPT = """You are a Finance Policy Assistant for an enterprise organization.

Rules:
1. Answer ONLY using the provided document context below.
2. If the answer is not in the context, say exactly:
   "I don't know - this information is not available in the provided finance documents."
3. Always cite your source: [filename, page X] after each point.
4. Be concise, professional, and policy-accurate.
5. Never fabricate policy details or numbers."""

# ─────────────────────────────────────────
# PIPELINE FUNCTIONS (cached)
# ─────────────────────────────────────────

def tokenize(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())

@st.cache_resource(show_spinner="Loading and indexing documents...")
def build_indexes():
    # Load all PDFs
    all_pages = []
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    for pdf_path in pdf_files:
        source = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                all_pages.append({"source": source, "page": i, "text": text})

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for p in all_pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append({"id": len(chunks), "source": p["source"],
                           "page": p["page"], "text": piece})

    # BM25
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    # FAISS
    texts = [c["text"] for c in chunks]
    embeddings = []
    for i in range(0, len(texts), 64):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+64])
        embeddings.extend([d.embedding for d in resp.data])
    arr = np.array(embeddings, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(arr)

    # Cross-encoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    sources = list({c["source"] for c in chunks})
    return chunks, bm25, index, cross_encoder, sources

def embed_query(query):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    arr  = np.array([resp.data[0].embedding], dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr

def search(query, chunks, bm25, faiss_index, cross_encoder):
    # BM25
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:CANDIDATES]
    bm25_hits = [{"rank": r+1, "doc_id": i, "score": float(s),
                  "text": chunks[i]["text"], "page": chunks[i]["page"],
                  "source": chunks[i]["source"]}
                 for r, (i, s) in enumerate(ranked)]

    # Vector
    q    = embed_query(query)
    D, I = faiss_index.search(q, CANDIDATES)
    vec_hits = [{"rank": r+1, "doc_id": int(i), "score": float(s),
                 "text": chunks[int(i)]["text"], "page": chunks[int(i)]["page"],
                 "source": chunks[int(i)]["source"]}
                for r, (i, s) in enumerate(zip(I[0], D[0])) if i != -1]

    # RRF fusion
    rrf_scores = {}
    items = {}
    for ranked_list in [bm25_hits, vec_hits]:
        for item in ranked_list:
            did = item["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60 + item["rank"])
            items[did] = item
    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:CANDIDATES]
    fused = [{"rank": r+1, "doc_id": did, "rrf_score": rrf_scores[did],
              "text": items[did]["text"], "page": items[did]["page"],
              "source": items[did]["source"]}
             for r, did in enumerate(top_ids)]

    # Rerank
    pairs  = [(query, c["text"]) for c in fused]
    scores = cross_encoder.predict(pairs)
    scored = [{**c, "ce_score": float(s)} for c, s in zip(fused, scores)]
    reranked = sorted(scored, key=lambda x: x["ce_score"], reverse=True)[:FINAL_K]

    return reranked

def generate_answer(query, retrieved):
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}, Page {r['page']}]\n{r['text']}"
        for r in retrieved
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nEmployee Question: {query}"},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content

# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Finance Policy Assistant",
    page_icon="📋",
    layout="wide"
)

st.title("Finance Policy Assistant")
st.caption("Ask any question about Nexora's internal finance policies. Answers are grounded in official documents.")

# Load indexes
chunks, bm25, faiss_index, cross_encoder, sources = build_indexes()

# Sidebar
with st.sidebar:
    st.header("Loaded Documents")
    for s in sorted(sources):
        st.success(f"📄 {s}")
    st.divider()
    st.metric("Total Chunks Indexed", len(chunks))
    st.metric("Retrieval Candidates", CANDIDATES)
    st.metric("Final Chunks to LLM", FINAL_K)
    st.divider()
    show_chunks = st.toggle("Show retrieved chunks", value=True)

# Demo queries
st.subheader("Try a sample query")
demo_queries = [
    "What is the reimbursement limit for client meals?",
    "What approvals are needed for vendor payments above 10 lakhs?",
    "What documents are required for procurement?",
    "What are the international travel entitlements?",
    "What is the TDS rate for professional fees?",
    "Can employees get a salary advance?",
    "What happens if I submit expenses after 30 days?",
    "What is the petty cash limit per department?",
]
cols = st.columns(4)
selected_demo = None
for i, q in enumerate(demo_queries):
    if cols[i % 4].button(q, use_container_width=True):
        selected_demo = q

st.divider()

# Query input
query = st.text_input(
    "Your Question",
    value=selected_demo or "",
    placeholder="e.g. What is the reimbursement limit for client travel?",
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Searching documents and generating answer..."):
        reranked = search(query, chunks, bm25, faiss_index, cross_encoder)
        answer   = generate_answer(query, reranked)

    st.subheader("Answer")
    st.info(answer)

    if show_chunks:
        st.subheader("Retrieved Source Chunks")
        for i, chunk in enumerate(reranked, 1):
            with st.expander(f"Chunk {i} — {chunk['source']} | Page {chunk['page']} | CE Score: {chunk['ce_score']:.4f}"):
                st.write(chunk["text"])
