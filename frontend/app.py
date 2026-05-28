"""
RAG Document Q&A — Streamlit Frontend
A clean, professional UI for uploading documents and asking questions.
"""

import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.3) 0%, transparent 70%);
    top: -80px; right: -80px;
    border-radius: 50%;
}
.hero h1 { color: white; font-size: 2.2rem; margin-bottom: 0.3rem; }
.hero p  { color: #94a3b8; font-size: 1rem; margin: 0; }

/* Source cards */
.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.source-card .meta {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.4rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.source-card .excerpt { font-size: 0.88rem; color: #334155; line-height: 1.6; }
.score-badge {
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 8px;
}

/* Answer box */
.answer-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #14532d;
}

/* Stat pills */
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
.stat-pill {
    background: #1e293b;
    color: #e2e8f0;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"Error: {detail}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("*RAG-powered Document Q&A*")
    st.divider()

    # Health check
    health = api_get("/health")
    if health:
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">📄 {health.get('documents_indexed', 0)} docs</div>
            <div class="stat-pill">🔢 {health.get('chunks_indexed', 0)} chunks</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("Backend connected ✅")
    else:
        st.error("Backend offline ❌")

    st.divider()

    # Upload
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files and st.button("Index Documents", type="primary", use_container_width=True):
        for f in uploaded_files:
            with st.spinner(f"Indexing {f.name}…"):
                res = api_post("/upload", files={"file": (f.name, f.getvalue(), f.type)})
                if res:
                    st.success(f"✅ {f.name} → {res['chunks_created']} chunks")

    st.divider()

    # List indexed docs
    docs_resp = api_get("/documents")
    if docs_resp and docs_resp.get("documents"):
        st.markdown("### Indexed Documents")
        for doc in docs_resp["documents"]:
            st.markdown(f"- 📄 **{doc['filename']}** ({doc['chunks']} chunks)")

    st.divider()
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        r = requests.delete(f"{API_BASE}/documents", timeout=10)
        if r.ok:
            st.success("Index cleared!")
            st.rerun()

    st.divider()
    st.markdown("""
    **How it works**
    1. Upload your documents
    2. They're split into chunks & embedded with `all-MiniLM-L6-v2`
    3. FAISS indexes the vectors
    4. Your question is embedded → top-k chunks retrieved
    5. Claude generates an answer from the context
    """)


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 DocMind</h1>
    <p>Ask anything about your documents — powered by RAG + Claude AI</p>
</div>
""", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f'<div class="answer-box">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s) used"):
                    for i, src in enumerate(msg["sources"], 1):
                        score_pct = int(src['relevance_score'] * 100)
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="meta">Source {i}: {src['source']} · Chunk #{src['chunk_id']}
                                <span class="score-badge">Relevance {score_pct}%</span>
                            </div>
                            <div class="excerpt">{src['excerpt']}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# Input
if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer…"):
            result = api_post("/query", json={"question": question, "top_k": 4})

        if result:
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])

            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            if sources:
                with st.expander(f"📎 {len(sources)} source(s) used"):
                    for i, src in enumerate(sources, 1):
                        score_pct = int(src['relevance_score'] * 100)
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="meta">Source {i}: {src['source']} · Chunk #{src['chunk_id']}
                                <span class="score-badge">Relevance {score_pct}%</span>
                            </div>
                            <div class="excerpt">{src['excerpt']}</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        else:
            st.error("Failed to get an answer. Is the backend running?")
