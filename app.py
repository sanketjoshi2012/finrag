import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from chain import ask
from ingestion import ingest
from retrieval import get_vector_store

st.set_page_config(page_title="FinRAG", page_icon="📊", layout="wide")
st.title("📊 FinRAG — Financial Document Q&A")

st.markdown("""
**FinRAG** is a Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions
about financial documents and get accurate, cited answers. Instead of reading through hundreds of pages,
just ask a question — FinRAG finds the relevant sections and generates a precise answer with source references.

Built with LangChain, ChromaDB, HuggingFace embeddings, and Claude.
""")

with st.expander("📂 About the dataset", expanded=False):
    st.markdown("""
**Domain:** Banking & Financial Services

**Current documents:**
- **JP Morgan Chase & Co. — 2024 Annual Report** (372 pages)
  - Covers: financial performance, revenue breakdown, risk management, regulatory compliance,
    technology strategy, business segment results (CCB, CIB, AWM, CB), capital adequacy, and outlook.

**Why this dataset?**
JP Morgan is the largest bank in the US (~$4T in assets). Their annual report is publicly available
and covers key financial topics — revenue, net interest income, risk factors, Basel III compliance,
AI/technology investments, and more. It's a real-world document that financial analysts actually read.

**Want to add your own documents?**
Drop any PDF into `data/documents/` and click **Ingest Documents** in the sidebar.
The system works with any financial document — earnings reports, SEC filings, Basel III guidelines, etc.
""")

SAMPLE_QUESTIONS = [
    "What was JP Morgan's total net revenue in 2024?",
    "How did net interest income change vs prior year?",
    "What are the key risk factors mentioned?",
    "What is JP Morgan's AI and technology strategy?",
    "What was the CET1 capital ratio and how does it compare to regulatory minimums?",
]

with st.sidebar:
    st.header("Document Management")

    if st.button("Ingest Documents", type="primary"):
        with st.spinner("Loading, chunking, embedding..."):
            store = ingest()
            st.success("Ingested into ChromaDB.")

    try:
        store = get_vector_store()
        count = store._collection.count()
        st.metric("Chunks in DB", count)
    except Exception:
        st.info("No documents ingested yet. Add PDFs to `data/documents/` and click Ingest.")

    st.divider()
    st.header("Sample Questions")
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Drop PDFs in `data/documents/`\n2. Click **Ingest Documents**\n3. Ask questions below or click a sample")

    st.divider()
    st.markdown("**Tech stack:**")
    st.markdown("- **LLM:** Claude (Anthropic)\n- **Embeddings:** HuggingFace all-MiniLM-L6-v2\n- **Vector DB:** ChromaDB\n- **Orchestration:** LangChain\n- **Frontend:** Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- **{s['file']}**, Page {s['page']}")

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask a question about your financial documents...")
question = pending or question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = ask(question)

        st.write(result["answer"])

        with st.expander("Sources"):
            for s in result["sources"]:
                st.write(f"- **{s['file']}**, Page {s['page']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
