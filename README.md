# FinRAG

A RAG-powered Q&A system for financial documents. Upload annual reports, SEC filings, or any financial PDF and ask questions in plain English. The system finds the relevant sections and gives you an answer with page-level source citations.

## What it does

1. You drop a PDF into the `data/documents/` folder
2. The system extracts text, splits it into chunks, and stores vector embeddings in ChromaDB
3. When you ask a question, it searches for the most relevant chunks using semantic similarity
4. Those chunks get sent to Claude along with your question
5. You get a concise answer with exact source citations (filename + page number)

Currently loaded with JP Morgan's 2024 Annual Report (372 pages, ~1900 chunks).

## Tech stack

| Component | Tool | Why |
|---|---|---|
| LLM | Claude (Anthropic) | Strong at document analysis and following instructions |
| Orchestration | LangChain | Connects the pipeline pieces together |
| Vector DB | ChromaDB | Free, runs locally, no setup needed |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Runs locally, no API key required |
| PDF parsing | PyMuPDF | Fast, handles most financial PDFs well |
| Frontend | Streamlit | Clean chat UI with minimal code |

## Setup

```bash
git clone https://github.com/sanketjoshi2012/finrag.git
cd finrag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your Anthropic API key to .env
```

## Running

```bash
source venv/bin/activate
streamlit run app.py
```

Then:
1. Open http://localhost:8501
2. Click "Ingest Documents" in the sidebar to process the PDFs
3. Ask a question or click one of the sample questions

## Project structure

```
finrag/
├── app.py                  # Streamlit frontend
├── requirements.txt
├── .env.example            # API key template
├── data/
│   └── documents/          # Drop PDFs here
├── src/
│   ├── config.py           # Settings (models, chunk size, API keys)
│   ├── ingestion.py        # PDF loading, chunking, embedding
│   ├── retrieval.py        # Vector similarity search
│   ├── generation.py       # Claude API call with prompt template
│   └── chain.py            # Ties retrieval + generation together
└── docs/
    ├── architecture.md     # Architecture decisions and reasoning
    └── decisions/          # Individual decision records
```

## How RAG works

```
Question --> Embed --> Search ChromaDB --> Top 4 chunks --> Claude --> Answer + Sources
```

The key idea: instead of sending an entire 372-page document to the LLM (which would exceed token limits and cost a fortune), we split it into small chunks, convert them to vectors, and only retrieve the few chunks that are actually relevant to the question. This keeps the context focused and the answers accurate.

## Architecture decisions

Detailed reasoning for every tech choice is documented in `docs/architecture.md`. The short version:

- **Why RAG over fine-tuning?** Cheaper, documents update instantly, no retraining needed
- **Why local embeddings?** Free, no API dependency, good enough for document search
- **Why ChromaDB?** Zero setup, runs locally, perfect for demos and prototyping
- **Why 1000 char chunks with 200 overlap?** Balances context preservation with search precision. The overlap prevents losing meaning at chunk boundaries.

## Sample questions

- What was JP Morgan's total net revenue in 2024?
- How did net interest income change compared to the prior year?
- What was the CET1 capital ratio and how does it compare to regulatory minimums?
- What are the key risk factors mentioned in the annual report?
- What is JP Morgan's strategy for AI and technology investment?
