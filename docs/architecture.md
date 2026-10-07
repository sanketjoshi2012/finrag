# FinRAG — Architecture Decision Record

## Problem Statement

Financial institutions deal with thousands of pages of regulatory documents, annual reports, and policy guidelines. Finding answers to specific questions means manually searching through hundreds of pages. FinRAG is a RAG-powered Q&A system that lets users ask natural language questions and get accurate answers with source citations.

## Why RAG?

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Fine-tuning | Model "learns" the documents | Expensive, slow, stale when docs update | ❌ |
| Full doc in prompt | Simple | Token limits — can't fit 500 pages | ❌ |
| RAG | Cheap, docs update instantly, cites sources | Needs vector DB + chunking | ✅ |

## Architecture

```
User Question → Embed → Search ChromaDB → Build Prompt → Claude API → Answer + Sources
```

## Tech Stack

| Component | Selected | Alternatives | Reasoning |
|---|---|---|---|
| LLM | Claude (Anthropic) | GPT-4, Gemini, LLaMA | Strong at document analysis, instruction following |
| Orchestration | LangChain | LlamaIndex, Haystack | Most requested in job descriptions, broad skill set |
| Vector DB | ChromaDB | Pinecone, FAISS, Weaviate | Free, local, zero setup, good for demos |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | OpenAI, Cohere | Free, local, no API key needed ([details](decisions/001-embeddings-model.md)) |
| Frontend | Streamlit | Gradio, Flask+React | Clean UI in minimal code, no frontend skills needed |
| PDF Parsing | PyMuPDF | Unstructured.io, LlamaParse | Simple, free, handles most PDFs well |
| Deployment | Local → AWS EC2 | Heroku, GCP, Railway | AWS most requested in job descriptions |

## Chunking Strategy

- Method: Recursive character text splitter
- Chunk size: 1000 characters
- Overlap: 200 characters
- Why: Balances context preservation with retrieval precision. Too large = noise. Too small = lost context. 200 char overlap ensures sentences aren't cut mid-thought.
