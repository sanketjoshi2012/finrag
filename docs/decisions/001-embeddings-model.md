# Decision 001: Embedding Model Selection

## Context
We need an embedding model to convert text chunks and user queries into vectors for similarity search.

## Options Considered

| Option | Cost | Quality | Setup |
|---|---|---|---|
| OpenAI text-embedding-3-small | ~$0.02/1M tokens | Excellent | Needs API key + internet |
| Cohere embed-v3 | Free tier available | Very good | Needs API key + internet |
| HuggingFace all-MiniLM-L6-v2 | Free | Good | Runs locally, no key needed |
| HuggingFace all-mpnet-base-v2 | Free | Better than MiniLM | Larger model, slower |

## Decision
**HuggingFace all-MiniLM-L6-v2**

## Reasoning
- Free and runs locally — no API dependency, no cost
- Fast and lightweight (80MB model) — good for demos and development
- Well-established in the industry — widely used in production RAG systems
- No API key management — reduces setup friction for anyone cloning the repo
- Demonstrates ability to run models locally, not just call APIs

## Trade-off
Slightly lower embedding quality than OpenAI, but for a document Q&A system with well-structured financial documents, the difference is negligible. If quality becomes an issue, upgrading to OpenAI embeddings is a one-line config change.
