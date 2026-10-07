from retrieval import retrieve
from generation import generate, format_context


def ask(question: str) -> dict:
    documents = retrieve(question)
    answer = generate(question, documents)
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        sources.append({"file": source, "page": page})
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context": format_context(documents),
    }
