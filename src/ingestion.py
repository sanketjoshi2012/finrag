import os
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")


def load_pdfs(directory: str = DATA_DIR) -> list[Document]:
    documents = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(directory, filename)
        doc = pymupdf.open(filepath)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_num},
                ))
        doc.close()
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks: list[Document]) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vector_store


def ingest(directory: str = DATA_DIR) -> Chroma:
    print(f"Loading PDFs from {directory}...")
    documents = load_pdfs(directory)
    print(f"Loaded {len(documents)} pages from {len(set(d.metadata['source'] for d in documents))} PDF(s).")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding and storing in ChromaDB...")
    vector_store = create_vector_store(chunks)
    print("Done. Vector store ready.")
    return vector_store


if __name__ == "__main__":
    ingest()
