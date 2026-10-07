from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

TOP_K = 4


def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve(query: str, k: int = TOP_K) -> list[Document]:
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)
