import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CHROMA_PERSIST_DIR = ".chroma"
COLLECTION_NAME = "financial_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

LLM_MODEL = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
