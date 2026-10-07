from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from config import ANTHROPIC_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """You are a financial document analyst. Answer questions using ONLY the provided context.
Rules:
- If the context doesn't contain the answer, say "I don't have enough information to answer this."
- Cite your sources using [Source: filename, Page X] format.
- Be concise and precise — this is financial data, accuracy matters."""

HUMAN_PROMPT = """Context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=1024,
    )


def format_context(documents: list[Document]) -> str:
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate(question: str, documents: list[Document]) -> str:
    llm = get_llm()
    context = format_context(documents)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
