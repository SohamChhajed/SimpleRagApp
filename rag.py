from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from config import get_db_url, get_gemini_api_key

COLLECTION_NAME = "sql_docs"

def build_prompt(context: str, question: str) -> str:
    return f"""
You are a helpful assistant.
Use ONLY the context below to answer the question.

Write a detailed answer.

If the answer is not present in the context, say: "I do not know the answer based on the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

def get_rag_components(k: int = 4):
    db_url = get_db_url()
    gemini_key = get_gemini_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=gemini_key
    )
    vectorstore = PGVector(
        connection=db_url,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=gemini_key,
        temperature=0.2
    )
    return retriever, llm

def answer_question(question: str, k: int = 4):
    retriever, llm = get_rag_components(k=k)
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant context found.", []
    context = "\n\n".join([d.page_content for d in docs])
    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)
    sources = [(d.metadata.get("source", "unknown"), d.metadata.get("page", "?")) for d in docs]
    return response.content, sources

