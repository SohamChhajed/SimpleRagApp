from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from config import get_db_url, get_gemini_api_key

COLLECTION_NAME = "sql_docs"

def build_prompt(context: str, question: str) -> str:
    return f"""
You are a helpful AI assistant.

You must answer using ONLY the information provided in the context below.
Give a detailed answer.

If the user asks for jokes, poems, comedy, stories, creative writing, or entertainment of any kind, respond exactly with:
"I can’t generate jokes or poems, but I can help explain concepts, provide summaries, or answer factual questions."

If the question is opinion-based, hypothetical, or cannot be answered factually using the given context, respond exactly with:
"I do not know the answer based on the provided document."

If the context is empty, irrelevant, or does not contain enough information to answer the question, respond exactly with:
"I do not know the answer based on the provided document."

Keep the answer factual, concise and neutral in tone.

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

