from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from config import get_db_url,get_gemini_api_key
col_name="sql_docs"

def build_prompt(context:str,question:str)->str:
    return f"""
You are a helpful assistant.Answer ONLY using the context below.
If the answer is not present in the context, say: "I do not know the answer".

CONTEXT:
{context}
QUESTION:
{question}
ANSWER:
""".strip()

def main():
    db_url=get_db_url()
    gemini_key=get_gemini_api_key()
    embeddings=GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=gemini_key
    )
    vectorstore=PGVector(
        connection=db_url,
        embeddings=embeddings,
        collection_name=col_name,
        use_jsonb=True
    )
    retrieve=vectorstore.as_retriever(search_kwargs={"k":4})
    llm=ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=gemini_key,
        temperature=0.2
    )
    print("Enter your question. Type 'exit' to exit.")
    while True:
        question=input().strip()
        if question.lower()=="exit":
            break

        try:
            docs=retrieve.get_relevant_documents(question)
        except Exception:
            docs=retrieve.invoke(question)
        
        if not docs:
            print("\nNo context found.")
            continue

        context="\n\n".join([d.page_content for d in docs])
        prompt=build_prompt(context=context,question=question)
        response=llm.invoke(prompt)
        print("\n")
        print(response.content)
        print("\n")
        for d in docs:
            src=d.metadata.get("source","unknown")
            page=d.metadata.get("page","?")
            print(f"-{src}(page {page})")

if __name__=="__main__":
    main()

