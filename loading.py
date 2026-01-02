from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from config import get_db_url,get_gemini_api_key

data_dir=Path("data/uploads")
pdf_name="sql.pdf"
col_name="sql_docs"

def main():
    db_url=get_db_url()
    gemini_key=get_gemini_api_key()
    pdf_path=data_dir/pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError("pdf not there")
    
    loader=PyPDFLoader(str(pdf_path))
    pages=loader.load()
    print(len(pages))

    for p in pages:
        p.metadata["source"]=pdf_name
    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    chunks=splitter.split_documents(pages)
    print(len(chunks))
    embeddings=GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=gemini_key
    )
    vectorstore=PGVector(
        connection=db_url,
        embeddings=embeddings,
        collection_name=col_name,
        use_jsonb=True,
    )
    vectorstore.add_documents(chunks)
if __name__=="__main__":
    main()