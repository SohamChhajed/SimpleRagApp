Simple RAG Application

This is a basic Retrieval Augmented Generation application built as part of my AI and ML internship work.

The application allows users to ask questions from a PDF document and generates answers based on relevant content from the document.

The PDF is split into smaller text chunks and stored as vector embeddings in a PostgreSQL database using pgvector.

When a question is asked, similar document chunks are retrieved using vector similarity search and provided as context to the language model.

A simple Streamlit interface is used to interact with the application.

Tech Stack Used

-Python

-LangChain

-PostgreSQL with pgvector

-Google Gemini

-Streamlit

How to Run

1.Install the required Python dependencies

2.Set the database URL and Gemini API key in a .env file

3.Run loading.py to ingest the document

4.Run streamlit run app.py to start the application
