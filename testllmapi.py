import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

resp = llm.invoke("How are you?")
print(resp)
print(resp.content)
