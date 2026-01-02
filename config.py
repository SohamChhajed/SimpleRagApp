import os
from dotenv import load_dotenv
load_dotenv()

def get_db_url():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE ERROR")
    return db_url

def get_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY ERROR")
    return api_key
