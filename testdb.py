import os
from dotenv import load_dotenv
import psycopg2
load_dotenv()
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("ERROR")

db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")


conn = psycopg2.connect(db_url)
cur = conn.cursor()

cur.execute("SELECT 1;")
result = cur.fetchone()

print(" Database connected successfully")

cur.close()
conn.close()
