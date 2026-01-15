from datetime import date
import psycopg2
from urllib.parse import urlparse
from config import get_db_url
from datetime import datetime
def get_psycopg2_conn():
        db_url = get_db_url()

        parsed = urlparse(db_url)

        return psycopg2.connect(
            dbname=parsed.path.lstrip("/"),
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port,
        )
def increment_thumbs_down():
    today = date.today()
    

    conn = get_psycopg2_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO rag_feedback_daily (date, thumbs_down_count)
        VALUES (%s, 1)
        ON CONFLICT (date)
        DO UPDATE SET thumbs_down_count =
            rag_feedback_daily.thumbs_down_count + 1;
    """, (today,))

    conn.commit()
    cur.close()
    conn.close()

def get_today_thumbs_down() -> int:
    today = date.today()

    conn = get_psycopg2_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT thumbs_down_count
        FROM rag_feedback_daily
        WHERE date = %s
    """, (today,))

    row = cur.fetchone()
    cur.close()
    conn.close()

    return row[0] if row else 0

def store_feedback_example(
    trace_id: str,
    question: str,
    context: list[str],
    model_answer: str,
    score: int | None = None,
    reason: str | None = None,
    comment: str | None = None,
):
    conn = get_psycopg2_conn()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT id FROM rag_feedback_examples 
            WHERE trace_id = %s
        """, (trace_id,))
        
        existing = cur.fetchone()
        if existing:
            cur.execute("""
                UPDATE rag_feedback_examples
                SET score = %s, 
                    reason = %s, 
                    comment = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE trace_id = %s
            """, (score, reason, comment, trace_id))
        else:
            cur.execute("""
                INSERT INTO rag_feedback_examples
                (trace_id, question, context, model_answer, score, reason, comment)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                trace_id,
                question,
                "\n\n".join(context),
                model_answer,
                score,
                reason,
                comment,
            ))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def get_last_gepa_run_time():
    conn = get_psycopg2_conn()
    cur = conn.cursor()
    cur.execute("SELECT MAX(created_at) FROM gepa_runs")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0]

def record_gepa_run(last_feedback_at):
    conn = get_psycopg2_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO gepa_runs (last_feedback_at) VALUES (%s)",
        (last_feedback_at,)
    )
    conn.commit()
    cur.close()
    conn.close()

from datetime import datetime

def count_feedback_since(timestamp):
    conn = get_psycopg2_conn()
    cur = conn.cursor()

    if timestamp:
        cur.execute("""
            SELECT COUNT(*)
            FROM rag_feedback_examples
            WHERE created_at > %s
        """, (timestamp,))
    else:
        cur.execute("""
            SELECT COUNT(*)
            FROM rag_feedback_examples
        """)

    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


