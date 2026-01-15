import os
import dspy

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from config import get_db_url, get_gemini_api_key
from dspy_config import configure_lm
from tracing_config import setup_mlflow_tracing
from langfuse_config import setup_langfuse
from dspy_rag import RAGModule

from langfuse import observe, propagate_attributes
from feedback_store import get_today_thumbs_down
from feedback_store import increment_thumbs_down

configure_lm()
setup_mlflow_tracing()

langfuse_client = setup_langfuse()

rag_module = RAGModule()
THUMBS_DOWN_THRESHOLD = 4

thumbs_down_today = get_today_thumbs_down()

if thumbs_down_today >= THUMBS_DOWN_THRESHOLD and os.path.exists("optimized_rag_gepa.json"):
    rag_module.load("optimized_rag_gepa.json")
    CURRENT_OPTIMIZER = "GEPA"
    print("GEPA enabled due to feedback threshold")
else:
    if os.path.exists("optimized_rag.json"):
        rag_module.load("optimized_rag.json")
        CURRENT_OPTIMIZER = "Baseline"
        print("Using baseline RAG")
    else:
        CURRENT_OPTIMIZER = "Baseline"
        print("No optimized model found")


# if os.path.exists("optimized_rag_gepa.json"):
#     rag_module.load("optimized_rag_gepa.json")
#     CURRENT_OPTIMIZER = "GEPA"
#     print("Loaded GEPA optimized")

# elif os.path.exists("optimized_rag_simba.json"):
#     rag_module.load("optimized_rag_simba.json")
#     CURRENT_OPTIMIZER = "SIMBA"
#     print("Loaded SIMBA optimized")

# if os.path.exists("optimized_rag_mipro.json"):
#     rag_module.load("optimized_rag_mipro.json")
#     CURRENT_OPTIMIZER = "MIPROv2"
#     print("Loaded MIPROv2 optimized")

# elif os.path.exists("optimized_rag_copro.json"):
#     rag_module.load("optimized_rag_copro.json")
#     CURRENT_OPTIMIZER = "COPRO"
#     print("Loaded COPRO optimized")

# elif os.path.exists("optimized_rag.json"):
#     rag_module.load("optimized_rag.json")
#     CURRENT_OPTIMIZER = "Basic"
#     print("Loaded basic optimized")

# else:
#     CURRENT_OPTIMIZER = "Baseline"
#     print("No optimized model found")


COLLECTION_NAME = "sql_docs"


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
    return retriever


def extract_usage_stats(lm_usage):
    if not lm_usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    prompt = completion = total = 0

    for _, stats in lm_usage.items():
        prompt += stats.get("prompt_tokens", 0)
        completion += stats.get("completion_tokens", 0)
        total += stats.get("total_tokens", 0)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total
    }


@observe(name="RAG_Query")
def answer_question(
    question: str,
    k: int = 4,
    user_id: str | None = None,
    session_id: str | None = None
):
    with propagate_attributes(
        user_id=user_id or "anonymous",
        session_id=session_id or "default_session",
        tags=["rag", "sql-assistant", f"optimizer-{CURRENT_OPTIMIZER}"],
        metadata={
            "retrieval_k": k,
            "optimizer": CURRENT_OPTIMIZER,
            "model": "gemini-2.5-flash"
        },
        version="1.0.0"
    ):
        retriever = get_rag_components(k=k)
        docs = retriever.invoke(question)

        trace_id = (
            langfuse_client.get_current_trace_id()
            if langfuse_client else None
        )

        if not docs:
            if langfuse_client:
                langfuse_client.update_current_trace(
                    input={"question": question},
                    output={"answer": "No relevant context found."}
                )
            return "No relevant context found.", []

        context_list = [d.page_content for d in docs]

        prediction = rag_module(
            context=context_list,
            question=question
        )

        lm_usage = prediction.get_lm_usage()
        usage = extract_usage_stats(lm_usage)

        dspy.inspect_history(n=1)
        print("Token usage:", usage)

        if langfuse_client:
            langfuse_client.update_current_trace(
                input={
                    "question": question,
                    "num_context_docs": len(context_list)
                },
                output={
                    "answer": prediction.answer,
                    "num_sources": len(docs),
                    "answer_length_chars": len(prediction.answer)
                },
                metadata={
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"]
                }
            )

        sources = [
            (d.metadata.get("source", "unknown"), d.metadata.get("page", "?"))
            for d in docs
        ]

        return prediction.answer, sources, trace_id, context_list

    
def log_feedback(trace_id: str, score: int):
    if not langfuse_client or not trace_id:
        return

    langfuse_client.create_score(
        trace_id=trace_id,
        name="user_feedback",
        value=score,
        comment="User feedback on answer quality"
    )
    if score==0:
        increment_thumbs_down()
