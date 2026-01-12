import os
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

def setup_langfuse():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    
    if not public_key or not secret_key:
        print(" Langfuse credentials not found in environment variables")
        return None
    langfuse = get_client()

    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
        print(f"  Connected to: {base_url}")
    else:
        print("Langfuse authentication failed. Please check your credentials.")
        return None

    DSPyInstrumentor().instrument()
    print(" DSPy instrumentation enabled for Langfuse")
    
    return langfuse