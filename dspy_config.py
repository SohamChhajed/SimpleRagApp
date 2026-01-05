import dspy
import os
from config import get_gemini_api_key
def configure_lm():
    api_key=get_gemini_api_key()
    lm=dspy.LM(
        model="gemini/gemini-2.5-flash",
        api_key=api_key,
        temperature=0.1
    )
    dspy.configure(lm=lm)