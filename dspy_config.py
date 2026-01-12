import dspy
import os
from config import get_gemini_api_key
def configure_lm():
    api_key=get_gemini_api_key()
    lm=dspy.LM(
        model="gemini/gemini-2.5-flash",
        api_key=api_key,
        temperature=0.1,
        track_usage=True,
    )
    dspy.configure(lm=lm)
# import dspy
# import time
# from config import get_gemini_api_key


# class RateLimitedLM(dspy.LM):
#     def __init__(self, base_lm, delay_seconds=15):
#         self.base_lm = base_lm
#         self.delay_seconds = delay_seconds

#     def __call__(self, *args, **kwargs):
#         time.sleep(self.delay_seconds)
#         return self.base_lm(*args, **kwargs)

#     def copy(self, **kwargs):
#         # MIPRO NEEDS THIS
#         copied_base = self.base_lm.copy(**kwargs)
#         return RateLimitedLM(copied_base, self.delay_seconds)

#     def __getattr__(self, name):
#         # Forward everything else to base LM
#         return getattr(self.base_lm, name)


# def configure_lm():
#     api_key = get_gemini_api_key()

#     base_lm = dspy.LM(
#         model="gemini/gemini-2.5-flash",
#         api_key=api_key,
#         temperature=0.1,
#         track_usage=True,
#     )

#     # Wrap with rate limiter
#     rate_limited_lm = RateLimitedLM(base_lm, delay_seconds=15)

#     dspy.configure(lm=rate_limited_lm)

