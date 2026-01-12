import dspy
import time

class RAGJudge(dspy.Signature):
    """
    Score how well the answer is correct AND grounded in the context.

    1.0 = fully correct and grounded
    0.5 = partially correct or weak grounding
    0.0 = incorrect or hallucinated
    """
    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Grounding score between 0.0 and 1.0")


class RAGMetric:
    __name__ = "rag_grounding_metric_soft"

    def __init__(self):
        self.judge = dspy.ChainOfThought(RAGJudge)
        self.cache = {}

    def __call__(self, example, prediction, trace=None):
        key = (example.question, prediction.answer)

        if key in self.cache:
            return self.cache[key]

        time.sleep(15)

        try:
            result = self.judge(
                context=example.context,
                question=example.question,
                answer=prediction.answer
            )

            score = max(0.0, min(1.0, float(result.score)))
            self.cache[key] = score
            return score

        except Exception:
            return 0.0
