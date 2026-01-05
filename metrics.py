import dspy
class RAGJudge(dspy.Signature):
    """
    Judge if the answer is correct and grounded in the context.
    Respond with YES or NO only.
    """
    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    verdict: str = dspy.OutputField(desc="YES or NO")

class RAGMetric:
    def __init__(self):
        self.judge = dspy.ChainOfThought(RAGJudge)

    def __call__(self, example, prediction,trace=None):
        result = self.judge(
            context=example.context,
            question=example.question,
            answer=prediction.answer
        )
        return 1 if result.verdict.strip().upper() == "YES" else 0
