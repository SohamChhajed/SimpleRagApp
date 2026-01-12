import dspy
import time
from typing import Optional, Union

class GEPARAGJudge(dspy.Signature):
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    expected_answer: str = dspy.InputField()

    verdict: str = dspy.OutputField(desc="YES or NO")
    feedback: str = dspy.OutputField(desc="Brief explanation")

class GEPARAGMetric:
    __name__ = "gepa_rag_metric"

    def __init__(self):
        self.judge = dspy.ChainOfThought(GEPARAGJudge)

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Optional[object] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[object] = None,
        return_outputs: bool = False
    ) -> Union[float, dict]:

        try:
            time.sleep(6)

            context = (
                "\n".join(gold.context)
                if isinstance(gold.context, list)
                else str(gold.context)
            )

            answer = pred.answer if hasattr(pred, "answer") else str(pred)

            result = self.judge(
                context=context,
                question=gold.question,
                answer=answer,
                expected_answer=gold.answer,
            )

            verdict = result.verdict.strip().upper()
            score = 1.0 if verdict == "YES" else 0.0

            feedback = result.feedback.strip() if hasattr(result, "feedback") else ""

            if not feedback:
                feedback = (
                    "The answer is correct."
                    if score == 1.0
                    else f"The answer is incorrect. Expected: '{gold.answer}'."
                )

            if pred_name:
                feedback = f"[{pred_name}] {feedback}"

            if return_outputs or pred_name is not None:
                return {
                    "score": score,
                    "feedback": feedback
                }

            return score

        except Exception as e:
            if return_outputs or pred_name is not None:
                return {
                    "score": 0.0,
                    "feedback": f"Evaluation error: {str(e)}"
                }
            return 0.0
