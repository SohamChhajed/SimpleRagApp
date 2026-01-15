import dspy
import time
from typing import Optional, Union


class GEPARAGJudge(dspy.Signature):
    """Evaluate if the model's answer is accurate and well-grounded in the retrieved context.
    
    The verdict MUST be exactly 'YES' or 'NO' - no other values are allowed.
    """
    context: str = dspy.InputField(desc="Retrieved documents")
    question: str = dspy.InputField(desc="User question")
    answer: str = dspy.InputField(desc="Model-generated answer")

    verdict: str = dspy.OutputField(desc="YES or NO")
    feedback: str = dspy.OutputField(desc="Brief explanation")


class HybridGEPARAGMetric:
    __name__ = "hybrid_gepa_rag_metric"

    def __init__(self):
        self.llm_judge = dspy.ChainOfThought(GEPARAGJudge)

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
        return_outputs=False,
    ) -> Union[float, dict]:

        if hasattr(gold, "human_score") and gold.human_score is not None:
            score = float(gold.human_score)
            feedback = getattr(
                gold,
                "human_feedback",
                "Human feedback provided"
            )

        else:
            try:
                time.sleep(6)

                context = (
                    "\n".join(gold.context)
                    if isinstance(gold.context, list)
                    else str(gold.context)
                )

                answer = pred.answer if hasattr(pred, "answer") else str(pred)

                result = self.llm_judge(
                    context=context,
                    question=gold.question,
                    answer=answer,
                )

                verdict = result.verdict.strip().upper()
                score = 1.0 if verdict == "YES" else 0.0
                feedback = result.feedback.strip()

            except Exception as e:
                score = 0.0
                feedback = f"Evaluation error: {str(e)}"

        if pred_name:
            feedback = f"[{pred_name}] {feedback}"

        return {
            "score": score,
            "feedback": feedback,
        }





# import dspy
# import time
# from typing import Optional, Union

# class GEPARAGJudge(dspy.Signature):
#     context: str = dspy.InputField()
#     question: str = dspy.InputField()
#     answer: str = dspy.InputField()
#     expected_answer: str = dspy.InputField()

#     verdict: str = dspy.OutputField(desc="YES or NO")
#     feedback: str = dspy.OutputField(desc="Brief explanation")

# class GEPARAGMetric:
#     __name__ = "gepa_rag_metric"

#     def __init__(self):
#         self.judge = dspy.ChainOfThought(GEPARAGJudge)

#     def __call__(
#         self,
#         gold: dspy.Example,
#         pred: dspy.Prediction,
#         trace: Optional[object] = None,
#         pred_name: Optional[str] = None,
#         pred_trace: Optional[object] = None,
#         return_outputs: bool = False
#     ) -> Union[float, dict]:

#         try:
#             time.sleep(6)

#             context = (
#                 "\n".join(gold.context)
#                 if isinstance(gold.context, list)
#                 else str(gold.context)
#             )

#             answer = pred.answer if hasattr(pred, "answer") else str(pred)

#             result = self.judge(
#                 context=context,
#                 question=gold.question,
#                 answer=answer,
#                 expected_answer=gold.answer,
#             )

#             verdict = result.verdict.strip().upper()
#             score = 1.0 if verdict == "YES" else 0.0

#             feedback = result.feedback.strip() if hasattr(result, "feedback") else ""

#             if not feedback:
#                 feedback = (
#                     "The answer is correct."
#                     if score == 1.0
#                     else f"The answer is incorrect. Expected: '{gold.answer}'."
#                 )

#             if pred_name:
#                 feedback = f"[{pred_name}] {feedback}"

#             if return_outputs or pred_name is not None:
#                 return {
#                     "score": score,
#                     "feedback": feedback
#                 }

#             return score

#         except Exception as e:
#             if return_outputs or pred_name is not None:
#                 return {
#                     "score": 0.0,
#                     "feedback": f"Evaluation error: {str(e)}"
#                 }
#             return 0.0




# import dspy
# import time
# from typing import Optional, Union


# class GEPARAGJudge(dspy.Signature):
#     """
#     Rubric-based judge.
#     No gold answer required.
#     """

#     context: str = dspy.InputField(desc="Retrieved documents")
#     question: str = dspy.InputField(desc="User question")
#     answer: str = dspy.InputField(desc="Model-generated answer")

#     verdict: str = dspy.OutputField(
#         desc="YES if the answer is correct, grounded in context, and helpful; otherwise NO"
#     )
#     feedback: str = dspy.OutputField(
#         desc="Brief explanation of the decision"
#     )


# class GEPARAGMetric:
#     __name__ = "gepa_rag_metric"

#     def __init__(self):
#         self.judge = dspy.ChainOfThought(GEPARAGJudge)

#     def __call__(
#         self,
#         gold: dspy.Example,
#         pred: dspy.Prediction,
#         trace: Optional[object] = None,
#         pred_name: Optional[str] = None,
#         pred_trace: Optional[object] = None,
#         return_outputs: bool = False
#     ) -> Union[float, dict]:

#         try:
#             # Rate-limit safety
#             time.sleep(6)

#             context = (
#                 "\n".join(gold.context)
#                 if isinstance(gold.context, list)
#                 else str(gold.context)
#             )

#             answer = pred.answer if hasattr(pred, "answer") else str(pred)

#             result = self.judge(
#                 context=context,
#                 question=gold.question,
#                 answer=answer
#             )

#             verdict = result.verdict.strip().upper()
#             score = 1.0 if verdict == "YES" else 0.0

#             feedback = result.feedback.strip()

#             if pred_name:
#                 feedback = f"[{pred_name}] {feedback}"

#             if return_outputs or pred_name is not None:
#                 return {
#                     "score": score,
#                     "feedback": feedback
#                 }

#             return score

#         except Exception as e:
#             if return_outputs or pred_name is not None:
#                 return {
#                     "score": 0.0,
#                     "feedback": f"Evaluation error: {str(e)}"
#                 }
#             return 0.0
