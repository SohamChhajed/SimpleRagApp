import time
from datetime import datetime, timedelta

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA

from dspy_config import configure_lm
from dspy_rag import RAGModule
from feedback_trainset import load_feedback_trainset
from gepa_metrics import HybridGEPARAGMetric

from langfuse import observe, propagate_attributes
from langfuse_config import setup_langfuse
from tracing_config import setup_mlflow_tracing

from feedback_store import (
    get_last_gepa_run_time,
    count_feedback_since,
    record_gepa_run,
)


setup_mlflow_tracing()
configure_lm()
langfuse_client = setup_langfuse()

rag_module = RAGModule()
metric = HybridGEPARAGMetric()

trainset = load_feedback_trainset(
    max_samples=4,
    positive_ratio=0.25
)



def extract_token_summary(lm_usage):
    if not lm_usage:
        return {
            "prompt_tokens": "0",
            "completion_tokens": "0",
            "total_tokens": "0",
        }

    prompt = completion = total = 0
    for _, stats in lm_usage.items():
        prompt += stats.get("prompt_tokens", 0)
        completion += stats.get("completion_tokens", 0)
        total += stats.get("total_tokens", 0)

    return {
        "prompt_tokens": str(prompt),
        "completion_tokens": str(completion),
        "total_tokens": str(total),
    }

@observe(name="Evaluate_Single_Example")
def tracked_metric(
    gold,
    pred,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    result = metric(
        gold,
        pred,
        trace=trace,
        pred_name=pred_name,
        pred_trace=pred_trace,
    )

    score = result["score"]
    feedback = result["feedback"]

    lm_usage = pred.get_lm_usage() if hasattr(pred, "get_lm_usage") else {}
    tokens = extract_token_summary(lm_usage)

    with propagate_attributes(
        metadata={
            "question": gold.question[:100] if hasattr(gold, "question") else None,
            "score": str(score),
            "feedback": feedback,
            "pred_name": pred_name,
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "total_tokens": tokens["total_tokens"],
        }
    ):
        pass
    return dspy.Prediction(score=score, feedback=feedback)

# @observe(name="Evaluate_Single_Example")
# def tracked_metric(
#     gold,
#     pred,
#     trace=None,
#     pred_name=None,
#     pred_trace=None,
#     return_outputs=False,
# ):
#     result = metric(
#         gold,
#         pred,
#         trace=trace,
#         pred_name=pred_name,
#         pred_trace=pred_trace,
#         return_outputs=True,
#     )

#     score = float(result["score"])
#     feedback = result.get("feedback")
#     lm_usage = pred.get_lm_usage() if hasattr(pred, "get_lm_usage") else {}
#     tokens = extract_token_summary(lm_usage)

#     with propagate_attributes(
#         metadata={
#             "question": gold.question[:100] if hasattr(gold, "question") else None,
#             "score": str(score),
#             "feedback": feedback,
#             "pred_name": pred_name,
#             "prompt_tokens": tokens["prompt_tokens"],
#             "completion_tokens": tokens["completion_tokens"],
#             "total_tokens": tokens["total_tokens"],
#         }
#     ):
#         pass
#     if return_outputs:
#         return result
#     else:
#         return score


@observe(name="GEPA_Optimization_Full")
def run_gepa_optimization():
    last_run = get_last_gepa_run_time()
    now = datetime.now()

    # if last_run and (now - last_run) < timedelta(hours=24):
    #     print("Skipping GEPA: last run < 24h ago")
    #     return None

    # new_feedback_count = count_feedback_since(last_run)
    # if new_feedback_count < 3:
    #     print(f"Skipping GEPA: only {new_feedback_count} new feedbacks")
    #     return None

    with propagate_attributes(
        tags=["optimization", "gepa", "training"],
        metadata={
            "optimizer": "GEPA",
            "num_examples": str(len(trainset)),
            "model": str(dspy.settings.lm),
        },
        version="1.0.0",
    ):
        start = time.time()

        gepa = GEPA(
            metric=tracked_metric,
            max_metric_calls=5,
            reflection_lm=dspy.settings.lm,
            reflection_minibatch_size=3,
            candidate_selection_strategy="current_best",
            component_selector="round_robin",
            skip_perfect_score=True,
            num_threads=1,
            track_stats=True,
            seed=42,
        )

        optimized_rag = gepa.compile(
            student=rag_module,
            trainset=trainset[:3]
        )

        duration = time.time() - start
        optimized_rag.save("optimized_rag_gepa.json")

        record_gepa_run(last_feedback_at=now)

        if langfuse_client:
            langfuse_client.update_current_trace(
                output={
                    "status": "optimization_complete",
                    "duration_seconds": str(round(duration, 2)),
                }
            )

        return optimized_rag
    
def score_only_metric(gold, pred):
    result = metric(gold, pred)
    if isinstance(result, dict):
        return result["score"]
    return result.score if hasattr(result, 'score') else float(result)

@observe(name="Final_Evaluation")
def run_final_evaluation(optimized_rag):
    evaluator = Evaluate(
        devset=trainset[:5],
        metric=score_only_metric,
        num_threads=1,
        display_progress=True,
        display_table=True,
    )

    score = evaluator(optimized_rag)
    print(f" Final GEPA score: {score}")
    return score

if __name__ == "__main__":
    optimized = run_gepa_optimization()

    if optimized is not None:
        run_final_evaluation(optimized)
    else:
        print("GEPA not triggered the system continues using current model")
