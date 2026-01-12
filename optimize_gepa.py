import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA
from dspy_config import configure_lm
from dspy_rag import RAGModule
from trainset import trainset
from gepa_metrics import GEPARAGMetric

from langfuse import observe, propagate_attributes
from langfuse_config import setup_langfuse
from tracing_config import setup_mlflow_tracing
import time


def extract_token_summary(lm_usage):
    if not lm_usage:
        return {
            "prompt_tokens": "0",
            "completion_tokens": "0",
            "total_tokens": "0",
        }

    total_prompt = 0
    total_completion = 0
    total_tokens = 0

    for _, stats in lm_usage.items():
        total_prompt += stats.get("prompt_tokens", 0)
        total_completion += stats.get("completion_tokens", 0)
        total_tokens += stats.get("total_tokens", 0)

    return {
        "prompt_tokens": str(total_prompt),
        "completion_tokens": str(total_completion),
        "total_tokens": str(total_tokens),
    }


setup_mlflow_tracing()
configure_lm()
langfuse_client = setup_langfuse()

rag_module = RAGModule()
metric = GEPARAGMetric()


@observe(name="Evaluate_Single_Example")
def tracked_metric(
    gold,
    pred,
    trace=None,
    pred_name=None,
    pred_trace=None,
    return_outputs=False,
):
    result = metric(
        gold,
        pred,
        trace=trace,
        pred_name=pred_name,
        pred_trace=pred_trace,
        return_outputs=return_outputs,
    )

    score = result["score"] if isinstance(result, dict) else result

    lm_usage = pred.get_lm_usage() if hasattr(pred, "get_lm_usage") else {}
    token_summary = extract_token_summary(lm_usage)

    with propagate_attributes(
        metadata={
            "question": gold.question[:100] if hasattr(gold, "question") else None,
            "score": str(score),
            "pred_name": pred_name,
            "prompt_tokens": token_summary["prompt_tokens"],
            "completion_tokens": token_summary["completion_tokens"],
            "total_tokens": token_summary["total_tokens"],
        }
    ):
        pass

    return result



@observe(name="GEPA_Optimization_Full")
def run_gepa_optimization():
    with propagate_attributes(
        tags=["optimization", "gepa", "training"],
        metadata={
            "optimizer": "GEPA",
            "num_examples": str(len(trainset)),
            "candidate_selection": "current_best",
            "component_selector": "round_robin",
            "reflection_minibatch_size": "5",
            "model": str(dspy.settings.lm),
        },
        version="1.0.0",
    ):
        start_time = time.time()

        gepa = GEPA(
            metric=tracked_metric, 
            max_metric_calls=5,
            reflection_lm=dspy.settings.lm,
            reflection_minibatch_size=5,
            candidate_selection_strategy="current_best",
            component_selector="round_robin",
            skip_perfect_score=True,
            num_threads=1,
            track_stats=True,
            seed=42,
        )

        optimized_rag = gepa.compile(
            student=rag_module,
            trainset=trainset[:3],
            valset=trainset[:2],
        )

        duration = time.time() - start_time
        optimized_rag.save("optimized_rag_gepa.json")

        if langfuse_client:
            langfuse_client.update_current_trace(
                output={
                    "status": "optimization_complete",
                    "optimization_duration_seconds": str(round(duration, 2)),
                }
            )

        return optimized_rag


@observe(name="Final_Evaluation")
def run_final_evaluation(optimized_rag):
    evaluate = Evaluate(
        devset=trainset[:5],
        metric=tracked_metric,
        num_threads=1,
        display_progress=True,
        display_table=True,
    )

    score = evaluate(optimized_rag)
    print(f"Final Score: {score}")
    return score


if __name__ == "__main__":
    optimized_rag = run_gepa_optimization()
    run_final_evaluation(optimized_rag)
