import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2
from dspy_config import configure_lm
from dspy_rag import RAGModule
from trainset import trainset
from metrics import RAGMetric
from tracing_config import setup_mlflow_tracing
from langfuse_config import setup_langfuse
from langfuse import observe, propagate_attributes
import time

setup_mlflow_tracing()
configure_lm()

langfuse_client = setup_langfuse()
rag_module = RAGModule()
metric = RAGMetric()

def extract_token_summary(lm_usage):
    if not lm_usage:
        return {
            "prompt_tokens": "0",
            "completion_tokens": "0",
            "total_tokens": "0"
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
        "total_tokens": str(total_tokens)
    }

@observe(name="Evaluate_Single_Example")
def tracked_metric(example, pred, trace=None):
    score = metric(example, pred, trace)

    lm_usage = pred.get_lm_usage() if hasattr(pred, "get_lm_usage") else {}
    token_summary = extract_token_summary(lm_usage)

    with propagate_attributes(
        metadata={
            "question": example.question[:100] if hasattr(example, "question") else None,
            "score": str(score),
            "has_context": str(hasattr(example, "context")),
            "prompt_tokens": token_summary["prompt_tokens"],
            "completion_tokens": token_summary["completion_tokens"],
            "total_tokens": token_summary["total_tokens"],
        }
    ):
        pass

    return score

@observe(name="MIPROv2_Optimization_Full")
def run_mipro_optimization():
    with propagate_attributes(
        tags=["optimization", "mipro", "training"],
        metadata={
            "optimizer": "MIPROv2",
            "num_examples": str(len(trainset)),
            "model": "gemini-2.5-flash",
            "num_candidates": "2",
            "num_trials": "2",
            "max_bootstrapped_demos": "1",
            "max_labeled_demos": "1",
        },
        version="1.0.0",
    ):
        start_time = time.time()

        mipro_optimizer = MIPROv2(
            prompt_model=dspy.settings.lm,
            task_model=dspy.settings.lm,
            metric=tracked_metric,
            num_candidates=2,
            init_temperature=1.0,
            verbose=True,
            auto=None,
        )

        try:
            optimized_rag = mipro_optimizer.compile(
                student=rag_module,
                trainset=trainset,
                num_trials=2,
                max_bootstrapped_demos=1,
                max_labeled_demos=1,
                requires_permission_to_run=False,
                minibatch_size=3,
            )

            optimization_duration = time.time() - start_time
            optimized_rag.save("optimized_rag_mipro.json")

            state = optimized_rag.dump_state()
            optimized_prompts = {}
            few_shot_demos = {}

            for key, value in state.items():
                if "signature_instructions" in value:
                    optimized_prompts[key] = value["signature_instructions"]
                if "demos" in value:
                    few_shot_demos[key] = len(value["demos"])

            lm_usage = optimized_rag.get_lm_usage() if hasattr(optimized_rag, "get_lm_usage") else {}
            token_summary = extract_token_summary(lm_usage)

            if langfuse_client:
                langfuse_client.update_current_trace(
                    output={
                        "status": "optimization_complete",
                        "saved_to": "optimized_rag_mipro.json",
                        "optimization_duration_seconds": str(round(optimization_duration, 2)),
                        "num_prompts_optimized": str(len(optimized_prompts)),
                        "few_shot_demos_per_module": str(few_shot_demos),
                        "total_demos": str(sum(few_shot_demos.values())),
                        "prompt_tokens": token_summary["prompt_tokens"],
                        "completion_tokens": token_summary["completion_tokens"],
                        "total_tokens": token_summary["total_tokens"],
                        "optimized_instructions_preview": str(optimized_prompts)[:500],
                    }
                )

            return optimized_rag, optimized_prompts, few_shot_demos

        except Exception as e:
            if langfuse_client:
                langfuse_client.update_current_trace(
                    output={
                        "status": "optimization_failed",
                        "error": str(e),
                    }
                )
            raise

@observe(name="Final_Evaluation")
def run_final_evaluation(optimized_rag):
    with propagate_attributes(
        tags=["evaluation", "mipro", "final-score"],
        metadata={
            "optimizer": "MIPROv2",
            "num_examples": str(len(trainset)),
        },
    ):
        evaluate = Evaluate(
            devset=trainset,
            metric=tracked_metric,
            num_threads=1,
            display_progress=True,
            display_table=True,
        )

        start_time = time.time()
        score = evaluate(optimized_rag)
        eval_duration = time.time() - start_time

        lm_usage = optimized_rag.get_lm_usage() if hasattr(optimized_rag, "get_lm_usage") else {}
        token_summary = extract_token_summary(lm_usage)

        if langfuse_client:
            langfuse_client.update_current_trace(
                output={
                    "final_score": str(score),
                    "evaluation_duration_seconds": str(round(eval_duration, 2)),
                    "num_threads": "1",
                    "prompt_tokens": token_summary["prompt_tokens"],
                    "completion_tokens": token_summary["completion_tokens"],
                    "total_tokens": token_summary["total_tokens"],
                }
            )

        dspy.inspect_history(n=2)
        return score

if __name__ == "__main__":
    try:
        optimized_rag, optimized_prompts, few_shot_demos = run_mipro_optimization()
        final_score = run_final_evaluation(optimized_rag)
        print(f"\nFinal Score: {final_score}")

    except Exception:
        import traceback
        traceback.print_exc()
