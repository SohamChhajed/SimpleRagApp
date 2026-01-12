import dspy
import time
from dspy.evaluate import Evaluate
from dspy.teleprompt import SIMBA

from dspy_config import configure_lm
from dspy_rag import RAGModule
from trainset import trainset
from metrics import RAGMetric
from tracing_config import setup_mlflow_tracing
from langfuse_config import setup_langfuse
from langfuse import observe, propagate_attributes


# -------------------- setup --------------------
setup_mlflow_tracing()
configure_lm()

langfuse_client = setup_langfuse()

rag_module = RAGModule()
metric = RAGMetric()


# -------------------- helpers --------------------
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


# -------------------- optimization --------------------
@observe(name="SIMBA_Optimization")
def run_simba_optimization():
    with propagate_attributes(
        tags=["optimization", "simba"],
        metadata={
            "optimizer": "SIMBA",
            "num_examples": str(len(trainset)),
            "model": "gemini-2.5-flash",
            "bsize": "2",
            "num_candidates": "2",
            "max_steps": "2",
            "max_demos": "1",
        },
        version="1.0.0",
    ):
        start_time = time.time()

        simba = SIMBA(
            metric=metric,
            bsize=2,
            num_candidates=2,
            max_steps=2,
            max_demos=1,
            prompt_model=dspy.settings.lm,
            temperature_for_sampling=0.3,
            temperature_for_candidates=0.3,
        )

        try:
            optimized_rag = simba.compile(
                student=rag_module,
                trainset=trainset,
            )

            duration = time.time() - start_time
            optimized_rag.save("optimized_rag_simba.json")

            # Extract optimized instructions
            state = optimized_rag.dump_state()
            optimized_prompts = {
                k: v["signature_instructions"]
                for k, v in state.items()
                if isinstance(v, dict) and "signature_instructions" in v
            }

            usage = optimized_rag.get_lm_usage() if hasattr(optimized_rag, "get_lm_usage") else {}
            token_summary = extract_token_summary(usage)

            if langfuse_client:
                langfuse_client.update_current_trace(
                    output={
                        "status": "optimization_complete",
                        "saved_to": "optimized_rag_simba.json",
                        "optimization_duration_seconds": str(round(duration, 2)),
                        "num_prompts_optimized": str(len(optimized_prompts)),
                        "prompt_tokens": token_summary["prompt_tokens"],
                        "completion_tokens": token_summary["completion_tokens"],
                        "total_tokens": token_summary["total_tokens"],
                        "optimized_instructions_preview": str(optimized_prompts)[:500],
                    }
                )

            return optimized_rag, optimized_prompts

        except Exception as e:
            if langfuse_client:
                langfuse_client.update_current_trace(
                    output={
                        "status": "optimization_failed",
                        "error": str(e),
                    }
                )
            raise

@observe(name="SIMBA_Final_Evaluation")
def run_final_evaluation(optimized_rag):
    with propagate_attributes(
        tags=["evaluation", "simba", "final-score"],
        metadata={
            "optimizer": "SIMBA",
            "num_examples": str(len(trainset)),
        },
    ):
        evaluate = Evaluate(
            devset=trainset,
            metric=metric,
            num_threads=1,
            display_progress=True,
            display_table=True,
        )

        start_time = time.time()
        score = evaluate(optimized_rag)
        duration = time.time() - start_time

        usage = optimized_rag.get_lm_usage() if hasattr(optimized_rag, "get_lm_usage") else {}
        token_summary = extract_token_summary(usage)

        if langfuse_client:
            langfuse_client.update_current_trace(
                output={
                    "final_score": str(score),
                    "evaluation_duration_seconds": str(round(duration, 2)),
                    "prompt_tokens": token_summary["prompt_tokens"],
                    "completion_tokens": token_summary["completion_tokens"],
                    "total_tokens": token_summary["total_tokens"],
                }
            )

        dspy.inspect_history(n=3)
        print(f"\nFinal SIMBA Score: {score}")

        return score

if __name__ == "__main__":
    try:
        optimized_rag, optimized_prompts = run_simba_optimization()
        final_score = run_final_evaluation(optimized_rag)
        print(f"\nFinal Score: {final_score}")

    except Exception:
        import traceback
        traceback.print_exc()
