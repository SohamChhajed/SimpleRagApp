import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import COPRO
from dspy_config import configure_lm
from dspy_rag import RAGModule
from trainset import trainset
from langfuse_config import setup_langfuse
from langfuse import observe, propagate_attributes
from metrics import RAGMetric
from tracing_config import setup_mlflow_tracing
import time

def stringify_metadata(d):
    """Langfuse metadata must be strings."""
    return {k: str(v) for k, v in d.items() if v is not None}


def extract_token_summary(lm_usage):
    """Extract token usage and return as strings for metadata"""
    if not lm_usage:
        return {
            "prompt_tokens": "0",
            "completion_tokens": "0",
            "total_tokens": "0"
        }
    
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    
    for model_name, stats in lm_usage.items():
        total_prompt += stats.get('prompt_tokens', 0)
        total_completion += stats.get('completion_tokens', 0)
        total_tokens += stats.get('total_tokens', 0)
    
    return {
        "prompt_tokens": str(total_prompt),
        "completion_tokens": str(total_completion),
        "total_tokens": str(total_tokens)
    }


setup_mlflow_tracing()
configure_lm()

langfuse_client = setup_langfuse()
rag_module = RAGModule()

metric = RAGMetric()


@observe(name="Evaluate_Single_Example")
def tracked_metric(example, pred, trace=None):
    score = metric(example, pred, trace)
    
    lm_usage = pred.get_lm_usage() if hasattr(pred, 'get_lm_usage') else {}
    token_summary = extract_token_summary(lm_usage)
    
    with propagate_attributes(
        metadata={
            "question": example.question[:100] if hasattr(example, 'question') else None,
            "score": str(score),
            "has_context": str(hasattr(example, 'context')),
            "prompt_tokens": token_summary["prompt_tokens"],
            "completion_tokens": token_summary["completion_tokens"],
            "total_tokens": token_summary["total_tokens"]
        }
    ):
        pass
    
    return score


@observe(name="COPRO_Optimization_Full")
def run_copro_optimization():
    with propagate_attributes(
        tags=["optimization", "copro", "training"],
        metadata={
            "optimizer": "COPRO",
            "num_examples": str(len(trainset)),
            "model": "gemini-2.5-flash",
            "breadth": "2",
            "depth": "1",
            "init_temperature": "1.0"
        },
        version="1.0.0"
    ):
        start_time = time.time()
        
        copro_optimizer = COPRO(
            prompt_model=dspy.settings.lm,
            metric=tracked_metric,
            breadth=2,
            depth=1,
            init_temperature=1.0,
            verbose=True
        )

        eval_kwargs = {
            'num_threads': 1,
            'display_progress': True,
            'display_table': True
        }

        try:
            
            optimized_rag = copro_optimizer.compile(
                student=rag_module,
                trainset=trainset[:3],
                eval_kwargs=eval_kwargs
            )
            
            optimization_duration = time.time() - start_time
            
            optimized_rag.save("optimized_rag_copro.json")
            
            state = optimized_rag.dump_state()
            optimized_prompts = {}
            for key, value in state.items():
                if 'signature_instructions' in value:
                    optimized_prompts[key] = value['signature_instructions']
            
            lm_usage = optimized_rag.get_lm_usage() if hasattr(optimized_rag, 'get_lm_usage') else {}
            token_summary = extract_token_summary(lm_usage)
            
            print(f"   Tokens - Prompt: {token_summary['prompt_tokens']}, Completion: {token_summary['completion_tokens']}, Total: {token_summary['total_tokens']}")

            if langfuse_client:
                langfuse_client.update_current_trace(
                    output={
                        "status": "optimization_complete",
                        "saved_to": "optimized_rag_copro.json",
                        "optimization_duration_seconds": str(round(optimization_duration, 2)),
                        "num_prompts_optimized": str(len(optimized_prompts)),
                        "prompt_tokens": token_summary["prompt_tokens"],
                        "completion_tokens": token_summary["completion_tokens"],
                        "total_tokens": token_summary["total_tokens"],
                        "optimized_instructions_preview": str(optimized_prompts)[:500]
                    }
                )
            
            return optimized_rag, optimized_prompts
            
        except Exception as e:
            if langfuse_client:
                langfuse_client.update_current_trace(
                    output={
                        "status": "optimization_failed",
                        "error": str(e)
                    }
                )
            raise


@observe(name="Final_Evaluation")
def run_final_evaluation(optimized_rag):
    with propagate_attributes(
        tags=["evaluation", "copro", "final-score"],
        metadata={
            "optimizer": "COPRO",
            "num_examples": str(len(trainset))
        }
    ):
        
        evaluate = Evaluate(
            devset=trainset,
            metric=tracked_metric,
            num_threads=1,
            display_progress=True,
            display_table=True
        )
        
        start_time = time.time()
        score = evaluate(optimized_rag)
        eval_duration = time.time() - start_time
        
        eval_usage = optimized_rag.get_lm_usage() if hasattr(optimized_rag, 'get_lm_usage') else {}
        token_summary = extract_token_summary(eval_usage)
        
        print(f"Final Score: {score}")
        print(f" Tokens - Prompt: {token_summary['prompt_tokens']}, Completion: {token_summary['completion_tokens']}, Total: {token_summary['total_tokens']}")
        
        if langfuse_client:
            langfuse_client.update_current_trace(
                output={
                    "final_score": str((score)),
                    "evaluation_duration_seconds": str(round(eval_duration, 2)),
                    "num_threads": "1",
                    "prompt_tokens": token_summary["prompt_tokens"],
                    "completion_tokens": token_summary["completion_tokens"],
                    "total_tokens": token_summary["total_tokens"]
                }
            )
        
        dspy.inspect_history(n=2)
        
        return score


if __name__ == "__main__":
    try:
        optimized_rag, optimized_prompts = run_copro_optimization()
        
        final_score = run_final_evaluation(optimized_rag)
        print(f"\nFinal Score: {final_score}")
        
    except Exception as e:
        print(f"\nError during optimization:")
        import traceback
        traceback.print_exc()