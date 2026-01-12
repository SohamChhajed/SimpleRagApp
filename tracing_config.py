import mlflow
import os

def setup_mlflow_tracing():
    """Configure MLflow tracing for DSPy"""
    try:

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        

        experiment_name = "SQL_RAG_Application"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
        except Exception:
            mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        mlflow.dspy.autolog()
        
        print(f"MLflow setup successful: {experiment_name}")
        
    except Exception as e:
        print(f" MLflow setup warning: {e}")
def disable_mlflow_tracing():
    """Disable MLflow tracing"""
    mlflow.dspy.autolog(disable=True)