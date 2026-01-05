import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy_config import configure_lm
from dspy_rag import rag_module
from trainset import trainset
from metrics import RAGMetric
configure_lm() 
metric = RAGMetric()
optimizer = BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=5
)
optimized_rag = optimizer.compile(
    rag_module,
    trainset=trainset
)
optimized_rag.save("optimized_rag.json")
