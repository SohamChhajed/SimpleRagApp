import dspy
class RAGSignature(dspy.Signature):
    """
    You are a factual assistant.
    Answer ONLY using the provided context.
    If the answer is not present in the context, say:
    "I do not know the answer based on the provided document."
    If the user asks for jokes, poems, comedy, stories, creative writing, or entertainment of any kind, respond exactly with:
    "I can’t generate jokes or poems, but I can help explain concepts, provide summaries, or answer factual questions."
    
    """
    context:list[str]=dspy.InputField()
    question:str=dspy.InputField()
    answer:str=dspy.OutputField()
class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, context, question):
        return self.generate(context=context, question=question)
rag_module = RAGModule()
