import dspy

# This loads the model directly into your Python process. 
# It will automatically detect your GPU if you have one, or fall back to CPU.
local_qwen = dspy.HFModel(model="Qwen/Qwen3-0.6B")

# Set it as the global default
dspy.configure(lm=local_qwen)

# Define your signature and module as normal
class CodeQuestion(dspy.Signature):
    """Answer a question about the codebase."""
    context: str = dspy.InputField(desc="Relevant code snippets")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa_bot = dspy.Predict(CodeQuestion)

# Run it
response = qa_bot(
    context="def add(a, b): return a + b", 
    question="What does this function do?"
)

print(response.answer)