import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
import chromadb

# 1. Connect DSPy to your local Ollama Server
ollama_qwen = dspy.LM(
    model="ollama/qwen3:0.6b",
    api_base="http://localhost:11434",
    api_key="none" # Ollama doesn't require an API key
)

# 2. Connect to your local ChromaDB
retriever = ChromadbRM(
    collection_name="codebase_chunks",
    persist_directory="./my_local_chromadb",
    k=2  # Fetch the top 2 most relevant code chunks
)

# 3. Configure DSPy globally
dspy.configure(lm=ollama_qwen, rm=retriever)

# 4. Define your RAG logic
class CodebaseRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=2)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Fetch code snippets from ChromaDB
        context_chunks = self.retrieve(question).passages
        
        # Ask Qwen3 to reason and answer
        prediction = self.generate_answer(context=context_chunks, question=question)
        
        return dspy.Prediction(context=context_chunks, answer=prediction.answer)

# --- Run the App ---
rag_bot = CodebaseRAG()

# Try asking a question (assuming you have ingested code into ChromaDB)
result = rag_bot(question="What does the authentication module do?")

print(f"Thought Process:\n{result.reasoning}\n")
print(f"Final Answer:\n{result.answer}")