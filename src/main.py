from typing import List
import dspy
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult

# Connection to Ollama and set Qwen as default language model 
ollama_qwen: dspy.LM = dspy.LM(
    model="ollama/qwen3:0.6b",
    api_base="http://localhost:11434",
    api_key="none"
)
dspy.configure(lm=ollama_qwen)  # type: ignore

# 2. Connect to ChromaDB directly
chroma_client: ClientAPI = chromadb.PersistentClient(path="./my_local_chromadb")
collection: Collection = chroma_client.get_or_create_collection(name="codebase_chunks")

# 3. Define the DSPy RAG Module
class CodebaseRAG(dspy.Module):
    def __init__(self) -> None:
        super().__init__()  # type: ignore
        
        self.generate_answer: dspy.ChainOfThought = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        # Step A: Query ChromaDB natively using its own API
        results: QueryResult = collection.query(
            query_texts=[question],
            n_results=2
        )
        
        # Step B: Extract the actual text documents safely based on the QueryResult shape
        raw_documents = results.get("documents")
        context_chunks: List[str] = raw_documents[0] if raw_documents else []
        
        # Step C: Pass the raw text and the question into the LLM
        prediction: dspy.Prediction = self.generate_answer(context=str(context_chunks), question=question)
        
        return dspy.Prediction(
            context=context_chunks, 
            reasoning=getattr(prediction, "reasoning", ""), 
            answer=getattr(prediction, "answer", "")
        )

# --- Run the App ---
rag_bot: CodebaseRAG = CodebaseRAG()

result: dspy.Prediction = rag_bot(question="What does the authentication module do?")

print(f"Thought Process:\n{getattr(result, 'reasoning', '')}\n")
print(f"Final Answer:\n{getattr(result, 'answer', '')}")