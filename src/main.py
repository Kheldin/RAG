from typing import List, Set
import dspy
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult

# 1. Connection to Ollama and set Qwen as default language model 
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
        
        # We define a cleaner prompt instructing the LLM to use the source markers we provide
        self.generate_answer: dspy.ChainOfThought = dspy.ChainOfThought(
            "context, question -> answer",
            instructions="Answer the question using the provided codebase context. Explicitly mention the file names you used from the context headers."
        )

    def forward(self, question: str) -> dspy.Prediction:
        # Step A: Query ChromaDB natively using its own API
        results: QueryResult = collection.query(
            query_texts=[question],
            n_results=3 # Boosted to 3 to get a slightly wider net of files
        )
        
        # Step B: Extract text documents AND their accompanying metadata
        raw_documents = results.get("documents")
        raw_metadatas = results.get("metadatas")
        
        context_chunks: List[str] = raw_documents[0] if raw_documents else []
        metadata_chunks = raw_metadatas[0] if raw_metadatas else []
        
        # Step C: Format context dynamically to bake the file names inside the prompt text
        # This allows the LLM to "see" which code belongs to which file
        formatted_context_list = []
        unique_sources: Set[str] = set()
        
        for doc, meta in zip(context_chunks, metadata_chunks):
            source_file = meta.get("source", "Unknown file") if meta else "Unknown file"
            unique_sources.add(source_file)
            
            # Wrap each chunk with an explicit header boundary
            formatted_context_list.append(f"--- File: {source_file} ---\n{doc}\n")
            
        context_str = "\n".join(formatted_context_list)
        
        # Step D: Pass the formatted text block and the question into the LLM
        prediction: dspy.Prediction = self.generate_answer(context=context_str, question=question)
        
        # Step E: Expose both the raw chunks and the cleaned unique list of file paths
        return dspy.Prediction(
            context=context_chunks,
            sources=list(unique_sources),
            reasoning=getattr(prediction, "reasoning", ""), 
            answer=getattr(prediction, "answer", "")
        )

# --- Run the App ---
rag_bot: CodebaseRAG = CodebaseRAG()

result: dspy.Prediction = rag_bot(question="What does the authentication module do?")

print(f"Thought Process:\n{getattr(result, 'reasoning', '')}\n")
print(f"Final Answer:\n{getattr(result, 'answer', '')}\n")

print("--- Sources Used ---")
for idx, source in enumerate(getattr(result, 'sources', []), 1):
    print(f"[{idx}] {source}")