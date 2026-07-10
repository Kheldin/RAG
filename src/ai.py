"""Model access and retrieval-augmented generation helpers."""

import os
import time
import socket
import subprocess
from ollama import Client # type: ignore
from .exceptions import LLMException, LogType, log_message
from src.models.models import MinimalSource
from .indexer import retrieve_chunks

def is_ollama_alive(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Return True when the Ollama server accepts connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0

class AI:
    """Interface to Ollama for answer generation."""
    
    def __init__(self) -> None:
        """Connect to Ollama and load the configured model."""
        self.model_name = "qwen3:0.6b"
        self._ensure_server_running()
        self.client = Client(host="http://localhost:11434", headers={})
        log_message(f"Loaded model {self.model_name}!", LogType.SUCCESS)

    def _ensure_server_running(self) -> None:
        """Start Ollama if the local server is not already running."""
        if not is_ollama_alive():
            log_message("Ollama server not running. Launching...", LogType.WARNING)
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid if os.name != "nt" else None,
                )
                while not is_ollama_alive():
                    time.sleep(1)
            except FileNotFoundError:
                raise LLMException("Ollama executable not found in PATH.")

    def RAG(self, query: str, k: int = 3, max_length: int = 1024) -> tuple[str, float]:
        """Answer a query with retrieved context and return the latency."""
        found_files: list[MinimalSource] = []
        chunks = retrieve_chunks(query, found_files, k=k)[::-1]

        context = "\n\n".join(
            f"[Document {len(chunks) - i}]\n{chunk}"
            for i, chunk in enumerate(chunks)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict, expert technical assistant. "
                    "You must answer the user's question using ONLY the provided Context. "
                    "Do not use markdown, code blocks, lists, or any formatting symbols. Answer in a single plain text paragraph. "
                    "If the answer cannot be confidently deduced from the Context, you must reply exactly with: 'Information not found in context.' "
                    "Be highly concise and direct."
                ),
            }
        ]
        
        # Upgraded Few-Shot examples to reinforce the "Plain Text" constraint
        examples = [
            ("Where can I find information about using generative models in vLLM?", 
             "Information about using generative models can be found on the generative models page as referenced in the supported models documentation."),
            ("What conditions must be met for ModelRunner to use CUDA graphs?", 
             "Two conditions must be met. First, prefill_meta must be None. Second, decode_meta.use_cuda_graph must be True. When both are satisfied, the ModelRunner uses the virtual engine graph runners instead of the regular model.")
        ]
        
        for q, a in examples:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}",
        })

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_gpu": 99,
                "num_predict": max(max_length, 512),
                "repeat_penalty": 1.15,
                "enable_thinking": False,
                "temperature": 0.0
            },
        }

        try:
            response = self.client.chat(**payload)
            answer = (
                response.message.content 
                if hasattr(response, "message") 
                else response.get("message", {}).get("content", "")
            )
            
            # Using dict .get() fallback for Ollama python client dict structures
            total_duration = getattr(response, "total_duration", response.get("total_duration", 0))
            load_duration = getattr(response, "load_duration", response.get("load_duration", 0))
            
            total_compute_time = (total_duration - load_duration) / 1e9
        except Exception as e:
            raise LLMException(f"Failed to communicate with Ollama: {e}")
            
        return answer.strip(), float(total_compute_time)