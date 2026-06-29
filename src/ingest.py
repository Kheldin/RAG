import os
import torch
from typing import List, Dict, Any
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

class CodebaseIndexer:
    """Scans a codebase, splits code/markdown into chunks, and indexes them into ChromaDB."""
    
    def __init__(
        self, 
        codebase_dir: str, 
        chroma_path: str = "./my_local_chromadb", 
        collection_name: str = "codebase_chunks",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 1000
    ):
        self.codebase_dir = codebase_dir
        self.batch_size = batch_size
        
        self.chroma_client: ClientAPI = chromadb.PersistentClient(path=chroma_path)
        self.collection: Collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading embedding model ({embedding_model_name}) on {device}...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=1000, 
            chunk_overlap=100
        )
        
        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        )
        self.markdown_text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=100
        )
        
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.chunk_counter: int = 0

    def _process_file(self, file_path: str, file_type: str) -> None:
        """Reads a file, splits it into chunks, and appends them to the processing queue."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            if not content.strip():
                return
                
            if file_type == 'python':
                for chunk in self.python_splitter.split_text(content):
                    self.documents.append(chunk)
                    self.metadatas.append({"source": file_path, "type": "python"})
                    self.ids.append(f"chunk_{self.chunk_counter}")
                    self.chunk_counter += 1
            
            elif file_type == 'markdown':
                # First split by headers to retain logical sections as metadata
                header_splits = self.markdown_header_splitter.split_text(content)
                # Then split those sections by length to avoid exceeding model context limits
                final_splits = self.markdown_text_splitter.split_documents(header_splits)
                
                for doc in final_splits:
                    self.documents.append(doc.page_content)
                    meta = {"source": file_path, "type": "markdown"}
                    meta.update(doc.metadata)
                    self.metadatas.append(meta)
                    self.ids.append(f"chunk_{self.chunk_counter}")
                    self.chunk_counter += 1
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def _flush_batch(self, force_all: bool = False) -> None:
        """Encodes the current queue of documents and inserts them into ChromaDB."""
        while len(self.documents) >= self.batch_size or (force_all and self.documents):
            # 3. Fixed Batching Logic to prevent OOM / DB payload limits
            take_count = min(len(self.documents), self.batch_size)
            
            batch_docs = self.documents[:take_count]
            batch_metas = self.metadatas[:take_count]
            batch_ids = self.ids[:take_count]

            print(f"Generating embeddings for {len(batch_docs)} chunks...")
            batch_embeddings = self.embedding_model.encode(batch_docs, show_progress_bar=False).tolist()
            
            self.collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"Successfully indexed {len(batch_docs)} chunks (Total: {self.chunk_counter})")
            
            self.documents = self.documents[take_count:]
            self.metadatas = self.metadatas[take_count:]
            self.ids = self.ids[take_count:]

    def index(self) -> None:
        """Main execution loop that walks the directory and orchestrates processing."""
        print(f"Scanning {self.codebase_dir} for Python and Markdown files...")
        
        ignore_dirs = {'.git', 'venv', 'env', '__pycache__', 'node_modules', 'build', 'dist', '.pytest_cache'}
        
        for root, dirs, files in os.walk(self.codebase_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ignore_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.py'):
                    self._process_file(file_path, 'python')
                elif file.endswith('.md'):
                    self._process_file(file_path, 'markdown')
                else:
                    continue
                
                # Check buffer and flush if we hit the batch limit
                if len(self.documents) >= self.batch_size:
                    self._flush_batch()

        # Flush any stragglers remaining after the loop finishes
        if self.documents:
            self._flush_batch(force_all=True)

        print(f"\nIndexing complete! Total chunks embedded: {self.chunk_counter}")

if __name__ == "__main__":
    indexer = CodebaseIndexer(
        codebase_dir="vllm-0.10.1",
        batch_size=1000
    )
    indexer.index()