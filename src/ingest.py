import os
import bm25s

from typing import Any

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)


class CodebaseIndexer:
    """Scans a codebase, splits code/markdown into chunks, and indexes them using bm25s."""
    
    def __init__(
        self, 
        codebase_dir: str, 
        max_chunk_size: int = 1000,
        index_path: str = "data/processed/bm25_index",
    ):
        self.codebase_dir = codebase_dir
        self.index_path = index_path
        
        overlap_size = max(10, int(300))
        
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=max_chunk_size, 
            chunk_overlap=overlap_size
        )
        
        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
            strip_headers=False
        )

        self.markdown_text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=max_chunk_size,
            chunk_overlap=overlap_size
        )
        
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.ids: list[str] = []
        self.chunk_counter: int = 0

    def _process_file(self, file_path: str, file_type: str) -> None:
        clean_path = file_path.replace("\\", "/")
        prefixes = ["vllm-0.10.1/", "data/raw/vllm-0.10.1/"]
        for prefix in prefixes:
            if clean_path.startswith(prefix):
                clean_path = clean_path[len(prefix):]
                break

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            if not content.strip():
                return
                
            if file_type == 'python':
                for chunk in self.python_splitter.split_text(content):
                    enriched_chunk = f"--- File: {clean_path} ---\n{chunk}"
                    
                    self.documents.append(enriched_chunk)
                    self.metadatas.append({"source": clean_path, "type": "python"})
                    self.ids.append(f"chunk_{self.chunk_counter}")
                    self.chunk_counter += 1
            
            elif file_type == 'markdown':
                for chunk in self.markdown_text_splitter.split_text(content):
                    enriched_chunk = f"--- File: {clean_path} ---\n{chunk}"
                    
                    self.documents.append(enriched_chunk)
                    self.metadatas.append({"source": clean_path, "type": "markdown"})
                    self.ids.append(f"chunk_{self.chunk_counter}")
                    self.chunk_counter += 1
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def run_index(self) -> None:
        print(f"Scanning {self.codebase_dir} for Python and Markdown files...")
        
        ignore_dirs = {'.git', 'venv', 'env', '__pycache__', 'node_modules', 'build', 'dist', '.pytest_cache'}
        
        # 1. Accumulate all documents
        for root, dirs, files in os.walk(self.codebase_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ignore_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.py'):
                    self._process_file(file_path, 'python')
                elif file.endswith('.md'):
                    self._process_file(file_path, 'markdown')

        if not self.documents:
            print("No documents found to index.")
            return

        print(f"Found {self.chunk_counter} chunks. Tokenizing and building BM25 index...")

        # 2. Tokenize the corpus
        corpus_tokens = bm25s.tokenize(self.documents)

        # 3. Create the BM25 model and index the tokens
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # 4. Prepare corpus payload mapping docs, ids, and metadata
        corpus_records = [
            {"id": doc_id, "text": doc_text, "metadata": doc_meta}
            for doc_id, doc_text, doc_meta in zip(self.ids, self.documents, self.metadatas)
        ]

        # 5. Save the index and the corpus for future retrieval
        os.makedirs(self.index_path, exist_ok=True)
        retriever.save(self.index_path, corpus=corpus_records)

        print(f"\nIndexing complete! {self.chunk_counter} chunks successfully saved to '{self.index_path}'")