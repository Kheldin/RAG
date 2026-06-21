import os
from typing import List, Dict, Any
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

# Import LangChain's intelligent structural splitters
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

# 1. Connect to ChromaDB
chroma_client: ClientAPI = chromadb.PersistentClient(path="./my_local_chromadb")
collection: Collection = chroma_client.get_or_create_collection(name="codebase_chunks")

# 2. Configuration
CODEBASE_DIR = "vllm-0.10.1"
BATCH_SIZE = 1000

# This specifically looks for classes and functions before falling back to character counts.
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=1500, 
    chunk_overlap=300
)

# This splits the document every time it sees a #, ##, or ###.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 3. Read and Prepare Files
documents: List[str] = []
metadatas: List[Dict[str, Any]] = []
ids: List[str] = []
chunk_counter = 0

print(f"Scanning {CODEBASE_DIR} for Python and Markdown files...")

for root, dirs, files in os.walk(CODEBASE_DIR):
    # Ignore hidden folders like .git, .venv, .vscode
    dirs[:] = [d for d in dirs if not d.startswith('.')]

    for file in files:
        if file.startswith('.'):
            continue
            
        file_path = os.path.join(root, file)
        
        # --- Handle Python Files ---
        if file.endswith(".py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split using Python syntax awareness
                chunks = python_splitter.create_documents([content])
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk.page_content)
                    metadatas.append({"file_path": file_path, "file_type": "python", "chunk_index": i})
                    ids.append(f"{file_path}_chunk_{i}")
                    chunk_counter += 1
            except Exception as e:
                print(f"Could not read {file_path}: {e}")

        # --- Handle Markdown Files ---
        elif file.endswith(".md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split based on markdown headers
                chunks = markdown_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk.page_content)
                    
                    # The brilliant thing about the Markdown splitter is that it saves 
                    # the headers it found directly into the chunk's metadata!
                    meta = {"file_path": file_path, "file_type": "markdown", "chunk_index": i}
                    meta.update(chunk.metadata) # Merges {"Header 2": "Installation", etc.}
                    
                    metadatas.append(meta)
                    ids.append(f"{file_path}_chunk_{i}")
                    chunk_counter += 1
            except Exception as e:
                print(f"Could not read {file_path}: {e}")

# 4. Batch Insert into ChromaDB
print(f"Found {chunk_counter} structural chunks. Injecting into ChromaDB in batches of {BATCH_SIZE}...")

for i in range(0, len(documents), BATCH_SIZE):
    collection.add(
        documents=documents[i:i + BATCH_SIZE],
        metadatas=metadatas[i:i + BATCH_SIZE],
        ids=ids[i:i + BATCH_SIZE]
    )
    print(f"Inserted batch {(i // BATCH_SIZE) + 1} of {(len(documents) // BATCH_SIZE) + 1}...")

print("Ingestion complete! Your codebase is logically chunked and ready.")