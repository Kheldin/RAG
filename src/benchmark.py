import os
import time
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer

codebase_dir = "vllm-0.10.1"

# 1. Walk and split benchmark (dry run)
start_time = time.time()
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=1000, 
    chunk_overlap=100
)
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
)

documents = []
metadatas = []
ids = []
chunk_counter = 0

print("Scanning files and splitting (no embeddings)...")
file_count = 0
for root, dirs, files in os.walk(codebase_dir):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith('.py') or file.endswith('.md'):
            file_count += 1
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if not content.strip():
                    continue
                if file.endswith('.py'):
                    for chunk in python_splitter.split_text(content):
                        documents.append(chunk)
                        metadatas.append({"source": file_path, "type": "python"})
                        ids.append(f"chunk_{chunk_counter}")
                        chunk_counter += 1
                elif file.endswith('.md'):
                    for doc in markdown_splitter.split_text(content):
                        documents.append(doc.page_content)
                        meta = {"source": file_path, "type": "markdown"}
                        meta.update(doc.metadata)
                        metadatas.append(meta)
                        ids.append(f"chunk_{chunk_counter}")
                        chunk_counter += 1
            except Exception as e:
                pass

print(f"Total files read/split: {file_count}")
print(f"Total chunks generated: {len(documents)}")
print(f"Time taken to scan & split: {time.time() - start_time:.2f} seconds")

# Let's take a sample of 2000 chunks to benchmark embedding speed
sample_docs = documents[:2000]
print(f"Benchmarking embedding on a sample of {len(sample_docs)} chunks...")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sequential encode
start_seq = time.time()
embeddings_seq = model.encode(sample_docs, batch_size=32, show_progress_bar=False)
seq_time = time.time() - start_seq
print(f"Sequential encode time (batch_size=32): {seq_time:.2f} seconds ({len(sample_docs)/seq_time:.1f} chunks/sec)")

# Sequential encode with larger batch size
start_seq_large = time.time()
embeddings_seq_large = model.encode(sample_docs, batch_size=256, show_progress_bar=False)
seq_large_time = time.time() - start_seq_large
print(f"Sequential encode time (batch_size=256): {seq_large_time:.2f} seconds ({len(sample_docs)/seq_large_time:.1f} chunks/sec)")

# Multi-process encode
try:
    print("Starting multi-process pool...")
    start_mp = time.time()
    pool = model.start_multi_process_pool()
    embeddings_mp = model.encode_multi_process(sample_docs, pool)
    model.stop_multi_process_pool(pool)
    mp_time = time.time() - start_mp
    print(f"Multi-process encode time: {mp_time:.2f} seconds ({len(sample_docs)/mp_time:.1f} chunks/sec)")
except Exception as e:
    print("Multi-process encode failed:", e)
