# chatbot_query_answering.py

import json
from pathlib import Path
from chatbot_llm import call_sambanova
from chatbot_retriever import hybrid_retrieve

# === CONFIGURATION ===
FAISS_DIR = Path("faiss_index")
DEFAULT_K = 5

def generate_prompt(context_chunks, question):
    context_text = "\n\n".join([chunk["chunk"]["text"] for chunk in context_chunks])
    return f"""
You are a due diligence assistant specializing in crypto investment funds.

Only answer based on the context provided below. If the answer is not in the context, say you don't know. Answer in English.

Context:
{context_text}

Question: {question}
"""

def get_latest_index_path():
    faiss_files = list(FAISS_DIR.rglob("*.faiss"))
    if not faiss_files:
        raise FileNotFoundError("No FAISS index found in the directory.")
    return max(faiss_files, key=lambda f: f.stat().st_mtime)

def answer_query_from_documents(question: str, k: int = DEFAULT_K, index_path: Path = None):
    if index_path is None:
        index_path = get_latest_index_path()

    top_chunks = hybrid_retrieve(query=question, index_path=index_path, top_k=k)

    if not top_chunks:
        return "No relevant information found in the uploaded documents.", []

    prompt = generate_prompt(top_chunks, question)
    answer = call_sambanova(prompt)

    sources = [res["chunk"]["metadata"]["source_file"] for res in top_chunks if "metadata" in res["chunk"]]
    return (answer.strip() if answer else "No answer could be generated."), list(set(sources))

