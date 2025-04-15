# chatbot_query_answering.py

import json
from pathlib import Path
import streamlit as st
from chatbot_llm import call_sambanova
from chatbot_retriever import hybrid_retrieve
from chatbot_web_scraper import get_web_chunks
from chatbot_web_filter import review_web_chunks
from chatbot_graphrag_retriever import graphrag_retrieve
from chatbot_answer_review import review_answer_with_context

FAISS_DIR = Path("faiss_index")
DEFAULT_K = 5

def generate_prompt(context_chunks, question):
    context_text = "\n\n".join([chunk["chunk"]["text"] for chunk in context_chunks if "chunk" in chunk])
    return f"""
You are a due diligence assistant specializing in crypto investment funds.

Only answer based on the context provided below. If the answer is not in the context, say you don't know. Answer in English.

Context:
{context_text}

Question: {question}
"""

def extract_missing_points(question: str, answer: str, context_chunks: list) -> list:
    context_text = "\n\n".join([chunk["chunk"]["text"] for chunk in context_chunks if "chunk" in chunk])

    prompt = f"""
You are an expert assistant. Given the context, question, and answer, identify which key points or facts are missing from the context that would allow a perfect and complete answer to the question.

Return the result strictly as a JSON array of strings (each string being a missing point).

Context:
{context_text}

Question:
{question}

Answer:
{answer}

MissingPoints:
"""

    response = call_sambanova(prompt)
    try:
        return json.loads(response)
    except Exception:
        return []

def get_latest_index_path():
    faiss_files = list(FAISS_DIR.rglob("*.faiss"))
    if not faiss_files:
        raise FileNotFoundError("No FAISS index found in the directory.")
    return max(faiss_files, key=lambda f: f.stat().st_mtime)


def answer_query_from_documents_debug(
    question: str,
    k: int = DEFAULT_K,
    index_path: Path = None,
    use_web: bool = True,
    use_graph: bool = False
):
    if index_path is None:
        index_path = get_latest_index_path()

    progress = st.empty()
    progress.text("🔍 Step 1/4: Retrieving relevant content...")

    # === 1. GraphRAG (strict)
    if use_graph:
        local_chunks = graphrag_retrieve(query=question, top_k=k)
        if not local_chunks:
            return {
                "answer": "❌ No relevant GraphRAG chunks found. Try uploading more content.",
                "sources": [],
                "review": {
                    "confidence_score": 0.0,
                    "hallucination_risk": "high",
                    "justification": "No chunks were returned by GraphRAG.",
                    "verdict": "❌ Unsupported"
                },
                "missing_points": []
            }
        for c in local_chunks:
            c["origin"] = "graph"
    else:
        local_chunks = hybrid_retrieve(query=question, index_path=index_path, top_k=k)
        for c in local_chunks:
            c["origin"] = "local"

    # === 2. Web augmentation
    if use_web:
        progress.text("🌐 Step 1.5: Retrieving from the web...")
        raw_web_chunks = get_web_chunks(question)
        reviewed_web_chunks = review_web_chunks(question, raw_web_chunks)
        web_chunks = [c for c in reviewed_web_chunks if c.get("relevance_score", 0) >= 0.3]
        for c in web_chunks:
            c["origin"] = "web"
        combined_chunks = local_chunks + web_chunks
    else:
        combined_chunks = local_chunks

    # === 3. Génération de réponse
    progress.text("🧠 Step 2/4: Generating answer with SambaNova...")
    prompt = generate_prompt(combined_chunks, question)
    answer = call_sambanova(prompt)
    final_answer = answer.strip() if answer else "No answer could be generated."

    # === 4. Évaluation intelligente
    progress.text("🧪 Step 3/4: Reviewing answer for reliability...")
    review = review_answer_with_context(
        question=question,
        answer=final_answer,
        context_chunks=combined_chunks
    )
    review["context_chunks"] = combined_chunks

    # === 5. Détection des informations manquantes
    progress.text("🔍 Step 4/4: Detecting missing points...")
    missing_points = extract_missing_points(question, final_answer, combined_chunks)

    # === Résumé final
    progress.text("✅ Displaying results")
    sources = []
    for res in combined_chunks:
        if res.get("origin") in ["graph", "local"] and "metadata" in res.get("chunk", {}):
            sources.append(res["chunk"]["metadata"]["source_file"])
        elif res.get("origin") == "web":
            sources.append(f"[WEB] {res.get('url')}")

    return {
        "answer": final_answer,
        "sources": list(set(sources)),
        "review": review,
        "missing_points": missing_points
    }
