# chatbot_query_answering.py

import json
from pathlib import Path
import streamlit as st
from chatbot_llm import call_sambanova
from chatbot_retriever import hybrid_retrieve
from chatbot_graphrag_retriever import graphrag_retrieve
from chatbot_answer_review import review_answer_with_context
from chatbot_fund_detector import detect_fund_name_from_file

FAISS_DIR = Path("faiss_index")
OUTPUT_DIR = Path("output")
DEFAULT_K = 5


def generate_prompt(context_chunks, question, fund_name):
    context_text = "\n\n".join([chunk["chunk"]["text"] for chunk in context_chunks if "chunk" in chunk])
    return f"""
You are a Senior Due Diligence Analyst working on behalf of institutional investors evaluating the crypto investment fund **"{fund_name}"**.

You must:
- Provide an audit-ready answer strictly based on the context.
- Avoid all assumptions or speculation.
- Clearly indicate missing information required to answer.
- Write in a neutral, professional, and regulatory-compliant tone.

Use this structure:
1. ✅ Direct Answer — Clear and factual based only on context
2. 📌 Key Evidence — Bullet points quoting or paraphrasing context
3. ⚖️ Compliance — Mention licenses, filings, regulations (if applicable)
4. ⚠️ Missing Information — List gaps or critical unknowns
5. 📝 Conclusion — Summary indicating sufficiency of information

Context:
{context_text}

Question:
{question}
"""


def extract_missing_points(question, answer, context_chunks, fund_name):
    context_text = "\n\n".join([chunk["chunk"]["text"] for chunk in context_chunks if "chunk" in chunk])
    prompt = f"""
You are an expert crypto fund auditor conducting due diligence on the fund "{fund_name}".

Analyze the context, question, and assistant's answer, and identify critical missing data points that are necessary to produce a fully complete and professional due diligence answer.

Focus on key due diligence areas:
- Regulatory compliance and licensing
- Governance structure
- Risk management and disclosures
- Investor protections and verification
- Custody of crypto assets
- Valuation policies
- Conflicts of interest

Return ONLY a JSON array of concise missing points like:
[
    "No mention of regulatory license",
    "Custody provider is not specified",
    "Missing disclosure of fee structure"
]

Context:
{context_text}

Question:
{question}

Answer:
{answer}

MissingPoints:
"""
    try:
        response = call_sambanova(prompt)
        return json.loads(response)
    except Exception:
        return []


def get_latest_index_path():
    faiss_files = list(FAISS_DIR.rglob("*.faiss"))
    if not faiss_files:
        raise FileNotFoundError("No FAISS index found.")
    return max(faiss_files, key=lambda f: f.stat().st_mtime)


def detect_main_fund_name() -> str:
    all_texts = []
    for folder in OUTPUT_DIR.glob("*"):
        extracted_file = folder / "extracted_text.txt"
        if extracted_file.exists():
            with open(extracted_file, "r", encoding="utf-8") as f:
                all_texts.append(f.read())
    return detect_fund_name_from_file(Path(folder) / "extracted_text.txt") if all_texts else "the fund"


def answer_query_from_documents_debug(
    question: str,
    k: int = DEFAULT_K,
    index_path: Path = None,
    use_graph: bool = False
):
    if index_path is None:
        index_path = get_latest_index_path()

    progress = st.empty()
    progress.text("🔍 Step 0: Detecting fund name...")
    fund_name = detect_main_fund_name()

    progress.text("🔍 Step 1/4: Retrieving relevant content...")
    if use_graph:
        local_chunks = graphrag_retrieve(query=question, top_k=k)
        if not local_chunks:
            return {
                "answer": "❌ No relevant GraphRAG chunks found.",
                "sources": [],
                "review": {
                    "confidence_score": 0.0,
                    "hallucination_risk": "high",
                    "justification": "No content retrieved from the graph index.",
                    "verdict": "❌ Unsupported"
                },
                "missing_points": [],
                "fund_name": fund_name,
                "chunks_used": [],
                "debug_log": {}
            }
        for c in local_chunks:
            c["origin"] = "graph"
    else:
        local_chunks = hybrid_retrieve(query=question, index_path=index_path, top_k=k)
        for c in local_chunks:
            c["origin"] = "local"

    combined_chunks = local_chunks

    # Step 2: Generation
    progress.text("🧠 Step 2/4: Generating answer...")
    prompt = generate_prompt(combined_chunks, question, fund_name)
    answer = call_sambanova(prompt)
    final_answer = answer.strip() if answer else "No answer could be generated."

    # Step 3: Review
    progress.text("🧪 Step 3/4: Reviewing answer...")
    review = review_answer_with_context(
        question=question,
        answer=final_answer,
        context_chunks=combined_chunks
    )
    review["context_chunks"] = combined_chunks

    # Step 4: Detect missing points
    progress.text("🔍 Step 4/4: Detecting missing points...")
    missing_points = extract_missing_points(question, final_answer, combined_chunks, fund_name)

    # Sources
    sources = []
    for res in combined_chunks:
        if res.get("origin") in ["graph", "local"] and "metadata" in res.get("chunk", {}):
            sources.append(res["chunk"]["metadata"]["source_file"])

    return {
        "answer": final_answer,
        "sources": list(set(sources)),
        "review": review,
        "missing_points": missing_points,
        "fund_name": fund_name,
        "chunks_used": combined_chunks,
        "debug_log": {
            "fund_name": fund_name,
            "prompt": prompt,
            "chunks_local": local_chunks,
            "question_original": question,
        }
    }
