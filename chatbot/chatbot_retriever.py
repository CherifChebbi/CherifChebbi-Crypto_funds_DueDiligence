# chatbot_retriever.py

import faiss
import numpy as np
import torch
import json
import logging
from pathlib import Path
from typing import List, Dict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re

# ---------------------------- CONFIGURATION ----------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
PARAPHRASE_MODEL_NAME = "google/flan-t5-base"
DIMENSION = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_K = 15
RERANK_K = 5
INDEX_DIR = "faiss_index"
BATCH_SIZE = 32

# ---------------------------- LOGGING ----------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------- CACHES MODELS ----------------------------

_embedding_tokenizer, _embedding_model = None, None
_cross_encoder = None
_paraphrase_tokenizer, _paraphrase_model = None, None


def load_embedding_model():
    global _embedding_tokenizer, _embedding_model
    if _embedding_tokenizer is None or _embedding_model is None:
        logger.info("📦 Chargement du modèle d'embedding...")
        _embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        _embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        _embedding_model.to(DEVICE).eval()
    return _embedding_tokenizer, _embedding_model


def load_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("📦 Chargement du cross-encoder...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=DEVICE)
    return _cross_encoder


def load_paraphrase_model():
    global _paraphrase_tokenizer, _paraphrase_model
    if _paraphrase_tokenizer is None or _paraphrase_model is None:
        logger.info("📦 Chargement du modèle de reformulation...")
        _paraphrase_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL_NAME)
        _paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL_NAME).to(DEVICE).eval()
    return _paraphrase_tokenizer, _paraphrase_model

# ---------------------------- UTILS ----------------------------

def encode_text(text: str, tokenizer, model) -> np.ndarray:
    inputs = tokenizer([text], truncation=True, padding=True, return_tensors="pt", max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return normalize(embedding)

def reformulate_query(query: str) -> str:
    try:
        tokenizer, model = load_paraphrase_model()
        prompt = f"Rephrase this question: {query}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        outputs = model.generate(**inputs, max_length=512, num_beams=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.warning(f"⚠️ Reformulation échouée : {e}")
        return query

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def build_bm25_corpus(metadata: List[Dict]) -> BM25Okapi:
    corpus = [tokenize(chunk["text"]) for chunk in metadata]
    return BM25Okapi(corpus)

def load_index_and_metadata(index_path: Path) -> tuple:
    index = faiss.read_index(str(index_path))
    metadata_path = index_path.with_suffix(".metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

# ---------------------------- RETRIEVAL ----------------------------

def hybrid_retrieve(query: str, index_path: Path, top_k: int = DEFAULT_K, rerank: bool = True,
                    alpha_dense: float = 0.5, alpha_sparse: float = 0.5) -> List[Dict]:
    logger.info(f"🔍 Requête reçue : {query}")

    tokenizer, model = load_embedding_model()
    cross_encoder = load_cross_encoder() if rerank else None

    query = reformulate_query(query)
    query_embedding = encode_text(query, tokenizer, model)

    index, metadata = load_index_and_metadata(index_path)

    if index.ntotal == 0:
        logger.warning(f"⚠️ Index vide : {index_path}")
        return []

    # Dense Search (FAISS)
    distances, indices = index.search(query_embedding, top_k)
    dense_results = [{"chunk": metadata[idx], "distance": float(dist)}
                     for dist, idx in zip(distances[0], indices[0]) if idx < len(metadata)]

    # Sparse Search (BM25)
    bm25 = build_bm25_corpus(metadata)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_sparse_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    sparse_results = [{"chunk": metadata[i], "bm25_score": float(bm25_scores[i])}
                      for i in top_sparse_indices if i < len(metadata)]

    # Fusion
    combined = {}
    for res in dense_results:
        cid = res["chunk"]["id"]
        combined[cid] = {"chunk": res["chunk"], "dense_score": 1.0 - res["distance"]}
    for res in sparse_results:
        cid = res["chunk"]["id"]
        if cid in combined:
            combined[cid]["sparse_score"] = res["bm25_score"]
        else:
            combined[cid] = {"chunk": res["chunk"], "sparse_score": res["bm25_score"]}

    # Score final
    results = []
    for val in combined.values():
        dense = val.get("dense_score", 0.0)
        sparse = val.get("sparse_score", 0.0)
        fusion_score = alpha_dense * dense + alpha_sparse * sparse
        val["fusion_score"] = fusion_score
        results.append(val)

    top_results = sorted(results, key=lambda x: x["fusion_score"], reverse=True)[:top_k]

    # Reranking (optional)
    if rerank:
        pairs = [(query, res["chunk"]["text"]) for res in top_results]
        scores = cross_encoder.predict(pairs, batch_size=BATCH_SIZE)
        for res, s in zip(top_results, scores):
            res["rerank_score"] = float(s)
        top_results = sorted(top_results, key=lambda x: x["rerank_score"], reverse=True)[:RERANK_K]

    logger.info(f"✅ {len(top_results)} résultats renvoyés.")
    return top_results

# ---------------------------- TEST ----------------------------

def main():
    test_query = "What is the fund's investment strategy?"
    index_dir = Path(INDEX_DIR)
    for index_path in index_dir.rglob("*.faiss"):
        logger.info(f"🧠 Recherche dans l'index : {index_path.name}")
        results = hybrid_retrieve(test_query, index_path)
        for i, res in enumerate(results):
            print(f"\nResult #{i+1}:")
            print(f"Section: {res['chunk']['metadata'].get('section', 'N/A')}")
            print(f"Text: {res['chunk']['text'][:300]}...")
            print(f"Fusion Score: {res.get('fusion_score'):.4f}, Rerank: {res.get('rerank_score', 'N/A')}")

if __name__ == "__main__":
    main()
