# chatbot_graphrag_retriever.py

import json
import pickle
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import torch
import os
from chatbot_graphrag_builder import build_graph_from_embeddings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

_tokenizer = None
_model = None


def load_embedding_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        _model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(DEVICE).eval()
    return _tokenizer, _model


def embed_text(text: str) -> np.ndarray:
    tokenizer, model = load_embedding_model()
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


def graphrag_retrieve(
    query: str,
    graph_path: str = "graph_index/graph.gpickle",
    top_k: int = 8,
    max_hops: int = 2
) -> List[Dict]:
    print("📡 Loading GraphRAG index...")

    if not os.path.exists(graph_path):
        print("⚠️ Graph not found. Auto-generating it...")
        build_graph_from_embeddings()

    with open(graph_path, "rb") as f:
        G: nx.Graph = pickle.load(f)

    print("📍 Embedding user query...")
    query_embedding = embed_text(query)

    node_embeddings = []
    node_ids = []

    for node_id, data in G.nodes(data=True):
        if "embedding" in data:
            node_embeddings.append(np.array(data["embedding"]))
            node_ids.append(node_id)

    if not node_embeddings:
        print("⚠️ No embeddings found in graph nodes.")
        return []

    node_embeddings = np.stack(node_embeddings)
    sims = cosine_similarity(query_embedding, node_embeddings)[0]
    best_idx = np.argmax(sims)
    best_node = node_ids[best_idx]

    print(f"🚀 Starting traversal from node: {best_node}")
    subgraph_nodes = nx.single_source_shortest_path_length(G, best_node, cutoff=max_hops)
    selected = sorted(subgraph_nodes.items(), key=lambda x: x[1])[:top_k]
    selected_ids = [node for node, _ in selected]

    chunks = []
    for n in selected_ids:
        node = G.nodes[n]
        if "text" in node:
            chunks.append({
                "chunk": {
                    "text": node["text"],
                    "metadata": node.get("metadata", {})
                },
                "origin": "graph"
            })

    if not chunks:
        print("⚠️ No chunks returned from GraphRAG. Verify node connectivity or embedding similarity.")

    return chunks
