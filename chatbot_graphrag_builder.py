# chatbot_graphrag_builder.py
import json
import os
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_chunks_with_embeddings(folder_path: Path) -> list:
    """Charge tous les fichiers *_with_embeddings.json et extrait les chunks."""
    all_chunks = []
    for path in folder_path.rglob("*_with_embeddings.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for chunk in data["chunks"]:
                chunk["embedding"] = np.array(chunk["embedding"], dtype=np.float32)
                chunk["id"] = chunk["metadata"].get("chunk_id", f"{path.stem}_{len(all_chunks)}")
                all_chunks.append(chunk)
    return all_chunks


def build_similarity_graph(chunks: list, similarity_threshold: float = 0.75) -> nx.Graph:
    """Construit un graphe basé sur la similarité cosine entre embeddings."""
    G = nx.Graph()

    for chunk in chunks:
        G.add_node(chunk["id"], **chunk)

    embeddings = np.stack([chunk["embedding"] for chunk in chunks])
    similarities = cosine_similarity(embeddings)

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            score = similarities[i, j]
            if score >= similarity_threshold:
                G.add_edge(chunks[i]["id"], chunks[j]["id"], weight=score)

    return G


def save_graph(graph: nx.Graph, output_path: Path):
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"✅ Graph saved to: {output_path}")


def build_graph_from_embeddings(input_folder="output", output_graph="graph_index/graph.gpickle", threshold=0.75):
    input_path = Path(input_folder)
    output_path = Path(output_graph)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("📥 Loading chunks...")
    chunks = load_chunks_with_embeddings(input_path)

    print(f"🔗 Building similarity graph (threshold={threshold})...")
    graph = build_similarity_graph(chunks, similarity_threshold=threshold)

    print(f"💾 Saving graph to {output_path}")
    save_graph(graph, output_path)


if __name__ == "__main__":
    build_graph_from_embeddings()
