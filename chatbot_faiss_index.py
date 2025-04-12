# chatbot_faiss_index.py

import faiss
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
EMBEDDINGS_DIR = "output"
INDEX_DIR = "faiss_index"
DIMENSION = 768

def create_faiss_index(embeddings: np.ndarray, output_path: Path) -> faiss.Index:
    if embeddings.shape[1] != DIMENSION:
        raise ValueError(f"Dimension mismatch: attendu {DIMENSION}, reçu {embeddings.shape[1]}")

    index = faiss.IndexFlatL2(DIMENSION)
    index.add(embeddings)
    faiss.write_index(index, str(output_path))
    logger.info(f"✅ Index FAISS sauvegardé dans : {output_path}")
    return index

def load_embeddings_and_metadata(embeddings_path: Path) -> tuple[np.ndarray, List[Dict]]:
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        logger.warning(f"⚠️ Fichier sans chunk, ignoré : {embeddings_path}")
        return None, []

    embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
    return embeddings, chunks

def process_embeddings_file(embeddings_path: Path, index_dir: Path = Path(INDEX_DIR)):
    logger.info(f"📂 Traitement : {embeddings_path}")
    index_output_path = index_dir / embeddings_path.parent.name / f"{embeddings_path.stem}.faiss"
    metadata_output_path = index_output_path.with_suffix(".metadata.json")

    if index_output_path.exists() and metadata_output_path.exists():
        logger.info(f"⏭️ Index déjà présent, on saute : {index_output_path}")
        return

    embeddings, chunks = load_embeddings_and_metadata(embeddings_path)
    if embeddings is None or len(chunks) == 0:
        return

    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    index = create_faiss_index(embeddings, index_output_path)

    for i, chunk in enumerate(chunks):
        chunk["id"] = i

    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    logger.info(f"✅ Métadonnées sauvegardées : {metadata_output_path}")

    return index, chunks

def search_faiss_index(index: faiss.Index, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    return index.search(query_embedding, k)

def main():
    embeddings_dir = Path(EMBEDDINGS_DIR)
    embeddings_files = list(embeddings_dir.rglob("*_with_embeddings.json"))

    if not embeddings_files:
        logger.warning(f"❌ Aucun fichier *_with_embeddings.json trouvé dans {EMBEDDINGS_DIR}")
        return

    logger.info(f"🔍 {len(embeddings_files)} fichiers à indexer.")
    for embeddings_file in embeddings_files:
        result = process_embeddings_file(embeddings_file)
        if result is None:
            continue

        index, chunks = result
        if chunks:
            sample_query_embedding = np.array([chunks[0]["embedding"]], dtype=np.float32)
            distances, indices = search_faiss_index(index, sample_query_embedding)
            logger.info(f"🔎 Test recherche FAISS - Indices: {indices[0]}, Distances: {distances[0]}")

if __name__ == "__main__":
    main()
