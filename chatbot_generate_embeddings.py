# chatbot_generate_embeddings.py

import os
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
import random

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
INPUT_DIR = "output"
OUTPUT_DIR = "output"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALIDATION_SIZE = 0.1

# Initialisation du modèle
_tokenizer = None
_model = None

def load_nomic_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info(f"Chargement du modèle {MODEL_NAME} sur {DEVICE}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model.to(DEVICE)
        _model.eval()
    return _tokenizer, _model

def generate_embeddings(texts, tokenizer, model, batch_size=BATCH_SIZE):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔄 Génération des embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        batch_embeddings = normalize(batch_embeddings)
        embeddings.append(batch_embeddings)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    embeddings = np.vstack(embeddings)
    logger.info(f"✅ Embeddings générés : {embeddings.shape}")
    return embeddings

def evaluate_embeddings(embeddings, chunks):
    metrics = {}

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Mismatch entre le nombre de chunks et d'embeddings!")

    norms = np.linalg.norm(embeddings, axis=1)
    metrics["avg_norm"] = float(np.mean(norms))
    metrics["std_norm"] = float(np.std(norms))

    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    metrics["avg_cosine_similarity"] = float(similarity_matrix.sum() / (len(embeddings) * (len(embeddings) - 1)))

    section_chunks = {}
    for i, chunk in enumerate(chunks):
        section = chunk["metadata"]["section"]
        section_chunks.setdefault(section, []).append(i)

    section_similarities = {}
    for section, indices in section_chunks.items():
        if len(indices) > 1:
            sec_embeddings = embeddings[indices]
            sec_sim_matrix = cosine_similarity(sec_embeddings)
            np.fill_diagonal(sec_sim_matrix, 0)
            section_similarities[section] = float(sec_sim_matrix.sum() / (len(indices) * (len(indices) - 1)))
    metrics["section_similarities"] = section_similarities

    inter_similarities = []
    sections = list(section_chunks.keys())
    for i in range(len(sections)):
        for j in range(i + 1, len(sections)):
            emb_i = embeddings[section_chunks[sections[i]]]
            emb_j = embeddings[section_chunks[sections[j]]]
            inter_similarities.append(cosine_similarity(emb_i, emb_j).mean())
    metrics["avg_inter_section_similarity"] = float(np.mean(inter_similarities)) if inter_similarities else 0.0

    logger.info(f"📊 Évaluation des embeddings : {json.dumps(metrics, indent=2)}")
    return metrics

def validate_embeddings(chunks, embeddings, tokenizer, model, sample_size=VALIDATION_SIZE):
    logger.info("🔍 Validation des embeddings sur un échantillon...")
    sample_size = max(2, int(len(chunks) * sample_size))
    sample_indices = random.sample(range(len(chunks)), sample_size)
    sample_texts = [chunks[i]["text"] for i in sample_indices]
    sample_embeddings = embeddings[sample_indices]

    regen_embeddings = generate_embeddings(sample_texts, tokenizer, model)
    stability = cosine_similarity(sample_embeddings, regen_embeddings).diagonal().mean()

    logger.info(f"✅ Stabilité moyenne des embeddings : {stability:.4f}")
    return {"stability_similarity": float(stability)}

def process_chunked_data(input_path, output_dir=OUTPUT_DIR, validate=True):
    logger.info(f"📂 Traitement du fichier : {input_path}")

    output_path = Path(output_dir) / input_path.parent.name / f"{input_path.stem}_with_embeddings.json"
    if output_path.exists():
        logger.info(f"⏭️ Fichier déjà enrichi détecté, on saute : {output_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        chunked_data = json.load(f)

    chunks = chunked_data.get("chunks", [])
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        logger.warning("⚠️ Aucun texte trouvé dans le fichier, skip.")
        return

    tokenizer, model = load_nomic_model()
    embeddings = generate_embeddings(texts, tokenizer, model)
    metrics = evaluate_embeddings(embeddings, chunks)

    if validate:
        validation = validate_embeddings(chunks, embeddings, tokenizer, model)
        metrics.update(validation)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    chunked_data["embedding_metrics"] = metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=4)

    logger.info(f"✅ Fichier sauvegardé : {output_path}")

def main():
    input_dir = Path(INPUT_DIR)
    chunk_files = list(input_dir.rglob("chunked_data.json"))

    if not chunk_files:
        logger.warning(f"⚠️ Aucun fichier chunked_data.json trouvé dans {INPUT_DIR}")
        return

    logger.info(f"🔎 {len(chunk_files)} fichiers à traiter...")
    for chunk_file in chunk_files:
        process_chunked_data(chunk_file, validate=False)  # Désactive la validation en batch

if __name__ == "__main__":
    main()
