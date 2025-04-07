# generate_embeddings.py

import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# === Mode debug (affiche les embeddings FAISS) ===
DEBUG = False

# === Chargement des variables d'environnement ===
load_dotenv()
hf_token = os.getenv("huggingface_token")
if hf_token is None:
    print("❌ Hugging Face token not found in .env file.")
    exit(1)
print("✅ Hugging Face token loaded.")

# === Configuration ===
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
OUTPUT_DIR = Path("output")
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBEDDING_DIM = 768

# === Chargement du modèle ===
print("🔄 Chargement du modèle d'embedding...")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, use_auth_token=hf_token)
print(f"✅ Modèle chargé : {MODEL_NAME}")

# === Chargement ou création de l'index FAISS ===
if Path(INDEX_PATH).exists() and Path(METADATA_PATH).exists():
    print("📥 Chargement de l’index FAISS existant...")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Index existant chargé avec {index.ntotal} vecteurs.")
else:
    print("⚙️ Aucun index existant. Création d’un nouvel index FAISS...")
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
    metadata = []

def generate_embeddings_for_new_files():
    """
    Génère des embeddings pour tous les fichiers 'cleaned_chunks.jsonl' non traités.
    Ajoute les embeddings à l'index FAISS et met à jour les métadonnées.
    """
    doc_folders = list(OUTPUT_DIR.iterdir())
    print(f"\n📁 {len(doc_folders)} documents à traiter...\n")

    for doc_folder in tqdm(doc_folders, desc="📚 Traitement des documents"):
        chunk_path = doc_folder / "cleaned_chunks.jsonl"

        if not chunk_path.exists():
            print(f"⚠️ Aucun 'cleaned_chunks.jsonl' pour {doc_folder.name}, ignoré.")
            continue

        # Vérifie si ce document est déjà dans les métadonnées
        if any(m["doc_name"] == doc_folder.name for m in metadata):
            print(f"⚠️ {doc_folder.name} déjà présent dans l'index. Ignoré.")
            continue

        print(f"🔍 Traitement de {doc_folder.name}...")

        with open(chunk_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"🧠 Embedding {doc_folder.name}", leave=False):
            data = json.loads(line)
            chunk_text = data["text"]
            chunk_id = data["chunk_id"]

            # Générer l'embedding
            try:
                embedding = model.encode(chunk_text)
                embedding_np = np.array([embedding])
                faiss.normalize_L2(embedding_np)
                index.add(embedding_np)
            except Exception as e:
                print(f"❌ Erreur lors de la génération de l'embedding pour {chunk_id}: {e}")
                continue

            # Mettre à jour les métadonnées
            metadata.append({
                "doc_name": doc_folder.name,
                "full_doc_path": str(doc_folder.resolve()),
                "chunk_id": chunk_id,
                "text": chunk_text[:100]  # Affiche les 100 premiers caractères du texte pour le log
            })

    # Sauvegarde
    print(f"\n💾 Sauvegarde des résultats...")
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ Index FAISS sauvegardé → {INDEX_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Métadonnées sauvegardées → {METADATA_PATH}")

    print("\n🎉 Embeddings générés avec succès !")

    # Debug : Afficher les 3 premiers vecteurs de FAISS
    if DEBUG:
        print("\n🧪 DEBUG : premiers vecteurs FAISS :")
        vectors = index.reconstruct_n(0, min(3, index.ntotal))
        for i, vec in enumerate(vectors):
            print(f"➡️ Vector {i} : {vec[:10]}...")  # Affiche les 10 premières valeurs

print(index.ntotal)

# === Exécution ===
if __name__ == "__main__":
    generate_embeddings_for_new_files()
