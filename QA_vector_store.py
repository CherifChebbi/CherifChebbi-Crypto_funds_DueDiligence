# QA_vector_store.py

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# === Chargement des variables d'environnement ===
load_dotenv()
hf_token = os.getenv("huggingface_token")

# === Chemins vers les fichiers FAISS ===
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_DIM = 768
TOP_K = 5  # Nombre de passages les plus pertinents à retourner

def check_faiss_index(index, metadata):
    """
    Vérifie si l'index FAISS contient des éléments et si les métadonnées sont valides.
    """
    # Vérification de la taille de l'index
    print(f"🔍 Taille de l'index FAISS : {index.ntotal}")
    print(f"🔍 Taille de la metadata : {len(metadata)}")
    
    if index.ntotal == 0:
        print("⚠️ L'index FAISS est vide. Veuillez vérifier la création de l'index.")
    if len(metadata) == 0:
        print("⚠️ Aucune donnée de métadonnées trouvée. Vérifiez la génération des embeddings.")

def test_embedding_generation(model):
    """
    Teste la génération des embeddings avec un exemple simple.
    """
    sample_query = "Test embedding"
    embedding = model.encode(sample_query)
    print(f"🔄 Embedding pour '{sample_query}': {embedding[:10]}...")  # Afficher les 10 premiers éléments de l'embedding
    if not embedding.any():
        print("⚠️ Problème avec la génération des embeddings.")
    else:
        print("✅ Embedding généré avec succès.")




# === Chargement du modèle d'embedding ===
print("🔄 Chargement du modèle d'embedding...")
model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, use_auth_token=hf_token)


# Test de la génération des embeddings
test_embedding_generation(model)

# === Chargement de l'index FAISS ===
print("📥 Chargement de l’index FAISS...")
index = faiss.read_index(INDEX_PATH)

# === Chargement des métadonnées associées ===
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"✅ Index et métadonnées chargés ({len(metadata)} chunks).")

# Test de l'index FAISS
check_faiss_index(index, metadata)

def search_similar_chunks(query, k=5):
    """
    Recherche les k chunks les plus similaires dans FAISS.
    """
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding])
    
    # Effectuer la recherche dans FAISS
    _, indices = index.search(query_embedding, k)
    
    # Déboguer les indices retournés
    print(f"Indices retournés par FAISS : {indices}")
    print(f"Taille de la metadata : {len(metadata)}")
    
    similar_chunks = []
    for idx in indices[0]:
        # Ajouter un contrôle pour s'assurer que l'indice est valide
        if idx >= 0 and idx < len(metadata):
            similar_chunks.append(metadata[idx])
        else:
            print(f"⚠️ L'index {idx} est hors de portée de la metadata. Ignoré.")
    
    return similar_chunks


def build_context_from_query(query: str, k: int = TOP_K) -> str:
    """
    Construit un contexte concaténé à partir des meilleurs chunks
    """
    chunks = search_similar_chunks(query, k)
    context = "\n---\n".join([c["text"] for c in chunks])
    return context

