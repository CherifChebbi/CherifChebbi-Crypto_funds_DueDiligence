import streamlit as st
from QA_vector_store import check_faiss_index, test_embedding_generation
from QA_rag_pipeline import test_sambanova_integration
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import faiss
import json

# === Chargement des variables d'environnement ===
load_dotenv()
hf_token = os.getenv("huggingface_token")

# === Constantes ===
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"

# === Interface Streamlit ===
st.title("🧪 Test de l'application")
st.write("Cette application teste l'intégration avec FAISS, le modèle d'embedding, et SambaNova.")

# === Bouton : Test de l'index FAISS ===
if st.button("✅ Tester l'Index FAISS"):
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        st.success("✅ Index et métadonnées chargés avec succès.")
        check_faiss_index(index, metadata)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de FAISS : {e}")

# === Bouton : Test de génération d'embeddings ===
if st.button("🔍 Tester Génération des Embeddings"):
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, use_auth_token=hf_token)
        test_embedding_generation(model)
        st.success("✅ Embedding généré avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des embeddings : {e}")

# === Bouton : Test de SambaNova ===
if st.button("🤖 Tester Intégration SambaNova"):
    try:
        test_sambanova_integration()
        st.success("✅ Intégration avec SambaNova réussie.")
    except Exception as e:
        st.error(f"❌ Erreur lors du test SambaNova : {e}")
