import streamlit as st
import os
from pathlib import Path
import time
from QA_rag_pipeline import answer_question_with_rag
from generate_chunks import process_all_cleaned_texts
from generate_embeddings import generate_embeddings_for_new_files
import shutil

# Configuration de l'interface
st.set_page_config(page_title="🧠 PDF Q&A RAG with SambaNova", layout="wide")

# Dossier pour stocker les uploads
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dossier de sortie
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Titre de l'application
st.title("🧠 Q&A Intelligent sur vos fichiers PDF")
st.markdown("Posez vos questions et obtenez des réponses **fiables et précises** à partir de vos propres documents. 🚀")

# Upload de fichier(s) PDF
uploaded_files = st.file_uploader("📂 Importer un ou plusieurs fichiers PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # Enregistrement du fichier
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.success(f"✅ Fichier enregistré : {uploaded_file.name}")

    # Option pour lancer le pipeline de traitement uniquement pour les nouveaux fichiers
    if st.button("🔄 Lancer le pipeline d'extraction (chunks et embeddings) pour les nouveaux fichiers"):
        with st.spinner("⏳ Traitement en cours..."):
            try:
                # Étape 1 : Générer les embeddings et indexer dans FAISS uniquement pour les nouveaux fichiers
                generate_embeddings_for_new_files()
                st.success("✅ Embeddings générés et indexés dans FAISS pour les nouveaux fichiers.")

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement des fichiers : {str(e)}")

# Initialiser l'historique du chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Interface de chat
st.subheader("💬 Posez une question")

question = st.text_input("Votre question :", placeholder="Exemple : Quel est le nom du fonds ?")

if st.button("📤 Envoyer") and question:
    with st.spinner("🧠 Analyse en cours..."):
        response = answer_question_with_rag(question)
        time.sleep(0.5)

        st.session_state.chat_history.append(("🧑‍💼 Vous", question))
        st.session_state.chat_history.append(("🤖 Assistant", response))

# Affichage du chat
st.subheader("📜 Historique du chat")

for role, message in st.session_state.chat_history:
    with st.chat_message(role.split()[1] if " " in role else role):
        st.markdown(message)

# Option pour reset l'historique
if st.button("🧹 Réinitialiser le chat"):
    st.session_state.chat_history = []
    st.success("🗑️ Chat réinitialisé.")

# Option pour réinitialiser les chunks et embeddings
if st.button("🗑️ Réinitialiser les chunks et embeddings"):
    try:
        # Effacer les fichiers de sortie
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        st.success("✅ Chunks et embeddings réinitialisés.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la réinitialisation : {str(e)}")
