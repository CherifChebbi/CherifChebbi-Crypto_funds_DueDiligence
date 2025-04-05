import streamlit as st
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# === Configuration ===
MODEL_NAME = "llama2"  # Ou "mistral" selon ton choix
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_METADATA_PATH = "faiss_metadata.json"
SIMILARITY_THRESHOLD = 0.7  # Seuil de similarité pour déclencher la recherche web
WEB_SEARCH_TRIGGER = False  # Si False, pas de recherche web (peut être activé selon besoin)

# === Chargement du modèle d'embedding pour la recherche ===
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1")
index = faiss.read_index(FAISS_INDEX_PATH)

# Chargement des métadonnées associées
with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Fonction pour récupérer les embeddings d'une question
def get_question_embedding(question):
    return embedding_model.encode(question)

# Fonction pour rechercher dans FAISS
def search_in_faiss(question_embedding):
    question_embedding_np = np.array([question_embedding])
    distances, indices = index.search(question_embedding_np, k=5)  # 5 meilleurs résultats
    return distances, indices

# Fonction pour générer des réponses avec RAG
def generate_answer_with_rag(question):
    question_embedding = get_question_embedding(question)

    # Recherche dans l'index FAISS
    distances, indices = search_in_faiss(question_embedding)

    # Vérifier si les résultats sont suffisamment proches
    if np.min(distances) < SIMILARITY_THRESHOLD:
        print(f"💡 Similarité insuffisante, activation de la recherche web...")
        # Optionnel : Mettre en place une recherche web si nécessaire
        if WEB_SEARCH_TRIGGER:
            # Code pour récupérer les informations via une recherche web (à ajouter ici)
            pass
        return "Désolé, je n'ai pas trouvé d'informations suffisantes dans les documents."

    # Sinon, générer une réponse avec les documents les plus pertinents
    relevant_texts = [metadata[i]["text"] for i in indices[0]]
    context = " ".join(relevant_texts)
    prompt = f"Voici quelques informations utiles pour répondre à la question :\n{context}\n\nRéponds à cette question : {question}"

    # Appel à Ollama pour générer la réponse
    response = ollama.chat(model=MODEL_NAME, messages=[
        {"role": "user", "content": prompt}
    ])
    
    return response["message"]["content"]

# === Streamlit Interface ===
st.title("Due Diligence Chatbot - Actifs Numériques")

# Historique des chats
if "history" not in st.session_state:
    st.session_state.history = []

# Fonction pour afficher l'historique des conversations
def display_history():
    for entry in st.session_state.history:
        st.chat_message("user").markdown(entry["question"])
        st.chat_message("assistant").markdown(entry["answer"])

# Entrée de la question
question = st.text_input("Posez votre question ici :")

# Bouton pour soumettre la question
if st.button("Envoyer"):
    if question:
        # Ajouter la question à l'historique
        st.session_state.history.append({"question": question, "answer": "..."})
        display_history()  # Afficher l'historique des messages
        with st.spinner("Traitement de la question..."):
            # Générer la réponse en utilisant le modèle
            answer = generate_answer_with_rag(question)
            st.session_state.history[-1]["answer"] = answer  # Mettre à jour la réponse
            display_history()  # Réafficher l'historique mis à jour
    else:
        st.warning("Veuillez poser une question.")

# Bouton pour réinitialiser l'historique
if st.button("Effacer l'historique des chats"):
    st.session_state.history = []
    st.experimental_rerun()

