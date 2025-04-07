# QA_rag_pipeline.py

from QA_model import call_sambanova
from QA_vector_store import build_context_from_query
import time

def build_prompt(context: str, question: str) -> str:
    """
    Construit un prompt clair et structuré à partir du contexte et de la question.
    """
    prompt = f"""You are a due diligence assistant. Answer the user's question strictly based on the provided context.
If the context does not contain enough information, respond with: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""
    return prompt

def sliding_window_context(context: str, max_tokens: int = 512) -> str:
    """
    Découpe le contexte en segments plus petits si nécessaire pour éviter de dépasser la limite du modèle.
    """
    tokens = context.split()  # Séparer le contexte en tokens
    context_chunks = []
    while tokens:
        context_chunk = " ".join(tokens[:max_tokens])
        context_chunks.append(context_chunk)
        tokens = tokens[max_tokens:]  # Enlever les tokens déjà utilisés
    return "\n---\n".join(context_chunks)  # Retourner le contexte découpé

def answer_question_with_rag(question: str) -> str:
    """
    Effectue une réponse à la question via RAG : contexte + génération
    """
    print(f"\n❓ Question reçue : {question}")
    
    # Étape 1 : Retrieve context
    context = build_context_from_query(question)
    print("🔍 Contexte extrait des chunks FAISS.")

    # Étape 2 : Gérer le contexte si nécessaire
    context = sliding_window_context(context)
    print("✍️ Contexte découpé en morceaux si nécessaire.")

    # Étape 3 : Formate le prompt
    prompt = build_prompt(context, question)
    print("✍️ Prompt construit.")

    # Étape 4 : Génère la réponse avec SambaNova
    response = None
    retry_attempts = 5
    for attempt in range(retry_attempts):
        response = call_sambanova(prompt)
        if response:
            break
        else:
            print("⚠️ Erreur lors de l'appel à l'API. Tentative de nouvelle génération...")
            time.sleep(2 ** attempt)  # Exponentially increasing delay (ex: 2, 4, 8... seconds)

    if response:
        print("✅ Réponse générée.")
        return response.strip()
    else:
        print("⚠️ Erreur lors de la génération après plusieurs tentatives.")
        return "An error occurred while generating the answer."


def test_sambanova_integration():
    """
    Teste l'intégration avec l'API Sambanova.
    """
    test_prompt = "What is the capital of France?"
    print("🔄 Test de l'intégration avec Sambanova...")
    response = call_sambanova(test_prompt)
    print(f"Réponse générée : {response}")
    if not response:
        print("⚠️ Erreur avec Sambanova.")
    else:
        print("✅ Sambanova fonctionne correctement.")


# Test de l'intégration avec Sambanova
test_sambanova_integration()
