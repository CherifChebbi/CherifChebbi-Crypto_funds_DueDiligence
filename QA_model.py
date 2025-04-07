# QA_model.py

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

# === Configuration Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Chargement des variables d'environnement ===
load_dotenv()
API_KEY = 'bfde6096-758b-46b8-a678-3d55567aa5e5'

if not API_KEY:
    logger.error("❌ Clé API SambaNova manquante dans .env")
    raise EnvironmentError("Ajoutez SAMBANOVA_API_KEY à votre fichier .env")

# === Initialisation du client OpenAI-like ===
client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=API_KEY
)

def call_sambanova(prompt: str,
                    system_prompt: str = "You are a helpful assistant that only responds based on the provided context.",
                    model: str = "Meta-Llama-3.1-70B-Instruct",
                    temperature: float = 0.3,
                    max_tokens: int = 512) -> Optional[str]:
    """
    Appel au modèle SambaNova avec gestion des erreurs
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"🚨 Erreur d'appel à l'API SambaNova : {e}")
        return None
