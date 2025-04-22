# chatbot_utils.py

import re

def clean_text(text: str) -> str:
    """
    Nettoie un texte brut (HTML, symboles, répétitions).
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()
