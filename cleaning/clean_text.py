import re
import unicodedata

def clean_text(text):
    # Normalisation Unicode
    text = unicodedata.normalize("NFKC", text)

    # Autoriser les caractères importants comme @ et / pour les emails/URLs
    text = re.sub(r"[^\w\s.,:%€$@/-]", " ", text)

    # Convertir en minuscules
    text = text.lower()

    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)

    # Supprimer les lignes vides
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.strip()
