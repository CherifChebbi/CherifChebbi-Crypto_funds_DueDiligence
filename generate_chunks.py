# generate_chunks.py

import os
import json
from pathlib import Path

# Configuration des paramètres de découpage
CHUNK_SEPARATOR = "\n\n"
CHUNK_SIZE_WORDS = 120  # Taille approximative en mots
OUTPUT_DIR = "output"

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE_WORDS):
    """
    Divise le texte en chunks de taille définie (en nombre de mots).
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def process_all_cleaned_texts():
    """
    Parcourt tous les fichiers 'cleaned.txt' dans les dossiers de 'output' 
    et génère des chunks sous forme de fichier JSONL.
    """
    for folder in os.listdir(OUTPUT_DIR):
        folder_path = os.path.join(OUTPUT_DIR, folder)
        cleaned_txt_path = os.path.join(folder_path, "cleaned.txt")

        if not os.path.exists(cleaned_txt_path):
            print(f"⚠️ Aucun fichier 'cleaned.txt' trouvé dans {folder}. Ignoré.")
            continue

        with open(cleaned_txt_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()

        chunks = split_text_into_chunks(cleaned_text)

        # Sauvegarde des chunks dans un fichier JSONL
        jsonl_path = os.path.join(folder_path, "cleaned_chunks.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                entry = {
                    "text": chunk.strip(),
                    "source": folder,
                    "chunk_id": f"{folder}_chunk_{i+1}"
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[✓] {folder} découpé en {len(chunks)} chunks → cleaned_chunks.jsonl")

# === Exécution ===
if __name__ == "__main__":
    process_all_cleaned_texts()
