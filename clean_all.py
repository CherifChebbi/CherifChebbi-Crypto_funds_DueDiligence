# clean_all.py

import os
from pathlib import Path
from cleaning.clean_text import clean_text

OUTPUT_DIR = "output"

def process_all_texts():
    for subdir in os.listdir(OUTPUT_DIR):
        full_path = os.path.join(OUTPUT_DIR, subdir)
        if os.path.isdir(full_path):
            raw_txt = os.path.join(full_path, f"{subdir}.txt")
            if os.path.exists(raw_txt):
                with open(raw_txt, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                cleaned = clean_text(raw_text)

                cleaned_path = os.path.join(full_path, "cleaned.txt")
                with open(cleaned_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                print(f"[✓] {subdir} nettoyé → cleaned.txt généré")

if __name__ == "__main__":
    process_all_texts()
