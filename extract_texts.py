# extract_texts.py

import os
import fitz  # PyMuPDF
from pathlib import Path
from parsers.text_parser import extract_text_pdfplumber
from parsers.ocr_parser import extract_text_ocr
from tqdm import tqdm

UPLOAD_DIR = "upload"
OUTPUT_DIR = "output"

def is_scanned_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        for page in doc:
            if page.get_text().strip():
                return False
    return True

def save_text_output(text, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"{filename}.txt")
    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump({
            "filename": filename,
            "num_chars": len(text),
            "preview": text[:500]
        }, f, indent=2)

def main():
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("Aucun fichier PDF trouvé dans le dossier 'upload/'")
        return

    for pdf_file in tqdm(pdf_files, desc="Traitement des PDFs"):
        pdf_path = os.path.join(UPLOAD_DIR, pdf_file)
        output_subdir = os.path.join(OUTPUT_DIR, Path(pdf_file).stem)

        try:
            if is_scanned_pdf(pdf_path):
                text = extract_text_ocr(pdf_path)
            else:
                text = extract_text_pdfplumber(pdf_path)

            save_text_output(text, output_subdir, Path(pdf_file).stem)
            print(f"[✓] {pdf_file} traité avec succès.")

        except Exception as e:
            print(f"[✗] Erreur lors du traitement de {pdf_file} : {e}")

if __name__ == "__main__":
    main()

