import os
import re
import json
import phonenumbers
from pathlib import Path
from dateparser import search as date_search

OUTPUT_DIR = "output"

# === REGEX patterns ===
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
URL_REGEX = r"https?://[^\s]+"
CIK_REGEX = r"CIK[:\s]+(\d{10})"
MONEY_REGEX = r"\$ ?[\d,]+(?:\.\d+)?"
FUND_NAME_REGEX = r"\b[A-Z][a-zA-Z&\s]{2,}(Capital|Partners|Fund|Investments|LLC)\b"

def extract_phone_numbers(text, region="US"):
    """Extrait et valide les numéros de téléphone avec phonenumbers."""
    potential_phones = []
    for match in phonenumbers.PhoneNumberMatcher(text, region):
        if phonenumbers.is_valid_number(match.number):
            formatted = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            potential_phones.append(formatted)
    return list(set(potential_phones))

def extract_entities(text):
    print("🔍 Début de l'extraction des entités...")

    emails = re.findall(EMAIL_REGEX, text)
    urls = re.findall(URL_REGEX, text)
    ciks = re.findall(CIK_REGEX, text)
    phones = extract_phone_numbers(text)
    moneys = re.findall(MONEY_REGEX, text)
    fund_names = re.findall(FUND_NAME_REGEX, text)
    dates = [str(dt.date()) for _, dt in date_search.search_dates(text) or []]

    return {
        "emails": list(set(emails)),
        "urls": list(set(urls)),
        "ciks": list(set(ciks)),
        "phones": phones,
        "amounts": list(set(moneys)),
        "fund_names": list(set(fund_names)),
        "dates": list(set(dates)),
    }

def process_cleaned_files():
    for folder in os.listdir(OUTPUT_DIR):
        folder_path = os.path.join(OUTPUT_DIR, folder)
        cleaned_txt_path = os.path.join(folder_path, "cleaned.txt")

        if not os.path.exists(cleaned_txt_path):
            continue

        with open(cleaned_txt_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()

        entities = extract_entities(cleaned_text)

        with open(os.path.join(folder_path, "metadata_extracted.json"), "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=4, ensure_ascii=False)

        print(f"[✓] Entités extraites pour {folder} → metadata_extracted.json")

if __name__ == "__main__":
    process_cleaned_files()
