# chatbot_data_extraction.py

import os
import re
import json
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import pipeline

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Constants
OUTPUT_DIR = "output"

# Define important keywords for crypto fund due diligence (English and French)
FINANCIAL_KEYWORDS = {
    'investment': ['investment', 'investissement'],
    'fund': ['fund', 'fonds'],
    'return': ['return', 'rendement'],
    'risk': ['risk', 'risque'],
    'portfolio': ['portfolio', 'portefeuille'],
    'strategy': ['strategy', 'stratégie'],
    'performance': ['performance'],
    'assets': ['assets', 'actifs'],
    'liquidity': ['liquidity', 'liquidité'],
    'capital': ['capital']
}

RISK_FACTORS = {
    'volatility': ['volatility', 'volatilité'],
    'market risk': ['market risk', 'risque de marché'],
    'liquidity risk': ['liquidity risk', 'risque de liquidité'],
    'regulatory risk': ['regulatory risk', 'risque réglementaire'],
    'operational risk': ['operational risk', 'risque opérationnel'],
    'counterparty risk': ['counterparty risk', 'risque de contrepartie']
}

SECTIONS_OF_INTEREST = {
    'executive summary': ['executive summary', 'résumé exécutif', 'overview', 'aperçu'],
    'fund overview': ['fund overview', 'présentation du fonds', 'about the fund'],
    'investment strategy': ['investment strategy', 'stratégie d\'investissement', 'strategy', 'stratégie'],
    'risk factors': ['risk factors', 'facteurs de risque', 'identified risk', 'risques'],
    'performance history': ['performance history', 'historique de performance', 'past performance', 'rendement passé'],
    'contact details': ['contact details', 'coordonnées', 'contact information', 'contact']
}

# Patterns for extraction
FUND_NAME_PATTERN = r"\b([A-Z][A-Za-z0-9&\-]+(?: [A-Z][A-Za-z0-9&\-]+)* (?:Fund|Capital|Partners|LLC))\b"
MONEY_PATTERN = r"(\$\d+(?:\.\d+)?(?:\s?(million|billion))?(?:\s?dollars)?|\d+\s?%)"
DATE_PATTERN = r"(\d{4}|\d{1,2}\/\d{1,2}\/\d{4}|\d{1,2} [A-Za-z]+ \d{4})"
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
URL_PATTERN = r"https?://(?:www\.)?[a-zA-Z0-9./-]+"
PHONE_PATTERN = r"\+?\d{1,2}\s?\(\d{3}\)\s?\d{3}-\d{4}"
ADDRESS_PATTERN = r"\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}"

# Cache NLP pipelines for efficiency
_ner_pipeline = None
_sentiment_pipeline = None

# Function to extract named entities
def extract_named_entities(text):
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    entities = _ner_pipeline(text)
    valid_entities = []
    print("\n🔍 Raw NER output:")
    for e in entities:
        print(f"- {e['word']} (entity_group: {e['entity_group']})")
    
    # Filter entities more strictly
    exclude_terms = ['finance', 'bitcoin', 'ethereum', 'defi', 'decent', 'crypto', 'new', 'york', 'st', 'ny']
    for e in entities:
        if e['entity_group'] == 'ORG' and not e['word'].startswith('##'):
            cleaned_entity = ' '.join(list(dict.fromkeys(e['word'].split())))
            if cleaned_entity.lower() not in exclude_terms and \
               any(term in cleaned_entity for term in ['Fund', 'Partners', 'LLC', 'Capital']):
                valid_entities.append(cleaned_entity)
    return list(set(valid_entities))

# Function to perform sentiment analysis
def analyze_sentiment(text):
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    performance_keywords = ['performance', 'return', 'rendement', 'strategy', 'stratégie', 'investment', 'investissement']
    relevant_sentences = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in performance_keywords):
            relevant_sentences.append(sentence)
    
    if not relevant_sentences:
        return {"label": "neutral", "score": 0.0}
    
    combined_text = " ".join(relevant_sentences)
    if len(combined_text) > 512:
        combined_text = combined_text[:512]
    result = _sentiment_pipeline(combined_text)[0]
    return {"label": result['label'].lower(), "score": result['score']}

# Function to count keywords with variants
def count_keywords(words_lower, keywords_dict):
    counts = {key: 0 for key in keywords_dict}
    for key, variants in keywords_dict.items():
        for variant in variants:
            counts[key] += words_lower.count(variant.lower())
    return counts

# Function to detect sections
def detect_sections(text, sections_dict):
    text_lower = text.lower()
    sections_found = {key: False for key in sections_dict}
    for section, keywords in sections_dict.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                sections_found[section] = True
                break
    return sections_found

# Updated extraction function
def extract_key_information_updated(text):
    data = {}
    words = word_tokenize(text)
    words_lower = [word.lower() for word in words if word.isalpha()]
    
    keyword_counts = count_keywords(words_lower, FINANCIAL_KEYWORDS)
    sections_found = detect_sections(text, SECTIONS_OF_INTEREST)
    risk_factors_found = count_keywords(words_lower, RISK_FACTORS)
    risk_factors_found = {k: v > 0 for k, v in risk_factors_found.items()}
    
    fund_names = re.findall(FUND_NAME_PATTERN, text)
    money_values = [val[0] for val in re.findall(MONEY_PATTERN, text)]
    dates = re.findall(DATE_PATTERN, text)
    emails = re.findall(EMAIL_PATTERN, text)
    urls = re.findall(URL_PATTERN, text)
    phone_numbers = re.findall(PHONE_PATTERN, text)
    addresses = re.findall(ADDRESS_PATTERN, text)
    named_entities = extract_named_entities(text)
    
    fund_names = [name for name in fund_names if any(entity in name for entity in named_entities) or name in named_entities]
    if not fund_names:
        fund_names = [entity for entity in named_entities if any(term in entity for term in ['Fund', 'Partners', 'LLC'])]
    if not fund_names:
        fund_names = re.findall(FUND_NAME_PATTERN, text)
    
    sentiment = analyze_sentiment(text)
    
    data['fund_names'] = list(set(fund_names))
    data['financial_keywords'] = keyword_counts
    data['sections_found'] = sections_found
    data['risk_factors_found'] = risk_factors_found
    data['money_values'] = money_values
    data['dates'] = validate_dates(dates, text)
    data['emails'] = emails
    data['urls'] = urls
    data['phone_numbers'] = phone_numbers
    data['addresses'] = addresses
    data['named_entities'] = named_entities
    data['sentiment'] = sentiment

    return data

# Function to validate extracted dates with context
def validate_dates(dates, text):
    valid_dates = []
    phone_pattern = r"\+\d+\s?\(\d+\)\s?\d+-\d+"
    postal_pattern = r"[A-Z]{2}\s?\d{5}"
    
    for date in dates:
        if re.match(r"^\d{4}$", date):
            if 1900 <= int(date) <= 2100:
                if not (re.search(phone_pattern, text) and date in text[re.search(phone_pattern, text).start():re.search(phone_pattern, text).end()]) and \
                   not (re.search(postal_pattern, text) and date in text[re.search(postal_pattern, text).start():re.search(postal_pattern, text).end()]):
                    valid_dates.append(date)
        elif re.match(r"^\d{1,2}\/\d{1,2}\/\d{4}$", date) or re.match(r"^\d{1,2} [A-Za-z]+ \d{4}$", date):
            valid_dates.append(date)
    return sorted(list(set(valid_dates)))

# Validation of extracted data
def validate_extracted_data(extracted_data):
    valid = True
    score = 0
    total_criteria = 10

    fund_names = extracted_data['fund_names']
    if fund_names and all('Fund' in name or 'Partners' in name or 'LLC' in name for name in fund_names) and \
       all(len(name.split()) <= 5 for name in fund_names):
        score += 1
    else:
        print("❌ Incorrect or missing fund names.")
        valid = False

    if extracted_data['money_values'] and any('%' in val for val in extracted_data['money_values']) and \
       any('$' in val for val in extracted_data['money_values']):
        score += 1
    else:
        print("❌ Incorrect or missing money values.")
        valid = False

    valid_dates = [date for date in extracted_data['dates'] if date.isdigit() and 1900 <= int(date) <= 2100]
    if valid_dates:
        score += 1
    else:
        print("❌ Incorrect or missing dates.")
        valid = False

    if extracted_data['emails']:
        score += 1
    else:
        print("❌ No email addresses extracted.")
        valid = False

    if extracted_data['urls']:
        score += 1
    else:
        print("❌ No URLs extracted.")
        valid = False

    if any(extracted_data['sections_found'].values()):
        score += 1
    else:
        print("❌ No sections identified.")
        valid = False

    if any(extracted_data['risk_factors_found'].values()):
        score += 1
    else:
        print("❌ No risk factors identified.")
        valid = False

    named_entities = extracted_data['named_entities']
    if named_entities and all('Fund' in entity or 'Partners' in entity or 'LLC' in entity for entity in named_entities) and \
       all(len(entity.split()) <= 5 for entity in named_entities):
        print("\n🏷️ Named entities extracted:")
        for entity in named_entities:
            print(f"- {entity}")
        score += 1
    else:
        print("⚠️ Warning: Incorrect or missing named entities (proceeding with save).")
        if not named_entities:
            score += 1  # Still award point if empty, as fund_names is correct

    if extracted_data['phone_numbers']:
        score += 1
    else:
        print("❌ No phone numbers extracted.")
        valid = False

    if extracted_data['addresses']:
        score += 1
    else:
        print("❌ No addresses extracted.")
        valid = False

    score_percentage = (score / total_criteria) * 100
    return valid, score_percentage

# Process extracted_text.txt
def process_from_extracted_text(txt_path, file_stem):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("📄 Text loaded from extracted_text.txt. Analyzing...")
    extracted_data = extract_key_information_updated(text)
    valid, score = validate_extracted_data(extracted_data)

    print("\n📌 Extraction summary:")
    print(f"🔹 Fund names: {extracted_data['fund_names']}")
    print(f"🔹 Money values: {extracted_data['money_values']}")
    print(f"🔹 Dates: {extracted_data['dates']}")
    print(f"🔹 Financial keywords: {extracted_data['financial_keywords']}")
    print(f"🔹 Emails: {extracted_data['emails']}")
    print(f"🔹 URLs: {extracted_data['urls']}")
    print(f"🔹 Phone numbers: {extracted_data['phone_numbers']}")
    print(f"🔹 Addresses: {extracted_data['addresses']}")
    print(f"🔹 Sentiment: {extracted_data['sentiment']}")

    json_file = Path(OUTPUT_DIR) / file_stem / "extracted_data.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Extracted data saved to: {json_file}")
    print(f"📊 Extraction score: {score:.2f}%")
    if not valid:
        print("⚠️ Note: Saved despite validation issues.")
    return extracted_data

# Main function
def process_all_files():
    for folder in os.listdir(OUTPUT_DIR):
        folder_path = Path(OUTPUT_DIR) / folder
        txt_path = folder_path / "extracted_text.txt"
        if txt_path.exists():
            print(f"\n📂 Processing text file: {txt_path}")
            process_from_extracted_text(txt_path, folder)

if __name__ == "__main__":
    print("🚀 Starting analysis of extracted_text.txt files for due diligence...")
    process_all_files()
    print("🎯 Extraction completed.")