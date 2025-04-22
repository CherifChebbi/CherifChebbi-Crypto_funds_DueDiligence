# chatbot_generate_chunks.py

import os
import json
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import textstat
import numpy as np

# Téléchargement des ressources NLTK
nltk.download('punkt')

# Constantes
INPUT_DIR = "output"
OUTPUT_DIR = "output"
SIMILARITY_THRESHOLD = 0.7
MAX_SENTENCES_PER_CHUNK = 7
MAX_CHARS_PER_CHUNK = 1500

# Initialisation globale du modèle
_model = None
def get_sentence_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Modèle Sentence-BERT chargé.")
    return _model

def split_into_sections(text, sections_found):
    section_patterns = {
        "investment strategy": r"Investment Strategy:|Investment Approach:",
        "contact details": r"Contact Information:|Contact Details:",
        "risk factors": r"Risk Factors:|Identified Risk:",
        "performance history": r"Performance History:|Past Performance:"
    }

    active_sections = {key: pattern for key, pattern in section_patterns.items() if sections_found.get(key, False)}
    sections = []
    remaining_text = text.strip()

    while remaining_text:
        earliest_match = None
        earliest_key = None
        earliest_pos = len(remaining_text)

        for key, pattern in active_sections.items():
            match = re.search(pattern, remaining_text)
            if match and match.start() < earliest_pos:
                earliest_match = match
                earliest_key = key
                earliest_pos = match.start()

        if earliest_match:
            if earliest_pos > 0:
                pre_section = remaining_text[:earliest_pos].strip()
                if pre_section:
                    sections.append(("misc", pre_section))

            section_start = earliest_pos
            section_end = re.search(r"|".join(active_sections.values()), remaining_text[section_start + 1:]) or len(remaining_text)
            if isinstance(section_end, re.Match):
                section_end = section_end.start() + section_start + 1
            else:
                section_end = len(remaining_text)

            section_content = remaining_text[section_start:section_end].strip()
            if section_content:
                sections.append((earliest_key, section_content))
            remaining_text = remaining_text[section_end:].strip()
        else:
            if remaining_text:
                sections.append(("misc", remaining_text))
            break

    return sections

def semantic_chunking(section_text, similarity_threshold=SIMILARITY_THRESHOLD, 
                      max_sentences=MAX_SENTENCES_PER_CHUNK, max_chars=MAX_CHARS_PER_CHUNK):
    sentences = sent_tokenize(section_text)
    if not sentences:
        return [{"text": section_text, "sentence_count": 0, "char_count": len(section_text)}]

    model = get_sentence_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current_chunk = []
    current_embeddings = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        sentence_embedding = embeddings[i]

        if not current_chunk:
            current_chunk.append(sentence)
            current_embeddings.append(sentence_embedding)
            current_length = len(sentence)
            continue

        chunk_embedding_avg = np.mean(current_embeddings, axis=0)
        similarity = cosine_similarity([chunk_embedding_avg], [sentence_embedding])[0][0]

        potential_length = current_length + len(sentence) + 1
        within_limits = len(current_chunk) < max_sentences and potential_length <= max_chars

        if similarity >= similarity_threshold and within_limits:
            current_chunk.append(sentence)
            current_embeddings.append(sentence_embedding)
            current_length = potential_length
        else:
            chunks.append({
                "text": " ".join(current_chunk),
                "sentence_count": len(current_chunk),
                "char_count": current_length
            })
            current_chunk = [sentence]
            current_embeddings = [sentence_embedding]
            current_length = len(sentence)

    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "sentence_count": len(current_chunk),
            "char_count": current_length
        })

    return chunks

def evaluate_chunk_quality(chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]
    model = get_sentence_model()
    embeddings = model.encode(chunk_texts, show_progress_bar=False)
    similarity_matrix = cosine_similarity(embeddings)
    avg_similarity = similarity_matrix.sum() / (len(chunks) * len(chunks) - len(chunks))
    return avg_similarity

def check_chunk_size(chunks):
    for chunk in chunks:
        sentence_count = chunk.get("sentence_count", len(sent_tokenize(chunk["text"])))
        char_count = chunk.get("char_count", len(chunk["text"]))
        if sentence_count > MAX_SENTENCES_PER_CHUNK:
            print(f"⚠️ Chunk trop long ({sentence_count} phrases): {chunk['text'][:100]}...")
        if char_count > MAX_CHARS_PER_CHUNK:
            print(f"⚠️ Chunk trop volumineux ({char_count} caractères): {chunk['text'][:100]}...")

def evaluate_readability(chunks):
    scores = []
    for chunk in chunks:
        score = textstat.flesch_reading_ease(chunk["text"])
        scores.append(score)
        if score < 30:
            print(f"⚠️ Faible lisibilité ({score}): {chunk['text'][:100]}...")
    return sum(scores) / len(scores)

def evaluate_chunks(chunked_data):
    chunks = chunked_data['chunks']
    avg_similarity = evaluate_chunk_quality(chunks)
    avg_readability = evaluate_readability(chunks)
    check_chunk_size(chunks)
    score = (avg_similarity + avg_readability) / 2
    print(f"✅ Score global des chunks: {score}")
    return score

def process_file(input_text_path, input_json_path=None, output_dir=OUTPUT_DIR, evaluate=True):
    output_path = Path(output_dir) / input_text_path.parent.name / "chunked_data.json"
    if output_path.exists():
        print(f"⏭️ Fichier déjà chunké, skip: {output_path}")
        return

    with open(input_text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"📄 Traitement: {input_text_path}")

    metadata = {}
    if input_json_path and input_json_path.exists():
        with open(input_json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    sections_found = metadata.get("sections_found", {
        "investment strategy": True,
        "risk factors": True,
        "performance history": True,
        "contact details": True
    })

    sections = split_into_sections(text, sections_found)

    chunked_data = {
        "source_file": str(input_text_path),
        "chunks": []
    }
    chunk_id_counter = 0

    for section_key, section_content in sections:
        chunks = semantic_chunking(section_content)
        for chunk in chunks:
            chunked_data["chunks"].append({
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": f"{input_text_path.stem}_{chunk_id_counter}",
                    "section": section_key,
                    "source_file": str(input_text_path),
                    "sentence_count": chunk["sentence_count"],
                    "char_count": chunk["char_count"]
                }
            })
            chunk_id_counter += 1

    if evaluate:
        evaluate_chunks(chunked_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Chunks sauvegardés dans: {output_path}")
    print(f"🔢 Total chunks: {len(chunked_data['chunks'])}")
    return chunked_data

def process_all_files(input_dir=INPUT_DIR):
    for folder in os.listdir(input_dir):
        folder_path = Path(input_dir) / folder
        text_path = folder_path / "extracted_text.txt"
        json_path = folder_path / "extracted_data.json"

        if text_path.exists():
            process_file(text_path, json_path if json_path.exists() else None, output_dir=OUTPUT_DIR, evaluate=False)

if __name__ == "__main__":
    print("🚀 Démarrage de la génération des chunks sémantiques...")
    process_all_files()
    print("🎯 Tous les fichiers ont été traités.")
