import json
import re
import time
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from model_samba import SambanovaModel

# === Configuration ===
OUTPUT_DIR = Path("output")
CATEGORY_LIST = [
    "Legal and Regulatory Compliance", "Financial Due Diligence", "Technical Due Diligence",
    "Operational Due Diligence", "Market and Competitive Analysis", "Governance",
    "Risk Assessment", "Community and Ecosystem", "Exit and Liquidity",
    "Environmental, Social, and Governance (ESG) Considerations", "Future Roadmap", "Miscellaneous"
]

# Initialize models
sambanova_model = SambanovaModel()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classification_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(CATEGORY_LIST))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def classify_question(question: str) -> str:
    """Classify a question into a category using BERT."""
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = classification_model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()

    # Utilisation d'une probabilité plus élevée pour augmenter la diversité
    category_scores = torch.nn.Softmax(dim=-1)(outputs.logits).squeeze().tolist()
    top_categories = sorted(zip(CATEGORY_LIST, category_scores), key=lambda x: x[1], reverse=True)

    # Retourner la catégorie avec la plus haute probabilité, mais permettre la diversité
    if top_categories[0][1] - top_categories[1][1] < 0.1:
        return top_categories[0][0]  # Si la différence est faible, prendre la catégorie la plus probable
    else:
        return top_categories[0][0]  # Sinon, il faut toujours prendre la catégorie principale


def generate_questions(chunk_text: str) -> list[str]:
    """Generate up to 3 questions from a text chunk"""
    prompt = f"""Analyze this text about digital assets and generate 3 insightful and varied questions:
    
    Text: {chunk_text}
    
    Generate:
    1. One question about the investment strategy, focusing on key financial strategies
    2. One question about potential risks, focusing on market volatility or regulatory challenges
    3. One question about regulatory aspects, addressing compliance and governance concerns
     
    Format each question on a new line with a number prefix (1., 2., 3.)."""

    response = sambanova_model.generate(
        prompt=prompt,
        system_prompt="You are an expert in digital assets analyzing documents to generate detailed and diverse questions.",
        temperature=0.5  # Increase temperature for more diversity
    )

    if not response:
        return []

    questions = []
    for line in response.split('\n'):
        line = re.sub(r'^\s*\d+\.\s*', '', line.strip())
        if line and len(line) > 10:
            questions.append(line)

    return questions[:3]


def process_documents():
    """Main pipeline for question generation and classification"""
    logger.info("🚀 Starting question generation...")

    for folder in OUTPUT_DIR.iterdir():
        if not folder.is_dir():
            continue

        input_file = folder / "cleaned_chunks.jsonl"
        output_file = folder / "generated_questions_samba.jsonl"

        if not input_file.exists():
            logger.warning(f"⛔ File not found: {input_file}")
            continue

        logger.info(f"📁 Processing folder: {folder.name}")

        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = [json.loads(line)["text"] for line in f]

        results = []

        for idx, chunk in enumerate(tqdm(chunks, desc=f"🧠 Generating for {folder.name}")):
            if not chunk.strip():
                continue

            questions = generate_questions(chunk)
            for q in questions:
                results.append({
                    "question": q,
                    "category": classify_question(q),
                    "source": folder.name,
                    "chunk_id": idx + 1
                })

            time.sleep(0.5)  # Rate limiting

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"✅ {len(results)} questions generated for {folder.name}")

if __name__ == "__main__":
    process_documents()
    logger.info("🎉 Question generation completed successfully!")