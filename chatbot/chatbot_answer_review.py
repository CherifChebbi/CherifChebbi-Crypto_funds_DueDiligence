# chatbot_answer_review.py

from chatbot_llm import call_sambanova
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_llm_json_output(raw_response: str) -> str:
    """
    Supprime les décorateurs Markdown comme ```json et les espaces inutiles.
    """
    return re.sub(r"^```json|```$", "", raw_response.strip()).strip()

def review_answer_with_context(question: str, answer: str, context_chunks: list) -> dict:
    context_text = "\n\n".join([chunk["chunk"]["text"] for chunk in context_chunks if "chunk" in chunk])

    base_prompt = f"""
You are an expert crypto compliance auditor.

Evaluate whether the assistant’s answer to the question is fully supported by the provided context.

Return your evaluation in valid JSON with the following keys:
- confidence_score (float, 0 to 100)
- hallucination_risk (str: none, low, medium, high)
- justification (short explanation)
- verdict ("✅ Reliable" or "❌ Unsupported")

Context:
{context_text}

Question:
{question}

Answer:
{answer}

Evaluation:
"""

    logger.info("Calling SambaNova LLM for answer review...")
    review_response = call_sambanova(base_prompt)
    logger.info(f"🧪 Raw Review Response: {review_response}")

    # Tentative 1 : nettoyage + parsing
    try:
        cleaned = clean_llm_json_output(review_response)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "confidence_score" in parsed:
            return parsed
    except Exception as e:
        logger.warning(f"First parsing attempt failed: {e}")

    # Tentative 2 : reformulation du prompt
    retry_prompt = base_prompt + "\nPlease reformat your answer strictly as JSON without any preamble."
    review_response_retry = call_sambanova(retry_prompt)
    logger.info(f"🧪 Retry Raw Review Response: {review_response_retry}")

    try:
        cleaned_retry = clean_llm_json_output(review_response_retry)
        parsed_retry = json.loads(cleaned_retry)
        if isinstance(parsed_retry, dict) and "confidence_score" in parsed_retry:
            return parsed_retry
    except Exception as e:
        logger.warning(f"Retry parsing failed: {e}")

    # Fallback en cas d’échec
    return {
        "confidence_score": 0.0,
        "hallucination_risk": "unknown",
        "justification": "Could not parse review after retry.",
        "verdict": "❌ Undetermined"
    }


# Test stub (pour vérification manuelle)
if __name__ == "__main__":
    dummy_question = "What is the investment strategy of the fund?"
    dummy_answer = "The fund focuses on diversified crypto assets across DeFi and Web3."
    dummy_chunks = [
        {"chunk": {"text": "The fund invests in a wide range of digital assets, including infrastructure tokens, Layer 1s, and emerging DeFi protocols."}},
        {"chunk": {"text": "It focuses on long-term growth and has exposure to several ecosystems like Ethereum, Solana, and Cosmos."}}
    ]

    review = review_answer_with_context(dummy_question, dummy_answer, dummy_chunks)
    print(json.dumps(review, indent=2))