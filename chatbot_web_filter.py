# chatbot_web_filter.py

from typing import List, Dict
import json
from chatbot_llm import call_sambanova

def review_web_chunks(question: str, web_chunks: List[Dict], max_chunks: int = 5) -> List[Dict]:
    """Evaluate web chunks for relevance and trust using LLM."""
    reviewed_chunks = []

    for chunk in web_chunks[:max_chunks]:
        review_prompt = f"""
You are an AI assistant evaluating the quality of web content to answer a user's question.

Your job is to assess how relevant and trustworthy the following content is with respect to the question.

Respond ONLY in JSON format:
{{
    "relevance_score": float (0 to 1),
    "trust_score": float (0 to 1),
    "reason": "short justification"
}}

Question:
{question}

Web Content:
{chunk['text']}

Source: {chunk.get('source', 'unknown')}

Evaluate now:
"""

        try:
            response = call_sambanova(review_prompt)
            parsed = json.loads(response)
            chunk["relevance_score"] = round(parsed.get("relevance_score", 0.0), 2)
            chunk["trust_score"] = round(parsed.get("trust_score", 0.0), 2)
            chunk["review_reason"] = parsed.get("reason", "")
        except Exception as e:
            chunk["relevance_score"] = 0.0
            chunk["trust_score"] = 0.0
            chunk["review_reason"] = f"Failed to parse review: {str(e)}"

        reviewed_chunks.append(chunk)

    return reviewed_chunks
