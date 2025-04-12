# chatbot_eval_qa.py

import json
from pathlib import Path
from chatbot_query_answering import answer_query_from_documents
from typing import List, Dict
from rich import print
from rich.table import Table

# === Test Cases ===
TEST_CASES = [
    {
        "question": "What is the investment strategy of the fund?",
        "expected_keywords": ["liquidity", "central bank", "not explicitly provided"],
        "index_path": None,
        "difficulty": "medium"
    },
    {
        "question": "Who is the main contact person for the fund?",
        "expected_keywords": ["contact", "email", "phone"],
        "index_path": None,
        "difficulty": "easy"
    },
    {
        "question": "What are the main risk factors identified?",
        "expected_keywords": ["market risk", "regulatory", "liquidity risk", "volatility"],
        "index_path": None,
        "difficulty": "hard"
    },
]

# === Evaluation function ===
def evaluate_response(response: str, expected_keywords: List[str]) -> Dict:
    found = [kw for kw in expected_keywords if kw.lower() in response.lower()]
    total = len(expected_keywords)
    keyword_score = len(found) / total if total > 0 else 0.0
    return {
        "keyword_score": round(keyword_score, 2),
        "keywords_found": found,
        "keywords_expected": expected_keywords
    }

# === Test runner ===
def run_test_cases():
    table = Table(title="📊 QA Evaluation Report")
    table.add_column("#", justify="right")
    table.add_column("Question", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Keywords Found", style="green")
    table.add_column("Difficulty", justify="center")

    for i, case in enumerate(TEST_CASES, 1):
        question = case["question"]
        expected = case["expected_keywords"]
        index_path = case.get("index_path")
        difficulty = case.get("difficulty", "unknown")

        try:
            answer, _ = answer_query_from_documents(question, index_path=index_path)
            result = evaluate_response(answer or "", expected)
            score = f"{result['keyword_score']*100:.0f}%"
            table.add_row(str(i), question, score, ", ".join(result["keywords_found"]), difficulty)
        except Exception as e:
            table.add_row(str(i), question, "ERROR", str(e), difficulty)

    print(table)

if __name__ == "__main__":
    run_test_cases()
