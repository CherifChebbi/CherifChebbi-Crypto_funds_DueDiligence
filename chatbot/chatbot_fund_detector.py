# chatbot_fund_detector.py

import re
from pathlib import Path
from collections import Counter

def extract_possible_fund_names(text: str) -> list:
    """
    Extract potential crypto fund names from raw text using regex patterns.
    """
    patterns = [
        r"(?:Fund Name|Crypto Fund|Investment Fund|The Fund|Fund):?\s*([A-Z][A-Za-z0-9\s\-&,]{3,100})",
        r"(?:This fund, called|known as|referred to as)\s+['\"]?([A-Z][A-Za-z0-9\s\-&,]{3,100})['\"]?",
        r"\b([A-Z][A-Za-z0-9\s\-&,]{2,100} (?:Capital|Fund|Partners|Investments|Management|LP|LLC|SPV))\b"
    ]

    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend([m.strip() for m in found if len(m.strip().split()) <= 10])

    return matches


def rank_candidates(candidates: list) -> str:
    """
    Rank candidate fund names by normalized frequency and capital weight.
    """
    if not candidates:
        return "the fund"

    counter = Counter(candidates)
    scores = {}

    for name, count in counter.items():
        cap_ratio = sum(1 for c in name if c.isupper()) / len(name)
        length_bonus = 1.0 if 5 <= len(name.split()) <= 8 else 0.8
        score = count * cap_ratio * length_bonus
        scores[name] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][0]


def detect_fund_name_from_text(text: str) -> str:
    """
    Detect the most likely crypto fund name from full text.
    """
    candidates = extract_possible_fund_names(text)
    return rank_candidates(candidates)


def detect_fund_name_from_file(file_path: Path) -> str:
    """
    Load a text file and detect the dominant fund name inside it.
    """
    if not file_path.exists():
        return "the fund"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return detect_fund_name_from_text(text)
    except Exception:
        return "the fund"
