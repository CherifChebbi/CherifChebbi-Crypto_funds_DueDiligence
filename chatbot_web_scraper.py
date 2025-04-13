# chatbot_smart_web_scraper.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from typing import List, Dict
import time


def search_bing_news(query: str, max_results: int = 5) -> List[Dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.bing.com/news/search?q={quote(query)}&FORM=HDRSC6"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Bing News error: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for card in soup.find_all("div", class_="news-card")[:max_results]:
        a_tag = card.find("a")
        snippet = card.find("div", class_="snippet") or card.find("div", class_="news-card-snippet")
        if a_tag and a_tag.get("href"):
            results.append({
                "title": a_tag.get_text(strip=True),
                "url": a_tag["href"],
                "text": f"{a_tag.get_text(strip=True)}. {snippet.get_text(strip=True) if snippet else ''}",
                "source": "Bing News",
                "score": None
            })
    return results


def search_sec(query: str, max_results: int = 3) -> List[Dict]:
    url = f"https://www.sec.gov/cgi-bin/srch-edgar?text={quote(query)}&first=2022&last=2025"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"❌ SEC.gov error: {e}")
        return []

    results = []
    for link in soup.find_all('a', href=True)[:max_results]:
        href = link['href']
        full_url = f"https://www.sec.gov{href}" if href.startswith("/") else href
        results.append({
            "title": link.text.strip(),
            "url": full_url,
            "text": link.text.strip(),
            "source": "SEC.gov",
            "score": None
        })
    return results


def search_reddit(query: str, max_results: int = 5) -> List[Dict]:
    url = f"https://www.reddit.com/search/?q={quote(query)}&sort=relevance"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"❌ Reddit error: {e}")
        return []

    posts = soup.find_all("a", href=True)
    chunks = []
    seen = set()

    for post in posts:
        href = post['href']
        text = post.get_text(strip=True)
        if "/r/" in href and text and href not in seen:
            seen.add(href)
            chunks.append({
                "title": text[:100],
                "url": f"https://www.reddit.com{href}" if href.startswith("/r/") else href,
                "text": text,
                "source": "Reddit",
                "score": None
            })
        if len(chunks) >= max_results:
            break
    return chunks


def get_web_chunks(query: str, max_results_per_source: int = 5) -> List[Dict]:
    print("🌐 Running smart multi-source web scraper...")
    bing = search_bing_news(query, max_results=max_results_per_source)
    sec = search_sec(query, max_results=max_results_per_source)
    reddit = search_reddit(query, max_results=max_results_per_source)
    all_chunks = bing + sec + reddit
    print(f"✅ Total chunks collected: {len(all_chunks)}")
    return all_chunks


# Test direct (supprimer ou commenter dans Streamlit)
if __name__ == "__main__":
    query = "BlackRock ESG controversies 2024"
    chunks = get_web_chunks(query)
    for i, c in enumerate(chunks):
        print(f"\n#{i+1} - {c['source']}: {c['title']}\n{c['url']}\n{c['text'][:200]}...\n")
