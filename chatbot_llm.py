# chatbot_llm.py

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

# === Logging Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Load Environment Variables ===
load_dotenv()
API_KEY = '6544f6fd-aecb-46a6-aaf7-e3fc3e368eb5'

if not API_KEY:
    logger.error("❌ Missing SambaNova API key in environment variables.")
    raise EnvironmentError("Please set SAMBANOVA_API_KEY in your .env file.")

# === Initialize OpenAI-compatible client for SambaNova ===
client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=API_KEY
)

def call_sambanova(
    prompt: str,
    system_prompt: str = "You are a helpful assistant that only responds based on the provided context.",
    model: str = "DeepSeek-R1-Distill-Llama-70B",
    temperature: float = 0.3,
    max_tokens: int = 512
) -> Optional[str]:
    """
    Call the SambaNova LLM via OpenAI-compatible API.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"🚨 Error calling SambaNova API: {e}")
        return None
