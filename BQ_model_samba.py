from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SambanovaModel:
    def __init__(self):
        self.api_key = 'bfde6096-758b-46b8-a678-3d55567aa5e5'
        
        if not self.api_key:
            logger.error("❌ SAMBANOVA_API_KEY not found in .env file")
            raise ValueError("API key is required. Please set SAMBANOVA_API_KEY in your .env file")

        self.client = OpenAI(
            base_url="https://api.sambanova.ai/v1",
            api_key=self.api_key
        )
        
        logger.info("🔐 API client initialized successfully")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that generates specific questions about digital assets.",
        model: str = "Meta-Llama-3.1-70B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Sends a prompt to the Sambanova model and returns the text response
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            
            logger.error("⚠️ Unexpected response format: No choices in response")
            return None

        except Exception as e:
            logger.error(f"🚨 Error calling Sambanova API: {e}")
            return None