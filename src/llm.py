"""
llm.py — LLM client using Groq API (Llama-3-8B-Instruct, free tier)
"""
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


class LLMClient:
    """
    Thin wrapper around the Groq API.
    Groq free tier: ~30 req/min, 14,400 req/day — sufficient for dev.
    """

    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = False):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Copy .env.example → .env and add your key.\n"
                "Get a free key at: https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        self.verbose = verbose

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        retries: int = 3,
    ) -> str:
        """
        Send a prompt to Groq and return the text response.
        Handles rate-limit errors with exponential backoff.
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content.strip()
                if self.verbose:
                    print(f"[LLM] tokens used: {response.usage.total_tokens}")
                return text

            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    wait = 2 ** attempt * 5   # 5s → 10s → 20s
                    print(f"[LLM] Rate limit hit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("LLM call failed after retries.")
