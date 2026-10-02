"""
llm.py — Unified LLM client supporting Groq API and Ollama (local).

Configure via .env:
    # Use Groq (cloud, free tier, rate-limited)
    LLM_PROVIDER=groq
    GROQ_API_KEY=your_key_here
    GROQ_MODEL=llama-3.1-8b-instant

    # Use Ollama (local, no rate limit, needs server running)
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=llama3.1:8b
    OLLAMA_BASE_URL=http://localhost:11434  # optional, this is the default
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


class LLMClient:
    """
    Unified LLM client that supports Groq (cloud) and Ollama (local).

    Switch provider by setting LLM_PROVIDER in .env:
        LLM_PROVIDER=groq    → Groq API (default)
        LLM_PROVIDER=ollama  → local Ollama server
    """

    def __init__(self, model: str | None = None, verbose: bool = False):
        self.verbose  = verbose
        self.provider = LLM_PROVIDER

        if self.provider == "ollama":
            self._init_ollama(model)
        else:
            self._init_groq(model)

    # ── Provider initialisers ─────────────────────────────────────────────────

    def _init_groq(self, model: str | None):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "Get a free key at: https://console.groq.com\n"
                "Or switch to Ollama: set LLM_PROVIDER=ollama in .env"
            )
        self.model  = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=api_key)
        if self.verbose:
            print(f"[LLM] Provider: Groq | Model: {self.model}")

    def _init_ollama(self, model: str | None):
        # Ollama exposes an OpenAI-compatible API — use openai SDK
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for Ollama backend.\n"
                "Run: pip install openai"
            )
        base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model  = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        if self.verbose:
            print(f"[LLM] Provider: Ollama | Model: {self.model} | URL: {base_url}")

    # ── Core generation method ────────────────────────────────────────────────

    def generate(
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.0,
        retries:     int   = 3,
    ) -> str:
        """
        Send a prompt and return the text response.
        Handles rate-limit errors (Groq) with exponential backoff.
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content or ""
                text = text.strip()
                if self.verbose:
                    usage = response.usage
                    print(f"[LLM] tokens used: {usage.total_tokens if usage else '?'}")
                return text

            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    wait = 2 ** attempt * 5  # 5s → 10s → 20s
                    print(f"[LLM] Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"LLM call failed after {retries} retries.")
