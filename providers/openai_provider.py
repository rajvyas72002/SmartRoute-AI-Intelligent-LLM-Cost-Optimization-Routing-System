"""OpenAI-compatible chat endpoint (works with OpenAI and compatible servers). Key comes from env only."""
import os, time, requests
from providers.base_provider import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY", "")
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def is_available(self, model_id: str) -> bool:
        return bool(self.key)

    def generate(self, model_id, prompt, timeout_s=60, max_retries=2) -> ProviderResponse:
        start, last_err = time.perf_counter(), None
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(f"{self.base}/chat/completions", timeout=timeout_s,
                                  headers={"Authorization": f"Bearer {self.key}"},
                                  json={"model": model_id, "messages": [{"role": "user", "content": prompt}]})
                if r.status_code == 429:                       # rate limit -> back off, retry
                    time.sleep(2 ** attempt); last_err = "rate limited"; continue
                r.raise_for_status()
                d = r.json()
                u = d.get("usage", {})
                return ProviderResponse(model_id, d["choices"][0]["message"]["content"],
                                        u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                                        False, time.perf_counter() - start)
            except requests.RequestException as e:
                last_err = type(e).__name__          # don't leak headers/keys into logs
                time.sleep(min(2 ** attempt, 4))
        return ProviderResponse(model_id, "", 0, 0, True, time.perf_counter() - start, error=last_err)
