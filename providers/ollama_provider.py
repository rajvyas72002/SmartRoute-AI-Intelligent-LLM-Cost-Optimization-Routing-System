import os, time, requests
from providers.base_provider import BaseProvider, ProviderResponse


class OllamaProvider(BaseProvider):
    def __init__(self, host: str | None = None):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def installed_models(self) -> set[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
            return {m["name"] for m in r.json().get("models", [])}
        except requests.RequestException:
            return set()

    def is_available(self, model_id: str) -> bool:
        names = self.installed_models()
        return model_id in names or f"{model_id}:latest" in names

    def generate(self, model_id, prompt, timeout_s=120, max_retries=2) -> ProviderResponse:
        start, last_err = time.perf_counter(), None
        for attempt in range(max_retries + 1):               # bounded retries
            try:
                r = requests.post(f"{self.host}/api/generate", timeout=timeout_s, json={
                    "model": model_id, "prompt": prompt, "stream": False,
                    "options": {"num_ctx": 4096}})
                r.raise_for_status()
                d = r.json()
                return ProviderResponse(model_id, d.get("response", ""), d.get("prompt_eval_count", 0),
                                        d.get("eval_count", 0), False, time.perf_counter() - start)
            except requests.RequestException as e:
                last_err = str(e)
                time.sleep(min(2 ** attempt, 4))
        return ProviderResponse(model_id, "", 0, 0, True, time.perf_counter() - start, error=last_err)
