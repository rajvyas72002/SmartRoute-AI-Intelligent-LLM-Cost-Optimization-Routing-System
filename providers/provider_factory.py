from providers.base_provider import BaseProvider
from providers.ollama_provider import OllamaProvider
from providers.openai_provider import OpenAIProvider

_REGISTRY = {"ollama": OllamaProvider, "openai": OpenAIProvider}


def get_provider(name: str) -> BaseProvider:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'")
    return _REGISTRY[name]()
