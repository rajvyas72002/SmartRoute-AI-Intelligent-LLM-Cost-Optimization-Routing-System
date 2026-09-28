from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderResponse:
    model_id: str
    text: str
    input_tokens: int
    output_tokens: int
    tokens_estimated: bool      # True if we guessed instead of reading provider usage
    latency_s: float
    error: Optional[str] = None


class BaseProvider(ABC):
    @abstractmethod
    def is_available(self, model_id: str) -> bool: ...

    @abstractmethod
    def generate(self, model_id: str, prompt: str, timeout_s: float = 120, max_retries: int = 2) -> ProviderResponse: ...
