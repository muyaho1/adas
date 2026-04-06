from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to LLM and return raw response string."""
        ...
