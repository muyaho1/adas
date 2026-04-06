import subprocess
import time
from src.llm.backend import LLMBackend
from src.cache.response_cache import ResponseCache


class ClaudeCodeBackend(LLMBackend):
    def __init__(
        self,
        cache_dir: str = ".cache/responses",
        cache_enabled: bool = True,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        self.cache = ResponseCache(cache_dir=cache_dir, enabled=cache_enabled)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    def generate(self, prompt: str) -> str:
        cached = self.cache.get(prompt, "")
        if cached is not None:
            return cached

        response = self._call_claude_code(prompt)
        self.cache.set(prompt, "", response)
        return response

    def _call_claude_code(self, prompt: str) -> str:
        for attempt in range(self.retry_attempts):
            try:
                result = subprocess.run(
                    ["claude", "-p", prompt, "--output-format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except subprocess.TimeoutExpired:
                pass

            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(
            f"Claude Code CLI failed after {self.retry_attempts} attempts"
        )
