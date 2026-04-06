import pytest
from unittest.mock import patch, MagicMock
from src.llm.backend import LLMBackend
from src.llm.claude_code_backend import ClaudeCodeBackend


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        LLMBackend()


def test_claude_code_backend_constructs(tmp_path):
    backend = ClaudeCodeBackend(cache_dir=str(tmp_path / "test_cache"), cache_enabled=False)
    assert backend is not None


@patch("subprocess.run")
def test_claude_code_backend_generate(mock_run, tmp_path):
    mock_run.return_value = MagicMock(
        stdout='{"state": {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.2}, "narrative": "The child fidgets and looks away."}',
        returncode=0,
    )
    backend = ClaudeCodeBackend(cache_dir=str(tmp_path / "test_cache"), cache_enabled=False)
    result = backend.generate("test prompt")
    assert "child" in result.lower() or "state" in result.lower()
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_claude_code_backend_uses_cache(mock_run, tmp_path):
    mock_run.return_value = MagicMock(
        stdout='{"state": {"distress_level": 0.5}, "narrative": "Response"}',
        returncode=0,
    )
    backend = ClaudeCodeBackend(cache_dir=str(tmp_path / "test_llm_cache"), cache_enabled=True)
    result1 = backend.generate("same prompt")
    result2 = backend.generate("same prompt")
    assert result1 == result2
    assert mock_run.call_count == 1


@patch("subprocess.run")
def test_claude_code_backend_retries_on_failure(mock_run, tmp_path):
    mock_run.side_effect = [
        MagicMock(stdout="", returncode=1),
        MagicMock(
            stdout='{"state": {"distress_level": 0.3}, "narrative": "OK"}',
            returncode=0,
        ),
    ]
    backend = ClaudeCodeBackend(
        cache_dir=str(tmp_path / "test_cache"), cache_enabled=False, retry_attempts=3, retry_delay=0.0
    )
    result = backend.generate("test prompt")
    assert result is not None
    assert mock_run.call_count == 2
