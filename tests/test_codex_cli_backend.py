from unittest.mock import MagicMock, patch

from src.llm.codex_cli_backend import CodexCLIBackend


VALID_RESPONSE = '{"state": {"distress_level": 0.3, "compliance": 0.6, "attention": 0.5, "escalation_risk": 0.2}, "narrative": "The student takes a breath and looks toward the teacher."}'


@patch("src.llm.codex_cli_backend.subprocess.run")
def test_codex_cli_backend_generate_reads_output_last_message(mock_run, tmp_path):
    def fake_run(command, input, capture_output, text, timeout, cwd, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(VALID_RESPONSE)
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = fake_run
    backend = CodexCLIBackend(cache_dir=str(tmp_path / "cache"), cache_enabled=False, retry_delay=0.0)

    result = backend.generate("simulate one classroom transition")

    assert '"narrative"' in result
    assert mock_run.call_count == 1


@patch("src.llm.codex_cli_backend.subprocess.run")
def test_codex_cli_backend_uses_cache(mock_run, tmp_path):
    def fake_run(command, input, capture_output, text, timeout, cwd, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(VALID_RESPONSE)
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = fake_run
    backend = CodexCLIBackend(cache_dir=str(tmp_path / "cache"), cache_enabled=True, retry_delay=0.0)

    result_1 = backend.generate("same prompt")
    result_2 = backend.generate("same prompt")

    assert result_1 == result_2
    assert mock_run.call_count == 1


@patch("src.llm.codex_cli_backend.subprocess.run")
def test_codex_cli_backend_retries_on_invalid_json(mock_run, tmp_path):
    responses = ["not-json", VALID_RESPONSE]

    def fake_run(command, input, capture_output, text, timeout, cwd, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(responses.pop(0))
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = fake_run
    backend = CodexCLIBackend(
        cache_dir=str(tmp_path / "cache"),
        cache_enabled=False,
        retry_attempts=2,
        retry_delay=0.0,
    )

    result = backend.generate("retry invalid output")

    assert '"state"' in result
    assert mock_run.call_count == 2
