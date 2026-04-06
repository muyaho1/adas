import pytest
from src.eval.metrics import EvaluationMetrics


def test_record_episode_success():
    metrics = EvaluationMetrics()
    metrics.record_episode(success=True, turns=4, distress_peak=0.45, safety_violations=0, distress_monotonic=True)
    summary = metrics.summary()
    assert summary["success_rate"] == 1.0
    assert summary["avg_turns"] == 4.0


def test_record_episode_failure():
    metrics = EvaluationMetrics()
    metrics.record_episode(success=False, turns=10, distress_peak=0.95, safety_violations=2, distress_monotonic=False)
    summary = metrics.summary()
    assert summary["success_rate"] == 0.0


def test_mixed_episodes():
    metrics = EvaluationMetrics()
    metrics.record_episode(success=True, turns=3, distress_peak=0.3, safety_violations=0, distress_monotonic=True)
    metrics.record_episode(success=True, turns=7, distress_peak=0.5, safety_violations=0, distress_monotonic=False)
    metrics.record_episode(success=False, turns=10, distress_peak=0.9, safety_violations=1, distress_monotonic=False)
    s = metrics.summary()
    assert abs(s["success_rate"] - 2 / 3) < 0.01
    assert s["avg_turns"] == 5.0
    assert abs(s["avg_distress_peak"] - 0.4) < 0.01
    assert s["safety_violation_rate"] == 1 / 3


def test_action_frequency():
    metrics = EvaluationMetrics()
    metrics.record_actions([0, 0, 1, 2, 0])
    freq = metrics.action_frequency()
    assert freq[0] == 3
    assert freq[1] == 1
    assert freq[2] == 1


def test_to_csv(tmp_path):
    metrics = EvaluationMetrics()
    metrics.record_episode(success=True, turns=4, distress_peak=0.3, safety_violations=0, distress_monotonic=True)
    path = str(tmp_path / "metrics.csv")
    metrics.to_csv(path)
    with open(path) as f:
        content = f.read()
    assert "success_rate" in content
