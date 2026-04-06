import pytest
import numpy as np
from src.environment.state_parser import StateParser


def test_parse_valid_json_response():
    response = '{"state": {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.2}, "narrative": "The child looks away and fidgets."}'
    parser = StateParser()
    state, narrative = parser.parse(response)
    assert state["distress_level"] == 0.5
    assert state["compliance"] == 0.3
    assert "fidgets" in narrative


def test_parse_extracts_json_from_mixed_text():
    response = 'Here is the response:\n```json\n{"state": {"distress_level": 0.6, "compliance": 0.2, "attention": 0.3, "escalation_risk": 0.4}, "narrative": "Child protests loudly."}\n```'
    parser = StateParser()
    state, narrative = parser.parse(response)
    assert state["distress_level"] == 0.6
    assert "protests" in narrative


def test_parse_clamps_values_to_valid_range():
    response = '{"state": {"distress_level": 1.5, "compliance": -0.3, "attention": 0.4, "escalation_risk": 0.2}, "narrative": "test"}'
    parser = StateParser()
    state, _ = parser.parse(response)
    assert state["distress_level"] == 1.0
    assert state["compliance"] == 0.0


def test_parse_returns_default_on_invalid():
    parser = StateParser()
    state, narrative = parser.parse("completely invalid response with no json")
    assert all(k in state for k in ["distress_level", "compliance", "attention", "escalation_risk"])
    assert narrative != ""


def test_state_to_array():
    parser = StateParser()
    state = {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.2}
    arr = parser.state_to_array(state)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,)
    assert arr[0] == 0.5
