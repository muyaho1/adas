import pytest
from src.reward.reward_function import RewardFunction


@pytest.fixture
def reward_fn():
    return RewardFunction()


def test_positive_reward_on_improvement(reward_fn):
    prev = {"distress_level": 0.6, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.3}
    curr = {"distress_level": 0.4, "compliance": 0.5, "attention": 0.6, "escalation_risk": 0.2}
    r = reward_fn.compute(prev, curr, turn=1)
    assert r > 0


def test_negative_reward_on_deterioration(reward_fn):
    prev = {"distress_level": 0.3, "compliance": 0.5, "attention": 0.6, "escalation_risk": 0.2}
    curr = {"distress_level": 0.7, "compliance": 0.2, "attention": 0.3, "escalation_risk": 0.5}
    r = reward_fn.compute(prev, curr, turn=1)
    assert r < 0


def test_safety_penalty_when_escalation_high(reward_fn):
    prev = {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.5}
    curr = {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.85}
    r = reward_fn.compute(prev, curr, turn=1)
    assert r <= -1.0


def test_efficiency_reward_higher_for_earlier_turns(reward_fn):
    prev = {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.3}
    curr = {"distress_level": 0.4, "compliance": 0.5, "attention": 0.5, "escalation_risk": 0.2}
    r_early = reward_fn.compute(prev, curr, turn=1)
    r_late = reward_fn.compute(prev, curr, turn=8)
    assert r_early > r_late


def test_episode_bonus_success(reward_fn):
    history = [0.5, 0.4, 0.3]
    bonus = reward_fn.episode_bonus(success=True, turns=4, distress_history=history)
    assert bonus >= 5.0


def test_episode_bonus_failure(reward_fn):
    history = [0.5, 0.7, 0.95]
    bonus = reward_fn.episode_bonus(success=False, turns=10, distress_history=history)
    assert bonus == -5.0


def test_episode_bonus_slow_success(reward_fn):
    history = [0.5, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
    bonus = reward_fn.episode_bonus(success=True, turns=7, distress_history=history)
    assert bonus >= 5.0


def test_custom_weights():
    custom = RewardFunction(w_compliance=0.5, w_distress=0.5, w_attention=0.0, w_efficiency=0.0)
    prev = {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.2}
    curr = {"distress_level": 0.3, "compliance": 0.5, "attention": 0.4, "escalation_risk": 0.2}
    r = custom.compute(prev, curr, turn=1)
    expected = 0.5 * 0.2 + 0.5 * 0.2
    assert abs(r - expected) < 0.01
