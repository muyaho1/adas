import pytest
import numpy as np
from src.agent.baselines import RandomAgent, RuleBasedAgent, SingleActionAgent


def test_random_agent_returns_valid_action():
    agent = RandomAgent(n_actions=12, seed=42)
    obs = np.zeros(9)
    action = agent.predict(obs)
    assert 0 <= action < 12


def test_random_agent_varies_actions():
    agent = RandomAgent(n_actions=12, seed=42)
    obs = np.zeros(9)
    actions = [agent.predict(obs) for _ in range(50)]
    assert len(set(actions)) > 1


def test_rule_based_agent_follows_sequence():
    agent = RuleBasedAgent()
    obs = np.zeros(9)
    actions = [agent.predict(obs) for _ in range(3)]
    assert actions == [0, 2, 10]


def test_rule_based_agent_cycles():
    agent = RuleBasedAgent(sequence=[0, 1, 2])
    obs = np.zeros(9)
    actions = [agent.predict(obs) for _ in range(6)]
    assert actions == [0, 1, 2, 0, 1, 2]


def test_single_action_agent():
    agent = SingleActionAgent(action=5)
    obs = np.zeros(9)
    actions = [agent.predict(obs) for _ in range(5)]
    assert all(a == 5 for a in actions)


def test_rule_based_agent_reset():
    agent = RuleBasedAgent(sequence=[0, 1])
    obs = np.zeros(9)
    agent.predict(obs)
    agent.reset()
    assert agent.predict(obs) == 0
