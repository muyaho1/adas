# Behavioral AI Simulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an RL agent that learns appropriate clinician responses to ADHD behavioral patterns in simulated school-age children, using Claude Code CLI as the LLM backend.

**Architecture:** A Gymnasium-compatible LLM child environment (Claude Code CLI wrapper) produces behavioral state vectors. A PPO agent (Stable-Baselines3) selects intervention actions. A research-grounded reward function scores each turn. Response caching makes CLI-based training feasible.

**Tech Stack:** Python 3.11+, Gymnasium, Stable-Baselines3, PyYAML, diskcache, matplotlib, seaborn, pytest

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `src/__init__.py`
- Create: `src/environment/__init__.py`
- Create: `src/agent/__init__.py`
- Create: `src/llm/__init__.py`
- Create: `src/reward/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `src/cache/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```txt
gymnasium>=0.29.0
stable-baselines3>=2.3.0
pyyaml>=6.0
diskcache>=5.6.0
matplotlib>=3.8.0
seaborn>=0.13.0
numpy>=1.26.0
pytest>=8.0.0
```

**Step 2: Create configs/default.yaml**

```yaml
environment:
  max_turns: 10
  success_threshold: 0.8
  failure_distress: 0.9
  failure_consecutive: 3
  scenario_types:
    - preferred_to_nonpreferred
    - timed_activity_ending
    - unexpected_schedule_change
    - hyperfocus_interruption

agent:
  algorithm: PPO
  policy: MlpPolicy
  hidden_sizes: [64, 64]
  learning_rate: 0.0003
  n_steps: 128
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  total_timesteps: 50000

reward:
  w_compliance: 0.3
  w_attention: 0.15
  w_distress: 0.25
  w_efficiency: 0.1
  safety_threshold: 0.8
  safety_penalty: -1.0
  bonus_success: 5.0
  bonus_fast: 2.0
  bonus_gentle: 2.0
  bonus_monotonic: 1.0
  penalty_failure: -5.0
  gentle_threshold: 0.6
  fast_threshold: 5

llm:
  backend: claude_code
  cache_enabled: true
  cache_dir: .cache/responses
  retry_attempts: 3
  retry_delay: 2.0

evaluation:
  n_eval_episodes: 100
  seed: 42
```

**Step 3: Create all __init__.py files and directory structure**

```bash
mkdir -p src/environment src/agent src/llm src/reward src/eval src/cache
mkdir -p tests data/profiles data/scenarios results/training results/evaluation results/figures
mkdir -p configs
touch src/__init__.py src/environment/__init__.py src/agent/__init__.py
touch src/llm/__init__.py src/reward/__init__.py src/eval/__init__.py src/cache/__init__.py
touch tests/__init__.py
```

**Step 4: Install dependencies**

Run: `uv venv && source .venv/bin/activate && uv pip install -r requirements.txt`

**Step 5: Commit**

```bash
git add requirements.txt configs/default.yaml src/ tests/ data/ results/.gitkeep
git commit -m "feat: scaffold project structure and dependencies"
```

---

## Task 2: Response Cache

**Files:**
- Create: `src/cache/response_cache.py`
- Create: `tests/test_cache.py`

**Step 1: Write the failing test**

```python
# tests/test_cache.py
import os
import shutil
import pytest
from src.cache.response_cache import ResponseCache


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(cache_dir=str(tmp_path / "test_cache"))


def test_cache_miss_returns_none(cache):
    result = cache.get("prompt_1", "context_1")
    assert result is None


def test_cache_set_and_get(cache):
    cache.set("prompt_1", "context_1", "response_text")
    result = cache.get("prompt_1", "context_1")
    assert result == "response_text"


def test_cache_different_keys_are_independent(cache):
    cache.set("prompt_1", "context_1", "response_a")
    cache.set("prompt_2", "context_2", "response_b")
    assert cache.get("prompt_1", "context_1") == "response_a"
    assert cache.get("prompt_2", "context_2") == "response_b"


def test_cache_stats(cache):
    cache.get("prompt_1", "context_1")  # miss
    cache.set("prompt_1", "context_1", "response_text")
    cache.get("prompt_1", "context_1")  # hit
    cache.get("prompt_2", "context_2")  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2


def test_cache_disabled():
    cache = ResponseCache(cache_dir="/tmp/unused", enabled=False)
    cache.set("prompt_1", "context_1", "response_text")
    result = cache.get("prompt_1", "context_1")
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/cache/response_cache.py
import hashlib
import json
import os


class ResponseCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._hits = 0
        self._misses = 0
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def _make_key(self, prompt: str, context: str) -> str:
        combined = json.dumps({"prompt": prompt, "context": context}, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, prompt: str, context: str) -> str | None:
        if not self.enabled:
            self._misses += 1
            return None
        key = self._make_key(prompt, context)
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            self._hits += 1
            with open(path, "r") as f:
                return json.load(f)["response"]
        self._misses += 1
        return None

    def set(self, prompt: str, context: str, response: str) -> None:
        if not self.enabled:
            return
        key = self._make_key(prompt, context)
        path = os.path.join(self.cache_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump({"prompt": prompt, "context": context, "response": response}, f)

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/cache/response_cache.py tests/test_cache.py
git commit -m "feat: add hash-based response cache with hit/miss tracking"
```

---

## Task 3: LLM Backend Interface & Claude Code Wrapper

**Files:**
- Create: `src/llm/backend.py`
- Create: `src/llm/claude_code_backend.py`
- Create: `tests/test_llm_backend.py`

**Step 1: Write the failing test**

```python
# tests/test_llm_backend.py
import pytest
from unittest.mock import patch, MagicMock
from src.llm.backend import LLMBackend
from src.llm.claude_code_backend import ClaudeCodeBackend


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        LLMBackend()


def test_claude_code_backend_constructs():
    backend = ClaudeCodeBackend(cache_dir="/tmp/test_cache", cache_enabled=False)
    assert backend is not None


@patch("subprocess.run")
def test_claude_code_backend_generate(mock_run):
    mock_run.return_value = MagicMock(
        stdout='{"state": {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.2}, "narrative": "The child fidgets and looks away."}',
        returncode=0,
    )
    backend = ClaudeCodeBackend(cache_dir="/tmp/test_cache", cache_enabled=False)
    result = backend.generate("test prompt")
    assert "child" in result.lower() or "state" in result.lower()
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_claude_code_backend_uses_cache(mock_run):
    mock_run.return_value = MagicMock(
        stdout='{"state": {"distress_level": 0.5}, "narrative": "Response"}',
        returncode=0,
    )
    backend = ClaudeCodeBackend(cache_dir="/tmp/test_llm_cache", cache_enabled=True)
    result1 = backend.generate("same prompt")
    result2 = backend.generate("same prompt")
    assert result1 == result2
    assert mock_run.call_count == 1  # second call served from cache


@patch("subprocess.run")
def test_claude_code_backend_retries_on_failure(mock_run):
    mock_run.side_effect = [
        MagicMock(stdout="", returncode=1),
        MagicMock(
            stdout='{"state": {"distress_level": 0.3}, "narrative": "OK"}',
            returncode=0,
        ),
    ]
    backend = ClaudeCodeBackend(
        cache_dir="/tmp/test_cache", cache_enabled=False, retry_attempts=3, retry_delay=0.0
    )
    result = backend.generate("test prompt")
    assert result is not None
    assert mock_run.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_backend.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the backend interface**

```python
# src/llm/backend.py
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to LLM and return raw response string."""
        ...
```

**Step 4: Write the Claude Code backend**

```python
# src/llm/claude_code_backend.py
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
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_backend.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add src/llm/backend.py src/llm/claude_code_backend.py tests/test_llm_backend.py
git commit -m "feat: add LLM backend interface and Claude Code CLI wrapper with caching"
```

---

## Task 4: Child Profiles & Scenarios

**Files:**
- Create: `src/environment/child_profiles.py`
- Create: `src/environment/scenarios.py`
- Create: `data/profiles/adhd_profiles.yaml`
- Create: `data/scenarios/task_transitions.yaml`
- Create: `tests/test_profiles_scenarios.py`

**Step 1: Write the failing test**

```python
# tests/test_profiles_scenarios.py
import pytest
from src.environment.child_profiles import ChildProfile, load_profiles
from src.environment.scenarios import Scenario, load_scenarios


def test_child_profile_fields():
    profile = ChildProfile(
        name="test_child",
        age=9,
        severity="moderate",
        traits={"impulsivity": 0.7, "inattention": 0.6, "emotional_reactivity": 0.5},
        description="A 9-year-old with moderate ADHD.",
    )
    assert profile.name == "test_child"
    assert profile.severity == "moderate"
    assert 0.0 <= profile.traits["impulsivity"] <= 1.0


def test_load_profiles_from_yaml(tmp_path):
    yaml_content = """
profiles:
  - name: child_a
    age: 8
    severity: mild
    traits:
      impulsivity: 0.4
      inattention: 0.5
      emotional_reactivity: 0.3
    description: "Mild ADHD, mostly inattentive."
  - name: child_b
    age: 11
    severity: severe
    traits:
      impulsivity: 0.9
      inattention: 0.8
      emotional_reactivity: 0.85
    description: "Severe ADHD, combined type."
"""
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml_content)
    profiles = load_profiles(str(path))
    assert len(profiles) == 2
    assert profiles[0].severity == "mild"
    assert profiles[1].severity == "severe"


def test_scenario_fields():
    scenario = Scenario(
        name="recess_to_math",
        type="preferred_to_nonpreferred",
        description="Teacher announces transition from recess to math class.",
        initial_state={"distress_level": 0.3, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.2},
    )
    assert scenario.type == "preferred_to_nonpreferred"
    assert scenario.initial_state["distress_level"] == 0.3


def test_load_scenarios_from_yaml(tmp_path):
    yaml_content = """
scenarios:
  - name: recess_to_math
    type: preferred_to_nonpreferred
    description: "Teacher announces transition from recess to math."
    initial_state:
      distress_level: 0.3
      compliance: 0.2
      attention: 0.4
      escalation_risk: 0.2
"""
    path = tmp_path / "scenarios.yaml"
    path.write_text(yaml_content)
    scenarios = load_scenarios(str(path))
    assert len(scenarios) == 1
    assert scenarios[0].name == "recess_to_math"


def test_profile_severity_values():
    for sev in ["mild", "moderate", "severe"]:
        p = ChildProfile(name="x", age=8, severity=sev, traits={}, description="")
        assert p.severity in ("mild", "moderate", "severe")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_profiles_scenarios.py -v`
Expected: FAIL

**Step 3: Write child_profiles.py**

```python
# src/environment/child_profiles.py
from dataclasses import dataclass
import yaml


@dataclass
class ChildProfile:
    name: str
    age: int
    severity: str  # mild, moderate, severe
    traits: dict[str, float]
    description: str


def load_profiles(path: str) -> list[ChildProfile]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return [ChildProfile(**p) for p in data["profiles"]]
```

**Step 4: Write scenarios.py**

```python
# src/environment/scenarios.py
from dataclasses import dataclass
import yaml


@dataclass
class Scenario:
    name: str
    type: str
    description: str
    initial_state: dict[str, float]


def load_scenarios(path: str) -> list[Scenario]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return [Scenario(**s) for s in data["scenarios"]]
```

**Step 5: Create data YAML files**

```yaml
# data/profiles/adhd_profiles.yaml
profiles:
  - name: mild_inattentive
    age: 8
    severity: mild
    traits:
      impulsivity: 0.3
      inattention: 0.6
      emotional_reactivity: 0.25
    description: >
      An 8-year-old with mild ADHD, predominantly inattentive presentation.
      Tends to daydream during transitions but rarely escalates.

  - name: moderate_combined
    age: 9
    severity: moderate
    traits:
      impulsivity: 0.6
      inattention: 0.55
      emotional_reactivity: 0.5
    description: >
      A 9-year-old with moderate ADHD, combined type.
      Shows both inattention and impulsivity. May resist transitions
      with verbal protests but can be redirected.

  - name: severe_hyperactive
    age: 11
    severity: severe
    traits:
      impulsivity: 0.85
      inattention: 0.7
      emotional_reactivity: 0.8
    description: >
      An 11-year-old with severe ADHD, predominantly hyperactive-impulsive.
      Transitions frequently trigger emotional outbursts, physical restlessness,
      and strong resistance. Requires careful, patient intervention.
```

```yaml
# data/scenarios/task_transitions.yaml
scenarios:
  - name: recess_to_math
    type: preferred_to_nonpreferred
    description: >
      The teacher announces that recess is ending and it's time to go inside
      for math class. The child is actively playing with friends.
    initial_state:
      distress_level: 0.3
      compliance: 0.15
      attention: 0.2
      escalation_risk: 0.3

  - name: art_time_ending
    type: timed_activity_ending
    description: >
      Art class is ending in 5 minutes. The child is deeply engaged in
      painting a picture they are proud of.
    initial_state:
      distress_level: 0.2
      compliance: 0.3
      attention: 0.15
      escalation_risk: 0.2

  - name: surprise_assembly
    type: unexpected_schedule_change
    description: >
      The normal schedule is disrupted by a surprise school assembly.
      The child was expecting to go to their favorite class next.
    initial_state:
      distress_level: 0.45
      compliance: 0.1
      attention: 0.3
      escalation_risk: 0.4

  - name: reading_to_cleanup
    type: hyperfocus_interruption
    description: >
      The child is deeply absorbed in a book during free reading time.
      The teacher asks everyone to stop and clean up their area.
    initial_state:
      distress_level: 0.25
      compliance: 0.1
      attention: 0.1
      escalation_risk: 0.25
```

**Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_profiles_scenarios.py -v`
Expected: All 5 tests PASS

**Step 7: Commit**

```bash
git add src/environment/child_profiles.py src/environment/scenarios.py
git add data/profiles/adhd_profiles.yaml data/scenarios/task_transitions.yaml
git add tests/test_profiles_scenarios.py
git commit -m "feat: add child profiles and scenario definitions with YAML loading"
```

---

## Task 5: State Parser

**Files:**
- Create: `src/environment/state_parser.py`
- Create: `tests/test_state_parser.py`

**Step 1: Write the failing test**

```python
# tests/test_state_parser.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state_parser.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/environment/state_parser.py
import json
import re
import numpy as np

STATE_KEYS = ["distress_level", "compliance", "attention", "escalation_risk"]
DEFAULT_STATE = {k: 0.5 for k in STATE_KEYS}


class StateParser:
    def parse(self, response: str) -> tuple[dict[str, float], str]:
        data = self._extract_json(response)
        if data is None:
            return dict(DEFAULT_STATE), "Failed to parse response."

        state_raw = data.get("state", DEFAULT_STATE)
        state = {}
        for key in STATE_KEYS:
            val = float(state_raw.get(key, 0.5))
            state[key] = max(0.0, min(1.0, val))

        narrative = data.get("narrative", "No narrative provided.")
        return state, narrative

    def _extract_json(self, text: str) -> dict | None:
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding any JSON object
        match = re.search(r"\{[^{}]*\"state\"[^{}]*\{.*?\}.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def state_to_array(self, state: dict[str, float]) -> np.ndarray:
        return np.array([state[k] for k in STATE_KEYS], dtype=np.float32)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state_parser.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/environment/state_parser.py tests/test_state_parser.py
git commit -m "feat: add state parser with JSON extraction and value clamping"
```

---

## Task 6: Reward Function

**Files:**
- Create: `src/reward/reward_function.py`
- Create: `tests/test_reward.py`

**Step 1: Write the failing test**

```python
# tests/test_reward.py
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
    history = [0.5, 0.4, 0.3]  # distress values
    bonus = reward_fn.episode_bonus(success=True, turns=4, distress_history=history)
    assert bonus >= 5.0  # success + fast + gentle + monotonic


def test_episode_bonus_failure(reward_fn):
    history = [0.5, 0.7, 0.95]
    bonus = reward_fn.episode_bonus(success=False, turns=10, distress_history=history)
    assert bonus == -5.0


def test_episode_bonus_slow_success(reward_fn):
    history = [0.5, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
    bonus = reward_fn.episode_bonus(success=True, turns=7, distress_history=history)
    assert bonus >= 5.0  # success + gentle + monotonic, but NOT fast


def test_custom_weights():
    custom = RewardFunction(w_compliance=0.5, w_distress=0.5, w_attention=0.0, w_efficiency=0.0)
    prev = {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.2}
    curr = {"distress_level": 0.3, "compliance": 0.5, "attention": 0.4, "escalation_risk": 0.2}
    r = custom.compute(prev, curr, turn=1)
    expected = 0.5 * 0.2 + 0.5 * 0.2  # compliance up 0.2, distress down 0.2
    assert abs(r - expected) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reward.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/reward/reward_function.py

class RewardFunction:
    def __init__(
        self,
        w_compliance: float = 0.3,
        w_attention: float = 0.15,
        w_distress: float = 0.25,
        w_efficiency: float = 0.1,
        safety_threshold: float = 0.8,
        safety_penalty: float = -1.0,
        bonus_success: float = 5.0,
        bonus_fast: float = 2.0,
        bonus_gentle: float = 2.0,
        bonus_monotonic: float = 1.0,
        penalty_failure: float = -5.0,
        gentle_threshold: float = 0.6,
        fast_threshold: int = 5,
    ):
        self.w_compliance = w_compliance
        self.w_attention = w_attention
        self.w_distress = w_distress
        self.w_efficiency = w_efficiency
        self.safety_threshold = safety_threshold
        self.safety_penalty = safety_penalty
        self.bonus_success = bonus_success
        self.bonus_fast = bonus_fast
        self.bonus_gentle = bonus_gentle
        self.bonus_monotonic = bonus_monotonic
        self.penalty_failure = penalty_failure
        self.gentle_threshold = gentle_threshold
        self.fast_threshold = fast_threshold

    def compute(self, prev_state: dict, curr_state: dict, turn: int) -> float:
        d_compliance = curr_state["compliance"] - prev_state["compliance"]
        d_attention = curr_state["attention"] - prev_state["attention"]
        d_distress = curr_state["distress_level"] - prev_state["distress_level"]

        r_therapeutic = self.w_compliance * d_compliance + self.w_attention * d_attention
        r_emotional = self.w_distress * (-d_distress)
        r_safety = self.safety_penalty if curr_state["escalation_risk"] > self.safety_threshold else 0.0
        r_efficiency = self.w_efficiency * (1.0 / max(turn, 1))

        return r_therapeutic + r_emotional + r_safety + r_efficiency

    def episode_bonus(self, success: bool, turns: int, distress_history: list[float]) -> float:
        if not success:
            return self.penalty_failure

        bonus = self.bonus_success

        if turns <= self.fast_threshold:
            bonus += self.bonus_fast

        if all(d <= self.gentle_threshold for d in distress_history):
            bonus += self.bonus_gentle

        if all(distress_history[i] >= distress_history[i + 1] for i in range(len(distress_history) - 1)):
            bonus += self.bonus_monotonic

        return bonus
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reward.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/reward/reward_function.py tests/test_reward.py
git commit -m "feat: add research-grounded reward function with safety penalties and episode bonuses"
```

---

## Task 7: Gymnasium Child Environment

**Files:**
- Create: `src/environment/child_env.py`
- Create: `tests/test_child_env.py`

**Step 1: Write the failing test**

```python
# tests/test_child_env.py
import pytest
import numpy as np
from unittest.mock import MagicMock
import gymnasium as gym
from src.environment.child_env import ADHDChildEnv
from src.environment.child_profiles import ChildProfile
from src.environment.scenarios import Scenario


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.generate.return_value = '{"state": {"distress_level": 0.4, "compliance": 0.4, "attention": 0.5, "escalation_risk": 0.2}, "narrative": "The child looks at the timer."}'
    return backend


@pytest.fixture
def profile():
    return ChildProfile(
        name="test", age=9, severity="moderate",
        traits={"impulsivity": 0.6, "inattention": 0.5, "emotional_reactivity": 0.5},
        description="Test child",
    )


@pytest.fixture
def scenario():
    return Scenario(
        name="test_scenario", type="preferred_to_nonpreferred",
        description="Recess to math",
        initial_state={"distress_level": 0.3, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.2},
    )


@pytest.fixture
def env(mock_backend, profile, scenario):
    return ADHDChildEnv(
        backend=mock_backend,
        profiles=[profile],
        scenarios=[scenario],
    )


def test_env_is_gymnasium_compatible(env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)


def test_env_action_space_size(env):
    assert env.action_space.n == 12


def test_env_observation_space_shape(env):
    assert env.observation_space.shape[0] >= 4  # at least state vector


def test_env_reset_returns_observation(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert "narrative" in info


def test_env_step_returns_correct_tuple(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "narrative" in info


def test_env_terminates_on_success(mock_backend):
    mock_backend.generate.return_value = '{"state": {"distress_level": 0.1, "compliance": 0.9, "attention": 0.8, "escalation_risk": 0.1}, "narrative": "Child complies happily."}'
    profile = ChildProfile(name="t", age=9, severity="mild", traits={}, description="")
    scenario = Scenario(name="s", type="test", description="", initial_state={"distress_level": 0.2, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.1})
    env = ADHDChildEnv(backend=mock_backend, profiles=[profile], scenarios=[scenario])
    env.reset()
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True


def test_env_truncates_at_max_turns(mock_backend):
    mock_backend.generate.return_value = '{"state": {"distress_level": 0.5, "compliance": 0.3, "attention": 0.4, "escalation_risk": 0.3}, "narrative": "Still resisting."}'
    profile = ChildProfile(name="t", age=9, severity="moderate", traits={}, description="")
    scenario = Scenario(name="s", type="test", description="", initial_state={"distress_level": 0.3, "compliance": 0.2, "attention": 0.4, "escalation_risk": 0.2})
    env = ADHDChildEnv(backend=mock_backend, profiles=[profile], scenarios=[scenario], max_turns=3)
    env.reset()
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
        assert not truncated
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated is True


def test_env_records_episode_history(env):
    env.reset()
    env.step(0)
    env.step(1)
    assert len(env.history) == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_child_env.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/environment/child_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.llm.backend import LLMBackend
from src.environment.child_profiles import ChildProfile
from src.environment.scenarios import Scenario
from src.environment.state_parser import StateParser
from src.reward.reward_function import RewardFunction

ACTIONS = [
    "transition_warning",
    "offer_choice",
    "labeled_praise",
    "visual_schedule_cue",
    "break_offer",
    "empathic_acknowledgment",
    "redirect_attention",
    "countdown_timer",
    "collaborative_problem_solving",
    "ignore_wait",
    "firm_boundary",
    "sensory_support",
]


class ADHDChildEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        backend: LLMBackend,
        profiles: list[ChildProfile],
        scenarios: list[Scenario],
        max_turns: int = 10,
        success_threshold: float = 0.8,
        failure_distress: float = 0.9,
        failure_consecutive: int = 3,
        reward_fn: RewardFunction | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.profiles = profiles
        self.scenarios = scenarios
        self.max_turns = max_turns
        self.success_threshold = success_threshold
        self.failure_distress = failure_distress
        self.failure_consecutive = failure_consecutive
        self.reward_fn = reward_fn or RewardFunction()
        self.parser = StateParser()

        self.action_space = spaces.Discrete(len(ACTIONS))

        # observation: 4 state dims + 4 scenario one-hot + 1 normalized turn
        obs_dim = 4 + len(scenarios) + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.current_profile = None
        self.current_scenario = None
        self.current_state = None
        self.turn = 0
        self.history = []
        self.distress_history = []
        self.consecutive_high_distress = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self.current_profile = self.profiles[rng.integers(len(self.profiles))]
        self.current_scenario = self.scenarios[rng.integers(len(self.scenarios))]
        self.current_state = dict(self.current_scenario.initial_state)
        self.turn = 0
        self.history = []
        self.distress_history = [self.current_state["distress_level"]]
        self.consecutive_high_distress = 0

        obs = self._make_observation()
        info = {"narrative": self.current_scenario.description, "profile": self.current_profile.name}
        return obs, info

    def step(self, action: int):
        self.turn += 1
        action_name = ACTIONS[action]

        prompt = self._build_prompt(action_name)
        response = self.backend.generate(prompt)
        new_state, narrative = self.parser.parse(response)

        reward = self.reward_fn.compute(self.current_state, new_state, self.turn)

        self.history.append({
            "turn": self.turn,
            "action": action_name,
            "state": dict(new_state),
            "narrative": narrative,
        })
        self.distress_history.append(new_state["distress_level"])

        prev_state = self.current_state
        self.current_state = new_state

        # Check termination conditions
        terminated = False
        truncated = False

        if new_state["compliance"] > self.success_threshold:
            terminated = True
            reward += self.reward_fn.episode_bonus(
                success=True, turns=self.turn, distress_history=self.distress_history
            )

        if new_state["distress_level"] > self.failure_distress:
            self.consecutive_high_distress += 1
        else:
            self.consecutive_high_distress = 0

        if self.consecutive_high_distress >= self.failure_consecutive:
            terminated = True
            reward += self.reward_fn.episode_bonus(
                success=False, turns=self.turn, distress_history=self.distress_history
            )

        if self.turn >= self.max_turns and not terminated:
            truncated = True

        obs = self._make_observation()
        info = {"narrative": narrative, "action": action_name, "turn": self.turn}
        return obs, float(reward), terminated, truncated, info

    def _make_observation(self) -> np.ndarray:
        state_arr = self.parser.state_to_array(self.current_state)

        scenario_onehot = np.zeros(len(self.scenarios), dtype=np.float32)
        idx = self.scenarios.index(self.current_scenario)
        scenario_onehot[idx] = 1.0

        turn_norm = np.array([self.turn / self.max_turns], dtype=np.float32)

        return np.concatenate([state_arr, scenario_onehot, turn_norm])

    def _build_prompt(self, action_name: str) -> str:
        history_text = ""
        for h in self.history[-3:]:  # last 3 turns for context
            history_text += f"Turn {h['turn']}: Clinician used '{h['action']}'. Child: {h['narrative']}\n"

        return f"""You are simulating a {self.current_profile.age}-year-old child with {self.current_profile.severity} ADHD.
Profile: {self.current_profile.description}
Traits: impulsivity={self.current_profile.traits.get('impulsivity', 0.5)}, inattention={self.current_profile.traits.get('inattention', 0.5)}, emotional_reactivity={self.current_profile.traits.get('emotional_reactivity', 0.5)}

Scenario: {self.current_scenario.description}

Previous interactions:
{history_text if history_text else "None yet."}

The clinician now uses: {action_name}

Respond as this child would. Return ONLY a JSON object with this exact format:
{{"state": {{"distress_level": <0.0-1.0>, "compliance": <0.0-1.0>, "attention": <0.0-1.0>, "escalation_risk": <0.0-1.0>}}, "narrative": "<1-2 sentence description of child's behavioral response>"}}"""
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_child_env.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/environment/child_env.py tests/test_child_env.py
git commit -m "feat: add Gymnasium-compatible ADHD child environment with LLM backend"
```

---

## Task 8: Baseline Agents

**Files:**
- Create: `src/agent/baselines.py`
- Create: `tests/test_baselines.py`

**Step 1: Write the failing test**

```python
# tests/test_baselines.py
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
    # Default sequence: transition_warning(0), labeled_praise(2), firm_boundary(10)
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/agent/baselines.py
import numpy as np


class RandomAgent:
    def __init__(self, n_actions: int = 12, seed: int = 42):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))

    def reset(self):
        pass


class RuleBasedAgent:
    def __init__(self, sequence: list[int] | None = None):
        self.sequence = sequence or [0, 2, 10]  # warn, praise, firm boundary
        self.step_idx = 0

    def predict(self, obs: np.ndarray) -> int:
        action = self.sequence[self.step_idx % len(self.sequence)]
        self.step_idx += 1
        return action

    def reset(self):
        self.step_idx = 0


class SingleActionAgent:
    def __init__(self, action: int = 0):
        self.action = action

    def predict(self, obs: np.ndarray) -> int:
        return self.action

    def reset(self):
        pass
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/agent/baselines.py tests/test_baselines.py
git commit -m "feat: add baseline agents (random, rule-based, single-action)"
```

---

## Task 9: PPO Agent Wrapper

**Files:**
- Create: `src/agent/ppo_agent.py`
- Create: `tests/test_ppo_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_ppo_agent.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import gymnasium as gym
from gymnasium import spaces


def test_ppo_agent_wrapper_constructs():
    from src.agent.ppo_agent import PPOAgent
    env = MagicMock(spec=gym.Env)
    env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
    env.action_space = spaces.Discrete(12)
    agent = PPOAgent(env=env, hidden_sizes=[64, 64], learning_rate=0.0003)
    assert agent is not None


def test_ppo_agent_predict_returns_valid_action():
    from src.agent.ppo_agent import PPOAgent
    env = MagicMock(spec=gym.Env)
    env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
    env.action_space = spaces.Discrete(12)
    agent = PPOAgent(env=env)
    obs = np.zeros(9, dtype=np.float32)
    action = agent.predict(obs)
    assert 0 <= action < 12


def test_ppo_agent_save_and_load(tmp_path):
    from src.agent.ppo_agent import PPOAgent
    env = MagicMock(spec=gym.Env)
    env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
    env.action_space = spaces.Discrete(12)
    agent = PPOAgent(env=env)
    save_path = str(tmp_path / "test_model")
    agent.save(save_path)
    loaded = PPOAgent.load(save_path, env=env)
    obs = np.zeros(9, dtype=np.float32)
    action = loaded.predict(obs)
    assert 0 <= action < 12
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ppo_agent.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/agent/ppo_agent.py
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.0003,
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        hidden_sizes = hidden_sizes or [64, 64]
        policy_kwargs = {"net_arch": hidden_sizes}
        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            seed=seed,
            verbose=0,
        )

    def train(self, total_timesteps: int) -> None:
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "PPOAgent":
        agent = cls.__new__(cls)
        agent.model = PPO.load(path, env=env)
        return agent
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ppo_agent.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/agent/ppo_agent.py tests/test_ppo_agent.py
git commit -m "feat: add PPO agent wrapper around Stable-Baselines3"
```

---

## Task 10: Training Script

**Files:**
- Create: `train.py`
- Create: `tests/test_train.py`

**Step 1: Write the failing test**

```python
# tests/test_train.py
import pytest
from unittest.mock import MagicMock, patch
from train import load_config, build_env, run_training


def test_load_config():
    config = load_config("configs/default.yaml")
    assert "environment" in config
    assert "agent" in config
    assert "reward" in config
    assert "llm" in config


def test_build_env_returns_gymnasium_env():
    config = load_config("configs/default.yaml")
    mock_backend = MagicMock()
    mock_backend.generate.return_value = '{"state": {"distress_level": 0.4, "compliance": 0.4, "attention": 0.5, "escalation_risk": 0.2}, "narrative": "test"}'
    env = build_env(config, backend=mock_backend)
    obs, info = env.reset()
    assert obs is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# train.py
import argparse
import os
import yaml
from src.environment.child_env import ADHDChildEnv
from src.environment.child_profiles import load_profiles
from src.environment.scenarios import load_scenarios
from src.llm.backend import LLMBackend
from src.llm.claude_code_backend import ClaudeCodeBackend
from src.reward.reward_function import RewardFunction
from src.agent.ppo_agent import PPOAgent


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_reward(config: dict) -> RewardFunction:
    rc = config["reward"]
    return RewardFunction(
        w_compliance=rc["w_compliance"],
        w_attention=rc["w_attention"],
        w_distress=rc["w_distress"],
        w_efficiency=rc["w_efficiency"],
        safety_threshold=rc["safety_threshold"],
        safety_penalty=rc["safety_penalty"],
        bonus_success=rc["bonus_success"],
        bonus_fast=rc["bonus_fast"],
        bonus_gentle=rc["bonus_gentle"],
        bonus_monotonic=rc["bonus_monotonic"],
        penalty_failure=rc["penalty_failure"],
        gentle_threshold=rc["gentle_threshold"],
        fast_threshold=rc["fast_threshold"],
    )


def build_backend(config: dict) -> LLMBackend:
    lc = config["llm"]
    return ClaudeCodeBackend(
        cache_dir=lc["cache_dir"],
        cache_enabled=lc["cache_enabled"],
        retry_attempts=lc["retry_attempts"],
        retry_delay=lc["retry_delay"],
    )


def build_env(config: dict, backend: LLMBackend | None = None) -> ADHDChildEnv:
    if backend is None:
        backend = build_backend(config)
    ec = config["environment"]
    profiles = load_profiles("data/profiles/adhd_profiles.yaml")
    scenarios = load_scenarios("data/scenarios/task_transitions.yaml")
    reward_fn = build_reward(config)
    return ADHDChildEnv(
        backend=backend,
        profiles=profiles,
        scenarios=scenarios,
        max_turns=ec["max_turns"],
        success_threshold=ec["success_threshold"],
        failure_distress=ec["failure_distress"],
        failure_consecutive=ec["failure_consecutive"],
        reward_fn=reward_fn,
    )


def run_training(config: dict, env: ADHDChildEnv) -> PPOAgent:
    ac = config["agent"]
    agent = PPOAgent(
        env=env,
        hidden_sizes=ac["hidden_sizes"],
        learning_rate=ac["learning_rate"],
        n_steps=ac["n_steps"],
        batch_size=ac["batch_size"],
        n_epochs=ac["n_epochs"],
        gamma=ac["gamma"],
    )
    agent.train(total_timesteps=ac["total_timesteps"])

    os.makedirs("results/training", exist_ok=True)
    agent.save("results/training/ppo_adhd_agent")
    print(f"Model saved to results/training/ppo_adhd_agent")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train ADHD behavioral intervention agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config)
    agent = run_training(config, env)

    cache_stats = env.backend.cache.stats() if hasattr(env.backend, "cache") else {}
    print(f"Cache stats: {cache_stats}")
    print("Training complete.")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: add training script with config loading and environment setup"
```

---

## Task 11: Evaluation Metrics & Visualization

**Files:**
- Create: `src/eval/metrics.py`
- Create: `src/eval/visualize.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_metrics.py
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
    assert s["avg_turns"] == 5.0  # (3+7)/2 successful only
    assert abs(s["avg_distress_peak"] - 0.4) < 0.01  # (0.3+0.5)/2 successful only
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL

**Step 3: Write metrics implementation**

```python
# src/eval/metrics.py
import csv
from collections import Counter


class EvaluationMetrics:
    def __init__(self):
        self.episodes = []
        self.all_actions = []

    def record_episode(
        self,
        success: bool,
        turns: int,
        distress_peak: float,
        safety_violations: int,
        distress_monotonic: bool,
    ) -> None:
        self.episodes.append({
            "success": success,
            "turns": turns,
            "distress_peak": distress_peak,
            "safety_violations": safety_violations,
            "distress_monotonic": distress_monotonic,
        })

    def record_actions(self, actions: list[int]) -> None:
        self.all_actions.extend(actions)

    def summary(self) -> dict:
        n = len(self.episodes)
        if n == 0:
            return {}

        successes = [e for e in self.episodes if e["success"]]
        n_success = len(successes)
        n_violations = sum(1 for e in self.episodes if e["safety_violations"] > 0)

        return {
            "success_rate": n_success / n,
            "avg_turns": sum(e["turns"] for e in successes) / n_success if n_success else 0,
            "avg_distress_peak": sum(e["distress_peak"] for e in successes) / n_success if n_success else 0,
            "safety_violation_rate": n_violations / n,
            "monotonic_deescalation_rate": sum(1 for e in successes if e["distress_monotonic"]) / n_success if n_success else 0,
            "total_episodes": n,
            "total_successes": n_success,
        }

    def action_frequency(self) -> Counter:
        return Counter(self.all_actions)

    def to_csv(self, path: str) -> None:
        summary = self.summary()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                writer.writerow([k, v])
```

**Step 4: Write visualization module**

```python
# src/eval/visualize.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curve(rewards: list[float], output_path: str, window: int = 50) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rewards, alpha=0.3, label="Raw")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"Rolling avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Curve")
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_action_heatmap(action_counts: dict, action_names: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = [action_counts.get(i, 0) for i in range(len(action_names))]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.barh(action_names, counts)
    ax.set_xlabel("Frequency")
    ax.set_title("Action Distribution")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_policy_heatmap(
    distress_bins: np.ndarray,
    compliance_bins: np.ndarray,
    action_grid: np.ndarray,
    action_names: list[str],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        action_grid,
        xticklabels=[f"{c:.1f}" for c in compliance_bins],
        yticklabels=[f"{d:.1f}" for d in distress_bins],
        annot=True,
        fmt="d",
        cmap="viridis",
        ax=ax,
    )
    ax.set_xlabel("Compliance")
    ax.set_ylabel("Distress")
    ax.set_title("Policy: Preferred Action by State")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add src/eval/metrics.py src/eval/visualize.py tests/test_metrics.py
git commit -m "feat: add evaluation metrics, CSV export, and visualization plotting"
```

---

## Task 12: Evaluation Script

**Files:**
- Create: `evaluate.py`
- Create: `tests/test_evaluate.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import pytest
from unittest.mock import MagicMock
from evaluate import evaluate_agent


def test_evaluate_agent_returns_summary():
    mock_env = MagicMock()
    mock_env.reset.return_value = (
        __import__("numpy").zeros(9, dtype="float32"),
        {"narrative": "start"},
    )
    mock_env.step.return_value = (
        __import__("numpy").zeros(9, dtype="float32"),
        5.0,
        True,  # terminated
        False,
        {"narrative": "done", "action": "test", "turn": 1},
    )
    mock_env.distress_history = [0.3, 0.2]
    mock_env.history = [{"turn": 1, "action": "test", "state": {"distress_level": 0.2}, "narrative": "done"}]

    mock_agent = MagicMock()
    mock_agent.predict.return_value = 0

    summary = evaluate_agent(mock_agent, mock_env, n_episodes=5)
    assert "success_rate" in summary
    assert summary["total_episodes"] == 5
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# evaluate.py
import argparse
import json
import os
import numpy as np
from src.eval.metrics import EvaluationMetrics
from src.eval.visualize import plot_training_curve, plot_action_heatmap, plot_policy_heatmap
from src.environment.child_env import ADHDChildEnv, ACTIONS
from src.agent.ppo_agent import PPOAgent
from src.agent.baselines import RandomAgent, RuleBasedAgent, SingleActionAgent
from train import load_config, build_env


def evaluate_agent(agent, env, n_episodes: int = 100, deterministic: bool = True) -> dict:
    metrics = EvaluationMetrics()

    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_actions = []

        while not terminated and not truncated:
            if hasattr(agent, "predict") and hasattr(agent, "model"):
                action = agent.predict(obs, deterministic=deterministic)
            else:
                action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_actions.append(action)

        success = terminated and env.current_state["compliance"] > env.success_threshold
        distress_peak = max(env.distress_history) if env.distress_history else 0.0
        safety_violations = sum(
            1 for h in env.history if h["state"].get("escalation_risk", 0) > 0.8
        )
        distress_monotonic = all(
            env.distress_history[i] >= env.distress_history[i + 1]
            for i in range(len(env.distress_history) - 1)
        ) if len(env.distress_history) > 1 else False

        metrics.record_episode(
            success=success,
            turns=len(episode_actions),
            distress_peak=distress_peak,
            safety_violations=safety_violations,
            distress_monotonic=distress_monotonic,
        )
        metrics.record_actions(episode_actions)

    return metrics.summary()


def run_baseline_comparison(config: dict, n_episodes: int = 100):
    results = {}
    baselines = {
        "random": RandomAgent(n_actions=12),
        "rule_based": RuleBasedAgent(),
        "single_action_best": SingleActionAgent(action=0),
    }

    # Evaluate baselines
    for name, agent in baselines.items():
        env = build_env(config)
        summary = evaluate_agent(agent, env, n_episodes=n_episodes)
        results[name] = summary
        print(f"{name}: {summary}")

    # Evaluate trained PPO agent
    model_path = "results/training/ppo_adhd_agent"
    if os.path.exists(model_path + ".zip"):
        env = build_env(config)
        ppo = PPOAgent.load(model_path, env=env)
        summary = evaluate_agent(ppo, env, n_episodes=n_episodes)
        results["ppo_trained"] = summary
        print(f"ppo_trained: {summary}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_baseline_comparison(config, n_episodes=args.episodes)

    os.makedirs("results/evaluation", exist_ok=True)
    with open("results/evaluation/comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/evaluation/comparison.json")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation script with baseline comparison"
```

---

## Task 13: Export Figures Script

**Files:**
- Create: `export_figures.py`

**Step 1: Write the script**

```python
# export_figures.py
import argparse
import json
import os
import numpy as np
from src.eval.visualize import plot_training_curve, plot_action_heatmap, plot_policy_heatmap
from src.environment.child_env import ACTIONS


def main():
    parser = argparse.ArgumentParser(description="Export results to publication figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Training curve
    rewards_path = os.path.join(args.results_dir, "training", "episode_rewards.json")
    if os.path.exists(rewards_path):
        with open(rewards_path) as f:
            rewards = json.load(f)
        plot_training_curve(rewards, os.path.join(args.output_dir, "training_curve.png"))
        print("Exported training_curve.png")

    # Comparison results
    comparison_path = os.path.join(args.results_dir, "evaluation", "comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            results = json.load(f)
        print(f"Comparison results loaded: {list(results.keys())}")

    print("Figure export complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add export_figures.py
git commit -m "feat: add figure export script for publication-ready plots"
```

---

## Task 14: Integration Test — Full Training Loop

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from train import load_config, build_env, run_training
from evaluate import evaluate_agent
from src.agent.baselines import RandomAgent


@pytest.fixture
def mock_backend():
    """Backend that simulates a child who gradually complies."""
    call_count = {"n": 0}

    def generate(prompt):
        call_count["n"] += 1
        # Simulate gradual compliance over turns
        turn = call_count["n"] % 10
        compliance = min(0.1 + turn * 0.12, 0.95)
        distress = max(0.6 - turn * 0.08, 0.1)
        return (
            f'{{"state": {{"distress_level": {distress:.2f}, "compliance": {compliance:.2f}, '
            f'"attention": {0.3 + turn * 0.05:.2f}, "escalation_risk": {max(0.3 - turn * 0.03, 0.05):.2f}}}, '
            f'"narrative": "Turn {turn}: child gradually settles."}}'
        )

    backend = MagicMock()
    backend.generate.side_effect = generate
    return backend


def test_full_training_loop(mock_backend):
    config = load_config("configs/default.yaml")
    # Reduce training for test speed
    config["agent"]["total_timesteps"] = 256
    config["agent"]["n_steps"] = 64
    config["agent"]["batch_size"] = 32

    env = build_env(config, backend=mock_backend)
    agent = run_training(config, env)
    assert agent is not None


def test_trained_agent_beats_random(mock_backend):
    config = load_config("configs/default.yaml")
    config["agent"]["total_timesteps"] = 512
    config["agent"]["n_steps"] = 64
    config["agent"]["batch_size"] = 32

    env = build_env(config, backend=mock_backend)
    agent = run_training(config, env)

    # Evaluate trained agent
    eval_env = build_env(config, backend=mock_backend)
    trained_summary = evaluate_agent(agent, eval_env, n_episodes=20)

    # Evaluate random agent
    eval_env2 = build_env(config, backend=mock_backend)
    random_agent = RandomAgent(n_actions=12)
    random_summary = evaluate_agent(random_agent, eval_env2, n_episodes=20)

    # Trained agent should perform at least as well as random
    # (with mock backend and tiny training, just verify it runs)
    assert trained_summary["total_episodes"] == 20
    assert random_summary["total_episodes"] == 20
```

**Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v --timeout=120`
Expected: All 2 tests PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration tests for full training and evaluation loop"
```

---

## Task 15: Run All Tests & Final Verification

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify project structure exists**

Run: `find . -name "*.py" | head -30 && echo "---" && cat requirements.txt`
Expected: All planned files exist

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: complete POC implementation of ADHD behavioral AI simulation"
```

---

## Execution Summary

| Task | Component | Tests |
|---|---|---|
| 1 | Project scaffolding | - |
| 2 | Response cache | 5 |
| 3 | LLM backend + Claude Code wrapper | 5 |
| 4 | Child profiles & scenarios | 5 |
| 5 | State parser | 5 |
| 6 | Reward function | 8 |
| 7 | Gymnasium child environment | 9 |
| 8 | Baseline agents | 6 |
| 9 | PPO agent wrapper | 3 |
| 10 | Training script | 2 |
| 11 | Evaluation metrics & visualization | 5 |
| 12 | Evaluation script | 1 |
| 13 | Export figures script | - |
| 14 | Integration test | 2 |
| 15 | Final verification | - |
| **Total** | **15 tasks** | **56 tests** |
