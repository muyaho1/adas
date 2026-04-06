from dataclasses import dataclass, field
import yaml


@dataclass
class Scenario:
    name: str
    type: str
    description: str
    initial_state: dict[str, float]
    evidence: list[dict[str, str]] = field(default_factory=list)
    behavioral_rationale: str = ""
    state_priors: dict[str, float] = field(default_factory=dict)
    expected_transition_sensitivity: dict[str, float] = field(default_factory=dict)


def load_scenarios(path: str) -> list[Scenario]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return [Scenario(**s) for s in data["scenarios"]]
