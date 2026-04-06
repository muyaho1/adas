from dataclasses import dataclass, field
import yaml


@dataclass
class ChildProfile:
    name: str
    age: int
    severity: str  # mild, moderate, severe
    traits: dict[str, float]
    description: str
    evidence: list[dict[str, str]] = field(default_factory=list)
    behavioral_rationale: str = ""
    state_priors: dict[str, float] = field(default_factory=dict)
    expected_transition_sensitivity: dict[str, float] = field(default_factory=dict)


def load_profiles(path: str) -> list[ChildProfile]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return [ChildProfile(**p) for p in data["profiles"]]
