from dataclasses import dataclass, field


@dataclass
class StudentSessionMemory:
    student_id: str
    teacher_trust: float = 0.5
    transition_tolerance: float = 0.5
    strategy_response_history: dict[str, float] = field(default_factory=dict)
    recent_trigger_patterns: list[str] = field(default_factory=list)
    sessions_seen: int = 0

    def summary(self) -> str:
        best_actions = sorted(
            self.strategy_response_history.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        action_text = ", ".join(f"{name}:{score:.2f}" for name, score in best_actions) or "none"
        triggers = ", ".join(self.recent_trigger_patterns[-3:]) or "none"
        return (
            f"teacher_trust={self.teacher_trust:.2f}, "
            f"transition_tolerance={self.transition_tolerance:.2f}, "
            f"effective_actions={action_text}, recent_triggers={triggers}"
        )

    def initial_state_adjustment(self, profile, scenario) -> dict[str, float]:
        severity_factor = {"mild": 0.5, "moderate": 1.0, "severe": 1.5}.get(
            profile.severity,
            1.0,
        )
        trust_gain = self.teacher_trust - 0.5
        tolerance_gain = self.transition_tolerance - 0.5
        return {
            "distress_level": -0.18 * trust_gain - 0.12 * tolerance_gain + 0.04 * severity_factor,
            "compliance": 0.20 * trust_gain + 0.16 * tolerance_gain - 0.03 * severity_factor,
            "attention": 0.10 * tolerance_gain - 0.04 * profile.traits.get("inattention", 0.5),
            "escalation_risk": -0.16 * trust_gain - 0.14 * tolerance_gain + 0.05 * severity_factor,
        }

    def update(
        self,
        action_name: str,
        prev_state: dict[str, float],
        next_state: dict[str, float],
        reward: float,
        scenario_type: str,
        terminated_successfully: bool,
    ) -> None:
        delta_compliance = next_state["compliance"] - prev_state["compliance"]
        delta_distress = prev_state["distress_level"] - next_state["distress_level"]
        delta_attention = next_state["attention"] - prev_state["attention"]
        response_score = 0.5 * delta_compliance + 0.3 * delta_distress + 0.2 * delta_attention
        if reward < 0:
            response_score += 0.15 * reward

        old_score = self.strategy_response_history.get(action_name, 0.0)
        self.strategy_response_history[action_name] = 0.7 * old_score + 0.3 * response_score

        trust_delta = 0.05 * delta_distress + 0.04 * delta_compliance
        if next_state["escalation_risk"] > 0.8:
            trust_delta -= 0.08
        if terminated_successfully:
            trust_delta += 0.05

        tolerance_delta = 0.04 * delta_compliance + 0.03 * delta_attention + 0.02 * delta_distress
        if terminated_successfully:
            tolerance_delta += 0.04

        self.teacher_trust = max(0.0, min(1.0, self.teacher_trust + trust_delta))
        self.transition_tolerance = max(
            0.0,
            min(1.0, self.transition_tolerance + tolerance_delta),
        )

        if scenario_type:
            self.recent_trigger_patterns.append(scenario_type)
            self.recent_trigger_patterns = self.recent_trigger_patterns[-10:]

    def mark_session_complete(self) -> None:
        self.sessions_seen += 1


class MemoryStore:
    def __init__(self):
        self._memories: dict[str, StudentSessionMemory] = {}

    def get(self, student_id: str) -> StudentSessionMemory:
        if student_id not in self._memories:
            self._memories[student_id] = StudentSessionMemory(student_id=student_id)
        return self._memories[student_id]

    def snapshot(self) -> dict[str, dict[str, float | int | str]]:
        return {
            student_id: {
                "teacher_trust": memory.teacher_trust,
                "transition_tolerance": memory.transition_tolerance,
                "sessions_seen": memory.sessions_seen,
                "summary": memory.summary(),
            }
            for student_id, memory in self._memories.items()
        }
