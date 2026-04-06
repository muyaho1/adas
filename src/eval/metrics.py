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
