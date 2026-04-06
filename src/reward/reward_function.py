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
        safety_violated = curr_state["escalation_risk"] > self.safety_threshold

        r_therapeutic = self.w_compliance * d_compliance + self.w_attention * d_attention
        r_emotional = self.w_distress * (-d_distress)
        r_safety = self.safety_penalty if safety_violated else 0.0
        # Keep the safety penalty "hard" by not offsetting it with efficiency reward.
        r_efficiency = 0.0 if safety_violated else self.w_efficiency * (1.0 / max(turn, 1))

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
