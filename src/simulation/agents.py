import numpy as np

from src.environment.child_env import ACTIONS


class TeacherAgent:
    def __init__(self, policy_agent, name: str = "Ms. Lee"):
        self.policy_agent = policy_agent
        self.name = name
        self._line_counts: dict[str, int] = {}

    def select_action(self, obs: np.ndarray) -> tuple[int, str]:
        if hasattr(self.policy_agent, "model"):
            action_idx = self.policy_agent.predict(obs, deterministic=True)
        else:
            action_idx = self.policy_agent.predict(obs)
        return action_idx, ACTIONS[action_idx]

    def narrate_action(self, action_name: str, scenario_name: str) -> str:
        templates = {
            "transition_warning": [
                "Let's get ready to switch. We have a short transition coming up.",
                "In a moment we're moving on, so start wrapping up this part.",
                "You have a little warning now, then we'll make the switch together.",
            ],
            "offer_choice": [
                "You can choose the first small step, but we still need to move together.",
                "Pick which part you want to do first, and then we'll keep going.",
                "Would you rather start with standing up or putting the first item away?",
            ],
            "labeled_praise": [
                "I noticed you paused and looked over. That was a helpful start.",
                "You shifted your attention when I asked. That helped the transition.",
                "That small step mattered. You listened and started to move with me.",
            ],
            "visual_schedule_cue": [
                "Let's check the visual schedule and see what comes next.",
                "Look here with me. This is where we are, and this is the next step.",
                "The schedule shows what comes now and what comes after that.",
            ],
            "break_offer": [
                "Take one quick regulation break, then we'll start the next activity.",
                "Let's do one short reset and then keep moving.",
                "Use a tiny break to settle your body, and then we transition.",
            ],
            "empathic_acknowledgment": [
                "I can tell this switch feels hard because you were really engaged.",
                "You were deep in that activity, so this change probably feels abrupt.",
                "It makes sense that stopping right now feels frustrating.",
            ],
            "redirect_attention": [
                "Look at this first part of the next task. Let's start there.",
                "Put your eyes here with me. We only need the first small step.",
                "Let's focus on one piece at a time so the switch feels smaller.",
            ],
            "countdown_timer": [
                "I'll set a short countdown so the switch is predictable.",
                "We're using a timer so you can see exactly when this changes.",
                "Watch the countdown with me, and then we'll move when it ends.",
            ],
            "collaborative_problem_solving": [
                "What would make this transition easier right now?",
                "Let's solve this together. What's getting in the way most?",
                "Tell me the smallest change that would help you move forward.",
            ],
            "ignore_wait": [
                "I'm going to give you a brief moment and stay nearby.",
                "I'll pause for a second so you can settle without extra pressure.",
                "You have a short quiet moment. I'm still right here when you're ready.",
            ],
            "firm_boundary": [
                "It's time to move to the next activity now.",
                "We're done with this part. The class is moving on now.",
                "I hear that this is hard, and the next step is still happening now.",
            ],
            "sensory_support": [
                "You can use a fidget or grounding strategy while we transition.",
                "Take your sensory support with you so your body has something steady.",
                "Use the grounding tool while you move to the next task.",
            ],
        }
        options = templates.get(action_name, [f"I am using {action_name} for {scenario_name}."])
        index = self._line_counts.get(action_name, 0) % len(options)
        self._line_counts[action_name] = self._line_counts.get(action_name, 0) + 1
        return options[index]


class PeerStudentAgent:
    def __init__(self, name: str, seat: tuple[int, int]):
        self.name = name
        self.seat = {"x": seat[0], "y": seat[1]}
        self._mood_counts = {"neutral": 0, "settled": 0, "concerned": 0}

    def react(self, target_state: dict[str, float], teacher_action: str) -> dict[str, object]:
        escalation = target_state.get("escalation_risk", 0.0)
        attention = target_state.get("attention", 0.0)
        if escalation > 0.75:
            mood = "concerned"
            options = [
                f"{self.name} glances over and loses focus because the room feels tense.",
                f"{self.name} pauses their own work and watches the escalation unfold.",
            ]
        elif attention > 0.65 and teacher_action in {"labeled_praise", "visual_schedule_cue", "transition_warning"}:
            mood = "settled"
            options = [
                f"{self.name} follows the classroom routine and settles back into the shared flow.",
                f"{self.name} notices the room calming and returns to the expected routine.",
            ]
        else:
            mood = "neutral"
            options = [
                f"{self.name} keeps working but still tracks what is happening nearby.",
                f"{self.name} stays mostly on task while quietly checking the transition.",
            ]
        index = self._mood_counts[mood] % len(options)
        self._mood_counts[mood] += 1
        return {"name": self.name, "mood": mood, "utterance": options[index], "seat": self.seat}


class ObserverAgent:
    def __init__(self, name: str = "Observer"):
        self.name = name

    def score_tick(
        self,
        prev_state: dict[str, float],
        next_state: dict[str, float],
        reward: float,
        action_name: str,
    ) -> dict[str, object]:
        delta_compliance = next_state["compliance"] - prev_state["compliance"]
        delta_distress = next_state["distress_level"] - prev_state["distress_level"]
        if next_state["escalation_risk"] > 0.8:
            note = f"{action_name} needs caution: escalation risk is high."
            quality = "safety_risk"
        elif delta_compliance > 0 and delta_distress <= 0:
            note = f"{action_name} helped the transition while keeping distress stable or lower."
            quality = "constructive"
        elif delta_compliance > 0:
            note = f"{action_name} improved compliance but distress rose, so pacing may need adjustment."
            quality = "mixed"
        else:
            note = f"{action_name} did not yet improve transition readiness."
            quality = "stalled"
        return {
            "observer": self.name,
            "quality": quality,
            "note": note,
            "reward": round(float(reward), 4),
        }
