from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.agent.baselines import RuleBasedAgent
from src.simulation.agents import ObserverAgent, PeerStudentAgent, TeacherAgent
from src.simulation.memory import MemoryStore


@dataclass
class ClassroomLayout:
    teacher_position: dict[str, int] = field(default_factory=lambda: {"x": 50, "y": 12})
    target_student_position: dict[str, int] = field(default_factory=lambda: {"x": 50, "y": 70})
    peer_positions: list[tuple[int, int]] = field(
        default_factory=lambda: [(22, 42), (78, 42), (24, 80), (76, 80)]
    )


class ClassroomWorld:
    def __init__(
        self,
        env,
        teacher_policy=None,
        memory_store: MemoryStore | None = None,
        peer_names: list[str] | None = None,
        observer: ObserverAgent | None = None,
        layout: ClassroomLayout | None = None,
    ):
        self.env = env
        self.memory_store = memory_store or getattr(env, "memory_store", None) or MemoryStore()
        self.env.memory_store = self.memory_store
        self.teacher = TeacherAgent(teacher_policy or RuleBasedAgent())
        self.layout = layout or ClassroomLayout()
        self.peers = [
            PeerStudentAgent(name=name, seat=seat)
            for name, seat in zip(
                peer_names or ["Jin", "Mina", "Haru", "Soo"],
                self.layout.peer_positions,
            )
        ]
        self.observer = observer or ObserverAgent()

    def run_session(self, session_id: int = 1) -> dict[str, object]:
        obs, info = self.env.reset()
        profile = self.env.current_profile
        scenario = self.env.current_scenario
        memory = self.memory_store.get(profile.name)

        events = [
            {
                "time": 0,
                "scene": scenario.name,
                "scene_type": scenario.type,
                "acting_agent": "environment",
                "speaker": "Narrator",
                "utterance": info["narrative"],
                "action": "scene_setup",
                "agent_positions": self._agent_positions(profile.name),
                "student_state": dict(self.env.current_state),
                "peer_reactions": [],
                "observer_note": {
                    "observer": self.observer.name,
                    "quality": "baseline",
                    "note": scenario.behavioral_rationale
                    or "Session starts from the scenario's transition context.",
                    "reward": 0.0,
                },
                "memory_update": memory.summary(),
                "termination_status": "running",
                "session_id": session_id,
            }
        ]

        total_reward = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            prev_state = dict(self.env.current_state)
            action_idx, action_name = self.teacher.select_action(obs)
            teacher_utterance = self.teacher.narrate_action(action_name, scenario.name)

            obs, reward, terminated, truncated, step_info = self.env.step(action_idx)
            total_reward += reward

            peer_reactions = [peer.react(self.env.current_state, action_name) for peer in self.peers]
            observer_note = self.observer.score_tick(
                prev_state=prev_state,
                next_state=self.env.current_state,
                reward=reward,
                action_name=action_name,
            )

            if terminated and self.env.current_state["compliance"] > self.env.success_threshold:
                status = "success"
            elif terminated:
                status = "failure"
            elif truncated:
                status = "timeout"
            else:
                status = "running"

            events.append(
                {
                    "time": step_info["turn"],
                    "scene": scenario.name,
                    "scene_type": scenario.type,
                    "acting_agent": self.teacher.name,
                    "speaker": self.teacher.name,
                    "utterance": teacher_utterance,
                    "action": action_name,
                    "agent_positions": self._agent_positions(profile.name),
                    "student_state": dict(self.env.current_state),
                    "student_narrative": step_info["narrative"],
                    "peer_reactions": peer_reactions,
                    "observer_note": observer_note,
                    "reward": round(float(reward), 4),
                    "total_reward": round(float(total_reward), 4),
                    "memory_update": step_info.get("memory"),
                    "termination_status": status,
                    "session_id": session_id,
                }
            )

        return {
            "session_id": session_id,
            "profile": {
                "name": profile.name,
                "age": profile.age,
                "severity": profile.severity,
                "description": profile.description,
            },
            "scenario": {
                "name": scenario.name,
                "type": scenario.type,
                "description": scenario.description,
                "behavioral_rationale": scenario.behavioral_rationale,
            },
            "events": events,
            "summary": {
                "status": events[-1]["termination_status"],
                "turns": self.env.turn,
                "total_reward": round(float(total_reward), 4),
                "final_state": dict(self.env.current_state),
                "memory": memory.summary(),
            },
        }

    def run_sessions(self, n_sessions: int = 3) -> dict[str, object]:
        sessions = [self.run_session(session_id=i + 1) for i in range(n_sessions)]
        return {
            "sessions": sessions,
            "memory_snapshot": self.memory_store.snapshot(),
        }

    def save_sessions(self, output_path: str, n_sessions: int = 3) -> dict[str, object]:
        log_data = self.run_sessions(n_sessions=n_sessions)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        return log_data

    def _agent_positions(self, student_name: str) -> dict[str, object]:
        return {
            "teacher": dict(self.layout.teacher_position),
            student_name: dict(self.layout.target_student_position),
            "peers": [
                {"name": peer.name, "seat": dict(peer.seat)}
                for peer in self.peers
            ],
        }
