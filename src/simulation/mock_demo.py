from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STATE_KEYS = ("distress_level", "compliance", "attention", "escalation_risk")
ACTION_EFFECTS = {
    "transition_warning": {"distress_level": -0.06, "compliance": 0.12, "attention": 0.09, "escalation_risk": -0.05},
    "offer_choice": {"distress_level": -0.05, "compliance": 0.13, "attention": 0.05, "escalation_risk": -0.05},
    "labeled_praise": {"distress_level": -0.03, "compliance": 0.09, "attention": 0.07, "escalation_risk": -0.03},
    "visual_schedule_cue": {"distress_level": -0.05, "compliance": 0.09, "attention": 0.11, "escalation_risk": -0.04},
    "break_offer": {"distress_level": -0.08, "compliance": 0.05, "attention": 0.04, "escalation_risk": -0.07},
    "empathic_acknowledgment": {"distress_level": -0.10, "compliance": 0.04, "attention": 0.03, "escalation_risk": -0.07},
    "redirect_attention": {"distress_level": -0.04, "compliance": 0.07, "attention": 0.12, "escalation_risk": -0.04},
    "countdown_timer": {"distress_level": -0.05, "compliance": 0.10, "attention": 0.08, "escalation_risk": -0.04},
    "collaborative_problem_solving": {"distress_level": -0.07, "compliance": 0.08, "attention": 0.06, "escalation_risk": -0.05},
    "ignore_wait": {"distress_level": -0.02, "compliance": 0.01, "attention": 0.00, "escalation_risk": -0.02},
    "firm_boundary": {"distress_level": 0.01, "compliance": 0.14, "attention": 0.05, "escalation_risk": 0.00},
    "sensory_support": {"distress_level": -0.08, "compliance": 0.05, "attention": 0.05, "escalation_risk": -0.06},
}
ACTION_LINES = {
    "transition_warning": [
        "The advance warning gives the student a chance to brace for the switch.",
        "Hearing the warning again, the student checks how soon the change will happen.",
    ],
    "offer_choice": [
        "Having a small choice lowers the feeling of being cornered.",
        "The student latches onto the choice and starts negotiating the first step.",
    ],
    "labeled_praise": [
        "Specific praise makes the student more willing to keep the transition going.",
        "The student seems to register that the teacher noticed the effort, not just the outcome.",
    ],
    "visual_schedule_cue": [
        "Seeing the routine laid out helps the student orient to what comes next.",
        "The visual cue makes the next step feel more concrete than the verbal cue alone.",
    ],
    "break_offer": [
        "The short break offer takes the pressure out of the moment.",
        "Knowing there is a quick regulation option helps the student loosen up a little.",
    ],
    "empathic_acknowledgment": [
        "Feeling understood reduces some of the immediate resistance.",
        "The student softens when the teacher names why the shift feels hard.",
    ],
    "redirect_attention": [
        "A concrete focal point gives the student somewhere to put their attention.",
        "The teacher's redirection narrows the task enough for the student to follow it.",
    ],
    "countdown_timer": [
        "The countdown makes the transition feel more predictable.",
        "Watching the timer gives the student a clearer endpoint for the current activity.",
    ],
    "collaborative_problem_solving": [
        "The student responds better when the transition feels like a joint plan.",
        "Problem solving together lowers the sense of being pushed without explanation.",
    ],
    "ignore_wait": [
        "A brief pause prevents the interaction from escalating further.",
        "The student uses the quiet space to cool off slightly before re-engaging.",
    ],
    "firm_boundary": [
        "The firm limit clarifies that the transition is still happening now.",
        "The student hears that the teacher is steady and not reopening the decision.",
    ],
    "sensory_support": [
        "The sensory support gives the student something regulating to hold onto during the switch.",
        "The extra regulation tool makes it easier to move without spiking distress.",
    ],
}


class ScriptedStudentBackend:
    def generate(self, prompt: str) -> str:
        action = self._extract_action(prompt)
        state = self._extract_state(prompt) or self._extract_named_values(prompt)
        memory = self._extract_memory(prompt)
        traits = self._extract_traits(prompt)
        history_count = len(re.findall(r"Turn\s+\d+:", prompt))
        repeat_count = len(re.findall(rf"'{re.escape(action)}'", prompt)) if action else 0

        if not state:
            state = {key: 0.5 for key in STATE_KEYS}
        if not action:
            action = "transition_warning"

        next_state = self._next_state(state, action, memory, traits, history_count, repeat_count)
        narrative = self._compose_narrative(action, state, next_state, memory, history_count, repeat_count)
        return json.dumps({"state": next_state, "narrative": narrative}, ensure_ascii=False)

    def _extract_action(self, prompt: str) -> str | None:
        match = re.search(r"The clinician now uses:\s*([a-z_]+)", prompt)
        return match.group(1) if match else None

    def _extract_state(self, prompt: str) -> dict[str, float]:
        match = re.search(
            r"Current observed state:\s*distress_level=([0-9.]+),\s*compliance=([0-9.]+),\s*attention=([0-9.]+),\s*escalation_risk=([0-9.]+)",
            prompt,
        )
        if not match:
            return {}
        values = [float(value) for value in match.groups()]
        return dict(zip(STATE_KEYS, values))

    def _extract_named_values(self, prompt: str) -> dict[str, float]:
        state = {}
        for key in STATE_KEYS:
            match = re.search(rf"{key}['\"]?\s*[:=]\s*([0-9.]+)", prompt)
            if match:
                state[key] = float(match.group(1))
        return state

    def _extract_memory(self, prompt: str) -> dict[str, object]:
        trust = self._extract_float(prompt, r"teacher_trust=([0-9.]+)", 0.5)
        tolerance = self._extract_float(prompt, r"transition_tolerance=([0-9.]+)", 0.5)
        action_scores: dict[str, float] = {}
        match = re.search(r"effective_actions=([^\n]+?)(?:,\s*recent_triggers=|$)", prompt)
        if match:
            raw = match.group(1).strip()
            if raw and raw != "none":
                for piece in raw.split(","):
                    if ":" not in piece:
                        continue
                    name, score = piece.strip().split(":", 1)
                    try:
                        action_scores[name] = float(score)
                    except ValueError:
                        continue
        return {"teacher_trust": trust, "transition_tolerance": tolerance, "action_scores": action_scores}

    def _extract_traits(self, prompt: str) -> dict[str, float]:
        match = re.search(
            r"Traits:\s*impulsivity=([0-9.]+),\s*inattention=([0-9.]+),\s*emotional_reactivity=([0-9.]+)",
            prompt,
        )
        if not match:
            return {"impulsivity": 0.5, "inattention": 0.5, "emotional_reactivity": 0.5}
        impulsivity, inattention, emotional_reactivity = [float(value) for value in match.groups()]
        return {
            "impulsivity": impulsivity,
            "inattention": inattention,
            "emotional_reactivity": emotional_reactivity,
        }

    def _extract_float(self, text: str, pattern: str, default: float) -> float:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else default

    def _next_state(
        self,
        state: dict[str, float],
        action: str,
        memory: dict[str, object],
        traits: dict[str, float],
        history_count: int,
        repeat_count: int,
    ) -> dict[str, float]:
        delta = ACTION_EFFECTS.get(action, ACTION_EFFECTS["transition_warning"])
        trust = float(memory["teacher_trust"]) - 0.5
        tolerance = float(memory["transition_tolerance"]) - 0.5
        action_score = float(memory["action_scores"].get(action, 0.0))
        repeat_penalty = max(0, repeat_count - 1) * 0.03
        practice_bonus = min(0.12, history_count * 0.02)

        distress = state["distress_level"] + delta["distress_level"]
        distress -= 0.08 * max(0.0, trust) + 0.04 * action_score
        distress += 0.03 * max(0.0, -trust) + 0.04 * traits["emotional_reactivity"] + repeat_penalty
        if action == "firm_boundary" and trust < 0:
            distress += 0.04
        if action in {"empathic_acknowledgment", "sensory_support", "break_offer"}:
            distress -= 0.03

        compliance = state["compliance"] + delta["compliance"]
        compliance += 0.06 * max(0.0, trust) + 0.05 * action_score + practice_bonus
        compliance += 0.03 * max(0.0, tolerance)
        compliance -= 0.04 * traits["impulsivity"] + repeat_penalty

        attention = state["attention"] + delta["attention"]
        attention += 0.05 * max(0.0, tolerance) + 0.04 * action_score + practice_bonus
        attention -= 0.05 * traits["inattention"] + 0.02 * repeat_penalty

        escalation = state["escalation_risk"] + delta["escalation_risk"]
        escalation -= 0.06 * max(0.0, trust) + 0.05 * action_score
        escalation += 0.04 * traits["emotional_reactivity"] + 0.02 * max(0.0, -trust) + repeat_penalty
        if action == "firm_boundary" and state["distress_level"] > 0.6:
            escalation += 0.03

        if compliance > 0.72 and attention > 0.55:
            distress -= 0.03
            escalation -= 0.03

        next_state = {
            "distress_level": self._clamp(distress),
            "compliance": self._clamp(compliance),
            "attention": self._clamp(attention),
            "escalation_risk": self._clamp(escalation),
        }
        return next_state

    def _compose_narrative(
        self,
        action: str,
        prev_state: dict[str, float],
        next_state: dict[str, float],
        memory: dict[str, object],
        history_count: int,
        repeat_count: int,
    ) -> str:
        trust = float(memory["teacher_trust"])
        action_score = float(memory["action_scores"].get(action, 0.0))
        lines = ACTION_LINES.get(action, ACTION_LINES["transition_warning"])
        action_line = lines[history_count % len(lines)]

        if history_count == 0:
            opener = "The student is still locked into the original activity and needs a beat before looking up."
        elif repeat_count > 1 and action_score < 0.04:
            opener = "The student notices the same cue returning and responds, but with less spark than the first time."
        elif history_count >= 3 or action_score >= 0.04 or trust >= 0.56:
            opener = "Because the teacher has stayed predictable across turns, the student orients faster this time."
        elif trust <= 0.45:
            opener = "The student hesitates at first and checks whether the teacher will stay steady through the transition."
        elif history_count == 1:
            opener = "The student pauses and starts checking the teacher's cues more deliberately."
        else:
            opener = "The student tests whether this cue feels manageable before committing to the transition."

        if next_state["compliance"] >= 0.82:
            closer = "They move into the next step with only mild hesitation."
        elif next_state["distress_level"] >= 0.62:
            closer = "Their body is still tense, and they stay partly pulled toward the old activity."
        elif next_state["attention"] > prev_state["attention"]:
            closer = "Attention is starting to shift, even if the student is not fully settled yet."
        else:
            closer = "The response is mixed, and the student still needs support to fully transition."

        return f"{opener} {action_line} {closer}"

    def _clamp(self, value: float) -> float:
        return round(max(0.0, min(1.0, value)), 3)


def _rr(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=min(radius, 4), fill=fill, outline=outline, width=width)


def _bubble(draw, box, title, text, fill=(248, 251, 255), ink=(12, 19, 34)):
    _rr(draw, box, 18, fill)
    draw.text((box[0] + 14, box[1] + 12), title, fill=(109, 123, 152), font=FONT_SMALL)
    draw.text((box[0] + 14, box[1] + 34), text[:68], fill=ink, font=FONT_SMALL)


def _character(draw, center, body_fill, label, glow, head_fill=(247, 235, 220)):
    x, y = center
    draw.rounded_rectangle((x - 62, y - 74, x + 62, y + 50), radius=4, fill=glow)
    draw.rounded_rectangle((x - 28, y - 72, x + 28, y - 16), radius=4, fill=head_fill, outline=(255, 255, 255), width=4)
    _rr(draw, (x - 38, y - 20, x + 38, y + 62), 4, body_fill)
    draw.rectangle((x - 12, y - 48, x - 4, y - 40), fill=(30, 36, 50))
    draw.rectangle((x + 4, y - 48, x + 12, y - 40), fill=(30, 36, 50))
    draw.text((x - 30, y + 78), label, fill=(222, 232, 249), font=FONT_SMALL)


def _metric(draw, x, y, w, label, value, color):
    draw.text((x, y), f"{label} {value:.2f}", fill=(150, 164, 194), font=FONT_SMALL)
    _rr(draw, (x, y + 24, x + w, y + 40), 8, (41, 52, 78))
    _rr(draw, (x, y + 24, x + int(w * max(0.0, min(1.0, value))), y + 40), 8, color)


FONT_TITLE = ImageFont.load_default()
FONT_BODY = ImageFont.load_default()
FONT_SMALL = ImageFont.load_default()


def export_preview_png(log_data: dict, output_path: str, title: str = "Classroom ADHD Simulation Preview") -> None:
    session = log_data["sessions"][0]
    event = session["events"][-1]
    state = event["student_state"]
    img = Image.new("RGB", (1600, 900), (8, 14, 25))
    draw = ImageDraw.Draw(img)

    _rr(draw, (28, 24, 1572, 104), 22, (16, 25, 44))
    draw.text((56, 42), title, fill=(238, 243, 255), font=FONT_TITLE)
    draw.text((56, 68), f"Session {session['session_id']} | Scenario: {session['scenario']['name']} | Profile: {session['profile']['name']}", fill=(150, 164, 194), font=FONT_SMALL)
    draw.text((1224, 56), f"Status: {session['summary']['status']} | Reward: {session['summary']['total_reward']:.2f}", fill=(112, 228, 165), font=FONT_SMALL)

    _rr(draw, (28, 130, 1060, 860), 30, (149, 201, 78))
    _rr(draw, (52, 154, 1036, 836), 26, (221, 214, 154), outline=(122, 139, 60), width=4)
    for x in range(52, 1036, 36):
        draw.line((x, 154, x, 836), fill=(174, 167, 114), width=1)
    for y in range(154, 836, 36):
        draw.line((52, y, 1036, y), fill=(174, 167, 114), width=1)

    _rr(draw, (364, 176, 716, 246), 18, (92, 118, 76), outline=(236, 243, 228), width=4)
    draw.text((484, 204), "FRONT BOARD", fill=(241, 248, 237), font=FONT_SMALL)
    _rr(draw, (444, 278, 592, 356), 18, (241, 245, 252), outline=(94, 108, 129), width=3)
    draw.text((472, 306), "Teacher Desk", fill=(73, 87, 112), font=FONT_SMALL)
    _rr(draw, (90, 222, 334, 392), 24, (243, 246, 252), outline=(112, 126, 148), width=3)
    draw.text((120, 252), "Reading Corner", fill=(73, 87, 112), font=FONT_SMALL)
    _rr(draw, (718, 222, 970, 392), 24, (243, 246, 252), outline=(112, 126, 148), width=3)
    draw.text((748, 252), "Cleanup Station", fill=(73, 87, 112), font=FONT_SMALL)
    _rr(draw, (388, 220, 648, 398), 24, (243, 246, 252), outline=(112, 126, 148), width=3)
    draw.text((438, 252), "Transition Lane", fill=(73, 87, 112), font=FONT_SMALL)
    _rr(draw, (90, 594, 334, 774), 24, (243, 246, 252), outline=(112, 126, 148), width=3)
    draw.text((132, 626), "Quiet Reset", fill=(73, 87, 112), font=FONT_SMALL)
    _rr(draw, (718, 594, 970, 774), 24, (243, 246, 252), outline=(112, 126, 148), width=3)
    draw.text((770, 626), "Group Tables", fill=(73, 87, 112), font=FONT_SMALL)
    _rr(draw, (416, 774, 664, 816), 18, (174, 202, 230), outline=(92, 110, 128), width=3)
    draw.text((488, 788), "Hallway Door", fill=(57, 71, 90), font=FONT_SMALL)

    for box in ((390, 442, 500, 512), (538, 442, 648, 512), (390, 548, 500, 618), (538, 548, 648, 618)):
        _rr(draw, box, 16, (199, 178, 107), outline=(124, 101, 43), width=3)

    draw.rounded_rectangle((504, 360, 556, 740), radius=22, fill=(246, 248, 252), outline=None)
    for y in range(368, 730, 28):
        draw.rectangle((520, y, 540, y + 14), fill=(201, 210, 220))

    _character(draw, (520, 400), (42, 65, 97), "MS. LEE", (37, 83, 76), head_fill=(229, 245, 241))
    _character(draw, (520, 664), (112, 131, 255), session['profile']['name'][:14].upper(), (56, 77, 170))
    _character(draw, (248, 466), (205, 216, 234), "JIN", (110, 120, 146), head_fill=(250, 251, 255))
    _character(draw, (812, 466), (205, 216, 234), "MINA", (110, 120, 146), head_fill=(250, 251, 255))
    _character(draw, (248, 678), (205, 216, 234), "HARU", (110, 120, 146), head_fill=(250, 251, 255))
    _character(draw, (812, 678), (205, 216, 234), "SOO", (110, 120, 146), head_fill=(250, 251, 255))

    _bubble(draw, (364, 282, 680, 376), event["speaker"], event.get("utterance", ""))
    _bubble(draw, (362, 500, 678, 602), "STUDENT RESPONSE", event.get("student_narrative", "Waiting for the next cue."))
    if event.get("peer_reactions"):
        _bubble(draw, (108, 392, 352, 480), "PEER", event["peer_reactions"][0]["utterance"])

    _rr(draw, (1092, 130, 1572, 860), 30, (16, 25, 44), outline=(34, 48, 74))
    draw.text((1124, 162), "SITUATION BRIEF", fill=(150, 164, 194), font=FONT_SMALL)
    draw.text((1124, 190), session["scenario"]["name"].replace("_", " "), fill=(238, 243, 255), font=FONT_BODY)
    draw.text((1124, 220), session["scenario"].get("behavioral_rationale", session["scenario"]["description"])[:118], fill=(150, 164, 194), font=FONT_SMALL)

    draw.text((1124, 292), "LIVE EVALUATION", fill=(150, 164, 194), font=FONT_SMALL)
    _metric(draw, 1124, 324, 392, "Distress", state["distress_level"], (241, 123, 147))
    _metric(draw, 1124, 388, 392, "Compliance", state["compliance"], (103, 217, 154))
    _metric(draw, 1124, 452, 392, "Attention", state["attention"], (122, 202, 255))
    _metric(draw, 1124, 516, 392, "Escalation", state["escalation_risk"], (255, 191, 105))

    draw.text((1124, 590), "OBSERVER NOTE", fill=(150, 164, 194), font=FONT_SMALL)
    draw.text((1124, 618), event.get("observer_note", {}).get("note", "")[:120], fill=(220, 228, 246), font=FONT_SMALL)
    draw.text((1124, 682), "MEMORY", fill=(150, 164, 194), font=FONT_SMALL)
    draw.text((1124, 710), session["summary"]["memory"][:120], fill=(220, 228, 246), font=FONT_SMALL)
    draw.text((1124, 786), f"Turn {event['time']} | Reward {event.get('total_reward', 0):.2f}", fill=(112, 228, 165), font=FONT_SMALL)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)
