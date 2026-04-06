import json

from src.simulation.mock_demo import ScriptedStudentBackend


BASE_PROMPT = """You are simulating a 9-year-old child with moderate ADHD.
Traits: impulsivity=0.6, inattention=0.55, emotional_reactivity=0.5
Scenario: Reading is ending and cleanup is starting.
Scenario rationale: Switching away from a preferred task is hard.
Scenario priors: {{'distress_level': 0.4, 'compliance': 0.1, 'attention': 0.2, 'escalation_risk': 0.3}}
Current observed state: distress_level=0.420, compliance=0.180, attention=0.220, escalation_risk=0.360

Previous interactions:
Turn 1: Clinician used 'transition_warning'. Child: The student paused and looked up.

Long-term student memory:
teacher_trust={teacher_trust:.2f}, transition_tolerance={transition_tolerance:.2f}, effective_actions={effective_actions}, recent_triggers=hyperfocus_interruption

The clinician now uses: {action}
"""


def test_scripted_student_backend_varies_by_action():
    backend = ScriptedStudentBackend()

    warning = json.loads(
        backend.generate(
            BASE_PROMPT.format(
                teacher_trust=0.50,
                transition_tolerance=0.50,
                effective_actions="transition_warning:0.03",
                action="transition_warning",
            )
        )
    )
    praise = json.loads(
        backend.generate(
            BASE_PROMPT.format(
                teacher_trust=0.50,
                transition_tolerance=0.50,
                effective_actions="labeled_praise:0.03",
                action="labeled_praise",
            )
        )
    )

    assert warning["narrative"] != praise["narrative"]
    assert warning["state"] != praise["state"]



def test_scripted_student_backend_uses_memory_strength():
    backend = ScriptedStudentBackend()

    low_memory = json.loads(
        backend.generate(
            BASE_PROMPT.format(
                teacher_trust=0.42,
                transition_tolerance=0.45,
                effective_actions="transition_warning:0.01",
                action="transition_warning",
            )
        )
    )
    high_memory = json.loads(
        backend.generate(
            BASE_PROMPT.format(
                teacher_trust=0.78,
                transition_tolerance=0.74,
                effective_actions="transition_warning:0.12",
                action="transition_warning",
            )
        )
    )

    assert high_memory["state"]["compliance"] > low_memory["state"]["compliance"]
    assert high_memory["state"]["distress_level"] < low_memory["state"]["distress_level"]
