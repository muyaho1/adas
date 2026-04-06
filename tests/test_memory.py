from src.environment.child_profiles import ChildProfile
from src.environment.scenarios import Scenario
from src.simulation.memory import MemoryStore, StudentSessionMemory



def test_memory_update_increases_trust_after_successful_transition():
    memory = StudentSessionMemory(student_id="moderate_combined")
    prev_state = {
        "distress_level": 0.6,
        "compliance": 0.2,
        "attention": 0.3,
        "escalation_risk": 0.4,
    }
    next_state = {
        "distress_level": 0.3,
        "compliance": 0.7,
        "attention": 0.6,
        "escalation_risk": 0.2,
    }

    memory.update(
        action_name="labeled_praise",
        prev_state=prev_state,
        next_state=next_state,
        reward=1.0,
        scenario_type="preferred_to_nonpreferred",
        terminated_successfully=True,
    )
    memory.mark_session_complete()

    assert memory.teacher_trust > 0.5
    assert memory.transition_tolerance > 0.5
    assert memory.sessions_seen == 1
    assert memory.strategy_response_history["labeled_praise"] > 0



def test_memory_adjusts_initial_state_in_safer_direction_when_trust_is_high():
    memory = StudentSessionMemory(
        student_id="mild_inattentive",
        teacher_trust=0.8,
        transition_tolerance=0.75,
    )
    profile = ChildProfile(
        name="mild_inattentive",
        age=8,
        severity="mild",
        traits={"inattention": 0.6},
        description="mild profile",
    )
    scenario = Scenario(
        name="reading_to_cleanup",
        type="hyperfocus_interruption",
        description="cleanup",
        initial_state={
            "distress_level": 0.3,
            "compliance": 0.2,
            "attention": 0.2,
            "escalation_risk": 0.3,
        },
    )

    adjustment = memory.initial_state_adjustment(profile, scenario)

    assert adjustment["distress_level"] < 0
    assert adjustment["compliance"] > 0
    assert adjustment["escalation_risk"] < 0



def test_memory_store_returns_same_object_for_same_student():
    store = MemoryStore()
    first = store.get("student-a")
    second = store.get("student-a")

    assert first is second
    assert "student-a" in store.snapshot()
