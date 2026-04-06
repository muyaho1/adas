from src.environment.transition_constraints import DEFAULT_ACTION_CONSTRAINTS, constrain_transition


def test_all_action_constraints_have_evidence():
    assert len(DEFAULT_ACTION_CONSTRAINTS) == 12
    for constraint in DEFAULT_ACTION_CONSTRAINTS.values():
        assert constraint.evidence
        assert constraint.rationale


def test_constrain_transition_clips_excessive_positive_jump():
    prev_state = {
        "distress_level": 0.4,
        "compliance": 0.2,
        "attention": 0.2,
        "escalation_risk": 0.3,
    }
    proposed_state = {
        "distress_level": 1.0,
        "compliance": 1.0,
        "attention": 1.0,
        "escalation_risk": 1.0,
    }

    clipped = constrain_transition("transition_warning", prev_state, proposed_state)

    assert clipped["compliance"] <= 0.38
    assert clipped["attention"] <= 0.4
    assert clipped["distress_level"] <= 0.480001
    assert clipped["escalation_risk"] <= 0.38


def test_constrain_transition_preserves_state_range():
    prev_state = {
        "distress_level": 0.05,
        "compliance": 0.95,
        "attention": 0.95,
        "escalation_risk": 0.05,
    }
    proposed_state = {
        "distress_level": -1.0,
        "compliance": 2.0,
        "attention": 2.0,
        "escalation_risk": -1.0,
    }

    clipped = constrain_transition("labeled_praise", prev_state, proposed_state)

    assert all(0.0 <= value <= 1.0 for value in clipped.values())

