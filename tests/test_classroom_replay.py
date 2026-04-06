from src.ui.classroom_replay import export_replay_html



def test_export_replay_html_embeds_session_log(tmp_path):
    log_data = {
        "sessions": [
            {
                "session_id": 1,
                "profile": {"name": "moderate_combined", "severity": "moderate"},
                "scenario": {"name": "recess_to_math", "type": "preferred_to_nonpreferred", "description": "transition"},
                "events": [
                    {
                        "session_id": 1,
                        "time": 0,
                        "scene": "recess_to_math",
                        "speaker": "Narrator",
                        "utterance": "Recess is ending.",
                        "action": "scene_setup",
                        "agent_positions": {
                            "teacher": {"x": 50, "y": 12},
                            "moderate_combined": {"x": 50, "y": 70},
                            "peers": [{"name": "Jin", "seat": {"x": 22, "y": 42}}],
                        },
                        "student_state": {
                            "distress_level": 0.3,
                            "compliance": 0.2,
                            "attention": 0.2,
                            "escalation_risk": 0.3,
                        },
                        "observer_note": {"note": "baseline"},
                        "memory_update": "teacher_trust=0.50",
                        "termination_status": "running",
                    }
                ],
                "summary": {"memory": "teacher_trust=0.50"},
            }
        ],
        "memory_snapshot": {},
    }
    output_path = tmp_path / "replay.html"

    export_replay_html(log_data, str(output_path), title="Classroom Replay")

    content = output_path.read_text(encoding="utf-8")
    assert "Classroom Replay" in content
    assert "recess_to_math" in content
    assert "teacher_trust=0.50" in content
    assert 'function escapeHtml(value)' in content
    assert '""":"&quot;"' not in content
    assert '<div class="controlDock"><div class="controlTray"><button class="ghost" id="prev">Back</button><button id="play">Play</button><button class="ghost" id="next">Next</button></div></div>' in content
    assert ".tb{left:50%;margin-left:-120px;top:calc(100% + 16px)}" in content
    assert ".sb{left:50%;margin-left:-120px;bottom:calc(100% + 16px)}" in content
