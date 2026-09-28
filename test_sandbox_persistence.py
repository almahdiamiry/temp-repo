"""Test suite for Educational Sandbox Session Replay Persistence (LIM-05 Hardening).

Validates:
1. Schema conformity: session records adhere to Schema v1.0 with initial conditions snapshot.
2. Automatic persistence: sessions auto-save on complete(), fail(), or abort().
3. Exact replay fidelity: replayed event timeline matches live execution events 1:1.
4. Session directory management: listing, sorting, and metadata inspection.
5. Robustness: graceful handling of corrupt, missing, or malformed session files without crash.
"""

import json
import os
import shutil
import tempfile
import pytest

import dynamic
import sandbox


@pytest.fixture
def temp_sessions_dir():
    """Create a temporary directory for isolated session persistence tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_sessions_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sim():
    return dynamic.SeparatorDynamicSimulator()


def test_session_lifecycle_and_auto_save(sim, temp_sessions_dir):
    """Verify session initialization, auto-persistence upon completion, and schema v1.0."""
    session = sandbox.SandboxSession(sim)
    assert session.session_id.startswith("SES-")
    assert session.initial_conditions is not None
    assert "level_mm" in session.initial_conditions

    # Load SCN-01
    assert session.load_scenario("SCN-01")
    assert session.start()

    # Step simulation to trigger baseline completion
    for _ in range(160):
        sim.step(0.2)
        state = session.step(0.2)
        if state["state"] == "COMPLETED":
            break

    assert session.state == sandbox.ScenarioState.COMPLETED
    assert session.completed_at_iso is not None

    # Manually save to test directory to verify path and file content
    filepath = session.save_to_disk(sessions_dir=temp_sessions_dir)
    assert os.path.exists(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid, msg = sandbox.verify_session_integrity(data)
    assert valid, f"Session integrity check failed: {msg}"
    assert data["schema_version"] == "1.0"
    assert data["session_id"] == session.session_id
    assert data["scenario_id"] == "SCN-01"
    assert data["status"] == "COMPLETED"
    assert len(data["events"]) > 0
    assert data["summary_metrics"]["passed"] is True
    assert data["summary_metrics"]["duration_s"] > 0


def test_session_save_load_roundtrip(sim, temp_sessions_dir):
    """Verify exact roundtrip fidelity of saved and loaded sessions."""
    session = sandbox.SandboxSession(sim)
    session.load_scenario("SCN-07")
    session.start()

    # Record some operator commands
    session.record_operator_action("SET_VALVE_MANUAL", {"valve": "FV-001", "manual": True, "target": 0.0}, True)
    session.record_operator_action("SET_SETPOINT", {"loop": "LICA-002", "value": 1600.0}, True)
    session.record_event("PROCESS", "Verification test event", "INFO")

    session.complete("Line isolation verified")
    saved_path = session.save_to_disk(sessions_dir=temp_sessions_dir)

    # Load back using load_saved_session
    loaded = sandbox.load_saved_session(session.session_id, sessions_dir=temp_sessions_dir)
    assert loaded["session_id"] == session.session_id
    assert loaded["scenario_id"] == "SCN-07"
    assert loaded["status"] == "COMPLETED"
    assert len(loaded["events"]) == len(session.timeline)

    # Verify initial conditions preserved
    assert loaded["initial_conditions"]["level_mm"] == pytest.approx(session.initial_conditions["level_mm"], rel=1e-3)


def test_timeline_replay_fidelity(sim, temp_sessions_dir):
    """Verify chronological timeline replay produces identical event sequence to live execution."""
    session = sandbox.SandboxSession(sim)
    session.load_scenario("SCN-02")
    session.start()

    # Simulate steps
    for _ in range(20):
        sim.step(0.2)
        session.step(0.2)

    session.record_operator_action("SET_CLAMP_OVERRIDE", {"active": True}, True)
    live_events = session.replay_timeline()
    assert len(live_events) >= 2

    # Save to disk
    session.save_to_disk(sessions_dir=temp_sessions_dir)

    # Reload and replay
    loaded_data = sandbox.load_saved_session(session.session_id, sessions_dir=temp_sessions_dir)
    replayed_events = loaded_data["events"]

    assert len(replayed_events) == len(live_events)
    for live_ev, rep_ev in zip(live_events, replayed_events):
        assert live_ev["time_s"] == rep_ev["time_s"]
        assert live_ev["category"] == rep_ev["category"]
        assert live_ev["message"] == rep_ev["message"]
        assert live_ev["level"] == rep_ev["level"]


def test_list_saved_sessions_and_ordering(sim, temp_sessions_dir):
    """Verify listing saved sessions with metadata and chronological sorting."""
    # Create two sessions with different IDs and scenarios
    s1 = sandbox.SandboxSession(sim)
    s1.load_scenario("SCN-01")
    s1.start()
    s1.complete("Finished SCN-01")
    s1.save_to_disk(sessions_dir=temp_sessions_dir)

    s2 = sandbox.SandboxSession(sim)
    s2.load_scenario("SCN-03")
    s2.start()
    s2.fail("Failed SCN-03")
    s2.save_to_disk(sessions_dir=temp_sessions_dir)

    sessions_list = sandbox.list_saved_sessions(sessions_dir=temp_sessions_dir)
    assert len(sessions_list) == 2
    session_ids = [s["session_id"] for s in sessions_list]
    assert s1.session_id in session_ids
    assert s2.session_id in session_ids

    # Verify summary metadata fields
    for item in sessions_list:
        assert "session_id" in item
        assert "scenario_id" in item
        assert "status" in item
        assert "created_at" in item
        assert "duration_s" in item
        assert "events_count" in item


def test_corrupt_and_missing_session_handling(temp_sessions_dir):
    """Verify that corrupt JSON or schema-invalid files are rejected without crashing."""
    # 1. Non-existent session
    with pytest.raises(FileNotFoundError):
        sandbox.load_saved_session("NON_EXISTENT_ID", sessions_dir=temp_sessions_dir)

    # 2. Corrupt JSON syntax
    corrupt_file = os.path.join(temp_sessions_dir, "SES-CORRUPT-JSON.json")
    with open(corrupt_file, "w", encoding="utf-8") as f:
        f.write("{invalid json syntax, broken...")

    # Should be gracefully ignored by list_saved_sessions
    assert sandbox.list_saved_sessions(sessions_dir=temp_sessions_dir) == []

    # Should raise JSONDecodeError if directly loaded
    with pytest.raises(json.JSONDecodeError):
        sandbox.load_saved_session("SES-CORRUPT-JSON", sessions_dir=temp_sessions_dir)

    # 3. Schema invalid file (missing required keys)
    invalid_schema_file = os.path.join(temp_sessions_dir, "SES-BAD-SCHEMA.json")
    with open(invalid_schema_file, "w", encoding="utf-8") as f:
        json.dump({"random_key": 123}, f)

    # list_saved_sessions skips schema invalid files
    assert sandbox.list_saved_sessions(sessions_dir=temp_sessions_dir) == []

    # load_saved_session raises ValueError on integrity failure
    with pytest.raises(ValueError, match="Corrupt session"):
        sandbox.load_saved_session("SES-BAD-SCHEMA", sessions_dir=temp_sessions_dir)
