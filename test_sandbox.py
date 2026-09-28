"""Phase 5 Educational Sandbox Verification Suite for CP2-V-71101.

Validates the Phase 5 Educational Sandbox Acceptance Gate:
1. Single Authority Preservation: Sandbox orchestrates learning; dynamic.py simulates process physics.
2. Scenario Catalog Completeness: All 7 scenarios (SCN-01 to SCN-07) implemented with bilingual objectives.
3. Scenario Lifecycle: READY -> ACTIVE -> COMPLETED / FAILED -> DEBRIEF, with clean RESET and ABORT.
4. Real-Time Event Timeline: Chronological recording of commands, alarms, interlocks, and scenarios.
5. Post-Mission Debriefing: Rigorous engineering analysis (outcomes, physical causes, design lessons)
   without game arcade mechanics (no XP, badges, leaderboards).
6. Note 26 Interlock Integration: Gas blow-by scenario triggers authentic protective closure.
7. Operating World Bridge HTTP Endpoints: /api/sandbox/* endpoints operational.
8. Browser Smoke Test: UI drawer, timeline feed, and scenario interaction verified.
"""

import json
import os
import sys
import time
import urllib.request
import pytest

sys.path.insert(0, os.path.abspath('simulator/stage1'))
import dynamic
import sandbox
import operating_world


@pytest.fixture
def sim():
    """Create a fresh SeparatorDynamicSimulator instance."""
    return dynamic.SeparatorDynamicSimulator("Case 1 x1.4")


@pytest.fixture
def session(sim):
    """Create a fresh SandboxSession instance."""
    return sandbox.SandboxSession(sim)


@pytest.fixture
def bridge():
    """Create a fresh OperatingWorldBridge instance with sandbox enabled."""
    return operating_world.OperatingWorldBridge()


# --- 1. Scenario Catalog Verification ---

def test_scenario_catalog_completeness():
    """Verify that all 7 required scenarios (SCN-01 through SCN-07) are registered."""
    expected_ids = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05", "SCN-06", "SCN-07"]
    for scn_id in expected_ids:
        assert scn_id in sandbox.SCENARIO_CATALOG, f"Scenario {scn_id} missing from catalog"
        scn = sandbox.SCENARIO_CATALOG[scn_id]
        assert scn.id == scn_id
        assert len(scn.title_en) > 0 and len(scn.title_ar) > 0
        assert len(scn.description_en) > 0 and len(scn.description_ar) > 0
        assert len(scn.learning_objective_en) > 0 and len(scn.learning_objective_ar) > 0
        assert len(scn.initial_condition_desc_en) > 0 and len(scn.initial_condition_desc_ar) > 0
        assert callable(scn.setup_func)
        assert callable(scn.evaluation_func)
        assert isinstance(scn.debrief_info, dict)
        assert "key_takeaway" in scn.debrief_info or "physical_cause_en" in scn.debrief_info


# --- 2. Scenario Lifecycle Verification ---

def test_scenario_lifecycle_transitions(session):
    """Test standard state transitions: READY -> ACTIVE -> ABORTED / RESET."""
    # 1. Load scenario
    loaded = session.load_scenario("SCN-01")
    assert loaded is True
    assert session.state == sandbox.ScenarioState.READY
    assert session.active_scenario.id == "SCN-01"
    assert len(session.timeline) == 1
    assert session.timeline[0].category == "SCENARIO"

    # 2. Start scenario
    session.start()
    assert session.state == sandbox.ScenarioState.ACTIVE
    assert len(session.timeline) == 2

    # 3. Abort scenario
    session.abort()
    assert session.state == sandbox.ScenarioState.ABORTED
    assert session.timeline[-1].level == "WARNING"

    # 4. Clean reset
    session.reset()
    assert session.state == sandbox.ScenarioState.READY


# --- 3. Scenario SCN-01: Normal Operation Baseline ---

def test_scn01_normal_baseline_execution(session):
    """Test SCN-01 baseline execution to successful completion."""
    session.load_scenario("SCN-01")
    session.start()
    assert session.state == sandbox.ScenarioState.ACTIVE

    # Step simulation through 32 seconds (160 steps of 0.2s)
    for _ in range(160):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)

    assert session.state == sandbox.ScenarioState.COMPLETED
    assert session.debrief is not None
    assert session.debrief["passed"] is True
    assert session.debrief["result"] == "COMPLETED"
    assert session.debrief["final_metrics"]["level_mm"] == pytest.approx(1550.0, abs=15.0)
    assert session.debrief["final_metrics"]["pressure_barg"] == pytest.approx(5.0, abs=0.2)


# --- 4. Scenario SCN-02: Cold Startup & Dynamic Fill ---

def test_scn02_cold_startup_setup_and_fill(session):
    """Test SCN-02 cold startup: initial dry level (<=50 mm) and fluid accumulation."""
    session.load_scenario("SCN-02")
    session.start()
    
    # Process authority starts dry (initial 50 mm avoids singularity)
    assert session.sim.state.level_mm <= 50.0
    assert session.sim.valv_fv001.position == 0.0

    # Step 50 seconds: liquid should accumulate above dry level
    for _ in range(250):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)

    assert session.sim.state.level_mm > 400.0
    assert len(session.timeline) >= 2
    assert "filling" in session.status_message_en.lower()


# --- 5. Scenario SCN-03: 1.4x Surge Challenge & Supervisor Clamp Override ---

def test_scn03_surge_and_clamp_override(session):
    """Test SCN-03 surge scenario and response to supervisor clamp override."""
    session.load_scenario("SCN-03")
    session.start()

    # Initial condition: high feed 140% design, clamp override inactive
    assert session.sim.case_name == "Case 1 x1.4"
    assert session.sim.state.clamp_override is False

    # Simulate 10 seconds: level rises due to 100k outflow clamp restriction
    init_lvl = session.sim.state.level_mm
    for _ in range(50):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
    assert session.sim.state.level_mm > init_lvl

    # Operator enables supervisor clamp override
    session.sim.apply_command("SET_CLAMP_OVERRIDE", {"active": True})
    session.record_operator_action("SET_CLAMP_OVERRIDE", {"active": True}, True)
    assert session.sim.state.clamp_override is True

    # Outflow can now exceed the 794.94 m3/h baseline cap to handle surge
    for _ in range(50):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
    assert session.sim.state.out_liquid_m3_h > 795.0


# --- 6. Scenario SCN-04: High Pressure & Split-Range Flaring ---

def test_scn04_split_range_flare_operation(session):
    """Test SCN-04 gas disruption: PV-003B restriction causes split-range flare via PV-003A."""
    session.load_scenario("SCN-04")
    session.start()

    # PV-003B is restricted
    assert session.sim.valv_pv003b.manual is True
    assert session.sim.valv_pv003b.position <= 0.35

    # Advance simulation: pressure rises toward flare threshold (7.0 barg)
    for _ in range(80):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)

    # Pressure must have triggered flaring
    assert session.sim.state.pressure_barg > 5.5
    assert session.sim.state.out_gas_flare_kg_h > 0.0 or session.sim.valv_pv003a.position > 0.0
    assert any("flaring" in ev.message.lower() or "flare" in ev.message.lower() for ev in session.timeline)


# --- 7. Scenario SCN-06: Gas Blow-by Emergency & Note 26 Interlock ---

def test_scn06_note26_gas_blowby_interlock(session):
    """Test SCN-06 gas blow-by: LALL 770 mm trips UZV-002, which automatically trips gas UZV-003."""
    session.load_scenario("SCN-06")
    session.start()

    # Step until LALL trips or 30s
    for _ in range(150):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
        if session.sim.state.alarm_status == "TRIP":
            break

    # Note 26 interlock: UZV-002 closed MUST cause UZV-003 closed
    assert session.sim.state.uzv002_open is False
    assert session.sim.state.uzv003_open is False
    assert any("note 26" in ev.message.lower() for ev in session.timeline)


# --- 8. Event Timeline Monotonicity & Tagging ---

def test_timeline_monotonicity_and_formatting(session):
    """Verify that timeline events maintain non-decreasing timestamps and valid categories."""
    session.load_scenario("SCN-01")
    session.start()
    session.record_operator_action("SET_SETPOINT", {"loop": "LICA-002", "value": 1600.0}, True)
    
    for _ in range(20):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)

    t_prev = -1.0
    valid_categories = {"SCENARIO", "COMMAND", "ALARM", "INTERLOCK", "PROCESS"}
    valid_levels = {"INFO", "WARNING", "DANGER", "SUCCESS"}

    for ev in session.timeline:
        assert ev.time_s >= t_prev
        t_prev = ev.time_s
        assert ev.category in valid_categories
        assert ev.level in valid_levels
        d = ev.to_dict()
        assert "time_s" in d and "category" in d and "message" in d and "level" in d


# --- 9. Post-Mission Debriefing Analysis ---

def test_post_mission_debrief_no_gamification(session):
    """Verify debrief focuses purely on engineering facts and contains no arcade scores/badges."""
    session.load_scenario("SCN-01")
    session.start()

    for _ in range(160):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)

    deb = session.debrief
    assert deb is not None
    # Must NOT contain arcade/game keys
    for arcade_key in ("xp", "exp", "coins", "points", "tier", "stars", "leaderboard", "badge"):
        assert arcade_key not in deb, f"Arcade key '{arcade_key}' found in engineering debrief!"

    # Must contain rigorous engineering fields
    assert "duration_s" in deb
    assert "final_metrics" in deb
    assert "physical_cause_en" in deb
    assert "design_lesson_en" in deb
    assert "educational_review" in deb


# --- 10. Operating World Bridge Sandbox Integration & HTTP Endpoints ---

def test_bridge_sandbox_http_endpoints(bridge):
    """Verify HTTP API endpoints for sandbox state, scenarios, start, reset, abort."""
    server, active_bridge = operating_world.start_bridge_server(port=8810, sim=bridge.sim)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    # 1. GET /api/sandbox/scenarios
    req = urllib.request.Request(f"{base_url}/api/sandbox/scenarios")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        scenarios = json.loads(resp.read().decode("utf-8"))
        assert len(scenarios) == 7
        assert any(s["id"] == "SCN-01" for s in scenarios)

    # 2. POST /api/sandbox/start
    post_data = json.dumps({"scenario_id": "SCN-01"}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/sandbox/start", data=post_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True
        assert data["sandbox_state"]["state"] == "ACTIVE"

    # 3. GET /api/sandbox/state
    req = urllib.request.Request(f"{base_url}/api/sandbox/state")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        sb_state = json.loads(resp.read().decode("utf-8"))
        assert sb_state["active_scenario_id"] == "SCN-01"
        assert len(sb_state["timeline"]) >= 2

    # 4. POST /api/sandbox/abort
    req = urllib.request.Request(f"{base_url}/api/sandbox/abort", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["sandbox_state"]["state"] == "ABORTED"

    # 5. POST /api/sandbox/reset
    req = urllib.request.Request(f"{base_url}/api/sandbox/reset", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["sandbox_state"]["state"] == "READY"


# --- 11. Playwright Browser Smoke Test ---

@pytest.mark.skipif(not os.environ.get("RUN_PLAYWRIGHT_SMOKE"), reason="Set RUN_PLAYWRIGHT_SMOKE=1 to run browser smoke test")
def test_browser_sandbox_smoke(bridge):
    """Automated browser smoke test verifying sandbox drawer, timeline, and scenario execution."""
    from playwright.sync_api import sync_playwright

    server, _ = operating_world.start_bridge_server(port=8820, sim=bridge.sim)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")

        # Verify page title
        assert "CP2-V-71101" in page.title()

        # Open Sandbox Drawer
        page.click("#btn-sandbox-header")
        page.wait_for_selector("#drawer-sandbox", state="visible")

        # Verify Scenario Dropdown contains SCN-01 through SCN-07
        options = page.eval_on_selector_all("#sb-scenario-select option", "elements => elements.map(el => el.value)")
        assert "SCN-01" in options
        assert "SCN-06" in options

        # Click Start Scenario
        page.click("#sb-btn-start")
        page.wait_for_timeout(500)

        # Verify timeline item appears
        timeline_items = page.query_selector_all(".sb-timeline-item")
        assert len(timeline_items) > 0

        browser.close()


# --- 12. Completion False-Positive Audit ---

def test_completion_false_positive_audit(session):
    """Mandatory Audit: Prove scenarios cannot falsely complete without satisfying physical invariants."""
    # Test 1: SCN-01 - Running only 5 seconds must NOT complete (requires 30s stability)
    session.load_scenario("SCN-01")
    session.start()
    for _ in range(25):  # 5 seconds
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
    assert session.state == sandbox.ScenarioState.ACTIVE
    assert session.debrief is None

    # Test 2: SCN-03 - Doing nothing during 1.4x surge must lead to FAILED (LAHH trip), NEVER false COMPLETED
    session.load_scenario("SCN-03")
    session.start()
    # Trainee fails to override clamp; level surges to LAHH 2700 mm
    for _ in range(500):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
        if session.state in (sandbox.ScenarioState.FAILED, sandbox.ScenarioState.COMPLETED):
            break
    assert session.state == sandbox.ScenarioState.FAILED
    assert session.debrief["passed"] is False
    assert "LAHH" in session.debrief["outcome_message"] or "overfilled" in session.debrief["outcome_message"]

    # Test 3: SCN-04 - Unrelated commands do NOT satisfy split-range flaring condition
    session.load_scenario("SCN-04")
    session.start()
    # Dispatch unrelated level command
    session.sim.apply_command("SET_SETPOINT", {"loop": "LICA-002", "value": 1600.0})
    session.record_operator_action("SET_SETPOINT", {"loop": "LICA-002", "value": 1600.0}, True)
    # Step only 5 seconds before flaring establishes
    for _ in range(25):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
    assert session.state == sandbox.ScenarioState.ACTIVE


# --- 13. Failure False-Positive Audit ---

def test_failure_false_positive_audit(session):
    """Verify that erroneous or unrelated operator actions do not produce false success."""
    session.load_scenario("SCN-05")
    session.start()

    # Wrong operator action: instead of throttling drain, open FV-001 wide open to 100%
    session.sim.apply_command("SET_VALVE_MANUAL", {"valve": "FV-001", "manual": True, "target": 1.0})
    session.record_operator_action("SET_VALVE_MANUAL", {"valve": "FV-001", "manual": True, "target": 1.0}, True)

    # Step until level crashes into LALL (770 mm)
    for _ in range(350):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
        if session.state in (sandbox.ScenarioState.FAILED, sandbox.ScenarioState.COMPLETED):
            break

    # Must transition to FAILED due to gas blow-by trip, never false COMPLETED
    assert session.state == sandbox.ScenarioState.FAILED
    assert session.debrief["passed"] is False


# --- 14. Initial-Condition Isolation Audit ---

def test_initial_condition_isolation(session):
    """Verify sequential execution SCN-A -> SCN-B does NOT leak stale state."""
    # 1. Run SCN-02 (Cold startup starts at 50 mm)
    session.load_scenario("SCN-02")
    session.start()
    for _ in range(50):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
    assert session.sim.state.level_mm < 600.0  # intermediate level

    # 2. Reset and load SCN-01 (Normal baseline requires 1550 mm NLL)
    session.reset()
    session.load_scenario("SCN-01")
    session.start()
    assert session.sim.state.level_mm == 1550.0  # pristine baseline restored
    assert session.sim.state.alarm_status == "NORMAL"
    assert session.sim.valv_fv001.manual is False

    # 3. Run SCN-06 (Gas blowby induces Note 26 ESD trip)
    session.load_scenario("SCN-06")
    session.start()
    for _ in range(150):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)
        if session.sim.state.alarm_status == "TRIP":
            break
    assert session.sim.state.uzv002_open is False
    assert session.sim.state.uzv003_open is False

    # 4. Reset and load SCN-01 again - MUST be completely pristine
    session.reset()
    session.load_scenario("SCN-01")
    session.start()
    assert session.sim.state.uzv002_open is True
    assert session.sim.state.uzv003_open is True
    assert session.sim.state.alarm_status == "NORMAL"
    assert len(session.sim.state.tripped_causes) == 0


# --- 15. Abort and Reset Cleanliness Audit ---

def test_abort_and_reset_cleanliness(session):
    """Verify ABORT terminates scenario cleanly and RESET clears all session artifacts."""
    session.load_scenario("SCN-03")
    session.start()
    for _ in range(20):
        session.sim.step(dt=0.2)
        session.step(dt=0.2)

    # Abort
    session.abort()
    assert session.state == sandbox.ScenarioState.ABORTED
    assert session.timeline[-1].category == "SCENARIO"
    assert "aborted" in session.timeline[-1].message.lower()

    # Further steps must not change state from ABORTED
    for _ in range(10):
        session.step(dt=0.2)
    assert session.state == sandbox.ScenarioState.ABORTED

    # Reset
    session.reset()
    assert session.state == sandbox.ScenarioState.READY
    assert session.debrief is None
    assert len(session.timeline) == 2  # load + reset messages


# --- 16. All Seven Scenarios Programmatic Determinism Audit ---

def test_all_seven_scenarios_deterministic_coverage(session):
    """Deterministic coverage: Prove each of SCN-01 to SCN-07 initializes, transitions, and evaluates."""
    catalog = sandbox.SCENARIO_CATALOG
    assert len(catalog) == 7

    for scn_id, scn_def in catalog.items():
        session.reset()
        ok = session.load_scenario(scn_id)
        assert ok is True, f"Failed to load {scn_id}"
        session.start()
        assert session.state == sandbox.ScenarioState.ACTIVE

        # Step 5 simulation seconds
        for _ in range(25):
            session.sim.step(dt=0.2)
            session.step(dt=0.2)

        # Confirm session state serializes cleanly
        s_state = session.get_session_state()
        assert s_state["active_scenario_id"] == scn_id
        assert len(s_state["timeline"]) >= 2
        assert s_state["state"] in ("ACTIVE", "COMPLETED", "FAILED")

