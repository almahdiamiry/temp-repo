"""Phase 4 One Operating World Verification Suite for CP2-V-71101.

Validates the Phase 4 Acceptance Gate:
1. Python dynamic simulation (dynamic.py) is the sole process authority.
2. Canonical runtime state snapshot (get_canonical_state) reaches 2D and 3D adapters.
3. Operator commands work through the canonical command boundary (apply_command).
4. Multi-view consistency: 2D and 3D views observe identical process truth.
5. Bidirectional 2D <-> 3D canonical tag synchronization.
6. Canonical alarm and safety interlock propagation (Note 26 gas blow-by).
7. In-memory educational trends recording and propagation.
8. Safe error boundary and negative command rejection.
9. Preserved Phase 3 accepted limitations (FV-001/FT-002 merged, UZV-051/052 outside skid).
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('simulator/stage1'))
import dynamic
import operating_world

REGISTRY_PATH = os.path.abspath('config/tag_registry.json')


@pytest.fixture
def bridge():
    """Create a fresh OperatingWorldBridge instance for each test."""
    sim = dynamic.SeparatorDynamicSimulator("Case 1 x1.4")
    return operating_world.OperatingWorldBridge(sim=sim)


# --- 1. Simulation Authority Verification ---

def test_single_simulation_authority(bridge):
    """Verify that Python SeparatorDynamicSimulator remains the single process authority."""
    assert isinstance(bridge.sim, dynamic.SeparatorDynamicSimulator)
    state = bridge.get_canonical_state()
    raw_sim_state = bridge.sim.get_canonical_state()

    # Numeric process values must match exact simulation state without client derivation
    assert state["process"]["level_mm"] == raw_sim_state["process"]["level_mm"]
    assert state["process"]["pressure_barg"] == raw_sim_state["process"]["pressure_barg"]
    assert state["valves"]["fv001_pct"] == raw_sim_state["valves"]["fv001_pct"]
    assert state["valves"]["pv003b_pct"] == raw_sim_state["valves"]["pv003b_pct"]


# --- 2. Canonical State Propagation ---

def test_canonical_state_structure_and_propagation(bridge):
    """Verify state snapshot completeness across all process and control sections."""
    cs = bridge.get_canonical_state()
    for sec in ("time_s", "process", "valves", "controllers", "alarms", "kpi", "trend_summary"):
        assert sec in cs, f"Missing section '{sec}' in canonical state snapshot"

    assert cs["process"]["level_mm"] == 1550.0 # Design NLL
    assert round(cs["process"]["pressure_barg"], 2) == 4.99 # Design 5.0 barg
    assert cs["valves"]["uzv002_open"] is True
    assert cs["valves"]["uzv003_open"] is True
    assert cs["alarms"]["status"] == "NORMAL"


# --- 3. Multi-View Consistency Verification ---

def test_multi_view_state_consistency(bridge):
    """Prove that 2D and 3D view adapters observe the exact same physical state."""
    v2d = bridge.get_2d_view_state()
    v3d = bridge.get_3d_view_state()
    cs = bridge.get_canonical_state()

    # Exact equality of core process variables
    assert v2d["level_mm"] == v3d["level_mm"] == cs["process"]["level_mm"]
    assert v2d["pressure_barg"] == v3d["pressure_barg"] == cs["process"]["pressure_barg"]
    assert v2d["alarm_status"] == v3d["alarm_status"] == cs["alarms"]["status"]

    # Valve positions consistency
    assert v2d["valves"]["FV-001"]["position_pct"] == cs["valves"]["fv001_pct"]
    assert v2d["valves"]["PV-003B"]["position_pct"] == cs["valves"]["pv003b_pct"]
    assert v3d["nodes"]["tag:PV-003B"]["pct"] == cs["valves"]["pv003b_pct"]
    assert v3d["nodes"]["tag:UZV-002"]["state"] == v2d["valves"]["UZV-002"]["state"]


# --- 4. Command Propagation & Round-Trip ---

def test_command_round_trip_setpoint(bridge):
    """Verify SET_SETPOINT command modifies simulation authority and reflects across views."""
    # Initial state
    init_sp = bridge.get_canonical_state()["controllers"]["lica002"]["sp"]
    assert init_sp == 1550.0

    # Dispatch command to change level setpoint to 1650 mm
    res = bridge.dispatch_command("SET_SETPOINT", {"loop": "LICA-002", "value": 1650.0})
    assert res["success"] is True
    assert res["canonical_state"]["controllers"]["lica002"]["sp"] == 1650.0

    # Verify 2D and 3D adapters immediately reflect new setpoint
    v2d = bridge.get_2d_view_state()
    assert v2d["controllers"]["LICA-002"]["sp"] == 1650.0

    # Pressure setpoint round-trip
    res_p = bridge.dispatch_command("SET_SETPOINT", {"loop": "PIC-003", "value": 6.20})
    assert res_p["success"] is True
    assert bridge.get_canonical_state()["controllers"]["pic003"]["sp"] == 6.20
    assert bridge.get_2d_view_state()["controllers"]["PIC-003"]["sp"] == 6.20


def test_command_round_trip_valve_manual(bridge):
    """Verify SET_VALVE_MANUAL command switches valve to manual and strokes position."""
    # Stroke FV-001 to 75%
    res = bridge.dispatch_command("SET_VALVE_MANUAL", {"valve": "FV-001", "manual": True, "target": 0.75})
    assert res["success"] is True

    # Advance simulation by 8 seconds (40 steps @ dt=0.2s) so 14s valve actuator slews to target
    bridge.step(dt=0.2, n_steps=40)

    cs = bridge.get_canonical_state()
    assert cs["valves"]["fv001_pct"] >= 75.0
    v2d = bridge.get_2d_view_state()
    assert v2d["valves"]["FV-001"]["mode"] == "MANUAL"


def test_command_clamp_override(bridge):
    """Verify SET_CLAMP_OVERRIDE toggles 100 kBOPD flow clamp."""
    assert bridge.get_canonical_state()["controllers"]["clamp_override"] is False
    res = bridge.dispatch_command("SET_CLAMP_OVERRIDE", {"active": True})
    assert res["success"] is True
    assert bridge.get_canonical_state()["controllers"]["clamp_override"] is True
    assert bridge.get_2d_view_state()["controllers"]["FIC-001"]["clamp_override"] is True


# --- 5. 2D <-> 3D Synchronization & Identity Linking ---

def test_canonical_tag_tracing(bridge):
    """Trace real canonical tags across registry, 2D view, and 3D scene."""
    # PT-003 (Pressure Transmitter)
    pt003_trace = bridge.trace_tag_identity("PT-003")
    assert pt003_trace["found"] is True
    assert pt003_trace["gltf_ref"] == "tag:PT-003"
    assert pt003_trace["physical_3d_presence"] == "verified"
    assert pt003_trace["addressable_3d_node"] == "verified"
    assert "pressure_barg" in pt003_trace["live_process_value"]

    # LT-002 (Level Transmitter)
    lt002_trace = bridge.trace_tag_identity("LT-002")
    assert lt002_trace["found"] is True
    assert lt002_trace["gltf_ref"] == "tag:LT-002"
    assert "level_mm" in lt002_trace["live_process_value"]

    # UZV-002 (Liquid SDV)
    uzv002_trace = bridge.trace_tag_identity("UZV-002")
    assert uzv002_trace["found"] is True
    assert uzv002_trace["gltf_ref"] == "tag:UZV-002"
    assert uzv002_trace["live_process_value"]["open"] is True


# --- 6. Alarms & Safety Interlock Propagation ---

def test_alarm_and_note26_interlock_propagation(bridge):
    """Verify level trip propagates to views and trips gas SDV UZV-003 per Note 26."""
    # Drain vessel to trip level (LALL <= 770 mm)
    bridge.sim.state.level_m = 0.700
    bridge.sim.state.level_mm = 700.0
    bridge.step(dt=0.2)

    cs = bridge.get_canonical_state()
    v2d = bridge.get_2d_view_state()
    v3d = bridge.get_3d_view_state()

    assert cs["alarms"]["status"] == "TRIP"
    assert v2d["alarm_status"] == "TRIP"
    assert v3d["alarm_status"] == "TRIP"

    # Note 26 interlock: LALL trips liquid SDV UZV-002 AND gas SDV UZV-003
    assert cs["valves"]["uzv002_open"] is False
    assert cs["valves"]["uzv003_open"] is False
    assert cs["valves"]["zsc002_closed"] is True

    # 2D and 3D views reflect closed valves
    assert v2d["valves"]["UZV-002"]["state"] == "CLOSED"
    assert v2d["valves"]["UZV-003"]["state"] == "CLOSED"
    assert v3d["nodes"]["tag:UZV-002"]["state"] == "CLOSED"
    assert v3d["nodes"]["tag:UZV-003"]["state"] == "CLOSED"

    # Reset ESD command returns valves to open when level restored
    bridge.sim.state.level_m = 1.550
    bridge.sim.state.level_mm = 1550.0
    res_reset = bridge.dispatch_command("RESET_ESD", {})
    assert res_reset["success"] is True
    bridge.step(dt=0.2)
    assert bridge.get_canonical_state()["alarms"]["status"] == "NORMAL"


# --- 7. In-Memory Rolling Trends ---

def test_in_memory_trends_recording(bridge):
    """Verify simulation steps populate the rolling in-memory trend buffer."""
    # Initial point was recorded on init
    assert len(bridge.get_trends()) >= 1

    # Advance 10 steps
    for _ in range(10):
        bridge.step(dt=0.2)

    trends = bridge.get_trends()
    assert len(trends) >= 11

    # Check data fields in recent point
    latest = trends[-1]
    for key in ("t", "level_mm", "level_sp_mm", "pressure_barg", "pressure_sp_barg", "out_liq_m3_h", "alarm_status"):
        assert key in latest, f"Missing trend key '{key}'"

    assert latest["level_mm"] > 0.0
    assert latest["pressure_barg"] > 0.0


# --- 8. Error Boundary & Negative Command Rejection ---

def test_negative_command_handling(bridge):
    """Verify invalid commands and out-of-bounds parameters are safely rejected without state corruption."""
    state_before = bridge.get_canonical_state()

    # 1. Unknown command name
    res1 = bridge.dispatch_command("MALICIOUS_CMD", {"foo": "bar"})
    assert res1["success"] is False
    assert "Rejected unknown command" in res1["error"]

    # 2. Unknown loop name
    res2 = bridge.dispatch_command("SET_SETPOINT", {"loop": "NON_EXISTENT_LOOP", "value": 100.0})
    assert res2["success"] is False
    assert "Rejected invalid controller loop" in res2["error"]

    # 3. Out-of-bounds level setpoint (e.g. 50,000 mm on a 4200 mm diameter vessel)
    res3 = bridge.dispatch_command("SET_SETPOINT", {"loop": "LICA-002", "value": 50000.0})
    assert res3["success"] is False
    assert "out of physical vessel bounds" in res3["error"]

    # 4. Out-of-bounds valve target (> 1.0)
    res4 = bridge.dispatch_command("SET_VALVE_MANUAL", {"valve": "FV-001", "target": 2.5})
    assert res4["success"] is False
    assert "out of fractional span" in res4["error"]

    # Verify state was not corrupted
    state_after = bridge.get_canonical_state()
    assert state_before["process"] == state_after["process"]
    assert state_before["controllers"] == state_after["controllers"]


# --- 9. Preserved Phase 3 Accepted Limitations ---

def test_preserved_accepted_limitations(bridge):
    """Verify honest representation of merged and outside-skid items."""
    v3d = bridge.get_3d_view_state()
    lims = v3d["merged_limitations"]

    # FV-001 is now an isolated verified 3D node
    assert "tag:FV-001" in v3d["nodes"]
    assert "pct" in v3d["nodes"]["tag:FV-001"]

    # FT-002: verified on P&ID but not modeled as discrete 3D solid in CAD
    assert lims["FT-002"]["addressable_node"] is False
    assert "CAD/IFC" in lims["FT-002"]["reason"]

    # UZV-051 and UZV-052: upstream outside separator skid module bounds
    assert lims["UZV-051"]["addressable_node"] is False
    assert "upstream outside separator skid bounds" in lims["UZV-051"]["reason"]
    assert lims["UZV-052"]["addressable_node"] is False
    assert "upstream outside separator skid bounds" in lims["UZV-052"]["reason"]

    # PSV-001A and PSV-001B: located downstream on flare header skid per authentic plant model
    assert lims["PSV-001A"]["addressable_node"] is False
    assert "flare header skid" in lims["PSV-001A"]["reason"]
    assert lims["PSV-001B"]["addressable_node"] is False
    assert "flare header skid" in lims["PSV-001B"]["reason"]



# --- 10. Real Browser Smoke Test (Audit Gate A) ---

def test_real_browser_smoke_round_trip():
    """Verify real browser (MS Edge via Playwright) end-to-end command round trip."""
    import time
    from playwright.sync_api import sync_playwright

    server, bridge = operating_world.start_bridge_server(port=8765)
    port = server.server_port

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
        time.sleep(0.8)

        # 1. Connected to Python authority
        assert page.evaluate("window.OTS_CONNECTED_TO_PYTHON") is True
        assert page.evaluate("window.CANONICAL_SIMULATION_AUTHORITY") == "python:dynamic.py"

        # 2. Initial state
        assert float(page.inner_text("#fp-lica-sp")) == 1550.0

        # 3. Open faceplate and set setpoint to 1650 mm
        page.click("#btn-faceplates-dock")
        time.sleep(0.2)
        page.fill("#fp-lica-sp-input", "1650")
        page.click("#fp-btn-apply-lica")
        time.sleep(0.5)

        # 4. Verify canonical state changed in Python and in both views
        assert bridge.get_canonical_state()["controllers"]["lica002"]["sp"] == 1650.0
        assert float(page.inner_text("#fp-lica-sp")) == 1650.0
        assert page.evaluate("sim.lica002_sp") == 1650.0

        # 5. Reset
        page.evaluate("resetCanonicalSim()")
        time.sleep(0.5)
        assert bridge.get_canonical_state()["controllers"]["lica002"]["sp"] == 1550.0
        assert float(page.inner_text("#fp-lica-sp")) == 1550.0

        browser.close()
