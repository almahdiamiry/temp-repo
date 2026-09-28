"""Phase 3 Canonical Model Verification Suite for CP2-V-71101.

Validates the two core Phase 3 objectives:
A. Faithful canonical geometry asset (final_twin.glb) and node addressability boundaries.
B. Single explicit canonical simulation/state boundary and command contract (dynamic.py).
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('simulator/stage1'))
import dynamic
import twin3d

REGISTRY_PATH = os.path.abspath('config/tag_registry.json')
GLB_PATH = os.path.abspath('simulator/stage1/assets/final_twin.glb')


@pytest.fixture(scope="module")
def registry():
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sim():
    return dynamic.SeparatorDynamicSimulator()


# --- A. Canonical Geometry Verification ---

def test_canonical_geometry_nodes_and_extents():
    """Verify final_twin.glb preserves the required canonical nodes and geometric extents."""
    import pygltflib
    assert os.path.exists(GLB_PATH), f"Canonical GLB missing at {GLB_PATH}"
    glb = pygltflib.GLTF2().load(GLB_PATH)

    node_names = {node.name for node in glb.nodes if node.name}

    expected_canonical_nodes = [
        "vessel_shell",
        "vessel_nozzles_saddles",
        "structure",
        # Service piping is addressed per battery-limit group rather than
        # through one flat `piping` node, so the scene graph says which
        # process boundary a run of pipe belongs to.
        "service_piping:02_INLET",
        "service_piping:03_LIQUID_OUTLET",
        "service_piping:04_GAS_OUTLET",
        "service_piping:06_RELIEF",
        "service_piping:99_REVIEW_UNCLASSIFIED",
        "tag:PV-003B",
        "tag:PV-003A",
        "tag:LT-002",
        "tag:PT-003",
        "tag:TT-051",
        "tag:UZV-002",
        "tag:UZV-003",
        "tag:UZV-054",
        "tag:FV-001",
    ]
    for node in expected_canonical_nodes:
        assert node in node_names, f"Expected canonical node '{node}' missing from final_twin.glb!"

    # The flat `piping` node is superseded by the per-group leaves above. It must
    # not linger, or the same geometry would be reachable through two names.
    assert "piping" not in node_names, (
        "A flat `piping` node reappeared alongside the per-group "
        "service_piping:* leaves; that would duplicate the addressability of the "
        "same geometry."
    )

    # Verify node extras are preserved
    named_nodes = [n for n in glb.nodes if n.name and n.name.startswith("tag:")]
    assert len(named_nodes) >= 10, "Telemetry tag nodes must preserve -kn -ke extras"
    for n in named_nodes:
        assert "tag" in n.extras or "category" in n.extras, f"Node {n.name} missing extras metadata"


def test_fv001_node_uniqueness_and_extras():
    """Verify tag:FV-001 exists, is unique, and preserves authentic control valve extras."""
    import pygltflib
    assert os.path.exists(GLB_PATH)
    glb = pygltflib.GLTF2().load(GLB_PATH)
    fv_nodes = [n for n in glb.nodes if n.name == "tag:FV-001"]
    assert len(fv_nodes) == 1, f"tag:FV-001 must exist and be strictly unique, found {len(fv_nodes)}"
    fv = fv_nodes[0]
    assert fv.extras.get("tag") == "FV-001"
    assert fv.extras.get("kind") == "control_valve"
    assert fv.extras.get("triCount", 0) > 1000
    assert "center" in fv.extras


def test_merged_and_out_of_bounds_geometry_honest_representation(registry):
    """Verify FT-002 remains unmodeled in CAD, FV-001 is verified addressable, and UZV-051 remains out-of-bounds."""
    # FT-002: in-line gas meter without discrete 3D solid in source CAD/IFC
    ft002 = registry["instruments"]["FT-002"]
    assert ft002["physical_3d_presence"] == "verified"
    assert ft002["addressable_3d_node"] == "not_yet_available"
    assert ft002["gltf_ref"] is None
    assert ft002["status"] == "PARTIALLY VERIFIED"

    # FV-001: isolated as standalone addressable GLTF node
    fv001 = registry["valves"]["FV-001"]
    assert fv001["physical_3d_presence"] == "verified"
    assert fv001["addressable_3d_node"] == "verified"
    assert fv001["gltf_ref"] == "tag:FV-001"
    assert fv001["status"] == "VERIFIED"

    # UZV-051: upstream manifold valve, addressable 3D node is not yet available in module GLB
    uzv051 = registry["valves"]["UZV-051"]
    assert uzv051["addressable_3d_node"] == "not_yet_available"
    assert uzv051["gltf_ref"] is None
    assert uzv051["status"] == "PARTIALLY VERIFIED"


# --- B. Canonical Simulation & State Boundary Verification ---

def test_canonical_simulation_state_boundary(sim):
    """Verify get_canonical_state() contract produces structured, self-contained snapshot."""
    state = sim.get_canonical_state()
    assert isinstance(state, dict)

    required_sections = ["time_s", "process", "valves", "controllers", "alarms", "kpi"]
    for sec in required_sections:
        assert sec in state, f"Section '{sec}' missing from canonical state!"

    # Process variables
    proc = state["process"]
    assert 0.0 < proc["level_m"] < dynamic.D
    assert 0.0 < proc["level_mm"] < dynamic.D * 1000.0
    assert proc["pressure_barg"] > 0.0
    assert proc["temp_c"] > 0.0
    assert proc["liquid_holdup_m3"] > 0.0
    assert proc["gas_holdup_m3"] > 0.0

    # Valve positions & states
    valves = state["valves"]
    for v_key in ("fv001_pct", "pv003b_pct", "pv003a_pct"):
        assert 0.0 <= valves[v_key] <= 100.0, f"Valve {v_key} out of range [0, 100]"
    for sdv in ("uzv051_open", "uzv052_open", "uzv002_open", "uzv003_open"):
        assert isinstance(valves[sdv], bool), f"SDV {sdv} must be boolean"

    # Controllers
    ctrls = state["controllers"]
    assert "lica002" in ctrls and "sp" in ctrls["lica002"] and "op" in ctrls["lica002"]
    assert "fic001" in ctrls and "sp" in ctrls["fic001"] and "op" in ctrls["fic001"]
    assert "pic003" in ctrls and "sp" in ctrls["pic003"] and "op" in ctrls["pic003"]

    # Alarms & KPI
    assert state["alarms"]["status"] in ("NORMAL", "ALARM", "TRIP")
    assert state["kpi"]["water_in_oil_ppm"] >= 0.0
    assert state["kpi"]["carryover_gal_mmscf"] >= 0.0


def test_canonical_simulation_command_contract():
    """Verify simulation command dispatcher executes valid commands and rejects invalid ones."""
    sim = dynamic.SeparatorDynamicSimulator()

    # 1. SET_SETPOINT
    assert sim.apply_command("SET_SETPOINT", {"loop": "LICA-002", "value": 1600.0})
    assert sim.state.lica002_sp_mm == 1600.0
    assert sim.pid_level.setpoint == 1600.0

    assert sim.apply_command("SET_SETPOINT", {"loop": "PIC-003", "value": 5.2})
    assert sim.state.pic003_sp_barg == 5.2
    assert sim.pid_press.setpoint == 5.2

    # 2. SET_CLAMP_OVERRIDE
    assert sim.apply_command("SET_CLAMP_OVERRIDE", {"active": True})
    assert sim.state.clamp_override is True
    assert sim.pid_level.output_max == 1400.0

    assert sim.apply_command("SET_CLAMP_OVERRIDE", {"active": False})
    assert sim.state.clamp_override is False
    assert sim.pid_level.output_max == dynamic.CAP_100KBOPD_M3_H

    # 3. SET_VALVE_MANUAL
    assert sim.apply_command("SET_VALVE_MANUAL", {"valve": "FV-001", "manual": True, "target": 0.75})
    assert sim.valv_fv001.manual is True
    assert sim.valv_fv001.target == 0.75

    # 4. SET_FEED_DISTURBANCE
    assert sim.apply_command("SET_FEED_DISTURBANCE", {"liquid_m3_h": 900.0, "gas_kg_h": 80000.0})
    assert sim.state.feed_liquid_m3_h == 900.0
    assert sim.state.feed_gas_kg_h == 80000.0

    # 5. STEP
    t_prev = sim.state.time_s
    assert sim.apply_command("STEP", {"dt": 0.2})
    assert sim.state.time_s == pytest.approx(t_prev + 0.2)

    # 6. RESET_ESD
    assert sim.apply_command("RESET_ESD")

    # 7. Scenario orchestration is not a simulation primitive (Phase 5 boundary)
    assert not sim.apply_command("APPLY_SCENARIO", {"scenario_tag": "SCN-01"})
    # Pre-existing apply_scenario function remains available directly
    dynamic.apply_scenario(sim, "SCN-01")
    assert sim.valv_fv001.manual is False


def test_negative_twin_command_and_state_separation():
    """Negative Twin: Verify commands do not conflate identities or accept spurious loops."""
    sim = dynamic.SeparatorDynamicSimulator()

    # Unknown loop or valve returns False
    assert not sim.apply_command("SET_SETPOINT", {"loop": "UNKNOWN-999", "value": 100.0})
    assert not sim.apply_command("SET_VALVE_MANUAL", {"valve": "UNKNOWN-VALVE", "manual": True})
    assert not sim.apply_command("NON_EXISTENT_COMMAND", {})

    # UZV-054 cannot be commanded as an independent valve state variable
    assert not sim.apply_command("SET_VALVE_MANUAL", {"valve": "UZV-054", "manual": True})

    # UZV-054 state is not conflated with UZV-051
    canonical_valves = sim.get_canonical_state()["valves"]
    assert "uzv051_open" in canonical_valves
    assert "uzv054_open" not in canonical_valves, "UZV-054 has no independent ODE simulation state!"


def test_browser_template_authority_and_initial_state():
    """Verify twin3d HTML template establishes Python authority and injects valid initial state."""
    rendered_html = twin3d.render_ots_app_html()

    assert 'window.CANONICAL_SIMULATION_AUTHORITY = "python:dynamic.py";' in rendered_html
    assert 'window.CLIENT_PHYSICS_MODE = "STANDALONE_FALLBACK_VIEWER";' in rendered_html
    assert 'window.INITIAL_CANONICAL_STATE = {' in rendered_html
    assert '__INITIAL_CANONICAL_STATE__' not in rendered_html
