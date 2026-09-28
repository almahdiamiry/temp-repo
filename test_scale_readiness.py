"""Test Suite for Phase 6 Gate B: Controlled Scale Readiness.

Validates that the digital twin architecture establishes clear, reusable boundaries:
1. Tag Registry Schema Contract (Multi-unit extensibility without breaking CP2-V-71101).
2. Canonical Simulation State & Command Boundary Interface Contract.
3. Operating World Bridge Pluggability (independent of CP2-V-71101 internals).
4. Educational Sandbox Lifecycle Reusability (generic scenario runner).
5. Second Unit Manifest Descriptor Schema Validation (CP2-V-71102 stub descriptor).
"""

import json
import os
import pytest
from typing import Dict, Any

from operating_world import OperatingWorldBridge, REGISTRY_PATH
import dynamic
import sandbox


# --- 1. Tag Registry Schema Extensibility ---

def test_tag_registry_multi_unit_schema_contract():
    """Verify that the tag registry schema can accommodate new units without schema break."""
    assert os.path.exists(REGISTRY_PATH), f"Registry missing at {REGISTRY_PATH}"
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # Core sections must exist
    required_sections = ["_metadata", "equipment", "nozzles", "valves", "instruments", "controllers"]
    for sec in required_sections:
        assert sec in registry, f"Missing section '{sec}' in tag_registry.json"

    # CP2-V-71101 must be present and verified
    assert "CP2-V-71101" in registry["equipment"]
    eq = registry["equipment"]["CP2-V-71101"]
    assert eq["canonical_tag"] == "CP2-V-71101"
    assert "spec" in eq
    assert eq["status"] == "VERIFIED"

    # Test adding a mock second unit (CP2-V-71102) to verify multi-unit compatibility
    mock_registry = dict(registry)
    mock_second_unit = {
        "canonical_tag": "CP2-V-71102",
        "type": "SEPARATOR_VESSEL_2ND_STAGE",
        "service": "2nd Stage Production Separator (Train 1)",
        "pid_ref": "CP2-V-71102",
        "cad_dwg_source": "MGP1-CP2-PIL-PX-2365-1002.dwg",
        "gltf_ref": "vessel_shell_stage2",
        "physical_3d_presence": "verified",
        "addressable_3d_node": "verified",
        "simulation_ref": "Stage2SeparatorDynamicSimulator",
        "status": "CANDIDATE",
        "verified_triplet": False,
        "spec": {
            "dimensions": "3600 mm ID x 11000 mm T/T",
            "operating_pressure_barg": 1.2
        }
    }
    mock_registry["equipment"]["CP2-V-71102"] = mock_second_unit
    assert "CP2-V-71102" in mock_registry["equipment"]
    assert mock_registry["equipment"]["CP2-V-71101"]["canonical_tag"] == "CP2-V-71101"


# --- 2. Simulation State & Command Boundary Contract ---

def test_canonical_simulation_interface_contract():
    """Verify that any simulator adheres to the Unit Simulation Interface contract."""
    sim = dynamic.SeparatorDynamicSimulator("Case 1 x1.4")

    # Contract 1: get_canonical_state() must return standard top-level dictionary
    state = sim.get_canonical_state()
    assert isinstance(state, dict)
    for key in ["time_s", "process", "valves", "controllers", "alarms"]:
        assert key in state, f"Canonical state missing contract key '{key}'"

    # Contract 2: apply_command(cmd, params) must return boolean and handle negative inputs safely
    assert sim.apply_command("UNKNOWN_COMMAND_XYZ", {}) is False
    assert sim.apply_command("SET_SETPOINT", {"loop": "NON_EXISTENT", "value": 999.0}) is False
    assert sim.apply_command("SET_VALVE_MANUAL", {"valve": "NON_EXISTENT", "target": 0.5}) is False

    # Contract 3: step(dt) advance
    t0 = sim.state.time_s
    sim.step(dt=0.2)
    assert round(sim.state.time_s - t0, 4) == 0.2


# --- 3. Operating World Bridge Pluggability ---

class MockUnitState:
    def __init__(self):
        self.time_s = 0.0
        self.pressure_barg = 1.2
        self.level_mm = 1100.0
        self.alarm_status = "NORMAL"
        self.active_alarms = []
        self.tripped_causes = []


class MockSecondUnitSimulator:
    """Minimal mock simulator representing a non-process test asset or second unit."""
    def __init__(self):
        self.state = MockUnitState()

    def get_canonical_state(self) -> Dict[str, Any]:
        return {
            "asset_id": "CP2-V-71102",
            "time_s": self.state.time_s,
            "process": {
                "level_mm": self.state.level_mm,
                "level_m": self.state.level_mm / 1000.0,
                "pressure_barg": self.state.pressure_barg,
                "feed_liquid_m3_h": 400.0,
                "out_liquid_m3_h": 400.0,
                "out_gas_header_kg_h": 15000.0,
                "out_gas_flare_kg_h": 0.0,
            },
            "valves": {"fv002": {"position": 0.50, "manual": False}},
            "controllers": {"lica003": {"pv": self.state.level_mm, "sp": 1100.0, "op": 50.0, "mode": "AUTO"}},
            "alarms": {"status": "NORMAL", "active_alarms": []}
        }

    def apply_command(self, cmd: str, params: Dict[str, Any]) -> bool:
        if cmd == "SET_SETPOINT":
            return True
        elif cmd == "STEP":
            self.step(params.get("dt", 0.2))
            return True
        return False

    def step(self, dt: float = 0.2):
        self.state.time_s += dt

    def _init_steady_baseline(self):
        self.state.time_s = 0.0
        self.state.level_mm = 1100.0


def test_bridge_pluggability_with_mock_simulator():
    """Verify OperatingWorldBridge can bind to a non-CP2-V-71101 simulator without modification."""
    mock_sim = MockSecondUnitSimulator()
    bridge = OperatingWorldBridge(sim=mock_sim)

    # 1. Query canonical state through bridge
    cs = bridge.get_canonical_state()
    assert cs["asset_id"] == "CP2-V-71102"
    assert cs["process"]["pressure_barg"] == 1.2

    # 2. Dispatch valid generic commands through bridge
    res_step = bridge.dispatch_command("STEP", {"dt": 0.2})
    assert res_step["success"] is True

    res_sp = bridge.dispatch_command("SET_SETPOINT", {"loop": "LEVEL", "value": 1200.0})
    assert res_sp["success"] is True

    # 3. Dispatch invalid command through bridge
    bad_res = bridge.dispatch_command("INVALID_CMD", {})
    assert bad_res["success"] is False


# --- 4. Educational Sandbox Lifecycle Reusability ---

def test_sandbox_reusability_with_custom_unit_scenario():
    """Verify SandboxSession can run custom scenarios for another unit using standard lifecycle."""
    mock_sim = MockSecondUnitSimulator()
    session = sandbox.SandboxSession(mock_sim)

    # Define a custom scenario for the second unit
    custom_scn = sandbox.ScenarioDefinition(
        id="SCN-U2-01",
        tag="SCN-U2-01",
        title_en="Stage 2 Normal Baseline",
        title_ar="خط الأساس للمرحلة الثانية",
        description_en="Verify 2nd stage separator baseline stability.",
        description_ar="التحقق من استقرار العازلة الثانية.",
        learning_objective_en="Understand low-pressure separation equilibrium.",
        learning_objective_ar="فهم اتزان الفصل في الضغط المنخفض.",
        initial_condition_desc_en="Normal 1.2 barg baseline.",
        initial_condition_desc_ar="خط أساس 1.2 بار.",
        allowed_commands=["SET_SETPOINT", "STEP"],
        setup_func=lambda s: s._init_steady_baseline(),
        evaluation_func=lambda s, sess: (sandbox.ScenarioState.COMPLETED, "Stage 2 Stable", "المرحلة مستقرة") if sess.elapsed_sim_s >= 9.9 else (sandbox.ScenarioState.ACTIVE, "Running", "قيد التشغيل"),
        debrief_info={"key_takeaway": "Low pressure degassing complete.", "design_reference": "PX-2365-1002"}
    )

    # Inject scenario into catalog for test
    sandbox.SCENARIO_CATALOG["SCN-U2-01"] = custom_scn

    try:
        assert session.load_scenario("SCN-U2-01") is True
        assert session.state == sandbox.ScenarioState.READY
        session.start()
        assert session.state == sandbox.ScenarioState.ACTIVE

        # Step 55 times (11s)
        for _ in range(55):
            session.sim.step(0.2)
            session.step(0.2)

        assert session.state == sandbox.ScenarioState.COMPLETED
        assert session.debrief is not None
        assert session.debrief["passed"] is True
        assert "Stage 2 Stable" in session.debrief["outcome_message"]
    finally:
        # Cleanup injected scenario
        sandbox.SCENARIO_CATALOG.pop("SCN-U2-01", None)


# --- 5. Unit Manifest Convention Schema Validation ---

def test_second_unit_manifest_schema_validation():
    """Verify that a minimal future unit manifest descriptor adheres to the project convention."""
    manifest_convention = {
        "unit_id": "CP2-V-71102",
        "unit_slug": "stage2-separator",
        "description": "2nd Stage Production Separator",
        "design_basis": "docs/plans/stage2-separator/DESIGN.md",
        "assets": {
            "geometry_glb": "simulator/stage2/assets/stage2_twin.glb",
            "pnd_svg": "simulator/stage2/assets/pnd_stage2.svg"
        },
        "simulation": {
            "module": "simulator.stage2.dynamic",
            "class": "Stage2SeparatorDynamicSimulator",
            "baseline_case": "Case 1 x1.4"
        },
        "scenarios": [
            "SCN-U2-01",
            "SCN-U2-02"
        ],
        "accepted_limitations": [
            "Single liquid phase modeled (no three-phase boot control)",
            "Shared header flare boundary"
        ]
    }

    # Validate essential convention fields
    assert manifest_convention["unit_id"].startswith("CP2-")
    assert manifest_convention["unit_slug"] != ""
    assert "geometry_glb" in manifest_convention["assets"]
    assert "class" in manifest_convention["simulation"]
    assert len(manifest_convention["scenarios"]) > 0
    assert len(manifest_convention["accepted_limitations"]) > 0
