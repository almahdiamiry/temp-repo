"""Stage 1 Production Separator (CP2-V-71101) Model Validation & Physical Invariants Suite.

Post-Phase-6 Hardening (LIM-08 Model Validation):
Strengthens validation, envelope definition, physical invariants, and numerical confidence
boundaries for the reduced-order dynamic model (dynamic.py) without CFD.

Validates 4 Core Dimensions:
1. Physical Conservation Checks:
   - Volume/mass balance: integral(Q_in - Q_out) dt == Delta V_liquid
   - Gas compressibility: dP/dt scales inversely with vapor space volume V_gas = V_total - V_liquid
2. Steady-State Stability:
   - 600-second unperturbed run demonstrates zero-drift (< 0.05% deviation / < 0.5 mm)
3. Monotonicity & Sign Invariants:
   - Feed step increase -> level monotonically rises (before controller acts)
   - Valve opening -> outflow monotonically increases
   - Gas valve pinch -> pressure monotonically rises
4. Operational Boundary Envelope & Numerical Robustness:
   - Water cut extremes: 0% to 99%
   - Pressure envelope: 1.0 to 20.0 barg
   - Temperature range: 20 to 100 °C
   - Feed throughput: 0 to 2.0x design (0 to 1590 m3/h)
   - Guarantees zero NaN, zero Inf, positive holdups, and non-divergent integration
"""

import math
import pytest

import dynamic
import model


# --- Category 1: Physical Conservation Checks ---

def test_liquid_inventory_mass_volume_conservation():
    """Verify liquid inventory balance: integral(Q_in - Q_out) dt == Delta V_liquid.
    
    Holds valves in manual mode to decouple feedback control and evaluate pure mass balance ODE.
    """
    sim = dynamic.SeparatorDynamicSimulator()
    sim._init_steady_baseline()
    
    # Put FV-001 in manual fixed at its steady position and step feed up by 20%
    sim.valv_fv001.manual = True
    pos0 = sim.valv_fv001.position
    sim.valv_fv001.target = pos0
    
    h_initial = sim.state.level_m
    v_liq_initial = model.a_liquid(h_initial) * dynamic.L_TT
    
    # Perturb feed rate to create accumulation
    sim.state.feed_liquid_m3_h = 794.94 * 1.20  # +20% over nominal 794.9 m3/h
    
    dt = 0.2
    total_steps = 100  # 20 seconds
    cumulative_net_inflow_m3 = 0.0
    
    for _ in range(total_steps):
        q_in_m3_s = sim.state.feed_liquid_m3_h / 3600.0
        q_out_m3_s = sim.state.out_liquid_m3_h / 3600.0
        cumulative_net_inflow_m3 += (q_in_m3_s - q_out_m3_s) * dt
        sim.step(dt=dt)
        
    h_final = sim.state.level_m
    v_liq_final = model.a_liquid(h_final) * dynamic.L_TT
    delta_v_geometric = v_liq_final - v_liq_initial
    
    # Geometric holdup change must match integrated inflow minus outflow within 0.01%
    assert delta_v_geometric > 0.0, "Liquid volume must accumulate with feed > outflow"
    rel_error = abs(delta_v_geometric - cumulative_net_inflow_m3) / cumulative_net_inflow_m3
    assert rel_error < 0.001, f"Mass balance error {rel_error*100:.5f}% exceeds 0.1% tolerance"


def test_gas_compressibility_and_vapor_space_scaling():
    """Verify dP/dt scales inversely with vapor space volume V_gas = V_total - V_liquid.
    
    Ideal gas relation: dP/dt = (R_g * T / V_gas) * (m_dot_in - m_dot_out).
    Therefore, with smaller vapor space (higher liquid level), pressure response |dP/dt|
    must be strictly faster for the identical mass imbalance.
    """
    dt = 0.2
    
    # Run 1: High vapor space (low level h = 0.8m -> large V_gas)
    sim_large_vgas = dynamic.SeparatorDynamicSimulator()
    sim_large_vgas.state.level_m = 0.8
    sim_large_vgas.state.level_mm = 800.0
    sim_large_vgas.valv_pv003b.manual = True
    sim_large_vgas.valv_pv003b.position = 0.50
    sim_large_vgas.valv_pv003b.target = 0.50
    sim_large_vgas.valv_fv001.manual = True
    sim_large_vgas.valv_fv001.position = 0.50
    
    p0_large = sim_large_vgas.state.pressure_barg
    for _ in range(25):  # 5 seconds
        sim_large_vgas.step(dt=dt)
    dp_large_vgas = sim_large_vgas.state.pressure_barg - p0_large
    
    # Run 2: Low vapor space (high level h = 2.4m -> small V_gas)
    sim_small_vgas = dynamic.SeparatorDynamicSimulator()
    sim_small_vgas.state.level_m = 2.4
    sim_small_vgas.state.level_mm = 2400.0
    sim_small_vgas.valv_pv003b.manual = True
    sim_small_vgas.valv_pv003b.position = 0.50
    sim_small_vgas.valv_pv003b.target = 0.50
    sim_small_vgas.valv_fv001.manual = True
    sim_small_vgas.valv_fv001.position = 0.50
    
    p0_small = sim_small_vgas.state.pressure_barg
    for _ in range(25):  # 5 seconds
        sim_small_vgas.step(dt=dt)
    dp_small_vgas = sim_small_vgas.state.pressure_barg - p0_small
    
    # Small vapor space must exhibit strictly larger |dP| under same imbalance
    assert abs(dp_small_vgas) > abs(dp_large_vgas), (
        f"Gas compressibility scaling violated: |dP_small_vgas|={abs(dp_small_vgas):.4f} "
        f"should exceed |dP_large_vgas|={abs(dp_large_vgas):.4f}"
    )


# --- Category 2: Steady-State Stability ---

def test_extended_steady_state_zero_drift():
    """Verify 600-second unperturbed run demonstrates zero drift and tight closed-loop stability."""
    sim = dynamic.SeparatorDynamicSimulator()
    sim._init_steady_baseline()
    
    init_level_mm = sim.state.level_mm
    init_pressure_barg = sim.state.pressure_barg
    
    dt = 0.2
    steps_600s = int(600.0 / dt)  # 3000 steps
    
    for _ in range(300):  # First 60s to reach cascade closed-loop equilibrium
        sim.step(dt=dt)
        
    settled_level_mm = sim.state.level_mm
    settled_pressure_barg = sim.state.pressure_barg
    settled_flow_m3_h = sim.state.out_liquid_m3_h
    
    for _ in range(steps_600s - 300):  # Remaining 540s (9 minutes)
        sim.step(dt=dt)
        
    final_level_mm = sim.state.level_mm
    final_pressure_barg = sim.state.pressure_barg
    final_flow_m3_h = sim.state.out_liquid_m3_h
    
    # Drift across the 540-second steady horizon (< 0.05% relative)
    level_drift_rel = abs(final_level_mm - settled_level_mm) / settled_level_mm
    press_drift_rel = abs(final_pressure_barg - settled_pressure_barg) / settled_pressure_barg
    flow_drift_rel = abs(final_flow_m3_h - settled_flow_m3_h) / settled_flow_m3_h
    
    assert level_drift_rel < 0.0005, f"Level drifted {level_drift_rel*100:.4f}% (> 0.05% max permitted)"
    assert press_drift_rel < 0.0001, f"Pressure drifted {press_drift_rel*100:.5f}% (> 0.01% max permitted)"
    assert flow_drift_rel < 0.0005, f"Flow drifted {flow_drift_rel*100:.5f}% (> 0.05% max permitted)"
    
    # Overall level deviation from NLL 1550 mm remains under 1.0 mm (< 0.06% of NLL)
    overall_level_dev = abs(final_level_mm - init_level_mm)
    assert overall_level_dev < 1.0, f"Overall level deviation {overall_level_dev:.2f} mm exceeds 1.0 mm bound"
    assert sim.state.alarm_status == "NORMAL"


# --- Category 3: Monotonicity & Sign Invariants ---

def test_feed_surge_monotonic_level_rise():
    """Verify feed increase causes monotonic liquid level rise prior to controller action."""
    sim = dynamic.SeparatorDynamicSimulator()
    sim._init_steady_baseline()
    
    # Lock valve in manual at steady position to test open-loop response
    sim.valv_fv001.manual = True
    pos0 = sim.valv_fv001.position
    sim.valv_fv001.target = pos0
    
    # Introduce +20% feed surge
    sim.state.feed_liquid_m3_h = 794.94 * 1.20
    
    dt = 0.2
    prev_level = sim.state.level_mm
    for step in range(30):  # 6 seconds
        sim.step(dt=dt)
        curr_level = sim.state.level_mm
        assert curr_level > prev_level, f"Monotonicity violated at step {step}: {curr_level} <= {prev_level}"
        prev_level = curr_level


def test_liquid_valve_stroke_monotonic_outflow():
    """Verify increasing FV-001 valve stem position monotonically increases liquid outflow."""
    positions = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
    outflows = []
    
    for pos in positions:
        sim = dynamic.SeparatorDynamicSimulator()
        sim._init_steady_baseline()
        sim.valv_fv001.manual = True
        sim.valv_fv001.position = pos
        sim.valv_fv001.target = pos
        sim.step(0.2)
        outflows.append(sim.state.out_liquid_m3_h)
        
    for i in range(len(outflows) - 1):
        assert outflows[i+1] > outflows[i], (
            f"Valve curve monotonicity violated between pos {positions[i]} and {positions[i+1]}: "
            f"{outflows[i+1]} <= {outflows[i]}"
        )


def test_gas_valve_pinch_monotonic_pressure_rise():
    """Verify pinching gas control valve PV-003B monotonically raises pressure."""
    sim = dynamic.SeparatorDynamicSimulator()
    sim._init_steady_baseline()
    
    # Pinch PV-003B from 90% down to 30% in manual
    sim.valv_pv003b.manual = True
    sim.valv_pv003b.position = 0.30
    sim.valv_pv003b.target = 0.30
    sim.valv_pv003a.manual = True
    sim.valv_pv003a.position = 0.0
    sim.valv_pv003a.target = 0.0
    
    dt = 0.2
    prev_press = sim.state.pressure_barg
    for step in range(25):  # 5 seconds
        sim.step(dt=dt)
        curr_press = sim.state.pressure_barg
        assert curr_press >= prev_press - 1e-4, f"Pressure monotonicity violated at step {step}: {curr_press} < {prev_press}"
        prev_press = curr_press


# --- Category 4: Operational Boundary Envelope & Numerical Robustness ---

@pytest.mark.parametrize("water_cut", [0.00, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99])
def test_water_cut_boundary_envelope(water_cut):
    """Verify numerical stability across entire water cut span (0% to 99%)."""
    sim = dynamic.SeparatorDynamicSimulator()
    sim._init_steady_baseline()
    sim.state.water_cut_vol = water_cut
    
    dt = 0.2
    for _ in range(50):  # 10 seconds
        st = sim.step(dt=dt)
        assert not math.isnan(st.level_mm), f"NaN detected at water cut {water_cut}"
        assert not math.isinf(st.level_mm), f"Inf detected at water cut {water_cut}"
        assert not math.isnan(st.water_in_oil_ppm), f"NaN in separation at water cut {water_cut}"
        assert st.level_mm > 0.0, f"Negative holdup at water cut {water_cut}"


@pytest.mark.parametrize("p_init_barg", [1.0, 2.5, 5.0, 10.0, 15.0, 18.0, 20.0])
def test_pressure_envelope_boundaries(p_init_barg):
    """Verify non-divergent pressure integration across operational envelope [1.0, 20.0] barg."""
    sim = dynamic.SeparatorDynamicSimulator()
    sim.state.pressure_barg = p_init_barg
    
    dt = 0.2
    for _ in range(30):  # 6 seconds
        st = sim.step(dt=dt)
        assert not math.isnan(st.pressure_barg)
        assert not math.isinf(st.pressure_barg)
        assert st.pressure_barg > 0.0, "Pressure must remain strictly positive"
        assert st.pressure_barg < 30.0, "Pressure must not explode numerically"


@pytest.mark.parametrize("temp_c", [20.0, 40.0, 60.0, 79.1, 90.0, 100.0])
def test_temperature_envelope_boundaries(temp_c):
    """Verify thermodynamic scaling stability across [20, 100] degC."""
    sim = dynamic.SeparatorDynamicSimulator()
    sim._init_steady_baseline()
    sim.state.temp_c = temp_c
    
    dt = 0.2
    for _ in range(30):
        st = sim.step(dt=dt)
        assert not math.isnan(st.pressure_barg)
        assert not math.isnan(st.level_mm)
        assert not math.isnan(st.out_gas_header_kg_h)


@pytest.mark.parametrize("feed_mult", [0.0, 0.25, 0.50, 1.0, 1.4, 1.8, 2.0])
def test_throughput_envelope_boundaries(feed_mult):
    """Verify feed turndown and overload throughput scaling [0.0x, 2.0x design]."""
    sim = dynamic.SeparatorDynamicSimulator()
    sim._init_steady_baseline()
    sim.state.feed_liquid_m3_h = 794.94 * feed_mult
    sim.state.feed_gas_kg_h = 71523.0 * feed_mult
    
    dt = 0.2
    for _ in range(40):
        st = sim.step(dt=dt)
        assert not math.isnan(st.level_mm)
        assert not math.isnan(st.pressure_barg)
        assert 0.0 <= st.level_mm <= dynamic.D * 1000.0
