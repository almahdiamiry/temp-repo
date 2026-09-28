"""Stage 1 production separator (CP2-V-71101 / CP2-V-71201) steady-state model.

Single source of truth for the HMI (6c). All locked values from
docs/plans/stage1-separator/DESIGN.md v1.0 (F02 Rev 003, PX-2105, PX-1206).
Stdlib only.
"""

import math

# --- Locked design data (DESIGN.md sections 2-8) ---

D = 4.2                      # vessel ID, m
R = D / 2                    # 2.1 m
L_TT = 13.6                  # tangent-tangent length, m (v1: active length = L_TT, A7)
A_TOTAL = math.pi * D * D / 4  # 13.854 m2
K_SB = 0.150                 # Souders-Brown K, m/s (F02 SulsepLL target)
A_GAS_NOZZLE = 0.177         # N3 gas outlet, 20" NB / 476 mm ID, m2

LEVELS_MM = {"T": 4200, "HH": 2700, "H": 2350, "NLL": 1550, "L": 1200, "LL": 770}
NLL_MM = LEVELS_MM["NLL"]

ENVELOPE_BAR_A = (6.0, 12.0)  # stated as 5-11 barg; check in bar a (DESIGN.md 8)
P_TRIP_BARG = 15.0            # PZT-001 trip (PX-3363)
H2S_DEFAULT_PPM = 4600        # design max; plant feed cases 2600 (avg) / 4600 (max) ppm mol (PX-1216 p.10); display only in v1

CASES = {
    "Case 1 x1.4": {
        "gas_kg_h": 100132, "gas_m3_h": 16885.700, "gas_rho": 5.930,
        "oil_kg_h": 864292, "oil_m3_h": 986.759, "oil_rho": 875.89,
        "water_kg_h": 190856, "water_m3_h": 197.711, "water_rho": 965.33,
        "liq_kg_h": 1055148, "liq_m3_h": 1184.469,
        "T_C": 79.10, "P_bar_a": 6.000,
        "dp_inlet_mbar": 48.32, "dp_gas_out_mbar": 20.60,
        "dp_liq_out_mbar": 3.15, "dp_total_mbar": 68.92,
    },
    "Case 4 x1.4": {
        "gas_kg_h": 81193, "gas_m3_h": 7223.577, "gas_rho": 11.240,
        "oil_kg_h": 871377, "oil_m3_h": 1001.721, "oil_rho": 869.88,
        "water_kg_h": 193189, "water_m3_h": 200.171, "water_rho": 965.12,
        "liq_kg_h": 1064566, "liq_m3_h": 1201.891,
        "T_C": 79.70, "P_bar_a": 12.000,
        "dp_inlet_mbar": 22.34, "dp_gas_out_mbar": 7.15,
        "dp_liq_out_mbar": 3.22, "dp_total_mbar": 29.49,
    },
}

# Source trace per displayed value (DESIGN.md section 11).
TRACE = {
    "p_bar_a": "F02 Rev 003 SulsepLL case table (pages 5-6)",
    "T_C": "F02 Rev 003 SulsepLL case table (pages 5-6)",
    "h2s_ppm": "PX-1216 p.10: plant feed cases 2600 (avg) / 4600 (max) ppm mol (design max per PX-5507 Rev 004); sim model default feed 1000/1500 (Table 4-2 p.12)",
    "flows_m3_h": "F02 Rev 003 SulsepLL case table",
    "vol_pct": "F02 case flows via DESIGN.md section 5 volume relations",
    "level_mm": "F02 SulsepLL elevation table (NLL 1550 mm)",
    "v_liquid_m3": "chord geometry A_L(h) * L_TT (DESIGN.md section 4, A7)",
    "v_liquid_geometric_m3": "active + 2x 2:1 SE heads V_heads(h) (DESIGN.md 4; ASME 2:1 SE)",
    "t_res_geometric_min": "geometric (active+heads) volume / Q_L (Phase 1 informative)",
    "gas_actual_m3_h": "mass-conserved gas expansion from F02 case (P,T) anchor (DESIGN.md 8, Phase 1)",
    "rho_gas_actual": "case gas density scaled (P/P_case)(T_case/T) ideal-gas (DESIGN.md 8)",
    "v_max_bulk_ms": "Souders-Brown with bulk liquid density (Phase 1 toggle; DESIGN.md 7)",
    "t_res_min": "t_res = V_L / Q_L at NLL (DESIGN.md section 6)",
    "v_gas_ms": "Q_gas / A_gas at NLL (DESIGN.md section 7)",
    "v_nozzle_ms": "Q_gas / A_N3; N3 = 0.177 m2 (PX-2105 nozzle data)",
    "v_max_ms": "Souders-Brown, K = 0.150 m/s (F02 SulsepLL target)",
    "dp_mbar": "F02 Rev 003 SulsepLL internal dP table",
    "alarms": "PX-2310 p.14, PX-3363 p.46, F02 SulsepLL elevation table",
}

def a_liquid(h_m: float) -> float:
    """Liquid cross-section (m2) at elevation h_m from shell bottom."""
    h = min(max(h_m, 0.0), D)
    return R * R * math.acos((R - h) / R) - (R - h) * math.sqrt(2 * R * h - h * h)


def v_heads(h_m: float) -> float:
    """Combined partial volume (m3) of the two 2:1 semi-ellipsoidal heads at level h_m.

    Full heads = pi*D^3/12 (19.4 m3). Informative (Phase 1); not part of the
    v1.0 active settling volume, which is calibrated to the F02 residence time.
    """
    x = min(max(h_m, 0.0), D) / D
    return (math.pi * D ** 3 / 12.0) * (3.0 * x * x - 2.0 * x * x * x)


def _alarm_states(level_mm: float, p_barg: float) -> list:
    rows = [
        ("CP2-711-LZT-001", f"Level HH > {LEVELS_MM['HH']} mm",
         level_mm > LEVELS_MM["HH"], "TRIP", "Close UZV-051, UZV-052, UZV-003"),
        ("CP2-711-LZT-001", f"Level LL < {LEVELS_MM['LL']} mm",
          level_mm < LEVELS_MM["LL"], "TRIP", "Close UZV-002, UZV-003 (liquid outlet)"),
        ("CP2-711-PZT-001", f"Pressure HH > {P_TRIP_BARG:.0f} barg",
         p_barg > P_TRIP_BARG, "TRIP", "Close UZV-051, UZV-052"),
        ("LA/H", f"Level H > {LEVELS_MM['H']} mm",
         level_mm > LEVELS_MM["H"], "ALARM", "Notify operator"),
        ("LA/L", f"Level L < {LEVELS_MM['L']} mm",
         level_mm < LEVELS_MM["L"], "ALARM", "Notify operator"),
    ]
    return [
        {"tag": tag, "condition": cond,
         "state": state if hit else "OK", "action": action}
        for tag, cond, hit, state, action in rows
    ]


def simulate(case: str = "Case 1 x1.4", level_mm: float = NLL_MM,
             h2s_ppm: float = H2S_DEFAULT_PPM, overrides=None) -> dict:
    """Steady-state result for one F02 case at one liquid level.

    overrides: optional dict of case-table fields to replace (operator inputs).
    """
    c = dict(CASES[case])
    if overrides:
        c.update(overrides)
    h = level_mm / 1000.0
    a_l = a_liquid(h)
    a_g = A_TOTAL - a_l
    q_liq = c["liq_m3_h"] / 3600.0
    v_l = a_l * L_TT                      # active (cylindrical) settling volume
    v_l_geo = v_l + v_heads(h)           # geometric (active + 2:1 heads), Phase 1
    p_bar_a = c["P_bar_a"]
    p_barg = p_bar_a - 1.013
    # Phase 1: mass-conserved gas expansion anchored to the selected F02 case point.
    # Factor == 1 at the case (P_case, T_case), so v1.0 results are unchanged; only
    # operator P/T overrides (or check_sensitivity's in-place CASES edit) are identity
    # here, and a lowered P inflates actual gas volume -> higher v_gas / k_actual.
    p_case = CASES[case]["P_bar_a"]
    t_case_k = CASES[case]["T_C"] + 273.15
    t_k = c["T_C"] + 273.15
    p_ratio = p_case / p_bar_a
    t_ratio = t_k / t_case_k
    q_gas = c["gas_m3_h"] / 3600.0 * p_ratio * t_ratio
    rho_gas = c["gas_rho"] * (1.0 / p_ratio) * (1.0 / t_ratio)
    wc = c["water_m3_h"] / c["liq_m3_h"] if c["liq_m3_h"] else 0.0
    rho_liq_bulk = wc * c["water_rho"] + (1.0 - wc) * c["oil_rho"]
    rho_ratio = (c["oil_rho"] - rho_gas) / rho_gas
    rho_ratio_bulk = (rho_liq_bulk - rho_gas) / rho_gas
    k_actual = q_gas / a_g / math.sqrt(rho_ratio)
    return {
        "case": case,
        "h2s_ppm": h2s_ppm,
        "T_C": c["T_C"],
        "p_bar_a": p_bar_a,
        "p_barg": p_barg,
        "in_envelope": ENVELOPE_BAR_A[0] <= p_bar_a <= ENVELOPE_BAR_A[1],
        "flows_m3_h": {"gas": c["gas_m3_h"], "oil": c["oil_m3_h"],
                       "water": c["water_m3_h"], "liquid": c["liq_m3_h"]},
        "flows_kg_h": {"gas": c["gas_kg_h"], "oil": c["oil_kg_h"],
                       "water": c["water_kg_h"], "liquid": c["liq_kg_h"]},
        "vol_pct": {
            "gas": c["gas_m3_h"] / (c["gas_m3_h"] + c["liq_m3_h"]) * 100,
            "oil": c["oil_m3_h"] / (c["gas_m3_h"] + c["liq_m3_h"]) * 100,
            "water": c["water_m3_h"] / (c["gas_m3_h"] + c["liq_m3_h"]) * 100,
            "water_of_liquid": c["water_m3_h"] / c["liq_m3_h"] * 100,
        },
        "level_mm": level_mm,
        "a_liquid_m2": a_l,
        "a_gas_m2": a_g,
        "v_liquid_m3": v_l,
        "v_liquid_geometric_m3": v_l_geo,
        "t_res_min": v_l / q_liq / 60.0,
        "t_res_geometric_min": v_l_geo / q_liq / 60.0,
        "v_gas_ms": q_gas / a_g,
        "v_nozzle_ms": q_gas / A_GAS_NOZZLE,
        "gas_actual_m3_h": q_gas * 3600.0,
        "rho_gas_actual": rho_gas,
        "v_max_ms": K_SB * math.sqrt(rho_ratio),
        "v_max_bulk_ms": K_SB * math.sqrt(rho_ratio_bulk),
        "k_actual_ms": k_actual,
        "k_ok": k_actual < K_SB,
        "dp_mbar": {"inlet": c["dp_inlet_mbar"], "gas_out": c["dp_gas_out_mbar"],
                    "liq_out": c["dp_liq_out_mbar"], "total": c["dp_total_mbar"]},
        "alarms": _alarm_states(level_mm, p_barg),
    }


def _self_check() -> None:
    r1 = simulate("Case 1 x1.4")
    r4 = simulate("Case 4 x1.4")
    checks = [
        ("t_res C1 (min)", r1["t_res_min"], 3.199, 0.005),
        ("t_res C4 (min)", r4["t_res_min"], 3.153, 0.005),
        ("v_gas C1 (m/s)", r1["v_gas_ms"], 0.5093, 0.0005),
        ("v_gas C4 (m/s)", r4["v_gas_ms"], 0.2179, 0.0005),
        ("k_actual C1 (m/s)", r1["k_actual_ms"], 0.0420, 0.0005),
        ("k_actual C4 (m/s)", r4["k_actual_ms"], 0.0249, 0.0005),
        ("water vol% C1", r1["vol_pct"]["water_of_liquid"], 16.69, 0.01),
        ("water vol% C4", r4["vol_pct"]["water_of_liquid"], 16.655, 0.01),
        ("gas vol% C1", r1["vol_pct"]["gas"], 93.45, 0.01),
        ("gas vol% C4", r4["vol_pct"]["gas"], 85.74, 0.01),
        ("dP total C1 (mbar)", r1["dp_mbar"]["total"], 68.92, 0.005),
        ("dP total C4 (mbar)", r4["dp_mbar"]["total"], 29.49, 0.005),
        ("dP sum C1 (mbar)", r1["dp_mbar"]["inlet"] + r1["dp_mbar"]["gas_out"], 68.92, 0.005),
        ("dP sum C4 (mbar)", r4["dp_mbar"]["inlet"] + r4["dp_mbar"]["gas_out"], 29.49, 0.005),
        ("L/D", L_TT / D, 3.238, 0.001),
        ("v_liq_geo C1 (m3)", r1["v_liquid_geometric_m3"], 69.15, 0.05),
        ("gas_actual C1 anchor (m3/h)", r1["gas_actual_m3_h"], 16885.700, 0.01),
        ("rho_gas C1 anchor", r1["rho_gas_actual"], 5.930, 0.005),
    ]
    failed = [(n, g, w) for n, g, w, t in checks if abs(g - w) > t]
    for n, g, w in failed:
        print(f"FAIL {n}: got {g:.5f}, want {w}")
    r1_drop = simulate("Case 1 x1.4", overrides={"P_bar_a": 5.0})
    assert r1_drop["gas_actual_m3_h"] > r1["gas_actual_m3_h"], "gas must expand as P drops (mass conserved)"
    assert r1["in_envelope"] and r4["in_envelope"], "pressure envelope"
    assert r1["k_ok"] and r4["k_ok"], "K check"
    assert not failed, f"{len(failed)} check(s) failed"
    print(f"OK {len(checks)} checks + envelope + K "
          f"(t_res C1 {r1['t_res_min']:.3f} / C4 {r4['t_res_min']:.3f} min; "
          f"k_actual {r1['k_actual_ms']:.4f} / {r4['k_actual_ms']:.4f} < K {K_SB})")


if __name__ == "__main__":
    _self_check()
