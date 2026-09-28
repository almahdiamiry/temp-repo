"""DESIGN.md 12 sensitivity check: +/-10% feed, +/-10% pressure, +/-5 degC
must not produce nonphysical values. One-off verification script."""

import math

import model

FLOW_KEYS = ("gas_kg_h", "gas_m3_h", "oil_kg_h", "oil_m3_h",
             "water_kg_h", "water_m3_h", "liq_kg_h", "liq_m3_h")
POS_KEYS = ("t_res_min", "v_gas_ms", "v_nozzle_ms", "v_max_ms",
            "k_actual_ms", "v_liquid_m3", "a_liquid_m2", "a_gas_m2")


def check(name: str, feed: float, pres: float, dtemp: float) -> None:
    orig = model.CASES[name]
    c = dict(orig)
    for k in FLOW_KEYS:
        c[k] *= feed
    c["P_bar_a"] = orig["P_bar_a"] * pres
    c["T_C"] = orig["T_C"] + dtemp
    model.CASES[name] = c
    try:
        r = model.simulate(name)
        for k in POS_KEYS:
            v = r[k]
            assert math.isfinite(v) and v > 0, f"{name} f{feed} p{pres} t{dtemp}: {k}={v}"
        # 0.01% tolerance: source table rounds liq_m3_h to 3 decimals
        assert abs(sum(r["vol_pct"][k] for k in ("gas", "oil", "water")) - 100.0) < 0.01
        assert r["k_ok"], f"{name}: k_actual {r['k_actual_ms']:.4f} >= K {model.K_SB}"
    finally:
        model.CASES[name] = orig


if __name__ == "__main__":
    n = 0
    for name in model.CASES:
        for feed in (0.9, 1.0, 1.1):
            for pres in (0.9, 1.0, 1.1):
                for dtemp in (-5.0, 0.0, 5.0):
                    check(name, feed, pres, dtemp)
                    n += 1
    print(f"OK sensitivity: {n} perturbations across {len(model.CASES)} cases, "
          f"all finite/positive, vol% sums 100, k_actual < K")
