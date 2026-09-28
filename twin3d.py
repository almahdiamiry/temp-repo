"""Interactive 3D Digital Twin & Gamified OTS Console for CP2-V-71101 (Three.js / WebGL).

Industrial-Grade Operator Training Simulator (OTS) running at 60 FPS:
- Authentic Industrial Asset 3D Rendering per Engineering Documents & GA Drawings:
  * Finished Ground Level FGL EL. +100.000 (y = -3.0m) with industrial concrete slab, grid, and yellow safety markings.
  * Reinforced concrete pedestal piers with Fixed Saddle (N - 629.950, x = -4.8m) and Sliding Saddle (N - 639.550, x = +4.8m, span 9.6m) with PTFE teflon slide pad and slotted expansion bolts.
  * Elevated Liquid Valve Platform PLTF EL. 101.111 with 4 structural steel support legs grounded at y = -3.0m and safety caged access ladder.
  * Upper Vapor Platform PLTF EL. 104.506 / 107.656 with diagonal structural knee braces and vertical columns anchored to vessel shell stiffeners.
- True Cylindrical Fluid Volume Conformance:
  * Liquid volumes (water, crude oil, emulsion band) conform 100% inside vessel cylinder using clipping planes and dynamic chord surface plates with ZERO protruding corners.
- Comprehensive Field Piping & Spacing per GA 4033-0197/0198:
  * Feed line (30") entering N1 at left dish head (x = -6.8m, y = +0.55m) with UZV-051 and UZV-052 SDVs.
  * Liquid outlet (24") from N2 (x = +5.8m, y = -2.10m) with UZV-002, 20x14 eccentric reducer (FOB), FV-001 (14" Koso S1200 globe with C300 piston actuator), and 711-FE-001 Coriolis mass flow meter.
  * Gas outlet (20") from N3 (x = +4.4m, y = +2.10m) with concentric reducer, 16" riser to B.O.P. EL. 109.005, UZV-003 (16" SDV), flare tee with PV-003A (14" Koso with acoustic baffle), PV-003B (14" Koso butterfly with AT1001U actuator & 49L volume tank), FT-002 (16" orifice meter with ABB 266 DP transmitter), and dual PSVs (PSV-001A/B).
- Primary P&ID Transmitters & Instrumentation Suite:
  * FT-002 gas orifice flow meter, PT-001/003 operating pressure transmitters, PZT-001 safety SIS PAHH transmitter, TT-001/TI-001 thermowell, GD-001 toxic gas detector, and CR-001 corrosion probe.
- Sulzer Chemtech Internals per Drawing MGP1-POMJ0S0006-B01-0001:
  * Schoepentoeter GIVS (20 vanes, length 1915 mm, y = +0.55m) with nozzle mounting brackets.
  * Perforated Baffle Plate (30% NFA at EL. +7000 mm -> x = +0.20m) with 8 circumferential wall weld lugs.
  * Mellachevron H2V2 droplet coalescer (EL. +9100 mm -> x = +2.30m) with structural support shelf angle.
  * Horizontal Demister Box directly under N3 (EL. +11200 mm -> x = +4.40m) suspended by 4 vertical threaded SS316 hanger rods.
  * Semi-transparent woven wire-mesh material for KnitMesh pads.
  * Cruciform vortex breaker inside N2 (EL. +12600 mm -> x = +5.80m).
- Universal Educational Cutaway Toggle:
  * 3 Modes: Full Plant Educational (vessel + hollow pipes + valve plugs/discs/cavities sliced open), Vessel Only, and Solid Exterior.
- Continuous CFD Multiphase Separation Flow & Schoepentoeter Waterfalls:
  * Dynamic continuous feed jet rushing into N1.
  * Continuous cascading curved liquid sheets (waterfalls) pouring from Schoepentoeter vanes into liquid pool.
  * Dynamic undulating wave ripples on oil/water surfaces with perforated baffle calming effect.
  * Interactive Cold Startup & Dynamic Vessel Fill mission from 0 mm to 1550 mm NLL.
"""

import os
from typing import Dict, Any, Tuple

CAP_100KBOPD_M3_H = 794.94


def get_three_js_bundle() -> Tuple[str, str, str]:
    """Read vendored Three.js, OrbitControls, and extracted Navisworks IFC4 CAD meshes."""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    three_path = os.path.join(assets_dir, "three.min.js")
    controls_path = os.path.join(assets_dir, "OrbitControls.js")
    navis_path = os.path.join(assets_dir, "navis_meshes.js")
    
    three_code = ""
    controls_code = ""
    navis_code = ""
    
    if os.path.exists(three_path):
        with open(three_path, "r", encoding="utf-8") as f:
            three_code = f.read()
    else:
        three_code = 'console.warn("three.min.js not found locally");'
        
    if os.path.exists(controls_path):
        with open(controls_path, "r", encoding="utf-8") as f:
            controls_code = f.read()

    if os.path.exists(navis_path):
        with open(navis_path, "r", encoding="utf-8") as f:
            navis_code = f.read()
            
    return three_code, controls_code, navis_code


def render_ots_app_html(
    is_arabic: bool = False,
    initial_scenario: str = "SCN-01"
) -> str:
    """Generate the complete, zero-flicker 60 FPS Digital Twin & OTS Application HTML."""
    three_code, controls_code, navis_code = get_three_js_bundle()
    template_path = os.path.join(os.path.dirname(__file__), "assets", "twin3d_template.html")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Twin3D template not found at {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    html = template.replace("__HTML_LANG__", "ar" if is_arabic else "en") \
                   .replace("__HTML_DIR__", "rtl" if is_arabic else "ltr") \
                   .replace("__IS_ARABIC__", "true" if is_arabic else "false") \
                   .replace("__THREE_CODE__", three_code) \
                   .replace("__CONTROLS_CODE__", controls_code) \
                   .replace("__NAVIS_MESHES_CODE__", navis_code)

    return html


def render_3d_twin_html(*args, **kwargs) -> str:
    """Backwards-compatible wrapper calling render_ots_app_html."""
    is_ar = kwargs.get("is_arabic", False)
    return render_ots_app_html(is_arabic=is_ar)
