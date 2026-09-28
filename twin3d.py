"""Interactive 3D Digital Twin & OTS Console for CP2-V-71101 (Three.js / WebGL).

Industrial-Grade Operator Training Simulator (OTS) running at 60 FPS.

PHYSICAL GEOMETRY AUTHORITY
---------------------------
All physical external equipment is rendered from exactly one source: the canonical
CAD-derived glTF binary ``assets/final_twin.glb``, loaded at runtime through
``GLTFLoader`` (Three.js r186 ES modules). There is no second physical
representation. The GLB supplies the vessel shell and nozzles/saddles, the skid
structure and platforms, every valve body, every process pipe run, and the
dedicated ``tag:*`` instrument nodes; the registry in ``config/tag_registry.json``
declares which canonical tag each GLB node serves.

Retained client-side geometry is deliberately NON-physical and educational only:
the Sulzer Chemtech internals, the conforming cylindrical fluid volumes, the
inlet feed stream, Schoepentoeter knockout cascades, vapour ribbons, and
multiphase particles. These are visualisation layers, not equipment
reconstructions, and they are absent from the GLB by design.

- Authentic Industrial Asset 3D Rendering per Engineering Documents & GA Drawings:
  * Finished Ground Level FGL EL. +100.000 (y = -3.0m) with industrial concrete slab, grid, and yellow safety markings.
- True Cylindrical Fluid Volume Conformance:
  * Liquid volumes (water, crude oil, emulsion band) conform 100% inside vessel cylinder using clipping planes and dynamic chord surface plates with ZERO protruding corners.
- Comprehensive Field Piping & Spacing per GA 4033-0197/0198 (served by the canonical GLB, not reconstructed in code):
  * Feed line (30") entering N1 at the left dish head with UZV-051 and UZV-052 SDVs, N1 flange and INSULATION/END1 dished cap. UZV-051/UZV-052 sit outside the local skid bounds, so their absence from the GLB is an authentic scope boundary; their control and interlock coupling is preserved in dynamic.py.
  * Liquid outlet (24") from N2 with UZV-002, 20x14 eccentric reducer (FOB), FV-001 (14" Koso S1200 globe with C300 piston actuator), and 711-FE-001 Coriolis mass flow meter.
  * Gas outlet (20") from N3 with concentric reducer, 16" riser to B.O.P. EL. 109.005, UZV-003 (16" SDV), flare tee with PV-003A (14" Koso with acoustic baffle), PV-003B (14" Koso butterfly with AT1001U actuator & 49L volume tank), FT-002 (16" orifice meter with ABB 266 DP transmitter), and dual PSVs (PSV-001A/B, located on the HP flare header skid).
- Primary P&ID Transmitters & Instrumentation Suite (served by the canonical GLB `tag:*` nodes):
  * LT-002 level bridle & guided wave radar, PT-003 operating pressure transmitter, TT-051 crude temperature transmitter, plus the upstream inlet HIPPS station UZV-053, UZV-054, PT-052, PZT-053 and SP-CP02.
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

CAP_100KBOPD_M3_H = 794.94


def get_asset_base_url(bridge_port: int) -> str:
    """Absolute origin under which the bridge server exposes ``/assets/*``.

    The renderer is a native ES module, so Three.js, GLTFLoader and the
    canonical GLB are fetched over HTTP rather than inlined into the document.
    An absolute base is required because the same HTML is served both from the
    bridge root and from a Streamlit-hosted iframe, whose document origin
    differs from the bridge origin.
    """
    return f"http://127.0.0.1:{bridge_port}/assets"


def render_ots_app_html(
    is_arabic: bool = False,
    initial_scenario: str = "SCN-01"
) -> str:
    """Generate the complete, zero-flicker 60 FPS Digital Twin & OTS Application HTML."""
    import json
    import dynamic

    template_path = os.path.join(os.path.dirname(__file__), "assets", "twin3d_template.html")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Twin3D template not found at {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Launch local bridge server if not already running
    bridge_port = 8765
    try:
        import operating_world
        server, bridge = operating_world.start_bridge_server(port=bridge_port)
        bridge_port = server.server_port
        sim = bridge.sim
    except Exception as e:
        sim = dynamic.SeparatorDynamicSimulator()
        if initial_scenario and initial_scenario != "SCN-01":
            dynamic.apply_scenario(sim, initial_scenario)

    initial_state_json = json.dumps(sim.get_canonical_state())

    html = template.replace("__HTML_LANG__", "ar" if is_arabic else "en") \
                   .replace("__HTML_DIR__", "rtl" if is_arabic else "ltr") \
                   .replace("__IS_ARABIC__", "true" if is_arabic else "false") \
                   .replace("__ASSET_BASE_URL__", get_asset_base_url(bridge_port)) \
                   .replace("__BRIDGE_PORT__", str(bridge_port)) \
                   .replace("__INITIAL_CANONICAL_STATE__", initial_state_json)

    return html


def render_3d_twin_html(*args, **kwargs) -> str:
    """Backwards-compatible wrapper calling render_ots_app_html."""
    is_ar = kwargs.get("is_arabic", False)
    return render_ots_app_html(is_arabic=is_ar)
