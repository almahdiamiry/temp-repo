"""Stage 1 Production Separator (CP2-V-71101) Interactive Dynamic OTS & HMI.

Provides a dual-mode industry-grade Operator Training Simulator (OTS):
1. 🎮 Dynamic OTS Playground & 3D Digital Twin (Phase 2):
   - Interactive 3D WebGL Digital Twin (Three.js) with cutaway view, Sulzer internals,
     Stokes settling particles, and raycasting click cards.
   - High-performance 2D DCS P&ID console (ISA-101 compliant animated SVG).
   - Real-time Two-Rate solver engine (dynamic.py) with Play/Pause/Step/Speed controls.
   - 7 Realistic operational scenarios (Surge challenge, Stuck valves, Emulsion, Gas blow-by).
   - Operator Career Progression and live KPI scoring engine (⭐⭐⭐ Master Console Operator).
   - Bilingual educational concept cards (English & Arabic).
2. 📋 Steady-State Design Model (Phase 1):
   - Single source of truth baseline from model.py for locked design point verification.

Run with:
    streamlit run simulator/stage1/app.py
"""

import html
import math
import time
import streamlit as st

import model
import dynamic
import twin3d

st.set_page_config(
    page_title="CP2-V-71101 Stage 1 Separator OTS",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("<style>.stApp { background-color: #12151a; color: #d7dde5; }</style>",
            unsafe_allow_html=True)

PXMM = 200.0 / 4200.0   # px per mm in the vessel drawing
YB = 415.0             # svg y of vessel bottom (vessel y=215..415, 200 px tall)

STATE_COLORS = {"OK": "#34c06c", "ALARM": "#ffb020", "TRIP": "#ff5252"}


def y_of(mm: float) -> float:
    return YB - mm * PXMM


def row_state(r: dict, tag_end: str, cond: str) -> str:
    for a in r.get("alarms", []):
        if a["tag"].endswith(tag_end) and cond in a["condition"]:
            return a["state"]
    return "OK"


def fv_open(r: dict) -> bool:
    return row_state(r, "LZT-001", "LL") != "TRIP"


def bubble(x, y, r, tag, val, vcolor="#d7dde5", square=False, tip=None):
    el = (f'<rect x="{x - r}" y="{y - r}" width="{2 * r}" height="{2 * r}" rx="4" '
          f'fill="#262b33" stroke="{vcolor}" stroke-width="1.5"/>' if square else
          f'<circle cx="{x}" cy="{y}" r="{r}" fill="#262b33" stroke="{vcolor}" stroke-width="1.5"/>')
    out = (el
           + f'<text x="{x}" y="{y - 4}" text-anchor="middle" font-size="10" fill="#8b95a3">{tag}</text>'
           + f'<text x="{x}" y="{y + 11}" text-anchor="middle" font-size="10" font-weight="bold" '
             f'fill="{vcolor}">{val}</text>')
    return f'<g><title>{html.escape(tip)}</title>{out}</g>' if tip else out


def valve(x, y, tag, state, ctrl=False, tip=None):
    c = STATE_COLORS[state]
    out = (f'<polygon points="{x - 13},{y - 8} {x},{y} {x - 13},{y + 8}" fill="none" stroke="{c}" stroke-width="1.5"/>'
           f'<polygon points="{x + 13},{y - 8} {x},{y} {x + 13},{y + 8}" fill="none" stroke="{c}" stroke-width="1.5"/>')
    ty = y - 14
    if ctrl:
        out += (f'<line x1="{x}" y1="{y - 8}" x2="{x}" y2="{y - 22}" stroke="{c}" stroke-width="1.5"/>'
                f'<polygon points="{x - 9},{y - 22} {x + 9},{y - 22} {x},{y - 32}" fill="none" stroke="{c}" stroke-width="1.5"/>')
        ty = y - 38
    out += f'<text x="{x}" y="{ty}" text-anchor="middle" font-size="10" font-weight="bold" fill="{c}">{tag}</text>'
    return f'<g><title>{html.escape(tip)}</title>{out}</g>' if tip else out


def sdv(x, y, tag, state, tip=None):
    c = STATE_COLORS[state]
    out = (f'<polygon points="{x - 13},{y - 8} {x},{y} {x - 13},{y + 8}" fill="none" stroke="{c}" stroke-width="1.5"/>'
           f'<polygon points="{x + 13},{y - 8} {x},{y} {x + 13},{y + 8}" fill="none" stroke="{c}" stroke-width="1.5"/>')
    out += (f'<line x1="{x}" y1="{y - 8}" x2="{x}" y2="{y - 18}" stroke="{c}" stroke-width="1.5"/>'
            f'<rect x="{x - 9}" y="{y - 30}" width="18" height="12" fill="none" stroke="{c}" stroke-width="1.5"/>')
    out += f'<text x="{x}" y="{y - 34}" text-anchor="middle" font-size="9" font-weight="bold" fill="{c}">{tag}</text>'
    return f'<g><title>{html.escape(tip)}</title>{out}</g>' if tip else out


def sig(*pts, w=1.2):
    p = " ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
    return f'<polyline points="{p}" fill="none" stroke="#4fa3e3" stroke-width="{w}" stroke-dasharray="5 3"/>'


TAGMAP = {"UZV-051": "UZV-051/052", "UZV-052": "UZV-051/052",
          "PV-003A": "PV-003A/B", "PV-003B": "PV-003A/B",
          "PSV-001A/B": "PSV-001"}

STRUCT_TIPS = {
    "vessel": "CP2-V-71101 - the stage 1 separator. A 4.2 m wide, 13.6 m long horizontal drum. "
              "The incoming stream settles by gravity: gas rises to the top, oil and water settle "
              "at the bottom. Design pressure 21 barg.",
    "gas": "Gas space inside the drum. The separated gas leaves through the top nozzle to the "
           "flashed gas header.",
    "oil": "The oil layer. In v1, oil and water settle together and leave as one liquid stream "
           "to the 2nd stage separator.",
    "water": "The water layer, below the oil. It leaves with the oil to the 2nd stage separator.",
    "feed": "The incoming stream from the production manifold. Two shutdown valves (UZV-051/052) "
            "can cut the feed automatically when a trip happens.",
    "gas_out": "Gas outlet: the 20 in N3 top nozzle splits into two lines - a 16 in line to PV-003B "
                "(then NRV-711005 and UZV-003) to the flashed gas header, and a 14 in line to PV-003A "
                "to the HP flare header. PV-003B runs max open in normal operation (note 19).",
    "flare_hdr": "The HP flare header. The 14 in line (PV-003A) and the PSVs discharge here - gas "
                 "that must be flared goes to the flare.",
    "gas_hdr": "The flashed gas header. The 16 in gas line (PV-003B, max open in normal operation) "
               "leaves here (normal operating range 5-11 barg).",
    "liq_line": "The liquid outlet line to the 2nd stage separator: shutdown valve UZV-002, flow "
                "meter (FE-001 orifice + FT-001), controller FIC-001, control valve FV-001.",
    "scale": "The level scale on the drum, measured from the bottom reference (SulsepLL): "
             "high-high 2700 mm, high 2350, normal (SP) 1550, low 1200, low-low 770.",
    "psv": "PSV-001A and PSV-001B - overpressure protection. Fed from the S1 nozzle at the top of the "
           "drum through a 6 in pipe; each opens automatically at 21 barg (the vessel's design pressure) "
           "and discharges through an 8 in line to the HP flare header. The last line of defence.",
    "sig1003": "Safety cable 1003: pressure high-high (> 15 barg) automatically closes the two "
               "feed shutdown valves UZV-051/052.",
    "sig1004": "Safety cable 1004: level high-high closes UZV-051/052/003; level low-low closes "
               "UZV-002/003 (note 26).",
    "sigpt": "Signal from PT-003 (the pressure gauge) to the two pressure control loops PICA-003A "
             "and PIC-003B.",
    "sigt": "Cascade signal: the level loop LT-002 tells the flow loop FIC-001 how much to flow, "
            "capped at 100 kBOPD (note 23).",
    "sigft": "Signal from the flow meter FT-001 to the flow controller FIC-001.",
    "sigfic": "Controller output: the command from FIC-001 to the FV-001 actuator.",
    "inlet_device": "Inlet device (Schoepentoeter, GIVS Type II) in the 30 in inlet nozzle. It breaks up the "
                   "incoming stream so it does not throw liquid around the drum and does the first gas-liquid "
                   "split - it separates gas from the total liquid, not oil from water. Design: it keeps the "
                   "inlet pressure drop down (48.3 mbar C1 / 22.3 C4 - the biggest part of the 68.9 mbar total). "
                   "(F02 SulsepLL; PX-2365 n.15)",
    "vane_baffle": "Mellachevron vane pack + perforated baffles (removable). They give the gas a large "
                   "disengagement area so droplets settle back by gravity before it leaves, and the draw pipe "
                   "sets the liquid level. Design: this keeps the gas velocity well under the Souders-Brown "
                   "limit - k_actual 0.042 (C1) / 0.025 (C4) m/s vs K = 0.150 m/s. (F02; PX-2365 n.15)",
    "demister": "Mesh-pad demister (2 nos, KnitMesh with grid, removable) on the gas path near the 20 in gas "
                "outlet. It catches the fine mist and entrained droplets and drips them back to the liquid. "
                "Design: it meets the carryover spec of 0.1 USgal/MMscf liquid in gas (Case 4 actual 0.04). "
                "(F02; PX-5527)",
    "vortex_breaker": "Vortex breaker (welded) at the 20 in liquid outlet. It stops the liquid from swirling "
                      "and pulling gas down into the 2nd-stage line (no gas blow-by). It is on the liquid path, "
                      "not the gas path. (MS-2105 SKETCH; PX-2365 n.15)",
}

PLAIN = {
    "PT-003": "A simple pressure gauge for the drum. It sends the vessel pressure to the two "
              "control loops that hold the gas header pressure.",
    "PZT-001": "A pressure high-high detector. If vessel pressure goes above 15 barg it "
               "automatically closes the two feed shutdown valves (safety cable 1003).",
    "PZIA-001": "A trip relay in the control panel: it carries the pressure trip from PZT-001 "
                "to the feed shutdown valves UZV-051/052.",
    "PICA-003A": "A control loop that trims PV-003A, the 14 in line to the HP flare header (the "
                  "flare/vent path). It works with PIC-003B on staggered setpoints (note 20) to "
                  "hold the gas pressure.",
    "PIC-003B": "A control loop for PV-003B, the 16 in main gas line to the flashed gas header. "
                "PV-003B runs nearly wide open in normal operation (note 19); it works with "
                "PICA-003A on staggered setpoints (note 20).",
    "LT-002": "The liquid level gauge of the drum. It does not drive a level valve here - instead "
              "it 'cascades' to the flow loop FIC-001 and steers the outlet flow (capped at "
              "100 kBOPD, note 23).",
    "LZT-001": "A level trip detector. Level above 2700 mm closes UZV-051/052/003; level below "
               "770 mm closes UZV-002/003 (note 26).",
    "LZIA-001": "A trip relay carrying the level trips to the shutdown valves (safety cable 1004).",
    "FT-001": "A flow meter: it measures the liquid flow from the pressure drop across the "
              "FE-001 orifice plate.",
    "FIC-001": "The flow control loop that holds the liquid outlet flow (normally 100 kBOPD). "
               "The level loop LT-002 can steer its target.",
    "PSV-001": "Overpressure protection. It opens automatically at 21 barg (the vessel design "
               "pressure) and discharges to the high-pressure flare - the last line of defence.",
    "PV-003A/B": "The two gas-outlet control valves: PV-003B (16 in) feeds the flashed gas header "
                  "(max open in normal op, note 19); PV-003A (14 in) feeds the HP flare header. "
                  "Both are trimmed by PICA-003A / PIC-003B on staggered setpoints (note 20).",
    "FV-001": "A control valve on the liquid outlet line - it opens and closes per the FIC-001 "
              "signal (and the LT-002 cascade).",
    "UZV-051/052": "Feed shutdown valves on the incoming line. They close on a pressure trip "
                   "(> 15 barg) or a level high-high trip - no operator action needed.",
    "UZV-002": "The first shutdown valve on the liquid outlet. It closes on level low-low to stop "
               "draining the drum.",
    "UZV-003": "A shutdown valve on the 16 in gas line to the flashed gas header. It closes on "
                "level high-high and level low-low (note 26), and also when UZV-002 closes.",
}

TAG_INFO = {
    "PT-003": {
        "eq": "PT-003 / PZT-001 reading; calibrated 0 to 25.0 barg (Honeywell STG74L)",
        "inputs": "Gas space pressure",
        "trace": "p_bar_a",
        "result": lambda r: f'{r["p_barg"]:.3f} barg ({r["p_bar_a"]:.3f} bar a)',
    },
    "PZT-001": {
        "eq": "Pressure trip; TRIP when pressure > 15.0 barg; routes to PZIA-001 (cable 1003)",
        "inputs": "PT-003 reading",
        "trace": "alarms",
        "result": lambda r: "; ".join(
            f'{a["condition"]}: {a["state"]}'
            for a in r["alarms"] if a["tag"].endswith("PZT-001")),
    },
    "PZIA-001": {
        "eq": "Pressure trip interlock (cable 1003): TRIP -> close UZV-051 and UZV-052",
        "inputs": "PZT-001",
        "trace": "alarms",
        "result": lambda r: "; ".join(
            f'{a["condition"]}: {a["state"]}'
            for a in r["alarms"] if a["tag"].endswith("PZT-001")),
    },
    "PICA-003A": {
        "eq": "Gas split-range controller: trims PV-003A (14 in to HP flare); opens when PV-003B saturated",
        "inputs": "PT-003 reading",
        "trace": "p_bar_a",
        "result": lambda r: f'SP {r["p_barg"]:.1f} barg; PV-003A closed in normal op',
    },
    "PIC-003B": {
        "eq": "Gas pressure controller: drives PV-003B (16 in to flashed gas header); runs max open (n.19)",
        "inputs": "PT-003 reading",
        "trace": "p_bar_a",
        "result": lambda r: f'SP {r["p_barg"]:.1f} barg; PV-003B max open (n.19)',
    },
    "PV-003A/B": {
        "eq": "Gas outlet control valves: PV-003B (16 in, Koso AB4001 Cv 3990) + PV-003A (14 in, Koso S1200 Cv 1170)",
        "inputs": "PICA-003A / PIC-003B controllers",
        "trace": "flows_m3_h",
        "result": lambda r: f'Gas flow {r["flows_m3_h"]["gas"]:,.1f} m3/h',
    },
    "PSV-001": {
        "eq": "PSV-001A / PSV-001B set 21.0 barg (vessel design pressure), fire case -> HP flare header",
        "inputs": "Gas space",
        "trace": "alarms",
        "result": lambda r: "SET 21 barg",
    },
    "LZT-001": {
        "eq": "Level trip; TRIP when level > 2700 mm (HH, UZ Grp 15) or < 770 mm (LL, UZ Grp 16); routes to LZIA-001 (cable 1004)",
        "inputs": "Level from GWR radar",
        "trace": "alarms",
        "result": lambda r: "; ".join(
            f'{a["condition"]}: {a["state"]}'
            for a in r["alarms"] if a["tag"].endswith("LZT-001")),
    },
    "LZIA-001": {
        "eq": "Level trip interlock (HH/LL, cable 1004): HH -> close UZV-051/052/003; LL -> close UZV-002/003",
        "inputs": "LZT-001",
        "trace": "alarms",
        "result": lambda r: "; ".join(
            f'{a["condition"]}: {a["state"]}'
            for a in r["alarms"] if a["tag"].endswith("LZT-001")),
    },
    "LT-002": {
        "eq": "Level transmitter /H /L (Magnetrol GWR X706, span 0-2500 mm); cascades to FIC-001",
        "inputs": "Liquid Level",
        "trace": "level_mm",
        "result": lambda r: f'{r["level_mm"]:.0f} mm (NLL 1550 mm)',
    },
    "FIC-001": {
        "eq": "Liquid flow controller; cascade from LT-002 (n.23) + FT-001; drives FV-001; cap 100 kBOPD",
        "inputs": "LT-002 cascade + FT-001",
        "trace": "flows_m3_h",
        "result": lambda r: f'{r["flows_m3_h"]["liquid"]:,.1f} m3/h',
    },
    "FV-001": {
        "eq": "Liquid flow control valve (14 in Koso S1200, Linear trim, Cv 2140); closes on LL trip (< 770 mm)",
        "inputs": "FIC-001 / LZT-001 LL trip state",
        "trace": "alarms",
        "result": lambda r: ("OPEN" if fv_open(r) else "CLOSED"),
    },
    "UZV-002": {
        "eq": "SDV (20 in TYPE 1, n.26) on the 20 in liquid line; closes on level LL trip; triggers UZV-003",
        "inputs": "LZT-001 LL trip state",
        "trace": "alarms",
        "result": lambda r: ("CLOSED" if row_state(r, "LZT-001", "LL") == "TRIP" else "OPEN"),
    },
    "UZV-003": {
        "eq": "SDV on the 16 in gas line; closes on level HH trip (> 2700 mm) and Note 26 LL trip",
        "inputs": "LZT-001 HH / LL trip states + Note 26",
        "trace": "alarms",
        "result": lambda r: ("CLOSED" if row_state(r, "LZT-001", "HH") == "TRIP"
                               or row_state(r, "LZT-001", "LL") == "TRIP" else "OPEN"),
    },
}


def pid_svg(r: dict, explain: bool = False) -> str:
    """Render high-contrast ISA-101 animated SVG of CP2-V-71101 and P&ID loops."""
    h = r["level_mm"]
    liq_top = y_of(h)
    frac_w = r["vol_pct"]["water_of_liquid"] / 100.0
    w_top = YB - h * frac_w * PXMM
    liq = r["flows_m3_h"]["liquid"]
    p = r["p_barg"]

    def tip_tag(tag):
        key = TAGMAP.get(tag, tag)
        info = TAG_INFO[key]
        return (f"{tag}\n{PLAIN.get(key, info['eq'])}\nNow: {info['result'](r)}\n"
                f"Source: {model.TRACE.get(info['trace'], 'Vendor Drawings')}\nClick the tag for details.")

    T = dict(STRUCT_TIPS) if explain else {}
    if explain:
        for tag in TAG_INFO:
            T[tag] = tip_tag(tag)
        for alias in TAGMAP:
            T[alias] = tip_tag(TAGMAP[alias])

    def tt(key):
        return f'<title>{html.escape(T[key])}</title>' if key in T else ""

    p_state = row_state(r, "PZT-001", "")
    fh_state = row_state(r, "LZT-001", "HH")
    ll_state = row_state(r, "LZT-001", "LL")
    lvl_state = "TRIP" if fh_state == "TRIP" or ll_state == "TRIP" else \
                "ALARM" if "ALARM" in (fh_state, ll_state) else "OK"
    fv_state = "OK" if fv_open(r) else "TRIP"
    uzv2_state = "OK" if ll_state != "TRIP" else "TRIP"
    uzv3_state = "OK" if fh_state != "TRIP" and ll_state != "TRIP" else "TRIP"
    uzv12_state = "OK" if p_state != "TRIP" and fh_state != "TRIP" else "TRIP"
    pv_col = "#34c06c" if r.get("in_envelope", True) else "#ff5252"

    ticks = ""
    for name, mm, lbl in [("HH", 2700, "LAHH"), ("H", 2350, "LAH"), ("NLL", 1550, "SP"),
                          ("L", 1200, "LAL"), ("LL", 770, "LALL")]:
        y = y_of(mm)
        c = "#d7dde5" if name == "NLL" else "#8b95a3"
        ticks += (f'<line x1="960" y1="{y:.1f}" x2="972" y2="{y:.1f}" stroke="{c}" stroke-width="1.5"/>'
                  f'<text x="976" y="{y + 3:.1f}" font-size="9" fill="{c}">{lbl} {mm}</text>')

    oil_h = max(0.0, w_top - liq_top)
    wat_h = max(0.0, YB - w_top)
    oil_rect = (f'<g>{tt("oil")}<rect x="310" y="{liq_top:.1f}" width="650" height="{oil_h:.1f}" '
                f'fill="#b27d14" opacity="0.45"/></g>') if oil_h > 0 else ""
    wat_rect = (f'<g>{tt("water")}<rect x="310" y="{w_top:.1f}" width="650" height="{wat_h:.1f}" '
                f'fill="#1f6feb" opacity="0.45"/></g>') if wat_h > 0 else ""

    oil_lbl = (f'<text x="635" y="{liq_top + oil_h / 2 + 4:.1f}" text-anchor="middle" '
               f'font-size="11" font-weight="bold" fill="#ffd166">OIL</text>') if oil_h > 15 else ""
    wat_lbl = (f'<text x="635" y="{w_top + wat_h / 2 + 4:.1f}" text-anchor="middle" '
               f'font-size="11" font-weight="bold" fill="#79c0ff">WATER</text>') if wat_h > 15 else ""

    svg = (
        f'<svg viewBox="0 0 1280 620" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" '
        f'style="background-color: #161b22; border-radius: 8px; border: 1px solid #30363d;">'
        f'<defs>'
        f'<marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 1 L 8 5 L 0 9 z" fill="#8b95a3"/></marker>'
        f'<marker id="arr-cyan" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 1 L 8 5 L 0 9 z" fill="#38d9a9"/></marker>'
        f'</defs>'
        f'<g>{tt("feed")}'
        f'<line x1="80" y1="315" x2="310" y2="315" stroke="#8b95a3" stroke-width="4" marker-end="url(#arr)"/>'
        f'<text x="90" y="300" font-size="11" fill="#8b95a3">From production manifold (30" N1)</text></g>'
        f'{sdv(160, 315, "UZV-051", uzv12_state, tip=T.get("UZV-051"))}'
        f'{sdv(240, 315, "UZV-052", uzv12_state, tip=T.get("UZV-052"))}'
        f'<g>{tt("vessel")}'
        f'<rect x="310" y="215" width="650" height="200" rx="30" fill="#21262d" stroke="#58a6ff" stroke-width="2"/>'
        f'<rect x="310" y="215" width="650" height="200" fill="#21262d" opacity="0.3"/>'
        f'{oil_rect}{wat_rect}'
        f'<text x="635" y="240" text-anchor="middle" font-size="13" font-weight="bold" fill="#79c0ff">CP2-V-71101 (1st Stage Separator)</text>'
        f'<text x="635" y="275" text-anchor="middle" font-size="11" fill="#8b95a3">GAS SPACE</text>'
        f'{oil_lbl}{wat_lbl}'
        f'</g>'
        f'<g>{ticks}</g>'
        f'<g>{tt("gas_out")}'
        f'<line x1="860" y1="215" x2="860" y2="160" stroke="#8b95a3" stroke-width="3"/>'
        f'<line x1="860" y1="160" x2="1080" y2="160" stroke="#8b95a3" stroke-width="3" marker-end="url(#arr)"/>'
        f'<line x1="940" y1="160" x2="940" y2="90" stroke="#8b95a3" stroke-width="3"/>'
        f'<line x1="940" y1="90" x2="1080" y2="90" stroke="#8b95a3" stroke-width="3" marker-end="url(#arr)"/>'
        f'<text x="1090" y="164" font-size="11" fill="#8b95a3">To Flashed Gas Header (16")</text>'
        f'<text x="1090" y="94" font-size="11" fill="#8b95a3">To HP Flare Header (14")</text>'
        f'</g>'
        f'{valve(1010, 160, "PV-003B", "OK", ctrl=True, tip=T.get("PV-003B"))}'
        f'{sdv(1060, 160, "UZV-003", uzv3_state, tip=T.get("UZV-003"))}'
        f'{valve(1010, 90, "PV-003A", "OK", ctrl=True, tip=T.get("PV-003A"))}'
        f'<g>{tt("liq_line")}'
        f'<line x1="860" y1="415" x2="860" y2="520" stroke="#8b95a3" stroke-width="3"/>'
        f'<line x1="860" y1="520" x2="1140" y2="520" stroke="#8b95a3" stroke-width="3" marker-end="url(#arr)"/>'
        f'<text x="1150" y="524" font-size="11" fill="#8b95a3">To 2nd Stage Separator (24")</text>'
        f'</g>'
        f'{sdv(910, 520, "UZV-002", uzv2_state, tip=T.get("UZV-002"))}'
        f'{bubble(970, 520, 18, "FT-001", f"{liq:,.0f}", tip=T.get("FT-001"))}'
        f'{bubble(1030, 520, 18, "FIC-001", "100k", square=True, tip=T.get("FIC-001"))}'
        f'{valve(1090, 520, "FV-001", fv_state, ctrl=True, tip=T.get("FV-001"))}'
        f'{bubble(720, 150, 20, "PT-003", f"{p:.2f}", pv_col, tip=T.get("PT-003"))}'
        f'{bubble(780, 150, 20, "PZT-001", f"{p:.1f}", STATE_COLORS[p_state], square=True, tip=T.get("PZT-001"))}'
        f'{bubble(860, 290, 20, "LT-002", f"{h:.0f}", tip=T.get("LT-002"))}'
        f'{bubble(920, 290, 20, "LZT-001", f"{h:.0f}", STATE_COLORS[lvl_state], square=True, tip=T.get("LZT-001"))}'
        f'{sig((780, 130), (780, 70), (160, 70), (160, 280))}'
        f'{sig((920, 310), (920, 480), (240, 480), (240, 340))}'
        f'{sig((920, 310), (920, 480), (910, 480), (910, 490))}'
        f'{sig((920, 270), (920, 60), (1060, 60), (1060, 130))}'
        f'</svg>'
    )
    return svg.replace("\n", "")


def dynamic_to_r(sim: dynamic.SeparatorDynamicSimulator) -> dict:
    """Adapt dynamic simulation state into steady-state dictionary for pid_svg rendering."""
    st = sim.state
    c = sim.case_data
    h_m = st.level_m
    a_l = model.a_liquid(h_m)
    a_g = model.A_TOTAL - a_l
    p_bar_a = st.pressure_bar_a
    p_barg = st.pressure_barg
    q_liq = st.out_liquid_m3_h
    q_gas = (st.out_gas_header_kg_h + st.out_gas_flare_kg_h) / max(c["gas_rho"], 1.0)
    
    alarms = model._alarm_states(st.level_mm, p_barg)
    if not st.uzv051_open or not st.uzv052_open:
        for a in alarms:
            if "PZT-001" in a["tag"] or "LAHH" in a["condition"]:
                a["state"] = "TRIP"
    if not st.uzv002_open:
        for a in alarms:
            if "LALL" in a["condition"]:
                a["state"] = "TRIP"

    return {
        "case": sim.case_name,
        "h2s_ppm": model.H2S_DEFAULT_PPM,
        "T_C": st.temp_c,
        "p_bar_a": p_bar_a,
        "p_barg": p_barg,
        "in_envelope": model.ENVELOPE_BAR_A[0] <= p_bar_a <= model.ENVELOPE_BAR_A[1],
        "flows_m3_h": {
            "gas": q_gas,
            "oil": q_liq * (1.0 - st.water_cut_vol),
            "water": q_liq * st.water_cut_vol,
            "liquid": q_liq
        },
        "flows_kg_h": {
            "gas": st.out_gas_header_kg_h + st.out_gas_flare_kg_h,
            "oil": q_liq * (1.0 - st.water_cut_vol) * c["oil_rho"],
            "water": q_liq * st.water_cut_vol * c["water_rho"],
            "liquid": q_liq * ((1.0 - st.water_cut_vol) * c["oil_rho"] + st.water_cut_vol * c["water_rho"])
        },
        "vol_pct": {
            "gas": q_gas / max(q_gas + q_liq, 1e-6) * 100.0,
            "oil": (q_liq * (1.0 - st.water_cut_vol)) / max(q_gas + q_liq, 1e-6) * 100.0,
            "water": (q_liq * st.water_cut_vol) / max(q_gas + q_liq, 1e-6) * 100.0,
            "water_of_liquid": st.water_cut_vol * 100.0,
        },
        "level_mm": st.level_mm,
        "a_liquid_m2": a_l,
        "a_gas_m2": a_g,
        "v_liquid_m3": st.liquid_holdup_m3,
        "v_liquid_geometric_m3": st.liquid_geometric_m3,
        "t_res_min": (st.liquid_holdup_m3 / (st.feed_liquid_m3_h / 3600.0) / 60.0) if st.feed_liquid_m3_h > 0 else 999.0,
        "t_res_geometric_min": (st.liquid_geometric_m3 / (st.feed_liquid_m3_h / 3600.0) / 60.0) if st.feed_liquid_m3_h > 0 else 999.0,
        "v_gas_ms": st.gas_velocity_m_s,
        "v_nozzle_ms": (q_gas / 3600.0) / model.A_GAS_NOZZLE,
        "gas_actual_m3_h": q_gas,
        "rho_gas_actual": c["gas_rho"] * (p_bar_a / c["P_bar_a"]),
        "v_max_ms": st.gas_v_max_m_s,
        "v_max_bulk_ms": st.gas_v_max_m_s,
        "k_actual_ms": (q_gas / 3600.0 / a_g) / math.sqrt(max((c["oil_rho"] - c["gas_rho"]) / c["gas_rho"], 1.0)),
        "k_ok": not st.demister_flooded,
        "dp_mbar": c["dp_total_mbar"],
        "alarms": alarms,
    }


def readout(label: str, val: str, color: str = "#d7dde5") -> str:
    return (f'<div style="background:#21262d;border:1px solid #30363d;border-radius:6px;'
            f'padding:6px 12px;min-width:110px;text-align:center;">'
            f'<div style="font-size:11px;color:#8b949e;">{label}</div>'
            f'<div style="font-size:15px;font-weight:bold;color:{color};">{val}</div>'
            f'</div>')


def render_educational_cards(is_arabic: bool = False):
    """Render interactive dual-language concept explainer cards."""
    if is_arabic:
        with st.expander("📚 المفاهيم الهندسية والمعادلات الفيزيائية (شرح تفصيلي)"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                ### 1. معامل تدفق الصمام ($C_v$)
                - **التعريف**: كمية المياه (بالجالون الأمريكي) عند 60°F التي تمر عبر الصمام خلال دقيقة واحدة بفارق ضغط 1 psi.
                - **صمام السائل `FV-001`**: صمام كروي 14 بوصة (Koso S1200) بسعة اسمية $C_v = 2140$ وخاصية فتح خطية (Linear).
                - **المعادلة**: $Q = \\frac{C_v}{4.403} \\sqrt{\\frac{\\Delta P}{SG}}$
                - **أخطار التصميم**: إذا كان $C_v$ صغيراً جداً، يختنق التدفق ويرتفع المنسوب مسبباً إغلاق المنشأة (LAHH). وإذا كان كبيراً جداً، فإن فتحة 2% تحدث صدمة تدفق هيدروليكية حادة.
                """)
                st.markdown("""
                ### 2. سرعة سودرز-براون القصوى ($v_{\\max}$)
                - **التعريف**: أقصى سرعة سطحية مسموح بها للغاز قبل أن تتغلب قوى السحب الهوائية على الجاذبية وتبدأ في حمل قطرات النفط رذاذاً.
                - **المعادلة**: $v_{\\max} = K_{SB} \\sqrt{\\frac{\\rho_L - \\rho_G}{\\rho_G}}$ (حيث $K_{SB} = 0.150\\text{ m/s}$).
                - **طوفان مانع الرذاذ (Demister Flooding)**: إذا تجاوزت سرعة الغاز $v_{\\max}$، يفقد مانع الرذاذ (KnitMesh) كفاءته ويتطاير السائل نحو الضواغط ومحطة الغاز.
                """)
            with c2:
                st.markdown("""
                ### 3. التحكم التتابعي المزدوج (Cascade Control $LICA \\to FIC$)
                - **التعريف**: متحكم المنسوب الرئيسي لا يغير فتحة الصمام مباشرة، بل يرسل قيمة تدفق مطلوبة لمتحكم التدفق التابع.
                - **السقف الحرج 100 kBOPD**: تدفق السائل مقيد بدقة عند $794.94\\text{ m}^3/\\text{h}$ لحماية السخانات والفواصل اللاحقة من الاختناق.
                """)
                st.markdown("""
                ### 4. قفل الحماية رقم 26 (منع تسرب الغاز عالي الضغط Note 26)
                - **آلية العمل**: عند هبوط منسوب السائل إلى $770\\text{ mm}$ (LALL)، يغلق صمام عزل السائل `UZV-002`، وفور تأكيد إغلاقه عبر المفتاح `ZSC-002`، يغلق تلقائياً صمام عزل الغاز `UZV-003`.
                - **الغاية**: منع اندفاع الغاز عالي الضغط (6 بار) إلى الفاصلة الثانية منخفضة الضغط (2 بار) مما يمنع انفجارها.
                """)
    else:
        with st.expander("📚 Engineering Concept Explainers & Mathematical Basis"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                ### 1. Valve Flow Coefficient ($C_v$)
                - **Definition**: Hydraulic capacity to pass 1 US gal/min of 60°F water with 1 psi pressure drop.
                - **Liquid Valve `FV-001`**: 14" Koso S1200 Globe Valve, rated $C_v = 2140$, **Linear Trim**.
                - **Formula**: $Q = \\frac{C_v}{4.403} \\sqrt{\\frac{\\Delta P}{SG}}$
                - **Sizing Hazard**: Undersized $C_v$ chokes liquid outflow and causes LAHH trip. Oversized $C_v$ causes erratic control hunting.
                """)
                st.markdown("""
                ### 2. Souders-Brown Superficial Velocity ($v_{\\max}$)
                - **Definition**: Maximum allowable vapor velocity before aerodynamic drag re-entrains liquid droplets into the gas stream.
                - **Formula**: $v_{\\max} = K_{SB} \\sqrt{\\frac{\\rho_L - \\rho_G}{\\rho_G}}$ with target $K_{SB} = 0.150\\text{ m/s}$.
                - **Demister Flooding**: If gas surge exceeds $v_{\\max}$, the Mellachevron vanes and Knit Mesh demister flood, blowing mist downstream.
                """)
            with c2:
                st.markdown("""
                ### 3. Cascade Level-to-Flow ($LICA-002 \\to FIC-001$)
                - **Architecture**: Level master PI outputs flow setpoint to flow slave PI, filtering out downstream backpressure noise.
                - **100 kBOPD Cap**: Setpoint is clamped at $794.94\\text{ m}^3/\\text{h}$ total liquid to protect downstream heating trains.
                """)
                st.markdown("""
                ### 4. P&ID Note 26 Interlock (Gas Blow-by Prevention)
                - **Mechanism**: Level low-low ($770\\text{ mm}$) trips liquid SDV `UZV-002`. Position switch `ZSC-002` close feedback trips gas SDV `UZV-003`.
                - **Safety Rationale**: Prevents 6 barg gas from blowing into the 2 barg 2nd-stage vessel.
                """)


def main() -> None:
    # --- State Initialization ---
    if "sim" not in st.session_state:
        st.session_state.sim = dynamic.SeparatorDynamicSimulator("Case 1 x1.4")
        st.session_state.sim_running = False
        st.session_state.sim_speed = 1.0
        st.session_state.sim_scenario = "SCN-01"
        st.session_state.app_mode = "OTS" # "OTS" or "STEADY"
        st.session_state.view_mode = "3D" # "3D" or "2D"
        st.session_state.is_arabic = False
        st.session_state.explain = False

    sim: dynamic.SeparatorDynamicSimulator = st.session_state.sim

    # --- Top Navigation Bar ---
    top_col1, top_col2, top_col3, top_col4 = st.columns([4.5, 2.8, 1.4, 1.1])
    
    with top_col1:
        st.markdown(
            '<h2 style="margin:0;padding:0;color:#58a6ff;">'
            'CP2-V-71101 — 1st Stage Production Separator OTS'
            '</h2>'
            '<span style="color:#8b949e;font-size:12px;">'
            'CPF-2 Majnoon Oil Field • Basrah Oil Company (BOC)'
            '</span>',
            unsafe_allow_html=True
        )

    with top_col2:
        mode = st.radio(
            "Mode",
            ["🎮 Dynamic OTS Simulator", "📋 Steady-State Model"],
            index=0 if st.session_state.app_mode == "OTS" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.app_mode = "OTS" if "Dynamic" in mode else "STEADY"

    with top_col3:
        lang = st.selectbox("Language / اللغة", ["English", "العربية"], index=1 if st.session_state.is_arabic else 0)
        st.session_state.is_arabic = (lang == "العربية")

    with top_col4:
        if st.button("💡 Explain (E)"):
            st.session_state.explain = not st.session_state.explain
            st.rerun()

    # --- Mode Branching ---
    if st.session_state.app_mode == "OTS":
        # =========================================================================
        # 🎮 DYNAMIC OPERATOR TRAINING SIMULATOR (ZERO-FLICKER 60 FPS DIGITAL TWIN)
        # =========================================================================
        
        # Render high-performance 60 FPS self-contained Digital Twin & DCS console
        html_ots = twin3d.render_ots_app_html(
            is_arabic=st.session_state.is_arabic,
            initial_scenario=st.session_state.sim_scenario
        )
        st.components.v1.html(html_ots, height=840, scrolling=False)

        # Bilingual Educational Concept Explanations
        render_educational_cards(st.session_state.is_arabic)

        # Python Engine Diagnostics & Verification Expander
        with st.expander("🔧 Python Dynamic Engine Verification & Batch Diagnostics"):
            st.markdown("""
            **Dynamic Engine Specifications (`dynamic.py`)**:
            - Two-Rate ODE scheme: Liquid level $dh/dt$ at $dt=0.2\\text{ s}$; Compressible gas sub-stepping at $dt_g=0.02\\text{ s}$.
            - 1-Hour mass & volume conservation drift: $< 0.0002\\%$ ($18,000$ steps).
            - Valve Actuators: Koso S1200 Linear ($C_v=2140$) and Modified Equal-% ($C_v=3990$) with $10\\text{ s}$ slew rate limit.
            - Interlocks: P&ID Note 26 Gas Blow-by Interlock (`LALL 770 mm -> UZV-002 close -> UZV-003 close`).
            """)
            if st.button("▶️ Run Automated Test Battery (11 Tests)"):
                with st.spinner("Executing dynamic test suite..."):
                    import subprocess
                    res = subprocess.run(["python", "simulator/stage1/test_dynamic.py"], capture_output=True, text=True)
                    if res.returncode == 0:
                        st.success("All 11 Dynamic & Steady-State Tests Passed 100%!")
                        st.code(res.stdout)
                    else:
                        st.error("Test failures detected:")
                        st.code(res.stderr)

    else:
        # =========================================================================
        # 📋 STEADY-STATE DESIGN MODEL (PHASE 1 - 100% UNMODIFIED BASELINE)
        # =========================================================================
        t1, t2, t3, t4, t5, t6 = st.columns([2.2, 1.4, 1.2, 1.2, 1.2, 1.8], gap="small")
        case = t2.selectbox("Case", list(model.CASES))
        base = model.CASES[case]
        q_liq = t3.number_input("Feed liq (m3/h)", min_value=0.0,
                                value=base["liq_m3_h"], step=10.0, key=f"q_{case}")
        p_barg = t4.number_input("Feed P (barg)", min_value=0.0,
                                 value=round(base["P_bar_a"] - 1.013, 3),
                                 step=0.1, format="%.3f", key=f"p_{case}")
        t_c = t5.number_input("Feed T (degC)", min_value=0.0,
                              value=base["T_C"], step=0.5, key=f"t_{case}")
        h2s = t6.number_input("H2S feed (ppm mol)", min_value=0, max_value=100000,
                              value=model.H2S_DEFAULT_PPM, step=100, key=f"h2s_{case}")

        f = q_liq / base["liq_m3_h"] if base["liq_m3_h"] else 1.0
        ovr = {"liq_m3_h": q_liq,
               "oil_m3_h": base["oil_m3_h"] * f,
               "water_m3_h": base["water_m3_h"] * f,
               "P_bar_a": p_barg + 1.013, "T_C": t_c}
        r = model.simulate(case, h2s_ppm=h2s, overrides=ovr)
        fv = fv_open(r)

        st.markdown(
            '<div style="display:flex;gap:10px;margin-top:8px;margin-bottom:12px;">'
            + readout("P (barg)", f'{r["p_barg"]:.2f}',
                      "#34c06c" if r["in_envelope"] else "#ff5252")
            + readout("Feed (m3/h)", f'{r["flows_m3_h"]["gas"] + r["flows_m3_h"]["liquid"]:,.0f}')
            + readout("Phase split G/O/W",
                      f'{r["vol_pct"]["gas"]:.1f}/{r["vol_pct"]["oil"]:.1f}/{r["vol_pct"]["water"]:.1f}')
            + readout("t_res (min)", f'{r["t_res_min"]:.2f}')
            + readout("V liquid (m3)", f'{r["v_liquid_m3"]:.2f}')
            + readout("v gas nozzle (m/s)", f'{r["v_nozzle_ms"]:.1f}')
            + readout("FV-001", "OPEN" if fv else "CLOSED",
                      "#34c06c" if fv else "#ff5252")
            + readout("H2S (ppm mol)", f'{r["h2s_ppm"]:,.0f}', "#ffb020")
            + "</div>", unsafe_allow_html=True)

        left, right = st.columns([3, 1.15], gap="large")
        left.markdown(pid_svg(r, st.session_state.explain), unsafe_allow_html=True)

        right.markdown("**Tags - click for equation, inputs, result, source**")
        sel = st.session_state.get("tag")
        for tag in TAG_INFO:
            if right.button(tag, key=f"btn_{tag}", width="stretch"):
                st.session_state.tag = None if sel == tag else tag
                st.rerun()
        if sel:
            info = TAG_INFO[sel]
            right.markdown(f"### {sel}")
            right.markdown(f"**Plain:** {PLAIN.get(sel, info['eq'])}")
            right.markdown(f"**Equation:** {info['eq']}")
            right.markdown(f"**Inputs:** {info['inputs']}")
            right.markdown(f"**Result:** {info['result'](r)}")
            right.markdown(f"**Source:** {model.TRACE.get(info['trace'], 'Vendor Drawings')}")


if __name__ == "__main__":
    main()
