"""Operating World Integration Layer for CP2-V-71101 Digital Twin (Phase 4).

Coordinates the single Python simulation authority with 2D DCS and 3D Field views:
1. Authority: simulator/stage1/dynamic.py (SeparatorDynamicSimulator).
2. State Snapshot: get_canonical_state() distributed to 2D and 3D adapters.
3. Command Dispatch: strictly validated operator commands (apply_command).
4. Synchronized Tag Navigation: 2D <-> 3D canonical tag cross-linking.
5. In-Memory Educational Trends: rolling buffer of key process variables.
6. Optional Lightweight Bridge Server: local HTTP/WebSocket endpoint for browser views.
"""

import json
import logging
import math
import os
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Tuple

import dynamic
import sandbox

logger = logging.getLogger("OperatingWorld")

REGISTRY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "tag_registry.json")
)


class OperatingWorldBridge:
    """Central coordinator ensuring One Process, One State, Two Synchronized Views."""

    def __init__(self, sim: Optional[dynamic.SeparatorDynamicSimulator] = None, max_trend_points: int = 300):
        self.sim = sim or dynamic.SeparatorDynamicSimulator("Case 1 x1.4")
        self.max_trend_points = max_trend_points
        self._trend_buffer: deque = deque(maxlen=max_trend_points)
        self._lock = threading.Lock()
        self._tag_registry: Dict[str, Any] = {}
        self._load_registry()
        
        # Educational Sandbox Session (Phase 5)
        self.sandbox = sandbox.SandboxSession(self.sim)
        
        # Record initial point in trend
        self._record_trend_point(self.sim.get_canonical_state())

    def _load_registry(self) -> None:
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                self._tag_registry = json.load(f)
        else:
            self._tag_registry = {}

    def get_canonical_state(self) -> Dict[str, Any]:
        """Obtain authoritative canonical runtime state from dynamic.py."""
        with self._lock:
            state = self.sim.get_canonical_state()
            state["trend_summary"] = {
                "points_recorded": len(self._trend_buffer),
                "max_points": self.max_trend_points,
            }
            return state

    def get_2d_view_state(self) -> Dict[str, Any]:
        """Format canonical state specifically for the 2D DCS P&ID view adapter."""
        with self._lock:
            cs = self.sim.get_canonical_state()
            proc = cs["process"]
            valves = cs["valves"]
            ctrl = cs["controllers"]
            alarms = cs["alarms"]

            # Calculate fluid geometry in SVG coordinates
            total_h_px = min(180.0, (proc["level_m"] / dynamic.D) * 180.0)
            water_cut = self.sim.state.water_cut_vol
            water_h_px = total_h_px * water_cut
            oil_h_px = total_h_px - water_h_px

            return {
                "asset_id": "CP2-V-71101",
                "time_s": cs["time_s"],
                "level_mm": proc["level_mm"],
                "level_m": proc["level_m"],
                "pressure_barg": proc["pressure_barg"],
                "feed_liquid_m3_h": proc["feed_liquid_m3_h"],
                "out_liquid_m3_h": proc["out_liquid_m3_h"],
                "out_gas_header_kg_h": proc["out_gas_header_kg_h"],
                "out_gas_flare_kg_h": proc["out_gas_flare_kg_h"],
                "svg_geometry": {
                    "vessel_x": 240,
                    "vessel_y": 160,
                    "vessel_w": 520,
                    "vessel_h": 180,
                    "water_layer": {
                        "x": 245,
                        "y": round(340.0 - water_h_px, 1),
                        "width": 510,
                        "height": round(max(0.0, water_h_px), 1),
                        "fill": "#0284c7"
                    },
                    "oil_layer": {
                        "x": 245,
                        "y": round(340.0 - total_h_px, 1),
                        "width": 510,
                        "height": round(max(0.0, oil_h_px), 1),
                        "fill": "#d97706"
                    }
                },
                "instruments": {
                    "PT-003": {"pv": proc["pressure_barg"], "units": "barg", "status": "OK"},
                    "LT-002": {"pv": proc["level_mm"], "units": "mm", "status": "OK"},
                    "FT-001": {"pv": proc["out_liquid_m3_h"], "units": "m3/h", "status": "OK"},
                    "FT-002": {"pv": proc["out_gas_header_kg_h"], "units": "kg/h", "status": "OK", "note": "merged in piping"}
                },
                "valves": {
                    "FV-001": {"position_pct": valves["fv001_pct"], "mode": "MANUAL" if self.sim.valv_fv001.manual else "AUTO", "state": "OPEN" if valves["fv001_pct"] > 2.0 else "CLOSED"},
                    "PV-003B": {"position_pct": valves["pv003b_pct"], "mode": "MANUAL" if self.sim.valv_pv003b.manual else "AUTO", "state": "OPEN" if valves["pv003b_pct"] > 2.0 else "CLOSED"},
                    "PV-003A": {"position_pct": valves["pv003a_pct"], "mode": "MANUAL" if self.sim.valv_pv003a.manual else "AUTO", "state": "OPEN" if valves["pv003a_pct"] > 1.0 else "CLOSED"},
                    "UZV-002": {"open": valves["uzv002_open"], "state": "OPEN" if valves["uzv002_open"] else "CLOSED"},
                    "UZV-003": {"open": valves["uzv003_open"], "state": "OPEN" if valves["uzv003_open"] else "CLOSED"},
                    "UZV-051": {"open": valves["uzv051_open"], "state": "OPEN" if valves["uzv051_open"] else "CLOSED"},
                    "UZV-052": {"open": valves["uzv052_open"], "state": "OPEN" if valves["uzv052_open"] else "CLOSED"}
                },
                "controllers": {
                    "LICA-002": {"pv": proc["level_mm"], "sp": ctrl["lica002"]["sp"], "op": ctrl["lica002"]["op"]},
                    "PIC-003": {"pv": proc["pressure_barg"], "sp": ctrl["pic003"]["sp"], "op": ctrl["pic003"]["op"]},
                    "FIC-001": {"pv": proc["out_liquid_m3_h"], "sp": ctrl["fic001"]["sp"], "op": ctrl["fic001"]["op"], "clamp_override": ctrl["clamp_override"]}
                },
                "alarm_status": alarms["status"],
                "active_alarms": alarms["active_alarms"],
                "tripped_causes": alarms["tripped_causes"]
            }

    def get_3d_view_state(self) -> Dict[str, Any]:
        """Format canonical state specifically for the 3D Field Three.js view adapter."""
        with self._lock:
            cs = self.sim.get_canonical_state()
            proc = cs["process"]
            valves = cs["valves"]
            alarms = cs["alarms"]

            # Compute physical 3D clipping planes and chord widths
            r_in = dynamic.D / 2.0 # 2.1 m
            vessel_r = dynamic.D / 2.0
            h_liq = max(0.005, min(dynamic.D - 0.01, proc["level_m"]))
            water_cut = self.sim.state.water_cut_vol
            h_water = max(0.002, h_liq * water_cut)
            h_oil = max(0.002, h_liq - h_water)

            y_water_level = -vessel_r + h_water
            y_oil_level = -vessel_r + h_liq

            w_water = 2.0 * math.sqrt(max(0.01, r_in * r_in - y_water_level * y_water_level))
            w_oil = 2.0 * math.sqrt(max(0.01, r_in * r_in - y_oil_level * y_oil_level))

            # Valve physical stem / disc positions
            fv001_stem_y = 0.45 + (valves["fv001_pct"] / 100.0) * 0.144
            pv003b_disc_rot_y = (valves["pv003b_pct"] / 100.0) * (math.pi / 2.0)

            return {
                "asset_id": "CP2-V-71101",
                "time_s": cs["time_s"],
                "level_m": proc["level_m"],
                "level_mm": proc["level_mm"],
                "pressure_barg": proc["pressure_barg"],
                "three_geometry": {
                    "y_water_level": round(y_water_level, 4),
                    "y_oil_level": round(y_oil_level, 4),
                    "w_water_chord": round(w_water, 4),
                    "w_oil_chord": round(w_oil, 4),
                    "water_visible": (h_water > 0.02),
                    "oil_visible": (h_oil > 0.02),
                    "fv001_stem_y": round(fv001_stem_y, 4),
                    "pv003b_disc_rot_y": round(pv003b_disc_rot_y, 4)
                },
                "nodes": {
                    "vessel_shell": {"status": "NORMAL"},
                    "tag:LT-002": {"pv": proc["level_mm"], "status": "NORMAL" if alarms["status"] == "NORMAL" else "ALARM"},
                    "tag:PT-003": {"pv": proc["pressure_barg"], "status": "NORMAL" if alarms["status"] == "NORMAL" else "ALARM"},
                    "tag:UZV-002": {"state": "OPEN" if valves["uzv002_open"] else "CLOSED"},
                    "tag:UZV-003": {"state": "OPEN" if valves["uzv003_open"] else "CLOSED"},
                    "tag:FV-001": {"pct": valves["fv001_pct"], "state": "OPEN" if valves["fv001_pct"] > 1.0 else "CLOSED"},
                    "tag:PV-003B": {"pct": valves["pv003b_pct"], "state": "OPEN" if valves["pv003b_pct"] > 2.0 else "CLOSED"},
                    "tag:PV-003A": {"pct": valves["pv003a_pct"], "state": "OPEN" if valves["pv003a_pct"] > 1.0 else "CLOSED"}
                },
                "merged_limitations": {
                    "FT-002": {"addressable_node": False, "reason": "Not modeled as a discrete 3D solid in source CAD/IFC (in-line meter on line CP2-16\"-P711001-3C6M-PP)"},
                    "UZV-051": {"addressable_node": False, "reason": "Located upstream outside separator skid bounds"},
                    "UZV-052": {"addressable_node": False, "reason": "Located upstream outside separator skid bounds"},
                    "PSV-001A": {"addressable_node": False, "reason": "Located downstream on HP flare header skid per authentic plant model, not mounted on vessel platform"},
                    "PSV-001B": {"addressable_node": False, "reason": "Located downstream on HP flare header skid per authentic plant model, not mounted on vessel platform"}
                },
                "alarm_status": alarms["status"]
            }

    def dispatch_command(self, cmd: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate and dispatch an operator command strictly through the canonical boundary."""
        params = params or {}
        cmd_u = str(cmd).strip().upper()

        # Strict validation of command existence
        allowed_commands = {
            "RESET_ESD",
            "SET_CLAMP_OVERRIDE",
            "SET_SETPOINT",
            "SET_VALVE_MANUAL",
            "SET_FEED_DISTURBANCE",
            "STEP"
        }
        if cmd_u not in allowed_commands:
            return {
                "success": False,
                "error": f"Rejected unknown command: '{cmd}'. Supported: {sorted(list(allowed_commands))}",
                "canonical_state": self.get_canonical_state()
            }

        # Parameter validation per command type
        if cmd_u == "SET_SETPOINT":
            loop = str(params.get("loop", "")).upper()
            if loop not in ("LICA-002", "LEVEL", "PIC-003", "PRESSURE", "FIC-001", "FLOW"):
                return {
                    "success": False,
                    "error": f"Rejected invalid controller loop: '{loop}'. Supported: LICA-002, PIC-003, FIC-001",
                    "canonical_state": self.get_canonical_state()
                }
            try:
                val = float(params.get("value", 0.0))
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "error": "Setpoint value must be numeric",
                    "canonical_state": self.get_canonical_state()
                }
            # Range sanity bounds
            if loop in ("LICA-002", "LEVEL") and not (100.0 <= val <= 4000.0):
                return {
                    "success": False,
                    "error": f"Level setpoint {val} mm out of physical vessel bounds [100, 4000] mm",
                    "canonical_state": self.get_canonical_state()
                }
            if loop in ("PIC-003", "PRESSURE") and not (0.5 <= val <= 22.0):
                return {
                    "success": False,
                    "error": f"Pressure setpoint {val} barg out of physical vessel bounds [0.5, 22.0] barg",
                    "canonical_state": self.get_canonical_state()
                }
            if loop in ("FIC-001", "FLOW") and not (0.0 <= val <= 2000.0):
                return {
                    "success": False,
                    "error": f"Flow setpoint {val} m3/h out of bounds [0.0, 2000.0] m3/h",
                    "canonical_state": self.get_canonical_state()
                }

        elif cmd_u == "SET_VALVE_MANUAL":
            valve = str(params.get("valve", "")).upper()
            supported_valves = ("FV-001", "FV001", "PV-003B", "PV003B", "PV-003A", "PV003A")
            if not any(v in valve for v in supported_valves):
                return {
                    "success": False,
                    "error": f"Rejected invalid valve: '{valve}'. Controllable: FV-001, PV-003B, PV-003A",
                    "canonical_state": self.get_canonical_state()
                }
            target = params.get("target")
            if target is not None:
                try:
                    t_val = float(target)
                    if not (0.0 <= t_val <= 1.0):
                        return {
                            "success": False,
                            "error": f"Manual valve target {t_val} out of fractional span [0.0, 1.0]",
                            "canonical_state": self.get_canonical_state()
                        }
                except (ValueError, TypeError):
                    return {
                        "success": False,
                        "error": "Valve target must be numeric between 0.0 and 1.0",
                        "canonical_state": self.get_canonical_state()
                    }

        # Dispatch to single process authority
        with self._lock:
            ok = self.sim.apply_command(cmd_u, params)
            new_state = self.sim.get_canonical_state()
            if ok:
                self._record_trend_point(new_state)
            self.sandbox.record_operator_action(cmd_u, params, ok)
            return {
                "success": ok,
                "command": cmd_u,
                "canonical_state": new_state,
                "sandbox_state": self.sandbox.get_session_state()
            }

    def step(self, dt: float = 0.2, n_steps: int = 1) -> Dict[str, Any]:
        """Advance authoritative Python process simulation by n_steps of dt."""
        with self._lock:
            for _ in range(max(1, n_steps)):
                self.sim.step(dt=dt)
                self.sandbox.step(dt=dt)
            new_state = self.sim.get_canonical_state()
            self._record_trend_point(new_state)
            return new_state

    def reset(self, initial_case: str = "Case 1 x1.4") -> Dict[str, Any]:
        """Reset simulation authority back to canonical baseline."""
        with self._lock:
            self.sim = dynamic.SeparatorDynamicSimulator(initial_case)
            self._trend_buffer.clear()
            self.sandbox = sandbox.SandboxSession(self.sim)
            init_state = self.sim.get_canonical_state()
            self._record_trend_point(init_state)
            return init_state

    def get_sandbox_state(self) -> Dict[str, Any]:
        """Obtain state of active educational scenario and timeline."""
        with self._lock:
            return self.sandbox.get_session_state()

    def start_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Initialize and start an educational scenario."""
        with self._lock:
            loaded = self.sandbox.load_scenario(scenario_id)
            if loaded:
                self.sandbox.start()
            return {
                "success": loaded,
                "sandbox_state": self.sandbox.get_session_state(),
                "canonical_state": self.sim.get_canonical_state()
            }

    def reset_scenario(self) -> Dict[str, Any]:
        """Cleanly reset current scenario and simulation state."""
        with self._lock:
            self.sandbox.reset()
            return {
                "success": True,
                "sandbox_state": self.sandbox.get_session_state(),
                "canonical_state": self.sim.get_canonical_state()
            }

    def abort_scenario(self) -> Dict[str, Any]:
        """Abort active scenario."""
        with self._lock:
            self.sandbox.abort()
            return {
                "success": True,
                "sandbox_state": self.sandbox.get_session_state(),
                "canonical_state": self.sim.get_canonical_state()
            }

    def _record_trend_point(self, cs: Dict[str, Any]) -> None:
        proc = cs.get("process", {})
        valves = cs.get("valves", {})
        ctrl = cs.get("controllers", {})
        alarms = cs.get("alarms", {})
        self._trend_buffer.append({
            "t": cs.get("time_s", 0.0),
            "level_mm": proc.get("level_mm", 0.0),
            "level_sp_mm": ctrl.get("lica002", {}).get("sp", 0.0),
            "pressure_barg": proc.get("pressure_barg", 0.0),
            "pressure_sp_barg": ctrl.get("pic003", {}).get("sp", 0.0),
            "out_liq_m3_h": proc.get("out_liquid_m3_h", 0.0),
            "out_gas_kg_h": proc.get("out_gas_header_kg_h", 0.0),
            "fv001_pct": valves.get("fv001_pct", 0.0),
            "pv003b_pct": valves.get("pv003b_pct", 0.0),
            "alarm_status": alarms.get("status", "NORMAL")
        })

    def get_trends(self, limit: int = 120) -> List[Dict[str, Any]]:
        """Return the most recent rolling trend history."""
        with self._lock:
            buf = list(self._trend_buffer)
            return buf[-limit:] if limit > 0 else buf

    def trace_tag_identity(self, tag: str) -> Dict[str, Any]:
        """Trace a canonical tag across P&ID, 3D GLB, and simulation authority."""
        tag_u = tag.strip().upper()
        reg_item = None
        category = None
        for cat in ("equipment", "nozzles", "valves", "instruments", "controllers"):
            if tag_u in self._tag_registry.get(cat, {}):
                reg_item = self._tag_registry[cat][tag_u]
                category = cat
                break
        
        if not reg_item:
            return {
                "tag": tag_u,
                "found": False,
                "error": f"Tag '{tag_u}' not found in canonical registry."
            }

        cs = self.get_canonical_state()
        sim_val = None
        # Link to live simulation state
        if tag_u == "LT-002" or tag_u == "LICA-002":
            sim_val = {"level_mm": cs["process"]["level_mm"], "sp_mm": cs["controllers"]["lica002"]["sp"]}
        elif tag_u == "PT-003" or tag_u == "PIC-003":
            sim_val = {"pressure_barg": cs["process"]["pressure_barg"], "sp_barg": cs["controllers"]["pic003"]["sp"]}
        elif tag_u == "FT-001" or tag_u == "FIC-001":
            sim_val = {"flow_m3_h": cs["process"]["out_liquid_m3_h"], "op_pct": cs["controllers"]["fic001"]["op"]}
        elif tag_u == "FV-001":
            sim_val = {"position_pct": cs["valves"]["fv001_pct"]}
        elif tag_u == "PV-003B":
            sim_val = {"position_pct": cs["valves"]["pv003b_pct"]}
        elif tag_u == "PV-003A":
            sim_val = {"position_pct": cs["valves"]["pv003a_pct"]}
        elif tag_u == "UZV-002":
            sim_val = {"open": cs["valves"]["uzv002_open"]}
        elif tag_u == "UZV-003":
            sim_val = {"open": cs["valves"]["uzv003_open"]}

        return {
            "tag": tag_u,
            "found": True,
            "category": category,
            "registry_entry": reg_item,
            "physical_3d_presence": reg_item.get("physical_3d_presence"),
            "addressable_3d_node": reg_item.get("addressable_3d_node"),
            "gltf_ref": reg_item.get("gltf_ref"),
            "simulation_ref": reg_item.get("simulation_ref"),
            "live_process_value": sim_val
        }


# --- Lightweight Local Bridge Server ---

class BridgeRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler for real-time canonical state and command dispatch."""

    bridge: Optional[OperatingWorldBridge] = None

    def log_message(self, format, *args):
        # Suppress routine request logging to avoid terminal clutter
        pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/state"):
            state = self.bridge.get_canonical_state() if self.bridge else {}
            body = json.dumps(state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/trends"):
            trends = self.bridge.get_trends() if self.bridge else []
            body = json.dumps(trends).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/2d"):
            v2d = self.bridge.get_2d_view_state() if self.bridge else {}
            body = json.dumps(v2d).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/3d"):
            v3d = self.bridge.get_3d_view_state() if self.bridge else {}
            body = json.dumps(v3d).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/sandbox/state"):
            s_state = self.bridge.get_sandbox_state() if self.bridge else {}
            body = json.dumps(s_state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/sandbox/scenarios"):
            scenarios = [s.to_dict() for s in sandbox.SCENARIO_CATALOG.values()]
            body = json.dumps(scenarios).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/sandbox/sessions"):
            sess_list = sandbox.list_saved_sessions()
            body = json.dumps(sess_list).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/sandbox/session"):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            q_params = urllib.parse.parse_qs(parsed.query)
            session_id = q_params.get("id", [""])[0]
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            if not session_id and len(parts) >= 4:
                session_id = parts[3]

            if not session_id:
                body = json.dumps({"error": "Missing session id"}).encode("utf-8")
                self.send_response(400)
            else:
                try:
                    sess_data = sandbox.load_saved_session(session_id)
                    body = json.dumps(sess_data).encode("utf-8")
                    self.send_response(200)
                except FileNotFoundError:
                    body = json.dumps({"error": f"Session '{session_id}' not found"}).encode("utf-8")
                    self.send_response(404)
                except ValueError as ve:
                    body = json.dumps({"error": str(ve)}).encode("utf-8")
                    self.send_response(422)

            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/" or self.path.startswith("/index.html"):
            import twin3d
            html_content = twin3d.render_ots_app_html()
            body = html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/assets/"):
            asset_rel = self.path[len("/assets/"):]
            asset_path = os.path.join(os.path.dirname(__file__), "assets", asset_rel)
            if os.path.exists(asset_path) and os.path.isfile(asset_path):
                with open(asset_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                content_type = "application/octet-stream"
                if asset_path.endswith(".glb"):
                    content_type = "model/gltf-binary"
                elif asset_path.endswith(".js"):
                    content_type = "application/javascript"
                elif asset_path.endswith(".html"):
                    content_type = "text/html"
                self.send_header("Content-Type", content_type)
                self._send_cors_headers()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            req_json = json.loads(post_data.decode("utf-8"))
        except Exception:
            req_json = {}

        if self.path.startswith("/api/command"):
            cmd = req_json.get("command", "")
            params = req_json.get("params", {})
            res = self.bridge.dispatch_command(cmd, params) if self.bridge else {"success": False}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200 if res.get("success") else 400)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/step"):
            dt = float(req_json.get("dt", 0.2))
            n = int(req_json.get("n_steps", 1))
            res = self.bridge.step(dt=dt, n_steps=n) if self.bridge else {}
            body = json.dumps({"success": True, "state": res}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/reset"):
            case = req_json.get("case", "Case 1 x1.4")
            res = self.bridge.reset(case) if self.bridge else {}
            body = json.dumps({"success": True, "state": res}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/sandbox/start"):
            scenario_id = req_json.get("scenario_id", "SCN-01")
            res = self.bridge.start_scenario(scenario_id) if self.bridge else {"success": False}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200 if res.get("success") else 400)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/sandbox/reset"):
            res = self.bridge.reset_scenario() if self.bridge else {"success": False}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/sandbox/abort"):
            res = self.bridge.abort_scenario() if self.bridge else {"success": False}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()


_global_server: Optional[HTTPServer] = None
_global_bridge: Optional[OperatingWorldBridge] = None


def start_bridge_server(port: int = 8765, sim: Optional[dynamic.SeparatorDynamicSimulator] = None) -> Tuple[HTTPServer, OperatingWorldBridge]:
    """Launch the local bridge server in a background daemon thread."""
    global _global_server, _global_bridge
    if _global_server is not None:
        return _global_server, _global_bridge

    _global_bridge = OperatingWorldBridge(sim=sim)
    BridgeRequestHandler.bridge = _global_bridge
    
    # Try preferred port, or fall back to adjacent port if occupied
    server = None
    for p in range(port, port + 10):
        try:
            server = HTTPServer(("127.0.0.1", p), BridgeRequestHandler)
            break
        except OSError:
            continue

    if server is None:
        raise RuntimeError(f"Could not bind bridge server on any port in range {port}..{port+9}")

    _global_server = server
    t = threading.Thread(target=server.serve_forever, daemon=True, name="OTS-OperatingWorldBridge")
    t.start()
    return server, _global_bridge


def get_bridge() -> OperatingWorldBridge:
    """Obtain or initialize the active OperatingWorldBridge singleton."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = OperatingWorldBridge()
    return _global_bridge


if __name__ == "__main__":
    import webbrowser
    import sys

    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    server, bridge = start_bridge_server(port=port)
    actual_port = server.server_port
    url = f"http://127.0.0.1:{actual_port}/"

    print("=" * 80)
    print("  CP2-V-71101 DIGITAL TWIN - OPERATOR TRAINING SIMULATOR (OTS)")
    print("  One Operating World (Phase 4) + Educational Sandbox (Phase 5)")
    print("=" * 80)
    print(f"  [+] Process Authority: dynamic.py active")
    print(f"  [+] Unified Console (2D DCS + 3D Field View + Sandbox): {url}")
    print(f"  [+] Press Ctrl+C to terminate.")
    print("=" * 80)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Shutting down server gracefully...")
        server.shutdown()
        print("[+] Simulator terminated.")

