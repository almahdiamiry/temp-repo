"""Educational Sandbox Engine for CP2-V-71101 Stage 1 Separator Digital Twin.

Phase 5: Orchestrates learning, scenario lifecycles, real-time timeline event recording,
and post-mission debriefing around the canonical process simulation (dynamic.py).

Guiding Principles:
1. Single Authority: Sandbox orchestrates learning; it does NOT simulate physics.
2. Canonical Commands: User and scenario actions route strictly through dynamic.py.
3. Event Traceability: Timeline records real canonical events (alarms, trips, commands).
4. Zero Second Physics: All process outcomes emerge naturally from dynamic.py equations.
"""

from dataclasses import dataclass, field
import datetime
from enum import Enum
import json
import math
import os
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
import uuid

import dynamic

DEFAULT_SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")


class ScenarioState(str, Enum):
    READY = "READY"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class TimelineEvent:
    time_s: float
    category: str  # "SCENARIO", "COMMAND", "ALARM", "INTERLOCK", "PROCESS"
    message: str
    level: str = "INFO"  # "INFO", "WARNING", "DANGER", "SUCCESS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": round(self.time_s, 1),
            "category": self.category,
            "message": self.message,
            "level": self.level,
        }


@dataclass
class ScenarioDefinition:
    id: str
    tag: str
    title_en: str
    title_ar: str
    description_en: str
    description_ar: str
    learning_objective_en: str
    learning_objective_ar: str
    initial_condition_desc_en: str
    initial_condition_desc_ar: str
    allowed_commands: List[str]
    setup_func: Callable[[dynamic.SeparatorDynamicSimulator], None]
    evaluation_func: Callable[[dynamic.SeparatorDynamicSimulator, 'SandboxSession'], Tuple[ScenarioState, str, str]]
    debrief_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tag": self.tag,
            "title_en": self.title_en,
            "title_ar": self.title_ar,
            "description_en": self.description_en,
            "description_ar": self.description_ar,
            "learning_objective_en": self.learning_objective_en,
            "learning_objective_ar": self.learning_objective_ar,
            "initial_condition_desc_en": self.initial_condition_desc_en,
            "initial_condition_desc_ar": self.initial_condition_desc_ar,
            "allowed_commands": self.allowed_commands,
        }


# --- Scenario Setup & Evaluation Functions ---

def setup_scn01(sim: dynamic.SeparatorDynamicSimulator) -> None:
    """SCN-01: Normal 100 kBOPD Baseline."""
    sim._init_steady_baseline()
    sim.set_clamp_override(False)
    sim.valv_fv001.manual = False
    sim.valv_pv003b.manual = False
    sim.valv_pv003a.manual = False


def eval_scn01(sim: dynamic.SeparatorDynamicSimulator, session: 'SandboxSession') -> Tuple[ScenarioState, str, str]:
    st = sim.state
    # Maintain baseline level and pressure for 30 simulation seconds
    if session.elapsed_sim_s >= 30.0:
        if 1500.0 <= st.level_mm <= 1600.0 and 4.8 <= st.pressure_barg <= 5.2 and st.alarm_status == "NORMAL":
            return (ScenarioState.COMPLETED,
                    "Baseline stability verified: Separator operated cleanly for 30s with zero alarms.",
                    "تم التحقق من استقرار نقطة التشغيل الأساسية: عملت العازلة بكفاءة لمدة 30 ثانية بدون أي إنذارات.")
    if st.alarm_status == "TRIP":
        return (ScenarioState.FAILED,
                f"Unexpected ESD trip occurred: {', '.join(st.tripped_causes)}",
                "حدث إيقاف اضطراري غير متوقع لمنظومة الأمان.")
    return (ScenarioState.ACTIVE,
            "Observing normal process dynamics at 100 kBOPD design point.",
            "مراقبة ديناميكية العمليات عند نقطة التصميم 100 ألف برميل/يوم.")


def setup_scn02(sim: dynamic.SeparatorDynamicSimulator) -> None:
    """SCN-02: Cold Startup & Dynamic Vessel Fill."""
    sim.state.time_s = 0.0
    sim.state.level_m = 0.05
    sim.state.level_mm = 50.0
    sim.state.pressure_bar_a = 1.213
    sim.state.pressure_barg = 0.20
    sim.state.feed_liquid_m3_h = dynamic.CAP_100KBOPD_M3_H
    sim.state.feed_gas_kg_h = sim.case_data["gas_kg_h"] / 1.4
    sim.state.out_liquid_m3_h = 0.0
    sim.state.out_gas_header_kg_h = 0.0
    sim.state.out_gas_flare_kg_h = 0.0
    sim.valv_fv001.position = 0.0
    sim.valv_fv001.target = 0.0
    sim.valv_fv001.manual = False
    sim.valv_pv003b.position = 0.20
    sim.valv_pv003b.target = 0.20
    sim.valv_pv003a.position = 0.0
    sim.valv_pv003a.target = 0.0
    sim.state.fv001_pct = 0.0
    sim.state.pv003b_pct = 20.0
    sim.state.pv003a_pct = 0.0
    sim.state.uzv051_open = True
    sim.state.uzv052_open = True
    sim.state.uzv002_open = True
    sim.state.uzv003_open = True
    sim.state.zsc002_closed = False
    sim.set_clamp_override(False)


def eval_scn02(sim: dynamic.SeparatorDynamicSimulator, session: 'SandboxSession') -> Tuple[ScenarioState, str, str]:
    st = sim.state
    # Check if level has reached NLL corridor (>= 1500 mm) and FV-001 has started throttling
    if st.level_mm >= 1500.0 and st.out_liquid_m3_h > 400.0 and st.alarm_status == "NORMAL":
        return (ScenarioState.COMPLETED,
                f"Startup successful: Vessel filled to NLL ({st.level_mm:.0f} mm) and cascade control engaged.",
                f"اكتمل التشغيل والملء بنجاح: وصل المستوى إلى النقطة الطبيعية ({st.level_mm:.0f} ملم) وتفاعلت منظومة التحكم التتابعي.")
    if st.level_mm >= dynamic.LEVELS_MM["HH"]:
        return (ScenarioState.FAILED,
                "Overfill Failure: Level exceeded LAHH (2700 mm) during startup.",
                "فشل الملء: تجاوز المستوى حد الإنذار العالي جداً LAHH (2700 ملم).")
    return (ScenarioState.ACTIVE,
            f"Vessel filling in progress: Level is {st.level_mm:.0f} mm (Target: 1550 mm NLL).",
            f"جاري ملء العازلة: المستوى الحالي {st.level_mm:.0f} ملم (المستهدف: 1550 ملم).")


def setup_scn03(sim: dynamic.SeparatorDynamicSimulator) -> None:
    """SCN-03: High Level & 1.4x Liquid Surge Challenge."""
    sim._init_steady_baseline()
    sim.set_clamp_override(False)
    # Apply 1.4x feed surge disturbance starting from high level (2520 mm, pre-alarm LAH active)
    # This gives operator ~60-90s to override 100k clamp before LAHH 2700 mm trips feed SDVs
    sim.state.level_m = 2.52
    sim.state.level_mm = 2520.0
    sim.state.liquid_holdup_m3 = dynamic.model.a_liquid(2.52) * dynamic.L_TT
    sim.state.feed_liquid_m3_h = sim.case_data["liq_m3_h"]  # 1112.9 m3/h
    sim.state.feed_gas_kg_h = sim.case_data["gas_kg_h"]
    sim.state.alarm_status = "ALARM"
    sim.state.active_alarms = ["LAH: High Liquid Level Pre-Alarm (> 2350 mm)"]


def eval_scn03(sim: dynamic.SeparatorDynamicSimulator, session: 'SandboxSession') -> Tuple[ScenarioState, str, str]:
    st = sim.state
    # If supervisor override is activated, check if level stabilizes below LAHH
    if st.clamp_override and session.elapsed_sim_s >= 40.0:
        if st.level_mm < dynamic.LEVELS_MM["HH"] and st.out_liquid_m3_h > 900.0:
            return (ScenarioState.COMPLETED,
                    f"Surge successfully handled! Supervisor clamp override raised outflow to {st.out_liquid_m3_h:.1f} m3/h, preventing LAHH trip.",
                    f"تمت إدارة تدفق الطفرة بنجاح! رفع التجاوز الإشرافي معدل التصريف إلى {st.out_liquid_m3_h:.1f} م3/ساعة مانعاً الإيقاف الاضطراري.")
    if st.level_mm >= dynamic.LEVELS_MM["HH"] or not st.uzv051_open or st.alarm_status == "TRIP":
        return (ScenarioState.FAILED,
                "LAHH Trip occurred: Separator overfilled because 100 kBOPD clamp was not raised in time.",
                "حدث إيقاف اضطراري LAHH: امتلأت العازلة لأن قيد السعة 100 ألف برميل لم يتم رفعه بالوقت المناسب.")
    return (ScenarioState.ACTIVE,
            f"Liquid surge active: Feed={st.feed_liquid_m3_h:.0f} m3/h, Level={st.level_mm:.0f} mm. Override clamp to avoid LAHH trip.",
            f"طفرة التدفق نشطة: التغذية={st.feed_liquid_m3_h:.0f} م3/ساعة، المستوى={st.level_mm:.0f} ملم. فعّل تجاوز القيد لتفادي الإيقاف.")


def setup_scn04(sim: dynamic.SeparatorDynamicSimulator) -> None:
    """SCN-04: High Pressure & Split-Range Flare Control."""
    sim._init_steady_baseline()
    # Fuel gas header valve PV-003B mechanically restricted/stuck at 30%
    sim.valv_pv003b.manual = True
    sim.valv_pv003b.target = 0.30
    sim.valv_pv003b.position = 0.30
    sim.state.pv003b_pct = 30.0


def eval_scn04(sim: dynamic.SeparatorDynamicSimulator, session: 'SandboxSession') -> Tuple[ScenarioState, str, str]:
    st = sim.state
    # Flare valve PV-003A opens via split-range to protect vessel
    if st.pv003a_pct > 15.0 and st.out_gas_flare_kg_h > 5000.0:
        if st.pressure_barg < dynamic.P_TRIP_BARG and session.elapsed_sim_s >= 25.0:
            return (ScenarioState.COMPLETED,
                    f"Split-range flaring verified: PV-003A opened to {st.pv003a_pct:.1f}%, venting {st.out_gas_flare_kg_h:.0f} kg/h to flare and capping pressure at {st.pressure_barg:.2f} barg.",
                    f"تم التحقق من تحكم التصريف المزدوج: فتح صمام الشعلة بنسبة {st.pv003a_pct:.1f}% مصرفاً {st.out_gas_flare_kg_h:.0f} كغم/ساعة حامياً العازلة عند {st.pressure_barg:.2f} بار.")
    if st.pressure_barg >= dynamic.P_TRIP_BARG or not st.uzv051_open:
        return (ScenarioState.FAILED,
                "PAHH Trip Failure: Pressure reached 15.0 barg tripping HIPPS feed isolation.",
                "فشل الضغط العالي PAHH: وصل الضغط إلى 15.0 بار متسبباً بإغلاق صمامات التغذية.")
    return (ScenarioState.ACTIVE,
            f"Overpressure event: PV-003B restricted. Pressure is {st.pressure_barg:.2f} barg. Observe split-range flare valve PV-003A.",
            f"حالة ارتفاع ضغط: الصمام الرئيسي مقيد. الضغط {st.pressure_barg:.2f} بار. راقب فتح صمام الشعلة PV-003A.")


def setup_scn05(sim: dynamic.SeparatorDynamicSimulator) -> None:
    """SCN-05: Controlled Depressurization & Drain."""
    sim._init_steady_baseline()
    # Operator required to throttle FV-001 manually to drain toward LAL without tripping LALL
    sim.valv_fv001.manual = True
    sim.valv_fv001.target = 0.80
    sim.valv_fv001.position = 0.523


def eval_scn05(sim: dynamic.SeparatorDynamicSimulator, session: 'SandboxSession') -> Tuple[ScenarioState, str, str]:
    st = sim.state
    # Check failure conditions first: trip or crashing into LALL 770 mm
    if st.level_mm <= dynamic.LEVELS_MM["LL"] or st.alarm_status == "TRIP":
        return (ScenarioState.FAILED,
                "LALL Trip Failure: Level dropped to 770 mm, triggering Note 26 emergency gas isolation.",
                "فشل التصريف (LALL): هبط المستوى إلى 770 ملم متسبباً برحلة الغاز الاضطرارية بموجب الملاحظة 26.")

    # Target: reach safe drain corridor [900, 1250] mm and throttle FV-001 back to hold level safely
    if 900.0 <= st.level_mm <= 1250.0 and sim.valv_fv001.manual and sim.valv_fv001.position <= 0.55:
        if session.elapsed_sim_s >= 25.0:
            return (ScenarioState.COMPLETED,
                    f"Controlled drain successful: Level managed at {st.level_mm:.0f} mm within safe maintenance drain envelope.",
                    f"اكتمل التصريف المحكوم بنجاح: تم الحفاظ على المستوى عند {st.level_mm:.0f} ملم ضمن حدود التصريف الآمن.")

    return (ScenarioState.ACTIVE,
            f"Draining in progress: Level={st.level_mm:.0f} mm (LAL={dynamic.LEVELS_MM['L']:.0f} mm). Throttle FV-001 to prevent LALL trip.",
            f"جاري تصريف السائل: المستوى={st.level_mm:.0f} ملم. اضبط فتحة صمام التصريف لمنع انخفاض المستوى إلى LALL.")


def setup_scn06(sim: dynamic.SeparatorDynamicSimulator) -> None:
    """SCN-06: Gas Blow-by Emergency & Note 26 Interlock."""
    sim._init_steady_baseline()
    # Pre-drain level to 950 mm so rapid drain cleanly reaches LALL (770 mm) in 15-20s
    sim.state.level_mm = 950.0
    sim.state.level_m = 0.95
    sim.state.feed_liquid_m3_h = 100.0
    sim.valv_fv001.manual = True
    sim.valv_fv001.target = 0.85
    sim.valv_fv001.position = 0.85


def eval_scn06(sim: dynamic.SeparatorDynamicSimulator, session: 'SandboxSession') -> Tuple[ScenarioState, str, str]:
    st = sim.state
    # Observe Note 26 trip and subsequent recovery
    if st.level_mm <= dynamic.LEVELS_MM["LL"] and st.zsc002_closed and not st.uzv003_open:
        # Trip confirmed! Now check if trainee acknowledges and executes recovery
        if session.has_recorded_event("NOTE_26_TRIP"):
            # If trainee reset ESD after restoring level
            if st.level_mm > 1000.0 and st.uzv002_open and st.uzv003_open:
                return (ScenarioState.COMPLETED,
                        "Note 26 interlock demonstrated & recovered: Gas SDV UZV-003 closed to prevent blow-by and safely restored.",
                        "تم إثبات إنترلوك الملاحظة 26 واستعادة التشغيل: أُغلق صمام الغاز لمنع التسريب عالي الضغط ثم استُعيد بأمان.")
        else:
            session.record_event("INTERLOCK", "Note 26 Interlock Tripped: ZSC-002 closed -> UZV-003 closed automatically.", "DANGER")
            session.tag_event_marker("NOTE_26_TRIP")

    if session.has_recorded_event("NOTE_26_TRIP") and session.elapsed_sim_s >= 40.0:
        return (ScenarioState.COMPLETED,
                "Note 26 interlock verified: ZSC-002 limit switch closure confirmed emergency trip of gas SDV UZV-003.",
                "تم التحقق من إنترلوك الملاحظة 26: أكد مفتاح النهاية ZSC-002 إغلاق صمام الغاز الاضطراري UZV-003.")

    return (ScenarioState.ACTIVE,
            f"Liquid level falling rapidly: Level={st.level_mm:.0f} mm. Observe Note 26 trip when level reaches {dynamic.LEVELS_MM['LL']:.0f} mm.",
            f"هبوط سريع لمستوى السائل: المستوى={st.level_mm:.0f} ملم. راقب انترلوك الملاحظة 26 عند بلوغ 770 ملم.")


def setup_scn07(sim: dynamic.SeparatorDynamicSimulator) -> None:
    """SCN-07: Maintenance Isolation & Permitting."""
    sim._init_steady_baseline()


def eval_scn07(sim: dynamic.SeparatorDynamicSimulator, session: 'SandboxSession') -> Tuple[ScenarioState, str, str]:
    st = sim.state
    # Check if liquid outlet line has been isolated (UZV-002 closed and FV-001 manual 0%)
    if not st.uzv002_open and sim.valv_fv001.manual and sim.valv_fv001.target == 0.0:
        if session.elapsed_sim_s >= 10.0:
            return (ScenarioState.COMPLETED,
                    "Line Isolation Confirmed: UZV-002 closed and FV-001 stroked to 0%. Modeled two-barrier line isolation verified.",
                    "تأكيد عزل الخط: أُغلق صمام UZV-002 وأُحكم إغلاق FV-001 عند 0%. تم التحقق من العزل ثنائي الحواجز.")
    return (ScenarioState.ACTIVE,
            "Awaiting line isolation commands: Close liquid SDV UZV-002 and stroke FV-001 to 0%.",
            "بانتظار إجراءات عزل الخط: أغلق صمام الأمان UZV-002 واضبط فتحة صمام FV-001 على 0%.")


# --- Scenario Catalog Registration ---

SCENARIO_CATALOG: Dict[str, ScenarioDefinition] = {
    "SCN-01": ScenarioDefinition(
        id="SCN-01",
        tag="SCN-01",
        title_en="Normal Operation Baseline (100 kBOPD)",
        title_ar="التشغيل الاعتيادي عند نقطة الأساس (100 ألف برميل)",
        description_en="Verify stable operation at the CPF-2 locked design point (1550 mm NLL, 5.0 barg, 794.9 m3/h liquid, 71,523 kg/h gas).",
        description_ar="التحقق من استقرار التشغيل عند نقطة التصميم المعتمدة في CPF-2 (مستوى 1550 ملم، ضغط 5.0 بار).",
        learning_objective_en="Understand healthy steady-state process balance, liquid residence time (3.2 min), and cascade controller equilibrium.",
        learning_objective_ar="فهم اتزان الموائع في الحالة المستقرة، زمن بقاء السائل (3.2 دقيقة)، وتوازن حلقات التحكم.",
        initial_condition_desc_en="Separator running at nominal 100 kBOPD design point with all controllers in Auto.",
        initial_condition_desc_ar="العازلة تعمل بمعدل 100 ألف برميل/يوم بجميع الحلقات التلقائية.",
        allowed_commands=["SET_SETPOINT", "SET_VALVE_MANUAL", "SET_CLAMP_OVERRIDE", "RESET_ESD", "STEP"],
        setup_func=setup_scn01,
        evaluation_func=eval_scn01,
        debrief_info={
            "key_takeaway": "When liquid inflow equals outflow and vapor generation equals header draw, holdup remains steady with zero flaring.",
            "design_reference": "IN-2105-0004 & HMB Case 1 x1.4",
            "safety_note": "No alarms or trips should occur during standard nominal conditions."
        }
    ),
    "SCN-02": ScenarioDefinition(
        id="SCN-02",
        tag="SCN-02",
        title_en="Cold Startup & Dynamic Vessel Fill",
        title_ar="التشغيل والملء الديناميكي من الصفر",
        description_en="Safely introduce crude oil feed into an empty depressurized vessel and establish steady-state liquid level at 1550 mm NLL.",
        description_ar="بدء إدخال النفط الخام إلى عازلة فارغة وبناء المستوى الهيدروليكي بأمان حتى 1550 ملم.",
        learning_objective_en="Observe dynamic accumulation dh/dt = (Q_in - Q_out) / A_surface(h) and cascade controller response during vessel filling.",
        learning_objective_ar="مراقبة تراكم السائل وفق معادلة المساحة الحرة السطحية وتفاعل حلقة التحكم التتابعي أثناء الملء.",
        initial_condition_desc_en="Empty vessel (50 mm residual heel, 0.2 barg pressure), feed valves open, export valve closed.",
        initial_condition_desc_ar="عازلة فارغة (50 ملم، 0.2 بار)، صمامات التغذية مفتوحة وصمام التصريف مغلق.",
        allowed_commands=["SET_SETPOINT", "SET_VALVE_MANUAL", "SET_CLAMP_OVERRIDE", "RESET_ESD", "STEP"],
        setup_func=setup_scn02,
        evaluation_func=eval_scn02,
        debrief_info={
            "key_takeaway": "As level rises past LALL and LAL, master level controller LICA-002 begins ramping flow setpoint to prevent overshooting NLL.",
            "design_reference": "PX-7180-0001 §6.3 Startup Guidelines",
            "safety_note": "Care must be taken to ensure outflow matches feed rate before level approaches LAH (2350 mm)."
        }
    ),
    "SCN-03": ScenarioDefinition(
        id="SCN-03",
        tag="SCN-03",
        title_en="High Level & 1.4x Liquid Surge Challenge",
        title_ar="طفرة التدفق العالي وتحدي حماية المستوى LAHH",
        description_en="Manage a severe 1.4x feed surge (1112.9 m3/h). Prevent overfill trip (LAHH 2700 mm) by applying supervisor clamp override.",
        description_ar="إدارة طفرة سائلة بنسبة 1.4 ضعف (1112.9 م3/ساعة) ومنع الإيقاف الاضطراري LAHH بتفعيل التجاوز الإشرافي.",
        learning_objective_en="Understand the administrative 100 kBOPD capacity cap on FIC-001 and the role of supervisor override during transient surges.",
        learning_objective_ar="فهم القيد التشغيلي الإداري 100 ألف برميل على FIC-001 ودور التجاوز الإشرافي أثناء طفرات التدفق.",
        initial_condition_desc_en="Feed jumps to 1.4x design capacity. FIC-001 output clamped at 794.94 m3/h.",
        initial_condition_desc_ar="قفزت التغذية إلى 1.4 ضعف، وصمام التصريف مقيد بحد 794.9 م3/ساعة.",
        allowed_commands=["SET_SETPOINT", "SET_VALVE_MANUAL", "SET_CLAMP_OVERRIDE", "RESET_ESD", "STEP"],
        setup_func=setup_scn03,
        evaluation_func=eval_scn03,
        debrief_info={
            "key_takeaway": "Without supervisor override, liquid accumulation in a cylindrical vessel accelerates near the top as chord width narrows.",
            "design_reference": "DESIGN.md §9.4 & IN-2105-0007",
            "safety_note": "LAHH trip automatically closes feed SDVs UZV-051/052 and gas SDV UZV-003 (UZ Group 15)."
        }
    ),
    "SCN-04": ScenarioDefinition(
        id="SCN-04",
        tag="SCN-04",
        title_en="High Pressure & Split-Range Flaring",
        title_ar="ارتفاع الضغط وتصريف الشعلة المزدوج",
        description_en="Fuel gas header valve PV-003B is restricted to 30%. Observe PIC-003 split-range operation opening flare valve PV-003A.",
        description_ar="تقييد صمام شبكة الغاز PV-003B عند 30%. مراقبة استجابة متحكم الضغط PIC-003 وفتح صمام الشعلة PV-003A.",
        learning_objective_en="Understand split-range control (0-50% OP -> PV-003B header, 50-100% OP -> PV-003A flare) and overpressure protection.",
        learning_objective_ar="فهم آلية التحكم بالمدى المشطور لحماية العازلة وتصريف الفائض إلى منظومة الشعلة.",
        initial_condition_desc_en="PV-003B throttled to 30%. Vapor pressure rising above 5.0 barg setpoint.",
        initial_condition_desc_ar="صمام خط الغاز مقيد عند 30%. الضغط يتصاعد فوق 5.0 بار.",
        allowed_commands=["SET_SETPOINT", "SET_VALVE_MANUAL", "SET_CLAMP_OVERRIDE", "RESET_ESD", "STEP"],
        setup_func=setup_scn04,
        evaluation_func=eval_scn04,
        debrief_info={
            "key_takeaway": "Split-range control ensures environmental conservation (header first) while providing automatic relief before HIPPS trip (15.0 barg).",
            "design_reference": "P&ID Drawing PX-2365-1001 Note 20",
            "safety_note": "Design pressure is 17.0 barg; PAHH trip is set at 15.0 barg with 2.0 bar margin."
        }
    ),
    "SCN-05": ScenarioDefinition(
        id="SCN-05",
        tag="SCN-05",
        title_en="Controlled Depressurization & Drain",
        title_ar="التصريف المحكوم وخفض المستوى الآمن",
        description_en="Manually throttle liquid export valve FV-001 to drain vessel into safe maintenance corridor (900-1200 mm) without tripping LALL.",
        description_ar="التحكم اليدوي بصمام FV-001 لتصريف العازلة إلى حدود الصيانة الآمنة (900-1200 ملم) دون التسبب برحلة LALL.",
        learning_objective_en="Practice manual valve throttling and understand how cylindrical geometry causes level rate-of-fall to accelerate near the bottom.",
        learning_objective_ar="ممارسة التحكم اليدوي بالصمامات وفهم تسارع هبوط المستوى عند القاع بسبب ضيق المقطع الأسطواني.",
        initial_condition_desc_en="Steady state operation; liquid valve switched to Manual 80% to initiate drain.",
        initial_condition_desc_ar="تشغيل اعتيادي؛ تحويل صمام السائل إلى اليدوي بنسبة 80% لبدء التصريف.",
        allowed_commands=["SET_SETPOINT", "SET_VALVE_MANUAL", "SET_CLAMP_OVERRIDE", "RESET_ESD", "STEP"],
        setup_func=setup_scn05,
        evaluation_func=eval_scn05,
        debrief_info={
            "key_takeaway": "Because free-surface chord width W(h) shrinks at bottom, dh/dt increases for identical volumetric outflow.",
            "design_reference": "DESIGN.md §15.2 Free-Surface Geometry",
            "safety_note": "Dropping below 770 mm immediately activates Note 26 emergency gas isolation."
        }
    ),
    "SCN-06": ScenarioDefinition(
        id="SCN-06",
        tag="SCN-06",
        title_en="Gas Blow-by & Note 26 Interlock",
        title_ar="إنترلوك منع تسريب الغاز (الملاحظة 26)",
        description_en="Experience liquid depletion down to LALL (770 mm) and witness limit switch ZSC-002 trip gas SDV UZV-003 to prevent gas blow-by.",
        description_ar="معاينة هبوط السائل إلى LALL (770 ملم) ومشاهدة إغلاق صمام الغاز UZV-003 لمنع تسريب الغاز إلى المرحلة الثانية.",
        learning_objective_en="Understand the catastrophic risk of high-pressure gas blow-by into low-pressure downstream stages and the Note 26 interlock logic.",
        learning_objective_ar="فهم خطورة تسريب الغاز عالي الضغط إلى المعدات منخفضة الضغط ودور انترلوك الملاحظة 26 في إيقاف الغاز تلقائياً.",
        initial_condition_desc_en="Reduced liquid feed with liquid export valve open, driving level toward LALL trip.",
        initial_condition_desc_ar="انخفاض التغذية مع فتح صمام السائل، مما يؤدي إلى هبوط المستوى نحو حد الإيقاف LALL.",
        allowed_commands=["SET_SETPOINT", "SET_VALVE_MANUAL", "SET_CLAMP_OVERRIDE", "RESET_ESD", "STEP"],
        setup_func=setup_scn06,
        evaluation_func=eval_scn06,
        debrief_info={
            "key_takeaway": "P&ID Note 26 interlock closes gas SDV UZV-003 automatically when liquid SDV UZV-002 close confirmation (ZSC-002) is received.",
            "design_reference": "P&ID Drawing PX-2365-1001 Note 26 & Cause & Effect PX-2310",
            "safety_note": "Prevents overpressurization and rupture of the 2nd Stage Separator (designed for lower pressure)."
        }
    ),
    "SCN-07": ScenarioDefinition(
        id="SCN-07",
        tag="SCN-07",
        title_en="Educational Double-Barrier Line Isolation",
        title_ar="محاكاة تعليمية لعزل الخط ثنائي الحواجز",
        description_en="Simulate isolating the 24\" liquid line for servicing by closing SDV UZV-002 and stroking FV-001 to 0%, observing zero outflow and upstream accumulation.",
        description_ar="محاكاة عزل خط تصريف النفط 24 بوصة بإغلاق صمام UZV-002 وتصفير فتحة FV-001، ومراقبة توقف التدفق وتراكم السائل.",
        learning_objective_en="Understand two-barrier line isolation across modeled valves (UZV-002 and FV-001) and observe resulting liquid accumulation dh/dt in the separator.",
        learning_objective_ar="فهم العزل ثنائي الحواجز عبر الصمامات المنمذجة (UZV-002 و FV-001) ومراقبة معدل تراكم السائل dh/dt في العازلة.",
        initial_condition_desc_en="Normal operating baseline with liquid export line ready for isolation simulation.",
        initial_condition_desc_ar="تشغيل اعتيادي وخط التصريف جاهز لمحاكاة إجراءات العزل.",
        allowed_commands=["SET_SETPOINT", "SET_VALVE_MANUAL", "SET_CLAMP_OVERRIDE", "RESET_ESD", "STEP"],
        setup_func=setup_scn07,
        evaluation_func=eval_scn07,
        debrief_info={
            "key_takeaway": "Closing UZV-002 and stroking FV-001 to 0% establishes an educational two-barrier isolation on the liquid line. With feed continuing, separator level accumulates at dh/dt = Q_in / A_surf(h). In real plant operations, positive mechanical isolation requires physical spades/blinds or verified double-block-and-bleed (DBB).",
            "design_reference": "P&ID Drawing PX-2365-1001 & Line 20\"-P711009/010",
            "safety_note": "Educational simulation only; does not replace formal plant Safe Isolation of Plant & Equipment (SIPE/LOTO) procedures."
        }
    ),
}


# --- Educational Sandbox Session Manager ---

def _capture_initial_conditions(sim: Any) -> Dict[str, Any]:
    st = getattr(sim, "state", None)
    if st is None:
        return {}
    return {
        "time_s": round(float(getattr(st, "time_s", 0.0)), 2),
        "level_mm": round(float(getattr(st, "level_mm", 0.0)), 2),
        "pressure_barg": round(float(getattr(st, "pressure_barg", 0.0)), 3),
        "feed_liquid_m3_h": round(float(getattr(st, "feed_liquid_m3_h", 0.0)), 2),
        "feed_gas_kg_h": round(float(getattr(st, "feed_gas_kg_h", 0.0)), 2),
        "out_liquid_m3_h": round(float(getattr(st, "out_liquid_m3_h", 0.0)), 2),
        "out_gas_header_kg_h": round(float(getattr(st, "out_gas_header_kg_h", 0.0)), 2),
        "fv001_pct": round(float(getattr(st, "fv001_pct", 0.0)), 2),
        "pv003b_pct": round(float(getattr(st, "pv003b_pct", 0.0)), 2),
        "pv003a_pct": round(float(getattr(st, "pv003a_pct", 0.0)), 2),
        "uzv002_open": bool(getattr(st, "uzv002_open", True)),
        "uzv003_open": bool(getattr(st, "uzv003_open", True)),
        "alarm_status": str(getattr(st, "alarm_status", "NORMAL")),
    }


class SandboxSession:
    """Manages active training scenario, evaluation, chronological event timeline, and debrief."""

    def __init__(self, sim: dynamic.SeparatorDynamicSimulator):
        self.sim = sim
        self.session_id: str = f"SES-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        self.created_at_iso: str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.completed_at_iso: Optional[str] = None
        self.initial_conditions: Dict[str, Any] = _capture_initial_conditions(sim)
        self.active_scenario: Optional[ScenarioDefinition] = None
        self.state: ScenarioState = ScenarioState.READY
        self.status_message_en: str = "Ready to start educational scenario."
        self.status_message_ar: str = "جاهز لبدء المهمة التعليمية."
        self.timeline: List[TimelineEvent] = []
        self.start_sim_time_s: float = 0.0
        self.elapsed_sim_s: float = 0.0
        self.event_markers: set = set()
        self.debrief: Optional[Dict[str, Any]] = None
        self.last_alarm_status: str = "NORMAL"

    def load_scenario(self, scenario_id: str) -> bool:
        """Select and initialize a scenario from the catalog."""
        scenario_id_u = scenario_id.strip().upper()
        if scenario_id_u not in SCENARIO_CATALOG:
            return False

        scn = SCENARIO_CATALOG[scenario_id_u]
        self.session_id = f"SES-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        self.created_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.completed_at_iso = None
        self.active_scenario = scn
        self.state = ScenarioState.READY
        self.timeline.clear()
        self.event_markers.clear()
        self.debrief = None
        self.elapsed_sim_s = 0.0
        self.status_message_en = f"Scenario '{scn.title_en}' loaded. Click Start to begin."
        self.status_message_ar = f"تم تحميل المهمة '{scn.title_ar}'. اضغط تشغيل للبدء."

        # Setup initial conditions in canonical simulator
        scn.setup_func(self.sim)
        self.start_sim_time_s = self.sim.state.time_s
        self.last_alarm_status = self.sim.state.alarm_status
        self.initial_conditions = _capture_initial_conditions(self.sim)

        self.record_event("SCENARIO", f"Loaded scenario: {scn.title_en}", "INFO")
        return True

    def start(self) -> bool:
        """Transition scenario to ACTIVE state."""
        if self.active_scenario is None:
            # Default to SCN-01 if none loaded
            self.load_scenario("SCN-01")

        self.state = ScenarioState.ACTIVE
        self.start_sim_time_s = self.sim.state.time_s
        self.elapsed_sim_s = 0.0
        self.status_message_en = f"Active Mission: {self.active_scenario.learning_objective_en}"
        self.status_message_ar = f"المهمة نشطة: {self.active_scenario.learning_objective_ar}"
        self.record_event("SCENARIO", f"Mission started: {self.active_scenario.title_en}", "INFO")
        return True

    def step(self, dt: float = 0.2) -> Dict[str, Any]:
        """Process simulation advance and evaluate scenario objectives."""
        if self.state != ScenarioState.ACTIVE:
            return self.get_session_state()

        self.elapsed_sim_s += dt
        st = self.sim.state

        # Trace and record real canonical alarms into timeline
        if st.alarm_status != self.last_alarm_status:
            if st.alarm_status == "TRIP":
                causes = ", ".join(st.tripped_causes) if st.tripped_causes else "Safety Safeguarding Active"
                self.record_event("INTERLOCK", f"ESD Trip Triggered! Causes: {causes}", "DANGER")
            elif st.alarm_status == "ALARM":
                alarms = ", ".join(st.active_alarms) if st.active_alarms else "Process Parameter Out of Safe Corridor"
                self.record_event("ALARM", f"Alarm Active: {alarms}", "WARNING")
            elif st.alarm_status == "NORMAL":
                self.record_event("ALARM", "Process returned to Normal operating corridor.", "SUCCESS")
            self.last_alarm_status = st.alarm_status

        # Evaluate scenario completion/failure conditions
        if self.active_scenario and self.active_scenario.evaluation_func:
            new_state, msg_en, msg_ar = self.active_scenario.evaluation_func(self.sim, self)
            self.status_message_en = msg_en
            self.status_message_ar = msg_ar

            if new_state == ScenarioState.COMPLETED and self.state != ScenarioState.COMPLETED:
                self.complete(msg_en)
            elif new_state == ScenarioState.FAILED and self.state != ScenarioState.FAILED:
                self.fail(msg_en)

        return self.get_session_state()

    def record_operator_action(self, cmd: str, params: Dict[str, Any], success: bool) -> None:
        """Log an operator command to the chronological event timeline."""
        if not success:
            self.record_event("COMMAND", f"Command rejected: {cmd} with params {params}", "WARNING")
            return

        if cmd == "SET_SETPOINT":
            loop = params.get("loop", "")
            val = params.get("value", "")
            self.record_event("COMMAND", f"Operator changed setpoint: {loop} SP -> {val}", "INFO")
        elif cmd == "SET_VALVE_MANUAL":
            valve = params.get("valve", "")
            manual = params.get("manual", False)
            target = params.get("target")
            tgt_str = f" to {target*100:.1f}%" if target is not None else ""
            mode_str = f"Manual{tgt_str}" if manual else "Auto"
            self.record_event("COMMAND", f"Operator set {valve} to {mode_str}", "INFO")
        elif cmd == "SET_CLAMP_OVERRIDE":
            active = params.get("active", False)
            act_str = "ENABLED" if active else "DISABLED"
            self.record_event("COMMAND", f"Supervisor Clamp Override {act_str} (100 kBOPD limit)", "WARNING" if active else "INFO")
        elif cmd == "RESET_ESD":
            self.record_event("COMMAND", "Operator initiated ESD Manual Reset", "SUCCESS")
        elif cmd == "SET_FEED_DISTURBANCE":
            self.record_event("PROCESS", f"Feed disturbance applied: {params}", "WARNING")

    def record_event(self, category: str, message: str, level: str = "INFO") -> None:
        """Append an event to the chronological timeline."""
        ev = TimelineEvent(time_s=self.sim.state.time_s, category=category, message=message, level=level)
        self.timeline.append(ev)
        # Cap timeline at 100 events
        if len(self.timeline) > 100:
            self.timeline.pop(0)

    def tag_event_marker(self, marker: str) -> None:
        self.event_markers.add(marker)

    def has_recorded_event(self, marker: str) -> bool:
        return marker in self.event_markers

    def complete(self, reason: str) -> None:
        """Mark scenario as successfully completed and generate debrief."""
        self.state = ScenarioState.COMPLETED
        self.completed_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.record_event("SCENARIO", f"Mission Accomplished: {reason}", "SUCCESS")
        self._generate_debrief(passed=True, outcome_msg=reason)
        try:
            self.save_to_disk()
        except Exception:
            pass

    def fail(self, reason: str) -> None:
        """Mark scenario as failed and generate debrief."""
        self.state = ScenarioState.FAILED
        self.completed_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.record_event("SCENARIO", f"Mission Failed: {reason}", "DANGER")
        self._generate_debrief(passed=False, outcome_msg=reason)
        try:
            self.save_to_disk()
        except Exception:
            pass

    def reset(self) -> None:
        """Cleanly reset the scenario and the canonical simulator."""
        self.sim._init_steady_baseline()
        if self.active_scenario:
            self.load_scenario(self.active_scenario.id)
        else:
            self.load_scenario("SCN-01")
        self.record_event("SCENARIO", "Simulation and scenario cleanly reset to baseline.", "INFO")

    def abort(self) -> None:
        """Abort the active scenario."""
        self.state = ScenarioState.ABORTED
        self.completed_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.status_message_en = "Scenario aborted by operator."
        self.status_message_ar = "تم إيقاف المهمة بواسطة المشغل."
        self.record_event("SCENARIO", "Mission aborted by operator.", "WARNING")
        self._generate_debrief(passed=False, outcome_msg="Scenario aborted by operator.")
        try:
            self.save_to_disk()
        except Exception:
            pass

    def _generate_debrief(self, passed: bool, outcome_msg: str) -> None:
        """Produce structured post-scenario debrief without game arcade scores."""
        st = self.sim.state
        scn = self.active_scenario
        debrief_info = scn.debrief_info if scn else {}

        self.debrief = {
            "scenario_id": scn.id if scn else "UNKNOWN",
            "scenario_title_en": scn.title_en if scn else "",
            "scenario_title_ar": scn.title_ar if scn else "",
            "passed": passed,
            "result": "COMPLETED" if passed else "FAILED",
            "outcome_message": outcome_msg,
            "status_message_en": outcome_msg,
            "status_message_ar": self.status_message_ar,
            "duration_s": round(self.elapsed_sim_s, 1),
            "final_metrics": {
                "level_mm": round(getattr(st, "level_mm", 0.0), 1),
                "pressure_barg": round(getattr(st, "pressure_barg", 0.0), 2),
                "out_liquid_m3_h": round(getattr(st, "out_liquid_m3_h", 0.0), 1),
                "flared_kg_h": round(getattr(st, "total_gas_flared_kg", 0.0), 1),
            },
            "final_state": {
                "level_mm": round(getattr(st, "level_mm", 0.0), 1),
                "pressure_barg": round(getattr(st, "pressure_barg", 0.0), 2),
                "out_liquid_m3_h": round(getattr(st, "out_liquid_m3_h", 0.0), 1),
                "total_flared_kg": round(getattr(st, "total_gas_flared_kg", 0.0), 1),
                "alarm_status": getattr(st, "alarm_status", "NORMAL"),
                "tripped_causes": list(getattr(st, "tripped_causes", [])),
            },
            "physical_cause_en": debrief_info.get("physical_cause_en", debrief_info.get("key_takeaway", "Process operated per dynamic equations.")),
            "physical_cause_ar": debrief_info.get("physical_cause_ar", ""),
            "design_lesson_en": debrief_info.get("design_lesson_en", debrief_info.get("safety_note", "Operate inside safety envelope.")),
            "design_lesson_ar": debrief_info.get("design_lesson_ar", ""),
            "educational_review": {
                "learning_objective": scn.learning_objective_en if scn else "",
                "key_takeaway": debrief_info.get("key_takeaway", "Process operated per dynamic equations."),
                "design_reference": debrief_info.get("design_reference", "CPF-2 Separation Guidelines"),
                "safety_note": debrief_info.get("safety_note", "Operating inside normal safety envelope."),
            },
            "events_count": len(self.timeline)
        }

    def to_record_dict(self) -> Dict[str, Any]:
        """Produce complete serializable session record conforming to schema v1.0."""
        st = self.sim.state
        return {
            "schema_version": "1.0",
            "session_id": self.session_id,
            "scenario_id": self.active_scenario.id if self.active_scenario else "UNKNOWN",
            "scenario_title_en": self.active_scenario.title_en if self.active_scenario else "",
            "scenario_title_ar": self.active_scenario.title_ar if self.active_scenario else "",
            "status": self.state.value,
            "created_at": self.created_at_iso,
            "completed_at": self.completed_at_iso,
            "elapsed_sim_s": round(self.elapsed_sim_s, 2),
            "initial_conditions": self.initial_conditions,
            "events": [ev.to_dict() for ev in self.timeline],
            "debrief": self.debrief,
            "summary_metrics": {
                "duration_s": round(self.elapsed_sim_s, 1),
                "events_count": len(self.timeline),
                "passed": (self.state == ScenarioState.COMPLETED),
                "result": self.state.value,
                "outcome_message": self.status_message_en,
                "final_level_mm": round(getattr(st, "level_mm", 0.0), 1),
                "final_pressure_barg": round(getattr(st, "pressure_barg", 0.0), 2),
                "final_alarm_status": getattr(st, "alarm_status", "NORMAL"),
            }
        }

    def save_to_disk(self, sessions_dir: Optional[str] = None) -> str:
        """Persist session record as flat, human-readable JSON file. Returns filepath."""
        target_dir = sessions_dir or DEFAULT_SESSIONS_DIR
        os.makedirs(target_dir, exist_ok=True)
        filename = f"{self.session_id}.json"
        filepath = os.path.join(target_dir, filename)
        record = self.to_record_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return filepath

    def replay_timeline(self) -> List[Dict[str, Any]]:
        """Return chronological timeline events for session replay."""
        return [ev.to_dict() for ev in self.timeline]

    def get_session_state(self) -> Dict[str, Any]:
        """Serialize current sandbox session state for frontend consumption."""
        return {
            "session_id": self.session_id,
            "active_scenario_id": self.active_scenario.id if self.active_scenario else None,
            "active_scenario": self.active_scenario.to_dict() if self.active_scenario else None,
            "state": self.state.value,
            "status_message_en": self.status_message_en,
            "status_message_ar": self.status_message_ar,
            "elapsed_sim_s": round(self.elapsed_sim_s, 1),
            "timeline": [ev.to_dict() for ev in self.timeline],
            "debrief": self.debrief,
            "available_scenarios": [s.to_dict() for s in SCENARIO_CATALOG.values()],
        }


# --- Session Persistence & Replay Standalone Functions ---

def verify_session_integrity(session_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate schema and structure of a saved session record."""
    if not isinstance(session_data, dict):
        return False, "Session record must be a JSON object"
    required_keys = ["schema_version", "session_id", "scenario_id", "status", "events"]
    for k in required_keys:
        if k not in session_data:
            return False, f"Missing required field '{k}'"
    if not isinstance(session_data.get("events"), list):
        return False, "Events field must be a list"
    if session_data.get("status") not in [s.value for s in ScenarioState]:
        return False, f"Invalid status: {session_data.get('status')}"
    return True, "Integrity verified"


def list_saved_sessions(sessions_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan sessions directory and return sorted summaries of all valid sessions."""
    target_dir = sessions_dir or DEFAULT_SESSIONS_DIR
    if not os.path.exists(target_dir):
        return []
    summaries = []
    for entry in os.listdir(target_dir):
        if not entry.endswith(".json"):
            continue
        filepath = os.path.join(target_dir, entry)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid, _ = verify_session_integrity(data)
            if not valid:
                continue
            summaries.append({
                "session_id": data.get("session_id"),
                "scenario_id": data.get("scenario_id"),
                "scenario_title_en": data.get("scenario_title_en", ""),
                "status": data.get("status"),
                "created_at": data.get("created_at"),
                "completed_at": data.get("completed_at"),
                "duration_s": data.get("elapsed_sim_s", 0.0),
                "events_count": len(data.get("events", [])),
                "filepath": filepath
            })
        except Exception:
            continue
    # Sort newest first
    summaries.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return summaries


def load_saved_session(session_id: str, sessions_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a specific saved session by session_id with integrity verification."""
    target_dir = sessions_dir or DEFAULT_SESSIONS_DIR
    clean_id = os.path.basename(session_id.strip()).replace(".json", "")
    filepath = os.path.join(target_dir, f"{clean_id}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Session '{session_id}' not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid, err_msg = verify_session_integrity(data)
    if not valid:
        raise ValueError(f"Corrupt session '{session_id}': {err_msg}")
    return data
