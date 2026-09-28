"""Stage 1 Production Separator (CP2-V-71101) High-Fidelity Dynamic Simulator.

Implements the locked dynamic engineering specification from:
    docs/plans/stage1-separator/DYNAMIC_SIMULATOR_PLAN.md
    docs/plans/stage1-separator/DESIGN.md (Section 15, locked 2026-09-18)

Key features:
1. Two-Rate ODE Solver Scheme:
   - Fast compressible gas sub-steps (dt_g = 0.02s, 10 sub-steps per 0.2s main step)
   - Liquid level integration (dt = 0.2s) with true free-surface chord geometry A_surface(h)
2. Slew-rate constrained physical valves (Koso Linear FV-001, Mod-Eq% PV-003B, Eq% PV-003A)
3. Industrial DCS Controllers with conditional integration Anti-Reset Windup:
   - LICA-002: Master level PI -> Remote flow setpoint capped at 100 kBOPD (794.94 m3/h)
   - FIC-001: Slave flow PI -> FV-001 valve stem %
   - PIC-003: Split-range pressure PI (0-50% to PV-003B header, 50-100% to PV-003A flare)
4. Safeguarding Interlocks & Cause & Effect Matrix:
   - PAHH 15 barg (UZ14 / HIPPS) -> closes feed SDVs UZV-051/052
   - LAHH 2700 mm (UZ15) -> closes feed SDVs UZV-051/052 and gas SDV UZV-003
   - LALL 770 mm (UZ16) -> closes liquid SDV UZV-002
   - Note 26 Interlock: ZSC-002 confirmation trips UZV-003 to prevent gas blow-by
   - PSV-001A/B mechanical overpressure relief at 21 barg
5. Stokes' Law & Richardson-Zaki Hindered Settling for oil-water separation & carryunder
6. Souders-Brown superficial gas loading & Mellachevron/mesh-pad demister flooding
7. Gamification & Operator Career Progression scenario engine & KPI grading
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

# Import steady-state truth constants from model.py
import model

# --- Locked Vessel Physical Constants (DESIGN.md §4 & §6) ---
D = model.D                    # 4.2 m ID
R = model.R                    # 2.1 m radius
L_TT = model.L_TT              # 13.6 m T/T length
A_TOTAL = model.A_TOTAL        # 13.854 m2
K_SB = model.K_SB              # 0.150 m/s Souders-Brown target
A_GAS_NOZZLE = model.A_GAS_NOZZLE  # 0.177 m2 (20" N3)

LEVELS_MM = model.LEVELS_MM    # HH 2700, H 2350, NLL 1550, L 1200, LL 770, T 4200
P_TRIP_BARG = model.P_TRIP_BARG  # 15.0 barg PAHH
P_PSV_BARG = 21.0              # PSV-001A/B setpoint
CAP_100KBOPD_M3_H = 794.936    # 100 kBOPD oil + 20 vol% dry-oil water cut (DESIGN.md §9.4)

# Valve Sizing & Rated Cv (IN-2105-0007 Datasheets Rev 004)
CV_RATED_FV001 = 2140.0        # 14" Koso S1200 Globe, Linear trim, FC
CV_RATED_PV003B = 3990.0       # 14" Koso AB4001 Butterfly, Mod Eq%, FL (49L tank)
CV_RATED_PV003A = 1170.0       # 14" Koso S1200 Globe with baffle, Eq%, FC
VALVE_STROKE_TIME_S = 14.0     # 1 s/inch for 14" valve (PX-7180-0001 §6.3.1)
MAX_VALVE_SLEW_PER_S = 1.0 / VALVE_STROKE_TIME_S  # ~0.0714 / s


def free_surface_area(h_m: float) -> float:
    """True liquid free-surface area (m2) A_surface(h) = 2*sqrt(2*R*h - h^2) * L_TT.
    
    CRITICAL: Never divide by cross-section A_L(h); dividing by A_L would accelerate
    level movement L_TT-fold and cause false rapid trips (DESIGN.md §15.2).
    """
    # Clamp h strictly inside [0.005, D - 0.005] to prevent numerical zero-division at bottom/top
    h_c = min(max(h_m, 0.005), D - 0.005)
    chord = 2.0 * math.sqrt(max(2.0 * R * h_c - h_c * h_c, 1e-6))
    return chord * L_TT


def gas_constant_per_case(case_name: str) -> float:
    """Specific gas constant R_g (J/(kg*K)) derived from F02 case anchor.
    
    Case 1: ~287.24 J/(kg*K)
    Case 4: ~302.57 J/(kg*K)
    """
    c = model.CASES[case_name]
    p_pa = c["P_bar_a"] * 1e5
    t_k = c["T_C"] + 273.15
    rho_g = c["gas_rho"]
    return p_pa / (rho_g * t_k)


@dataclass
class ValveState:
    """Represents physical valve position, target, and dynamic slew rate."""
    position: float = 0.0      # Actual stem position [0.0, 1.0]
    target: float = 0.0        # Commanded target position [0.0, 1.0]
    manual: bool = False       # Manual override / stuck valve
    slew_rate: float = MAX_VALVE_SLEW_PER_S  # Maximum travel speed per second

    def update(self, dt: float) -> float:
        """Advance valve travel constrained by physical actuator slew rate."""
        if self.manual:
            # Under manual mode, target is explicitly pinned by user
            diff = self.target - self.position
        else:
            diff = self.target - self.position
        max_step = self.slew_rate * dt
        if abs(diff) <= max_step:
            self.position = self.target
        else:
            self.position += math.copysign(max_step, diff)
        self.position = min(max(self.position, 0.0), 1.0)
        return self.position


@dataclass
class DiscretePI:
    """Industrial Discrete PI Controller with true Conditional Integration Anti-Reset Windup.
    
    Direct/Reverse acting configurable. Freezes I-term when output is saturated at limits
    and the error would drive it further into saturation (DESIGN.md §9.3).
    """
    kp: float
    ki: float
    output_min: float = 0.0
    output_max: float = 100.0
    direct_acting: bool = True  # True: error = PV - SP (level/pressure), False: error = SP - PV
    integral: float = 0.0
    prev_output: float = 0.0
    setpoint: float = 0.0
    initial_output: float = 0.0

    def __post_init__(self):
        if self.initial_output != 0.0:
            self.integral = self.initial_output
            self.prev_output = self.initial_output

    def update(self, pv: float, dt: float) -> float:
        error = (pv - self.setpoint) if self.direct_acting else (self.setpoint - pv)
        
        # Anti-reset windup: freeze integration if already pinned at boundary and error reinforces it
        saturated_high = (self.prev_output >= self.output_max and error > 0.0)
        saturated_low = (self.prev_output <= self.output_min and error < 0.0)
        
        if not (saturated_high or saturated_low):
            self.integral += self.ki * error * dt
            # Guard against infinite integral drift
            self.integral = min(max(self.integral, -self.output_max * 2.0), self.output_max * 2.0)

        raw_output = self.kp * error + self.integral
        clamped_output = min(max(raw_output, self.output_min), self.output_max)
        self.prev_output = clamped_output
        return clamped_output

    def reset(self, initial_output: float = 0.0):
        self.prev_output = initial_output
        self.integral = initial_output


@dataclass
class SeparatorState:
    """Instantaneous dynamic state of CP2-V-71101 and associated instrumentation."""
    time_s: float = 0.0
    level_m: float = 1.550            # NLL = 1.550 m (1550 mm)
    level_mm: float = 1550.0
    pressure_bar_a: float = 6.000     # Absolute gas space pressure
    pressure_barg: float = 4.987      # Gauge pressure (PT-003)
    
    # Fluid Holdups
    liquid_holdup_m3: float = 63.16   # Active liquid volume
    liquid_geometric_m3: float = 69.13 # Geometric liquid volume (cylinder + 2:1 heads)
    gas_holdup_kg: float = 547.0      # Compressible gas mass in vapor space
    gas_holdup_m3: float = 92.2       # Vapor space volume
    
    # Inlet Conditions
    feed_liquid_m3_h: float = 794.94  # 100 kBOPD normal baseline
    feed_gas_kg_h: float = 71523.0    # 1.0x per-train design gas mass
    water_cut_vol: float = 0.1669     # 16.69% of liquid (20% on dry oil basis)
    crude_api: float = 28.5           # Zubair / Mishrif crude API
    temp_c: float = 79.10
    
    # Outflow Rates
    out_liquid_m3_h: float = 794.94   # FT-001 liquid outflow
    out_gas_header_kg_h: float = 71523.0 # Gas to flashed gas header via PV-003B
    out_gas_flare_kg_h: float = 0.0   # Gas to HP flare via PV-003A
    out_gas_psv_kg_h: float = 0.0     # Gas to HP flare via PSV-001A/B
    total_gas_flared_kg: float = 0.0  # Cumulative flared gas for scoring
    
    # Physical Valves Positions (0.0 to 1.0)
    fv001_pct: float = 52.3           # Liquid control valve % open
    pv003b_pct: float = 90.0          # Main gas butterfly valve % open (P&ID note 19)
    pv003a_pct: float = 0.0           # Flare gas globe valve % open
    
    # Safety Shutdown Valves (True = OPEN, False = CLOSED)
    uzv051_open: bool = True          # Feed SDV 1 (1003/1004)
    uzv052_open: bool = True          # Feed SDV 2 (1003/1004)
    uzv002_open: bool = True          # Liquid outlet SDV (1004 LL trip)
    uzv003_open: bool = True          # Gas outlet SDV (1004 HH / Note 26 LL trip)
    zsc002_closed: bool = False       # Note 26: UZV-002 close feedback confirmation
    
    # Controller Setpoints and Outputs
    lica002_sp_mm: float = 1550.0     # Level setpoint (NLL)
    lica002_op_flow: float = 794.94   # Flow setpoint to FIC-001 (capped at 100 kBOPD)
    fic001_sp_flow: float = 794.94    # Active setpoint of FIC-001
    fic001_op_pct: float = 52.3       # Command to FV-001
    pic003_sp_barg: float = 5.0       # Pressure setpoint
    pic003_op_pct: float = 45.0       # Split-range output (0-50% PV-003B, 50-100% PV-003A)
    clamp_override: bool = False      # Supervisor override raising 100 kBOPD cap
    
    # Physical Separation Kinetics
    stokes_settling_vel_mm_s: float = 0.85  # Stokes settling velocity of water droplets
    water_in_oil_ppm: float = 250.0   # Water carryunder in export oil
    oil_in_water_ppm: float = 150.0   # Oil carryover in produced water
    gas_velocity_m_s: float = 0.38    # Superficial gas velocity in vapor space
    gas_v_max_m_s: float = 1.81       # Souders-Brown maximum allowable velocity
    demister_flooded: bool = False    # True if v_g > v_max (droplet re-entrainment)
    carryover_gal_mmscf: float = 0.03 # Mist carryover in gas (spec < 0.1 USgal/MMscf)
    
    # Interlocks & Alarms
    alarm_status: str = "NORMAL"      # NORMAL, ALARM, TRIP
    tripped_causes: List[str] = field(default_factory=list)
    active_alarms: List[str] = field(default_factory=list)


class SeparatorDynamicSimulator:
    """High-fidelity dynamic process simulator for CPF-2 1st Stage Production Separator.
    
    Engine Architecture:
    - Incompressible liquid mass balance integrated at dt = 0.2s.
    - Compressible vapor mass & pressure balance integrated over 10 sub-steps of dt_g = 0.02s.
    - Real-world Koso valve curves (Linear & Equal-Percentage) with 14s stroke actuators.
    - Complete Cause & Effect logic matching PX-2310 and Note 26 interlock.
    """

    def __init__(self, case_name: str = "Case 1 x1.4", initial_state: Optional[SeparatorState] = None):
        self.case_name = case_name
        self.R_g = gas_constant_per_case(case_name)
        self.case_data = model.CASES[case_name]
        
        # Physical valves
        self.valv_fv001 = ValveState(position=0.523, target=0.523)
        self.valv_pv003b = ValveState(position=0.900, target=0.900)
        self.valv_pv003a = ValveState(position=0.000, target=0.000)
        
        # Industrial Controllers (tuned for CP2-V-71101 dynamics)
        # LICA-002: Master Level -> Flow Remote SP (0 to 794.94 m3/h cap)
        self.pid_level = DiscretePI(
            kp=0.75, ki=0.015,
            output_min=0.0, output_max=CAP_100KBOPD_M3_H,
            direct_acting=True,
            setpoint=1550.0,
            initial_output=CAP_100KBOPD_M3_H
        )
        
        # FIC-001: Slave Flow -> FV-001 % stem position
        self.pid_flow = DiscretePI(
            kp=0.045, ki=0.012,
            output_min=0.0, output_max=100.0,
            direct_acting=False, # Flow below setpoint -> open valve more
            setpoint=CAP_100KBOPD_M3_H,
            initial_output=52.3
        )
        
        # PIC-003: Pressure Master -> Split-Range Output (0-100%)
        initial_sp_barg = self.case_data["P_bar_a"] - 1.013
        self.pid_press = DiscretePI(
            kp=12.0, ki=0.35,
            output_min=0.0, output_max=100.0,
            direct_acting=True, # Pressure above setpoint -> open gas valves
            setpoint=initial_sp_barg,
            initial_output=45.0
        )
        
        # Initialize state
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = SeparatorState()
            self._init_steady_baseline()

    def _init_steady_baseline(self):
        """Initialize simulator exactly at the 100 kBOPD steady-state baseline."""
        self.state.level_m = 1.550
        self.state.level_mm = 1550.0
        self.state.pressure_bar_a = self.case_data["P_bar_a"]
        self.state.pressure_barg = self.state.pressure_bar_a - 1.013
        self.state.temp_c = self.case_data["T_C"]
        
        # Fluid properties
        self.state.liquid_holdup_m3 = model.a_liquid(1.550) * L_TT
        self.state.liquid_geometric_m3 = self.state.liquid_holdup_m3 + model.v_heads(1.550)
        self.state.gas_holdup_m3 = (A_TOTAL - model.a_liquid(1.550)) * L_TT
        
        # Initial gas mass from ideal-gas relationship P*V / (R_g * T)
        t_k = self.state.temp_c + 273.15
        p_pa = self.state.pressure_bar_a * 1e5
        self.state.gas_holdup_kg = (p_pa * self.state.gas_holdup_m3) / (self.R_g * t_k)
        
        # Baseline flows
        self.state.feed_liquid_m3_h = CAP_100KBOPD_M3_H
        # Baseline 1.0x gas mass = F02 gas mass / 1.4
        self.state.feed_gas_kg_h = self.case_data["gas_kg_h"] / 1.4
        self.state.out_liquid_m3_h = CAP_100KBOPD_M3_H
        self.state.out_gas_header_kg_h = self.state.feed_gas_kg_h
        self.state.out_gas_flare_kg_h = 0.0
        
        # Initial valve positions
        dp_bar = max(self.state.pressure_barg - 2.0, 0.1)
        dp_psi = dp_bar * 14.5038
        sg_liq = 0.89
        cv_needed = (self.state.feed_liquid_m3_h * 4.40287) / math.sqrt(dp_psi / sg_liq)
        init_pos = min(max(cv_needed / CV_RATED_FV001, 0.05), 0.95)
        self.valv_fv001.position = init_pos
        self.valv_fv001.target = init_pos
        self.valv_pv003b.position = 0.900
        self.valv_pv003b.target = 0.900
        self.valv_pv003a.position = 0.000
        self.valv_pv003a.target = 0.000
        
        self.state.fv001_pct = init_pos * 100.0
        self.state.pv003b_pct = 90.0
        self.state.pv003a_pct = 0.0
        
        self.pid_level.reset(CAP_100KBOPD_M3_H)
        self.pid_flow.reset(self.state.fv001_pct)
        self.pid_press.reset(45.0)

    def set_clamp_override(self, active: bool):
        """Supervisor Override: Temporarily raise FIC-001 flow cap above 100 kBOPD."""
        self.state.clamp_override = active
        if active:
            self.pid_level.output_max = 1400.0  # Allow flow to match full surge capacity
        else:
            self.pid_level.output_max = CAP_100KBOPD_M3_H

    def reset_esd(self):
        """Operator ESD Reset: Reopen latched SDVs if process conditions are within safe bounds."""
        if self.state.pressure_barg < P_TRIP_BARG and LEVELS_MM["LL"] < self.state.level_mm < LEVELS_MM["HH"]:
            self.state.uzv051_open = True
            self.state.uzv052_open = True
            self.state.uzv002_open = True
            self.state.uzv003_open = True
            self.state.zsc002_closed = False
            self.state.tripped_causes.clear()
            self.state.alarm_status = "NORMAL"

    def step(self, dt: float = 0.2,
             p_back_liq_barg: float = 2.0,
             p_back_gas_barg: float = 4.5,
             p_back_flare_barg: float = 0.2) -> SeparatorState:
        """Advance simulation by dt seconds (default 0.2s) using the Two-Rate Scheme.
        
        Sub-steps:
        1. Compressible Gas Sub-Stepping (10 iterations at dt_g = 0.02s)
        2. Liquid Level Integration (dh/dt) across chord free-surface A_surface(h)
        3. Control Loop Execution (LICA-002 -> FIC-001 -> FV-001, PIC-003 -> Split Range)
        4. Cause & Effect Interlock & Safeguarding Evaluation (PAHH, LAHH, LALL, Note 26)
        5. Stokes Settling & Separation Kinetics Evaluation
        """
        st = self.state
        st.time_s += dt
        
        # --- 1. Fast Compressible Gas Sub-Stepping (10 x 0.02s) ---
        n_gas_steps = 10
        dt_g = dt / n_gas_steps
        t_k = st.temp_c + 273.15
        
        # Vapor space volume for this step based on current level
        a_liq = model.a_liquid(st.level_m)
        v_gas = max((A_TOTAL - a_liq) * L_TT, 1.0)
        st.gas_holdup_m3 = v_gas
        
        for _ in range(n_gas_steps):
            # Gas pressure calculation from ideal-gas relation P = m * R_g * T / V
            p_pa = (st.gas_holdup_kg * self.R_g * t_k) / v_gas
            p_bar_a = p_pa / 1e5
            p_barg = max(p_bar_a - 1.013, 0.0)
            
            # Update pressure controller PIC-003
            pic_op = self.pid_press.update(p_barg, dt_g)
            st.pic003_op_pct = pic_op
            
            # Split-range mapping (P&ID note 20):
            # 0-50% OP -> PV-003B (0-100% Mod Eq. %)
            # 50-100% OP -> PV-003B remains 100%, PV-003A (0-100% Eq. %) opens to flare
            if not self.valv_pv003b.manual:
                if pic_op <= 50.0:
                    self.valv_pv003b.target = pic_op / 50.0
                else:
                    self.valv_pv003b.target = 1.0
            
            if not self.valv_pv003a.manual:
                if pic_op <= 50.0:
                    self.valv_pv003a.target = 0.0
                else:
                    self.valv_pv003a.target = (pic_op - 50.0) / 50.0
            
            # Update gas valve physical positions with actuator slew rate
            self.valv_pv003b.update(dt_g)
            self.valv_pv003a.update(dt_g)
            
            # Gas outflow calculation (IEC 60534 control valve capacity)
            # PV-003B: Modified equal-percentage characteristic
            # Cv(x) = Cv_rated * (exp(3.4 * x) - 1) / (exp(3.4) - 1)
            x_b = self.valv_pv003b.position
            cv_b = CV_RATED_PV003B * (math.exp(3.4 * x_b) - 1.0) / (math.exp(3.4) - 1.0) if x_b > 0.001 else 0.0
            
            # Outflow to flashed gas header (requires UZV-003 open)
            dp_gas_b = max(p_barg - p_back_gas_barg, 0.0)
            if st.uzv003_open and dp_gas_b > 0.001:
                # Flow proportional to Cv * sqrt(dP * P_in)
                # Calibrated factor k_b matches 71,523 kg/h at normal Cv=2800 (90% open), dP=0.487 bar
                k_b = 16.39
                gas_flow_b_kg_s = (k_b * cv_b * math.sqrt(dp_gas_b * max(p_barg, 0.5))) / 3600.0
            else:
                gas_flow_b_kg_s = 0.0
            
            # PV-003A: Equal-percentage characteristic to HP flare
            x_a = self.valv_pv003a.position
            cv_a = CV_RATED_PV003A * (math.exp(3.5 * x_a) - 1.0) / (math.exp(3.5) - 1.0) if x_a > 0.001 else 0.0
            dp_gas_a = max(p_barg - p_back_flare_barg, 0.0)
            if dp_gas_a > 0.001 and cv_a > 0.1:
                k_a = 16.39
                gas_flow_a_kg_s = (k_a * cv_a * math.sqrt(dp_gas_a * max(p_barg, 0.5))) / 3600.0
            else:
                gas_flow_a_kg_s = 0.0
            
            # PSV-001A/B mechanical overpressure relief at 21 barg
            if p_barg >= P_PSV_BARG:
                # PSV capacity 50,000 kg/h per valve
                psv_flow_kg_s = 100000.0 / 3600.0
            else:
                psv_flow_kg_s = 0.0
            
            # Net gas mass accumulation
            gas_in_kg_s = (st.feed_gas_kg_h / 3600.0) if (st.uzv051_open and st.uzv052_open) else 0.0
            gas_out_kg_s = gas_flow_b_kg_s + gas_flow_a_kg_s + psv_flow_kg_s
            
            st.gas_holdup_kg += (gas_in_kg_s - gas_out_kg_s) * dt_g
            st.gas_holdup_kg = max(st.gas_holdup_kg, 10.0) # Prevent zero mass
            
            # Accumulate flared gas
            st.total_gas_flared_kg += (gas_flow_a_kg_s + psv_flow_kg_s) * dt_g
            
            # Store instant gas flows in kg/h
            st.out_gas_header_kg_h = gas_flow_b_kg_s * 3600.0
            st.out_gas_flare_kg_h = gas_flow_a_kg_s * 3600.0
            st.out_gas_psv_kg_h = psv_flow_kg_s * 3600.0
        
        # Update pressure states after gas sub-steps
        st.pressure_bar_a = (st.gas_holdup_kg * self.R_g * t_k) / (v_gas * 1e5)
        st.pressure_barg = max(st.pressure_bar_a - 1.013, 0.0)
        st.pv003b_pct = self.valv_pv003b.position * 100.0
        st.pv003a_pct = self.valv_pv003a.position * 100.0
        
        # --- 2. Liquid DCS Control (LICA-002 -> FIC-001 -> FV-001) ---
        # LICA-002 Master Level Controller -> Flow Remote Setpoint
        flow_remote_sp = self.pid_level.update(st.level_mm, dt)
        st.lica002_op_flow = flow_remote_sp
        st.fic001_sp_flow = flow_remote_sp
        
        # FIC-001 Slave Flow Controller -> Valve Command %
        self.pid_flow.setpoint = flow_remote_sp
        fic_op_pct = self.pid_flow.update(st.out_liquid_m3_h, dt)
        st.fic001_op_pct = fic_op_pct
        
        # Update FV-001 valve target
        if not self.valv_fv001.manual:
            self.valv_fv001.target = fic_op_pct / 100.0
        
        # Move valve stem with 14s actuator slew rate
        self.valv_fv001.update(dt)
        st.fv001_pct = self.valv_fv001.position * 100.0
        
        # --- 3. Liquid Outflow Hydraulics (Koso S1200 Linear Cv = 2140) ---
        # Cv(x) = 2140 * x (Linear trim)
        cv_fv = CV_RATED_FV001 * self.valv_fv001.position
        dp_liq_bar = max(st.pressure_barg - p_back_liq_barg, 0.0)
        
        # Specific gravity of liquid mixture (~0.89)
        wc = st.water_cut_vol
        rho_liq_bulk = wc * self.case_data["water_rho"] + (1.0 - wc) * self.case_data["oil_rho"]
        sg_liq = rho_liq_bulk / 1000.0
        
        # Flow calculation: Q (m3/h) = Cv * sqrt(dP_psi / SG) / 4.40287
        # Requires liquid SDV UZV-002 to be OPEN
        if st.uzv002_open and dp_liq_bar > 0.01 and cv_fv > 0.5:
            dp_liq_psi = dp_liq_bar * 14.5038
            q_out_gpm = cv_fv * math.sqrt(dp_liq_psi / sg_liq)
            st.out_liquid_m3_h = q_out_gpm / 4.40287
        else:
            st.out_liquid_m3_h = 0.0
        
        # --- 4. Liquid Level Integration (Two-Rate Main Step dh/dt) ---
        q_in_m3_s = (st.feed_liquid_m3_h / 3600.0) if (st.uzv051_open and st.uzv052_open) else 0.0
        q_out_m3_s = st.out_liquid_m3_h / 3600.0
        
        # Surface area from free-surface chord width W(h) * L_TT
        a_surf = free_surface_area(st.level_m)
        dh_dt = (q_in_m3_s - q_out_m3_s) / a_surf
        
        st.level_m += dh_dt * dt
        st.level_m = min(max(st.level_m, 0.05), D - 0.05) # Bound within vessel
        st.level_mm = st.level_m * 1000.0
        
        # Update liquid volumes
        st.liquid_holdup_m3 = model.a_liquid(st.level_m) * L_TT
        st.liquid_geometric_m3 = st.liquid_holdup_m3 + model.v_heads(st.level_m)
        
        # --- 5. Interlock Safeguarding & Note 26 Logic (PX-2310 C&E) ---
        st.tripped_causes.clear()
        st.active_alarms.clear()
        
        # PAHH Trip (> 15.0 barg): UZ Grp 14 / HIPPS closes feed SDVs
        if st.pressure_barg >= P_TRIP_BARG:
            st.uzv051_open = False
            st.uzv052_open = False
            st.tripped_causes.append("PAHH > 15.0 barg: Feed SDVs (UZV-051/052) Tripped")
        
        # LAHH Trip (>= 2700 mm): UZ Grp 15 closes feed SDVs & gas SDV UZV-003
        if st.level_mm >= LEVELS_MM["HH"]:
            st.uzv051_open = False
            st.uzv052_open = False
            st.uzv003_open = False
            st.tripped_causes.append("LAHH >= 2700 mm: Feed SDVs & Gas SDV (UZV-003) Tripped")
        
        # LALL Trip (<= 770 mm): UZ Grp 16 closes liquid SDV UZV-002
        if st.level_mm <= LEVELS_MM["LL"]:
            st.uzv002_open = False
            st.tripped_causes.append("LALL <= 770 mm: Liquid SDV (UZV-002) Tripped")
            
            # NOTE 26 Interlock: ZSC-002 close confirmation automatically trips UZV-003
            # to prevent high pressure gas blow-by into low-pressure 2nd stage
            st.zsc002_closed = True
            st.uzv003_open = False
            st.tripped_causes.append("Note 26 Interlock: ZSC-002 closed -> Gas SDV (UZV-003) Tripped (Gas Blow-by Prevention)")
        
        # Pre-Alarms
        if st.level_mm >= LEVELS_MM["H"] and st.level_mm < LEVELS_MM["HH"]:
            st.active_alarms.append("LAH: High Liquid Level Pre-Alarm (> 2350 mm)")
        elif st.level_mm <= LEVELS_MM["L"] and st.level_mm > LEVELS_MM["LL"]:
            st.active_alarms.append("LAL: Low Liquid Level Pre-Alarm (< 1200 mm)")
        
        if st.tripped_causes:
            st.alarm_status = "TRIP"
        elif st.active_alarms:
            st.alarm_status = "ALARM"
        else:
            st.alarm_status = "NORMAL"
        
        # --- 6. Stokes' Law & Separation Kinetics ---
        # Stokes droplet settling velocity
        # Viscosity mu_o depends on crude API and temperature
        # Baseline: 28.5 API -> mu ~ 3.5 cP = 0.0035 Pa*s at 79 C
        api_factor = math.exp(0.08 * (28.5 - st.crude_api)) # heavier crude -> exponential viscosity increase
        mu_oil = 0.0035 * api_factor
        rho_water = self.case_data["water_rho"]
        rho_oil = self.case_data["oil_rho"] * (1.0 + 0.005 * (28.5 - st.crude_api))
        
        d_drop = 150e-6 # 150 um median water droplet diameter
        g = 9.81
        v_stokes = (g * d_drop * d_drop * (rho_water - rho_oil)) / (18.0 * mu_oil)
        
        # Richardson-Zaki hindered settling correction
        v_settle = v_stokes * ((1.0 - st.water_cut_vol) ** 4.65)
        st.stokes_settling_vel_mm_s = v_settle * 1000.0
        
        # Residence time in vessel
        t_res_s = (st.liquid_holdup_m3 / (st.feed_liquid_m3_h / 3600.0)) if st.feed_liquid_m3_h > 0 else 600.0
        
        # Water carryunder in export oil
        # Normal baseline ~ 250 ppm. Slower settling or shorter residence time inflates carryunder
        settling_ratio = (v_settle * t_res_s) / max(st.level_m, 0.5)
        st.water_in_oil_ppm = max(200.0, 250.0 / max(settling_ratio, 0.05))
        
        # Superficial gas velocity and Souders-Brown check
        a_gas = max(A_TOTAL - a_liq, 0.5)
        q_gas_actual_m3_s = (st.out_gas_header_kg_h + st.out_gas_flare_kg_h) / (self.case_data["gas_rho"] * 3600.0)
        st.gas_velocity_m_s = q_gas_actual_m3_s / a_gas
        
        rho_g = max((st.gas_holdup_kg / v_gas), 1.0)
        st.gas_v_max_m_s = K_SB * math.sqrt(max((rho_oil - rho_g) / rho_g, 1.0))
        
        if st.gas_velocity_m_s > st.gas_v_max_m_s:
            st.demister_flooded = True
            st.carryover_gal_mmscf = 0.15 + (st.gas_velocity_m_s - st.gas_v_max_m_s) * 0.25
        else:
            st.demister_flooded = False
            st.carryover_gal_mmscf = 0.03 * (st.gas_velocity_m_s / max(st.gas_v_max_m_s, 0.1))
        
        return st


# --- Preset Scenario Engine & Operator Grading Catalog ---

@dataclass
class ScenarioDefinition:
    tag: str
    title_en: str
    title_ar: str
    description_en: str
    description_ar: str
    initial_case: str
    init_func: Any
    win_condition_desc_en: str
    win_condition_desc_ar: str


def apply_scenario(sim: SeparatorDynamicSimulator, scenario_tag: str):
    """Inject disturbances and initialize the simulator for a specific training mission."""
    st = sim.state
    
    if scenario_tag == "SCN-01":
        # SCN-01: Normal 100 kBOPD Baseline
        sim._init_steady_baseline()
        sim.set_clamp_override(False)
        sim.valv_fv001.manual = False
        sim.valv_pv003b.manual = False
        sim.valv_pv003a.manual = False
        
    elif scenario_tag == "SCN-02":
        # SCN-02: 1.4x Liquid Surge Challenge
        sim._init_steady_baseline()
        sim.set_clamp_override(False)
        # Flow jumps to 1.4x surge rate (1184.47 m3/h)
        st.feed_liquid_m3_h = sim.case_data["liq_m3_h"]
        st.feed_gas_kg_h = sim.case_data["gas_kg_h"]
        
    elif scenario_tag == "SCN-03":
        # SCN-03: Liquid Valve FV-001 Stuck Closed
        sim._init_steady_baseline()
        sim.valv_fv001.manual = True
        sim.valv_fv001.target = 0.0
        sim.valv_fv001.position = 0.0
        
    elif scenario_tag == "SCN-04":
        # SCN-04: Gas Valve PV-003B Stuck Locked
        sim._init_steady_baseline()
        sim.valv_pv003b.manual = True
        # Lock at 30% (restricted)
        sim.valv_pv003b.target = 0.30
        sim.valv_pv003b.position = 0.30
        
    elif scenario_tag == "SCN-05":
        # SCN-05: Heavy Crude Transition (Emulsion)
        sim._init_steady_baseline()
        st.crude_api = 19.5 # Heavy crude, viscosity x3
        st.water_cut_vol = 0.28 # High water cut
        
    elif scenario_tag == "SCN-06":
        # SCN-06: Gas Blow-by Emergency (Note 26)
        sim._init_steady_baseline()
        st.feed_liquid_m3_h = 200.0 # Low feed
        sim.valv_fv001.manual = True
        sim.valv_fv001.target = 0.85 # Open outflow draining vessel
        sim.valv_fv001.position = 0.85
        
    elif scenario_tag == "SCN-07":
        # SCN-07: Free Sandbox (Chaos Mode)
        pass


@dataclass
class OperatorScore:
    """Evaluates trainee performance and computes official certification grade."""
    score: float = 1000.0
    flaring_penalty: float = 0.0
    alarm_penalty: float = 0.0
    trip_penalty: float = 0.0
    carryover_penalty: float = 0.0
    grade_tier: str = "⭐⭐⭐ Master Console Operator"
    grade_tier_ar: str = "⭐⭐⭐ مشغل غرفة تحكم رئيسي"
    passed: bool = True

    def calculate(self, sim: SeparatorDynamicSimulator, duration_s: float):
        st = sim.state
        # Flaring penalty: $50 per 1000 m3 (~700 kg) flared
        flared_kg = st.total_gas_flared_kg
        self.flaring_penalty = (flared_kg / 700.0) * 50.0
        
        # Alarms penalty
        if st.alarm_status == "ALARM":
            self.alarm_penalty += 50.0
        
        # Trip penalty: Automatic 500 pt loss or fail
        if st.tripped_causes:
            self.trip_penalty = 500.0
            self.passed = False
        
        # Quality carryover penalty
        if st.water_in_oil_ppm > 500.0 or st.carryover_gal_mmscf > 0.10:
            self.carryover_penalty = 100.0
        
        self.score = max(0.0, 1000.0 - (self.flaring_penalty + self.alarm_penalty + self.trip_penalty + self.carryover_penalty))
        
        if self.score >= 900.0 and self.passed:
            self.grade_tier = "⭐⭐⭐ Master Console Operator"
            self.grade_tier_ar = "⭐⭐⭐ مشغل غرفة تحكم رئيسي"
        elif self.score >= 750.0 and self.passed:
            self.grade_tier = "⭐⭐ Competent Operator"
            self.grade_tier_ar = "⭐⭐ مشغل مؤهل"
        elif self.score >= 600.0:
            self.grade_tier = "⭐ Junior Operator"
            self.grade_tier_ar = "⭐ مشغل مبتدئ"
        else:
            self.grade_tier = "❌ Re-training Required"
            self.grade_tier_ar = "❌ إعادة تدريب مطلوبة"
            self.passed = False
