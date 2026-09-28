# Stage 1 Production Separator Simulator (`CP2-V-71101 / CP2-V-71201`)

Interactive educational simulator for the 1st Stage Production Separator at Majnoon CPF-2 (Basrah Oil Company, Train 1 100 kBOPD).

---

## 1. Quick Start

### Option A: One-Click Batch Script (Windows)
Double-click [`start.bat`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/start.bat) or run in terminal:
```cmd
start.bat
```

### Option B: Command Line (PowerShell / CMD)
Ensure dependencies are installed:
```powershell
pip install -r ../../requirements.txt
```
Run the Streamlit application:
```powershell
streamlit run app.py
```

### Run Model Verification, Sensitivity & Dynamic Tests
```powershell
python model.py             # 18 design checks, envelope, and K factor
python check_sensitivity.py # 54 physical perturbations across Case 1 & Case 4
python test_dynamic.py      # Comprehensive dynamic tests (Two-rate ODE, 1.4x surge, interlocks, Note 26)
```

---

## 2. File Overview

| File | Purpose |
|---|---|
| [`dynamic.py`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/dynamic.py) | **Dynamic Physics & OTS Engine**: Stiff Two-Rate ODE solver ($0.2\text{s}$ liquid main step, $0.02\text{s}$ gas sub-steps), Koso valves ($C_v = 2140$ linear, $3990$ mod-eq%, $1170$ eq%), discrete PI controllers with anti-reset windup, Stokes & Richardson-Zaki droplet settling, and Cause & Effect interlocks (Note 26). |
| [`twin3d.py`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/twin3d.py) | **Interactive 3D Digital Twin (Three.js/WebGL)**: Cutaway vessel, Sulzer internals (Schoepentoeter, Mellachevron, Baffle, Demister, Vortex Breaker), dynamic dual-phase fluid layers (oil/water), Stokes settling particles, and raycast inspection cards. |
| [`test_dynamic.py`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/test_dynamic.py) | **Dynamic Validation Suite**: Validates 1-hour mass balance drift (<0.01%), 1.4x surge challenge response (~2 mm/s, ~9.6 min to LAHH), 14s valve slew rates, split-range flare venting, and Note 26 gas blow-by prevention. |
| [`model.py`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/model.py) | **Physics Engine (Steady-State)**: Standard-library only. Contains vessel geometry ($4.2\text{ m ID} \times 13.6\text{ m TT}$), Sulzer F02 design cases, phase split calculations, residence time, and Souders-Brown $K=0.150\text{ m/s}$ gas velocity checks. |
| [`app.py`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/app.py) | **Operator HMI & Training Console**: Streamlit app with dual modes (🎮 Dynamic OTS & 📋 Steady-State), 3D Digital Twin vs 2D DCS P&ID console toggle, 7 operational scenarios, live KPI grading, and bilingual concept explainers. |
| [`check_sensitivity.py`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/check_sensitivity.py) | **Automated Validation Suite**: Executes 54 perturbations across pressure, temperature, water cut, and level to guarantee zero numerical collapse and 100% physically valid outputs. |
| [`start.bat`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/start.bat) | Windows launcher script for instant browser startup. |
| [`assets/`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/assets/) | Vendored Three.js, OrbitControls, and high-resolution P&ID assets for 100% offline operation. |
| [`.streamlit/config.toml`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/simulator/stage1/.streamlit/config.toml) | UI configuration and styling. |

---

## 3. Engineering Source Basis

All equations, dimensions, volumes, and alarm thresholds strictly adhere to the locked design document:
- Design Specification: [`docs/plans/stage1-separator/DESIGN.md`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/docs/plans/stage1-separator/DESIGN.md)
- Development Roadmap: [`docs/plans/stage1-separator/ROADMAP.md`](file:///c:/Users/Almahdi-BOC/Documents/Projects/3-phase-sperator/docs/plans/stage1-separator/ROADMAP.md)
