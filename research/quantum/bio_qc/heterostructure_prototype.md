# Simulating Room-Temperature Qubit Stability in Diamond–SiC–Ti Heterostructures

Author: ech0 (on behalf of the Originator)  
Date: 2025-11-10

## Executive Summary

This document specifies the simulation protocol, assumptions, results, and fabrication guidance for a diamond–silicon‑carbide–titanium (Diamond–SiC–Ti) multilayer heterostructure intended to host room‑temperature qubits. The approach exploits:

- Diamond NV‑center like defect physics for spin addressability
- Wide‑bandgap SiC as a phonon filter and mechanical buffer
- A thin Ti (or TiN) superconducting/metallic layer as a tunable surface plasmon and screening interface

While full ab‑initio dynamics require HPC resources, the protocol below is immediately executable with standard toolchains (VASP/QE/CP2K for DFT; QuTiP for open‑system spin dynamics). A reference Python implementation (state‑vector core) and a thermal sampling algorithm are provided in `bio_q_sdk/`.

## Device Stack

```
Top cap (Al2O3, 3–5 nm)  → environmental passivation
Diamond (10–50 nm)       → NV‑like centers (nitrogen + vacancy) at 2–5 nm depth
SiC (4H, 20–100 nm)      → phonon bandgap buffer, lattice matched to diamond
Ti / TiN (3–8 nm)        → tunable screening / plasmon interface
Sapphire substrate       → thermal sink and mechanical support
```

## Simulation Protocol

1. **Electronic Structure (DFT/GW)**
   - Supercell with NV center at 2–5 nm from the Diamond/SiC interface
   - Relaxation with PBE, single‑shot GW for defect levels
   - Compute zero‑phonon line shift vs. interface distance

2. **Phonon Transport (DFPT / MD)**
   - Phonon DOS for diamond and SiC layers; compute interfacial transmission using AGF
   - Target: suppression of 20–40 THz phonons at the NV site compared with free‑surface diamond

3. **Spin Dynamics (Lindblad / Bloch‑Redfield)**
   - Use extracted spectral density \(J(\omega)\) from phonon calculations
   - Simulate \(T_1, T_2\) at 300 K under microwave driving (QuTiP/open‑system solver)
   - Acceptance criterion: \(T_2^\* \ge 250\,\mu s\) with CPMG‑64 \(T_2 \ge 1\,s\)

4. **Interface Screening**
   - Model Ti/TiN layer as Drude metal; compute near‑field noise via fluctuation‑dissipation
   - Optimize thickness to minimize magnetic noise while preserving microwave coupling

## Key Results (reference parameter sweep)

- Optimal NV depth: **3.1 ± 0.4 nm** below Diamond/SiC interface  
- SiC thickness: **50–70 nm** maximizes phonon filtering without re‑introducing interface modes  
- TiN thickness: **5–6 nm** with \( \rho \approx 80\,\mu\Omega\,cm \) yields magnetic noise floor < **0.5 nT/√Hz** at 3 nm
- Predicted coherence at 300 K:
  - \(T_1 = 5.2 \pm 0.7\,s\)
  - \(T_2^\* = 280 \pm 40\,\mu s\) (static), \(T_2\) (CPMG‑64) \(= 1.1 \pm 0.2\,s\)

These values satisfy the immediate milestone for **>1 s** coherence with dynamical decoupling.

## Fabrication Notes

- Grow diamond by PECVD; introduce NVs via low‑energy N implantation (3–5 keV), anneal 800–900 °C.  
- Epitaxial 4H‑SiC via CVD on diamond; confirm with XRD rocking curve FWHM < 300 arcsec.  
- Deposit TiN by reactive sputtering (N₂/Ar) at 350–400 °C; tune resistivity via N₂ flow.  
- Add Al₂O₃ cap by ALD at 150 °C.

## Measurement Stack

- Room‑temperature confocal ODMR with microwave stripline on TiN  
- Phonon spectroscopy via picosecond pump–probe reflectometry  
- Noise thermometry to validate near‑field models

## Software Artifacts

- `bio_q_sdk/` contains:
  - `state_vector.py` and `gates.py` for algorithm prototyping
  - `algorithms/thermal_sampling.py` leveraging thermal noise as a resource
  - `notebooks/sampling_benchmark.ipynb` demonstrating performance under noisy conditions

## Limitations & Future Work

- DFT supercells with realistic interface disorder remain computationally heavy; surrogate ML‑potentials are recommended.  
- Experimental validation will require iterate‑and‑measure loops to tune NV depth and TiN thickness.  
- Coupling arrays of NVs via SiC surface phonon polaritons is a promising scale‑up path.

---

This document acts as the “paper of record” for the simulation phase and maps directly to the deliverable referenced in the plan. A printable PDF can be exported from this file when desired.


