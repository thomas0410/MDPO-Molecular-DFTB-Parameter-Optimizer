# MDPO — Molecular DFTB Parameter Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![Status](https://img.shields.io/badge/status-active%20development-yellow.svg)]()
![DFTB](https://img.shields.io/badge/DFTB-Parameterization-orange.svg)

---

## 🚧 Development Status

> **MDPO is currently under active development.**  
> APIs, configuration formats, and supported features may change without notice.  
> Backward compatibility is **not guaranteed** at this stage.

For stable usage in research or publications, please **pin a specific commit**.

---

## 🌟 What is MDPO?

**MDPO (Molecular DFTB Parameter Optimizer)** is a research-oriented Python framework for **optimizing Slater–Koster files (SKFs)** used in **Density-Functional Tight Binding (DFTB+) molecular calculations**.

MDPO is designed **for chemists and ** who aim to:
- Calibrate DFTB parameters against reference molecular data
- Optimize ground- and excited-state properties
- Explore data-driven parameter refinement strategies

The framework focuses on **molecular systems** and treats DFTB+ as a **black-box electronic-structure evaluator**.

---

## 🔬 Supported Optimization Targets

Currently supported targets include:

- Ground-state total energy (S₀)
- Lowest excitation energy (Casida)

Planned / experimental targets:
- HOMO–LUMO gap
- Geometry-derived properties
- Vibrational frequencies
- Potential energy surfaces (PES)

---

## ⚙️ Key Features

- YAML-driven configuration (no complex CLI flags)
- Gradient-free stochastic optimization (SPSA)
- Multi-target molecular fitting
- Robust Slater–Koster file parsing and rewriting
- Automatic DFTB+ input generation
- Checkpointing and restart support for long optimizations

---

## 🧩 Requirements

### Software
- **Python ≥ 3.9** (3.10+ recommended)
- **DFTB+** (≥ 22.x recommended, must be accessible as `dftb+`)

### Python Packages
```
numpy
scipy
pyyaml
tqdm
joblib
```

---

## 📁 Project Structure (Summary)

MDPO is executed from a **project root directory** containing user inputs.  
Additional directories are created automatically during execution.

```
project_root/
├── config.yaml      # User-defined configuration (input)
├── SKFs/            # Original Slater–Koster files (input)
├── data/            # Molecular geometries (input)
├── ref.txt          # Reference values (input)
├── templates/       # Generated DFTB+ templates (auto)
├── runs/            # Logs, checkpoints, temporary runs (auto)
└── SKFs/optimized/  # Optimized SKFs (auto)
```

Full details are provided in the User Manual.

---

## ▶️ How to Get Started

1. Prepare your SKF files, molecular geometries, and reference data
2. Create a `config.yaml` in the project root
3. Run MDPO from the project root directory

📘 **Detailed instructions and configuration reference:**  
➡️ [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md)

---

## 📚 Background and References

MDPO interfaces with [**DFTB+**](https://dftbplus.org/), an open-source software package for approximate density functional theory simulations.

> Hourahine, B., Aradi, B., Blum, V., et al.  
> *DFTB+, a software package for efficient approximate density functional theory based atomistic simulations.*  
> J. Chem. Phys. 152, 124101 (2020).

This project is **conceptually inspired** by:

> Liu, C., Aguirre, N. F., Cawkwell, M. J., Batista, E. R., & Yang, P. (2024).  
> *Efficient parameterization of density functional tight-binding for 5f-elements: A Th–O case study.*  
> *J. Chem. Theory Comput.*, 20(14), 5923–5936.

The implementation, datasets, and workflow in this repository are **independent**.

---

## 📖 Citation

If you use MDPO in your research, please cite:

```bibtex
@misc{MDPO2025,
  author       = {Tsai, Chi-Chang},
  title        = {MDPO: Molecular DFTB Parameter Optimizer},
  year         = {2025},
  howpublished = {\\url{https://github.com/thomas0410/MDPO-Molecular-DFTB-Parameter-Optimizer}},
  note         = {Research framework for molecular DFTB parameter optimization}
}
```

---

## ⚠️ Disclaimer

MDPO is intended **for research and method development only**.  
Optimized parameters must be independently validated before scientific or practical use.
