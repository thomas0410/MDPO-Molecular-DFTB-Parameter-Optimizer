

# MDPO — Molecular DFTB Parameter Optimizer  
## User Manual

*(Under Active Development)*

---

## ⚠️ Important Notice

MDPO is **under active development**.

- Internal APIs, configuration formats, and supported targets may change.
- Backward compatibility is **not guaranteed**.
- Some features are experimental or partially implemented.

Users are **strongly advised to pin a specific commit** when using this framework for production calculations or publications.

---

## 🎯 Intended Scope and Target Users

MDPO is designed **for chemists and molecular scientists** working on:

- Molecular electronic structure
- Density-Functional Tight Binding (DFTB+)
- Slater–Koster parameter development
- Data-driven or optimization-based parameter fitting

This framework focuses on **molecular systems** and is **not intended** as a black-box production tool.

Users are expected to be familiar with:
- DFTB+ input/output structure
- Slater–Koster formalism
- Basic numerical optimization concepts

---

## 1. System and Software Requirements

### 1.1 Operating System

- Linux (recommended)
- macOS (limited testing)
- Windows is not officially supported (WSL may work)

---

### 1.2 External Dependency

- **DFTB+**
  - Version ≥ 22.x recommended
  - Must be accessible as a command-line executable (`dftb+`)
- OpenMP-capable runtime is recommended for parallel execution

---

### 1.3 Python Environment

- **Python ≥ 3.9** (3.10+ recommended)

Required Python packages:

```
numpy
scipy
pyyaml
tqdm
joblib
```

Optional (diagnostics / visualization):

```
matplotlib
```

---

## 2. Overview

**MDPO (Molecular DFTB Parameter Optimizer)** is a Python-based framework for optimizing **Slater–Koster files (SKFs)** used in **DFTB+ molecular calculations**.

The goal is to adjust selected SKF parameters so that DFTB+ predictions reproduce reference molecular properties such as:

- Ground-state total energies
- Low-lying excitation energies
- Frontier orbital gaps *(planned)*

DFTB+ is treated as a **black-box evaluator**, and no analytical gradients are required.

---

## 3. Optimization Strategy

MDPO employs **Simultaneous Perturbation Stochastic Approximation (SPSA)**, which is suitable for:

- High-dimensional parameter spaces
- Expensive electronic-structure evaluations
- Noisy objective functions (SCF / excited-state solvers)

Only **two DFTB+ evaluations per iteration** are required, independent of parameter dimension.

---

## 4. Configuration via `config.yaml`

MDPO is fully configured using a **YAML configuration file**.

All physics choices, optimization settings, dataset handling rules, and DFTB+ execution options are defined explicitly in this file.

---

### 4.1 Minimal Example

```yaml
targets: [energy, excitation]

pairs:
  - C-C
  - C-H
  - O-H

basis: poly
opt_scope: both
K: 2

dftb_timeout: 300
```

This is the **smallest valid configuration** that will run an optimization.

---

## 4.2 Target Selection

```yaml
targets: energy
# or
targets: [energy, excitation]
# or
targets: all
```

Supported targets:

| Target | Meaning | Status |
|------|--------|--------|
| `energy` | Ground-state total energy (S₀) | Implemented |
| `excitation` | Lowest Casida excitation | Implemented |
| `HLGap` | HOMO–LUMO gap | Placeholder |

---

## 4.3 Target Weights (Optional)

```yaml
weights:
  energy: 1.0
  excitation: 0.5
```

If omitted, all targets default to equal weight.

---

## 4.4 Slater–Koster Pairs

```yaml
pairs:
  - C-C
  - C-H
  - O-H
```

Notes:
- Only listed pairs are optimized.
- Directional symmetry (`A–B` ↔ `B–A`) is handled automatically.
- All required SKFs must exist in the `SKFs/` directory.

---

## 4.5 Correction Model

```yaml
basis: poly        # poly | bspline | sigmoid
opt_scope: both    # ham | repulsive | both
K: 2               # basis size / polynomial order
```

Optional smoothness regularization:

```yaml
smooth_lambda: 1.0e-3
```

---

## 4.6 Dataset Sampling Strategy

For small datasets, MDPO automatically disables sampling.

```yaml
auto_all_threshold: 100
```

For large datasets:

```yaml
permanent: "1,201,401"
strata: "1-200,201-400,401-600"
k_per_pool: 20
batch_P: 10
batch_F: 40
no_strata: false
```

Sampling is deterministic and reproducible.

---

## 4.7 Global Search (Optional)

```yaml
global_phase: none    # none | generic | asha
```

Random search:

```yaml
global_phase: generic
global_evals: 32
global_budget: 64
```

ASHA search:

```yaml
global_phase: asha
asha_eta: 3
asha_R: 256
```

---

## 5. Project Directory Structure (Inputs vs Generated Outputs)

MDPO expects you to run it from the **project root** (the directory that contains `SKFs/`, `data/`, and `ref.txt`).

### 5.1 Required inputs (you must create these)

```
project_root/
├── config.yaml              # (Input) YAML configuration file. Place it in the project root.
├── SKFs/                    # (Input) Original Slater–Koster files used by DFTB+
│   ├── C-C.skf
│   ├── C-H.skf
│   ├── O-H.skf
│   └── ...
├── data/                    # (Input) Molecular geometries (e.g., .xyz)
│   ├── mol1.xyz
│   ├── mol2.xyz
│   └── ...
└── ref.txt                  # (Input) Reference values for each geometry in `data/`
                             # Format:
                             # <geometry_file> <target1> <target2> ...
                             # Example (energy + excitation):
                             # mol1.xyz  -150.1234  3.50
                             # mol2.xyz  -160.5678  3.20
```

### 5.2 Auto-generated during execution (MDPO creates these)

After you run MDPO, it will create additional directories/files:

```
project_root/
├── templates/               # (Generated) DFTB+ input template directory
│   └── dftb_in.hsd          # (Generated) DFTB+ template used for all runs
├── runs/                    # (Generated) All runtime logs, checkpoints, and temporary DFTB+ work dirs
│   ├── skf_opt.log          # (Generated) Log file (name can be configured via `log_file`)
│   ├── spsa_ckpt.pkl        # (Generated) SPSA checkpoint (auto-saved)
│   ├── spsa_ckpt.STOP       # (Optional) If present, MDPO will stop gracefully after the current step
│   └── spsa_eval_*/         # (Generated) Temporary per-batch DFTB+ working directories
└── SKFs/
    └── optimized/           # (Generated) Final optimized SKF files (written at the end / on best checkpoint)
        ├── C-C.skf
        ├── C-H.skf
        └── ...
```

### Notes

- **Input files are never modified in place.** Your original `SKFs/*.skf` remain untouched.
- All DFTB+ executions happen under `runs/` (including temporary per-structure subdirectories).
- Optimized SKFs are written to **`SKFs/optimized/`** (this folder is created if missing).

## 6. Workflow Summary

1. Prepare initial SKF files
2. Define molecular reference data
3. Configure `config.yaml`
4. Run SPSA optimization
5. Export optimized SKFs and logs

---

## 7. Limitations

- HOMO–LUMO gap parsing is not fully implemented
- API and configuration format may change
- No automatic physical constraint enforcement

---

## 8. Disclaimer

This software is intended **for research and method development only**.

Optimized parameters may overfit specific datasets or behave poorly outside the training domain.  
**Independent validation is the responsibility of the user.**
