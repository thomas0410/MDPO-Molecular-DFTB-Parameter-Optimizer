# MDPO — Molecular DFTB Parameter Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
![DFTB](https://img.shields.io/badge/DFTB-Parameterization-orange.svg)


---

## 🌟 Overview

**MDPO (Molecular DFTB Parameter Optimizer)** is a **lightweight and automated Python tool** designed to optimize molecular properties through **DFTB parameter calibration**.  
It enables researchers to efficiently tune, validate, and test DFTB parameters on molecular systems using automated workflows and modern optimization algorithms.

---

## ⚙️ Features

- 🔹 Automated DFTB parameter optimization workflow  
- 🔹 Support for molecular systems and dataset-level fitting  
- 🔹 Lightweight, modular structure for research and educational use  
- 🔹 Easy to customize input/output routines for different molecular datasets  
---

## 🧩 Requirements and Environment Setup

### 🐍 Python Environment
This project is written in **Python 3.4+** and relies primarily on the Python standard library and **NumPy**.

The following Python modules are required:
```python
from __future__ import annotations
import argparse, os, re, json, subprocess, time, pickle, tempfile, shutil, concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence, Optional, Callable, Dict, Any, Set
import numpy as np
```
---

### ⚙️ DFTB+ Dependency
**MDPO** serves as an external Python-based optimizer that interfaces with [**DFTB+**](https://dftbplus.org/),  
a well-established open-source software package for performing self-consistent **Density Functional Tight Binding (DFTB)** simulations.

To run this tool successfully, make sure that:
1. **DFTB+** is installed and accessible from the command line (i.e., the command `dftb+` works in your terminal).  
   Installation guide: [https://dftbplus.org/download/](https://dftbplus.org/download/)
2. You have appropriate **Slater–Koster parameter sets** (e.g., mio, 3ob, or your own customized files).  
   MDPO automatically calls DFTB+ for single-point calculations and extracts total energies, dipoles, and other target properties.

---

### 🧠 Recommended System Setup
- **Operating System:** Linux or macOS  
- **Python:** 3.4 or higher  
---

### 🔬 References
- **DFTB+**: Aradi, B., Hourahine, B., & Frauenheim, T. (2007).  
  *DFTB+, a sparse matrix-based implementation of the DFTB method.*  
  *Journal of Physical Chemistry A*, 111(26), 5678–5684.  
  [https://doi.org/10.1021/jp070186p](https://doi.org/10.1021/jp070186p)

The MDPO code independently provides an automated interface for molecular DFTB parameter tuning  
and can be combined with existing DFTB+ workflows for property-driven optimization.

---

## 📚 Citation and Acknowledgement

This project was **inspired by** and conceptually related to the methodology proposed in the following publication:

> Liu, C., Aguirre, N. F., Cawkwell, M. J., Batista, E. R., & Yang, P. (2024).  
> *Efficient parameterization of density functional tight-binding for 5f-elements: A Th–O case study.*  
> *Journal of Chemical Theory and Computation*, 20(14), 5923–5936.  
> https://doi.org/10.1021/acs.jctc.4c00123

The ideas in this repository were developed **independently**, drawing general inspiration from the above study but not directly reproducing its implementation or datasets.  
The code and workflow presented here were written by **Tsai, Chi-Chang (Thomas)** for research and educational purposes.

If you use this repository or part of its code in your research, please cite the paper above and acknowledge this repository as follows:

```bibtex
@misc{MDPO2025,
  author       = {Tsai, Chi-Chang},
  title        = {MDPO: Molecular DFTB Parameter Optimizer},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-username>/MDPO}},
  note         = {Conceptually inspired by Liu et al., J. Chem. Theory Comput., 2024.}
}
