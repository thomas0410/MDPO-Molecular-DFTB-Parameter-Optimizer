#!/usr/bin/env python3
"""
用於優化 Slater-Koster 檔案的 Python 腳本 ( Slater-Koster File Optimizer using SPSA)。

此腳本根據指定的專案目錄結構進行了重構，使其能夠：
 * 從 `ref.txt` 讀取參考結構和能量（或其他物理量）數據。
 * 讀取 `SKFs/` 目錄下的初始 Slater-Koster 檔案，並分別解析 H/S 表格與 Repulsive 部分。
 * 在 `runs/` 目錄下執行所有計算。
 * 對 SK 檔案中的哈密頓積分或是Repulsive 部分應用修正，並能完整重構檔案。
 * 使用 SPSA (Simultaneous Perturbation Stochastic Approximation) 演算法進行優化。
 * 支援多目標加權優化 (例如：基態能量 + 激發能)。
 * 支援多種基底來建構修正函數 (Polynomial / B-spline / Sigmoid)。
"""

# =============================================================================
# Program Usage (Default Settings)
# =============================================================================
#
# 預設設定：
#   --targets both
#       → 同時優化 基態能量 (S0) 與 激發能 (Excitation energy)。
#       → `ref.txt` 每行需包含兩個數值：S0 以及 Excitation energy。
#
#   --basis poly
#       → 修正函數 g(r) 使用 Polynomial 展開。
#       → 最高次數由 --K 控制 (預設 K=2)。
#
#   --opt-scope both
#       → 同時優化 Hamiltonian (H) 和 Repulsive potential。
#
# =============================================================================
# Table of Contents
# =============================================================================
# [1] Imports & Global Dependencies
# [2] Core Data Structures & SKF File I/O
# [3] Basis Functions & Correction Model
# [4] SPSA Training Algorithm & Utilities
# [5] DFTB+ Calculation & Objective Function
# [6] Dataset Loading & Utilities
# [7] Main Entry Point
# =============================================================================

# =============================================================================
# [1] Imports & Global Dependencies
# =============================================================================
from __future__ import annotations
import argparse, os, re, json, subprocess, time, pickle, tempfile, shutil, concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence, Optional, Callable, Dict, Any, Set
import numpy as np
import logging

# =============================================================================
# Logging setup
# =============================================================================
def setup_logger(log_path: str) -> logging.Logger:
    """Create a module-level logger that writes to both console and a file (PBS-tail friendly)."""
    logger = logging.getLogger("skf_opt")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicates on resume
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # 在建立日誌處理程式之前，確保日誌檔案的目錄存在
    try:
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create log directory {log_dir}. Error: {e}")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # try to flush immediately (useful on PBS)
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    return logger
    
import fcntl  # Add to imports at top

def _save_ckpt(state: Dict[str, Any], ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ckpt_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'wb') as fo:
            # Lock file during write
            try:
                fcntl.flock(fo.fileno(), fcntl.LOCK_EX)
            except (AttributeError, OSError):
                # fcntl not available on Windows, skip locking
                pass
            pickle.dump(state, fo)
        # Atomic rename
        temp_path.replace(ckpt_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e

# =============================================================================
# [2] Core Data Structures & SKF File I/O
# =============================================================================

@dataclass
class ParsedSKFile:
    """
    一個用於解析、儲存和寫入 Slater-Koster 檔案 (.skf) 的資料類別。
    這個類別封裝了所有與 SKF 檔案格式相關的 I/O 邏輯。
    """
    filepath: str
    header_lines: List[str]
    trailing_lines: List[str]
    grid_dist: float
    radii: List[float]
    H: Dict[str, List[float]]
    S: Dict[str, List[float]]
    repulsive: Dict[str, Any]
    num_h_cols: int
    rep_in_header: bool = False
    rep_rel_start: int = -1
    rep_rel_end: int = -1
    rep_mass: Optional[float] = None  # <-- 【新增】儲存 mass 欄位
    
    # --- Private Helper Methods (formerly standalone functions) ---
    
    @staticmethod
    def _expand_tokens(line: str) -> List[float]:
        """處理 SKF 的速記法，例如 '3*0.0' -> [0.0, 0.0, 0.0]"""
        out = []
        for tok in re.split(r'[,\s]+', line.strip()):
            if not tok: continue
            m = re.fullmatch(r'(\d+)\*(\-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)', tok)
            out.extend([float(m.group(2))] * int(m.group(1)) if m else [float(tok)])
        return out

    @classmethod
    def _parse_polynomial_repulsive(cls, line: str) -> Optional[Dict[str, Any]]:
        """
        [新函式] 僅嘗試從「單一字串行」中解析 Polynomial Repulsive 資料。
        """
        try:
            tokens = line.strip().split()
            # 根據 ，至少要有 mass + c2-c9 (8) + rcut (1) = 10 個欄位
            if len(tokens) >= 10:
                nums = cls._expand_tokens(line)
                if len(nums) >= 10:
                    mass = nums[0]  # <-- 【修正】保存 mass 欄位
                    coeffs = nums[1:] # c2 在 coeffs[0], rcut 在 coeffs[8]
                    rep_data = {
                        'type': 'polynomial',
                        'mass': mass,
                        'c': coeffs[:8], # 抓取 c2 到 c9 (共 8 個)
                        'rcut': coeffs[8] if len(coeffs) >= 9 else 0.0
                    }
                    return rep_data
        except (ValueError, IndexError):
            return None # 解析失敗
        return None

    @classmethod
    def _parse_spline_repulsive(cls, lines: List[str]) -> Tuple[Optional[Dict], int, int]:
        """
        [新函式] 僅在給定的行列表（例如 trailing_lines）中搜尋「Spline」區塊。
        """
        text_lines = [ln.strip() for ln in lines]
        for i, ln in enumerate(text_lines):
            if ln.lower().startswith("spline"):
                try:
                    nInt, cutoff = cls._expand_tokens(text_lines[i+1])
                    nInt, cutoff = int(nInt), float(cutoff)
                    a1, a2, a3 = cls._expand_tokens(text_lines[i+2])
                    segs = []
                    # 解析 nInt-1 個標準 spline 段
                    for j in range(nInt-1):
                        s, e, c0, c1, c2, c3 = cls._expand_tokens(text_lines[i+3+j])
                        segs.append({'start': s, 'end': e, 'c': [c0,c1,c2,c3]})
                    # 解析最後一個 spline 段 (格式不同)
                    last = cls._expand_tokens(text_lines[i+3+(nInt-1)])
                    s, e, c0, c1, c2, c3, c4, c5 = last
                    segs.append({'start': s, 'end': e, 'c': [c0,c1,c2,c3,c4,c5]})
                    
                    rep_data = {'type': 'spline', 'cutoff': cutoff, 'nInt': nInt, 'exp': (a1,a2,a3), 'segments': segs}
                    block_len = 3 + nInt
                    # 回傳 data 和在 lines 中的「相對」位置
                    return rep_data, i, i + block_len
                except (ValueError, IndexError) as e:
                    # 如果找到了 "Spline" 關鍵字但解析失敗，這代表檔案格式錯誤
                    raise ValueError(f"Found 'Spline' keyword but failed to parse block: {e}")
        
        return None, -1, -1 # 沒找到 Spline

    def _format_repulsive_block(self, rep: Dict[str, Any]) -> List[str]:
        """將 Repulsive 資料字典格式化為要寫入檔案的字串列表"""
        lines = []
        if rep.get('type') == 'spline':
            nInt, cutoff = int(rep['nInt']), float(rep['cutoff'])
            a1, a2, a3 = rep['exp']
            lines.extend(["Spline\n", f"{nInt:d} {cutoff:.6f}\n", f"{a1:.6f} {a2:.6f} {a3:.6f}\n"])
            
            segs = rep['segments']
            for seg in segs[:-1]:
                s, e, c = seg['start'], seg['end'], seg['c']
                lines.append(f"{s:.6f} {e:.6f} " + " ".join(f"{ci:.6e}" for ci in c[:4]) + "\n")
            
            last = segs[-1]
            s, e, c = last['start'], last['end'], last['c']
            lines.append(f"{s:.6f} {e:.6f} " + " ".join(f"{ci:.6e}" for ci in c[:6]) + "\n")
        elif rep.get('type') == 'polynomial':
            mass = rep.get('mass', self.rep_mass or 0.0) # 獲取儲存的 mass
            c2_to_c9, rcut = rep['c'], rep.get('rcut', 0.0)
            
            # 根據 ，格式為 mass c2..c9 rcut d1..d10
            # 我們補上 10 個 0.0 作為 d1-d10 的佔位符
            placeholders = " ".join(f"{0.0:.6e}" for _ in range(10))
            
            coeffs = " ".join(f"{v:.6e}" for v in ([mass] + list(c2_to_c9) + [rcut]))
            lines.append(f"{coeffs} {placeholders}\n")
        return lines

    # --- Public API Methods ---

    @classmethod
    def from_file(cls, filepath: str) -> "ParsedSKFile":
        """
        [工廠方法] 從一個 .skf 檔案路徑讀取並解析資料，
        然後回傳一個 ParsedSKFile 物件實例。
        
        此函式已重構，可正確處理 Simple/Extended 格式  
        並在正確位置解析 Polynomial  或 Spline。
        """
        path = Path(filepath)
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        if not lines: raise ValueError(f"File {filepath} is empty")

        pair_name = path.stem
        elements = pair_name.split('-')
        is_homo = len(elements) == 2 and elements[0] == elements[1]

        # --- 1. 偵測格式並設定索引 ---
        # 檢查第一行是否有 '@' 來判斷是否為 "Extended Format"
        is_extended = lines[0].strip().startswith('@')
        
        # 根據格式規範，決定標頭的行數和 Poly Repulsive 所在行
        poly_rep_line_idx = -1
        if is_homo:
            if is_extended:
                line_1_idx = 1 # gridDist 在 Line 2 (index 1)
                start_idx = 4  # 表格從 Line 5 (index 4) 開始
                poly_rep_line_idx = 3 # Poly 在 Line 4 (index 3)
            else: # homo-simple
                line_1_idx = 0 # gridDist 在 Line 1 (index 0)
                start_idx = 3  # 表格從 Line 4 (index 3) 開始
                poly_rep_line_idx = 2 # Poly 在 Line 3 (index 2)
        else: # is_hetero
            if is_extended:
                line_1_idx = 1 # gridDist 在 Line 3 (index 2)
                start_idx = 3  # 表格從 Line 4 (index 3) 開始
                poly_rep_line_idx = 2 # Poly 在 Line 2 (index 1)
            else: # hetero-simple
                line_1_idx = 0 # gridDist 在 Line 1 (index 0)
                start_idx = 2  # 表格從 Line 3 (index 2) 開始
                poly_rep_line_idx = 1 # Poly 在 Line 2 (index 1)

        # --- 2. 解析標頭和 Polynomial Repulsive ---
        first = cls._expand_tokens(lines[line_1_idx])
        grid_dist, n_grid = float(first[0]), int(first[1])
        n_rows = n_grid - 1
        table_end_idx = start_idx + n_rows
        if table_end_idx > len(lines): raise ValueError(f"Insufficient lines in {filepath}")

        repulsive_data = None
        rep_mass = None
        self_rep_in_header = False
        rep_rel_start = -1
        rep_rel_end = -1
        
        # 僅在規範指定的位置嘗試解析 Polynomial
        if poly_rep_line_idx != -1:
            repulsive_data = cls._parse_polynomial_repulsive(lines[poly_rep_line_idx])
        
        if repulsive_data:
            # 如果成功解析，從 header_lines 中「移除」該行
            header_lines = lines[:poly_rep_line_idx] + lines[poly_rep_line_idx+1:start_idx]
            rep_mass = repulsive_data.get('mass')
            self_rep_in_header = True
            rep_rel_start = poly_rep_line_idx # 記錄「絕對」位置
            rep_rel_end = poly_rep_line_idx + 1
        else:
            # 未找到 Poly，標頭保持原樣
            header_lines = lines[:start_idx]

        # --- 3. 解析 H 和 S 表格 ---
        table_lines = lines[start_idx:table_end_idx]
        if not table_lines: raise ValueError(f"No table data in {filepath}")
        num_cols = len(cls._expand_tokens(table_lines[0]))
        if num_cols == 0 or num_cols % 2 != 0: raise ValueError(f"Invalid column count: {num_cols}")
        num_h_cols = num_cols // 2
        table_rows = [cls._expand_tokens(ln) for ln in table_lines]
        H_cols = list(zip(*[row[:num_h_cols] for row in table_rows]))
        S_cols = list(zip(*[row[num_h_cols:] for row in table_rows]))
        H = {f"H_{i}": list(col) for i, col in enumerate(H_cols)}
        S = {f"S_{i}": list(col) for i, col in enumerate(S_cols)}
        radii = [grid_dist * (i + 1) for i in range(n_rows)]

        # --- 4. 解析 Spline Repulsive (僅在未找到 Poly 時) ---
        trailing_lines = lines[table_end_idx:]
        
        if repulsive_data is None:
            # 規範說 Spline 是可選的 ，在表格之後
            # 我們只在 trailing_lines 中尋找 Spline
            spline_data, rel_start, rel_end = cls._parse_spline_repulsive(trailing_lines)
            
            if spline_data:
                repulsive_data = spline_data
                self_rep_in_header = False
                rep_rel_start = rel_start # 相對於 trailing_lines 的位置
                rep_rel_end = rel_end
                # 從 trailing_lines 中「移除」Spline 區塊
                trailing_lines = trailing_lines[:rel_start] + trailing_lines[rel_end:]
        
        # 如果 Poly 和 Spline 都沒找到，repulsive_data 保持為 None
        if repulsive_data is None:
            repulsive_data = {} # 設為空字典，表示沒有 Repulsive
            self_rep_in_header = False
            rep_rel_start = 0 # 預設插入到 trailing_lines 的開頭
            rep_rel_end = 0

        return cls(
            filepath=filepath, header_lines=header_lines, trailing_lines=trailing_lines,
            grid_dist=grid_dist, radii=radii, H=H, S=S, repulsive=repulsive_data,
            num_h_cols=num_h_cols, rep_in_header=self_rep_in_header,
            rep_rel_start=rep_rel_start, rep_rel_end=rep_rel_end,
            rep_mass=rep_mass # 儲存 mass
        )

    def write(self, outpath: str, modified_H: Dict[str, List[float]]) -> None:
        """
        [實例方法] 將物件內儲存的 SKF 資料（可選擇傳入修改過的 H 和 Repulsive）
        寫入到指定的檔案路徑。
        """
        h_keys = [f"H_{i}" for i in range(self.num_h_cols)]
        s_keys = [f"S_{i}" for i in range(self.num_h_cols)]
        h_cols_modified = [modified_H.get(key, self.H[key]) for key in h_keys]
        s_cols_original = [self.S[key] for key in s_keys]
        h_rows = list(zip(*h_cols_modified))
        s_rows = list(zip(*s_cols_original))
        table_lines_new = [" ".join(f"{val:.12e}" for val in h_row + s_row) + "\n" for h_row, s_row in zip(h_rows, s_rows)]
        header_lines, trailing_lines = list(self.header_lines), list(self.trailing_lines)
        # 檢查是否有 Repulsive 需要被覆寫
        rep_override = modified_H.get("__REPULSIVE__")
        current_rep = self.repulsive if rep_override is None else rep_override
        new_block = self._format_repulsive_block(current_rep)
        if self.rep_in_header:
            header_lines = header_lines[:self.rep_rel_start] + new_block + header_lines[self.rep_rel_start:]
        else:
            trailing_lines = trailing_lines[:self.rep_rel_start] + new_block + trailing_lines[self.rep_rel_start:]
        with open(outpath, "w", encoding='utf-8') as f:
            f.writelines(header_lines + table_lines_new + trailing_lines)

# =============================================================================
# [3] Basis Functions & Correction Model
# =============================================================================

@dataclass
class CorrectionConfig:
    """用於修正模型的設定資料類別。"""
    basis: str
    opt_scope: str
    K: int
    bspline_degree: int
    smooth_lambda: float

def map_to_unit(r: np.ndarray) -> np.ndarray:
    """Map radius array to [0,1]"""
    r = np.asarray(r, dtype=float)
    rmin, rmax = float(np.min(r)), float(np.max(r))
    return np.zeros_like(r) if rmax == rmin else (r - rmin) / (rmax - rmin)

def legendre_vander_01(xi: np.ndarray, K: int) -> np.ndarray:
    """Legendre polynomial Vandermonde matrix on [0,1]"""
    x = 2.0*xi - 1.0
    V = np.empty((xi.size, K+1), dtype=float)
    V[:,0] = 1.0
    if K >= 1: V[:,1] = x
    for n in range(1, K):
        V[:,n+1] = ((2*n+1)*x*V[:,n] - n*V[:,n-1])/(n+1)
    return V

def bspline_basis_matrix(x: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """B-spline basis matrix"""
    x = np.asarray(x, dtype=float)
    a, b = float(x[0]), float(x[-1])
    p = degree
    m = n_basis + p + 1
    n_internal = m - 2*(p+1)
    if n_internal < 0: raise ValueError("n_basis too small for degree")
    
    t_internal = np.linspace(a, b, n_internal+2)[1:-1] if n_internal > 0 else np.array([], dtype=float)
    t = np.concatenate([np.full(p+1, a), t_internal, np.full(p+1, b)])
    
    B = np.zeros((x.size, n_basis + p))
    for i in range(n_basis + p):
        if i == n_basis + p - 1:
            B[:, i] = ((x >= t[i]) & (x <= t[i+1])).astype(float)
        else:
            B[:, i] = ((x >= t[i]) & (x < t[i+1])).astype(float)
    
    for k in range(1, p + 1):
        for i in range(n_basis + p - k):
            d1, d2 = t[i+k] - t[i], t[i+k+1] - t[i+1]
            term1 = ((x - t[i]) / d1) * B[:, i] if d1 > 1e-9 else 0
            term2 = ((t[i+k+1] - x) / d2) * B[:, i+1] if d2 > 1e-9 else 0
            B[:, i] = term1 + term2
    return B[:, :n_basis]

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def softplus(z): return np.where(z > 20, z, np.log1p(np.exp(z)))

class TabulatedModel:

    """
    管理 SKF 修正模型的類別。

    本類別是連接優化參數 `theta` 與物理 SKF 表格的橋樑。它接收一組 `theta` 參數，
    並根據指定的基底函數 (如 Polynomial, B-spline) 建構出修正函數 g(r)，
    最終生成一套修正後的哈密頓量 (H) 與排斥位能 (Repulsive) 資料，
    用於後續的 DFTB+ 計算。
    """
    
    def __init__(self, sk_data_map: Dict[str, ParsedSKFile], theta_mapping: Dict[str, str], cfg: CorrectionConfig):
        self.sk_data_map, self.theta_mapping, self.cfg = sk_data_map, theta_mapping, cfg
        self.param_keys = sorted(list(set(theta_mapping.values())))
        self.M = len(self.param_keys)
        
        first_sk = next(iter(sk_data_map.values()))
        self.r = np.asarray(first_sk.radii, dtype=float).copy()
        self.N, self.xi = self.r.size, map_to_unit(self.r)
        self.Phi = self._build_basis_matrix()
        
        self.do_ham = self.cfg.opt_scope in ("ham", "both")
        self.do_rep = self.cfg.opt_scope in ("repulsive", "both")
        
        if self.cfg.basis in ["poly", "bspline"]:
            self.ham_theta_len = int(self.Phi.shape[1])
        elif self.cfg.basis == "sigmoid":
            self.ham_theta_len = 3 * int(self.cfg.K)
        else:
            raise ValueError(f"Unknown basis: {self.cfg.basis}")
        
        self.rep_theta_len = 4
    
    def _build_basis_matrix(self):
        if self.cfg.basis == "poly":
            Phi = legendre_vander_01(self.xi, self.cfg.K)
        elif self.cfg.basis == "bspline":
            Phi = bspline_basis_matrix(self.xi, n_basis=self.cfg.K, degree=self.cfg.bspline_degree)
        elif self.cfg.basis == "sigmoid":
            return None
        else:
            raise ValueError(f"Unknown basis: {self.cfg.basis}")
        
        s = np.std(Phi, axis=0)
        s[s == 0] = 1.0
        return Phi * (1.0 / s)
    
    def theta_shape(self) -> Tuple[int, ...]:
        total = 0
        if self.do_ham: total += self.ham_theta_len
        if self.do_rep: total += self.rep_theta_len
        if total <= 0: raise ValueError("No optimization targets")
        return (total,)
    
    def _split_theta(self, theta: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.do_ham and self.do_rep:
            return theta[:self.ham_theta_len], theta[self.ham_theta_len:self.ham_theta_len + self.rep_theta_len]
        elif self.do_ham:
            return theta, None
        elif self.do_rep:
            return None, theta
        return None, None
    
    def _log_fc_one(self, theta: np.ndarray) -> np.ndarray:
        """根據 Hamiltonian 的 theta 參數，建構單一的對數修正曲線 log(f(r))。"""
        theta_h, _ = self._split_theta(theta)
        if theta_h is None: return np.zeros_like(self.xi)
        
        if self.cfg.basis in ["poly", "bspline"]:
            g = self.Phi @ theta_h
        elif self.cfg.basis == "sigmoid":
            g = np.zeros_like(self.xi)
            for m in range(self.cfg.K):
                a, b_raw, mu_raw = theta_h[3*m:3*m+3]
                g += a * sigmoid((softplus(b_raw) + 1e-6) * (self.xi - sigmoid(mu_raw)))
        else:
            raise ValueError("Unknown basis")
        
        return g - g[-1]
    
    def _repulsive_override(self, sk: ParsedSKFile, theta_rep: np.ndarray) -> Optional[Dict[str, Any]]:
        rep = sk.repulsive
        if rep.get('type') != 'spline' or theta_rep is None: return None
        
        da1, da2, da3, dscale = theta_rep.tolist()
        a1, a2, a3 = rep['exp']
        new_a1, new_a2, new_a3 = a1 * (1.0 + 0.05 * da1), a2 * (1.0 + 0.05 * da2), a3 * (1.0 + 0.05 * da3)
        coef_scale = 1.0 + 0.05 * dscale
        
        segs = [{'start': seg['start'], 'end': seg['end'], 'c': [ci * coef_scale for ci in seg['c']]} for seg in rep['segments']]
        
        return {'type': 'spline', 'cutoff': rep['cutoff'], 'nInt': rep['nInt'], 'exp': (new_a1, new_a2, new_a3), 'segments': segs}
    
    def synthesize(self, theta_list: Sequence[np.ndarray]) -> Tuple[List[Dict[str, List[float]]], List[np.ndarray]]:
        Y_by_file, all_logs = [], []
        param_map = {key: theta for key, theta in zip(self.param_keys, theta_list)}
        
        for pair_name in self.sk_data_map.keys():
            theta_key = self.theta_mapping.get(pair_name)
            if theta_key in param_map:
                theta = param_map[theta_key]
                theta_h, theta_rep = self._split_theta(theta)
                
                log_fc = self._log_fc_one(theta)
                fc = np.exp(log_fc)
                if self.do_ham: all_logs.append(log_fc)
                
                rep_override = None
                if self.do_rep: rep_override = self._repulsive_override(self.sk_data_map[pair_name], theta_rep)
                
                original_H = self.sk_data_map[pair_name].H
                modified_H = {key: np.array(values) * fc for key, values in original_H.items()}
                if rep_override is not None: modified_H["__REPULSIVE__"] = rep_override
                Y_by_file.append(modified_H)
        
        return Y_by_file, all_logs
    
    def smooth_penalty(self, log_fc_list: List[np.ndarray]) -> float:
        lam = self.cfg.smooth_lambda
        if lam <= 0.0 or not self.do_ham: return 0.0
        tot = sum(np.mean((g[2:] - 2*g[1:-1] + g[:-2])**2) for g in log_fc_list)
        return lam * tot / max(len(log_fc_list), 1)

from dataclasses import dataclass
from typing import List

# =============================================================================
# [4] SPSA Training Algorithm & Utilities
# =============================================================================
@dataclass
class TrainConfig:
    groups: List[List[int]]
    starts: int
    cycles: int
    steps_per_group: int
    batch_P: int
    batch_F: int
    seed: int = 123
    # SPSA hyperparameters with reasonable defaults
    a0: float = 1e-1
    c0: float = 1e-2
    alpha: float = 0.602
    gamma: float = 0.101
    reps: int = 1
"""
# --------------------------------------------------------------------------
# Simultaneous Perturbation Stochastic Approximation (SPSA)
#
# ▸ 目的：SPSA 是一種隨機梯度下降演算法，特別適用於目標函式 (損失函式) 的
#   梯度難以或無法直接計算的場景。在我們的案例中，目標函式涉及執行大量
#   DFTB+ 計算，其對 `theta` 參數的解析梯度是不可得的。
#
# ▸ 梯度近似：SPSA 的核心思想是，在每一步中，只對參數 `theta` 施加一個隨機的
#   擾動 `Δ` (其中 `Δ` 的每個元素是 +1 或 -1)，然後評估目標函式在 `θ+c_kΔ`
#   和 `θ−c_kΔ` 兩點的值 (y+ 和 y−)。僅僅透過這兩次評估，就可以近似出
#   整個高維梯度向量：
#   ĝ_i = (y+ − y−) / (2 * c_k * Δ_i)
#   這使得每一步的計算成本與參數維度無關，非常高效。
#
# ▸ 參數更新：θ_{k+1} = θ_k − a_k * ĝ
#   其中 a_k 和 c_k 是隨迭代次數 k 遞減的序列，控制學習率和擾動幅度。
#
# ▸ 本實作特點：
#   - 多起點 (multi-start)：多次從不同的隨機初始點開始訓練，以增加找到全域最優解的機會。
#   - 分組更新 (grouping)：允許將多組 `theta` 參數分組，每次只更新一組，適用於參數間耦合較弱的情況。
#   - 梯度均值 (reps)：多次計算隨機梯度並取平均，以降低梯度估計的噪聲。
#   - 並行評估：`evaluate_rmse` 函式內部使用多執行緒並行執行 DFTB+ 計算，加速評估過程。
#   - 檢查點 (checkpointing)：定期保存訓練狀態，允許中斷後從上次進度繼續。
#   - 驗證集監控：使用獨立的驗證集來追蹤最佳參數 (`best_theta`)，避免過擬合。
#   - CRN (Common Random Numbers)：在計算 y+ 和 y− 時使用相同的隨機種子，
#     可以顯著降低 `y+ - y-` 的方差，使梯度估計更穩定。
#
# ▸ 變數說明：
#   - s/cyc/gpos/step/tstep：進度指標，分別代表起點、循環、組、組內步驟和總步驟。
#   - theta_list：一個列表，儲存了所有需要優化的參數組 (例如 theta_C-C, theta_C-H 等)。
#   - idx_offsets：用於將多個 `theta` 組合成一個扁平向量，方便梯度計算，之後再拆分回來。
# --------------------------------------------------------------------------
"""
def _save_ckpt(state: Dict[str, Any], ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, 'wb') as fo: pickle.dump(state, fo)

def _load_ckpt(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'rb') as fi: return pickle.load(fi)
        except: return None
    return None

def _detect_cores() -> int:
    try: return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count() or 1

def spsa_train(
    model: TabulatedModel, evaluate_rmse: Callable, train_cfg: TrainConfig,
    P_ids: np.ndarray, F_pool_ids: np.ndarray, val_ids: Optional[np.ndarray] = None,
    checkpoint_path: Optional[str] = None, save_secs: int = 120, resume: bool = True,
    stop_flag: Optional[Dict[str, bool]] = None, strata_pools: Optional[List[Tuple[np.ndarray, int]]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[np.ndarray], float, Dict[str, Any]]:
    """SPSA training with checkpointing and multi-start"""
    
    # Validation
    if train_cfg.steps_per_group <= 0:
        raise ValueError(f"steps_per_group must be > 0, got {train_cfg.steps_per_group}")
    if train_cfg.starts <= 0:
        raise ValueError(f"starts must be > 0, got {train_cfg.starts}")
    if train_cfg.cycles <= 0:
        raise ValueError(f"cycles must be > 0, got {train_cfg.cycles}")
    
    rng_master = np.random.default_rng(train_cfg.seed)
    theta_len = model.theta_shape()[0]
    all_runs, ckpt_path = [], Path(checkpoint_path) if checkpoint_path else None
    last_save = time.time()
    
    s = cyc = gpos = step = tstep = 0
    theta_list, best_theta, best_val = [], [], np.inf
    history, group_order = {"val": [], "t": []}, []
    
    if resume and ckpt_path:
        state = _load_ckpt(ckpt_path)
        if state:
            (logger or logging.getLogger("skf_opt")).info(f"Resuming from {ckpt_path}...")
            s = state['s']
            cyc = state['cyc']
            gpos = state['gpos']
            step = state['step']
            tstep = state['tstep']
            best_val = state['best_val']
            history = state['history']
            group_order = state['group_order']
            theta_list = [np.array(t) for t in state.get('theta_list', [])]
            best_theta = [np.array(t) for t in state.get('best_theta', [])]
            rng_master.bit_generator.state = state.get('rng_master_state', rng_master.bit_generator.state)
    
    while s < train_cfg.starts:
        (logger or logging.getLogger("skf_opt")).info(f"\n--- SPSA Multi-start {s+1}/{train_cfg.starts} ---")
        rng = np.random.default_rng(rng_master.integers(1<<30))
        
        if not theta_list:
            theta_list = [rng.normal(0.0, 0.05, size=theta_len) for _ in range(model.M)]
            best_theta, best_val = [t.copy() for t in theta_list], np.inf
            history, tstep = {"val": [], "t": []}, 0
            cyc = gpos = step = 0
            group_order = []
        
        while cyc < train_cfg.cycles:
            if not group_order:
                group_order = list(range(len(train_cfg.groups)))
                rng.shuffle(group_order)
            
            g_idx = group_order[gpos]
            g_ids = train_cfg.groups[g_idx]
            
            flat_theta, idx_offsets = [], []
            current_offset = 0
            for i in g_ids:
                idx_offsets.append((i, current_offset, current_offset + theta_len))
                flat_theta.append(theta_list[i])
                current_offset += theta_len
            flat_theta = np.concatenate(flat_theta) if flat_theta else np.array([], dtype=float)
            
            while step < train_cfg.steps_per_group:
                tstep += 1
                a_t = train_cfg.a0 / ((tstep + 100)**train_cfg.alpha)
                c_t = train_cfg.c0 / (tstep**train_cfg.gamma)
                
                if strata_pools:
                    sampled_chunks = []
                    for pool, k_target in strata_pools:
                        if pool.size == 0: continue
                        k = min(k_target, pool.size)
                        sampled_chunks.append(rng.choice(pool, size=k, replace=False))
                    f_batch = np.concatenate(sampled_chunks) if sampled_chunks else np.array([], dtype=int)
                    batch = np.unique(np.concatenate([P_ids, f_batch])).astype(int)
                else:
                    if F_pool_ids.size > 0:
                        k = min(train_cfg.batch_F, F_pool_ids.size)
                        f_batch = rng.choice(F_pool_ids, size=k, replace=False)
                        batch = np.concatenate([P_ids, f_batch])
                    else:
                        batch = P_ids
                
                grad = np.zeros_like(flat_theta)
                for _ in range(train_cfg.reps):
                    Delta = rng.choice([-1.0, 1.0], size=flat_theta.size)
                    theta_plus, theta_minus = [t.copy() for t in theta_list], [t.copy() for t in theta_list]
                    
                    for i, lo, hi in idx_offsets:
                        d = Delta[lo:hi]
                        theta_plus[i] += c_t * d
                        theta_minus[i] -= c_t * d
                    
                    state_rng = rng.bit_generator.state
                    Yp, logs_p = model.synthesize(theta_plus)
                    f_plus = evaluate_rmse(Yp, batch, rng) + model.smooth_penalty(logs_p)
                    rng.bit_generator.state = state_rng
                    Ym, logs_m = model.synthesize(theta_minus)
                    f_minus = evaluate_rmse(Ym, batch, rng) + model.smooth_penalty(logs_m)
                    
                    grad += ((f_plus - f_minus) / (2.0 * c_t)) * Delta
                
                flat_theta -= (a_t / train_cfg.reps) * grad
                for i, lo, hi in idx_offsets:
                    theta_list[i] = flat_theta[lo:hi]
                
                if val_ids is not None and val_ids.size > 0:
                    Yv, logv = model.synthesize(theta_list)
                    v = evaluate_rmse(Yv, val_ids, rng) + model.smooth_penalty(logv)
                    history["val"].append(v)
                    history["t"].append(tstep)
                    if v < best_val:
                        best_val, best_theta = v, [t.copy() for t in theta_list]
                        (logger or logging.getLogger("skf_opt")).info(f"  > Step {tstep}: New best val loss: {best_val:.6f}")
                
                now = time.time()
                should_save = ckpt_path and ((now - last_save) >= save_secs)
                killed = bool(stop_flag and stop_flag.get('kill')) or (ckpt_path and ckpt_path.with_suffix('.STOP').exists())
                
                if should_save or killed:
                    if ckpt_path:
                        _save_ckpt({
                            's': s, 'cyc': cyc, 'gpos': gpos, 'step': step, 'tstep': tstep,
                            'theta_list': [t.copy() for t in theta_list],
                            'best_theta': [t.copy() for t in best_theta],
                            'best_val': best_val, 'history': history, 'group_order': group_order,
                            'rng_master_state': rng_master.bit_generator.state,
                        }, ckpt_path)
                        last_save = now
                        (logger or logging.getLogger("skf_opt")).info(f"--- Checkpoint saved at step {tstep} ---")
                    if killed:
                        (logger or logging.getLogger("skf_opt")).info("Stop requested, checkpoint saved.")
                        logs = {"history": history, "resume_ckpt": str(ckpt_path) if ckpt_path else None}
                        return best_theta, best_val, logs
                
                step += 1
            
            step = 0
            gpos += 1
            if gpos >= len(group_order):
                gpos = 0
                group_order = []
                cyc += 1
                (logger or logging.getLogger("skf_opt")).info(f"--- Completed Cycle {cyc}/{train_cfg.cycles} ---")
        
        all_runs.append((best_val, [t.copy() for t in best_theta], {"history": history}))
        theta_list = []
        best_theta = []
        best_val = np.inf
        history = {"val": [], "t": []}
        cyc = gpos = step = 0
        s += 1
        
        if ckpt_path:
            _save_ckpt({
                's': s, 'cyc': 0, 'gpos': 0, 'step': 0, 'tstep': 0,
                'theta_list': [], 'best_theta': [], 'best_val': np.inf,
                'history': {"val": [], "t": []}, 'group_order': [],
                'rng_master_state': rng_master.bit_generator.state,
            }, ckpt_path)
    
    if not all_runs: raise RuntimeError("No training results")
    all_runs.sort(key=lambda x: x[0])
    best_val, best_theta, hist = all_runs[0]
    logs = {"history": hist["history"], "all_runs": [(bv, h) for bv, _, h in all_runs]}
    return best_theta, best_val, logs

# =============================================================================
# [5] DFTB+ Calculation & Objective Function
# =============================================================================

class DFTBPlusRunner:
    """
    將執行 DFTB+ 計算的相關邏輯封裝成一個類別。

    這個類別負責管理工作目錄、HSD 輸入檔的生成、執行 DFTB+ 子程序、
    並行處理多個計算任務，以及解析輸出結果。
    """
    def __init__(self, base_work_dir: str, base_skf_dir: str, dftb_template: str,
                 calculation_requirements: Set[str]):
        """
        初始化 DFTB+ 執行器。

        Args:
            base_work_dir: 所有計算運行的根目錄 (例如 'runs/')。
            base_skf_dir: 包含原始 (非優化) SKF 檔案的目錄 (例如 'SKFs/')。
            dftb_template: dftb_in.hsd 檔案的模板內容。
            calculation_requirements: 一個集合，包含需要執行的計算類型 (例如 {'ground_state', 'casida'})。
        """
        self.base_work_dir = Path(base_work_dir)
        self.base_skf_dir = Path(base_skf_dir).resolve()
        self.dftb_template = dftb_template
        self.calc_reqs = calculation_requirements
        
        try:
            # 確保即使環境變數設為 "0" 或空字串也能處理
            omp_threads = int(os.environ.get("OMP_NUM_THREADS") or 1)
            if omp_threads <= 0:
                omp_threads = 1
        except ValueError:
            omp_threads = 1
        
        total_cores = _detect_cores()
        self.max_workers = max(1, total_cores // omp_threads)

    def _run_single_calculation(self, geometry_path: str, modified_sk_dir: Path, work_dir: Path) -> Optional[List[float]]:
        """執行單一的 DFTB+ 計算並解析結果。"""
        stem = Path(geometry_path).stem
        geom_dst = work_dir / f"{stem}.xyz"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(geometry_path, geom_dst)
        except FileNotFoundError:
            logging.getLogger("skf_opt").warning(f"Error: Geometry file {geometry_path} not found")
            return None
        # Avoid str.format() because HSD uses many literal braces `{}`; do simple placeholder replacement instead
        hsd_content = (self.dftb_template
                       .replace("{geometry_filename}", geom_dst.name)
                       .replace("{opt_dir}", str(modified_sk_dir.resolve()) + os.sep)
                       .replace("{SKF_DIR}", str(self.base_skf_dir) + os.sep))
        (work_dir / "dftb_in.hsd").write_text(hsd_content, encoding='utf-8')
        log_path = work_dir / "output"
        try:
            with open(log_path, "w", encoding='utf-8') as log_file:
                subprocess.run(["dftb+"], cwd=work_dir, stdout=log_file, stderr=subprocess.STDOUT, check=True, timeout=300)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        # --- 動態解析結果 ---
        results = {}
        output_content = log_path.read_text(encoding='utf-8', errors='ignore')
        if 'ground_state' in self.calc_reqs:
            match = re.search(r"Total Energy:.*?\s+([\-\d\.]+)\s+eV", output_content)
            if match:
                results['S0'] = float(match.group(1))
        if 'casida' in self.calc_reqs:
            exc_dat_path = work_dir / "EXC.DAT"
            if exc_dat_path.exists():
                try:
                    # Read lines keeping UTF-8 but be forgiving about encoding
                    lines = exc_dat_path.read_text(encoding='utf-8', errors='ignore').splitlines()
                    
                    if not lines:
                        logging.getLogger("skf_opt").warning(f"Warning: {exc_dat_path.name} is empty")
                    else:
                        # Find separator line
                        sep_idx = -1
                        for i, ln in enumerate(lines):
                            if ln.strip() and set(ln.strip()) == {'='}:
                                sep_idx = i
                                break

                    # Start scanning after the separator if found; otherwise from top
                    start_idx = sep_idx + 1 if sep_idx != -1 else 0

                    # Regex to capture a leading float number (with optional exponent) at line start
                    num_re = re.compile(r"^[\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[\s]+")

                    first_energy = None
                    for j in range(start_idx, len(lines)):
                        ln = lines[j].strip()
                        if not ln or ln.startswith('#'):
                            continue  # skip blanks/comments
                        m = num_re.match(ln)
                        if m:
                            # Column 1 is w [eV]
                            first_energy = float(m.group(1))
                            break
                    if first_energy is not None:
                        results['Excitation'] = first_energy
                except Exception as e:
                    # Keep going, but surface a concise warning for debugging
                    logging.getLogger("skf_opt").warning(f"Warning: Error parsing {exc_dat_path.name}: {e}")
        return [results.get('S0'), results.get('Excitation')]

    def evaluate_batch(self, Y_by_file: List[Dict[str, List[float]]],
                       sk_data_map: Dict[str, ParsedSKFile],
                       batch_jobs: List[Tuple[str, List[float]]]) -> List[Tuple[Optional[List[float]], List[float]]]:
        """
        對一批結構執行並行計算。
        
        此方法會建立一個臨時目錄來存放該批次所需的、經過修改的 SKF 檔案。
        """
        if not batch_jobs:
            return []
        # 1. 建立唯一的臨時 SKF 目錄
        run_id = np.random.randint(1 << 30)
        modified_sk_dir = self.base_work_dir / f"spsa_eval_{run_id}"
        modified_sk_dir.mkdir(parents=True, exist_ok=True)
        try:
            # 2. 將模型合成的 SKF 寫入臨時目錄
            for i, (pair_name, sk_data) in enumerate(sk_data_map.items()):
                out_path = modified_sk_dir / f"{pair_name}.skf"
                sk_data.write(out_path, Y_by_file[i])
            # 3. 準備計算任務
            tasks = []
            for gpath, ref_vals in batch_jobs:
                struct_work_dir = modified_sk_dir / f"struct_{Path(gpath).stem}"
                tasks.append((gpath, modified_sk_dir, struct_work_dir))
            # 4. 並行執行
            results = []
            if self.max_workers > 1 and len(tasks) > 1:
                with cf.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_ref = {executor.submit(self._run_single_calculation, *task): batch_jobs[i][1]
                                     for i, task in enumerate(tasks)}
                    for future in cf.as_completed(future_to_ref):
                        ref_vals = future_to_ref[future]
                        try:
                            pred_vals = future.result()
                            results.append((pred_vals, ref_vals))
                        except Exception as exc:
                            logging.getLogger("skf_opt").warning(f"A calculation task generated an exception: {exc}")
                            results.append((None, ref_vals))
            else: # 單線程執行
                for i, task in enumerate(tasks):
                    pred_vals = self._run_single_calculation(*task)
                    results.append((pred_vals, batch_jobs[i][1]))
        finally:
            # 5. 清理臨時目錄
            shutil.rmtree(modified_sk_dir, ignore_errors=True)
        return results

def create_objective_function(
    dataset: List[Tuple[str, List[float]]],
    dftb_runner: DFTBPlusRunner,
    sk_data_map: Dict[str, ParsedSKFile],
    target_weights: List[float],
    active_targets: List[Dict[str, Any]]
) -> Callable:
    """工廠函式：建立 RMSE 目標函式。"""
    
    def evaluate_rmse(Y_by_file: List[Dict[str, List[float]]], batch_ids: np.ndarray, rng: np.random.Generator) -> float:
        """
        SPSA 所使用的目標函式 (或稱損失函式)。
        它接收模型參數 `theta` (隱含在 Y_by_file 中)，回傳 RMSE。
        """
        # 從完整資料集中選取當前批次的任務
        batch_jobs = [(dataset[idx][0], dataset[idx][1]) for idx in batch_ids]
        
        # 使用 DFTBPlusRunner 執行計算
        results = dftb_runner.evaluate_batch(Y_by_file, sk_data_map, batch_jobs)
        
        err_e_sq_sum = 0.0
        valid_calcs = 0
        
        for e_pred_full, e_ref in results:
            if e_pred_full is None:
                continue
            
            try:
                # 根據 active_targets 選擇需要比較的預測值
                selected_preds = [e_pred_full[t['calc_idx']] for t in active_targets]
                
                # 檢查所有需要的預測值是否都成功計算出來
                if any(p is None for p in selected_preds):
                    continue
                
                if len(selected_preds) == len(e_ref):
                    weighted_squared_error = sum(
                        w * (pred - ref) ** 2
                        for w, pred, ref in zip(target_weights, selected_preds, e_ref)
                    )
                    err_e_sq_sum += weighted_squared_error
                    valid_calcs += 1
            except IndexError:
                continue
        
        return 1e10 if valid_calcs == 0 else np.sqrt(err_e_sq_sum / valid_calcs)
    
    return evaluate_rmse

# =============================================================================
# [6] Dataset Loading & Utilities
# =============================================================================

def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    return [int(x) for x in re.split(r'[,\s]+', s) if x] if s else []

def _parse_ranges(s: str) -> List[Tuple[int, int]]:
    s = s.strip()
    if not s: return []
    out = []
    for part in re.split(r'[,\s]+', s):
        if not part: continue
        if '-' in part:
            a, b = part.split('-', 1)
            lo, hi = int(a), int(b)
        else:
            lo = hi = int(part)
        if lo > hi: lo, hi = hi, lo
        out.append((lo, hi))
    return out

def load_dataset(ref_file: str, data_dir: str, num_targets: int) -> List[Tuple[str, List[float]]]:
    """
    從 ref.txt 讀取參考數據集。

    檔案格式應為：
    <結構檔案名稱> <參考值1> <參考值2> ...

    Args:
        ref_file: 參考檔案的路徑 (e.g., "ref.txt")。
        data_dir: 結構檔案所在的目錄 (e.g., "data/")。
        num_targets: 每個結構對應的參考值數量。

    Returns:
        一個包含 (結構檔案絕對路徑, [參考值列表]) 的元組列表。
    """
    dataset = []
    if not Path(ref_file).exists(): raise FileNotFoundError(f"Reference file {ref_file} not found")
    
    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) != 1 + num_targets:
                print(f"Warning: Line '{line[:50]}...' has wrong field count (expected {1+num_targets}), skipping")
                continue
            
            struct_name = parts[0]
            value_strs = parts[1:]
            struct_path = Path(data_dir) / struct_name
            
            if not struct_path.exists():
                print(f"Warning: Structure file {struct_path} not found, skipping")
                continue
            
            try:
                values = [float(v) for v in value_strs]
                dataset.append((str(struct_path), values))
            except ValueError:
                print(f"Warning: Cannot parse values '{value_strs}', skipping")
                continue
    
    if not dataset: raise ValueError("Dataset is empty")
    return dataset

def create_template_files(
    data_dir: str,
    templates_dir: str,
    ref_file: str,
    skf_dir: str,
    all_element_pairs: List[str],
    optimizing_skf_pairs: List[str],
    extra_hsd_content: str = ""
):
    """
    根據設定，動態創建專案所需的模板檔案。

    Args:
        data_dir: 存放幾何結構檔案的目錄。
        templates_dir: 存放 dftb_in.hsd 模板的目錄。
        ref_file: 參考數據檔案的路徑。
        skf_dir: 存放基礎 SKF 檔案的目錄。
        all_element_pairs: 計算中所有會用到的元素配對 (e.g., ["C-H", "Ni-O"])。
        optimizing_skf_pairs: 指定要優化的 SKF 配對。
        extra_hsd_content: 可選的、要附加到 dftb_in.hsd 中的額外內容字串。
    """
    Path(data_dir).mkdir(exist_ok=True)
    Path(templates_dir).mkdir(exist_ok=True)
    Path(skf_dir).mkdir(exist_ok=True)

    # --- 動態生成 SlaterKosterFiles 區塊 ---
    def _generate_skf_block(all_pairs, opt_pairs_set):
        lines = []
        processed_pairs = set()
        for pair in all_pairs:
            elements = sorted(pair.split('-'))
            sorted_pair = f"{elements[0]}-{elements[1]}"
            if sorted_pair in processed_pairs:
                continue
            
            # 決定是用 opt_dir 還是 SKF_DIR
            directory = "{opt_dir}" if pair in opt_pairs_set or f"{elements[1]}-{elements[0]}" in opt_pairs_set else "{SKF_DIR}"
            
            # 同時處理 A-B 和 B-A 的情況
            e1, e2 = pair.split('-')
            lines.append(f"        {e1}-{e2} = \"{directory}{e1}-{e2}.skf\"")
            if e1 != e2:
                lines.append(f"        {e2}-{e1} = \"{directory}{e2}-{e1}.skf\"")
            processed_pairs.add(sorted_pair)
        return "\n".join(lines)

    optimizing_set = set(optimizing_skf_pairs)
    skf_block_str = _generate_skf_block(all_element_pairs, optimizing_set)

    # --- 組合完整的 dftb_in.hsd 內容 ---
    dftb_template_path = Path(templates_dir, "dftb_in.hsd")
    if not dftb_template_path.exists():
        print(f"Creating dynamic template: {dftb_template_path}")
        
        hsd_template = f"""Geometry = xyzFormat {{
    <<< "{{geometry_filename}}"
}}

Hamiltonian = DFTB {{
    SCC = Yes
    MaxAngularMomentum {{
        H  = "s"
        C  = "p"
        O  = "p"
        Fe = "d"
    }}
    SlaterKosterFiles {{
{skf_block_str}
    }}
}}

{extra_hsd_content}

Options = {{ WriteDetailedOut = No }}
ParserOptions {{ ParserVersion = 14 }}
"""
        dftb_template_path.write_text(hsd_template.strip())

    # --- 創建 ref.txt 和範例結構 (與之前相同) ---
    ref_file_path = Path(ref_file)
    if not ref_file_path.exists():
        print(f"Creating template: {ref_file_path}")
        ref_file_path.write_text(
            "# Format depends on --targets flag.\n"
            "# For --targets both: Structure_file S0_energy(eV) Excitation_energy(eV)\n"
            "struct1.xyz  -150.1234  3.5\n"
            "struct2.xyz  -160.5678  3.2\n"
        )
    
    for i in range(1, 3):
        struct_path = Path(data_dir, f"struct{i}.xyz")
        if not struct_path.exists():
            print(f"Creating template: {struct_path}")
            struct_path.write_text("4\n\nC 0.0 0.0 0.0\nH 1.1 0.0 0.0\nH -0.4 0.9 0.0\nSn 0.0 0.0 1.8\n")

# =============================================================================
# [7] Main Entry Point & Project Orchestrator
# =============================================================================

class SKFOptimizationProject:
    """
    一個總控類別，用於管理整個 SKF 優化專案的生命週期。
    它封裝了設定、初始化、執行和結果儲存的所有步驟。
    """
    def __init__(self, args: argparse.Namespace, skf_pairs_to_optimize: List[str], extra_hsd: str):
        self.args = args
        self.skf_pairs_to_optimize = skf_pairs_to_optimize
        self.extra_hsd = extra_hsd

        self.paths = {
            "data": Path("data"), "ref": Path("ref.txt"), "templates": Path("templates"),
            "skf": Path("SKFs"), "runs": Path("runs")
        }
        self.paths["ckpt"] = self.paths["runs"] / "spsa_ckpt.pkl"
        self.paths["optimized_skf"] = self.paths["skf"] / "optimized"

        # 初始化元件，將在 setup() 中被建立
        self.model: Optional[TabulatedModel] = None
        self.dftb_runner: Optional[DFTBPlusRunner] = None
        self.dataset: List[Tuple[str, List[float]]] = []
        self.sk_data_map: Dict[str, ParsedSKFile] = {}
        
        # 處理優化目標設定
        self._process_target_settings()
        
        self.logger: Optional[logging.Logger] = None

    def _process_target_settings(self):
        """根據 args 解析並設定優化目標、權重和計算需求。"""
        target_defs = {
            'S0':       {'calc_idx': 0, 'default_weight': 2.0, 'requires': {'ground_state'}},
            'Excitation': {'calc_idx': 1, 'default_weight': 1.0, 'requires': {'ground_state', 'casida'}},
        }
        target_mapping = {'energy': ['S0'], 'excitation': ['Excitation'], 'both': ['S0', 'Excitation']}
        
        active_names = target_mapping[self.args.targets]
        self.num_targets = len(active_names)
        self.active_targets = [{'name': name, 'calc_idx': target_defs[name]['calc_idx']} for name in active_names]
        
        self.calc_reqs = {req for name in active_names for req in target_defs[name]['requires']}
        
        if self.num_targets > 1:
            self.target_weights = [target_defs[t['name']]['default_weight'] for t in self.active_targets]
        else:
            self.target_weights = [1.0]
            
        print(f"Targets: {[t['name'] for t in self.active_targets]} with weights {self.target_weights}.")
        print(f"Expecting {self.num_targets} value(s) in {self.paths['ref']}.")

    def _setup_project_templates(self):
        """掃描 SKF 目錄並建立 dftb_in.hsd 模板和 ref.txt 範例。"""
        self.logger.info("Scanning SKF directory and setting up templates...")
        self.paths["skf"].mkdir(exist_ok=True)
        discovered_skfs = sorted([p.stem for p in self.paths["skf"].glob("*.skf")])
        self.logger.info(f"Discovered {len(discovered_skfs)} SKF pairs in '{self.paths['skf']}/'.")

        create_template_files(
            data_dir=str(self.paths["data"]),
            templates_dir=str(self.paths["templates"]),
            ref_file=str(self.paths["ref"]),
            skf_dir=str(self.paths["skf"]),
            all_element_pairs=discovered_skfs,
            optimizing_skf_pairs=self.skf_pairs_to_optimize,
            extra_hsd_content=self.extra_hsd
        )

    def setup(self):
        """執行所有優化前的準備工作。"""
        # initialize logger (place the log under runs/ if a relative path is provided)
        log_path = Path(self.args.log_file)
        if not log_path.is_absolute():
            log_path = self.paths["runs"] / log_path
        self.logger = setup_logger(str(log_path))
        self.logger.info("\n--- [1/3] Setting up Project ---")
        self.paths["runs"].mkdir(exist_ok=True)
        self._setup_project_templates()

        # 載入資料集
        self.dataset = load_dataset(str(self.paths["ref"]), str(self.paths["data"]), num_targets=self.num_targets)
        self.logger.info(f"Loaded {len(self.dataset)} structures from {self.paths['ref']}")
        
        # 建立完整的 SKF 優化列表 (含 A-B, B-A) 和參數映射
        full_opt_list = []
        processed = set()
        for p in self.skf_pairs_to_optimize:
            elements = sorted(p.split('-'))
            sorted_pair = f"{elements[0]}-{elements[1]}"
            if sorted_pair in processed: continue
            
            full_opt_list.append(f"{elements[0]}-{elements[1]}")
            if elements[0] != elements[1]:
                full_opt_list.append(f"{elements[1]}-{elements[0]}")
            processed.add(sorted_pair)
        
        theta_mapping = {p: f"theta_{'-'.join(sorted(p.split('-')))}" for p in full_opt_list}
        
        # 載入 SKF 檔案 (使用重構後的 ParsedSKFile)
        self.sk_data_map = {p: ParsedSKFile.from_file(str(self.paths["skf"] / f"{p}.skf")) for p in full_opt_list}
        
        # 初始化模型 (TabulatedModel)
        correction_cfg = CorrectionConfig(
            basis=self.args.basis, opt_scope=self.args.opt_scope, K=self.args.K,
            bspline_degree=self.args.bspline_degree, smooth_lambda=self.args.smooth_lambda
        )
        self.model = TabulatedModel(self.sk_data_map, theta_mapping, correction_cfg)
        
        # 初始化 DFTB+ 執行器
        dftb_template = (self.paths["templates"] / "dftb_in.hsd").read_text(encoding='utf-8')
        self.dftb_runner = DFTBPlusRunner(
            base_work_dir=str(self.paths["runs"]),
            base_skf_dir=str(self.paths["skf"]),
            dftb_template=dftb_template,
            calculation_requirements=self.calc_reqs
        )
        self.logger.info("Project setup complete.")

    def _prepare_data_pools(self) -> Tuple[np.ndarray, np.ndarray, Optional[List[Tuple[np.ndarray, int]]]]:
        """準備永久集、浮動集和分層抽樣池。"""
        n = len(self.dataset)
        if n == 0: raise ValueError("Dataset is empty, cannot run optimization.")
        
        # =================================================================
        # ▼▼▼ 在這裡加入自動偵測邏輯 ▼▼▼
        # =================================================================
        # Use configurable threshold instead of hardcoded
        AUTO_ALL_DATA_THRESHOLD = self.args.auto_all_threshold

        # 判斷數據集大小是否小於閾值
        if n < AUTO_ALL_DATA_THRESHOLD:
            self.logger.info(f"Dataset size ({n}) is less than the threshold ({AUTO_ALL_DATA_THRESHOLD}).")
            self.logger.info("Automatically using the entire dataset for optimization.")
            self.logger.info("All sampling-related arguments (--permanent, --strata, etc.) will be ignored.")
            
            # 準備返回：P_ids 設為全部, F_pool_ids 設為空, strata_pools 設為 None
            all_indices = np.arange(n)
            return all_indices, np.array([], dtype=int), None
        # =================================================================
        # ▲▲▲ 自動偵測邏輯結束 ▲▲▲
        # =================================================================

        # 如果數據量大於等於閾值，則執行原始的抽樣邏輯
        self.logger.info(f"Dataset size ({n}) is greater than or equal to the threshold.")
        self.logger.info("Using sampling strategy defined by command-line arguments.")
        P_ids = np.array(_parse_int_list(self.args.permanent), dtype=int) - 1
        P_ids = P_ids[(P_ids >= 0) & (P_ids < n)]
        
        all_indices = np.arange(n)
        F_pool_ids = np.setdiff1d(all_indices, P_ids)
        
        strata_pools = None
        if not self.args.no_strata:
            ranges = _parse_ranges(self.args.strata)
            pools = [np.setdiff1d(np.arange(max(0, a-1), min(n, b)), P_ids) for a, b in ranges]
            strata_pools = [(p, min(self.args.k_per_pool, p.size)) for p in pools if p.size > 0]
            
        return P_ids, F_pool_ids, strata_pools
        
    def _save_results(self, best_theta_list: List[np.ndarray], best_val: float):
        """合成並儲存優化後的 SKF 檔案及參數。"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Optimization Complete!")
        self.logger.info(f"Best objective value: {best_val:.6e}")
        
        self.paths["optimized_skf"].mkdir(exist_ok=True)
        final_H_by_file, _ = self.model.synthesize(best_theta_list)
        
        for i, pair_name in enumerate(self.sk_data_map.keys()):
            sk_data = self.sk_data_map[pair_name]
            out_path = self.paths["optimized_skf"] / f"{pair_name}.skf"
            sk_data.write(out_path, final_H_by_file[i]) # 使用重構後的 write 方法
            self.logger.info(f"Saved optimized {pair_name}.skf to {out_path}")
        
        optimized_params = {key: theta.tolist() for key, theta in zip(self.model.param_keys, best_theta_list)}
        self.logger.info("\nFinal optimized parameters:\n" + json.dumps(optimized_params, indent=2))
        self.logger.info("="*50)

    def run(self):
        """執行 SPSA 優化流程。"""
        if not all([self.model, self.dftb_runner, self.dataset]):
            raise RuntimeError("Project is not set up. Please call project.setup() before running.")

        self.logger.info("\n--- [2/3] Starting SPSA Optimization ---")
        
        # 建立目標函式
        evaluate_rmse_func = create_objective_function(
            dataset=self.dataset, dftb_runner=self.dftb_runner, sk_data_map=self.sk_data_map,
            target_weights=self.target_weights, active_targets=self.active_targets
        )
        
        # 準備資料池
        P_ids, F_pool_ids, strata_pools = self._prepare_data_pools()
        val_ids = np.arange(len(self.dataset))

        # 設定 SPSA 訓練參數
        train_cfg = TrainConfig(
            groups=[list(range(self.model.M))], starts=2, cycles=3, steps_per_group=20,
            batch_P=self.args.batch_P, batch_F=self.args.batch_F, seed=123
        )

        # 執行訓練
        best_theta_list, best_val, _ = spsa_train(
            self.model, evaluate_rmse_func, train_cfg, P_ids, F_pool_ids, val_ids=val_ids,
            checkpoint_path=str(self.paths["ckpt"]), resume=True, strata_pools=strata_pools, logger=self.logger
        )
        
        # 清理停止標記
        stop_marker = self.paths["ckpt"].with_suffix('.STOP')
        if stop_marker.exists(): stop_marker.unlink()
        
        self.logger.info("\n--- [3/3] Processing Final Results ---")
        self._save_results(best_theta_list, best_val)

def main():
    """主程式進入點：解析參數並驅動優化專案。"""
    parser = argparse.ArgumentParser(description="Optimize SKFs with SPSA")
    # ... (此處保留所有 argparse 的設定，與原始版本相同)
    parser.add_argument("--targets", choices=["energy", "excitation", "both"], default="both", help="Optimization target(s).")
    parser.add_argument("--basis", choices=["poly", "bspline", "sigmoid"], default="poly")
    parser.add_argument("--opt-scope", choices=["ham", "repulsive", "both"], default="both")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--bspline-degree", type=int, default=3)
    parser.add_argument("--smooth-lambda", type=float, default=1e-3)
    parser.add_argument("--permanent", type=str, default="1,201,401")
    parser.add_argument("--strata", type=str, default="1-200,201-400,401-600")
    parser.add_argument("--k-per-pool", type=int, default=20)
    parser.add_argument("--no-strata", action="store_true")
    parser.add_argument("--batch-P", type=int, default=10)
    parser.add_argument("--batch-F", type=int, default=40)
    parser.add_argument("--log-file", type=str, default="skf_opt.log")
    parser.add_argument("--auto-all-threshold", type=int, default=100,
                    help="Dataset size threshold below which all data is used (default: 100)")
    args, _ = parser.parse_known_args()
    
    # =================================================================
    # 集中設定區：在此處自由修改您的專案設定
    # =================================================================
    # 1. 指定您這次想要優化的 SKF 配對
    SKF_PAIRS_TO_OPTIMIZE = [""]
    
    # 2. 在此處加入任何您想客製化的 DFTB+ HSD 內容
    EXTRA_HSD_BLOCK = """
ExcitedState {
    Casida {
        NrOfExcitations = 10
        Symmetry = Singlet
        Diagonaliser = Arpack{}
    }
}
""".strip()
    # =================================================================
    
    try:
        # 初始化專案
        project = SKFOptimizationProject(
            args=args,
            skf_pairs_to_optimize=SKF_PAIRS_TO_OPTIMIZE,
            extra_hsd=EXTRA_HSD_BLOCK
        )
        
        # 執行專案流程
        project.setup()
        project.run()
        
    except Exception as e:
        import traceback
        logging.getLogger("skf_opt").exception(f"\nFATAL ERROR: Script execution failed. Reason: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
