"""
Task D Part 3 (GPU-capable): Confinement — Consolidated Runner (Echo model)

Run from:
  C:\GitHub\hilbert_substrate\experiments>python conf_part3.py --mode both

What it does:
  - Ladder (Part 1): V(R) = -ln|<W>|; fits V(R)=sigma*R+const.
  - Grid   (Part 2): area-vs-perimeter discrimination using rectangular Wilson loops.
  - Adds: Creutz ratios (grid), standardized JSON+CSV output, GPU backend option.

GPU:
  - Uses CuPy + cupyx.scipy.sparse.linalg.eigsh with a matrix-free LinearOperator.
  - If CuPy is missing, falls back to NumPy+SciPy.
  - WARNING: This is still exponential in number of bonds.

Notes:
  - Wilson loop operator is implemented as product of Pauli-X on the bonds along the loop.
    This matches Parts 1–2 assumption: link variable encoded in X basis.

Examples (Windows one-liners):
  python conf_part3.py --mode both --backend auto
  python conf_part3.py --mode ladder --backend cupy --ladder_L 4,5,6,7 --couplings 0.15,0.25,0.35,0.45
  python conf_part3.py --mode grid --backend auto --grid_sizes 3x3,4x3,3x4 --couplings 0.25,0.35,0.45
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# -------------------------
# Imports: your echo model
# -------------------------
# These are expected to exist in your repo (as in Parts 1/2/3).
# If their location differs, tweak PYTHONPATH or adjust imports accordingly.
try:
    from bond_hamiltonian_b1 import EchoLattice, decompose_in_pauli, hamming_weight
    from bond_hamiltonian_final import exact_H_eff
except Exception as e:
    raise ImportError(
        "Could not import bond_hamiltonian_b1 / bond_hamiltonian_final.\n"
        "Make sure you run from your repo experiments directory and the modules are importable.\n"
        f"Original import error: {e}"
    )

# -------------------------
# Backend selection
# -------------------------
def get_backend(name: str):
    name = name.lower().strip()
    if name not in ("auto", "numpy", "cupy"):
        raise ValueError("--backend must be one of: auto, numpy, cupy")

    if name in ("auto", "cupy"):
        try:
            import cupy as cp
            import cupyx
            import cupyx.scipy.sparse.linalg as cpx_linalg
            return "cupy", cp, cpx_linalg
        except Exception:
            if name == "cupy":
                raise
            # fall through to numpy

    import numpy as np
    from scipy.sparse.linalg import LinearOperator as spLinearOperator
    from scipy.sparse.linalg import eigsh as sp_eigsh
    return "numpy", np, (spLinearOperator, sp_eigsh)

# -------------------------
# Pauli-term representation
# -------------------------
@dataclass
class PauliTerm:
    coeff: complex
    flip_mask: int         # bits to flip (X or Y)
    z_mask: int            # bits that contribute Z sign
    y_bits: Tuple[int, ...]  # positions (bit indices) that are Y (for ±i phases)

def _bitpos(n_qubits: int, qubit_idx: int) -> int:
    # We interpret qubit 0 as most-significant bit for consistency with earlier code
    return n_qubits - 1 - qubit_idx

def pauli_label_to_term(label: str, coeff: complex, n_qubits: int) -> PauliTerm:
    flip_mask = 0
    z_mask = 0
    y_bits: List[int] = []
    for q, ch in enumerate(label):
        if ch == "I":
            continue
        bit = _bitpos(n_qubits, q)
        if ch == "X":
            flip_mask |= (1 << bit)
        elif ch == "Y":
            flip_mask |= (1 << bit)
            y_bits.append(bit)
        elif ch == "Z":
            z_mask |= (1 << bit)
        else:
            raise ValueError(f"Unknown Pauli char: {ch}")
    return PauliTerm(coeff=coeff, flip_mask=flip_mask, z_mask=z_mask, y_bits=tuple(y_bits))

# -------------------------
# Reference term extraction (4-bond plaquette)
# -------------------------
def get_reference_terms(coupling: float = 0.3) -> Tuple[Dict[str, complex], Dict[str, complex]]:
    """
    Extract averaged single-bond (weight-1) and plaquette (weight-4) Pauli terms
    from a 4-bond reference plaquette, matching your Part 1/2 logic.
    """
    ref = EchoLattice(4, [(0, 1), (1, 2), (2, 3), (0, 3)], d_B=2)
    H_ref = exact_H_eff(ref, coupling=coupling)
    coeffs = decompose_in_pauli(H_ref, 4)

    w1_terms: Dict[int, Dict[str, complex]] = {}
    w4_terms: Dict[str, complex] = {}

    for label, c in coeffs.items():
        if abs(c) < 1e-14:
            continue
        w = hamming_weight(label)
        if w == 1:
            for pos, ch in enumerate(label):
                if ch != "I":
                    w1_terms.setdefault(pos, {})[ch] = c
        elif w == 4:
            chars = "".join(ch for ch in label if ch != "I")
            w4_terms[chars] = c

    # Average single-bond terms across positions
    tmp: Dict[str, List[complex]] = {}
    for _, terms in w1_terms.items():
        for ch, c in terms.items():
            tmp.setdefault(ch, []).append(c)
    avg_w1: Dict[str, complex] = {ch: complex(sum(vals) / len(vals)) for ch, vals in tmp.items()}

    return avg_w1, w4_terms

# -------------------------
# Lattices
# -------------------------
@dataclass
class Ladder:
    L: int

    def __post_init__(self):
        L = self.L
        self.n_bonds = 3 * L - 2
        self.dim = 2 ** self.n_bonds

        self.top = list(range(0, L - 1))
        self.bot = list(range(L - 1, 2 * (L - 1)))
        self.rung = list(range(2 * (L - 1), 3 * L - 2))

        # Plaquettes: [top[i], rung[i+1], bot[i], rung[i]]
        self.plaquettes: List[List[int]] = []
        for i in range(L - 1):
            self.plaquettes.append([self.top[i], self.rung[i + 1], self.bot[i], self.rung[i]])

    def wilson_loop(self, start: int, width: int) -> Optional[List[int]]:
        if start + width >= self.L:
            return None
        bonds: List[int] = []
        for i in range(start, start + width):     # top
            bonds.append(self.top[i])
        bonds.append(self.rung[start + width])    # right rung
        for i in range(start + width - 1, start - 1, -1):  # bottom reverse
            bonds.append(self.bot[i])
        bonds.append(self.rung[start])            # left rung
        return bonds

@dataclass
class Grid:
    Nx: int
    Ny: int

    def __post_init__(self):
        # Bonds: horizontal + vertical on open boundaries
        # index mapping:
        #  horiz: (x,y)-> id in [0, (Nx-1)*Ny)
        #  vert : (x,y)-> id in [horiz_count, horiz_count + Nx*(Ny-1))
        self.horiz_count = (self.Nx - 1) * self.Ny
        self.vert_count = self.Nx * (self.Ny - 1)
        self.n_bonds = self.horiz_count + self.vert_count
        self.dim = 2 ** self.n_bonds

    def hbond(self, x: int, y: int) -> int:
        # bond from (x,y) to (x+1,y)
        return y * (self.Nx - 1) + x

    def vbond(self, x: int, y: int) -> int:
        # bond from (x,y) to (x,y+1)
        return self.horiz_count + y * self.Nx + x

    def rect_loop_bonds(self, x0: int, y0: int, w: int, h: int) -> List[int]:
        """
        Rectangle of width w and height h in plaquettes:
          corners: (x0,y0) to (x0+w, y0+h)
        """
        bonds: List[int] = []
        # bottom edge (right)
        for x in range(x0, x0 + w):
            bonds.append(self.hbond(x, y0))
        # right edge (up)
        for y in range(y0, y0 + h):
            bonds.append(self.vbond(x0 + w, y))
        # top edge (left)
        for x in range(x0 + w - 1, x0 - 1, -1):
            bonds.append(self.hbond(x, y0 + h))
        # left edge (down)
        for y in range(y0 + h - 1, y0 - 1, -1):
            bonds.append(self.vbond(x0, y))
        return bonds

    def enumerate_rectangles(self) -> List[Tuple[int,int,int,int,int,int]]:
        """
        Return list of rectangles with metadata:
          (w, h, area, perimeter, x0, y0)
        where w,h are in plaquettes, area=w*h, perimeter=2*(w+h)
        """
        rects = []
        for w in range(1, self.Nx):         # plaquettes width
            for h in range(1, self.Ny):     # plaquettes height
                for x0 in range(0, self.Nx - w):
                    for y0 in range(0, self.Ny - h):
                        area = w * h
                        perim = 2 * (w + h)
                        rects.append((w, h, area, perim, x0, y0))
        return rects

# -------------------------
# Matrix-free apply of Pauli terms
# -------------------------
def apply_pauli_terms(xp, v, terms: List[PauliTerm]):
    """
    Compute y = sum_t coeff_t * P_t @ v, where each P_t is a Pauli string.

    v is xp array (np or cp) complex128
    """
    dim = v.shape[0]
    idx = xp.arange(dim, dtype=xp.int64)
    y = xp.zeros_like(v)

    # helpers if available
    has_bitcount = hasattr(xp, "bit_count")  # cupy has this in newer versions; numpy has it too

    for t in terms:
        j = idx ^ t.flip_mask
        phase = xp.ones(dim, dtype=xp.complex128)

        if t.z_mask != 0:
            masked = idx & t.z_mask
            if has_bitcount:
                parity = xp.bit_count(masked) & 1
            else:
                # slow fallback: xor-fold parity (works on numpy/cupy but slower)
                parity = xp.zeros(dim, dtype=xp.int8)
                tmp = masked.copy()
                while True:
                    parity ^= (tmp & 1).astype(xp.int8)
                    tmp >>= 1
                    if int(xp.max(tmp).get() if xp.__name__ == "cupy" else tmp.max()) == 0:
                        break
            phase *= xp.where(parity == 0, 1.0, -1.0)

        # Y phases: multiply by +i if input bit=0, -i if input bit=1, per Y bit
        # Equivalent to (1j) * (-1)**bit for each Y position
        for b in t.y_bits:
            inp = (idx >> b) & 1
            phase *= xp.where(inp == 0, 1j, -1j)

        y += t.coeff * phase * v[j]

    return y

# -------------------------
# Build Hamiltonian term lists
# -------------------------
def build_ladder_terms(lad: Ladder, coupling: float, ref_cache: Dict[float, Tuple[Dict[str, complex], Dict[str, complex]]]) -> List[PauliTerm]:
    if coupling not in ref_cache:
        ref_cache[coupling] = get_reference_terms(coupling)
    w1, w4 = ref_cache[coupling]

    terms: List[PauliTerm] = []
    n = lad.n_bonds

    # single-bond terms
    for b in range(n):
        for ch, c in w1.items():
            label = ["I"] * n
            label[b] = ch
            terms.append(pauli_label_to_term("".join(label), c, n))

    # plaquette terms: map 4-bond pattern across each plaquette
    # pattern keys are strings like "XXXX", "ZZZZ", etc. (no I)
    for plaq in lad.plaquettes:
        for pat, c in w4.items():
            if len(pat) != 4:
                continue
            label = ["I"] * n
            for k, ch in enumerate(pat):
                label[plaq[k]] = ch
            terms.append(pauli_label_to_term("".join(label), c, n))

    return terms

def build_grid_terms(gr: Grid, coupling: float, ref_cache: Dict[float, Tuple[Dict[str, complex], Dict[str, complex]]]) -> List[PauliTerm]:
    if coupling not in ref_cache:
        ref_cache[coupling] = get_reference_terms(coupling)
    w1, w4 = ref_cache[coupling]

    terms: List[PauliTerm] = []
    n = gr.n_bonds

    # single-bond terms
    for b in range(n):
        for ch, c in w1.items():
            label = ["I"] * n
            label[b] = ch
            terms.append(pauli_label_to_term("".join(label), c, n))

    # plaquettes on open grid: each cell has 4 bonds in order [bottom, right, top, left]
    # bottom: hbond(x,y)
    # right : vbond(x+1,y)
    # top   : hbond(x,y+1)
    # left  : vbond(x,y)
    for x in range(gr.Nx - 1):
        for y in range(gr.Ny - 1):
            plaq = [gr.hbond(x, y), gr.vbond(x + 1, y), gr.hbond(x, y + 1), gr.vbond(x, y)]
            for pat, c in w4.items():
                if len(pat) != 4:
                    continue
                label = ["I"] * n
                for k, ch in enumerate(pat):
                    label[plaq[k]] = ch
                terms.append(pauli_label_to_term("".join(label), c, n))

    return terms

# -------------------------
# Ground state solver
# -------------------------
def solve_ground_state(backend_name: str, xp, linalg, terms: List[PauliTerm], dim: int,
                       dense_cutoff: int, maxiter: int, tol: float):
    """
    Returns (E0, psi0, gap, method)
    """
    if backend_name == "cupy":
        cp = xp
        cpx_linalg = linalg
        # Dense cutoff on GPU (still expensive, but ok for <= 4096-ish)
        if dim <= dense_cutoff:
            # Build dense H by applying terms to basis vectors (slow but manageable at small dim)
            H = cp.zeros((dim, dim), dtype=cp.complex128)
            eye = cp.eye(dim, dtype=cp.complex128)
            for k in range(dim):
                H[:, k] = apply_pauli_terms(cp, eye[:, k], terms)
            w, v = cp.linalg.eigh(H)
            E0 = float(cp.asnumpy(w[0].real))
            psi0 = cp.asnumpy(v[:, 0])
            gap = float(cp.asnumpy((w[1] - w[0]).real))
            return E0, psi0, gap, "cupy-dense"

        # Matrix-free eigsh via LinearOperator
        def mv(vec):
            return apply_pauli_terms(cp, vec, terms)

        A = cpx_linalg.LinearOperator((dim, dim), matvec=mv, dtype=cp.complex128)
        # Compute lowest 2 eigenpairs
        w, v = cpx_linalg.eigsh(A, k=2, which="SA", maxiter=maxiter, tol=tol)
        # sort
        w = cp.asnumpy(w)
        v = cp.asnumpy(v)
        idx = w.argsort()
        E0 = float(w[idx[0]].real)
        psi0 = v[:, idx[0]]
        gap = float((w[idx[1]] - w[idx[0]]).real)
        return E0, psi0, gap, "cupy-eigsh"

    # numpy/scipy backend
    import numpy as np
    from scipy.sparse.linalg import LinearOperator as spLinearOperator
    from scipy.sparse.linalg import eigsh as sp_eigsh

    if dim <= dense_cutoff:
        # dense build by matvec on basis vectors (same idea, but small only)
        H = np.zeros((dim, dim), dtype=np.complex128)
        eye = np.eye(dim, dtype=np.complex128)
        for k in range(dim):
            H[:, k] = apply_pauli_terms(np, eye[:, k], terms)
        w, v = np.linalg.eigh(H)
        return float(w[0].real), v[:, 0], float((w[1] - w[0]).real), "numpy-dense"

    def mv(vec):
        return apply_pauli_terms(np, vec, terms)

    A = spLinearOperator((dim, dim), matvec=mv, dtype=np.complex128)
    w, v = sp_eigsh(A, k=2, which="SA", maxiter=maxiter, tol=tol)
    idx = w.argsort()
    return float(w[idx[0]].real), v[:, idx[0]], float((w[idx[1]] - w[idx[0]]).real), "scipy-eigsh"

# -------------------------
# Wilson loop / potentials
# -------------------------
def wilson_expval_x(loop_bonds: List[int], psi: Any) -> float:
    """
    Expectation of product of X on loop bonds.
    psi: numpy array (CPU). We always keep psi on CPU for reporting simplicity.
    """
    import numpy as np
    n = int(round(math.log2(psi.shape[0])))
    dim = psi.shape[0]
    idx = np.arange(dim, dtype=np.int64)

    flip_mask = 0
    for b in loop_bonds:
        bit = _bitpos(n, b)
        flip_mask |= (1 << bit)

    j = idx ^ flip_mask
    # X has no phase; just permutation
    val = np.vdot(psi, psi[j]).real
    return float(val)

def potential_from_w(W: float, eps: float) -> float:
    # V = -ln(|W|) but clamp to avoid inf/nan
    a = max(abs(W), eps)
    return float(-math.log(a))

def linear_fit(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """
    Fit y = a*x + b. Returns (a, b, R^2).
    """
    import numpy as np
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a * x + b
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r2)

def multi_fit_area_perim(areas: List[float], perims: List[float], Vs: List[float]) -> Tuple[float, float, float, float]:
    """
    Fit V = a*Area + b*Perim + c. Returns (a, b, c, R^2).
    """
    import numpy as np
    A = np.vstack([np.array(areas), np.array(perims), np.ones(len(Vs))]).T
    y = np.array(Vs)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    a, b, c = coeffs
    yhat = A @ coeffs
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(c), float(r2)

def creutz_ratio(W_RT: float, W_Rm1T: float, W_RTm1: float, W_Rm1Tm1: float, eps: float) -> Optional[float]:
    # chi = -ln( W(R,T)*W(R-1,T-1) / (W(R-1,T)*W(R,T-1)) )
    # Use abs values and eps clamp; skip if any are too tiny or non-finite.
    vals = [W_RT, W_Rm1T, W_RTm1, W_Rm1Tm1]
    if any((not math.isfinite(v)) for v in vals):
        return None
    if any(abs(v) < eps for v in vals):
        return None
    num = abs(W_RT) * abs(W_Rm1Tm1)
    den = abs(W_Rm1T) * abs(W_RTm1)
    if den <= 0 or num <= 0:
        return None
    return float(-math.log(num / den))

# -------------------------
# Runs
# -------------------------
def run_ladder(backend_name: str, xp, linalg, couplings: List[float], ladder_L: List[int],
               dense_cutoff: int, maxiter: int, tol: float, eps: float,
               outdir: str, ref_cache: Dict[float, Tuple[Dict[str, complex], Dict[str, complex]]]) -> Dict[str, Any]:

    results = {"runs": []}

    for L in ladder_L:
        lad = Ladder(L)
        for g in couplings:
            terms = build_ladder_terms(lad, g, ref_cache)

            E0, psi0, gap, method = solve_ground_state(
                backend_name, xp, linalg, terms, lad.dim,
                dense_cutoff=dense_cutoff, maxiter=maxiter, tol=tol
            )

            # Compute V(R) using largest loop at start=0 for each width
            Rs: List[int] = []
            Vs: List[float] = []
            Ws: List[float] = []
            for width in range(1, L):  # max width is L-1
                loop = lad.wilson_loop(0, width)
                if loop is None:
                    continue
                W = wilson_expval_x(loop, psi0)
                V = potential_from_w(W, eps)
                Rs.append(width)
                Vs.append(V)
                Ws.append(W)

            sigma, c0, r2 = linear_fit([float(r) for r in Rs], Vs) if len(Rs) >= 2 else (float("nan"), float("nan"), float("nan"))

            results["runs"].append({
                "L": L,
                "g": g,
                "n_bonds": lad.n_bonds,
                "dim": lad.dim,
                "solver": method,
                "E0": E0,
                "gap": gap,
                "R": Rs,
                "W": Ws,
                "V": Vs,
                "sigma": sigma,
                "intercept": c0,
                "R2": r2,
            })

    # CSV summary
    csv_path = os.path.join(outdir, "taskD_part3_ladder_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["L", "g", "n_bonds", "dim", "solver", "E0", "gap", "sigma", "intercept", "R2"])
        for r in results["runs"]:
            w.writerow([r["L"], r["g"], r["n_bonds"], r["dim"], r["solver"], r["E0"], r["gap"], r["sigma"], r["intercept"], r["R2"]])

    return results

def run_grid(backend_name: str, xp, linalg, couplings: List[float], grid_sizes: List[Tuple[int,int]],
             dense_cutoff: int, maxiter: int, tol: float, eps: float,
             outdir: str, ref_cache: Dict[float, Tuple[Dict[str, complex], Dict[str, complex]]]) -> Dict[str, Any]:

    results = {"runs": []}

    for (Nx, Ny) in grid_sizes:
        gr = Grid(Nx, Ny)
        rects = gr.enumerate_rectangles()

        for g in couplings:
            terms = build_grid_terms(gr, g, ref_cache)
            E0, psi0, gap, method = solve_ground_state(
                backend_name, xp, linalg, terms, gr.dim,
                dense_cutoff=dense_cutoff, maxiter=maxiter, tol=tol
            )

            # Collect loop values
            loop_records = []
            for (w_, h_, area, perim, x0, y0) in rects:
                bonds = gr.rect_loop_bonds(x0, y0, w_, h_)
                W = wilson_expval_x(bonds, psi0)
                V = potential_from_w(W, eps)
                loop_records.append({
                    "w": w_, "h": h_, "area": area, "perim": perim, "x0": x0, "y0": y0,
                    "W": W, "V": V
                })

            # Regression tests (filtering: all are finite because we clamp)
            areas = [float(r["area"]) for r in loop_records]
            perims = [float(r["perim"]) for r in loop_records]
            Vs = [float(r["V"]) for r in loop_records]

            aA, cA, r2A = linear_fit(areas, Vs) if len(Vs) >= 2 else (float("nan"), float("nan"), float("nan"))
            aP, cP, r2P = linear_fit(perims, Vs) if len(Vs) >= 2 else (float("nan"), float("nan"), float("nan"))
            a, b, c, r2AP = multi_fit_area_perim(areas, perims, Vs) if len(Vs) >= 3 else (float("nan"), float("nan"), float("nan"), float("nan"))

            # Creutz ratios for rectangles: treat w as R, h as T (only where both >=2)
            # We need mean W for each (w,h) type across positions to reduce noise/position bias.
            by_shape: Dict[Tuple[int,int], List[float]] = {}
            for r in loop_records:
                by_shape.setdefault((r["w"], r["h"]), []).append(r["W"])
            Wmean = {k: float(sum(v) / len(v)) for k, v in by_shape.items()}

            creutz: Dict[str, float] = {}
            for (w_, h_) in list(Wmean.keys()):
                if w_ < 2 or h_ < 2:
                    continue
                key = f"{w_}x{h_}"
                W_RT = Wmean.get((w_, h_), None)
                W_Rm1T = Wmean.get((w_-1, h_), None)
                W_RTm1 = Wmean.get((w_, h_-1), None)
                W_Rm1Tm1 = Wmean.get((w_-1, h_-1), None)
                if None in (W_RT, W_Rm1T, W_RTm1, W_Rm1Tm1):
                    continue
                chi = creutz_ratio(W_RT, W_Rm1T, W_RTm1, W_Rm1Tm1, eps=eps)
                if chi is not None and math.isfinite(chi):
                    creutz[key] = chi

            results["runs"].append({
                "Nx": Nx, "Ny": Ny,
                "g": g,
                "n_bonds": gr.n_bonds,
                "dim": gr.dim,
                "solver": method,
                "E0": E0,
                "gap": gap,
                "area_fit_sigma": aA,
                "area_fit_intercept": cA,
                "R2_area": r2A,
                "perim_fit_mu": aP,
                "perim_fit_intercept": cP,
                "R2_perim": r2P,
                "combined_sigma": a,
                "combined_mu": b,
                "combined_c": c,
                "R2_combined": r2AP,
                "creutz": creutz,
            })

    # CSV summary
    csv_path = os.path.join(outdir, "taskD_part3_grid_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Nx","Ny","g","n_bonds","dim","solver","E0","gap",
            "R2_area","R2_perim","R2_combined",
            "area_fit_sigma","perim_fit_mu","combined_sigma","combined_mu",
            "creutz_keys","creutz_values"
        ])
        for r in results["runs"]:
            keys = ";".join(sorted(r["creutz"].keys()))
            vals = ";".join([f"{r['creutz'][k]:.8f}" for k in sorted(r["creutz"].keys())])
            w.writerow([
                r["Nx"], r["Ny"], r["g"], r["n_bonds"], r["dim"], r["solver"], r["E0"], r["gap"],
                r["R2_area"], r["R2_perim"], r["R2_combined"],
                r["area_fit_sigma"], r["perim_fit_mu"], r["combined_sigma"], r["combined_mu"],
                keys, vals
            ])

    return results

# -------------------------
# CLI / main
# -------------------------
def parse_list_of_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_of_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_grid_sizes(s: str) -> List[Tuple[int,int]]:
    out = []
    for part in s.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError("grid size must look like 3x3,4x3,...")
        a,b = part.split("x", 1)
        out.append((int(a), int(b)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["both","ladder","grid"])
    ap.add_argument("--backend", default="auto", choices=["auto","numpy","cupy"],
                    help="auto uses CuPy if available; otherwise NumPy")
    ap.add_argument("--couplings", default="0.15,0.25,0.35,0.45")
    ap.add_argument("--ladder_L", default="4,5,6,7",
                    help="ladder lengths. WARNING: L=8 -> dim=2^(22)=4,194,304 (may work on GPU with enough VRAM)")
    ap.add_argument("--grid_sizes", default="3x3,4x3,3x4",
                    help="open grids (Nx x Ny). WARNING: 4x4 -> bonds=24 -> dim=16,777,216 (usually too big).")
    ap.add_argument("--dense_cutoff", type=int, default=4096,
                    help="if dim <= dense_cutoff, do dense diagonalization (GPU or CPU)")
    ap.add_argument("--maxiter", type=int, default=6000)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--eps", type=float, default=1e-300,
                    help="Clamp for Wilson loop magnitudes: V=-ln(max(|W|,eps)) to avoid infinities.")
    args = ap.parse_args()

    couplings = parse_list_of_floats(args.couplings)
    ladder_L = parse_list_of_ints(args.ladder_L)
    grid_sizes = parse_grid_sizes(args.grid_sizes)

    backend_name, xp, linalg = get_backend(args.backend)

    # Output directory relative to this file (experiments/hsf_out/...)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base_dir, "hsf_out")
    os.makedirs(out_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(out_root, f"{stamp}_taskD_part3")
    os.makedirs(outdir, exist_ok=True)

    ref_cache: Dict[float, Tuple[Dict[str, complex], Dict[str, complex]]] = {}

    all_results: Dict[str, Any] = {
        "meta": {
            "timestamp": stamp,
            "mode": args.mode,
            "backend": backend_name,
            "couplings": couplings,
            "ladder_L": ladder_L,
            "grid_sizes": grid_sizes,
            "dense_cutoff": args.dense_cutoff,
            "maxiter": args.maxiter,
            "tol": args.tol,
            "eps": args.eps,
        }
    }

    if args.mode in ("both", "ladder"):
        all_results["ladder"] = run_ladder(
            backend_name, xp, linalg,
            couplings=couplings,
            ladder_L=ladder_L,
            dense_cutoff=args.dense_cutoff,
            maxiter=args.maxiter,
            tol=args.tol,
            eps=args.eps,
            outdir=outdir,
            ref_cache=ref_cache
        )

    if args.mode in ("both", "grid"):
        all_results["grid"] = run_grid(
            backend_name, xp, linalg,
            couplings=couplings,
            grid_sizes=grid_sizes,
            dense_cutoff=args.dense_cutoff,
            maxiter=args.maxiter,
            tol=args.tol,
            eps=args.eps,
            outdir=outdir,
            ref_cache=ref_cache
        )

    json_path = os.path.join(outdir, "taskD_part3_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("============================================================")
    print("Task D Part 3 complete")
    print(f"Backend: {backend_name}")
    print(f"Output folder: {outdir}")
    print(f"JSON: {json_path}")
    if args.mode in ("both", "ladder"):
        print(f"CSV ladder: {os.path.join(outdir, 'taskD_part3_ladder_summary.csv')}")
    if args.mode in ("both", "grid"):
        print(f"CSV grid:   {os.path.join(outdir, 'taskD_part3_grid_summary.csv')}")
    print("============================================================")

if __name__ == "__main__":
    main()
