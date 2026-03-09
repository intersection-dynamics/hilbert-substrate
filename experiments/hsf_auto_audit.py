#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hsf_auto_audit.py
=================

One-shot, self-running audit for HSF endpoint/link extraction.

Run:
python hsf_auto_audit.py
"""

from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# -----------------------------
# Filesystem helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Core math (robust, simple)
# -----------------------------

def hermitize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)

def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - np.trace(A) * np.eye(d, dtype=A.dtype) / d

def hs_inner(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.real(np.trace(A.conj().T @ B)))

def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, hs_inner(A, A))))

def normalize_hs(A: np.ndarray) -> np.ndarray:
    n = hs_norm(A)
    if n < 1e-15:
        return A.copy()
    return A / n

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def gram_stats(basis: List[np.ndarray]) -> Dict[str, float]:
    k = len(basis)
    if k == 0:
        return {"diag_mean": 0.0, "offdiag_maxabs": 0.0, "offdiag_meanabs": 0.0}
    G = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            G[i, j] = hs_inner(basis[i], basis[j])
    off = G - np.diag(np.diag(G))
    return {
        "diag_mean": float(np.mean(np.diag(G))),
        "offdiag_maxabs": float(np.max(np.abs(off))),
        "offdiag_meanabs": float(np.mean(np.abs(off))),
    }

def closure_residual_correct(basis: List[np.ndarray]) -> Dict[str, float]:
    """
    Correct closure diagnostic for Hermitian basis:
      Y = -i[T_a, T_b]  (Hermitian)
    Project Y into span{T_c} and measure residual norms.
    """
    k = len(basis)
    if k == 0:
        return {"mean": 0.0, "max": 0.0}

    G = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            G[i, j] = hs_inner(basis[i], basis[j])
    Ginv = np.linalg.pinv(G, rcond=1e-12)

    def project(Y: np.ndarray) -> np.ndarray:
        ip = np.array([hs_inner(basis[i], Y) for i in range(k)], dtype=float)
        alpha = Ginv @ ip
        P = np.zeros_like(Y)
        for i in range(k):
            P += alpha[i] * basis[i]
        return P

    vals = []
    for a in range(k):
        for b in range(k):
            Y = (-1.0j) * comm(basis[a], basis[b])
            P = project(Y)
            vals.append(hs_norm(Y - P))
    v = np.array(vals, dtype=float)
    return {"mean": float(v.mean()), "max": float(v.max())}

def cross_energy(L: List[np.ndarray], R: List[np.ndarray]) -> Dict[str, float]:
    k = min(len(L), len(R))
    if k == 0:
        return {"k_used": 0, "energy_sum_sq": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0, "min": 0.0}
    vals = []
    for a in range(k):
        for b in range(k):
            vals.append(hs_norm(comm(L[a], R[b])))
    v = np.array(vals, dtype=float)
    return {
        "k_used": int(k),
        "energy_sum_sq": float(np.sum(v * v)),
        "max": float(v.max()),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "min": float(v.min()),
    }

def commuting_min_eigs(L: List[np.ndarray], R: List[np.ndarray]) -> Dict[str, List[float]]:
    """
    Smallest few eigenvalues of commuting quadratic forms:
      ML(v)=Σ_b ||[Σ_a v_a L_a, R_b]||^2
      MR(w)=Σ_b ||[Σ_a w_a R_a, L_b]||^2
    """
    k = min(len(L), len(R))
    if k == 0:
        return {"L_min_eigs": [], "R_min_eigs": [], "k_used": 0}

    ML = np.zeros((k, k), dtype=float)
    MR = np.zeros((k, k), dtype=float)

    for a in range(k):
        for ap in range(k):
            s = 0.0
            for b in range(k):
                C1 = comm(L[a], R[b])
                C2 = comm(L[ap], R[b])
                s += hs_inner(C1, C2)
            ML[a, ap] = s

    for a in range(k):
        for ap in range(k):
            s = 0.0
            for b in range(k):
                C1 = comm(R[a], L[b])
                C2 = comm(R[ap], L[b])
                s += hs_inner(C1, C2)
            MR[a, ap] = s

    ML = 0.5 * (ML + ML.T)
    MR = 0.5 * (MR + MR.T)

    eL = np.sort(np.real(np.linalg.eigvalsh(ML)))
    eR = np.sort(np.real(np.linalg.eigvalsh(MR)))

    m = min(5, k)
    return {"k_used": int(k), "L_min_eigs": eL[:m].tolist(), "R_min_eigs": eR[:m].tolist()}


# -----------------------------
# NPZ discovery + loading
# -----------------------------

@dataclass
class BasisPair:
    file: str
    mode: str
    full: bool
    k: int
    d: int
    L: List[np.ndarray]
    R: List[np.ndarray]
    keys_present: List[str]
    keys_used: Dict[str, str]

def find_npz_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".npz"):
                out.append(os.path.join(dirpath, fn))
    return out

def choose_latest(files: List[str], name_contains: Optional[str] = None) -> Optional[str]:
    cand = []
    for f in files:
        if name_contains is None or name_contains in os.path.basename(f):
            try:
                cand.append((os.path.getmtime(f), f))
            except Exception:
                pass
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]

def load_basis_pair(npz_path: str, mode: str, prefer_full: bool = True) -> Optional[BasisPair]:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None

    keys = list(data.keys())

    def key(which: str, full: bool) -> str:
        return f"basis_{which}_{mode}_full" if full else f"basis_{which}_{mode}"

    full_ok = (key("left", True) in keys) and (key("right", True) in keys)
    std_ok = (key("left", False) in keys) and (key("right", False) in keys)

    use_full = prefer_full and full_ok
    if not use_full and not std_ok:
        return None

    kL = key("left", use_full)
    kR = key("right", use_full)

    arrL = data[kL]
    arrR = data[kR]

    if arrL.ndim != 3 or arrR.ndim != 3 or arrL.shape[1] != arrL.shape[2] or arrR.shape[1] != arrR.shape[2]:
        return None
    if arrL.shape != arrR.shape:
        return None

    k, d, _ = arrL.shape

    L = []
    R = []
    for i in range(k):
        X = np.array(arrL[i], dtype=complex)
        X = normalize_hs(traceless(hermitize(X)))
        L.append(X)
        Y = np.array(arrR[i], dtype=complex)
        Y = normalize_hs(traceless(hermitize(Y)))
        R.append(Y)

    return BasisPair(
        file=npz_path,
        mode=mode,
        full=use_full,
        k=int(k),
        d=int(d),
        L=L,
        R=R,
        keys_present=keys,
        keys_used={"L": kL, "R": kR},
    )

def classify_dataset(npz_path: str) -> str:
    bn = os.path.basename(npz_path).lower()
    if "qutrit" in bn and "lr" in bn:
        return "qutrit_lr"
    if "link9" in bn and "lr" in bn:
        return "link9_lr"
    if "link9" in bn:
        return "link9"
    if "qutrit" in bn:
        return "qutrit"
    return "unknown"


# -----------------------------
# Audit runner
# -----------------------------

def verdict_text(result: Dict) -> str:
    clL = result["closure"]["L"]["mean"]
    clR = result["closure"]["R"]["mean"]
    k = result["dims"]["k"]
    d = result["dims"]["d_link"]
    crossE = result["cross"]["energy_sum_sq"]
    min_eL = result["commuting"]["L_min_eigs"][0] if result["commuting"]["L_min_eigs"] else None
    min_eR = result["commuting"]["R_min_eigs"][0] if result["commuting"]["R_min_eigs"] else None

    closes_L = (clL < 1e-6)
    closes_R = (clR < 1e-6)
    has_comm = (min_eL is not None and min_eL < 1e-6) or (min_eR is not None and min_eR < 1e-6)

    lines = []
    lines.append(f"k={k}, d_link={d}")
    lines.append(f"Correct closure mean residual: L={clL:.3e}, R={clR:.3e}  -> closes(L)={closes_L}, closes(R)={closes_R}")
    lines.append(f"Cross noncommutativity energy Σ||[L,R]||^2 = {crossE:.6g}")
    if min_eL is not None:
        lines.append(f"Min commuting eigs: L={min_eL:.3e}, R={min_eR:.3e} -> any commuting direction={has_comm}")
    else:
        lines.append("Min commuting eigs: (none)")
    return "\n".join(lines)

def analyze_pair(bp: BasisPair) -> Dict:
    gL = gram_stats(bp.L)
    gR = gram_stats(bp.R)
    clL = closure_residual_correct(bp.L)
    clR = closure_residual_correct(bp.R)
    ce = cross_energy(bp.L, bp.R)
    cm = commuting_min_eigs(bp.L, bp.R)

    return {
        "file": bp.file,
        "mode": bp.mode,
        "used_full": bp.full,
        "keys_used": bp.keys_used,
        "dims": {"k": bp.k, "d_link": bp.d},
        "gram": {"L": gL, "R": gR},
        "closure": {"L": clL, "R": clR},
        "cross": ce,
        "commuting": cm,
    }

def main():
    here = os.path.abspath(os.getcwd())
    roots = [here]
    hsf_out = os.path.join(here, "hsf_out")
    if os.path.isdir(hsf_out):
        roots.append(hsf_out)

    files = []
    for r in roots:
        files.extend(find_npz_files(r))
    files = sorted(set(files))

    if not files:
        print("No .npz files found under current directory or ./hsf_out. Nothing to do.")
        sys.exit(1)

    qutrit_npz = choose_latest(files, "qutrit_su3_lr_bases") or choose_latest(files, "qutrit_su3_lr") or choose_latest(files, "qutrit")
    link9_npz = choose_latest(files, "link9_su3_lr_bases") or choose_latest(files, "link9_su3_lr") or choose_latest(files, "link9")

    selected = []
    if qutrit_npz:
        selected.append(qutrit_npz)
    if link9_npz and link9_npz != qutrit_npz:
        selected.append(link9_npz)

    if not selected:
        files_by_time = sorted([(os.path.getmtime(f), f) for f in files], key=lambda x: x[0], reverse=True)
        selected = [x[1] for x in files_by_time[:2]]

    results = []
    for f in selected:
        for mode in ("aligned", "mixed"):
            bp = load_basis_pair(f, mode=mode, prefer_full=True)
            if bp is None:
                bp = load_basis_pair(f, mode=mode, prefer_full=False)
            if bp is None:
                continue
            res = analyze_pair(bp)
            res["dataset_guess"] = classify_dataset(f)
            results.append(res)

    if not results:
        print("Found NPZ files, but none contained basis_left/basis_right keys in aligned/mixed.")
        print("Files checked:")
        for f in selected:
            print("  ", f)
        sys.exit(2)

    out_dir = os.path.join(hsf_out if os.path.isdir(hsf_out) else here, "audit")
    ensure_dir(out_dir)
    tag = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{tag}_audit.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({"results": results}, fp, indent=2)

    print("\n====================== HSF AUTO AUDIT ======================\n")
    for r in results:
        print(f"FILE: {os.path.basename(r['file'])}   mode={r['mode']}   full={r['used_full']}   guess={r['dataset_guess']}")
        print(verdict_text(r))
        print("-" * 58)
    print(f"\n[SAVED REPORT] {out_path}\n")
    print("============================================================\n")


if __name__ == "__main__":
    main()