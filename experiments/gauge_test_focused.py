#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gauge_test_focused.py
=====================

Composite-link SU(3) gauge invariance + Gauss-kernel test
---------------------------------------------------------

System:
  site_A(3) ⊗ link(9=3⊗3) ⊗ site_B(3)  => 81-dimensional total Hilbert space.

Key fix vs earlier versions:
  To admit a Gauss-law singlet (nonempty Gauss subspace), each vertex must have
  a (3 ⊗ 3bar) or (3bar ⊗ 3) pairing, since:
      3 ⊗ 3bar = 1 ⊕ 8   (contains singlet)
      3 ⊗ 3    = 6 ⊕ 3bar (no singlet)
      3bar ⊗ 3bar = 3 ⊕ 6bar (no singlet)

We enforce:
  - site_A in anti-fundamental (3bar):   T_A^a = -(T^a)^T
  - link-left factor in fundamental (3): T_L^a =  T^a ⊗ I
  - link-right factor in anti-fund (3bar): T_R^a = I ⊗ (-(T^a)^T)
  - site_B in fundamental (3):           T_B^a =  T^a

Then:
  G_left^a  = T_A^a + (link-left)^a
  G_right^a = (link-right)^a + T_B^a
and we build H that commutes with both sets of Gauss generators.

Outputs:
  - prints gauge invariance commutator norms
  - prints Gauss-kernel dimension (dim of states with G^2 ~ 0)
  - writes JSON to ./hsf_out/gauge_tests/

No SciPy required.
"""

from __future__ import annotations

import os
import json
import time
import math
import numpy as np


# -------------------------
# Linear algebra helpers
# -------------------------

def hermitize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)

def traceless(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    return A - (np.trace(A) / d) * np.eye(d, dtype=complex)

def hs_inner(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.real(np.trace(A.conj().T @ B)))

def hs_norm(A: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, hs_inner(A, A))))

def fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))

def normalize_hs(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = hs_norm(A)
    if n < eps:
        return A.copy()
    return A / n

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for X in ops[1:]:
        out = np.kron(out, X)
    return out

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------
# su(3) generators (HS-orthonormal Hermitian traceless)
# -------------------------

def su_generators_gellmann_hs(d: int = 3) -> list[np.ndarray]:
    gens: list[np.ndarray] = []

    # symmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = 1.0
            M[j, i] = 1.0
            gens.append(M)

    # antisymmetric off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[i, j] = -1.0j
            M[j, i] = 1.0j
            gens.append(M)

    # diagonals (d-1)
    for k in range(1, d):
        M = np.zeros((d, d), dtype=complex)
        for i in range(k):
            M[i, i] = 1.0
        M[k, k] = -float(k)
        # standard normalization factor to match su_generators in our other scripts
        M = M * math.sqrt(2.0 / (k * (k + 1.0)))
        gens.append(M)

    out = []
    for G in gens:
        H = hermitize(traceless(G))
        out.append(normalize_hs(H))
    if len(out) != d * d - 1:
        raise RuntimeError("Generator count mismatch")
    return out


def structure_constants(T: list[np.ndarray]) -> np.ndarray:
    """
    For Hermitian HS-orthonormal basis:
      f^{abc} = (1/(2i)) Tr([T^a, T^b] T^c)
    """
    n = len(T)
    f = np.zeros((n, n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            C = comm(T[a], T[b])
            for c in range(n):
                f[a, b, c] = float((np.trace(C @ T[c]) / (2.0j)).real)
    return f


# -------------------------
# Build gauge-invariant H and Gauss generators (with singlet-admitting reps)
# -------------------------

def main() -> None:
    dS = 3
    dB = 9      # link = 3 ⊗ 3
    D = dS * dB * dS  # 81
    n_gen = 8

    I3 = np.eye(3, dtype=complex)
    I9 = np.eye(9, dtype=complex)

    # Fundamental su(3) generators on 3
    T_f = su_generators_gellmann_hs(3)                 # 8 generators
    # Anti-fundamental rep for Hermitian basis
    T_af = [-(X.T) for X in T_f]                       # 8 generators

    # Assign reps:
    T_A = T_af   # site_A is 3bar
    T_B = T_f    # site_B is 3

    # Link endpoint actions on 9 = 3 ⊗ 3
    # left factor: fundamental
    link_left = [kron(T_f[a], I3) for a in range(n_gen)]       # 9x9
    # right factor: anti-fundamental so it pairs with site_B(3) as (3bar ⊗ 3)
    link_right = [kron(I3, T_af[a]) for a in range(n_gen)]     # 9x9

    # Build two-leg Hamiltonian:
    # H = Σ_a [ (T_A^a ⊗ link_left^a ⊗ I)  +  (I ⊗ link_right^a ⊗ T_B^a) ]
    H = np.zeros((D, D), dtype=complex)
    for a in range(n_gen):
        H += kron(T_A[a], link_left[a], I3)
        H += kron(I3, link_right[a], T_B[a])
    H = hermitize(H)

    print("=" * 78)
    print("COMPOSITE LINK GAUGE INVARIANCE TEST (SINGLET-ADMITTING REPS)")
    print("System: site_A(3bar) ⊗ link(9=3⊗3) ⊗ site_B(3) = 81 dim")
    print("=" * 78)
    print(f"\n||H||_F = {fro(H):.6f}")
    print(f"Hermiticity: ||H-H†||_F = {fro(H - H.conj().T):.2e}")

    # Build Gauss generators:
    # G_L^a = T_A^a ⊗ I ⊗ I + I ⊗ (link_left^a) ⊗ I
    # G_R^a = I ⊗ (link_right^a) ⊗ I + I ⊗ I ⊗ T_B^a
    GL = []
    GR = []
    for a in range(n_gen):
        GL.append(kron(T_A[a], I9, I3) + kron(I3, link_left[a], I3))
        GR.append(kron(I3, link_right[a], I3) + kron(I3, I9, T_B[a]))

    # Gauge invariance check: max ||[H, G]||
    comm_L = np.array([fro(comm(H, G)) for G in GL], dtype=float)
    comm_R = np.array([fro(comm(H, G)) for G in GR], dtype=float)
    maxL = float(comm_L.max())
    maxR = float(comm_R.max())
    maxTot = max(maxL, maxR)

    print("\nGAUGE INVARIANCE:")
    print(f"  max ||[H, G_L^a]||_F = {maxL:.3e}")
    print(f"  max ||[H, G_R^a]||_F = {maxR:.3e}")
    print(f"  max total           = {maxTot:.3e}")
    gauge_invariant = (maxTot < 1e-10)
    print(f"  VERDICT: {'*** GAUGE INVARIANT ***' if gauge_invariant else 'NOT gauge invariant'}")

    # Independence check: [link_left, link_right] should be 0 because they act on different factors
    max_cross = 0.0
    for a in range(n_gen):
        for b in range(n_gen):
            max_cross = max(max_cross, fro(comm(link_left[a], link_right[b])))
    print("\nALGEBRA STRUCTURE:")
    print(f"  max ||[link_left^a, link_right^b]||_F = {max_cross:.2e} (0 = independent factors)")

    # Compare structure constants norms (optional sanity)
    f_site = structure_constants(T_f)
    # normalize link generators to HS-orthonormal (they already are if T_f is HS-orthonormal)
    TLn = [normalize_hs(traceless(hermitize(x))) for x in link_left]
    TRn = [normalize_hs(traceless(hermitize(x))) for x in link_right]
    f_left = structure_constants(TLn)
    f_right = structure_constants(TRn)

    print(f"  ||f_site||           = {np.linalg.norm(f_site):.6f}")
    print(f"  ||f_left - f_site||  = {np.linalg.norm(f_left - f_site):.2e}")
    print(f"  ||f_right - f_site|| = {np.linalg.norm(f_right - f_site):.2e}")

    # Gauss subspace: compute G^2 = Σ_a (G_L^a)^2 + (G_R^a)^2 and count near-zero eigenvalues
    print("\nGAUSS SUBSPACE:")
    G2 = np.zeros((D, D), dtype=complex)
    for a in range(n_gen):
        G2 += GL[a] @ GL[a] + GR[a] @ GR[a]
    G2 = hermitize(G2)

    # eigenvalues real (Hermitian). use a tolerance.
    evals = np.linalg.eigvalsh(G2.real)
    tol = 1e-8
    dim_gauss = int(np.sum(np.abs(evals) < tol))
    print(f"  Gauss subspace dimension: {dim_gauss} / {D}")
    print(f"  Smallest G² eigenvalues: {evals[:8]}")

    # Save JSON locally (Windows friendly)
    out_dir = os.path.join(os.path.dirname(__file__), "hsf_out", "gauge_tests")
    safe_mkdir(out_dir)
    tag = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gauge_test_focused_{tag}.json")

    out = {
        "timestamp": tag,
        "dims": {"site": 3, "link": 9, "total": 81},
        "representation_choice": {
            "site_A": "anti-fund (3bar): -(T^a)^T",
            "link_left": "fund on left factor: T^a ⊗ I",
            "link_right": "anti-fund on right factor: I ⊗ (-(T^a)^T)",
            "site_B": "fund (3): T^a",
            "singlet_condition": "3bar⊗3 contains singlet at each vertex",
        },
        "H_fro_norm": fro(H),
        "gauge_invariance": {
            "max_comm_L": maxL,
            "max_comm_R": maxR,
            "max_comm_total": maxTot,
            "passes": bool(gauge_invariant),
        },
        "algebra": {
            "max_comm_link_left_right": max_cross,
            "f_site_norm": float(np.linalg.norm(f_site)),
            "f_left_minus_site_norm": float(np.linalg.norm(f_left - f_site)),
            "f_right_minus_site_norm": float(np.linalg.norm(f_right - f_site)),
        },
        "gauss_subspace": {
            "tol": tol,
            "dim": dim_gauss,
            "smallest_eigs": [float(x) for x in evals[:12]],
        },
        "paths": {"saved_json": out_path},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[saved] {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()