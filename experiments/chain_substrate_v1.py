# chain_substrate_v1.py

r"""
HSF-style chain substrate (v1)

What this is
------------
A more HSF-shaped next step than the previous ecology toy:

- explicit local Hilbert states (qubits)
- reversible local unitary dynamics
- strictly local nearest-neighbor interactions
- inferred link/interface sectors from bond Schmidt spectra
- finite-bandwidth proxy via bond-dimension cap and truncation loss

What it is not
--------------
- not full exact many-body Hilbert simulation in arbitrary geometry
- not gauge-complete
- not a proof of no-refolding

It is a scalable 1D TEBD / MPS experiment that lets many subsystems run loose
while staying meaningfully quantum.

Interpretation
--------------
- subsystem = one site/qubit
- link/interface = Schmidt sector across a bond
- active link dimension = number of significant Schmidt values on that bond
- finite bandwidth = chi_max cap on kept Schmidt rank
- truncation loss = pressure from bandwidth limits
- clusters = contiguous regions of active bonds

Outputs
-------
summary.json
timeseries.csv
bond_entropy.png
bond_rank.png
largest_cluster.png
truncation_loss.png
final_bond_entropy_hist.png
final_bond_rank_hist.png

Example
-------
python chain_substrate_v1.py --outdir chain_out --sites 32 --steps 400 --chi_max 32 --seed 0

Bigger run
----------
python chain_substrate_v1.py --outdir chain_big --sites 48 --steps 600 --chi_max 48 --seed 0
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Basic linear algebra
# ============================================================

def dagger(x: np.ndarray) -> np.ndarray:
    return x.conj().T


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows, path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_state(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def matrix_exp_hermitian(H: np.ndarray, dt: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    return evecs @ np.diag(np.exp(-1j * dt * evals)) @ dagger(evecs)


# ============================================================
# Local operators
# ============================================================

def paulis():
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def random_single_site_unitary(rng: np.random.Generator, scale: float) -> np.ndarray:
    I, X, Y, Z = paulis()
    coeffs = rng.normal(size=3)
    H = coeffs[0] * X + coeffs[1] * Y + coeffs[2] * Z
    H = 0.5 * (H + dagger(H))
    return matrix_exp_hermitian(H, scale)


def build_two_site_gate(
    rng: np.random.Generator,
    dt: float,
    heisenberg_scale: float,
    field_scale: float,
) -> np.ndarray:
    I, X, Y, Z = paulis()

    jx, jy, jz = heisenberg_scale * rng.normal(size=3)
    hx1, hy1, hz1 = field_scale * rng.normal(size=3)
    hx2, hy2, hz2 = field_scale * rng.normal(size=3)

    H = (
        jx * np.kron(X, X) +
        jy * np.kron(Y, Y) +
        jz * np.kron(Z, Z) +
        hx1 * np.kron(X, I) +
        hy1 * np.kron(Y, I) +
        hz1 * np.kron(Z, I) +
        hx2 * np.kron(I, X) +
        hy2 * np.kron(I, Y) +
        hz2 * np.kron(I, Z)
    )
    H = 0.5 * (H + dagger(H))
    return matrix_exp_hermitian(H, dt)


# ============================================================
# MPS utilities
# ============================================================

def init_product_mps(n_sites: int, rng: np.random.Generator, init: str = "random") -> List[np.ndarray]:
    """
    MPS tensors A[i] with shape (Dl, d, Dr), d=2.
    Product-state initialization => all bond dims = 1.
    """
    mps = []
    for i in range(n_sites):
        if init == "random":
            v = rng.normal(size=2) + 1j * rng.normal(size=2)
            v = normalize_state(v)
        elif init == "up":
            v = np.array([1.0, 0.0], dtype=complex)
        elif init == "neel":
            v = np.array([1.0, 0.0], dtype=complex) if i % 2 == 0 else np.array([0.0, 1.0], dtype=complex)
        else:
            raise ValueError(f"Unknown init mode: {init}")

        A = np.zeros((1, 2, 1), dtype=complex)
        A[0, :, 0] = v
        mps.append(A)
    return mps


def apply_one_site_gate(A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    A: (Dl, d, Dr), U: (d, d)
    """
    return np.einsum("ab,ibr->iar", U, A, optimize=True)


def apply_two_site_gate_mps(
    A: np.ndarray,
    B: np.ndarray,
    U2: np.ndarray,
    chi_max: int,
    svd_cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply a 2-site gate to neighboring MPS tensors and split with SVD.

    Returns:
      A_new, B_new, kept_singular_values, discarded_weight
    """
    Dl, d1, Dm = A.shape
    Dm2, d2, Dr = B.shape
    if Dm != Dm2:
        raise ValueError("MPS bond mismatch")

    # theta: (Dl, d1, d2, Dr)
    theta = np.einsum("idr,rje->idje", A, B, optimize=True)

    # apply gate on physical legs
    theta = theta.reshape(Dl, d1 * d2, Dr)
    theta = np.einsum("ab,ibr->iar", U2, theta, optimize=True)
    theta = theta.reshape(Dl, d1, d2, Dr)

    # split
    theta_mat = np.reshape(theta, (Dl * d1, d2 * Dr))
    U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

    keep = min(chi_max, len(S))
    if svd_cutoff > 0.0:
        keep = min(keep, max(1, int(np.sum(S > svd_cutoff))))
    kept = S[:keep].copy()
    discarded = S[keep:].copy()

    disc_weight = float(np.sum(discarded ** 2))
    kept_norm = math.sqrt(max(1e-16, float(np.sum(kept ** 2))))
    kept /= kept_norm

    U = U[:, :keep]
    Vh = Vh[:keep, :]

    A_new = U.reshape(Dl, d1, keep)
    B_new = (np.diag(kept) @ Vh).reshape(keep, d2, Dr)

    return A_new, B_new, kept, disc_weight


def bond_entropy_from_singulars(s: np.ndarray, eps: float = 1e-15) -> float:
    if len(s) == 0:
        return 0.0
    p = np.real(s * s)
    p = p[p > eps]
    if len(p) == 0:
        return 0.0
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


def active_rank_from_singulars(s: np.ndarray, rel_thresh: float) -> int:
    if len(s) == 0 or s[0] < 1e-15:
        return 0
    rel = s / s[0]
    return int(np.sum(rel > rel_thresh))


def one_site_rho_from_mps(A: np.ndarray) -> np.ndarray:
    """
    Approximate local rho from tensor alone:
      rho_{ab} = sum_{l,r} A[l,a,r] conj(A[l,b,r])
    This is not globally exact unless gauge is favorable, but it is a useful local proxy.
    """
    rho = np.einsum("lar,lbr->ab", A, np.conjugate(A), optimize=True)
    tr = np.trace(rho)
    if abs(tr) > 1e-15:
        rho = rho / tr
    rho = 0.5 * (rho + dagger(rho))
    return rho


# ============================================================
# Main model
# ============================================================

class ChainSubstrate:
    def __init__(
        self,
        n_sites: int,
        chi_max: int,
        dt: float,
        heisenberg_scale: float,
        field_scale: float,
        kick_scale: float,
        svd_cutoff: float,
        rank_rel_thresh: float,
        seed: int,
        init: str,
    ):
        self.rng = np.random.default_rng(seed)
        self.n = n_sites
        self.chi_max = chi_max
        self.dt = dt
        self.heisenberg_scale = heisenberg_scale
        self.field_scale = field_scale
        self.kick_scale = kick_scale
        self.svd_cutoff = svd_cutoff
        self.rank_rel_thresh = rank_rel_thresh

        self.mps = init_product_mps(n_sites, self.rng, init=init)

        # Per-bond fixed local generators (no refolding / no rewiring)
        self.bond_gates_even = []
        self.bond_gates_odd = []
        for b in range(n_sites - 1):
            U2 = build_two_site_gate(
                rng=self.rng,
                dt=dt,
                heisenberg_scale=heisenberg_scale,
                field_scale=field_scale,
            )
            if b % 2 == 0:
                self.bond_gates_even.append((b, U2))
            else:
                self.bond_gates_odd.append((b, U2))

        # Cached bond diagnostics
        self.bond_singulars: List[np.ndarray] = [np.array([1.0], dtype=float) for _ in range(n_sites - 1)]
        self.bond_entropy = np.zeros(n_sites - 1, dtype=float)
        self.bond_rank = np.zeros(n_sites - 1, dtype=int)

        self.total_truncation_loss = 0.0
        self.step_truncation_loss = 0.0

        # persistence
        self.bond_hot_persistence = np.zeros(n_sites - 1, dtype=float)

    def apply_random_kicks(self):
        if self.kick_scale <= 0.0:
            return
        # sparse small kicks
        n_kicks = max(1, self.n // 12)
        sites = self.rng.choice(self.n, size=n_kicks, replace=False)
        for i in sites:
            U1 = random_single_site_unitary(self.rng, self.kick_scale)
            self.mps[i] = apply_one_site_gate(self.mps[i], U1)

    def half_sweep(self, parity: int):
        gate_list = self.bond_gates_even if parity == 0 else self.bond_gates_odd
        for b, U2 in gate_list:
            A = self.mps[b]
            B = self.mps[b + 1]
            A_new, B_new, svals, disc = apply_two_site_gate_mps(
                A=A,
                B=B,
                U2=U2,
                chi_max=self.chi_max,
                svd_cutoff=self.svd_cutoff,
            )
            self.mps[b] = A_new
            self.mps[b + 1] = B_new
            self.bond_singulars[b] = svals
            self.bond_entropy[b] = bond_entropy_from_singulars(svals)
            self.bond_rank[b] = active_rank_from_singulars(svals, self.rank_rel_thresh)

            self.step_truncation_loss += disc
            self.total_truncation_loss += disc

    def update_persistence(self):
        if len(self.bond_entropy) == 0:
            return
        thresh = float(np.quantile(self.bond_entropy, 0.75))
        hot = (self.bond_entropy >= thresh).astype(float)
        self.bond_hot_persistence = 0.97 * self.bond_hot_persistence + 0.03 * hot

    def step(self):
        self.step_truncation_loss = 0.0
        self.apply_random_kicks()
        self.half_sweep(0)
        self.half_sweep(1)
        self.update_persistence()

    def local_z_activity(self) -> np.ndarray:
        _, _, _, Z = paulis()
        vals = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            rho = one_site_rho_from_mps(self.mps[i])
            vals[i] = float(np.real(np.trace(rho @ Z)))
        # convert to activity-like quantity
        return 1.0 - np.abs(vals)

    def largest_active_cluster(self, entropy_quantile: float = 0.75) -> Tuple[int, int]:
        if len(self.bond_entropy) == 0:
            return 0, 0
        thresh = float(np.quantile(self.bond_entropy, entropy_quantile))
        active = self.bond_entropy >= thresh

        largest = 0
        n_clusters = 0
        run = 0
        for x in active:
            if x:
                run += 1
            else:
                if run > 0:
                    n_clusters += 1
                    largest = max(largest, run + 1)  # bonds -> sites
                run = 0
        if run > 0:
            n_clusters += 1
            largest = max(largest, run + 1)

        return largest, n_clusters

    def mean_active_path(self, entropy_quantile: float = 0.75) -> float:
        if len(self.bond_entropy) == 0:
            return 0.0
        thresh = float(np.quantile(self.bond_entropy, entropy_quantile))
        active = self.bond_entropy >= thresh
        lengths = []
        run = 0
        for x in active:
            if x:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                run = 0
        if run > 0:
            lengths.append(run)
        if not lengths:
            return 0.0
        return float(np.mean(lengths))

    def snapshot_metrics(self) -> Dict[str, float]:
        local_act = self.local_z_activity()
        largest_cluster, n_clusters = self.largest_active_cluster()
        mean_path = self.mean_active_path()

        return {
            "mean_local_activity": float(np.mean(local_act)),
            "max_local_activity": float(np.max(local_act)),
            "mean_bond_entropy": float(np.mean(self.bond_entropy)) if len(self.bond_entropy) else 0.0,
            "max_bond_entropy": float(np.max(self.bond_entropy)) if len(self.bond_entropy) else 0.0,
            "mean_bond_rank": float(np.mean(self.bond_rank)) if len(self.bond_rank) else 0.0,
            "max_bond_rank": int(np.max(self.bond_rank)) if len(self.bond_rank) else 0,
            "largest_cluster": int(largest_cluster),
            "n_clusters": int(n_clusters),
            "mean_active_path": float(mean_path),
            "step_truncation_loss": float(self.step_truncation_loss),
            "total_truncation_loss": float(self.total_truncation_loss),
            "mean_bond_persistence": float(np.mean(self.bond_hot_persistence)) if len(self.bond_hot_persistence) else 0.0,
            "max_bond_persistence": float(np.max(self.bond_hot_persistence)) if len(self.bond_hot_persistence) else 0.0,
        }


# ============================================================
# Plotting
# ============================================================

def make_plots(outdir: str, rows: List[Dict[str, float]], final_entropy: np.ndarray, final_rank: np.ndarray):
    steps = [r["step"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["mean_bond_entropy"] for r in rows], label="mean")
    plt.plot(steps, [r["max_bond_entropy"] for r in rows], label="max")
    plt.xlabel("step")
    plt.ylabel("bond entropy")
    plt.title("Bond entropy over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bond_entropy.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["mean_bond_rank"] for r in rows], label="mean")
    plt.plot(steps, [r["max_bond_rank"] for r in rows], label="max")
    plt.xlabel("step")
    plt.ylabel("bond rank proxy")
    plt.title("Bond rank over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bond_rank.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["largest_cluster"] for r in rows], label="largest cluster")
    plt.plot(steps, [r["n_clusters"] for r in rows], label="n clusters")
    plt.xlabel("step")
    plt.ylabel("cluster size / count")
    plt.title("Active clusters over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "largest_cluster.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["step_truncation_loss"] for r in rows], label="step")
    plt.plot(steps, [r["total_truncation_loss"] for r in rows], label="total")
    plt.xlabel("step")
    plt.ylabel("truncation loss")
    plt.title("Finite-bandwidth / truncation pressure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truncation_loss.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(final_entropy, bins=20)
    plt.xlabel("final bond entropy")
    plt.ylabel("count")
    plt.title("Final bond entropy histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "final_bond_entropy_hist.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    max_rank = int(max(1, np.max(final_rank))) if len(final_rank) else 1
    bins = np.arange(-0.5, max_rank + 1.5, 1)
    plt.hist(final_rank, bins=bins)
    plt.xlabel("final bond rank proxy")
    plt.ylabel("count")
    plt.title("Final bond rank histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "final_bond_rank_hist.png"), dpi=160)
    plt.close()


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="HSF-style chain substrate")
    p.add_argument("--outdir", type=str, default="chain_out")
    p.add_argument("--sites", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--snapshot_every", type=int, default=10)
    p.add_argument("--chi_max", type=int, default=32)
    p.add_argument("--dt", type=float, default=0.08)
    p.add_argument("--heisenberg_scale", type=float, default=0.9)
    p.add_argument("--field_scale", type=float, default=0.25)
    p.add_argument("--kick_scale", type=float, default=0.05)
    p.add_argument("--svd_cutoff", type=float, default=1e-8)
    p.add_argument("--rank_rel_thresh", type=float, default=0.12)
    p.add_argument("--init", type=str, default="random", choices=["random", "up", "neel"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    model = ChainSubstrate(
        n_sites=args.sites,
        chi_max=args.chi_max,
        dt=args.dt,
        heisenberg_scale=args.heisenberg_scale,
        field_scale=args.field_scale,
        kick_scale=args.kick_scale,
        svd_cutoff=args.svd_cutoff,
        rank_rel_thresh=args.rank_rel_thresh,
        seed=args.seed,
        init=args.init,
    )

    print("=" * 72)
    print("CHAIN SUBSTRATE (v1)")
    print("=" * 72)
    print(f"outdir: {args.outdir}")
    print(
        f"sites={args.sites}, steps={args.steps}, chi_max={args.chi_max}, "
        f"dt={args.dt}, init={args.init}, seed={args.seed}"
    )
    print(
        f"heisenberg_scale={args.heisenberg_scale}, field_scale={args.field_scale}, "
        f"kick_scale={args.kick_scale}, rank_rel_thresh={args.rank_rel_thresh}"
    )
    print()

    rows = []
    for step in range(args.steps + 1):
        if step > 0:
            model.step()

        if step % args.snapshot_every == 0 or step == args.steps:
            m = model.snapshot_metrics()
            row = {"step": step, **m}
            rows.append(row)
            print(
                f"step={step:>5}  "
                f"mean_S={m['mean_bond_entropy']:.3f}  "
                f"mean_rank={m['mean_bond_rank']:.3f}  "
                f"largest_cluster={m['largest_cluster']:>3}  "
                f"trunc_step={m['step_truncation_loss']:.3e}"
            )

    final_metrics = model.snapshot_metrics()
    final_entropy = model.bond_entropy.copy()
    final_rank = model.bond_rank.copy()

    summary = {
        "params": vars(args),
        "final_metrics": final_metrics,
        "final_bond_entropy": final_entropy.tolist(),
        "final_bond_rank": final_rank.tolist(),
    }

    save_json(summary, os.path.join(args.outdir, "summary.json"))
    save_csv(rows, os.path.join(args.outdir, "timeseries.csv"))
    make_plots(args.outdir, rows, final_entropy, final_rank)

    print()
    print("Saved:")
    print(f"  {os.path.join(args.outdir, 'summary.json')}")
    print(f"  {os.path.join(args.outdir, 'timeseries.csv')}")
    print(f"  {os.path.join(args.outdir, 'bond_entropy.png')}")
    print(f"  {os.path.join(args.outdir, 'bond_rank.png')}")
    print(f"  {os.path.join(args.outdir, 'largest_cluster.png')}")
    print(f"  {os.path.join(args.outdir, 'truncation_loss.png')}")
    print(f"  {os.path.join(args.outdir, 'final_bond_entropy_hist.png')}")
    print(f"  {os.path.join(args.outdir, 'final_bond_rank_hist.png')}")


if __name__ == "__main__":
    main()