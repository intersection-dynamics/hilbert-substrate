"""
defrag_lumps_demo_v2.py

Toy demonstration of emergent "lumps" (localized, low-entropy structures)
in a Hilbert-space substrate with local dynamics.

This version fixes a key issue in v1: the environment never actually became
entangled. Here we explicitly:

    1. Define "lump" sites and "environment" sites in factor index space.
    2. Initialize a product state with:
           - lump sites in |1>
           - environment sites in |0>
    3. Apply several layers of RANDOM LOCAL UNITARIES on the environment only:
           - random single-qubit rotations + a fixed two-qubit entangler
      so the environment becomes a genuine noisy bath:
           - high single-site entropy S_i ~ 1
           - low coherence C_i ~ 0
    4. Then turn on chain XX+YY dynamics + strong Z pinning on lump sites.
    5. Track:
           - single-factor entropy S_i(t)
           - coherence C_i(t) = 1 - S_i(t)
           - high-coherence segments as "lumps"
           - lump centers and counts over time

This is still a TOY model, not a full implementation of the defrag mechanism.
It is designed to exercise:

    - the core Substrate engine,
    - entropy / coherence diagnostics,
    - lump detection and tracking in a noisy environment.

No CLI; edit the CONFIG block below and run:

    python engine/defrag_lumps_demo_v2.py
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np

from hilbert_substrate_core import Config, Substrate


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Number of qubit factors in the substrate
N_FACTORS: int = 12

# Main dynamics parameters
STEPS: int = 60           # number of discrete time steps in the main run
DT_PAIR: float = 0.15     # time step for pairwise XX+YY unitaries
DT_PIN: float = 0.15      # time step for local Z "pinning" unitaries
J_PAIR: float = 1.0       # coupling strength for XX+YY terms
H_PIN: float = 1.0        # strength of Z pinning on lump sites

# Environment warm-up (pre-entangling via random local unitaries)
ENV_WARMUP_LAYERS: int = 20     # number of random circuit layers
ENV_RNG_SEED: int = 12345       # separate seed for env randomization (optional)

# RNG / backend for the whole run
SEED: int = 42
USE_GPU: bool = False

# Lump / environment definition in factor index space
# Two small lumps; everything else is environment.
LUMP_SEGMENTS: List[Tuple[int, int]] = [(2, 3), (8, 9)]

# Coherence threshold for defining a site as "inside a lump"
COHERENCE_THRESHOLD: float = 0.6

# Max number of lumps we will track per time step (for output array shape)
MAX_LUMPS: int = 4

# Output file
OUT_FILE: str = "defrag_lumps_results_v2.npz"


# ---------------------------------------------------------------------------
# Generic local unitary helpers (operate directly on Substrate.state)
# ---------------------------------------------------------------------------

def apply_single_qubit_unitary(sub: Substrate, site: int, U: np.ndarray) -> None:
    """
    Apply a single-qubit unitary U to factor `site` of the substrate's state |Ψ>.

    This modifies sub.state in place.

    Parameters
    ----------
    sub : Substrate
        The Hilbert substrate instance.
    site : int
        Index of the factor to which U is applied.
    U : np.ndarray
        2x2 unitary matrix, assumed to act on local_dim == 2.
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("apply_single_qubit_unitary assumes local_dim == 2.")

    xp = sub.xp
    n = sub.cfg.n_factors
    d = sub.cfg.local_dim

    if not (0 <= site < n):
        raise ValueError(f"Site index {site} out of range [0, {n-1}].")

    U_backend = xp.asarray(np.asarray(U, dtype=np.complex128))
    psi = sub.state.reshape((d,) * n)

    # Move target axis last
    non_sites = [i for i in range(n) if i != site]
    perm = non_sites + [site]
    psi_perm = xp.transpose(psi, axes=perm)

    rest_dim = d ** (n - 1)
    psi_flat = psi_perm.reshape(rest_dim, d)

    psi_flat = psi_flat @ U_backend.T

    psi_perm = psi_flat.reshape((d,) * n)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    psi_new = xp.transpose(psi_perm, axes=inv_perm)

    sub.state = psi_new.reshape(-1)


def apply_two_qubit_unitary(sub: Substrate, site1: int, site2: int, U4: np.ndarray) -> None:
    """
    Apply a two-qubit unitary U4 (4x4) to factors `site1` and `site2` of |Ψ>.

    Modifies sub.state in place.

    Parameters
    ----------
    sub : Substrate
        The Hilbert substrate instance.
    site1, site2 : int
        Indices of the two factors.
    U4 : np.ndarray
        4x4 unitary matrix acting on the tensor product of the two qubits.
        We assume local_dim == 2 for both sites.
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("apply_two_qubit_unitary assumes local_dim == 2.")

    xp = sub.xp
    n = sub.cfg.n_factors
    d = sub.cfg.local_dim

    if not (0 <= site1 < n) or not (0 <= site2 < n):
        raise ValueError("Site indices out of range.")
    if site1 == site2:
        raise ValueError("site1 and site2 must be distinct.")

    i, j = sorted([site1, site2])

    U_backend = xp.asarray(np.asarray(U4, dtype=np.complex128))
    psi = sub.state.reshape((d,) * n)

    # Move the two target axes to the last two positions
    remaining = [k for k in range(n) if k not in (i, j)]
    perm = remaining + [i, j]
    psi_perm = xp.transpose(psi, axes=perm)

    # Reshape to (rest_dim, 4)
    rest_dim = d ** (n - 2)
    psi_flat = psi_perm.reshape(rest_dim, d * d)  # shape: (rest_dim, 4)

    psi_flat = psi_flat @ U_backend.T  # apply on the 4-dim subspace

    psi_perm = psi_flat.reshape((d,) * n)
    inv_perm = [0] * n
    for idx, p in enumerate(perm):
        inv_perm[p] = idx
    psi_new = xp.transpose(psi_perm, axes=inv_perm)

    sub.state = psi_new.reshape(-1)


# ---------------------------------------------------------------------------
# Random environment warm-up: random local unitary circuit
# ---------------------------------------------------------------------------

def random_single_qubit_unitary(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a simple random single-qubit unitary using random Euler angles.

    Not Haar-perfect, but more than enough to stir the environment.
    """
    # Random angles
    alpha = 2 * np.pi * rng.random()
    beta = 2 * np.pi * rng.random()
    gamma = 2 * np.pi * rng.random()

    # Z(alpha) Y(beta) Z(gamma) decomposition
    ca, sa = np.cos(alpha / 2), np.sin(alpha / 2)
    cb, sb = np.cos(beta / 2), np.sin(beta / 2)
    cg, sg = np.cos(gamma / 2), np.sin(gamma / 2)

    # Z(alpha)
    Za = np.array([[np.exp(-1j * alpha / 2), 0],
                   [0, np.exp(1j * alpha / 2)]], dtype=np.complex128)
    # Y(beta)
    Yb = np.array([[cb, -sb],
                   [sb, cb]], dtype=np.complex128)
    # Z(gamma)
    Zg = np.array([[np.exp(-1j * gamma / 2), 0],
                   [0, np.exp(1j * gamma / 2)]], dtype=np.complex128)

    U = Za @ Yb @ Zg
    return U


def fixed_two_qubit_entangler() -> np.ndarray:
    """
    Return a fixed 4x4 entangling gate.

    We'll use a CZ-like gate in the computational basis:

        CZ = diag(1, 1, 1, -1)

    which entangles superpositions of |11> with other basis states.
    """
    CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
    return CZ


def warmup_environment_random(
    sub: Substrate,
    env_sites: List[int],
    n_layers: int,
    rng: np.random.Generator,
) -> None:
    """
    Apply a random local unitary circuit on the environment sites only.

    Each layer:
        - random single-qubit unitaries on all env sites
        - random subset of nearest-neighbor env pairs get a fixed entangler

    This is not using the Substrate.local_terms / step mechanism; instead
    we operate directly on sub.state using our apply_* helpers.
    """
    if len(env_sites) == 0:
        return

    env_sites_sorted = sorted(env_sites)
    entangler = fixed_two_qubit_entangler()

    for layer in range(n_layers):
        # 1. Single-qubit random unitaries
        for s in env_sites_sorted:
            U1 = random_single_qubit_unitary(rng)
            apply_single_qubit_unitary(sub, s, U1)

        # 2. Random entanglers on random neighbor pairs in env_sites
        #    (neighbor in index space, not geometry)
        for s in env_sites_sorted:
            # with some probability, entangle this site with the next one
            if rng.random() < 0.5:
                s_next = s + 1
                if s_next in env_sites_sorted:
                    apply_two_qubit_unitary(sub, s, s_next, entangler)


# ---------------------------------------------------------------------------
# Main chain XX+YY + pinning dynamics using Substrate.local_terms
# ---------------------------------------------------------------------------

def build_chain_xx_yy_and_pinning(
    sub: Substrate,
    dt_pair: float,
    dt_pin: float,
    J_pair: float,
    H_pin: float,
    pin_sites: List[int],
) -> None:
    """
    Register local-unitary dynamics on the substrate for the main run:

    - Nearest-neighbor XX+YY pairwise interactions along the factor index line.
    - Local Z "pinning" fields on selected lump sites.
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("build_chain_xx_yy_and_pinning is implemented for qubits.")

    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    H_pair = J_pair * (np.kron(X, X) + np.kron(Y, Y))
    U_pair = Substrate.hermitian_to_unitary(H_pair, dt_pair, use_gpu=sub.on_gpu)

    H_pin_single = H_pin * Z
    U_pin_single = Substrate.hermitian_to_unitary(H_pin_single, dt_pin, use_gpu=sub.on_gpu)

    for i in range(sub.cfg.n_factors - 1):
        sub.add_local_unitary(sites=[i, i + 1], unitary=U_pair)

    for i in pin_sites:
        if 0 <= i < sub.cfg.n_factors:
            sub.add_local_unitary(sites=[i], unitary=U_pin_single)


# ---------------------------------------------------------------------------
# Entropy and coherence diagnostics
# ---------------------------------------------------------------------------

def single_factor_entropy(sub: Substrate, index: int) -> float:
    """Von Neumann entropy S(ρ_i) (bits) of single-factor reduced density."""
    rho = sub.reduced_density([index])
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals.real, 1e-15, 1.0)
    S = -np.sum(evals * np.log2(evals))
    return float(S)


def entropy_and_coherence_per_factor(sub: Substrate) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute von Neumann entropy S_i and coherence C_i for each factor i.

    For qubits:
        S_i in [0, 1] bits (0 = pure, 1 = maximally mixed).
        C_i = 1 - S_i.
    """
    if sub.cfg.local_dim != 2:
        raise ValueError("entropy_and_coherence_per_factor assumes local_dim == 2.")

    n = sub.cfg.n_factors
    S = np.zeros(n, dtype=float)
    for i in range(n):
        S[i] = single_factor_entropy(sub, i)
    C = 1.0 - S
    return S, C


# ---------------------------------------------------------------------------
# Lump detection from coherence profile
# ---------------------------------------------------------------------------

def detect_lumps_from_coherence(
    coherence: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int]]:
    """
    Detect "lump" regions from a coherence profile C_i.

    A site i is part of a lump if C_i >= threshold.
    Contiguous such sites form lump segments.
    """
    n = coherence.shape[0]
    lumps: List[Tuple[int, int]] = []

    in_lump = False
    start = 0
    for i in range(n):
        if coherence[i] >= threshold:
            if not in_lump:
                in_lump = True
                start = i
        else:
            if in_lump:
                in_lump = False
                lumps.append((start, i - 1))

    if in_lump:
        lumps.append((start, n - 1))

    return lumps


def lump_centers_from_segments(
    segments: List[Tuple[int, int]],
) -> List[float]:
    """Compute lump centers as arithmetic means of segment endpoints."""
    centers: List[float] = []
    for (start, end) in segments:
        centers.append(0.5 * (start + end))
    return centers


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_defrag_lumps_demo_v2(
    n_factors: int,
    steps: int,
    dt_pair: float,
    dt_pin: float,
    J_pair: float,
    H_pin: float,
    env_warmup_layers: int,
    env_rng_seed: int,
    seed: int,
    use_gpu: bool,
    lump_segments: List[Tuple[int, int]],
    coherence_threshold: float,
    max_lumps: int,
) -> Dict[str, Any]:
    """
    Run the v2 "defrag lumps" demonstration with random environment warm-up.
    """
    # Build lump and environment site lists
    lump_sites = set()
    for (a, b) in lump_segments:
        for i in range(a, b + 1):
            if 0 <= i < n_factors:
                lump_sites.add(i)
    env_sites = [i for i in range(n_factors) if i not in lump_sites]
    lump_sites_list = sorted(list(lump_sites))

    # Initial product state:
    #   - Lump sites in |1>
    #   - Environment sites in |0>
    product_state = [0] * n_factors
    for i in lump_sites:
        product_state[i] = 1

    cfg = Config(
        n_factors=n_factors,
        local_dim=2,
        use_gpu=use_gpu,
        seed=seed,
        product_state=product_state,
    )
    sub = Substrate(cfg)

    print("[Defrag v2] Initial norm:", sub.norm())
    print("[Defrag v2] Lump sites:", lump_sites_list)
    print("[Defrag v2] Env sites:", env_sites)
    print("[Defrag v2] Initial product_state:", product_state)

    # -----------------------------------------------------------------------
    # 1) Pre-entangle the environment with random local unitaries
    # -----------------------------------------------------------------------
    if env_warmup_layers > 0 and len(env_sites) >= 1:
        print(f"[Defrag v2] Environment warm-up: layers={env_warmup_layers}")
        env_rng = np.random.default_rng(env_rng_seed)
        warmup_environment_random(sub, env_sites=env_sites, n_layers=env_warmup_layers, rng=env_rng)
    else:
        print("[Defrag v2] Skipping environment warm-up (no env sites or layers=0).")

    # -----------------------------------------------------------------------
    # 2) Main dynamics: chain XX+YY + pinning on lump sites
    # -----------------------------------------------------------------------
    build_chain_xx_yy_and_pinning(
        sub,
        dt_pair=dt_pair,
        dt_pin=dt_pin,
        J_pair=J_pair,
        H_pin=H_pin,
        pin_sites=lump_sites_list,
    )

    # -----------------------------------------------------------------------
    # Storage for diagnostics
    # -----------------------------------------------------------------------
    t_vals = np.arange(steps + 1, dtype=float)
    S_ts = np.zeros((n_factors, steps + 1), dtype=float)
    C_ts = np.zeros((n_factors, steps + 1), dtype=float)

    lump_mask = np.zeros((steps + 1, n_factors), dtype=bool)
    lump_centers_arr = np.full((steps + 1, max_lumps), np.nan, dtype=float)
    lump_counts = np.zeros(steps + 1, dtype=int)

    # t = 0 (after env warm-up, before main evolution)
    S0, C0 = entropy_and_coherence_per_factor(sub)
    S_ts[:, 0] = S0
    C_ts[:, 0] = C0
    segments0 = detect_lumps_from_coherence(C0, coherence_threshold)
    centers0 = lump_centers_from_segments(segments0)
    lump_counts[0] = len(centers0)
    for m, (start, end) in enumerate(segments0[:max_lumps]):
        lump_mask[0, start:end + 1] = True
    for m, center in enumerate(centers0[:max_lumps]):
        lump_centers_arr[0, m] = center

    print(f"[Defrag v2] t=0: entropies={S0}")
    print(f"[Defrag v2] t=0: coherences={C0}")
    print(f"[Defrag v2] t=0: segments={segments0}, centers={centers0}")

    # Main time evolution
    for step in range(1, steps + 1):
        sub.step(n_steps=1)
        S, C = entropy_and_coherence_per_factor(sub)
        S_ts[:, step] = S
        C_ts[:, step] = C

        segments = detect_lumps_from_coherence(C, coherence_threshold)
        centers = lump_centers_from_segments(segments)
        lump_counts[step] = len(centers)

        for m, (start, end) in enumerate(segments[:max_lumps]):
            lump_mask[step, start:end + 1] = True
        for m, center in enumerate(centers[:max_lumps]):
            lump_centers_arr[step, m] = center

        if step % max(1, steps // 10) == 0:
            print(f"[Defrag v2] step {step}/{steps}: "
                  f"mean S={S.mean():.3f}, mean C={C.mean():.3f}, "
                  f"segments={segments}")

    results = {
        "t": t_vals,
        "S": S_ts,
        "C": C_ts,
        "lump_mask": lump_mask,
        "lump_centers": lump_centers_arr,
        "lump_counts": lump_counts,
        "cfg": {
            "n_factors": n_factors,
            "steps": steps,
            "dt_pair": dt_pair,
            "dt_pin": dt_pin,
            "J_pair": J_pair,
            "H_pin": H_pin,
            "env_warmup_layers": env_warmup_layers,
            "env_rng_seed": env_rng_seed,
            "seed": seed,
            "use_gpu": use_gpu,
            "lump_segments": lump_segments,
            "coherence_threshold": coherence_threshold,
            "max_lumps": max_lumps,
        },
    }
    return results


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

def quick_plots(results: Dict[str, Any]) -> None:
    """Generate simple plots of entropy, coherence, and lump centers over time."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Defrag v2] matplotlib not available; skipping plots.")
        return

    t = results["t"]
    S = results["S"]
    C = results["C"]
    lump_centers = results["lump_centers"]

    n, T1 = S.shape
    assert T1 == len(t)

    # Entropy heatmap
    plt.figure()
    plt.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, n - 1],
    )
    plt.colorbar(label="S_i(t) (bits)")
    plt.xlabel("time step")
    plt.ylabel("factor index i")
    plt.title("Single-factor entropy S_i(t)")
    plt.tight_layout()

    # Coherence heatmap with lump centers
    plt.figure()
    plt.imshow(
        C,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, n - 1],
    )
    plt.colorbar(label="C_i(t) = 1 - S_i(t)")
    plt.xlabel("time step")
    plt.ylabel("factor index i")
    plt.title("Coherence C_i(t) with lump centers")
    for k in range(lump_centers.shape[1]):
        centers_k = lump_centers[:, k]
        mask = ~np.isnan(centers_k)
        plt.plot(t[mask], centers_k[mask], ".", markersize=3)
    plt.tight_layout()

    # Lump count vs time
    lump_counts = results["lump_counts"]
    plt.figure()
    plt.plot(t, lump_counts, marker="o")
    plt.xlabel("time step")
    plt.ylabel("number of lumps")
    plt.title("Lump count vs time")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Script entrypoint (no CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Defrag Lumps Demo v2 (random env warm-up, no CLI) ===")
    print(f"n_factors={N_FACTORS}, steps={STEPS}")
    print(f"dt_pair={DT_PAIR}, dt_pin={DT_PIN}, J_pair={J_PAIR}, H_pin={H_PIN}")
    print(f"env_warmup_layers={ENV_WARMUP_LAYERS}, env_rng_seed={ENV_RNG_SEED}")
    print(f"seed={SEED}, use_gpu={USE_GPU}")
    print(f"lump_segments={LUMP_SEGMENTS}")
    print(f"coherence_threshold={COHERENCE_THRESHOLD}, max_lumps={MAX_LUMPS}")
    print(f"Output file: {OUT_FILE}")

    res = run_defrag_lumps_demo_v2(
        n_factors=N_FACTORS,
        steps=STEPS,
        dt_pair=DT_PAIR,
        dt_pin=DT_PIN,
        J_pair=J_PAIR,
        H_pin=H_PIN,
        env_warmup_layers=ENV_WARMUP_LAYERS,
        env_rng_seed=ENV_RNG_SEED,
        seed=SEED,
        use_gpu=USE_GPU,
        lump_segments=LUMP_SEGMENTS,
        coherence_threshold=COHERENCE_THRESHOLD,
        max_lumps=MAX_LUMPS,
    )

    np.savez_compressed(OUT_FILE, **res)
    print(f"[Defrag v2] Saved results to {OUT_FILE}")

    quick_plots(res)
    print("[Defrag v2] Done.")
