#!/usr/bin/env python3
"""
subsystem_branching.py
=======================
HSF — Subsystem Creation as a Branching Process

HYPOTHESIS:
  Subsystem creation is a branching process driven by interaction events.
  Early on, interaction rate grows with N, so N(t) grows approximately
  exponentially. Finite bandwidth and no-refolding then act like a
  'resource cap' that reduces the effective branching rate, turning
  pure exponential growth into a regulated growth regime.

MODEL:
  State: N subsystems, each with dimension d, connected by links.
  Each link occupies d² dimensions (composite link requirement).

  At each timestep:
    1. INTERACTION EVENTS: Each existing pair of connected subsystems
       can interact. Rate ~ number of active links.
    2. LINK CREATION: When two unlinked subsystems interact (proximity),
       a new link register is created. The link itself factors into two
       subsystem-like entities (d_B = N² → two d-dim factors).
    3. CONSTRAINTS:
       - Finite bandwidth: each subsystem has a maximum number of
         links it can sustain (channel capacity κ).
       - No-refolding: once a link exists and is populated (carries
         excitations), it cannot be dissolved. Links are permanent.
       - No-signaling: links between non-adjacent subsystems are
         forbidden (locality of interaction graph).

  We track N(t) under different constraint regimes:
    (a) Unconstrained: pure branching, no limits
    (b) +Finite bandwidth only: κ limits connections per node
    (c) +No-refolding only: links permanent, network rigidifies
    (d) Full HSF: all constraints active

  PREDICTION:
    (a) Pure exponential growth (N ~ e^{αt})
    (b) Bandwidth → logistic-like saturation (N → N_max ~ κ·N₀)
    (c) No-refolding → network freezes, growth stalls
    (d) Full HSF → regulated growth: fast early, controlled late,
        with a characteristic "settling" scale

DEPENDENCIES: numpy
RUN: python subsystem_branching.py
"""

import numpy as np
import json
import os
import time
import argparse
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════
#  Substrate graph model
# ═══════════════════════════════════════════════════════════════════════

class SubstrateGraph:
    """
    A growing graph of subsystems connected by links.
    
    Each node = subsystem (dimension d)
    Each edge = link register (dimension d², factored as d ⊗ d)
    
    The link itself contributes 2 new "interface subsystems" to the
    total count, representing the independent left/right factors.
    
    NO-SIGNALING IMPLEMENTATION:
      Each subsystem has an "activation time" — the timestep at which
      it was first reached by the causal frontier. A link can only
      trigger a spawn event if BOTH endpoints are active. New subsystems
      become active one timestep after creation (signal propagation delay).
      
      This enforces a light cone: the colonization wave advances at most
      one graph hop per timestep, giving N(t) ~ t^d instead of e^{αt}.
    """
    
    def __init__(self, n_initial=2, d=3, bandwidth=None, 
                 no_refolding=False, locality_radius=2,
                 spatial_dim=1, no_signaling=False):
        self.d = d
        self.bandwidth = bandwidth
        self.no_refolding = no_refolding
        self.locality_radius = locality_radius
        self.spatial_dim = spatial_dim
        self.no_signaling = no_signaling
        
        self.n_subsystems = n_initial
        self.n_links = 0
        self.n_interface_subsystems = 0
        self.links = set()
        self.degree = defaultdict(int)
        self.timestep = 0
        
        # Activation tracking for no-signaling
        # activation_time[i] = timestep when node i became active
        self.activation_time = {}
        for i in range(n_initial):
            self.activation_time[i] = 0  # initial nodes are active at t=0
        
        # Spatial positions
        self.positions = {}
        for i in range(n_initial):
            self.positions[i] = np.random.randn(spatial_dim) * 0.5
        
        # Connect initial subsystems as a chain
        for i in range(n_initial - 1):
            self._add_link(i, i + 1)
        
        self.history = {
            'total_subsystems': [self.total_subsystems()],
            'site_subsystems': [self.n_subsystems],
            'interface_subsystems': [self.n_interface_subsystems],
            'n_links': [self.n_links],
            'mean_degree': [self._mean_degree()],
            'max_degree': [self._max_degree()],
            'total_hilbert_log': [self._log_hilbert_dim()],
            'n_active': [n_initial],
            'frontier_radius': [0.0],
        }
    
    def total_subsystems(self):
        """Total = site subsystems + interface subsystems from links."""
        return self.n_subsystems + self.n_interface_subsystems
    
    def _mean_degree(self):
        if self.n_subsystems == 0:
            return 0
        return 2 * self.n_links / max(1, self.n_subsystems)
    
    def _max_degree(self):
        if not self.degree:
            return 0
        return max(self.degree.values())
    
    def _log_hilbert_dim(self):
        """log2 of total Hilbert space dimension."""
        n_site = self.n_subsystems
        n_link = self.n_links
        # Each site: d dimensions. Each link: d² dimensions.
        return n_site * np.log2(self.d) + n_link * 2 * np.log2(self.d)
    
    def _can_link(self, i, j):
        """Check if a new link between i and j is allowed."""
        if i == j:
            return False
        edge = (min(i, j), max(i, j))
        if edge in self.links:
            return False
        
        # Bandwidth constraint
        if self.bandwidth is not None:
            if self.degree[i] >= self.bandwidth or self.degree[j] >= self.bandwidth:
                return False
        
        # Locality constraint: check spatial distance
        if i in self.positions and j in self.positions:
            dist = np.linalg.norm(self.positions[i] - self.positions[j])
            if dist > self.locality_radius:
                return False
        
        return True
    
    def _add_link(self, i, j):
        """Create a link between subsystems i and j."""
        edge = (min(i, j), max(i, j))
        self.links.add(edge)
        self.n_links += 1
        self.degree[i] += 1
        self.degree[j] += 1
        
        # The link register factors as d ⊗ d, creating 2 interface subsystems
        self.n_interface_subsystems += 2
    
    def _add_subsystem(self, near_node=None):
        """Create a new site subsystem, positioned near an existing one."""
        new_id = self.n_subsystems
        self.n_subsystems += 1
        
        if near_node is not None and near_node in self.positions:
            # Place near the parent with some jitter
            self.positions[new_id] = (self.positions[near_node] + 
                                       np.random.randn(self.spatial_dim) * 0.3)
        else:
            self.positions[new_id] = np.random.randn(self.spatial_dim)
        
        return new_id
    
    def _is_active(self, node_id):
        """Check if a node is causally active at current timestep."""
        if not self.no_signaling:
            return True  # everything active when NS not enforced
        return (node_id in self.activation_time and 
                self.activation_time[node_id] < self.timestep)
    
    def _frontier_radius(self):
        """Max distance from origin of any active node."""
        origin = np.zeros(self.spatial_dim)
        max_r = 0.0
        for i, t_act in self.activation_time.items():
            if t_act <= self.timestep and i in self.positions:
                r = np.linalg.norm(self.positions[i] - origin)
                max_r = max(max_r, r)
        return max_r
    
    def _n_active(self):
        """Count of currently active nodes."""
        return sum(1 for t in self.activation_time.values() 
                   if t < self.timestep)
    
    def _record(self):
        """Record current state to history."""
        self.history['total_subsystems'].append(self.total_subsystems())
        self.history['site_subsystems'].append(self.n_subsystems)
        self.history['interface_subsystems'].append(self.n_interface_subsystems)
        self.history['n_links'].append(self.n_links)
        self.history['mean_degree'].append(self._mean_degree())
        self.history['max_degree'].append(self._max_degree())
        self.history['total_hilbert_log'].append(self._log_hilbert_dim())
        self.history['n_active'].append(self._n_active())
        self.history['frontier_radius'].append(self._frontier_radius())
    
    def step(self, interaction_rate=0.3, spawn_rate=0.1, max_subsystems=5000):
        """
        One timestep of the branching process.
        
        With no_signaling=True, only links where BOTH endpoints were
        activated in a PREVIOUS timestep can trigger spawns. This
        enforces a one-hop-per-step speed limit (light cone).
        """
        self.timestep += 1
        
        if self.n_subsystems >= max_subsystems:
            self._record()
            return
        
        new_subsystems = []
        
        # Phase 1: Active links spawn new subsystems
        for (i, j) in list(self.links):
            if self.n_subsystems >= max_subsystems:
                break
            # Causal check: both endpoints must already be active
            if not (self._is_active(i) and self._is_active(j)):
                continue
            if np.random.random() < interaction_rate:
                parent = i if np.random.random() < 0.5 else j
                new_id = self._add_subsystem(near_node=parent)
                new_subsystems.append((new_id, parent))
                # Activates at current timestep → becomes active next step
                self.activation_time[new_id] = self.timestep
        
        # Phase 2: New subsystems link to their parents
        for new_id, parent in new_subsystems:
            if self._can_link(new_id, parent):
                self._add_link(new_id, parent)
        
        # Phase 3: Nearby active pairs may form new links
        n_samples = min(self.n_subsystems * 2, 500)
        for _ in range(n_samples):
            i = np.random.randint(self.n_subsystems)
            j = np.random.randint(self.n_subsystems)
            if i != j and self._can_link(i, j):
                # Both must be active for no-signaling
                if self._is_active(i) and self._is_active(j):
                    if np.random.random() < spawn_rate:
                        self._add_link(i, j)
        
        self._record()


# ═══════════════════════════════════════════════════════════════════════
#  Experiment: compare constraint regimes
# ═══════════════════════════════════════════════════════════════════════

def run_regime(label, n_steps=200, n_initial=2, d=3,
               bandwidth=None, no_refolding=False, 
               locality_radius=2.0, interaction_rate=0.3,
               spawn_rate=0.1, n_trials=20, spatial_dim=1,
               max_subsystems=3000, no_signaling=False):
    """Run multiple trials of a constraint regime and return statistics."""
    
    all_total = []
    all_sites = []
    all_links = []
    all_degree = []
    all_hilbert = []
    all_active = []
    all_frontier = []
    
    for trial in range(n_trials):
        np.random.seed(1000 * hash(label) % 100000 + trial)
        
        g = SubstrateGraph(
            n_initial=n_initial, d=d, bandwidth=bandwidth,
            no_refolding=no_refolding, locality_radius=locality_radius,
            spatial_dim=spatial_dim, no_signaling=no_signaling
        )
        
        for t in range(n_steps):
            g.step(interaction_rate=interaction_rate, 
                   spawn_rate=spawn_rate,
                   max_subsystems=max_subsystems)
        
        all_total.append(g.history['total_subsystems'])
        all_sites.append(g.history['site_subsystems'])
        all_links.append(g.history['n_links'])
        all_degree.append(g.history['mean_degree'])
        all_hilbert.append(g.history['total_hilbert_log'])
        all_active.append(g.history['n_active'])
        all_frontier.append(g.history['frontier_radius'])
    
    # Compute statistics
    all_total = np.array(all_total)
    all_sites = np.array(all_sites)
    all_links = np.array(all_links)
    all_degree = np.array(all_degree)
    all_hilbert = np.array(all_hilbert)
    all_active = np.array(all_active)
    all_frontier = np.array(all_frontier)
    
    result = {
        'label': label,
        'n_steps': n_steps,
        'n_trials': n_trials,
        'params': {
            'bandwidth': bandwidth,
            'no_refolding': no_refolding,
            'no_signaling': no_signaling,
            'locality_radius': locality_radius,
            'interaction_rate': interaction_rate,
            'spawn_rate': spawn_rate,
        },
        'total_subsystems': {
            'mean': all_total.mean(axis=0).tolist(),
            'std': all_total.std(axis=0).tolist(),
            'final_mean': float(all_total[:, -1].mean()),
            'final_std': float(all_total[:, -1].std()),
        },
        'site_subsystems': {
            'mean': all_sites.mean(axis=0).tolist(),
            'final_mean': float(all_sites[:, -1].mean()),
        },
        'n_links': {
            'mean': all_links.mean(axis=0).tolist(),
            'final_mean': float(all_links[:, -1].mean()),
        },
        'mean_degree': {
            'mean': all_degree.mean(axis=0).tolist(),
            'final_mean': float(all_degree[:, -1].mean()),
        },
        'hilbert_log': {
            'mean': all_hilbert.mean(axis=0).tolist(),
            'final_mean': float(all_hilbert[:, -1].mean()),
        },
        'n_active': {
            'mean': all_active.mean(axis=0).tolist(),
            'final_mean': float(all_active[:, -1].mean()),
        },
        'frontier_radius': {
            'mean': all_frontier.mean(axis=0).tolist(),
            'final_mean': float(all_frontier[:, -1].mean()),
        },
    }
    
    return result


def fit_growth(t, N_t):
    """
    Fit growth curve to distinguish exponential vs power-law vs linear.
    Power law N ~ t^α is the prediction for no-signaling constrained growth.
    """
    t = np.array(t, dtype=float)
    N_t = np.array(N_t, dtype=float)
    
    start = max(1, len(t) // 10)
    t_fit = t[start:]
    N_fit = N_t[start:]
    
    if N_fit[-1] <= N_fit[0] + 1:
        return {'type': 'stalled', 'rate': 0.0, 'power': 0.0}
    
    # Test exponential: log(N) vs t should be linear
    log_N = np.log(np.maximum(N_fit, 1))
    coeffs_exp = np.polyfit(t_fit, log_N, 1)
    exp_rate = coeffs_exp[0]
    pred_exp = np.polyval(coeffs_exp, t_fit)
    ss_res_exp = np.sum((log_N - pred_exp)**2)
    ss_tot_exp = np.sum((log_N - log_N.mean())**2)
    r2_exp = 1 - ss_res_exp / max(ss_tot_exp, 1e-10)
    
    # Test power law: log(N) vs log(t) should be linear → N ~ t^α
    # Shift t to avoid log(0): use t_fit which starts after transient
    log_t = np.log(np.maximum(t_fit, 1))
    coeffs_pow = np.polyfit(log_t, log_N, 1)
    power_exp = coeffs_pow[0]
    pred_pow = np.polyval(coeffs_pow, log_t)
    ss_res_pow = np.sum((log_N - pred_pow)**2)
    ss_tot_pow = np.sum((log_N - log_N.mean())**2)
    r2_pow = 1 - ss_res_pow / max(ss_tot_pow, 1e-10)
    
    # Test linear: N vs t
    coeffs_lin = np.polyfit(t_fit, N_fit, 1)
    lin_rate = coeffs_lin[0]
    pred_lin = np.polyval(coeffs_lin, t_fit)
    ss_res_lin = np.sum((N_fit - pred_lin)**2)
    ss_tot_lin = np.sum((N_fit - N_fit.mean())**2)
    r2_lin = 1 - ss_res_lin / max(ss_tot_lin, 1e-10)
    
    result = {
        'exp_rate': float(exp_rate),
        'exp_r2': float(r2_exp),
        'power_exp': float(power_exp),
        'power_r2': float(r2_pow),
        'lin_rate': float(lin_rate),
        'lin_r2': float(r2_lin),
    }
    
    # Classify by best R² on log(N) fits
    # Power law and exponential both fit in log space, so compare directly
    fits = [
        ('exponential', r2_exp),
        ('power_law', r2_pow),
        ('linear', r2_lin),
    ]
    fits.sort(key=lambda x: -x[1])
    
    # Exponential beats power law if it has higher R² AND rate > 0
    # Power law beats exponential if α is close to an integer (t^d prediction)
    if fits[0][0] == 'exponential' and exp_rate > 0.001:
        result['type'] = 'exponential'
    elif fits[0][0] == 'power_law' or (r2_pow > 0.98 and r2_pow > r2_exp - 0.02):
        result['type'] = 'power_law'
    elif fits[0][0] == 'linear':
        result['type'] = 'linear'
    else:
        result['type'] = fits[0][0]
    
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Main experiment
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='HSF Subsystem Branching Process')
    parser.add_argument('--nsteps', type=int, default=200)
    parser.add_argument('--ntrials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    t_start = time.time()
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  HSF: Subsystem Branching Process                                  ║")
    print("║  Does interaction-driven growth regulate under HSF constraints?     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    n_steps = args.nsteps
    n_trials = args.ntrials
    common = dict(n_steps=n_steps, n_initial=2, d=3, n_trials=n_trials,
                  interaction_rate=0.05, spawn_rate=0.05, spatial_dim=1,
                  max_subsystems=2000)
    
    regimes = [
        # ── No constraints ──
        ('(a) Unconstrained',
         dict(bandwidth=None, no_refolding=False, no_signaling=False,
              locality_radius=100.0)),
        
        # ── Single constraints ──
        ('(b) +No-signaling only',
         dict(bandwidth=None, no_refolding=False, no_signaling=True,
              locality_radius=100.0)),
        
        ('(c) +Bandwidth only (κ=4)',
         dict(bandwidth=4, no_refolding=False, no_signaling=False,
              locality_radius=100.0)),
        
        ('(d) +Locality only (r=1.5)',
         dict(bandwidth=None, no_refolding=False, no_signaling=False,
              locality_radius=1.5)),
        
        # ── Pairs ──
        ('(e) NS + Bandwidth (κ=4)',
         dict(bandwidth=4, no_refolding=False, no_signaling=True,
              locality_radius=100.0)),
        
        ('(f) NS + Locality (r=1.5)',
         dict(bandwidth=None, no_refolding=False, no_signaling=True,
              locality_radius=1.5)),
        
        # ── Full HSF ──
        ('(g) Full HSF (κ=4,r=1.5)',
         dict(bandwidth=4, no_refolding=True, no_signaling=True,
              locality_radius=1.5)),
        
        ('(h) Full HSF tight (κ=3,r=1)',
         dict(bandwidth=3, no_refolding=True, no_signaling=True,
              locality_radius=1.0)),
        
        # ── Spatial dimension test (1D vs 2D vs 3D) ──
    ]
    
    # Add spatial dimension variants with full HSF
    for sdim in [1, 2, 3]:
        regimes.append((
            f'(s{sdim}) Full HSF {sdim}D',
            dict(bandwidth=4, no_refolding=True, no_signaling=True,
                 locality_radius=1.5, spatial_dim=sdim)
        ))
    
    all_results = {}
    
    for label, params in regimes:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        
        t0 = time.time()
        r = run_regime(label, **{**common, **params})
        dt = time.time() - t0
        
        # Fit growth curve
        times = list(range(n_steps + 1))
        growth = fit_growth(times, r['total_subsystems']['mean'])
        r['growth_fit'] = growth
        
        final = r['total_subsystems']['final_mean']
        final_std = r['total_subsystems']['final_std']
        final_links = r['n_links']['final_mean']
        final_degree = r['mean_degree']['final_mean']
        final_hilbert = r['hilbert_log']['final_mean']
        
        print(f"    Final subsystems: {final:.0f} ± {final_std:.0f}")
        print(f"    Final links: {final_links:.0f}")
        print(f"    Mean degree: {final_degree:.1f}")
        print(f"    log₂(dim H): {final_hilbert:.0f}")
        print(f"    Growth type: {growth['type']}")
        print(f"      exp:  rate={growth['exp_rate']:.4f}  R²={growth['exp_r2']:.4f}")
        print(f"      power: α={growth['power_exp']:.2f}    R²={growth['power_r2']:.4f}")
        print(f"      linear: rate={growth['lin_rate']:.2f}  R²={growth['lin_r2']:.4f}")
        print(f"    ({dt:.1f}s)")
        
        all_results[label] = r
    
    # ─── Comparative analysis ──────────────────────────────────────
    elapsed = time.time() - t_start
    
    print(f"\n\n{'═' * 70}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'═' * 70}\n")
    
    print(f"  {'Regime':<30} {'Final N':>8} {'Type':>12} {'α(pow)':>8} {'R²(pow)':>8} {'R²(exp)':>8}")
    print(f"  {'─'*30} {'─'*8} {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
    
    for label in [l for l, _ in regimes]:
        r = all_results[label]
        final = r['total_subsystems']['final_mean']
        g = r['growth_fit']
        alpha = g['power_exp']
        
        print(f"  {label:<30} {final:>8.0f} {g['type']:>12}"
              f" {alpha:>8.2f} {g['power_r2']:>8.4f} {g['exp_r2']:>8.4f}")
    
    # ─── Key comparisons ───────────────────────────────────────────
    print(f"\n  KEY TESTS:")
    
    uncon = all_results['(a) Unconstrained']
    ns = all_results['(b) +No-signaling only']
    
    print(f"\n  1. Does no-signaling change exponential → power law?")
    print(f"     Unconstrained: type={uncon['growth_fit']['type']}"
          f"  exp_rate={uncon['growth_fit']['exp_rate']:.4f}"
          f"  R²(exp)={uncon['growth_fit']['exp_r2']:.4f}")
    print(f"     +No-signaling: type={ns['growth_fit']['type']}"
          f"  α={ns['growth_fit']['power_exp']:.2f}"
          f"  R²(pow)={ns['growth_fit']['power_r2']:.4f}"
          f"  R²(exp)={ns['growth_fit']['exp_r2']:.4f}")
    
    # Spatial dimension comparison
    sdim_results = {}
    for label in [l for l, _ in regimes]:
        if label.startswith('(s'):
            d_str = label.split(')')[0][-1]
            sdim_results[int(d_str)] = all_results[label]
    
    if sdim_results:
        print(f"\n  2. Does spatial dimension set the power law exponent?")
        print(f"     Prediction: N(t) ~ t^d → α = d")
        for sdim in sorted(sdim_results):
            r = sdim_results[sdim]
            alpha = r['growth_fit']['power_exp']
            print(f"     {sdim}D: α = {alpha:.2f}"
                  f"  (predicted: {sdim})"
                  f"  R²(pow)={r['growth_fit']['power_r2']:.4f}"
                  f"  {'✓' if abs(alpha - sdim) < 0.5 else '✗'}")
    
    bw = all_results.get('(c) +Bandwidth only (κ=4)', {})
    full = all_results.get('(g) Full HSF (κ=4,r=1.5)', {})
    
    if bw and full:
        print(f"\n  3. Full HSF vs individual constraints:")
        print(f"     Unconstrained:   {uncon['total_subsystems']['final_mean']:.0f}")
        print(f"     +NS only:        {ns['total_subsystems']['final_mean']:.0f}")
        print(f"     +BW only:        {bw['total_subsystems']['final_mean']:.0f}")
        print(f"     Full HSF:        {full['total_subsystems']['final_mean']:.0f}")
    
    print(f"\n  HYPOTHESIS CHECK:")
    uncon_type = uncon['growth_fit']['type']
    ns_type = ns['growth_fit']['type']
    full_type = full['growth_fit']['type'] if full else 'N/A'
    
    print(f"    Unconstrained:  {uncon_type} (exp_rate={uncon['growth_fit']['exp_rate']:.4f})")
    print(f"    +No-signaling:  {ns_type} (α={ns['growth_fit']['power_exp']:.2f})")
    print(f"    Full HSF:       {full_type}")
    
    if ns['growth_fit']['power_r2'] > ns['growth_fit']['exp_r2']:
        print(f"    ✓ No-signaling converts exponential → power-law growth")
    else:
        print(f"    ? No-signaling effect unclear (exp still fits better)")
    
    if sdim_results:
        alphas = {d: sdim_results[d]['growth_fit']['power_exp'] for d in sdim_results}
        if all(abs(alphas[d] - d) < 1.0 for d in alphas):
            print(f"    ✓ Power law exponent tracks spatial dimension (α ≈ d)")
        else:
            print(f"    ? Power law exponents: {alphas}")
    
    print(f"\n  Total runtime: {elapsed:.1f}s")
    
    # Save
    os.makedirs('hsf_out', exist_ok=True)
    outpath = 'hsf_out/subsystem_branching.json'
    
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    with open(outpath, 'w') as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()