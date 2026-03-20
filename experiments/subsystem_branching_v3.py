#!/usr/bin/env python3
"""
subsystem_branching_v3.py
==========================
HSF — Subsystem Branching with Light Cone + Energy Cost

V2 RESULT: Full HSF gives N(t) ~ t^d with α tracking spatial dimension.
But post-inflation cosmology gives t^{3/2} (radiation) or t^2 (matter),
not t^3. The missing ingredient: creation costs energy, and energy
density dilutes as volume expands, slowing growth.

V3 ADDITION: Energy conservation.
  - The substrate starts with a finite energy pool E₀
  - Creating a subsystem costs energy ε (drawn from the pool)
  - The creation rate at a link is modulated by local energy density:
      p_create = interaction_rate × (ρ_E / ρ₀)^γ
    where ρ_E = E_remaining / V(frontier) is the current energy density,
    ρ₀ is the initial density, and γ controls how strongly density
    affects creation rate.
  - As space expands, energy density drops, creation slows.

PREDICTION:
  Without energy cost: N(t) ~ t^d  (v2 result, constant frontier density)
  With energy cost:    N(t) ~ t^α where α < d
    The exponent α depends on γ (energy sensitivity):
    γ = 0: recovers v2 (energy doesn't matter)  → α = d
    γ = 1: creation rate ~ 1/V ~ 1/t^d          → α < d
    γ = 1/2: intermediate                        → radiation-like?

COSMOLOGY TARGETS (3D):
  Radiation era: V ~ t^{3/2}  → α_eff ≈ 1.5
  Matter era:    V ~ t^2      → α_eff ≈ 2.0

DEPENDENCIES: numpy
RUN: python subsystem_branching_v3.py [--nsteps 400] [--ntrials 25]
"""

import numpy as np
import json
import os
import time
import argparse
from collections import defaultdict


class SubstrateGraph:
    """
    Growing graph with a hard light-cone constraint in physical space.
    """
    
    def __init__(self, n_initial=2, d=3, bandwidth=None,
                 no_refolding=False, spatial_dim=1,
                 light_speed=1.0, no_signaling=False,
                 spatial_exclusion=False, exclusion_radius=0.2,
                 energy_total=None, energy_cost=1.0, energy_gamma=0.0):
        """
        Args:
            energy_total: initial energy pool (None = infinite)
            energy_cost: energy per subsystem creation
            energy_gamma: exponent for density-dependent creation rate
                          p_create *= (ρ_E / ρ₀)^γ
                          γ=0: no energy effect (recovers v2)
                          γ=1: creation rate proportional to energy density
        """
        self.d = d
        self.bandwidth = bandwidth
        self.no_refolding = no_refolding
        self.spatial_dim = spatial_dim
        self.light_speed = light_speed
        self.no_signaling = no_signaling
        self.spatial_exclusion = spatial_exclusion
        self.exclusion_radius = exclusion_radius
        
        # Energy tracking
        self.energy_total = energy_total  # None = infinite
        self.energy_remaining = energy_total if energy_total is not None else float('inf')
        self.energy_cost = energy_cost
        self.energy_gamma = energy_gamma
        self.initial_energy_density = None  # set after first volume calc
        
        self.timestep = 0
        self.n_subsystems = n_initial
        self.n_links = 0
        self.n_interface_subsystems = 0
        self.links = set()
        self.degree = defaultdict(int)
        
        self.positions = {}
        for i in range(n_initial):
            self.positions[i] = np.random.randn(spatial_dim) * 0.1
        
        self.activation_time = {i: 0 for i in range(n_initial)}
        self._all_pos_dirty = True
        
        for i in range(n_initial - 1):
            self._add_link(i, i + 1)
        
        self.history = {
            'total_subsystems': [self.total_subsystems()],
            'site_subsystems': [self.n_subsystems],
            'interface_subsystems': [self.n_interface_subsystems],
            'n_links': [self.n_links],
            'mean_degree': [self._mean_degree()],
            'hilbert_log': [self._log_hilbert_dim()],
            'frontier_radius': [0.0],
            'density': [float(self.n_subsystems)],
            'energy_remaining': [self.energy_remaining if self.energy_total else 0],
            'energy_density': [0.0],
            'creation_rate_modifier': [1.0],
        }
    
    def total_subsystems(self):
        return self.n_subsystems + self.n_interface_subsystems
    
    def _mean_degree(self):
        if self.n_subsystems == 0: return 0
        return 2 * self.n_links / max(1, self.n_subsystems)
    
    def _log_hilbert_dim(self):
        return (self.n_subsystems * np.log2(self.d) +
                self.n_links * 2 * np.log2(self.d))
    
    def _frontier_radius(self):
        if not self.positions: return 0.0
        origin = np.zeros(self.spatial_dim)
        return max(np.linalg.norm(p - origin) for p in self.positions.values())
    
    def _volume(self, r):
        d = self.spatial_dim
        if d == 1: return 2 * r
        elif d == 2: return np.pi * r**2
        else: return (4/3) * np.pi * r**3
    
    def _density(self):
        vol = self._volume(self._frontier_radius())
        if vol < 1e-10: return float(self.n_subsystems)
        return self.n_subsystems / vol
    
    def _energy_density(self):
        """Current energy density = remaining energy / frontier volume."""
        if self.energy_total is None:
            return 0.0
        vol = self._volume(max(self._frontier_radius(), 0.1))
        return self.energy_remaining / vol
    
    def _creation_rate_modifier(self):
        """
        Multiplicative factor on creation rate due to energy density.
        Returns (ρ_E / ρ₀)^γ, clamped to [0, 1].
        """
        if self.energy_total is None or self.energy_gamma == 0:
            return 1.0
        
        if self.energy_remaining <= 0:
            return 0.0
        
        rho_E = self._energy_density()
        
        # Set initial density on first real call
        if self.initial_energy_density is None or self.initial_energy_density <= 0:
            self.initial_energy_density = rho_E
            return 1.0
        
        ratio = rho_E / self.initial_energy_density
        modifier = ratio ** self.energy_gamma
        return min(modifier, 1.0)
    
    def _is_active(self, node_id):
        return (node_id in self.activation_time and
                self.activation_time[node_id] < self.timestep)
    
    def _can_link(self, i, j):
        if i == j: return False
        edge = (min(i, j), max(i, j))
        if edge in self.links: return False
        if self.bandwidth is not None:
            if self.degree[i] >= self.bandwidth or self.degree[j] >= self.bandwidth:
                return False
        if i in self.positions and j in self.positions:
            dist = np.linalg.norm(self.positions[i] - self.positions[j])
            if dist > self.light_speed * 1.5:
                return False
        return True
    
    def _add_link(self, i, j):
        edge = (min(i, j), max(i, j))
        self.links.add(edge)
        self.n_links += 1
        self.degree[i] += 1
        self.degree[j] += 1
        self.n_interface_subsystems += 2
    
    def _propose_position(self, parent):
        parent_pos = self.positions[parent]
        direction = np.random.randn(self.spatial_dim)
        direction /= np.linalg.norm(direction) + 1e-10
        distance = np.random.uniform(0.1, self.light_speed * 0.8)
        return parent_pos + direction * distance
    
    def _rebuild_active_positions(self):
        """Cache active positions as numpy array for fast distance checks."""
        active_ids = [nid for nid, ta in self.activation_time.items()
                      if ta < self.timestep]
        if active_ids:
            self._active_pos_array = np.array([self.positions[nid] for nid in active_ids])
        else:
            self._active_pos_array = np.empty((0, self.spatial_dim))
        self._active_cache_step = self.timestep
    
    def _in_light_cone(self, pos):
        """Hard spatial speed limit: pos must be within c of an active node."""
        if not self.no_signaling:
            return True
        if not hasattr(self, '_active_cache_step') or self._active_cache_step != self.timestep:
            self._rebuild_active_positions()
        if len(self._active_pos_array) == 0:
            return False
        dists = np.linalg.norm(self._active_pos_array - pos, axis=1)
        return np.min(dists) <= self.light_speed
    
    def _check_exclusion(self, pos):
        if not self.spatial_exclusion:
            return True
        if not hasattr(self, '_all_pos_array') or self._all_pos_dirty:
            ids = sorted(self.positions.keys())
            if ids:
                self._all_pos_array = np.array([self.positions[i] for i in ids])
            else:
                self._all_pos_array = np.empty((0, self.spatial_dim))
            self._all_pos_dirty = False
        if len(self._all_pos_array) == 0:
            return True
        dists = np.linalg.norm(self._all_pos_array - pos, axis=1)
        return np.min(dists) >= self.exclusion_radius
    
    def _add_subsystem(self, pos):
        new_id = self.n_subsystems
        self.n_subsystems += 1
        self.positions[new_id] = pos
        self.activation_time[new_id] = self.timestep
        self._all_pos_dirty = True
        return new_id
    
    def _record(self):
        self.history['total_subsystems'].append(self.total_subsystems())
        self.history['site_subsystems'].append(self.n_subsystems)
        self.history['interface_subsystems'].append(self.n_interface_subsystems)
        self.history['n_links'].append(self.n_links)
        self.history['mean_degree'].append(self._mean_degree())
        self.history['hilbert_log'].append(self._log_hilbert_dim())
        self.history['frontier_radius'].append(self._frontier_radius())
        self.history['density'].append(self._density())
        self.history['energy_remaining'].append(
            self.energy_remaining if self.energy_total else 0)
        self.history['energy_density'].append(self._energy_density())
        self.history['creation_rate_modifier'].append(self._creation_rate_modifier())
    
    def step(self, interaction_rate=0.1, spawn_rate=0.05, max_subsystems=10000):
        self.timestep += 1
        if self.n_subsystems >= max_subsystems:
            self._record()
            return
        
        # Energy check: can we create anything?
        if self.energy_total is not None and self.energy_remaining < self.energy_cost:
            self._record()
            return
        
        # Creation rate modifier from energy density
        rate_mod = self._creation_rate_modifier()
        effective_rate = interaction_rate * rate_mod
        
        # Rebuild caches for this timestep
        self._rebuild_active_positions()
        self._all_pos_dirty = True
        
        new_subsystems = []
        
        # Phase 1: Active links spawn new subsystems
        for (i, j) in list(self.links):
            if self.n_subsystems >= max_subsystems:
                break
            if self.energy_total is not None and self.energy_remaining < self.energy_cost:
                break
            if not (self._is_active(i) and self._is_active(j)):
                continue
            if np.random.random() < effective_rate:
                parent = i if np.random.random() < 0.5 else j
                pos = self._propose_position(parent)
                if not self._in_light_cone(pos):
                    continue
                if not self._check_exclusion(pos):
                    continue
                
                # Pay energy cost
                if self.energy_total is not None:
                    self.energy_remaining -= self.energy_cost
                
                new_id = self._add_subsystem(pos)
                new_subsystems.append((new_id, parent))
        
        # Phase 2: Link new subsystems to parents
        for new_id, parent in new_subsystems:
            if self._can_link(new_id, parent):
                self._add_link(new_id, parent)
        
        # Phase 3: Nearby active unlinked pairs form links
        n_samples = min(self.n_subsystems * 2, 500)
        for _ in range(n_samples):
            i = np.random.randint(self.n_subsystems)
            j = np.random.randint(self.n_subsystems)
            if i != j and self._can_link(i, j):
                if self._is_active(i) and self._is_active(j):
                    if np.random.random() < spawn_rate:
                        self._add_link(i, j)
        
        self._record()


# ═══════════════════════════════════════════════════════════════════════

def fit_growth(t, N_t):
    t = np.array(t, dtype=float)
    N_t = np.array(N_t, dtype=float)
    start = max(2, len(t) // 7)
    tf, Nf = t[start:], N_t[start:]
    
    if Nf[-1] <= Nf[0] + 1:
        return {'type': 'stalled', 'exp_rate': 0, 'power_exp': 0,
                'lin_rate': 0, 'exp_r2': 0, 'power_r2': 0, 'lin_r2': 0}
    
    logN = np.log(np.maximum(Nf, 1))
    logt = np.log(np.maximum(tf, 1))
    
    def r2(y, yp):
        ss = np.sum((y - yp)**2)
        st = np.sum((y - y.mean())**2)
        return 1 - ss / max(st, 1e-10)
    
    ce = np.polyfit(tf, logN, 1)
    r2e = r2(logN, np.polyval(ce, tf))
    
    cp = np.polyfit(logt, logN, 1)
    r2p = r2(logN, np.polyval(cp, logt))
    
    cl = np.polyfit(tf, Nf, 1)
    r2l = r2(Nf, np.polyval(cl, tf))
    
    result = {
        'exp_rate': float(ce[0]), 'exp_r2': float(r2e),
        'power_exp': float(cp[0]), 'power_r2': float(r2p),
        'lin_rate': float(cl[0]), 'lin_r2': float(r2l),
    }
    
    if r2p > r2e + 0.005:
        result['type'] = 'power_law'
    elif r2e > r2p + 0.005:
        result['type'] = 'exponential'
    else:
        result['type'] = 'power_law' if r2p > 0.95 else 'exponential'
    
    return result


def fit_frontier(t, r_t):
    t = np.array(t, dtype=float)
    r_t = np.array(r_t, dtype=float)
    start = max(2, len(t) // 7)
    tf, rf = t[start:], r_t[start:]
    if rf[-1] <= rf[0] + 0.01:
        return {'speed': 0, 'r2': 0}
    c = np.polyfit(tf, rf, 1)
    ss = np.sum((rf - np.polyval(c, tf))**2)
    st = np.sum((rf - rf.mean())**2)
    return {'speed': float(c[0]), 'r2': float(1 - ss / max(st, 1e-10))}


# ═══════════════════════════════════════════════════════════════════════

def run_regime(label, n_steps=300, n_initial=2, d=3,
               bandwidth=None, no_refolding=False,
               spatial_dim=1, light_speed=1.0,
               no_signaling=False, spatial_exclusion=False,
               exclusion_radius=0.2,
               energy_total=None, energy_cost=1.0, energy_gamma=0.0,
               interaction_rate=0.1, spawn_rate=0.05,
               n_trials=20, max_subsystems=10000):
    
    collectors = defaultdict(list)
    
    for trial in range(n_trials):
        np.random.seed(42 + trial * 1000 + hash(label) % 10000)
        
        g = SubstrateGraph(
            n_initial=n_initial, d=d, bandwidth=bandwidth,
            no_refolding=no_refolding, spatial_dim=spatial_dim,
            light_speed=light_speed, no_signaling=no_signaling,
            spatial_exclusion=spatial_exclusion,
            exclusion_radius=exclusion_radius,
            energy_total=energy_total, energy_cost=energy_cost,
            energy_gamma=energy_gamma,
        )
        
        for t in range(n_steps):
            g.step(interaction_rate=interaction_rate,
                   spawn_rate=spawn_rate,
                   max_subsystems=max_subsystems)
        
        for key in g.history:
            collectors[key].append(g.history[key])
    
    result = {'label': label, 'n_steps': n_steps, 'n_trials': n_trials,
              'params': {
                  'bandwidth': bandwidth, 'no_refolding': no_refolding,
                  'no_signaling': no_signaling, 'spatial_dim': spatial_dim,
                  'light_speed': light_speed, 'interaction_rate': interaction_rate,
                  'spawn_rate': spawn_rate, 'spatial_exclusion': spatial_exclusion,
                  'exclusion_radius': exclusion_radius,
                  'energy_total': energy_total, 'energy_cost': energy_cost,
                  'energy_gamma': energy_gamma,
              }}
    
    for key in collectors:
        arr = np.array(collectors[key])
        result[key] = {
            'mean': arr.mean(axis=0).tolist(),
            'std': arr.std(axis=0).tolist(),
            'final_mean': float(arr[:, -1].mean()),
            'final_std': float(arr[:, -1].std()),
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='HSF Branching v3: Energy Cost')
    parser.add_argument('--nsteps', type=int, default=300)
    parser.add_argument('--ntrials', type=int, default=20)
    args = parser.parse_args()
    
    t_start = time.time()
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  HSF Subsystem Branching v3: Light Cone + Energy Cost              ║")
    print("║  Creation costs energy. Energy density dilutes as volume grows.    ║")
    print("║  Prediction: α < d, approaching cosmological exponents.            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    n_steps = args.nsteps
    n_trials = args.ntrials
    
    # Full HSF baseline (all constraints, no energy = v2 recovery)
    hsf_base = dict(no_signaling=True, bandwidth=4, no_refolding=True,
                    spatial_exclusion=True, exclusion_radius=0.15)
    
    # Energy: E₀ large enough to not run out, but density dilution matters
    # With ~5000 subsystems at cost 1.0 each, E₀=50000 won't exhaust
    E0 = 50000
    
    common = dict(n_steps=n_steps, n_initial=2, d=3, n_trials=n_trials,
                  interaction_rate=0.08, spawn_rate=0.03,
                  light_speed=1.0, max_subsystems=6000)
    
    regimes = [
        # ── v2 baseline: no energy cost ──
        ('(a) v2 baseline 3D',
         dict(**hsf_base, spatial_dim=3,
              energy_total=None, energy_gamma=0)),
        
        # ── γ sweep in 3D: how does energy sensitivity change α? ──
        ('(b) γ=0.25 3D',
         dict(**hsf_base, spatial_dim=3,
              energy_total=E0, energy_cost=1.0, energy_gamma=0.25)),
        
        ('(c) γ=0.50 3D',
         dict(**hsf_base, spatial_dim=3,
              energy_total=E0, energy_cost=1.0, energy_gamma=0.50)),
        
        ('(d) γ=0.75 3D',
         dict(**hsf_base, spatial_dim=3,
              energy_total=E0, energy_cost=1.0, energy_gamma=0.75)),
        
        ('(e) γ=1.00 3D',
         dict(**hsf_base, spatial_dim=3,
              energy_total=E0, energy_cost=1.0, energy_gamma=1.00)),
        
        ('(f) γ=1.50 3D',
         dict(**hsf_base, spatial_dim=3,
              energy_total=E0, energy_cost=1.0, energy_gamma=1.50)),
        
        # ── Best γ across dimensions ──
        ('(g) γ=0.50 1D',
         dict(**hsf_base, spatial_dim=1,
              energy_total=E0, energy_cost=1.0, energy_gamma=0.50)),
        
        ('(h) γ=0.50 2D',
         dict(**hsf_base, spatial_dim=2,
              energy_total=E0, energy_cost=1.0, energy_gamma=0.50)),
        
        # (c) already covers γ=0.50 3D
        
        ('(i) γ=1.00 1D',
         dict(**hsf_base, spatial_dim=1,
              energy_total=E0, energy_cost=1.0, energy_gamma=1.00)),
        
        ('(j) γ=1.00 2D',
         dict(**hsf_base, spatial_dim=2,
              energy_total=E0, energy_cost=1.0, energy_gamma=1.00)),
        
        # (e) already covers γ=1.00 3D
    ]
    
    all_results = {}
    
    for label, params in regimes:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        
        t0 = time.time()
        r = run_regime(label, **{**common, **params})
        dt = time.time() - t0
        
        times = list(range(n_steps + 1))
        growth = fit_growth(times, r['total_subsystems']['mean'])
        r['growth_fit'] = growth
        front = fit_frontier(times, r['frontier_radius']['mean'])
        r['frontier_fit'] = front
        
        final = r['total_subsystems']['final_mean']
        final_std = r['total_subsystems']['final_std']
        final_r = r['frontier_radius']['final_mean']
        
        # Energy info
        e_info = ""
        if 'energy_remaining' in r:
            e_rem = r['energy_remaining']['final_mean']
            e_mod = r['creation_rate_modifier']['final_mean']
            if params.get('energy_total') is not None:
                e_info = f"  E_rem={e_rem:.0f}/{params['energy_total']}  mod={e_mod:.3f}"
        
        print(f"    N = {final:.0f} ± {final_std:.0f}   R = {final_r:.1f}{e_info}")
        print(f"    Growth: {growth['type']}  α={growth['power_exp']:.2f}"
              f"  R²p={growth['power_r2']:.3f}  R²e={growth['exp_r2']:.3f}")
        print(f"    ({dt:.1f}s)")
        
        all_results[label] = r
    
    elapsed = time.time() - t_start
    
    # ═══ Summary ═══
    print(f"\n\n{'═' * 78}")
    print(f"  RESULTS")
    print(f"{'═' * 78}\n")
    
    print(f"  {'Regime':<20} {'dim':>3} {'γ':>5} {'N':>7} {'α':>6} {'R²p':>6}"
          f" {'R²e':>6} {'E_rem':>7} {'mod':>6}")
    print(f"  {'─'*20} {'─'*3} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*6}")
    
    for label, params in regimes:
        r = all_results[label]
        g = r['growth_fit']
        sd = params.get('spatial_dim', 1)
        gam = params.get('energy_gamma', 0)
        e_rem = r['energy_remaining']['final_mean'] if params.get('energy_total') else float('inf')
        mod = r['creation_rate_modifier']['final_mean']
        e_str = f"{e_rem:>7.0f}" if e_rem < float('inf') else "    inf"
        
        print(f"  {label:<20} {sd:>3} {gam:>5.2f} "
              f"{r['total_subsystems']['final_mean']:>7.0f} "
              f"{g['power_exp']:>6.2f} {g['power_r2']:>6.3f} {g['exp_r2']:>6.3f}"
              f" {e_str} {mod:>6.3f}")
    
    # ═══ γ sweep analysis (3D) ═══
    print(f"\n  ═══ γ SWEEP (3D): How energy sensitivity changes growth exponent ═══")
    print(f"  Cosmology targets: radiation α≈1.5, matter α≈2.0, v2 baseline α≈3.0\n")
    
    gamma_labels = [
        ('(a) v2 baseline 3D', 0),
        ('(b) γ=0.25 3D', 0.25),
        ('(c) γ=0.50 3D', 0.50),
        ('(d) γ=0.75 3D', 0.75),
        ('(e) γ=1.00 3D', 1.00),
        ('(f) γ=1.50 3D', 1.50),
    ]
    
    print(f"    {'γ':>5} {'α':>6} {'R²':>6} {'N(fin)':>8} {'Interpretation'}")
    print(f"    {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*30}")
    
    for lab, gam in gamma_labels:
        if lab in all_results:
            g = all_results[lab]['growth_fit']
            n = all_results[lab]['total_subsystems']['final_mean']
            alpha = g['power_exp']
            
            if alpha > 2.7:
                interp = "≈ geometric (v2-like)"
            elif alpha > 2.2:
                interp = "matter-like"
            elif alpha > 1.7:
                interp = "≈ MATTER ERA (t²)"
            elif alpha > 1.3:
                interp = "≈ RADIATION ERA (t^{3/2})"
            elif alpha > 0.8:
                interp = "sub-radiation"
            else:
                interp = "stalled"
            
            print(f"    {gam:>5.2f} {alpha:>6.2f} {g['power_r2']:>6.3f}"
                  f" {n:>8.0f} {interp}")
    
    # ═══ Dimension test with energy ═══
    print(f"\n  ═══ α vs DIMENSION with energy cost ═══")
    
    for gam_val, gam_label in [(0.5, 'γ=0.50'), (1.0, 'γ=1.00')]:
        print(f"\n  {gam_label}:")
        dim_labs = []
        if gam_val == 0.5:
            dim_labs = [('(g) γ=0.50 1D', 1), ('(h) γ=0.50 2D', 2), ('(c) γ=0.50 3D', 3)]
        else:
            dim_labs = [('(i) γ=1.00 1D', 1), ('(j) γ=1.00 2D', 2), ('(e) γ=1.00 3D', 3)]
        
        for lab, dim in dim_labs:
            if lab in all_results:
                g = all_results[lab]['growth_fit']
                print(f"    {dim}D: α = {g['power_exp']:.2f}  R²={g['power_r2']:.3f}"
                      f"  (v2 pred: {dim}, cosmo rad: {dim/2:.1f}, cosmo mat: {2*dim/3:.1f})")
    
    print(f"\n  Runtime: {elapsed:.1f}s")
    
    # Save
    os.makedirs('hsf_out', exist_ok=True)
    outpath = 'hsf_out/subsystem_branching_v3.json'
    
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if obj == float('inf'): return None
        return obj
    
    with open(outpath, 'w') as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()