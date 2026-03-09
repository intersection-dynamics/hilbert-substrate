"""
Task D Part 1: Wilson loops on ladder lattices
Robust eigensolver + area law analysis
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import sys
sys.path.insert(0, '/mnt/user-data/outputs')
from bond_hamiltonian_b1 import EchoLattice, decompose_in_pauli, hamming_weight
from bond_hamiltonian_final import exact_H_eff

np.set_printoptions(precision=8, suppress=True, linewidth=120)
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

def pauli_string_sparse(n_qubits, ops_dict, coeff=1.0):
    """Build sparse Pauli string. ops_dict: {qubit_idx: 'X'|'Y'|'Z'}"""
    dim = 2**n_qubits
    flip_mask = 0; z_mask = 0; y_positions = []
    for q, p in ops_dict.items():
        bit = n_qubits - 1 - q
        if p == 'X': flip_mask |= (1 << bit)
        elif p == 'Y': flip_mask |= (1 << bit); y_positions.append(bit)
        elif p == 'Z': z_mask |= (1 << bit)
    rows = np.arange(dim, dtype=np.int64)
    cols = rows ^ flip_mask
    phases = np.ones(dim, dtype=complex)
    if z_mask:
        z_bits = rows & z_mask
        z_par = np.zeros(dim, dtype=int)
        temp = z_bits.copy()
        while np.any(temp > 0):
            z_par ^= (temp & 1).astype(int); temp >>= 1
        phases *= (-1.0)**z_par
    for bit in y_positions:
        inp = (rows >> bit) & 1
        phases *= np.where(inp == 0, 1j, -1j)
    return csr_matrix((coeff * phases, (cols, rows)), shape=(dim, dim))

# Extract reference Hamiltonian terms from single plaquette
def get_reference_terms(coupling=0.3):
    """Get single-bond and plaquette Pauli terms from 4-site reference."""
    ref = EchoLattice(4, [(0,1),(1,2),(2,3),(0,3)], d_B=2)
    H_ref = exact_H_eff(ref, coupling=coupling)
    coeffs = decompose_in_pauli(H_ref, 4)
    
    w1_terms = {}  # single-bond terms
    w4_terms = {}  # plaquette terms
    for label, c in coeffs.items():
        if abs(c) < 1e-14: continue
        w = hamming_weight(label)
        if w == 1:
            for pos, ch in enumerate(label):
                if ch != 'I': w1_terms.setdefault(pos, {})[ch] = c
        elif w == 4:
            chars = ''.join(ch for ch in label if ch != 'I')
            w4_terms[chars] = c
    
    # Average single-bond terms over positions
    avg_w1 = {}
    for pos, terms in w1_terms.items():
        for ch, c in terms.items():
            avg_w1.setdefault(ch, []).append(c)
    avg_w1 = {ch: np.mean(vals) for ch, vals in avg_w1.items()}
    
    return avg_w1, w4_terms

class Ladder:
    """Ladder lattice: L rungs, 2 rails + L rungs = 2(L-1) + L bonds"""
    def __init__(self, L):
        self.L = L
        # Bonds: top rail + bottom rail + rungs
        # top[i] = bond between top sites i and i+1
        # bot[i] = bond between bottom sites i and i+1
        # rung[i] = bond between top i and bottom i
        self.n_top = L - 1
        self.n_bot = L - 1
        self.n_rung = L
        self.n_bonds = 2*(L-1) + L  # = 3L - 2
        self.dim = 2**self.n_bonds
        
        # Bond indices
        self.top = list(range(0, L-1))
        self.bot = list(range(L-1, 2*(L-1)))
        self.rung = list(range(2*(L-1), 3*L-2))
        
        # Plaquettes: each square uses top[i], rung[i+1], bot[i], rung[i]
        self.plaquettes = []
        for i in range(L-1):
            self.plaquettes.append([
                self.top[i], self.rung[i+1], 
                self.bot[i], self.rung[i]
            ])
    
    def wilson_loop(self, start, width):
        """Wilson loop from rung 'start' spanning 'width' squares."""
        if start + width >= self.L: return None
        bonds = []
        # Top: start to start+width-1
        for i in range(start, start + width):
            bonds.append(self.top[i])
        # Right rung
        bonds.append(self.rung[start + width])
        # Bottom: start+width-1 back to start
        for i in range(start + width - 1, start - 1, -1):
            bonds.append(self.bot[i])
        # Left rung
        bonds.append(self.rung[start])
        return bonds

def build_ladder_H(ladder, coupling=0.3):
    """Build effective bond Hamiltonian for ladder."""
    w1, w4 = get_reference_terms(coupling)
    n = ladder.n_bonds
    dim = 2**n
    H = csr_matrix((dim, dim), dtype=complex)
    
    # Single-bond terms on each bond
    for b in range(n):
        for ch, c in w1.items():
            H += pauli_string_sparse(n, {b: ch}, coeff=c)
    
    # Plaquette terms on each square
    for plaq in ladder.plaquettes:
        for chars, c in w4.items():
            ops = {plaq[k]: chars[k] for k in range(4)}
            H += pauli_string_sparse(n, ops, coeff=c)
    
    return H

def ground_state(H, dim):
    """Find ground state robustly."""
    if dim <= 4096:
        M = H.toarray() if hasattr(H, 'toarray') else H
        ev, evc = np.linalg.eigh(M)
        return ev[0], evc[:, 0], ev[1] - ev[0]
    # Sparse: shift-invert for robustness
    try:
        ev, evc = eigsh(H, k=2, which='SA', maxiter=20000, tol=1e-10)
        idx = np.argsort(ev)
        return ev[idx[0]], evc[:, idx[0]], ev[idx[1]] - ev[idx[0]]
    except:
        # Fallback: shift-invert
        ev, evc = eigsh(H, k=2, sigma=-10.0, which='LM', maxiter=20000)
        idx = np.argsort(ev)
        return ev[idx[0]], evc[:, idx[0]], ev[idx[1]] - ev[idx[0]]

def wilson_expval(loop_bonds, n_bonds, psi):
    """<psi| prod_b sigma_x^b |psi>"""
    W = pauli_string_sparse(n_bonds, {b: 'X' for b in loop_bonds})
    return (psi.conj() @ W @ psi).real

# ============ MAIN ============
print("="*60)
print("TASK D PART 1: LADDER WILSON LOOPS")
print("="*60)

couplings = [0.15, 0.25, 0.35, 0.45]
results = {}

for L in [4, 5, 6, 7]:
    lat = Ladder(L)
    if lat.dim > 65536:
        print(f"\nSkipping L={L} (dim={lat.dim})")
        continue
    
    for g in couplings:
        print(f"\n--- L={L}, g={g:.2f} (n_bonds={lat.n_bonds}, dim={lat.dim}) ---")
        H = build_ladder_H(lat, coupling=g)
        E0, psi, gap = ground_state(H, lat.dim)
        
        print(f"  E0={E0:.6f}, gap={gap:.6f}")
        
        wl = {}
        for R in range(1, L):
            loop = lat.wilson_loop(0, R)
            if loop is None: continue
            
            # Average over all starting positions
            vals = []
            for start in range(L - R):
                lp = lat.wilson_loop(start, R)
                if lp is not None:
                    vals.append(wilson_expval(lp, lat.n_bonds, psi))
            
            avg = np.mean(vals)
            V = -np.log(abs(avg)) if abs(avg) > 1e-15 else float('inf')
            wl[R] = {'mean': avg, 'V': V, 'area': R, 'perim': 2*R+2}
            print(f"    R={R}: <W>={avg:+.8f}, V(R)={V:.4f}, area={R}, perim={2*R+2}")
        
        results[(L, g)] = {'wl': wl, 'E0': E0, 'gap': gap}

# Fit V(R) = sigma*R + const
print("\n" + "="*60)
print("LINEAR FIT: V(R) = sigma*R + const")
print("="*60)

for key in sorted(results.keys()):
    L, g = key
    wl = results[key]['wl']
    Rs = sorted(wl.keys())
    if len(Rs) < 2: continue
    
    R_arr = np.array([wl[R]['area'] for R in Rs])
    V_arr = np.array([wl[R]['V'] for R in Rs])
    
    mask = np.isfinite(V_arr)
    if np.sum(mask) < 2: continue
    
    p = np.polyfit(R_arr[mask], V_arr[mask], 1)
    sigma = p[0]
    
    Vfit = np.polyval(p, R_arr[mask])
    SS_res = np.sum((V_arr[mask] - Vfit)**2)
    SS_tot = np.sum((V_arr[mask] - np.mean(V_arr[mask]))**2)
    R2 = 1 - SS_res/SS_tot if SS_tot > 0 else 0
    
    print(f"  L={L}, g={g:.2f}: sigma={sigma:.6f}, R2={R2:.4f}")
    results[key]['sigma'] = sigma
    results[key]['R2'] = R2

print("\nDone.")