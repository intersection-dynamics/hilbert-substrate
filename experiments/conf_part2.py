"""
Task D Part 2: Grid Wilson loops
Area vs Perimeter discrimination requires 2D loops with different shapes.
Key test: loops with SAME perimeter but DIFFERENT area.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import sys
sys.path.insert(0, '/mnt/user-data/outputs')
from bond_hamiltonian_b1 import EchoLattice, decompose_in_pauli, hamming_weight
from bond_hamiltonian_final import exact_H_eff

I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)

def pauli_string_sparse(n_qubits, ops_dict, coeff=1.0):
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
        z_bits = rows & z_mask; z_par = np.zeros(dim, dtype=int); temp = z_bits.copy()
        while np.any(temp > 0): z_par ^= (temp & 1).astype(int); temp >>= 1
        phases *= (-1.0)**z_par
    for bit in y_positions:
        inp = (rows >> bit) & 1; phases *= np.where(inp == 0, 1j, -1j)
    return csr_matrix((coeff * phases, (cols, rows)), shape=(dim, dim))

def get_reference_terms(coupling=0.3):
    ref = EchoLattice(4, [(0,1),(1,2),(2,3),(0,3)], d_B=2)
    H_ref = exact_H_eff(ref, coupling=coupling)
    coeffs = decompose_in_pauli(H_ref, 4)
    w1, w4 = {}, {}
    for label, c in coeffs.items():
        if abs(c) < 1e-14: continue
        w = hamming_weight(label)
        if w == 1:
            for pos, ch in enumerate(label):
                if ch != 'I': w1.setdefault(pos, {})[ch] = c
        elif w == 4:
            chars = ''.join(ch for ch in label if ch != 'I')
            w4[chars] = c
    avg_w1 = {}
    for pos, terms in w1.items():
        for ch, c in terms.items():
            avg_w1.setdefault(ch, []).append(c)
    return {ch: np.mean(v) for ch, v in avg_w1.items()}, w4

class Grid:
    """Open-boundary rectangular grid Lx columns × Ly rows."""
    def __init__(self, Lx, Ly):
        self.Lx, self.Ly = Lx, Ly
        self.n_h = (Lx-1)*Ly  # horizontal bonds
        self.n_v = Lx*(Ly-1)  # vertical bonds
        self.n_bonds = self.n_h + self.n_v
        self.dim = 2**self.n_bonds
        
        # Bond indexing: horizontal first, then vertical
        # h(x, y) = bond between site (x,y)-(x+1,y), x=0..Lx-2, y=0..Ly-1
        # v(x, y) = bond between site (x,y)-(x,y+1), x=0..Lx-1, y=0..Ly-2
        self.h_idx = {}
        idx = 0
        for x in range(Lx-1):
            for y in range(Ly):
                self.h_idx[(x, y)] = idx; idx += 1
        self.v_idx = {}
        for x in range(Lx):
            for y in range(Ly-1):
                self.v_idx[(x, y)] = idx; idx += 1
        
        # Plaquettes: (x,y) for x=0..Lx-2, y=0..Ly-2
        # bonds: bottom h(x,y), right v(x+1,y), top h(x,y+1), left v(x,y)
        self.plaquettes = []
        for x in range(Lx-1):
            for y in range(Ly-1):
                self.plaquettes.append([
                    self.h_idx[(x, y)],
                    self.v_idx[(x+1, y)],
                    self.h_idx[(x, y+1)],
                    self.v_idx[(x, y)],
                ])
    
    def wilson_loop(self, x0, y0, W, H):
        """Rectangular loop starting at (x0,y0), width W, height H."""
        if x0 + W >= self.Lx or y0 + H >= self.Ly: return None
        bonds = []
        # Bottom
        for x in range(x0, x0 + W):
            bonds.append(self.h_idx[(x, y0)])
        # Right
        for y in range(y0, y0 + H):
            bonds.append(self.v_idx[(x0 + W, y)])
        # Top (reverse)
        for x in range(x0 + W - 1, x0 - 1, -1):
            bonds.append(self.h_idx[(x, y0 + H)])
        # Left (reverse)
        for y in range(y0 + H - 1, y0 - 1, -1):
            bonds.append(self.v_idx[(x0, y)])
        return bonds

def build_grid_H(grid, coupling=0.3):
    w1, w4 = get_reference_terms(coupling)
    n = grid.n_bonds; dim = 2**n
    H = csr_matrix((dim, dim), dtype=complex)
    for b in range(n):
        for ch, c in w1.items():
            H += pauli_string_sparse(n, {b: ch}, coeff=c)
    for plaq in grid.plaquettes:
        for chars, c in w4.items():
            ops = {plaq[k]: chars[k] for k in range(4)}
            H += pauli_string_sparse(n, ops, coeff=c)
    return H

def wilson_expval(loop_bonds, n_bonds, psi):
    W = pauli_string_sparse(n_bonds, {b: 'X' for b in loop_bonds})
    return (psi.conj() @ W @ psi).real

# ============ MAIN ============
print("="*60)
print("TASK D PART 2: GRID WILSON LOOPS")
print("="*60)

for Lx, Ly in [(3,3), (4,3), (3,4)]:
    grid = Grid(Lx, Ly)
    print(f"\nGrid {Lx}x{Ly}: {grid.n_bonds} bonds, dim={grid.dim}, "
          f"{len(grid.plaquettes)} plaquettes")
    
    if grid.dim > 262144:
        print("  SKIPPED (too large)")
        continue
    
    for g in [0.30, 0.40]:
        print(f"\n  --- g = {g:.2f} ---")
        H = build_grid_H(grid, coupling=g)
        
        if grid.dim <= 4096:
            M = H.toarray()
            evals, evecs = np.linalg.eigh(M)
            psi = evecs[:, 0]; E0 = evals[0]; gap = evals[1] - evals[0]
        else:
            evals, evecs = eigsh(H, k=2, which='SA', maxiter=30000, tol=1e-10)
            idx = np.argsort(evals)
            psi = evecs[:, idx[0]]; E0 = evals[idx[0]]; gap = evals[idx[1]] - evals[idx[0]]
        
        print(f"  E0={E0:.6f}, gap={gap:.6f}")
        
        # Enumerate all rectangular Wilson loops
        loop_data = {}
        for W in range(1, Lx):
            for H_loop in range(1, Ly):
                vals = []
                for x0 in range(Lx - W):
                    for y0 in range(Ly - H_loop):
                        lp = grid.wilson_loop(x0, y0, W, H_loop)
                        if lp is not None:
                            vals.append(wilson_expval(lp, grid.n_bonds, psi))
                if vals:
                    A = W * H_loop
                    P = 2*(W + H_loop)
                    avg = np.mean(vals)
                    V = -np.log(abs(avg)) if abs(avg) > 1e-15 else float('inf')
                    loop_data[(W, H_loop)] = {
                        'area': A, 'perim': P, 'W_avg': avg, 'V': V, 'count': len(vals)
                    }
        
        print(f"\n  {'WxH':>5} {'Area':>5} {'Perim':>6} {'<W>':>12} {'V':>10} {'n':>3}")
        print(f"  {'-'*46}")
        for k in sorted(loop_data.keys(), key=lambda x: (loop_data[x]['area'], loop_data[x]['perim'])):
            d = loop_data[k]
            print(f"  {k[0]}x{k[1]:>3} {d['area']:>5} {d['perim']:>6} {d['W_avg']:>12.8f} "
                  f"{d['V']:>10.4f} {d['count']:>3}")
        
        # KEY TEST: same perimeter, different area
        by_perim = {}
        for k, d in loop_data.items():
            by_perim.setdefault(d['perim'], []).append((k, d))
        
        print(f"\n  AREA vs PERIMETER DISCRIMINATION:")
        for P in sorted(by_perim.keys()):
            entries = by_perim[P]
            if len(entries) >= 2:
                areas = [e[1]['area'] for e in entries]
                Vs = [e[1]['V'] for e in entries]
                labels = [f"{e[0][0]}x{e[0][1]}" for e in entries]
                print(f"    Perim={P}: {', '.join(f'{l}(A={a},V={v:.2f})' for l,a,v in zip(labels,areas,Vs))}")
                # Area law predicts larger A -> larger V
                if len(set(areas)) > 1:
                    sorted_by_a = sorted(zip(areas, Vs))
                    if all(sorted_by_a[i][1] <= sorted_by_a[i+1][1] for i in range(len(sorted_by_a)-1)):
                        print(f"      -> V increases with area: AREA LAW ✓")
                    else:
                        print(f"      -> V does NOT increase with area: check perimeter law")
        
        # Fit: V = sigma*A + mu*P + c
        if len(loop_data) >= 3:
            A_arr = np.array([d['area'] for d in loop_data.values()])
            P_arr = np.array([d['perim'] for d in loop_data.values()])
            V_arr = np.array([d['V'] for d in loop_data.values()])
            
            mask = np.isfinite(V_arr)
            if np.sum(mask) >= 3:
                Am, Pm, Vm = A_arr[mask], P_arr[mask], V_arr[mask]
                
                # Area-only fit
                Xa = np.column_stack([Am, np.ones_like(Am)])
                ca, _, _, _ = np.linalg.lstsq(Xa, Vm, rcond=None)
                R2a = 1 - np.sum((Vm - Xa@ca)**2) / np.sum((Vm - np.mean(Vm))**2)
                
                # Perimeter-only fit
                Xp = np.column_stack([Pm, np.ones_like(Pm)])
                cp, _, _, _ = np.linalg.lstsq(Xp, Vm, rcond=None)
                R2p = 1 - np.sum((Vm - Xp@cp)**2) / np.sum((Vm - np.mean(Vm))**2)
                
                # Combined fit
                Xc = np.column_stack([Am, Pm, np.ones_like(Am)])
                cc, _, _, _ = np.linalg.lstsq(Xc, Vm, rcond=None)
                R2c = 1 - np.sum((Vm - Xc@cc)**2) / np.sum((Vm - np.mean(Vm))**2)
                
                print(f"\n  FITS:")
                print(f"    Area:  V = {ca[0]:.4f}*A + {ca[1]:.4f}  (R2={R2a:.6f})")
                print(f"    Perim: V = {cp[0]:.4f}*P + {cp[1]:.4f}  (R2={R2p:.6f})")
                print(f"    Both:  V = {cc[0]:.4f}*A + {cc[1]:.4f}*P + {cc[2]:.4f}  (R2={R2c:.6f})")
                
                if R2a > R2p:
                    print(f"    => AREA LAW WINS (R2_area={R2a:.4f} > R2_perim={R2p:.4f})")
                else:
                    print(f"    => PERIMETER LAW WINS (R2_perim={R2p:.4f} > R2_area={R2a:.4f})")

print("\nDone.")