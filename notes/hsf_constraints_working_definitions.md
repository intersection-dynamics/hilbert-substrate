# HSF Constraints — Working Definitions (Project Info)

This page is a **living glossary** for the four core HSF constraints. Each definition has:
- **Core statement** (what it forbids)
- **Operational test** (how we recognize it in sims)
- **What it tends to select** (the pressure it applies)
- **Failure modes** (how we can fool ourselves)

---

## 1) No‑signaling

### Core statement
Influence cannot propagate everywhere at once. A change localized to one subsystem must affect other subsystems only through an allowed chain of interactions.

### Operational test (in our current tooling)
- In an “endpoint/link” view: left and right endpoint actions on a link should be **independent interfaces**. A strong proxy is **commutativity** of endpoint actions on the link DOF:
  - \(\|[L^a, R^b]\|\) small for all \(a,b\).
- In a lattice/graph view: couplings are restricted to an **interaction graph**; no all‑to‑all instantaneous updates.

### What it tends to select
- **Interface structure**: information moves via boundary degrees of freedom.
- Often yields **commuting subalgebras** when the model has enough DOF.

### Failure modes / caveats
- Commutativity alone is **too weak**: it allows reducible “hide‑in‑a‑corner” solutions where most DOF are inert.

---

## 2) No‑forgetting

### Core statement
The substrate cannot erase information. Evolution preserves distinguishability (reversibility), so histories remain recoverable in principle.

### Operational test
- Dynamics is **unitary / reversible** (or an isometry on the relevant closed system).
- Practically: invariants (spectra, norms) are preserved under allowed evolution.

### What it tends to select
- **Conserved structures** and persistent sectors.
- Promotes “memory” as **correlation structure**, not stored classical records.

### Failure modes / caveats
- Open‑system reductions (partial tracing) can look like forgetting unless we track the full Hilbert space.

---

## 3) No‑refolding

### Core statement
Once structure (matter/excitations) exists, you cannot arbitrarily re‑factor or rewire the substrate without paying a physical cost. Existing excitations constrain dynamically reachable configurations.

### Operational test (candidate forms)
We track this in two nonexclusive ways (we’ll choose as the project stabilizes):
1. **Factorization stability**: once a link behaves like \(V\otimes\bar V\), that tensor split is not freely mutable.
2. **Slack collapse**: large “do‑nothing” commutants are disfavored because they represent refolding freedom.

### What it tends to select
- **Committed interfaces**: links become genuine bidirectional registers rather than arbitrary algebras inside a single space.
- Suppresses gauge‑like “degeneracy solutions” that leave most DOF inert.

### Failure modes / caveats
- If we only enforce no‑signaling, we can accidentally allow massive refolding slack (large commutants) while still having commuting endpoints.

---

## 4) Finite bandwidth

### Core statement
A link cannot carry unlimited simultaneous influence. The interface has limited capacity to mirror/mediate changes between subsystems.

### Operational test (HSF‑native, no ‘bits/sec’ needed)
We treat bandwidth as a **structural bottleneck** on the *operator channels* that can pass through a link in one step.

Concrete proxy options (we can standardize one):
- **Channel rank profile**: build an influence map \(\Phi_{A\to L}\) from a basis of “pokes” on A to induced changes on the link endpoint, then inspect singular values.
  - “Finite bandwidth” ↔ only a limited number of singular directions are significant.
- **Active‑operator budget**: only a limited number of coupling directions (in a fixed operator basis) may be large simultaneously.
- **Spectral spread bound**: constrain \(\|H_{\text{coupling}}\|\) / spectral width so influence can’t be arbitrarily strong across all channels at once.

### What it tends to select
- **Compression/hygiene**: the system learns to represent influence using a limited set of stable interface modes.
- Encourages genuine \(V\otimes\bar V\) links when two independent endpoints must coexist under capacity limits.

### Failure modes / caveats
- If bandwidth isn’t implemented explicitly, optimizers may find degenerate commuting solutions that underuse DOF (large commutants).

---

# Cross‑constraint notes

## Bidirectional link definition (algebraic)
A “real gauge‑like bidirectional link” means:
- \(\mathcal H_{\text{link}} \cong V\otimes\bar V\) as independent tensor factors
- endpoint actions act as \(\rho(g)\otimes I\) and \(I\otimes\bar\rho(g)\) (automatically commuting)

A sharp witness for \(d=N^2\):
- \(\dim\,\mathrm{Comm}(L)=N^2\)
- \(\dim\,\mathrm{Comm}(R)=N^2\)
- \(\dim\,\mathrm{Comm}(L,R)=1\)

## Singlet‑admitting vertex pairing
For SU(3):
- \(3\otimes\bar 3\) contains a singlet: \(3\otimes\bar 3 = 1\oplus 8\)
- \(3\otimes 3\) does not.
So Gauss‑kernel nonemptiness naturally points to **conjugate pairing** at vertices.

---

# What we’ve learned so far (summary anchors)
- A single qutrit link (\(d=3\)) can support clean su(3) endpoint algebras but cannot host two commuting endpoints (the “48 wall”).
- A composite link (\(d=9=3\otimes 3\)) can host a gauge‑invariant Hamiltonian and, with conjugate pairing, a nonempty Gauss kernel.
- No‑signaling alone (implemented as “make endpoints commute”) does not force \(V\otimes\bar V\) factorization or Gauss‑kernel emergence; it admits reducible high‑commutant solutions.

---

# To decide next
We still need to standardize one operational definition of **finite bandwidth** in the codebase (rank‑profile, active‑operator budget, or spectral bound), because that’s the most likely missing pressure that collapses commutant slack and stabilizes true bidirectional links.
