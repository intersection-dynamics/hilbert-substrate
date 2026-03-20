# HSF Observable Dictionary

## Purpose

This note defines the current observables used in the HSF mesoscape program in behavioral terms. It does **not** assume that these quantities already correspond to familiar particle-physics observables. Instead, each observable is treated as a diagnostic of a difference in regime, organization, or persistence behavior within the evolving HSF substrate.

The guiding idea is simple:

**An observable is useful when it separates one kind of system behavior from another.**

In the current HSF program, the main distinctions we care about are:

- churn vs accretion
- diffuse activity vs organized support
- transient hotspot vs persistent organizer
- line-like corridor vs hub vs triangulated shared-edge complex
- static cluster vs cooperative object with changing members

This note is meant to give each observable a clear working meaning so the code outputs are interpretable and cumulative rather than just a pile of numbers.

---

## Background framing

HSF begins from a globally unitary substrate and studies how persistent structure appears under operational constraints. In that framework, the important question is not “what classical object is hidden in the state?” but:

- which subsystems matter disproportionately,
- which structures persist,
- and what kinds of support organization emerge over time.

In Paper I, persistence was framed primarily in terms of **stable identity and ordering** of dominant pathways, not frozen magnitudes. That is the spirit used here as well.

---

## 1. Growth and organization observables

These observables tell us whether the substrate is actively generating and retaining relational structure.

### Birth count
**Computed from:** number of subsystem birth events over the run.

**Detects:** whether the substrate is dynamically producing new effective structure.

**Interpretation:**  
A high birth count means the system is productive, but by itself it does not distinguish meaningful growth from aimless boiling.

---

### Persistent births
**Computed from:** births that survive through the settling/lookahead windows.

**Detects:** whether newly formed structure remains in the graph long enough to count as durable organization.

**Interpretation:**  
High persistent-birth count suggests accretion of mesoscopic structure rather than transient fluctuation.

---

### Remerge-prone births
**Computed from:** births that disappear or reintegrate after short timescale evaluation.

**Detects:** how much of the substrate activity is temporary churn.

**Interpretation:**  
If most births are remerge-prone, the substrate is boiling without building much durable organization.

---

### Active nodes
**Computed from:** number of nodes currently participating in the active graph.

**Detects:** how much of the substrate has been recruited into the evolving structure.

**Interpretation:**  
More active nodes can mean growth, but also can simply mean spread. This must be interpreted together with morphology and continuity observables.

---

### Active edges
**Computed from:** number of active edges in the graph.

**Detects:** degree of relational connectivity in the active support.

**Interpretation:**  
Growth in active edges can indicate shell-building or densification. Too many edges relative to corridor indicators can also mean the object is becoming mesh-like rather than pathway-like.

---

### Metric extent
**Computed from:** spread of the active graph in a spectral-embedding or graph-induced metric.

**Detects:** whether relational organization is expanding outward in the induced metric sense.

**Interpretation:**  
Increasing extent suggests exploration or elaboration of relational space.  
Flattening extent with persistent organizer identity suggests consolidation around a chosen support.

**Important caution:**  
This is a morphology diagnostic, not yet a proof of physical distance.

---

### Total edge length
**Computed from:** sum of embedded or graph-derived edge lengths.

**Detects:** geometric elaboration of the active support.

**Interpretation:**  
Useful as a secondary growth marker. Should not by itself be read as evidence of emergent space.

---

## 2. Organizer identity observables

These tell us where the system’s current organizer is and whether it settles into a recognizable dominant structure.

### Dominant core pair
**Computed from:** the pair of nodes with highest mesoscopic importance score in a snapshot.

**Detects:** the local center of organization at that time.

**Interpretation:**  
This answers: “where is the object organizing right now?”

**Important caution:**  
This does not automatically mean “the particle is this pair.” It marks the current organizer center, not necessarily a complete object.

---

### Core switches
**Computed from:** number of times the dominant core pair changes between snapshots.

**Detects:** volatility of organizer identity.

**Interpretation:**  
Many switches indicate boiling, migration, or weak commitment.  
Few switches with a long stable late epoch indicate condensation.

---

### Longest-lived core
**Computed from:** longest contiguous epoch in which the same dominant pair remains dominant.

**Detects:** whether the system develops an attractor-like organizer.

**Interpretation:**  
One of the strongest current objecthood indicators.

---

### Dominant-pair counts
**Computed from:** total number of snapshots in which each pair is dominant.

**Detects:** whether one pair dominates, several recur, or dominance is fragmented.

**Interpretation:**  
A single dominant pair suggests a stable attractor.  
Several neighboring dominant pairs suggest corridor migration or a handoff chain.

---

### Epoch structure
**Computed from:** segmentation of the dominant-core history into time intervals.

**Detects:** formation, migration, stabilization, and replacement phases.

**Interpretation:**  
Often more informative than summary numbers alone.  
A sequence like `[6,8] -> [10,11] -> [11,12] -> [12,13]` suggests migrating condensation rather than random switching.

---

## 3. Morphology observables

These tell us what kind of organizer the system is building.

### Line-likeness
**Computed from:** how closely the support resembles a path or corridor rather than a hub or mesh.

**Detects:** serial/linear organization.

**Interpretation:**  
High line-likeness means the object is corridor-like.  
Low line-likeness means it is not naturally a 1D support.

**Important caution:**  
This is a shape diagnostic, not a direct fermion detector.

---

### Path coverage
**Computed from:** fraction of organizer support captured by the inferred backbone path.

**Detects:** how much of the support lies on a single principal corridor.

**Interpretation:**  
High path coverage suggests serial organization.  
Low path coverage means much of the support lies off-path.

---

### Path edge concentration
**Computed from:** fraction of support edges that belong to the inferred backbone path.

**Detects:** whether relational activity concentrates onto a narrow support.

**Interpretation:**  
A high value indicates corridor concentration.  
A low value suggests branching or broad support.

---

### Branch penalty
**Computed from:** excess branching burden of the support graph.

**Detects:** how far the organizer deviates from serial corridor behavior.

**Interpretation:**  
High branch penalty means the support is acting like a branching conduit or hub, not a thin line.

---

### Triangle density
**Computed from:** density of triangles in the support neighborhood.

**Detects:** prevalence of shared-edge simplicial organization.

**Interpretation:**  
High triangle density means the object is triangulated or plaquette-rich rather than chain-like.

**Why this matters:**  
This may turn out to be one of the most important current observables.  
It distinguishes 1D-string expectations from simplicial/shared-edge transport structure.

---

### Triangle penalty
**Computed from:** burden of loops/triangles relative to an idealized line support.

**Detects:** departure from thin-corridor topology.

**Interpretation:**  
Useful when testing whether a JW-like witness is even appropriate.

---

### Topology class
**Computed from:** rule-based classification of support as corridor, hub, triangulated, mesh, or mixed.

**Detects:** coarse morphology of the organizer.

**Interpretation:**  
Useful summary label for comparison across runs, but should always be backed by the lower-level metrics above.

---

## 4. Continuity and persistence observables

These tell us whether the organizer persists as a pattern through time.

### Path Jaccard
**Computed from:** overlap of inferred backbone path between consecutive snapshots.

**Detects:** continuity of the corridor/backbone support.

**Interpretation:**  
High path Jaccard means the inferred support remains stable through time even if it is not especially line-like.

---

### Support Jaccard
**Computed from:** overlap of total organizer support between consecutive snapshots.

**Detects:** temporal persistence of the broader local relational zone.

**Interpretation:**  
High support overlap means the same region remains organized, even if its local details change.

---

### Core-on-previous-path
**Computed from:** whether the new dominant core lies on or near the previous inferred path.

**Detects:** continuity of migration.

**Interpretation:**  
Distinguishes corridor handoff from arbitrary relocation.

---

### Support persistence
**Computed from:** temporal overlap of shell/core neighborhoods over an epoch.

**Detects:** whether the same support structure survives.

**Interpretation:**  
High support persistence indicates real object continuity.

---

### Member turnover
**Computed from:** change in node membership of the support between snapshots or over an epoch.

**Detects:** whether the object is microscopically static or dynamically renewed.

**Interpretation:**  
This is crucial for HSF. If a pattern persists while its exact members change, then the object is a **cooperative organizer**, not a frozen cluster.

---

## 5. Collective objecthood observables

These ask whether a persistent pattern deserves to be treated as an object in its own right.

### Collective identity score
**Computed from:** combined measure of support persistence, topology coherence, and tolerance to member turnover.

**Detects:** whether the system sustains the same kind of organizer despite internal churn.

**Interpretation:**  
High score means the system supports a cooperative object.  
Low score means there may be stable local structures, but not yet a strong persistent identity.

---

### Strongest epoch
**Computed from:** epoch with highest collective identity score.

**Detects:** when the clearest object-like organization occurs.

**Interpretation:**  
The strongest particle-like phase may be early and compact rather than late and sprawling.  
This is important because the biggest late structure is not always the cleanest object.

---

### Organizer accretion
**Computed from:** shell growth around a stable or semi-stable core/support.

**Detects:** whether the object is growing while preserving identity.

**Interpretation:**  
Distinguishes an accreting object from a simply expanding cloud.

---

## 6. Constraint-witness observables

These connect the measured behavior back to HSF’s underlying constraint logic.

### Rank stability / dominant-identity stability
**Computed from:** persistence of ordered importance ranking over time.

**Detects:** stable pathways of significance.

**Interpretation:**  
Directly aligned with the original HIP idea: what matters is not frozen intensity values but stable identity and ordering of dominant pathways.

---

### Committed-interface concentration
**Computed from:** concentration of activity/influence on a limited support set.

**Detects:** whether the system is selecting a few committed channels rather than spreading activity arbitrarily.

**Interpretation:**  
Can be read as a proxy for no-refolding plus finite-bandwidth effects.

---

### Bandwidth hygiene / thinness proxies
**Computed from:** low effective channel count, narrow support, concentrated active directions.

**Detects:** whether influence is represented through a limited number of active modes.

**Interpretation:**  
Proxy for finite bandwidth.  
Not yet a theorem, but a useful diagnostic.

---

### Memory-as-correlation proxies
**Computed from:** persistence of organizer identity and support across local change.

**Detects:** whether structure survives as relational memory rather than static storage.

**Interpretation:**  
Aligned with the HSF idea that no-forgetting is preservation of history in correlation structure.

---

## 7. Provisional charge-like and label observables

These are exploratory. They do not yet deserve strong physical names.

### Shell-integrated Cartan charges (`Q3`, `Q8`)
**Computed from:** coarse-grained shell/core operator sums in the chosen local basis.

**Detects:** whether metastable neighborhoods carry persistent orientation-like or charge-like signatures.

**Interpretation:**  
These are candidate coarse labels, not yet proven quantum numbers.

---

### Casimir-like norm
**Computed from:** norm of the coarse charge vector.

**Detects:** total magnitude of structured shell/core labeling.

**Interpretation:**  
Potentially useful as a scale or identity marker if it stabilizes over epochs.

---

### Shell entropy
**Computed from:** entropy of shell/complement reduced-state split.

**Detects:** how entangled the organizer neighborhood is with its outside.

**Interpretation:**  
This is not “internal disorder” in the ordinary sense.  
It is a boundary entanglement diagnostic.

---

### Reduced-state spectrum / participation ratio
**Computed from:** eigenvalues of reduced shell/core states.

**Detects:** effective rank and concentration of local mixed-state structure.

**Interpretation:**  
May eventually help distinguish simple coherent modes from diffuse mixed support.

---

## 8. Special case: the JW witness family

These were introduced to test whether the mesoscopic organizer is becoming a linear Jordan–Wigner-like support.

### JW string score
**Computed from:** combined measure favoring path continuity, low branching, low triangulation, and line-like support.

**Detects:** how close the organizer is to a thin serial corridor.

**Interpretation:**  
This is a **linearity witness**, not a general particle witness.

**Important conclusion from current runs:**  
If JW score stays low while support continuity stays high, that means the organizer is real but not naturally 1D.

---

### What the JW family means when it fails
If the support is persistent but has:

- high branch penalty,
- nontrivial triangle density,
- low line-likeness,
- high path continuity,

then the correct reading is **not** “nothing is there.”

The correct reading is:

**the mesoscopic organizer is persistent, but its natural topology is branched/shared-edge rather than linear.**

That is an important result, because it tells us the wrong witness class may have been used.

---

## 9. What these observables do *not* mean yet

This section is critical.

### They do not yet prove particles
A stable core, shell, or collective organizer is not automatically a particle in the standard sense.

### They do not yet prove fermions
A failed JW witness does not mean there are no fermionic-like structures.  
It may only mean that a 1D ordering witness is inappropriate for a triangulated or simplicial support.

### They do not yet prove gravity or geometry
Metric extent and edge length are morphology diagnostics. They are not yet proofs of emergent spacetime geometry.

### They do not yet derive measurement
Reduced states and entropy are part of the decoherence-style machinery, but they do not by themselves solve the measurement problem or derive the Born rule.

---

## 10. Working interpretation for the current mesoscape program

At the moment, the observables support the following cautious picture:

1. The substrate produces real mesoscopic organization, not just random churn.
2. That organization can condense around dominant cores and persistent late-time support.
3. The organizer is often stable in identity but not naturally line-like.
4. Larger runs increasingly suggest branched, shared-edge, triangle-rich support rather than 1D corridor structure.
5. The most promising current “objecthood” indicators are:
   - longest-lived core
   - dominant-pair counts
   - support persistence
   - member turnover under stable support
   - collective identity score
6. The first robust labels may be **structural** rather than **charge-like**.

---

## 11. Practical usage rule

When interpreting any observable, ask:

1. What behavioral difference does it detect?
2. Does it separate two real regimes seen in the runs?
3. Is it measuring organizer identity, morphology, continuity, or constraint pressure?
4. Are we overreading it as a particle label or physical law before that is justified?

If an observable cannot answer those questions clearly, it needs to be revised or removed.

---

## 12. Bottom line

In the HSF mesoscape program, an observable is valuable when it helps answer one of the following:

- Is the system organizing at all?
- Where is the organizer?
- What kind of organizer is it?
- Does that organizer persist through time?
- Does it survive member turnover as a cooperative object?
- Which HSF constraints appear to be selecting it?

That is the current dictionary.

Future versions should expand this note only when a new observable clearly separates a new regime or clarifies an existing ambiguity.