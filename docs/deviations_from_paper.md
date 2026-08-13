# Physics deviations and source clarifications

This is the authoritative record for differences between this implementation
and Springer *et al.*  It is intentionally separate from the historical
`physics_fidelity_audit.md`, whose pre-SI findings about realized-ensemble
`<A/V>` and residence-time `k_io` labels are superseded by the MADI I
supporting information.

Primary sources:

- MADI I [Supporting Information](https://doi.org/10.1002/nbm.4781), especially §§S.I--S.IV and Eqs. S7a--S14.
- Springer *et al.*, [MADI I](https://doi.org/10.1002/nbm.4781), *NMR in Biomedicine* 36:e4781 (2023), especially §§2--3 and Eqs. 3--5.
- Springer *et al.*, [MADI II](https://doi.org/10.1002/nbm.4782), *NMR in Biomedicine* 36:e4782 (2023), especially §4.4.1.

## Finite rectangular gradient lobes

- **Papers do or appear to do:** MADI I describes endpoint-displacement
  encoding in its simulation discussion.  The SI describes geometry and the
  random walk but contains no phase-accumulation section; its numbering jumps
  from S.IV to S.VI.
- **Code does:** `madi/signal.py` and `madi/walker_gpu.py` use
  `Y(δ) + Y(Δ) - Y(Δ+δ)`, where `Y(t)=∫x(t)dt`, to evaluate the exact phase
  integral for rectangular lobes.  `narrow_pulse` is diagnostic-only.
- **Why:** `φ=γ∫G(t)·r(t)dt` is the acquisition physics.  The endpoint form
  is only a narrow-pulse approximation in restricted or exchanging tissue.
- **Quantified consequence:** the prior finite-lobe audit measured a signal
  difference up to 12.5% at `b=1000 s/mm²` in restricted tissue.  The
  free-water analytic test remains the decisive unit check; it passes for ten
  timing pairs through `δ/Δ=0.6`.
- **Published MADI values comparable:** no, unless the original phase model
  is independently established and matched.
- **Status:** deliberate improvement; source-paper ambiguity remains.

## Dense masked log-coordinate grid

- **Papers do or appear to do:** MADI II used discrete `v_i` hyperbolae and
  calls that restriction a current deficiency (§4.4.1).
- **Code does:** `madi/library.py` defines a dense masked grid uniform in
  `log(rho)` and `log(V)`, with analytic quadrature weights and a free-water
  atom.
- **Why:** topology is a numerical estimator choice, not forward physics;
  the weights prevent Bayesian grid-density bias.
- **Quantified consequence:** the actual spacing is stored with each build;
  no remediated production artifact exists yet.
- **Published MADI values comparable:** not entry-for-entry.
- **Status:** deliberate improvement.

## Walk duration and step proposal

- **Papers do or appear to do:** MADI I uses a 1-µs, 3-D Gaussian proposal
  with component variance `2D0 t_s` (§S.IV.b.1) and an 80-ms walk for its
  fixed timing.
- **Code does:** `SimConfig` retains that Gaussian proposal and uses an
  operational RMS step `sqrt(6 D0 t_s)=134 nm`; the universal timing library
  uses 128 ms to cover its stored windows.
- **Why:** a 128-ms walk is needed when `Δ+δ` reaches 110 ms.
- **Quantified consequence:** the SI Eq. S8 domain becomes 554.256 µm at
  `D0=3 µm²/ms`, `t_RW=128 ms` when its upper walk-length bound binds.
- **Published MADI values comparable:** signal estimates can differ if the
  longer window is used; shared timing columns remain comparable only after
  phase-model resolution.
- **Status:** deliberate extension, with SI-consistent step law.

## Eq. 5 k_io calibration

- **Papers do or appear to do:** SI §S.IV assigns `⟨A/V⟩` to Eq. 5,
  `⟨V/A⟩⁻¹` to inverse mean intracellular lifetime, and `⟨A⟩/⟨V⟩` to a third,
  distinct quantity.  It specifies an untrimmed mean across five million
  independent single-cell, `rho=1`, `kappa=0.9` ensembles and 26 linearly
  spaced `alpha*` values.
- **Code does:** direct tagged-start-cell/residence-time calibration and its
  `kio_measured_se` storage have been removed.  `madi/walker_gpu.py` uses
  Eq. 5 only; `scripts/build_si_geometry_reference.py` generates the required
  untrimmed process reference table in shards.
- **Why:** a residence-time measurement estimates the wrong ensemble average,
  not `⟨k_io⟩`.
- **Quantified consequence:** the old 5--95% trim is absent.  The completed
  five-million-cell / 26-alpha artifact is
  `data/geometry_reference_si_kappa_0p9.npz` (SHA-256
  `6a2957eb3d6f89fdebc61d83e7cabb65c67812415879e00b5c07614680785e47`),
  composed of eight contiguous 625,000-cell shards.  Its normalized
  `⟨A/V⟩` ranges from 9.59314 at `v_i=0.40` to 6.30045 at `v_i=1.00`, with
  relative Monte-Carlo SE below 0.016%.  After the required `rho^(1/3)`
  scaling, this is 0.76141 and 0.50229 `um^-1`, respectively, at
  `rho=5e5 cells/uL`.  The SI Fig. S6b 0.75-pL endpoint corresponds to
  `v_i=0.375`, just outside the production table; a linear diagnostic
  extrapolation gives about 0.783 `um^-1` and is not used by production.
- **Published MADI values comparable:** yes in definition when the same SI
  reference geometry and step convention are used.
- **Status:** SI-specified method; the reference artifact is complete.  A
  pilot remains blocked only on the Sol CPU/GPU full-facet golden check.

## S7a geometry and full-facet classification

- **Papers do or appear to do:** SI Eq. S2 and §S.II define a contracted cell
  as the conjunction of *all* shifted Voronoi half-spaces.  S7a and the SI
  `⟨A/V⟩` cloud are measured from full convex hulls of those cells.  SI §S.IV.a
  instead evaluates S13--S14 for only the second-nearest seed.
- **Code does:** `madi/ensemble.py` inverts S7a directly, applies
  `alpha_i=min(alpha*,0.9 d_nn/2)`, and tests every potentially binding facet
  at every endpoint.  For nearest seed `c_1`, the CPU and CUDA paths first
  obtain `d_1`, then query every seed within `d_1+2 alpha_1`, calculate
  `(d_j²-d_1²)/(2|c_1-c_j|)`, and label intracellular only when every margin
  is at least `alpha_1`.  This is an exact adaptive-radius search, not a
  fixed-neighbour count or a cache.  The reference generator stores
  `contraction_rule=all_shifted_voronoi_facets_S2` and the runtime loader
  rejects a table without it.
- **Why:** neighbour order by `d_j` does not order the normalized facet margin,
  because the denominator is `|c_1-c_j|`.  The second-nearest rule can only
  omit a binding facet and therefore moves points from gap into cell.  If
  `m_j < alpha_1`, triangle inequality gives
  `d_j²-2 alpha_1 d_j-d_1²-2 alpha_1 d_1 < 0`, whose positive root is
  `d_j=d_1+2 alpha_1`; no seed outside that radius can bind.  Full facets are
  required so the walker geometry is the same object that supplied S7a and
  the Eq. 5 `⟨A/V⟩` calibration.
- **Quantified consequence:** Tier-A, 50,000-point measurements in the
  128-ms Eq. S8 domain passed at all nine targets `v_i=0.40…0.99` and at
  `rho=1e5, 5e5, 1e6 cells/uL`; all errors were within `4 SE` (and the
  0.005 absolute finite-realisation allowance).  At `rho=5e5`, a 100,000-point
  two-nearest versus full-facet curve gives excess `v_i` of 0.02870 at 0.40,
  0.00333 at 0.80, 0.00190 at 0.85, 0.00113 at 0.90, and 0.00001 at 0.99;
  those are disagreement rates of 2.870%, 0.333%, 0.190%, 0.113%, and
  0.001%, respectively.  The radius classifier had zero disagreements with
  brute-force all-seed Eq. S2 evaluations (64 independent points at each
  target).  See `scripts/validate_full_facet_geometry.py`.
- **Published MADI values comparable:** yes to the Eq. S2/S7a geometry; no to
  an implementation that literally uses only the §S.IV.a two-nearest shortcut.
- **Status:** resolved deliberate correction of an internal source-document
  inconsistency.  The Sol CPU/GPU golden check remains the required hardware
  transliteration confirmation before a pilot build.

## Finite simulation boundary

- **Papers do or appear to do:** SI §S.III Eqs. S8--S12 uses a finite
  `Omega_sim`, a concentric `0.4W` source cube, an encompassing populated
  domain, and treats any walker escape as an error.
- **Code does:** `madi/ensemble.py` uses Eq. S8 and numerically certifies the
  S10--S12 population conditions; `madi/walker_gpu.py` raises on any escape.
  Periodic wrapping and all per-column survivor bookkeeping were removed.
- **Why:** escape-conditioned signal averaging changes the ensemble.
- **Quantified consequence:** at 128 ms, Eq. S8 gives `W=554.256 µm` where
  its upper walk-length bound applies and `W=430.887 µm` at the planned
  `rho=1e7 cells/uL` edge.  Fixed-seed 512-walker free-water checks in both
  source cubes produced zero escapes; a forced exit raises rather than
  returning a selected subset.  The previous 200-µm diagnostic box produced
  two escapes among 512 walkers.
- **Published MADI values comparable:** yes in boundary intent, subject to
  the independently unresolved phase-model difference above.
- **Status:** SI-specified correction.

## Common random numbers and optional fitting modes

- **Papers do or appear to do:** no common-random-number library strategy or
  Bayesian/NNLS/Rician variants are specified.
- **Code does:** neighboring entries share random streams; Bayesian, NNLS,
  Rician, and free-S0 modes are opt-in.  The fixed-S0 linear exhaustive MAP
  reference matcher remains isolated.
- **Why:** shared streams smooth Monte-Carlo residual surfaces; optional
  fitters do not alter the reference route.
- **Quantified consequence:** entries are correlated, so `1/sqrt(N)` is not
  a valid per-entry uncertainty estimate; P2-M will use held-out ensembles.
- **Published MADI values comparable:** reference MAP only.
- **Status:** deliberate additions.

## Retained snapping and direction limitation

- **Papers do or appear to do:** MADI II used direction averaging; MADI III
  reports material one-direction versus three-direction parameter shifts.
- **Code does:** snapping stays at 30 `s/mm²` and 1.5 ms but is recorded,
  while fit configuration declares a direction scheme and records b-tensor
  checks.
- **Why:** visible acquisition provenance is required to interpret maps.
- **Quantified consequence:** MADI III reports, for one versus three
  directions, `v_i` −8.4% and `k_ioV` +16.9%, among other shifts.
- **Published MADI values comparable:** only with matching direction handling.
- **Status:** deliberate provenance improvement and inherited limitation.

## Source-document inconsistencies (not implementation targets)

- MADI I Figure 1’s `rho`, `V`, and `v_i` caption values do not satisfy the
  unit identity simultaneously.
- The MADI I/SI printed Gaussian-step mean and median expressions are not the
  3-D Gaussian values; Eq. 3 operationally fixes the RMS step at 134 nm.
- SI §S.IV.a's two-nearest S13--S14 shortcut is inconsistent with the all-facet
  contracted-cell definition in SI Eq. S2 and §S.II.  At `rho=5e5` it raises
  measured `v_i` by 0.00333 at target 0.80, 0.00190 at 0.85, and 0.00113 at
  0.90 (100,000 samples); the effect reaches 0.02870 at target 0.40.  The
  discrepancy is one-sided because omitted facets can only turn gap into cell.
- MADI II Table 1’s NA-prostate row lies below its stated library `v_i` floor.
- MADI III Table 1 combines medians, so its median `rho` and `V` do not imply
  its listed median `v_i`.
- MADI I and II disagree on pmol versus fmol scale for the same NKA example.
- MADI III prints an ADC expression with multiplication by `b` where units
  require division.
