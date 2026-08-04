# MADI I SI interim remediation update

Date: 2026-08-03.  Source read in full:
[nbm4781-sup-0001](papers/nbm4781-sup-0001-supporting%20information.docx).

This update supersedes the earlier remediation direction only where the SI is
explicit.  P1 remains paused.  The initially discovered S7a/S13--S14 conflict
is resolved: production follows the full-facet SI Eq. S2 / §S.II geometry,
rather than the inconsistent two-nearest §S.IV.a shortcut.

## Item-by-item status

1. **Withdrawn realized-ensemble `<A/V>` calibration — removed.**
   `madi/ensemble.py` now requests `⟨A/V⟩` from a separate governing-process
   reference table.  The table loader rejects missing or uncertified tables;
   it does not substitute a finite `Omega_sim` statistic.  The new
   `scripts/build_si_geometry_reference.py` makes the SI’s untrimmed,
   5-million-single-cell / 26-alpha artifact in reproducible CPU shards.
   The completed artifact is
   `data/geometry_reference_si_kappa_0p9.npz` (SHA-256
   `6a2957eb3d6f89fdebc61d83e7cabb65c67812415879e00b5c07614680785e47`):
   eight contiguous 625,000-cell shards, `kappa=0.9`, full-facet metadata,
   and 26 monotone rows from `v_i=0.40` to 1.00.  The strict production loader
   accepts it.

2. **Withdrawn residence-time/direct `k_io` labeling — removed.**
   Tagged start-cell tracking, first-exit rates, survival fits, direct-
   calibration response curves, `calibrate_exchange`, and
   `kio_measured_se` library storage are gone.  New labels use analytic
   MADI I Eq. 5 with the SI process `⟨A/V⟩`; `p_p` outside `[0,1]` now raises.
   The former Tier-B direct-rate analysis script and tests were deleted.

3. **`kappa=0.9` — changed.**
   The only contraction cap is now
   `min(alpha_star, 0.9*d_nn/2)` in the geometry constructor and the reference
   generator.  The old `0.95` setting is absent from the production path.

4. **S7a inversion — changed.**
   The 25-point alpha lookup/cache and its clamp are gone.  The code solves
   the stated cubic on its physical branch.  At `v_i=0.784`,
   `rho=5e5 cells/uL`, it returns `alpha*=0.50485148 um`, consistent with the
   SI check value (~0.504 um).

5. **Finite source-domain boundary — changed.**
   Eq. S8 now determines `W`; walkers begin in a concentric `0.4W` source
   cube, population seeds are enlarged/certified against S10--S12, and an
   exit is fatal.  Periodic wrapping and every successful-return
   frozen/drop/survivor-column path were removed.  (The kernel may stop an
   escaped walker only long enough to report the fatal error.)  At 128 ms the
   S8 upper bound is `W=554.25626
   um` and the source edge is `221.70250 um`.  At the planned
   `rho=1e7 cells/uL` edge, `W=430.88694 um`; a fixed-seed 512-walker
   free-water stress run had zero escapes in that smaller source domain.

6. **Exact classifier — changed again after resolving the SI conflict.**
   The 1-um voxel-centre candidate cache and the two-nearest classifier are
   gone from production.  CPU and CUDA identify the nearest seed and test all
   potentially binding shifted facets inside the exact `d1+2*alpha1` radius
   bound.  The CPU golden fixture was regenerated for this full-facet case;
   the Sol GPU result must be rerun before a pilot build.

7. **Endpoint membrane count and Gaussian proposal — confirmed.**
   The code already uses per-axis variance `2D0*t_s`, endpoint-only
   `m={0,1,2}`, acceptance `u<p_p**m`, and rejection by literal reversion.
   The operational step length remains RMS 134 nm.

8. **Finite-lobe encoding — unchanged.**
   The SI gives no phase model.  The finite rectangular-lobe integral remains
   the production model and narrow pulse remains diagnostic-only.  The
   microsecond-integral and free-water analytic tests pass.

## Resolved P0-A: SI Eq. S2 versus the SI §S.IV.a shortcut

SI Eq. S2 and §S.II are authoritative because they define the geometric object
used by S7a and the full-facet convex-hull `⟨A/V⟩` reference.  The §S.IV.a
two-nearest rule is a shortcut, not an equivalent evaluation: neighbour order
by distance is not order by normalized shifted-facet margin.  It can only
overestimate intracellular volume.

The replacement queries all seeds within `d1+2*alpha1`, evaluates the exact
margin for each, and requires every one to pass.  The bound is exact: a facet
can bind only if its seed distance satisfies `dj < d1+2 alpha1`.  The certified
50,000-point, 128-ms Eq. S8 grid passed for all nine `v_i` targets at three
densities.  At `rho=5e5`, the two-nearest excess `v_i` is 0.00333 at target
0.80, 0.00190 at 0.85, and 0.00113 at 0.90; it is 0.02870 at 0.40 and decays
to 0.00001 at 0.99.  Brute-force all-seed checks found zero radius-bound
disagreements.

The geometry reference builder now explicitly records its full Eq. S2
contraction rule, and the runtime rejects any reference without that metadata.
The constructor retains its measured-`v_i` build gate, but it now validates the
same geometry that the random walkers inhabit.

## Deleted or superseded machinery

- `analysis/p0b_kio_calibration_table.py`
- `tests/physics_audit/test_p0b_calibration_study.py`
- legacy periodic/cache/direct-exchange portions of the geometry, walker,
  library schema, Sol pilot arguments, and golden harness
- `scripts/run_simulation.py`, which used removed pre-SI APIs and stale
  periodic/cache configuration

The repository-wide stale-document audit requested for P2 has not begun;
for example, the currently dirty README still references the deleted legacy
simulation script.  That is deliberately deferred rather than editing the
user’s unrelated in-progress documentation changes during this SI stop point.

## Tests run

Tier A:

```text
python -m scripts.validate_full_facet_geometry \
  --geometry-reference data/geometry_reference_si_kappa_0p9.npz \
  --output logs/p0a_full_facet_certified_validation.json \
  --volume-samples 50000 --discrepancy-samples 100000 --brute-force-samples 64

All 27 packing rows passed; all 9 adaptive-radius versus all-seed checks had
zero disagreements.  The 5-million-cell artifact passed eight-shard coverage,
metadata, monotonicity, and strict-loader checks.
```

The regenerated CPU golden replay and the existing P0/reference-path checks
also pass locally.  Do not submit the pilot or production build until the
five-million-cell reference exists and the Sol GPU golden job passes.
