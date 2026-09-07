# P0 remediation pilot validation

Tier C build `60223180`; Tier A artifact validation: 2026-08-03.

## Verdict

**PASS — structural and provenance pilot.** This clears the P0 pilot gate for
discussion of a full rebuild. It does **not** validate production Monte Carlo
precision and must not be used for fitting data.

## Returned artifact

| Shard | SHA-256 | Entries |
|---|---|---:|
| `000` | `1d13cb23b244e6344084648f18a01a1190a8550fe3e1d16b129dae2151e93302` | 7 |
| `001` | `11782e3dbb82aaf5b6814b65d7ebddae8565fe0e59d02dfbd9e8a0ea9c2b8b1f` | 6 |
| `002` | `3657ac7574d6c30de233a4829b224c1e78112a73b0fd5f186c69c92f3876aac0` | 6 |
| `003` | `5f1cfa82842b67abf64598bf5a3fccb559dac139ec922d0f19e7a99c4aceb13a` | 6 |

The combined artifact contains 25 entries: 24 cellular entries (eight masked
log-grid `(rho,V)` coordinates times `k_io={0,20,130} s^-1`) and one distinct
free-water atom.

## Acceptance results

- All four expected shard IDs are present, with `7,6,6,6` entries.
- The full 25-entry specification is covered without holes or duplicate
  nominal labels.
- Every vector has the required four `(delta,Delta)` pairs — `(7,25)`,
  `(7,50)`, `(20,25)`, `(20,50)` ms — and six b values
  `0,500,1000,2000,4000,6000 s/mm2`; every stored `S(b=0)` is exactly one.
- The metadata declares `finite_lobe`, `T_max=128 ms`, `kappa=0.9`,
  `si_fatal_escape`, untrimmed certified SI `A/V`, the exact SI Eq. S2
  full-facet classifier, and common random numbers across `(rho,V,k_io)`.
- All entry weights are finite and positive. The free-water atom and the
  eight cellular `k_io=0` entries are present.
- Every cellular entry has zero escapes, valid `p_p` (`0..0.03909`), and an
  analytic Eq. 5 label equal to its nominal `k_io`.
- The maximum direct-volume realization error is `0.0022866` in `v_i`, below
  the recorded `0.005` tolerance; labels and realized-geometry metadata agree.
- There are no non-finite signals and no exact cross-label signal ties. The
  minimum pairwise L2 distance is `0.0669528`.

## Validator correction

The first validation invocation falsely reported each shard's nominal
coordinates as mismatched. The builder assigns `(rho,V)` groups round-robin
after sorting by the documented `rho*V` cost proxy; the validator had used
lexicographic tuple ordering. The combined coverage was always correct. The
validator now uses the same cost-proxy ordering and explicitly verifies each
realized `v_i` against the build's stored tolerance. Revalidation of the
unchanged four files passes.

## Pilot-only Monte Carlo limitation

This pilot intentionally used one 512-walker ensemble, or 1,536 axis-walks
per entry, solely to exercise the storage path. It contains 50 negative tail
samples (minimum `-0.03527`), all below the `0.015` signal trust floor, and
seven adjacent-b increases where both samples exceed that floor. The
free-water vector differs from `exp(-b D0)` by at most `0.0451` at this small
sample size. These are expected low-sample Monte Carlo effects, not evidence
of a geometry, exchange-label, or CPU/GPU discrepancy. They reinforce that
P2-L's trust-floor enforcement must be in place for any fitting library.

The full dense production configuration is much larger: 369 retained
`(rho,V)` pairs, 18,819 cellular entries plus free water, 31,125 columns per
entry, and 12,000,000 axis-walks per entry. Pilot wall times (150--224 s per
array task) are not a reliable linear estimate for that configuration because
both the walker count and stored-column count differ by orders of magnitude.
No production submission is authorized by this pilot record alone.
