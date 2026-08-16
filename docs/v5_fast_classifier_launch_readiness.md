# v5 exact-classifier cache and launch readiness

## Verdict: PENDING FULL-A100 RESOURCE GATE — do not submit production

The runtime diagnosis establishes that the exact full-facet classifier, not
geometry construction or signal reduction, causes the production build cost.
The new two-tier cache is locally equivalent to the exact classifier on the
fixed CPU golden fixture.  The Sol GPU golden check, parameter sweep, and
50,000-walker speed gate all passed.  One resource-planning gate remains:
establish the cached group time on a full A100, because the high-rho group is
too slow for the existing 24-hour layout on a 20-GB MIG slice.  No production
or W4 job is authorized by this record.

## Why the cache is safe

The reference classifier asks the KD tree for the nearest seed and then tests
every potentially binding shifted Voronoi facet.  The cache never replaces
that test with an approximation.

Tier 1 stores a reference position, its compartment label, and a conservative
lower bound on the distance to the nearest shifted facet.  A later endpoint is
short-circuited only if its **cumulative displacement from that fixed
reference** is strictly smaller than this bound.  It therefore cannot have
crossed a membrane.  Comparing only one microstep would be unsafe; subtracting
path length would be safe but unnecessarily pessimistic.

Tier 2 handles the cases where Tier 1 cannot fire, especially extracellular
walkers.  It stores every seed inside

``d1(reference) + 2 max(alpha) + 2 delta_max``.

This corrects the initially proposed bound, which used only the reference
seed's annulus.  After a displacement, the nearest seed can change, so the
finite-ensemble `max(alpha)` is required for a proof that the cache contains
every seed in the subsequent exact adaptive-radius query.  A cache buffer
overflow disables Tier 2 and falls through to full exact classification; it
never truncates the set.

The cached implementation leaves the geometry, SI Eq. S2 contraction rule,
and the endpoint-only membrane model unchanged.  Its validation is therefore
an equivalence test against the certified current classifier, not a second
derivation of the P0-A geometry validation.

## Local deterministic equivalence result

The committed CPU/GPU golden fixture contains 64 walkers × 4,000 microsteps
(0.256 million endpoint requests).  This was a fixed deterministic replay,
not a CPU performance run or a production-like simulation.

| Check | Result |
|---|---:|
| Cached Y(t) bit-identical to reference | PASS |
| Cached Y(t) bit-identical to committed golden | PASS |
| Occupancy trace and fatal-escape state | PASS |
| Classifier-request accounting | 256,064 / 256,064 |
| Forced label and accept/reject checks | 32,000 / 32,000 PASS |
| Tier-2 candidate-superset checks | 29,767 PASS |
| Deliberate near-shifted-facet Tier-1 refusals | 8 / 8 |
| Candidate-buffer overflows in fixture | 0 |

The GPU golden gate repeats this fixture on Sol and compares it to the
established CPU output.  The speed gate measures the exact and cached normal
walk-plus-reduction path at the production grid's low, centre, and high rho
points, recording cache hits separately for intracellular and extracellular
endpoints.

## Completed Sol GPU checks

### GPU golden replay: job 61531014

The cached GPU kernel passed the established 64-walker, 4,000-step golden
fixture.  It made 256,064 classifier requests: 3,923 full classifications,
14,330 Tier-1 hits, and 237,811 Tier-2 hits; these sum exactly to the request
count.  There were zero cache-capacity overflows and zero fatal escapes.

The GPU signal agreed with the stored CPU golden signal to an absolute maximum
of `5.68e-14` (relative maximum `1.18e-15`), far inside the predeclared
`5e-12` absolute and `5e-11` relative tolerances.  Occupancy and escape state
were exactly equal.  CPU-versus-GPU reductions are not bit-identical because
their floating-point reduction order differs; that is a baseline GPU golden
property, not a cache discrepancy.  The direct cached-versus-exact GPU
comparison below is bit-identical.

The log contains only Numba low-occupancy warnings for the deliberately tiny
golden/reduction grids.  It contains no Python, CUDA, escape, geometry, or
classifier error.  This fixture uses its declared non-certified test geometry,
not the production geometry reference, as appropriate for an equivalence test.

### Cache-parameter sweep: array job 61530730

Each task used one production-grid geometry, one 5,000-walker ensemble, the
full 31,125-column production storage grid, and `k_io=20 s^-1`.  Geometry
build time and CUDA JIT warm-up were excluded from the timing comparison;
every cached run reused the exact run's geometry and walk seed.  At all three
densities, every tested setting was bit-identical to the exact GPU run in
cosine sum, sine sum, occupancy counts, and fatal-escape count.

| `delta_max` (um), minimum Tier-1 radius (um) | Low rho speedup | Centre speedup | High rho speedup |
|---|---:|---:|---:|
| 0.5, 0 | 2.58x | 2.95x | 2.71x |
| 1.0, 0 | 5.66x | 6.15x | 5.51x |
| **2.0, 0** | **10.09x** | **10.93x** | **6.05x** |
| 1.0, 0.25 | 5.64x | 6.17x | 5.51x |
| 1.0, 0.5 | 5.65x | 6.15x | 5.47x |

`delta_max=2.0 um`, zero minimum Tier-1 radius, and a 256-seed cache is the
unambiguous candidate for the full-size speed gate: it is fastest at every
density and clears the predeclared 3x threshold even at high rho.  The tested
minimum-radius guard did not improve the 1.0-um timing (differences were at
most 0.3%) and merely moved otherwise safe Tier-1 work into Tier 2.  Zero is
therefore the simpler measured choice.  This is parameter selection, not yet
a production selection: the full-size gate remains mandatory.

For the selected 2.0-um setting, Tier-1 plus Tier-2 avoided a full KD-tree
classification on 99.70%, 99.60%, and 99.69% of endpoints at low, centre,
and high rho, respectively.  Intracellular fast-path rates were 99.75%,
99.59%, and 99.71%; extracellular rates were 99.59%, 99.61%, and 99.67%.
Tier 2 supplied all extracellular hits by construction and became especially
important at high rho (97.3% of all requests); this is why Tier 1 alone would
not be sufficient.  Candidate-buffer overflow was zero for every setting and
all three geometries.  Overflow would fall back to full classification, so it
could affect speed but never correctness; the three-point sweep is not proof
that no production group will overflow.

The low-rho task used a 1g.20gb A100 MIG slice and the centre/high tasks used
2g.20gb slices.  Each reported speedup compares exact and cached runs within
the same task, so it is valid.  Absolute seconds across different density
tasks should not be compared as a pure density effect because the MIG compute
profiles differ.

### Full-size cache gate: array job 61532125

This is the required direct measurement at the production 50,000 walkers per
ensemble, again at low, centre, and high rho with the full 31,125-column grid.
The selected `delta_max=2.0 um`, zero-radius-guard, 256-candidate cache was
bit-identical to the exact GPU path in all four recorded quantities at every
density: cosine sum, sine sum, occupancy counts, and fatal-escape count.
There were zero candidate-buffer overflows.  The benchmark would have aborted
on any fatal escape, so its successful JSON outputs also establish zero
escapes in these six exact/cached walks.

| Density | GPU slice | Exact walk+reduction | Cached walk+reduction | Measured speedup |
|---|---|---:|---:|---:|
| Low rho | A100 MIG 2g.20gb | 94.32 s | 9.26 s | **10.18x** |
| Centre rho | A100 MIG 1g.20gb | 271.56 s | 27.11 s | **10.02x** |
| High rho | A100 MIG 2g.20gb | 271.80 s | 53.95 s | **5.04x** |

This clears the predeclared 3x gate at every density.  The high-rho value is
the governing result; it is lower because Tier 2 must handle nearly every
endpoint there, but it is still a five-fold reduction in the measured normal
walk-plus-reduction cost.  At the selected setting the full-classification
fractions were only 0.302%, 0.399%, and 0.306%, respectively.  Accounting
again closed exactly: full + Tier 1 + Tier 2 equalled all 6,400,050,000
endpoint requests in each run.

The Slurm jobs were also healthy: they completed in 2:01, 5:18, and 6:05,
with 186 MB, 188 MB, and 340 MB maximum RSS, respectively.  Those wall times
include geometry construction, CUDA compilation warm-up, and both the exact
and cached reference runs; they are not projected production group times.
The only stderr content was the expected Numba warning about the benchmark's
one-ensemble reduction launch.  There were no Python, CUDA, classifier,
geometry, or escape errors.

The production dense preset now records this validated exact cache in artifact
metadata: `exact_cached`, `delta_max=2.0 um`, minimum safety radius `0`, and
candidate capacity `256`.  This changes only a provably equivalent route to
the existing SI Eq. S2 classifier; it does not change the geometry, membrane
rule, RNG stream, walk time step, or v5 schema.

## Monte-Carlo allocation

The production allocation is now **40 ensembles × 50,000 walkers × 3 axes =
6,000,000 axis-walks per entry**.  The stencil probe found that in the top 5%
of derivative magnitudes—the columns carrying most Fisher information—the
largest beta values were 4.786e-5 (rho), 9.201e-5 (V), and 2.769e-4 (k_io).
Halving walkers conservatively doubles them; the largest becomes 5.538e-4.
That is an uncorrected Fisher inflation of 1.00055, which is scientifically
negligible for the planned CRLB analysis.

The independent-sampling floor is `1/sqrt(6,000,000) = 4.082e-4`, about 36.7
times below the 0.015 trust floor.  Stored `signal_variance` remains the
authoritative uncertainty estimate; this nominal floor is not a substitute
for it.

## Cost and sharding after the 20-GB MIG speed result

There are 51 k_io values and 40 ensembles per `(rho,V)` group.  Multiplying
the measured cached one-ensemble, one-k_io time by `51 x 40` gives the
following direct first-order task estimates on the actual MIG profiles used
by job 61532125.  Geometry is built once per group and adds only 3--23 seconds
to these values.

| Location | Cached group time on measured MIG slice |
|---|---:|
| Low rho | 5.25 h |
| Centre rho | 15.36 h |
| High rho | **30.57 h** |

The high-rho case exceeds the one-day task limit before any safety margin is
applied.  Therefore the existing `a100.20gb:1` production request is not
acceptable for the one-group-per-task, 369-shard layout.  Resharding alone
would not reduce total compute.  Splitting a group by k_io would duplicate
geometry and add operational complexity; it is not selected while the simpler
full-A100 option has not yet been measured.

The earlier direct exact benchmark ran on a full A100 and was substantially
faster per walker than the MIG measurements (about 3--9x, depending on
density and slice profile).  That is consistent with the smaller MIG compute
partition, but it is not a substitute for a direct cached measurement.  A
short serial full-A100 benchmark is therefore the remaining resource gate.

The 369 retained `(rho,V)` pairs are now ordered by ascending rho and assigned
one per task.  This eliminates the former `rho*V = v_i` proxy, which was
nearly constant inside the mask and did not balance the cost-driving rho
dependence.  The exact production bands are 127 tasks (0–126), 121 tasks
(127–247), and 121 tasks (248–368).  The free-water atom is added to task 0.

The cache speed gate is complete.  Per-band wall limits and the production GRES
will be set from the remaining full-A100 measurement, at roughly twice the
measured worst case for each band.

## Array scheduling and storage checks still required on Sol

Slurm supports a `%N` suffix on `--array` to limit the number of simultaneous
array tasks; its maximum valid index is controlled by `MaxArraySize`.
Schedulers also reserve only a bounded number of future array tasks during
backfill, so a 369-task array is normal but should not be submitted by issuing
hundreds of separate `sbatch` calls.  See the [Slurm array
documentation](https://slurm.schedmd.com/job_array.html), the
[`sbatch --array` documentation](https://slurm.schedmd.com/sbatch.html), and
[ASU's array examples](https://docs.rc.asu.edu/slurm-job-array-examples/).

The recommended policy is three submissions—high rho first, then centre, then
low—with a moderate array throttle such as `%24`, subject to the actual public
QOS limits.  This changes queue behaviour only, not total GPU-hours.  A
throttle must not be chosen blindly: the following Sol checks are a launch
gate.

```bash
scontrol show config | grep -iE 'MaxArraySize|MaxJobCount'
sacctmgr show qos public format=Name,MaxSubmitJobsPU,MaxJobsPU,MaxTRESPU
myquota
```

The raw v5 matrix floor remains 9.29 GiB regardless of walker count.  Scratch
must accommodate all 369 shards plus the merged artifact and simultaneous
merge inputs/outputs; no sufficient quota figure has yet been recorded.

## Remaining GPU-only resource gate

After the small follow-up script is committed and pulled to Sol, run these
exact commands from the established checkout:

```bash
cd /scratch/tksimmo2/madi/custom_madi
git status --short
git pull --ff-only
git log -1 --oneline
sha256sum -c data/geometry_reference_si_kappa_0p9.npz.sha256
sbatch scripts/benchmark_exact_cache_full_a100.sbatch
```

This is a serial, three-task, one-hour `htc` array: one exact-plus-cached
50,000-walker ensemble at each density.  It requests one full A100 rather
than a MIG slice, one CPU, and Sol's required 24-GiB host-memory minimum.
The one-CPU request is supported by job 61532125, which consumed only one
core across all tasks.  Serial submission (`%1`) caps this measurement at one
GPU at a time.  It creates validation JSON only and does not write a library.
The production GRES, storage headroom, per-band wall limits, exact production
commands, and GO/NO-GO decision follow only after its outputs are reviewed.
