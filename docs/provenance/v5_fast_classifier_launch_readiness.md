# v5 exact-classifier cache and launch readiness

## Verdict: GO — configuration is technically ready; submission still requires user approval

The runtime diagnosis establishes that the exact full-facet classifier, not
geometry construction or signal reduction, causes the production build cost.
The new two-tier cache is locally equivalent to the exact classifier on the
fixed CPU golden fixture.  The Sol GPU golden check, parameter sweep,
50,000-walker speed gate, and full-A100 resource gate all passed.  The launch
uses 369 monotonic-rho shards, a full A100, one CPU, and 24 GiB host memory.
The three independent rho bands have measured twofold-margin limits of 6 h,
11 h, and 19 h.  This record prepares commands only; submission remains the
user's decision, and W4 remains unauthorized.

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
applied.  Therefore the former `a100.20gb:1` production request is not
acceptable for the one-group-per-task, 369-shard layout.  Resharding alone
would not reduce total compute.  Splitting a group by k_io would duplicate
geometry and add operational complexity, so it is not selected.

### Full-A100 resource gate: array job 61533979

Job 61533979 repeated the same selected 50,000-walker exact/cached comparison
on a full NVIDIA A100-SXM4-80GB, one CPU, and 24 GiB host memory.  It passed
the same bit-identity and zero-overflow checks as the MIG jobs.

| Density | Cached one-ensemble, one-kio time | Cached 51-kio x 40-ensemble group time |
|---|---:|---:|
| Low rho | 3.533 s | 2.00 h |
| Centre rho | 5.105 s | 2.89 h |
| High rho | 16.260 s | **9.21 h** |

The associated exact times were 34.825 s, 52.957 s, and 87.590 s, giving
measured cache speedups of 9.86x, 10.37x, and 5.39x.  Each cached result was
bit-identical to exact; full + Tier 1 + Tier 2 equalled all 6,400,050,000
endpoint requests; and candidate-buffer overflow was zero.  The high-rho
group is the governing case: twice 9.21 h is 18.43 h, inside a 24-hour limit.

The production request is therefore a full A100 (`-G a100:1`), one CPU, and
24 GiB host memory.  The larger GPU request is about compute partitions, not
memory: the full-A100 test used only 188--493 MB RSS, but a 20-GB MIG slice
was too slow at high rho.  One CPU is supported by 89--94% CPU efficiency in
the full-A100 test; the host process uses one core while the walk runs on the
GPU.

The 369 retained `(rho,V)` pairs are now ordered by ascending rho and assigned
one per task.  This eliminates the former `rho*V = v_i` proxy, which was
nearly constant inside the mask and did not balance the cost-driving rho
dependence.  The exact production bands are 127 tasks (0–126), 121 tasks
(127–247), and 121 tasks (248–368).  The free-water atom is added to task 0.

The measured per-ensemble cached times were interpolated linearly in log time
between the production-grid anchors (rho indices 0, 21, and 63) and multiplied
by each band's actual masked-pair count.  This is a planning estimate; the
wall limits use twice the predicted band maximum, not the average.

| Rho band / task IDs | Pairs | Predicted maximum group | Chosen limit | Estimated GPU-h in band |
|---|---:|---:|---:|---:|
| indices 0--21 / 0--126 | 127 | 2.89 h | 6 h | 307 |
| indices 22--42 / 127--247 | 121 | 5.16 h | 11 h | 481 |
| indices 43--63 / 248--368 | 121 | 9.21 h | 19 h | 858 |
| **All cellular groups** | **369** | — | — | **~1,646** |

The total is the measured cached walk-and-reduction projection before the
small one-time geometry and archive-I/O costs.  The comparable full-A100
exact-classifier projection is ~12,607 GPU-h, so the measured aggregate
reduction is about 7.7x.  The limits provide modest additional rounding above
the strict twofold margins (5.79 h, 10.33 h, and 18.43 h).

## Array scheduling and storage checks

Slurm supports a `%N` suffix on `--array` to limit the number of simultaneous
array tasks; its maximum valid index is controlled by `MaxArraySize`.
Schedulers also reserve only a bounded number of future array tasks during
backfill, so a 369-task array is normal but should not be submitted by issuing
hundreds of separate `sbatch` calls.  See the [Slurm array
documentation](https://slurm.schedmd.com/job_array.html), the
[`sbatch --array` documentation](https://slurm.schedmd.com/sbatch.html), and
[ASU's array examples](https://docs.rc.asu.edu/slurm-job-array-examples/).

Sol reports `MaxArraySize=50000` and `MaxJobCount=300000`. The public QOS
permits 300,000 submitted/running jobs per user and `cpu=7500`, so all 369
one-CPU tasks are permitted; no GPU-specific per-user cap was reported.

The selected policy is three independent, **unthrottled** submissions—high
rho first, then centre, then low. Omitting `%N` lets Slurm start as many jobs
as fair-share and available full A100s allow, minimizing calendar time without
changing the ~1,646 GPU-hour allocation. A `%N` applies to each array, not to
all three arrays together: the former `%8` commands would have permitted up
to 24 simultaneous tasks, not eight. The largest meaningful per-array values
are `%121`, `%121`, and `%127`; they are equivalent to no throttle because
those are the respective array lengths. The scheduler, rather than an
artificial cap, still determines the actual concurrency.

```text
MaxArraySize            = 50000
MaxJobCount             = 300000
public QOS: MaxSubmitJobsPU=300000, MaxJobsPU=300000, MaxTRESPU=cpu=7500
scratch: 40.59 MiB used of 100.00 TiB
```

The raw v5 matrix floor remains 9.29 GiB regardless of walker count.  Even a
conservative 30 GiB allowance for all shards, merged output, hashes, and
simultaneous merge inputs is negligible relative to ~100 TiB available scratch.
File-count headroom is also ample: 422 of 20,000,000 allowed files are used.

## Exact pre-flight and submission commands, only after explicit approval

First verify the exact committed code and immutable geometry input.  These
commands do not launch a production job:

```bash
cd /scratch/tksimmo2/madi/custom_madi
git status --short
git pull --ff-only
git log -1 --oneline
sha256sum -c data/geometry_reference_si_kappa_0p9.npz.sha256
python -m pytest tests/physics_audit/test_v5_production_configuration.py tests/physics_audit/test_remediation_production_sharding.py -q
```

Only if those pass and the user explicitly elects to spend the allocation,
submit the independent bands in this order.  Do not add a dependency; they are
independent and a dependency would delay short work behind the high-rho band.

```bash
PROD_HIGH_JOB="$(sbatch --partition=public --qos=public --array=248-368 --time=0-19:00:00 --cpus-per-task=1 --mem=24G -G a100:1 scripts/build_lib.sbatch production 369 | awk '{print $4}')"
printf 'High-rho job: %s\n' "${PROD_HIGH_JOB}"
PROD_CENTRE_JOB="$(sbatch --partition=public --qos=public --array=127-247 --time=0-11:00:00 --cpus-per-task=1 --mem=24G -G a100:1 scripts/build_lib.sbatch production 369 | awk '{print $4}')"
printf 'Centre-rho job: %s\n' "${PROD_CENTRE_JOB}"
PROD_LOW_JOB="$(sbatch --partition=public --qos=public --array=0-126 --time=0-06:00:00 --cpus-per-task=1 --mem=24G -G a100:1 scripts/build_lib.sbatch production 369 | awk '{print $4}')"
printf 'Low-rho job: %s\n' "${PROD_LOW_JOB}"
squeue -j "${PROD_HIGH_JOB},${PROD_CENTRE_JOB},${PROD_LOW_JOB}"
```

The production launch must not proceed if the final commit/checksum/test
pre-flight fails.  W4 remains unauthorized regardless of this GO verdict.

## Completion and post-merge commands

After all three arrays finish, check task states and scan every production log
explicitly. Exit status alone is insufficient because the build is required
to fail loudly on fatal escape, geometry-target, and permeation-probability
assertions.

```bash
sacct -j "${PROD_HIGH_JOB},${PROD_CENTRE_JOB},${PROD_LOW_JOB}" --format=JobID,JobName,Elapsed,TotalCPU,ReqMem,MaxRSS,State,AllocTRES
seff "${PROD_HIGH_JOB}_248"
rg -n -i 'fatal.*escape|geometry.*assert|geometry.*target|p_p.*range|traceback|error:' logs/madi_dense_${PROD_HIGH_JOB}_*.out logs/madi_dense_${PROD_HIGH_JOB}_*.err logs/madi_dense_${PROD_CENTRE_JOB}_*.out logs/madi_dense_${PROD_CENTRE_JOB}_*.err logs/madi_dense_${PROD_LOW_JOB}_*.out logs/madi_dense_${PROD_LOW_JOB}_*.err
find libraries -maxdepth 1 -type f -name 'madi_dense_universal_remediated.shard*.npz' | wc -l
```

The scan must print no matches and the count must be exactly 369. Only then
submit the CPU-only post-processing job. It hashes all shards before merge,
uses `--require-shards 369`, runs the merged v5 validator, and refuses to
overwrite the current `madi_dense_universal.npz`.

```bash
PROD_POST_JOB="$(sbatch scripts/postprocess_v5_production.sbatch | awk '{print $4}')"
printf 'Production post-processing job: %s\n' "${PROD_POST_JOB}"
squeue -j "${PROD_POST_JOB}"
```

The post-processing script establishes the exact 369-shard coverage before it
merges. The validator then reports the v5 schema, exact 18,820-entry grid
coverage, diagnostic-array shape/dtype and subset reconstruction,
imaginary-symmetry maximum Student statistic and Bonferroni threshold,
negative-signal count/minimum, and positive-b signal-SE median/IQR. A failed
validator is a stop: retain the new artifact and evidence, but do not replace
the current production library.
