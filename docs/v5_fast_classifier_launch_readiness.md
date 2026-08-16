# v5 exact-classifier cache and launch readiness

## Verdict: PENDING GPU GATE — do not submit production

The runtime diagnosis establishes that the exact full-facet classifier, not
geometry construction or signal reduction, causes the production build cost.
The new two-tier cache is locally equivalent to the exact classifier on the
fixed CPU golden fixture.  It still needs the Sol GPU equivalence and measured
speed gates before it can be selected for production.  No production or W4
job is authorized by this record.

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

The GPU gate repeats this fixture on Sol and compares it to the established
CPU output.  The speed gate then measures the exact and cached normal
walk-plus-reduction path at the production grid's low, centre, and high rho
points, recording cache hits separately for intracellular and extracellular
endpoints.

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

## Cost and sharding before the GPU speed result

The prior exact-classifier benchmark projected 51-k_io, 40-ensemble groups of
26.5 h (low rho), 34.3 h (centre), and 93.4 h (high rho) at 100,000 walkers.
Geometry was only 0.13% of a group.  Halving walkers therefore gives the
following near-linear pre-cache planning values:

| Location | 50k-walker group before cache | At 3× cache speed | At 5× cache speed |
|---|---:|---:|---:|
| Low rho | 13.2 h | 4.4 h | 2.6 h |
| Centre | 17.2 h | 5.7 h | 3.4 h |
| High rho | 46.7 h | 15.6 h | 9.3 h |
| Full build | ~9,750 GPU-h | ~3,250 GPU-h | ~1,950 GPU-h |

The 369 retained `(rho,V)` pairs are now ordered by ascending rho and assigned
one per task.  This eliminates the former `rho*V = v_i` proxy, which was
nearly constant inside the mask and did not balance the cost-driving rho
dependence.  The exact production bands are 127 tasks (0–126), 121 tasks
(127–247), and 121 tasks (248–368).  The free-water atom is added to task 0.

These numbers are planning arithmetic, not a measured result.  The cache
speed must be at least 3× to clear the stated gate.  Per-band wall limits will
be set to roughly twice the measured worst case after that result returns.

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

## Next GPU-only commands

After this change set is committed and pulled to Sol, run these exact commands
from the established checkout:

```bash
cd /scratch/tksimmo2/madi/custom_madi
git status --short
git pull --ff-only
git log -1 --oneline
sha256sum -c data/geometry_reference_si_kappa_0p9.npz.sha256
sbatch scripts/validate_exact_cache_gpu.sbatch
sbatch scripts/benchmark_exact_cache_sweep.sbatch
```

The jobs use a 20-GB A100 MIG slice, two CPUs, and Sol's required 24-GiB host
memory minimum.  They create validation JSON only and do not write a library.
The speed-mode job, exact production commands, final cache constants, storage
headroom, and GO/NO-GO decision follow only after these outputs are reviewed.
