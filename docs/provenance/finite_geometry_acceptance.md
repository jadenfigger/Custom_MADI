# Finite Ω_sim geometry acceptance

## Incident

Production array job `61617744` completed every intermediate- and high-density
shard, but 30 low-density shards failed before walking.  The representative
exception was:

```text
target=0.415266, measured=0.409555, allowed=0.005000
```

The failure occurs in `create_ensemble`, after the SI Eq. S7a value of
`alpha_star` and the complete SI Eq. S2 classifier have already been chosen.
It is not a CUDA, Mamba, Slurm, boundary-escape, or Eq. 5 calibration error.

## Cause

The old gate used

```text
max(0.005, 4 × sqrt(v_i (1 - v_i) / N_points))
```

with `N_points = 200,000`.  That standard error assumes the volume-probe
points are independent Bernoulli observations.  They are not: a finite
low-density Ω_sim contains relatively few, large, contracted Poisson--Voronoi
cells, so nearby probe points share the same cell geometry.  The pointwise
SE consequently understates the finite-realisation packing variation.  The
fixed 0.005 threshold then rejected otherwise valid finite SI geometries;
the observed target--measurement difference was 0.005711.

The original 9-by-3 geometry test did not expose this because it used only
20,000 points, for which the old four-point-SE allowance was already larger
than 0.005.  Production's 200,000-point probe reduced only the conditional
sampling SE, revealing the incorrect independence assumption.

## Correction

The simulator still:

- derives `alpha_star` directly from SI Eq. S7a;
- uses all shifted Voronoi facets under SI Eq. S2;
- retains the SI Eq. S8 domain and its S10--S12 population certificate; and
- uses the certified untrimmed process `<A/V>` for Eq. 5.

No finite geometry is used to recalibrate any of those quantities.

The acceptance gate uses the largest of the absolute 0.005 tolerance, four
times the existing pointwise binomial SE, and **five** times the SE of
equal-volume spatial batch means (8 batches along each axis by default). The
five-SE spatial term is a familywise guard for the 369 x 40 finite-geometry
production ensemble set: it avoids rejecting the expected rare 4--5 SE tail
while remaining stricter than a one-geometry uncertainty allowance.
The latter captures the missing finite-domain component.  Each ensemble's
metadata records `realised_vi_spatial_se` and the number of batches per axis;
the library metadata records the full acceptance rule.  This preserves a
strict rejection path for a genuine geometry mismatch while no longer
mistaking correlated samples from a valid low-density realization for
independent trials.

## Restart scope

Only the following failed tasks need resubmission:

```text
0-2,6-7,11-13,17-20,23-25,29-31,35-36,41-45,47,53,59,81-82
```

They are all in the original low-rho band and retain their original seed,
grid, shard IDs, output names, GPU request, and six-hour wall-time limit.
