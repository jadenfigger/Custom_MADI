# Fisher information and CRLB analysis: phased plan

Status: planning document. Written while the v5 production library build is
running on Sol. It specifies the next project phase and does not authorize any
build, submission, or code change by itself.

Companion documents: `archive/v5_schema_prebuild_note.md` (the three diagnostic arrays
this analysis consumes), `provenance/v5_stencil_probe.md` (the measured derivative-noise
values this plan relies on), `provenance/v5_crn_diagnostic.md` (why the pilot could not
answer the identifiability question), `fitting_methods.md` (the three
estimators compared in Phase 5),
[`marginal_s0_fisher_crlb_implementation_plan.md`](marginal_s0_fisher_crlb_implementation_plan.md)
(the nuisance-amplitude specification adopted in section 2.7),
[`fisher_phase01_framework.md`](fisher_phase01_framework.md) (the executed
Phase-0/1 record and its results),
[`fisher_phase2.md`](fisher_phase2.md) (the executed Phase-2 record: the
protocol sweep, the CRLB and `kappa` maps, and the `n0_eff` sweep),
[`fisher_domain_audit.md`](fisher_domain_audit.md) (the 2026-09-06 audit that
separated the reusable substrate from the conditional acquisition analyses, and
the remediation it required).

This document is amended in place rather than superseded by parallel documents.
Every change is recorded in the **amendment log** at the end of the file, and
`madi/fisher_crlb_preregistration.json` carries a matching `amendment_log`.

---

## 1. What this phase is

The library is a forward machine: give it `(rho, V, k_io)` and it predicts a
signal curve. The Fisher analysis asks the inverse question. When a noisy
measured curve comes in, how much does it actually pin down each of the three
parameters, and which combinations of them does it fail to resolve at all?

The answer is assembled from two ingredients: how much the predicted curve
moves when each parameter is nudged (the derivatives), and how much the curve
jiggles under measurement noise. Everything in Phase 0 and Phase 1 exists to
obtain the first ingredient honestly. Phases 2 through 5 turn it into four
deliverables:

1. a quantitative statement of how well `rho`, `V`, and `k_io` can be separated
   from a single-diffusion-time PGSE acquisition;
2. a map of the parameter-space degeneracy, and a test of the hypothesis that
   it runs along constant-`v_i` hyperbolae;
3. a mechanistic explanation for the unrealistic-cell-volume artifact reported
   in MADI III and characterized in the Jackson thesis;
4. an optimized, physically feasible acquisition protocol, plus a measurement
   of how efficiently the existing estimators actually use the available
   information.

This is the analysis phase of the planned first-author manuscript. It answers a
question the originating group raised in its own Supporting Information ("very
similar curves may have quite different parameter sets... we are most likely
facing a clustered parameter space") and left unquantified.

---

## 2. The mathematics, stated once

### 2.1 The Fisher matrix

Model a measurement at acquisition column `c`, meaning one `(delta, Delta, b)`
triple, as

    m_c = S_c(theta) + eps_c,      eps_c ~ N(0, sigma_c^2)

with the parameter vector

    theta = (log rho, log V, k_io)

The Fisher information matrix is

    F_jk = sum_c (1 / sigma_c^2) * (dS_c/dtheta_j) * (dS_c/dtheta_k)
         = J^T Sigma^-1 J

where `J_cj = dS_c/dtheta_j` is the Jacobian. The Cramer-Rao lower bound states
that any unbiased estimator satisfies

    Cov(theta_hat) >= F^-1,     sigma_theta_j >= sqrt( [F^-1]_jj )

### 2.2 Why log parameters

Two reasons, both practical.

The grid is already uniform in `log rho` and `log V`, so the natural
finite-difference step is a constant in log coordinates:

    h_rho = ln(1.115884) = 0.109700
    h_V   = ln(1.170228) = 0.157230

A constant step means one truncation-bias analysis covers the whole grid.

More usefully, the CRLB on a log parameter *is* the fractional precision. If
`sigma_{log V} = 0.20` then `sigma_V / V` is about 0.20, a 20% coefficient of
variation. That is directly interpretable and directly comparable between
parameters carrying different units. `k_io` stays linear because its grid
includes zero.

### 2.3 The three reported quantities

**CRLB.** `sqrt([F^-1]_jj)`, the best achievable precision under a stated
protocol and noise level.

**Degeneracy inflation factor.** There are two distinct questions one can ask
about how well `V` is determined. First: if `rho` and `k_io` were known
exactly, how well could `V` be measured? That is answered by `1/sqrt(F_VV)`.
Second: how well can `V` be measured when all three are estimated jointly?
That is `sqrt([F^-1]_VV)`. Matrix algebra guarantees `[F^-1]_jj >= 1/F_jj`, so
define

    kappa_j = sqrt( [F^-1]_jj * F_jj )  >= 1

`kappa_j = 1` means the parameter is informationally orthogonal to the other
two and joint estimation costs nothing. `kappa_j = 10` means a tenfold
precision loss caused purely by parameter trade-off, with no reference to noise
level at all. This isolates degeneracy from SNR and is the intended headline
number of the analysis.

**Sloppy and stiff directions.** Eigendecompose the non-dimensionalized Fisher
matrix `F_tilde = D F D` with `D = diag(1, 1, k_io_ref)`, so all three axes are
measured in comparably meaningful steps. The eigenvector with the largest
eigenvalue is the parameter combination the data constrains hardest; the
smallest is the combination it barely sees. Report the spectrum and the
condition number `lambda_1 / lambda_3`.

### 2.4 The hyperbola hypothesis

In `(log rho, log V)` the direction `(1, -1)/sqrt(2)` holds `v_i = rho*V`
constant, which is exactly a constant-`v_i` hyperbola. The direction
`(1, 1)/sqrt(2)` changes `v_i`.

**Hypothesis: the sloppy eigenvector lies close to `(1, -1)`, meaning `rho` and
`V` trade off against each other along a hyperbola while `v_i` remains well
determined.**

There is a physical mechanism behind this, visible in the certified geometry
reference itself. The reference stores a normalized `<A/V>` and the runtime
multiplies by a density prefactor:

    <A/V> = (rho / 1e9)^(1/3) * g(v_i)

Moving along a constant-`v_i` hyperbola leaves `g(v_i)` completely unchanged;
only the weak cube-root prefactor varies. A full one-node step in `rho` (a
factor of 1.1159) changes `<A/V>` by only `1.1159^(1/3) - 1 = 3.7%`. Since
membrane encounter rate is what the signal is most sensitive to, motion along
the hyperbola barely changes the predicted curve.

If confirmed, this also explains a historical detail worth a paragraph in the
discussion: the original OHSU library was constructed as 1,300 `rho*V`
hyperbolae for computational convenience. If the sloppy direction runs along
those hyperbolae, their layout was an accidental empirical map of the model's
own degeneracy, discretized most coarsely in exactly the direction the data
cannot resolve.

### 2.5 The bias trap, and why the v5 schema exists

Derivatives are obtained by finite differences between neighbouring library
entries. There is no analytic derivative available; the library stores only
collapsed signal.

Fisher information is **quadratic** in the derivative. If the estimate is the
true derivative plus Monte Carlo noise, then

    E[ J_hat^2 ] = J^2 + Var(J_hat)

The noise adds. It does not cancel. So Monte Carlo noise inflates the estimated
Fisher information, which deflates the estimated CRLB, which makes the method
appear better conditioned than it is. A naive finite-difference Fisher analysis
returns precisely the reassuring answer that should be trusted least.

The correction subtracts the measured derivative variance from the diagonal:

    F_jj_corrected = sum_c (1/sigma_c^2) * [ J_hat_cj^2 - Var(J_hat_cj) ]

For a central difference over step `h`,

    Var(J_hat_cj) = [ Var(S+) + Var(S-) - 2 Cov(S+, S-) ] / h^2

**`Var(S+-)` here is the variance of the stored entry *mean*, which is
`signal_variance / n_ensembles`, not the stored `signal_variance` itself.** The
stored array is the between-ensemble sample variance of per-ensemble means, as
its own build metadata states ("consumer SE is `sqrt(signal_variance /
n_ensembles)`"). Both endpoint terms and the covariance term carry that
`1/n_ensembles`. This is spelled out because dropping it on the endpoint terms
alone is a defect that survived one full Phase-1 execution; see the amendment
log, entry `2026-09-05-varj-normalization`. The inflation it causes is
`n_ensembles (1 - r/n_ensembles) / (1 - r)`, which grows with the
common-random-number correlation `r` and therefore damages the best-correlated
axis worst.

This requires a per-entry per-column variance (the v5 `signal_variance` array,
W2) and the covariance between common-random-number partnered entries (the v5
`ensemble_means_subset` array, W3). Those arrays were added to the schema for
this analysis and for no other reason.

Off-diagonal elements are left uncorrected. Because central differences use
four disjoint entries (`rho +/- k` at fixed `V`, and `V +/- k` at fixed `rho`),
their Monte Carlo noises are independent, so the off-diagonals are already
unbiased. Those are the elements that encode the degeneracy, which is a
fortunate property of the stencil choice rather than an accident.

### 2.6 The noise model, and why it decides the protocol answer

If `sigma_c` is treated as one constant across all columns, the protocol
optimizer will recommend very large `Delta`, because longer diffusion times
give more exchange contrast. That recommendation is worthless: large `Delta`
forces a long `TE` and the signal decays away before it can be measured.

Every column carries a price. A column with timing `(delta, Delta)` requires at
minimum `TE >~ Delta + delta + t_epi`, and available signal falls as
`exp(-TE/T2)`. So the noise on the normalized signal for that column is

    sigma_c = (sigma_0 / sqrt(n_c)) * exp( (Delta_c + delta_c + t_epi) / T2 )

with `n_c` the number of averages allocated to that column. `t_epi` is the
`delta`/`Delta`-independent TE overhead and is pre-registered at **30 ms**
(roughly half a 40-60 ms EPI echo train to the k-space centre, plus refocusing
and crusher time; the mid-range of the real 20-40 ms window). It was 0 ms
through the Phase-1 execution, which prices no readout at all and so makes
long-`Delta` columns look cheaper than they are -- biasing a protocol optimizer
toward long diffusion times for a reason that is not physical. See the amendment
log, entry `2026-09-05-t-epi`. At fixed total scan
time `N = sum_c n_c`, protocol design becomes a real constrained optimization.

Three filters apply before a column is even a candidate **of a declared
acquisition**. They are evaluation-time conditions on a named scenario, not
statements about the model, and §2.8 is explicit that none of them may decide
what the Phase-1 substrate contains. The first is a property of the column
alone; the second and third are properties of a `(node, column)` pair and are
evaluated inside that node's Fisher sum:

- **Gradient feasibility.** `G = sqrt( b / (gamma^2 delta^2 (Delta - delta/3)) )`
  must satisfy `G <= G_max`. Small `delta` at high `b` is not achievable on real
  hardware. The stored grid contains columns requiring roughly 15.9 T/m, which
  no scanner produces; the builder deliberately stores them without a hardware
  model, so the mask is an analysis-time responsibility. **It is an
  evaluation-time responsibility specifically**, applied where the scenario is
  named in the result. Applying it at extraction removed 7,014 of 31,125 stored
  columns from the reusable substrate for eight days; see the amendment log,
  entry `2026-09-06-substrate-domain`, and
  [`fisher_domain_audit.md`](fisher_domain_audit.md).
- **Trust floor.** `S/S0 < 0.015` is a hard exclusion, not a weighting, but it
  is evaluated per `(entry, column)` inside that node's Fisher sum.  A global
  per-column minimum is retained only as a conservative diagnostic: using it
  as the rule would discard high-b measurements merely because a different
  tissue entry has decayed below the trust floor.  Report the surviving-column
  count per node because Fisher matrices need not use identical column sets.
- **Rician validity.** The Gaussian noise model underlying the Fisher formula
  holds only where magnitude SNR exceeds roughly 3. Below that the noise floor
  biases the signal and the formula does not apply. **The SNR that matters is
  the one the acquisition achieves**, `S / (sigma_1/sqrt(n_c))`, not the
  one-average value: an acquisition that spends `n_c` averages on a column
  measures it `sqrt(n_c)` times better. `averages_per_column = 1` in the
  pre-registered noise model is the Phase-1 screen for which columns are *ever*
  usable, and is not the right basis for evaluating a specific acquisition; at
  one average the median grid node retains 2-3 usable columns out of 24 and no
  three-parameter Fisher matrix exists anywhere on the grid. See the amendment
  log, entry `2026-09-05-phase2-averaging-aware-mask`.

Design criteria, in increasing order of trustworthiness:

- **A-optimality:** minimize `tr(W F^-1 W)`. Minimizes total variance.
- **D-optimality:** maximize `det F`. Minimizes confidence-ellipsoid volume,
  but can hide one badly determined direction behind two good ones.
- **Minimax relative CRLB:** minimize `max_j sqrt([F^-1]_jj) / theta_j`. This
  is the recommended primary criterion, because it refuses to sacrifice one
  parameter to flatter the others, which is exactly the failure mode under
  investigation.

### 2.7 The unknown amplitude, and why every CRLB is reported twice

*Adopted 2026-09-05; see the amendment log, entry `2026-09-05-s0-marginalization`.
Full specification:* [`marginal_s0_fisher_crlb_implementation_plan.md`](marginal_s0_fisher_crlb_implementation_plan.md).

Everything above treats the library's normalized `S/S0` curve as the forward
model. That silently asserts that the amplitude `S0` is known exactly. No real
acquisition knows it exactly. Model the measurement instead as

    m_c = a * S_c(theta) + eps_c,     a = S0

which promotes `a` to a fourth, unwanted parameter. In blocks,

    F_full = [ F_tt   F_ta ]     F_tt = a^2 sum_c J_cj J_ck / sigma_c^2
             [ F_at   F_aa ]     F_ta = a   sum_c J_cj S_c  / sigma_c^2
                                 F_aa =     sum_c S_c^2     / sigma_c^2 + lambda

and the bound on `theta` alone is the Schur complement

    F_eff = F_tt - F_ta (F_aa)^-1 F_at

`F_ta F_aa^-1 F_at` is a rank-one positive-semidefinite outer product, so
`F_eff <= F_tt` in the Loewner order for every `lambda >= 0`. Marginalizing over
an unknown amplitude can only lose information, never add it, and every CRLB can
only grow. This is verified numerically rather than assumed
(`tests/physics_audit/test_fisher_crlb.py`), including against the tissue block
of the explicit 4x4 inverse.

**The reporting rule: wherever a CRLB is computed, report both bounds and the
gap between them.** The gap, `sqrt([F_eff^-1]_jj) / sqrt([F_tt^-1]_jj) >= 1`,
is its own quantity. It measures how much identifiability is spent on amplitude
uncertainty alone, with no reference to any particular fitter.

`lambda >= 0` is an independent Gaussian prior precision on the amplitude and
selects the three regimes:

| Regime | `lambda` | Meaning |
|---|---|---|
| Known amplitude | `-> infinity` | `F_eff -> F_tt`; the old fixed-`S0` bound |
| Finite `b0` precision | `n0_eff * S(b_ref)^2 / sigma_ref^2` | a real reference shell of finite quality |
| Unknown amplitude | `0` | nothing constrains `a` but the tissue columns themselves |

The finite case is the realistic one, and the Jackson-thesis acquisition is its
worked example: its lowest shell is `b = 50 s/mm2`, not a true `b = 0`, so `S0`
there is neither perfectly known nor freely unconstrained. Using that shell as
the normalizer asserts `S(b_ref) = 1` and supplies amplitude precision
`n0_eff S(b_ref)^2 / sigma^2` **while discarding the shell's tissue-derivative
content**. The information thrown away by that collapse is the difference
between the prior treatment and simply retaining `b_ref` as an ordinary column,
and that difference is exactly what Phase 4 hypothesis H4 is about. H4 must
therefore reuse this implementation rather than duplicate it.

One honest limitation of the worked example: the stored `b` grid starts at 0 and
steps by 500, so **there is no stored `b = 50` column**. `S(50)` is extrapolated
from the lowest stored positive shell under a local mono-exponential law. That
extrapolation sets only a scalar prior precision, and `S(50)` sits within a few
percent of 1, so it cannot carry a conclusion; it is labelled as an
extrapolation wherever it is reported.

Implementation: `madi.fisher_crlb.amplitude_marginal_fisher`,
`amplitude_marginal_diagnostics`, `amplitude_prior_precision`; reported by
`scripts/report_s0_marginal_crlb.py`; pre-registered under `amplitude_model`.

### 2.8 The analysis domain: reusable substrate versus conditional analysis

*Adopted 2026-09-06; see the amendment log, entry `2026-09-06-substrate-domain`,
and the audit that produced it,* [`fisher_domain_audit.md`](fisher_domain_audit.md).
*Pre-registered under `analysis_domain_architecture`.*

Everything this plan computes falls into one of two layers, and confusing them
is a specific, recurring failure mode rather than a hypothetical one.

**The reusable substrate.** The stored signal `S`, the central-difference
Jacobians `J`, the derivative variance `Var(J_hat)` and its CRN covariance, and
the Richardson truncation-bias fields. These follow from the validated library
alone. No scanner, `TE`, `T2`, `SNR`, averaging budget or trust threshold enters
them. Their domain is therefore the whole stored acquisition grid: all 1,245
`(delta, Delta)` pairs times all 25 `b` values, 31,125 columns.

**The conditional analysis.** Gradient feasibility at a chosen `G_max`, the
TE/`T2` noise model, Rician validity at a chosen averaging, the `S/S0` trust
floor, the budget `N`, protocol optimisation, estimator efficiency. Each is a
statement about a *declared* acquisition. Each is legitimate, each is reported
with its declaration attached, and §5's two gradient scenarios are the model of
how such an assumption should enter: computed separately, never pooled, and left
unchosen where the user has not chosen.

The rule between them:

> Preserve and analyse the full domain supplied by the validated forward
> model/library unless an exclusion is inherent to the validated model,
> mathematically required for the requested analysis, or explicitly approved as
> part of the scientific question. Downstream feasibility quantities may be
> calculated and retained as metadata without censoring the reusable upstream
> analysis substrate.

Three consequences are binding on the rest of this plan.

1. **A conditional quantity may annotate the substrate; it may never select
   it.** Calculating `G` for every stored column, storing it, stratifying by it
   and masking a *declared* Fisher sum with it are all correct. Deciding on its
   basis whether a derivative is computed or cached is not, because it makes the
   modelled information unrecoverable at any other setting without regenerating
   the substrate.
2. **Every cache and report declares its domain.** `madi.fisher_crlb.ColumnDomain`
   is the machine-checkable form, and `require_columns` refuses to let a
   restricted cache be read as universal. A downstream analysis that asks for a
   column the substrate lacks raises rather than quietly scoring a smaller
   acquisition under the requested name.
3. **A computational limit may change the representation, never the domain.** If
   a required quantity cannot be materialised densely at full width, the answer
   is chunked, streamed or on-demand storage — not a smaller scientific domain.
   At full width the Phase-1 fields total 8.61 GiB and each Phase-2 cache member
   is 2.18 GiB, so no such limit binds here.

A fidelity caveat is an annotation under this rule too, not a censor. The
rectangular-lobe approximation is least accurate at `delta = 1-3 ms`
(`universal_library.md` §9); those columns are labelled, reported, and retained.

---

## 3. Phase 0: work available while the build runs

None of this requires the merged library. All of it can be written and tested
against the v5 pilot artifact or the partial shards.

### 0.1 Shard verification

A script that walks every completed shard and logs a per-shard table. Each
production shard is one `(rho, V)` group, so within a shard all 51 entries
share identical realized `rho` and `V` (geometry is built once per group and
reused across the `k_io` sweep) while `k_io` varies.

Checks per shard:

- required arrays present with expected shapes and dtypes;
- `vectors[:, b=0] == 1` exactly, since the `b = 0` column is deterministic;
- `signal_variance >= 0` everywhere, no non-finite values in any array;
- averaging `ensemble_means_subset` over the ensemble axis reproduces the
  corresponding `vectors` columns to float32 tolerance, which validates subset
  indexing end to end;
- realized `rho` and `V` bit-identical across the 51 entries of a group;
- negative-signal count and minimum at high `b`, against the `4.082e-4`
  independent-sampling floor at 6,000,000 axis-walks;
- the recorded CRN ensemble-index contract present in metadata.

The purpose is to catch a systematic problem days before the merge, while
intervention is still cheap.

### 0.2 Migrate `madi/identifiability.py`

Its derivative and Fisher core is sound and already differentiates on realized
entry labels rather than nominal grid nodes, which is correct. It is broken on
v5 at the metadata and column-selection layer only: it imports the removed
`_pair_indices`, expects `meta["deltas"]` where v5 provides `delta_pairs`, and
models acquisition columns as `(Delta, b)` with no `delta`.

Migrate that layer to the v5 `(delta, Delta, b)` contract. Then add what the
plan requires and the old tool lacks:

- log-parameter Jacobians (`d/dlog rho`, `d/dlog V`, linear `k_io`);
- the bias-correction term of §2.5;
- the covariance path reading `ensemble_means_subset`;
- variable stencil half-width `k`;
- the `kappa_j` degeneracy inflation factor;
- eigendecomposition of the non-dimensionalized matrix.

### 0.3 The free-water validation gates

The stored free-water atom is deliberately generated by the ordinary
Monte-Carlo walker, so its signal is not deterministic.  Split the former
single gate into a blocking software test and an informational artifact test.

**Gate A — blocking analytic pipeline test.** Feed synthetic, analytically
generated values (not the library row) through the analysis pipeline:

    S(b) = exp(-b D0)
    dS/dD0 = -b S
    F_D0D0 = sum_c (b_c^2 S_c^2) / sigma_c^2
    CRLB(D0) = 1 / sqrt(F_D0D0)

The pipeline must reproduce the derivative, Fisher element, and CRLB to a
predeclared absolute tolerance of `1e-12`. **This is the blocking gate. Nothing
downstream is trustworthy if it fails.**

**Gate B — informational free-water Monte-Carlo convergence check.** Compare
the stored signal to `exp(-b D0)` and report its maximum standardized
deviation.  The predeclared criterion is below 4 sigma, using the independent
sampling reference `1/sqrt(6,000,000)` and the recorded observed/nominal SE
ratio.  This validates the library statistically; it must never require the
library atom itself to be analytic or trigger a rebuild merely for ordinary
2--3 sigma extrema over many correlated columns.

### 0.4 Noise model and column feasibility mask

Implement §2.6: the `G_max` gradient mask (defaults for clinical 80 mT/m and
research 300 mT/m), the `S/S0 = 0.015` trust floor as a hard per-entry
exclusion, the Rician-validity threshold, and the `TE`-coupled noise model.
Report global-minimum diagnostics and the distribution of per-entry surviving
counts; only the latter is operative in a Fisher sum.

**Every output of 0.4 is a conditional annotation over the stored column grid.**
It is an input to §5 and to Phase 5.1, and it is reported per scenario so a later
conditional analysis can be reconstructed without re-running it. It is not an
extraction directive. The report's `derivative_column_selection` key, which was
one until 2026-09-06, is retained as DEPRECATED so the historical restricted
Phase-1 basis stays reproducible; see §2.8 and the amendment log, entry
`2026-09-06-substrate-domain`.

### 0.5 Pre-register the analysis grid

Written down before the data exists, to inoculate the manuscript against
garden-of-forking-paths criticism:

- stencil half-widths to evaluate: `k = 1..2` for `rho` and for `V`. The
  mask-band geometry allows `k` up to 4 on `rho` and 2 on `V` (the band is
  `log10(0.99/0.40) = 0.394` decades wide, one `rho` node is 0.0476 decades and
  one `V` node is 0.0683), and `k = 3` and `k = 4` were originally
  pre-registered on that basis. They were **dropped on 2026-09-05** once Phase 1
  settled the stencil question against them; see the amendment log, entry
  `2026-09-05-drop-k3-k4`. `k = 2` is retained *only* so the Richardson estimate
  can quantify truncation bias; **no Fisher matrix is built from `k = 2`**;
- protocols to evaluate: the MADI II timing, the MADI III `(7, 25)` ms timing,
  the Jackson thesis protocol, plus declared multi-`Delta` candidates;
- SNR levels: 50 (matching the MADI II precision test), plus 20 and 100;
- the `kappa` threshold above which a parameter is declared unidentified.

---

## 4. Phase 1: foundations, first week after the merge

### 1.1 Post-merge validation

Run the committed merged-artifact validator, plus the Phase-0.1 checks applied
to the merged file.

### 1.2 Free-water gate on the production artifact

Pass or fail. A failure stops the phase.

### 1.3 Derivative field computation

*Amended 2026-09-06; see the amendment log, entry `2026-09-06-substrate-domain`.
The previous wording restricted extraction to the research feasibility mask and
is quoted there.*

Compute central differences over **every stored `(delta, Delta, b)` column**.
Phase 1 is the reusable substrate of §2.8: a finite difference of the library
depends on no scanner, `TE`, `T2`, `SNR`, averaging budget or trust threshold, so
no such quantity may decide what it contains. For every interior grid node,
compute central differences in `log rho`, `log V`, and `k_io` at each declared
stencil width, using realized labels in the denominators. Store `J`, and for the
200 diagnostic columns also `Var(J_hat)` including the covariance term.

The Phase-0.4 feasibility report may be supplied and is then recorded in the
manifest as a conditional annotation. Restricting the substrate to it requires
the explicit `--restrict-columns-to-feasibility` flag, which stamps the manifest
RESTRICTED and exists only so the pre-2026-09-06 basis stays reproducible.

At full width the fields total 8.61 GiB, against 7.15 GiB for the restricted
basis; the storage economy the restriction bought was never the binding
constraint.

Central differences are used deliberately rather than forward differences: the
four entries involved are disjoint, which keeps the off-diagonal Fisher
elements unbiased (§2.5).

### 1.4 Derivative quality audit

Per column and axis, compute

    SNR_partial = |J_hat| / SE(J_hat)

and report the fraction of columns clearing a threshold of about 3. Columns
below it contribute mostly noise.

Compare stencil widths by Richardson extrapolation. For a central difference
the error is `O(h^2)`, so

    J_improved = (4 J_h - J_2h) / 3

and the difference between `J_h` and `J_improved` estimates the truncation
bias. There is a real tension to report: widening the stencil reduces noise but
increases truncation bias, and narrowing it does the reverse. Where that
trade-off balances for this library is a concrete methodological result.

**Truncation bias must be reported relative to `|J|`, not as an absolute RMS.**
`beta = Var(J_hat)/J_hat^2` is a squared *relative* quantity; an absolute bias
RMS and a beta cannot be read against each other, and reading the trade-off off
the two numbers side by side is the entire point of computing both. Report
`|J_h - J_improved| / |J_h|` and its square on the same cells the beta audit
uses (the 200 diagnostic columns, top 5% of `|J|`), at each stencil width, so
that total relative squared error is approximately
`beta + (relative truncation bias)^2` and the minimizing width can be read
directly. The absolute RMS is retained alongside, not replaced. See the
amendment log, entry `2026-09-05-relative-truncation-bias`.

Expected outcome, from the stencil probe: in the top 5% of derivative
magnitudes the largest `beta = Var(J_hat)/J^2` was `2.769e-4`, so this should
come back clean. The probe covered 39 entries at one grid location; Phase 1.4
is the full-grid confirmation. *(Executed. The full-grid confirmation held once
a `Var(J_hat)` normalization defect was fixed; the first execution reported
`beta` inflated by roughly 60-1000x depending on axis. Results and the
correction are recorded in* [`fisher_phase01_framework.md`](fisher_phase01_framework.md)*.)*

Note that `beta` will be strongly structured rather than uniform, because it
scales as `1/J^2` and therefore diverges wherever `J -> 0`, which is precisely
along the sloppy direction. Report where `beta` is largest and whether those
columns are ones the analysis relies on. Do not summarize `beta` by its median.

---

## 5. Phase 2: information content (weeks 2 to 3)

**Objective: quantify how much information a single-diffusion-time PGSE
acquisition carries about `(rho, V, k_io)`.**

**Question answered: statistical observability under a declared measurement
model, and acquisition optimisation within a declared hardware scenario.**
Required assumptions, all declared and all reported with the result: a gradient
ceiling (`G_max`, both scenarios, never pooled), `T2`, `t_epi`, SNR at `b = 0`,
the budget `N`, the equal-averaging allocation, the trust floor, and Rician
validity at the arm's own averaging. Phase 2 is therefore **not** a statement
about intrinsic model identifiability over the stored acquisition domain, and
its results must not be read as one; §2.8 is the boundary and
[`fisher_domain_audit.md`](fisher_domain_audit.md) §5 states the scope of the
executed Phase-2 conclusions precisely.

*Reframed 2026-09-05. Phase 2 no longer reproduces three named historical
acquisitions; it sweeps the feasible acquisition space and optimizes within it.
See the amendment log, entry `2026-09-05-protocol-sweep`, and the
`protocol_sweep` block of* `madi/fisher_crlb_preregistration.json`, *which is
the authoritative specification.*

- **2.1** Debiased Fisher matrix at every interior node, evaluated over the
  declared **sweep** of realistic `(delta, Delta)` pairs and `b`-value subsets
  drawn from the library's own stored grid (`delta = 1..30` ms; `Delta = 1..50`
  by 1 then `55..80` by 5; triangular `Delta >= delta`; 1,245 stored pairs; 25
  stored `b` values), restricted to the feasibility mask at the pre-registered
  noise model. Clinical (80 mT/m) and research (300 mT/m) gradient scenarios are
  reported separately and never pooled.
- **2.2** Maps across the `(rho, V)` plane of relative CRLB per parameter and
  of `kappa_j`.
- **2.3** The **single-`Delta` versus multi-`Delta` comparison** at **matched
  total measurement count**, so the comparison is fair on scan time rather than
  on image count. Operationally: a single-`Delta` arm spends its whole averaging
  budget on the declared `b`-subset at one `(delta, Delta)` pair; a
  multi-`Delta` arm splits the same budget `N` across `m` pairs, `m` in
  `{2, 3, 4}`. `m = 2` is the smallest set that varies diffusion time at all;
  `m <= 4` keeps per-pair averaging high enough that the Rician-validity mask
  does not begin deleting the high-`b` columns carrying the `k_io` signal, and
  keeps the search tractable.

**Judging "optimal".** The minimax relative CRLB
`min max_j sqrt([F^-1]_jj)/theta_j` survives the reframing as the primary
criterion, with A- and D-optimality secondary, but a genuine sweep needs three
things that three fixed points did not, all now pre-registered:

1. **A declared evaluation node set.** Every node carrying all three `k = 1`
   stencils: the retained `(rho, V)` mask nodes crossed with the full interior
   `k_io` grid, `1` to `125 s^-1`. **No `k_io` ceiling is imposed** (amendment
   `2026-09-05-withdraw-kio-restriction`; a previous revision restricted the set
   to `k_io <= 30` and that restriction is withdrawn by user decision). The
   Phase-1 result that no column at any `k_io > 30` node reaches the top 5% of
   `|dS/dk_io|` still stands, so the criterion may saturate on `k_io`; the
   response is to *measure* that rather than prevent it. Every sweep therefore
   reports the argmax parameter per protocol, the distribution of argmax
   parameters across the sweep, the spread of scores, and the fraction of
   protocols within 5% of the best. Per-parameter relative CRLBs are reported
   separately and never pooled, A- and D-optimality alongside, and a
   `(log rho, log V)`-only minimax as a **labelled diagnostic contrast that is
   not a criterion and not a substitute for the unrestricted result**.
2. **A relative-CRLB floor for `k_io`.** `k_io` is linear and its grid reaches
   `1 s^-1`, so an unfloored `sqrt([F^-1]_jj)/theta_j` is dominated by the
   smallest-`k_io` node as an artifact of division. The denominator is
   `max(k_io, 5 s^-1)`.
3. **A `b`-subset selection rule.** `C(24, 8) = 735,471` subsets per timing pair
   times 1,245 pairs is not enumerable, so subset sizes `{8, 12, 16}` are chosen
   by greedy forward selection under the primary criterion, with the greedy gap
   bounded against exhaustive enumeration on a declared reduced subproblem.

**The S0-marginalized bound (section 2.7) is used wherever a fitted-`S0`
acquisition is being modeled**; the fixed-`S0` bound is reported alongside it,
never substituted for it.

Predicted headline: `k_io` is weakly determined at a single `Delta` and
substantially improved by multi-`Delta` sampling. If the prediction fails, the
null result is still publishable and arguably more valuable, since it would
vindicate the single-`Delta` design used by all three MADI papers.

*(Executed 2026-09-06;* [`fisher_phase2.md`](fisher_phase2.md)*. The prediction
holds and is stronger than stated: no single-`Delta` acquisition reaches a finite
minimax at all — 0 of 3,820 arms — and the binding parameter at a single
diffusion time is `log rho`, not `k_io`. Two diffusion times roughly double the
identifiable-node fraction and produce a finite criterion.)*

The three previously named protocols (MADI II `(20, 50)`, MADI III `(7, 25)`,
and the Jackson thesis) are retained under `reference_protocols` as manuscript
context: a paragraph situating published acquisitions inside the swept space.
That paragraph does not need protocol-exact timing. The Jackson entry stays
marked `requires_source_timing_confirmation` and is **deprioritized**; its known
structure (16 `b` values from 50 to 4700 s/mm2, TE 111.5 ms, lowest shell
`b = 50` rather than a true `b = 0`) places it approximately, and the real
comparison is structural rather than protocol-exact.

---

## 6. Phase 3: the degeneracy map (weeks 3 to 4)

**Objective: characterize the degeneracy and test the hyperbola hypothesis.**

**Question answered: the geometry of the model's own degeneracy.** This is the
closest thing in the plan to an intrinsic-identifiability statement, and its one
unavoidable assumption must be stated rather than absorbed: a Fisher matrix is
`J^T Sigma^-1 J`, so an eigendecomposition is always relative to a column
weighting `Sigma` and a column set. Phase 3 therefore reports its spectra with the
declared column set and noise model named, and reports the sloppy-direction angle
of §3.2 — which is a property of the *direction*, not of the scale — as the
weighting-robust quantity. Because the substrate now spans every stored column
(§2.8), Phase 3 can be run over the full stored domain, over a declared hardware
scenario, or over both for contrast, **without regenerating any derivative
field.** Which of those is the reported result is a scientific choice for the
project owner, not an implementation default.

- **3.1** Eigendecompose the non-dimensionalized Fisher matrix at every
  interior node; report spectra and condition numbers.
- **3.2** Compute the angle between the sloppy eigenvector and the
  constant-`v_i` direction `(1, -1)/sqrt(2)`.
- **3.3** The structural figure: the `(rho, V)` plane with the sloppy
  eigenvector drawn as a short line segment at each node, overlaid on
  constant-`v_i` hyperbolae. If the hypothesis holds, the alignment is visible
  at a glance. The figure is the argument.

---

## 7. Phase 4: the unrealistic-volume pathology (weeks 4 to 6)

**Objective: explain the artifact that MADI III named and could not account
for, and that the Jackson thesis characterized and handled with a 20 pL
threshold.**

**Question answered: explanation of a fitting pathology.** H3 and H4 use the
trust floor and the amplitude prior deliberately, as *experimental conditions
that are switched on and off* — which is the correct use of a conditional
quantity and the opposite of the defect §2.8 exists to prevent. H1 needs Phase 3
and inherits its weighting statement. Note the standing coverage limitation: 136
of 369 `(rho, V)` pairs carry no Fisher matrix at all, being the mask-band edge,
and the unrealistic-volume hypothesis is a mask-boundary hypothesis.

Background: plotting ADC against recovered `V` voxelwise separates the data
into two branches, a plausible linear-like branch and an exponential-like
branch running to about 180 pL/cell, against MADI II medians of 6.0 pL (cortical
GM) and 0.91 pL (WM). The thesis found the same structure in rodent data from
the originating group, was advised against removing it because nobody
understood it, and ruled out the one published hypothesis by showing no
correlation between `v_i` pegging and ill-fitting pixels.

Hypotheses, tested in order of cost:

| # | Hypothesis | Prediction if true | Cost |
|---|---|---|---|
| H4 | `S0` mismatch (thesis data's lowest shell is `b = 50`, not 0) | blow-up shrinks with `--fit-s0` | hours |
| H3 | trust-floor violation (high-`b` library values below 0.015 are noise) | blow-up shrinks when those columns are masked | hours |
| H1 | degeneracy ridge plus mask boundary | ill-fit voxels have LOW residuals; concentrate on the boundary; posterior is banana-shaped | needs Phase 3 |
| H2 | out-of-model signal (free water, CSF partial volume, IVIM) | ill-fit voxels have HIGH residuals | larger |

- **4.1** Test H4. One refit. **H4 reuses the section 2.7 amplitude
  implementation rather than duplicating it**: the thesis's `b = 50` lowest
  shell is exactly the finite-`b0`-precision regime, and the information the
  normalize-by-lowest-shell step discards is the difference between treating
  that shell as an amplitude prior and retaining it as an ordinary column.
  `madi.fisher_crlb.amplitude_prior_precision` takes the reference shell's
  realized signal for this reason.
- **4.2** Test H3. One masking flag.
- **4.3** The residual map, which cleanly discriminates H1 from H2 and is nearly
  free: H1 predicts low residuals (many entries fit comparably, one was chosen
  arbitrarily), H2 predicts high residuals (nothing fits).
- **4.4** If H1 survives: overlay ill-fitting voxel locations on the Phase-3
  `kappa` map. The mechanism's signature is MAP estimates sliding along the
  ridge until they reach the mask boundary. Note that the reachable maximum is
  `V_max = 0.99 / (rho_min * 1e-6) = 99 pL` at `rho = 1e4`, and the observed
  blow-ups in the OHSU library reached about 180 pL against its stated 206 pL
  ceiling. Blow-up values landing at the boundary of the reachable region is
  not what a random fitting failure would produce.
- **4.5** The principled replacement for the 20 pL cutoff, if H1 is confirmed:
  report the stiff combination (`log v_i = log rho + log V`) with a CRLB-derived
  error bar, and report `rho` and `V` separately only where `kappa` is below the
  pre-registered threshold; use the Bayes posterior standard deviation as a
  per-voxel, probabilistically meaningful quality flag; and validate that flag
  against the CRLB prediction, since the Fisher matrix predicts where the ridge
  is worst before any fitting is done.

Honest limit to state in the manuscript: if H1 is the mechanism, the blow-up is
not a bug and cannot be fixed by better code. It is the model reporting that it
cannot separate `rho` from `V` for that voxel at that SNR. The fix is to stop
asking it to, and to report the combination it can determine.

---

## 8. Phase 5: protocol design and estimator efficiency (weeks 6 to 8)

**Question answered: 5.1 is acquisition/protocol optimisation under declared
hardware and budget constraints — the one place in the plan where an engineering
assumption is the point rather than a caveat. 5.2 is estimator efficiency against
a declared bound.** 5.1 must name its `G_max`, `T2`, `t_epi`, SNR and budget in
every reported protocol, and 5.2 must name which bound each efficiency number
divided by. Neither may write its assumptions back into the substrate: a change
of scanner profile at Phase 5 is a re-evaluation, not a regeneration.

**Objective 3: derive an optimal, physically feasible protocol.**

- **5.1** Optimize column selection and average allocation under a fixed
  scan-time budget using the minimax relative-CRLB criterion. Report both the
  recommended feasible protocol and the gap between it and an unconstrained
  ideal, since that gap is a useful planning number for anyone designing a MADI
  study.

**Objective 5: measure how efficiently the existing estimators use the
available information.**

- **5.2** Generate synthetic voxels at known parameters with Rician noise at
  the three declared SNR levels; fit with `map`, `bayes`, and `amico`; report
  efficiency as achieved RMSE divided by the CRLB. **For any `--fit-s0` fitter
  run the divisor must be the S0-marginalized bound of section 2.7, never the
  fixed-`S0` bound.** A fitter that estimates its own amplitude cannot reach the
  fixed-`S0` bound, so dividing by it would understate that fitter's efficiency
  against a bound it structurally cannot attain. Report which bound each
  efficiency number used. (Phase 5 is out of scope for the task that adopted
  this requirement; the requirement is recorded here so it binds when Phase 5
  runs.) MAP is confined to grid
  nodes, so at high SNR its error floors at the grid spacing rather than at the
  CRLB; quantify where that crossover occurs. This is the concrete quantitative
  argument for posterior estimation over nearest-neighbour matching.
- **5.3** The three-voxel figure. For a well-conditioned white-matter voxel, a
  degenerate gray-matter voxel, and a blown-up voxel, show side by side: the 2D
  posterior over `(log rho, log V)` at the MAP `k_io`, with constant-`v_i`
  hyperbolae, grid nodes, and the mask boundary overlaid; the CRLB ellipse from
  the Fisher matrix on the same axes; and three 1D marginals with MAP point,
  posterior mean, and CRLB interval marked.

  The point the figure makes in one glance: voxel 1's posterior is a compact
  blob the ellipse encloses tightly, voxel 2 is elongated along the hyperbola,
  and voxel 3 is a banana running to the mask boundary with its MAP point at
  the edge and its posterior mean far from it. And the CRLB ellipse, computed
  from the Fisher matrix with no reference to any data, predicts the shape of
  all three. That is what turns a diagnostic picture into an argument: the
  information geometry predicted the fitting pathology before the fit was run.

  Implementation note: the Bayes fitter already computes the full per-voxel
  weight vector `w_i` and then collapses it to mean and standard deviation. The
  figure needs `--export-voxel` extended to dump `w_i` rather than only the
  summary. That is a small addition to an existing code path.

---

## 9. Phase 6: W4 replicate and wrap-up (weeks 8 to 9)

**Question answered: calibration of the stored `signal_variance`, and the
methods ledger.** W4 is a property of the library, not of any acquisition; it
carries no hardware assumption. If it forces a recalibration, the quantity that
changes is `Var(J_hat)` in the substrate, and every conditional analysis is then
re-evaluated from it rather than re-extracted.

- **6.1** Launch the W4 independent-seed replicate (about 190 declared entries,
  a distinct build seed, already specified in
  `data/madi_v5_replicate_entry_subset.json`). Compare the replicate spread
  against the stored `signal_variance`. This is the only check that establishes
  whether the W2 between-ensemble estimator is calibrated. If it disagrees, the
  Phase-2 bias correction must be recomputed with recalibrated variances, which
  is why W4 must land before the manuscript is frozen rather than after.
- **6.2** Assemble the methods ledger: every departure of this implementation
  from the published method, what it changes, and its measured or bounded
  effect. Assemble the final figure set.

---

## 10. Dependencies

```
0.1 ─┐
0.2 ─┼─► 1.2 (gate) ─► 1.3 ─► 1.4 ─► Phase 2 ─► Phase 3 ─► 4.4 ─► 4.5
0.3 ─┘                                     │
0.4 ─────────────────────────────────────► Phase 2, 5.1      (annotation only)
0.5 ─────────────────────────────────────► Phase 2

4.1, 4.2, 4.3  need only the merged library and the fitters; can start week 1
5.2            needs the library and 0.4, not the Fisher matrix
6.1            can launch any time after the build; must land before freeze
```

**0.4 does not gate 1.3.** The arrow from 0.4 runs to the conditional analyses
only. Corrected 2026-09-06: until then 0.4's output selected 1.3's extraction
domain, which is exactly the dependency §2.8 forbids. The practical property this
buys is that every downstream engineering choice — a different `G_max`, a
different `T2`, a different budget, a preclinical profile — is a re-evaluation
against the existing substrate, never a regeneration of it.

---

## 11. Risks

| Risk | Response |
|---|---|
| Free-water gate fails | Stop. Nothing downstream is meaningful. This is why it is built in Phase 0 against the pilot rather than after the merge. |
| Derivative SNR poor at `k = 1` on the `rho` axis | Widen the stencil; the geometric ceiling is `k = 4` and Richardson extrapolation quantifies the truncation cost. The stencil probe suggests this is unlikely. |
| W4 contradicts `signal_variance` | Recompute the Phase-2 correction with recalibrated variances. The schedule absorbs this if W4 launches promptly after the build. |
| All of H1 through H4 fail on the volume pathology | Report the ablation as a negative result. The artifact characterization is publishable without a confirmed mechanism. |
| `beta` large in load-bearing columns | Report it and correct for it; that is what the correction exists for. A large, honestly corrected `beta` is a finding, not a failure. |

---

## 12. Known structural limitations to carry into the manuscript

These are properties of the library and the model, not of the analysis, and
should be stated rather than discovered by a reviewer.

- **The `rho` derivative is the noisiest of the three.** Changing `rho` changes
  both the Poisson seed count and the populated domain size, so neighbouring
  `rho` entries are independent tissue realizations rather than one tissue
  perturbed. Common random numbers buy little on that axis. This was accepted
  deliberately rather than redesigning a validated geometry generator; the
  mitigation is a wider analysis-time stencil.
- **The `V` and `k_io` derivatives are well conditioned.** At fixed `rho` the
  seed positions are reused and only the contraction radius changes; across the
  `k_io` sweep the geometry and the walker random stream are both shared.
- **Waveform is PGSE only.** TRSE approximated as PGSE with effective timing is
  an accepted, documented limitation.
- **Nearest-column matching, no interpolation.** A measured protocol that does
  not land on the stored grid is matched to the nearest column and the mismatch
  is accepted as error.
- **The trust floor is an analysis-time and fit-time mask**, not a builder
  property, and must be applied explicitly. Like every filter in §2.6 it is an
  *evaluation-time* mask; §2.8 forbids it from selecting the substrate.
- **The short-`delta`, high-`b` corner is stored but hard to play.** 7,044 of the
  31,125 stored columns require more than 300 mT/m and 21,126 require more than
  80 mT/m; at `delta = 1 ms` the median timing pair has no diffusion-weighted
  column reachable at 300 mT/m at all. Those columns are in the model and in the
  substrate, and any statement about them is a statement about the model rather
  than about an achievable acquisition. This is also the region where the
  rectangular-lobe approximation is least accurate (`universal_library.md` §9),
  so results there carry two labels, not a deletion.

---

## 13. Amendment log

This plan is amended in place. Each entry records what the document previously
said, what it says now, and why, so a decision can be reconstructed without
diffing git history by hand. Nothing is deleted; superseded wording is quoted
here. `madi/fisher_crlb_preregistration.json` carries a matching
`amendment_log` array.

### 2026-09-06-substrate-domain — the reusable substrate spans every stored column

- **Previously:** section 1.3 read *"After feasibility analysis, compute central
  differences only for columns surviving the declared research feasibility mask
  at at least one cellular node, plus all 200 diagnostic columns. This preserves
  every physically usable measurement while avoiding fields for globally
  unreachable columns."* `scripts/analyze_fisher_feasibility.py` emitted that
  set as `derivative_column_selection` and `scripts/run_fisher_phase1.py`
  required it. There was no section 2.8, and the pre-registration carried no
  statement about the analysis domain at all.
- **Now:** new **section 2.8** states the substrate/conditional boundary and its
  governing rule; section 1.3 extracts over every stored `(delta, Delta, b)`
  column; section 0.4's outputs are conditional annotations; the dependency
  graph's 0.4 arrow no longer reaches 1.3; sections 5 to 9 each name the question
  they answer and the assumptions it requires. The pre-registration gains an
  `analysis_domain_architecture` block.
- **Why:** "globally unreachable" meant unreachable *by a 300 mT/m scanner*, not
  absent from the model. The restriction removed **7,014 of the 31,125 stored
  columns** — the entire short-`delta`, high-`b` corner, including 84 timing pairs
  that lost every diffusion-weighted column — from the reusable substrate, so no
  later analysis at any other gradient setting could reach them and no
  notebook-level filter could recover them. It was never pre-registered and
  appears in no amendment log, so it entered as an implementation decision rather
  than an approved scientific choice. The full audit, with the classification of
  every other restriction in Phases 0 to 6, is
  [`fisher_domain_audit.md`](fisher_domain_audit.md).
- **Measured effect:** Phase-1 column coverage 24,111 → 31,125; fields 7.15 →
  8.61 GiB. **No executed number changes.** Every Phase-2 result was already
  conditioned on a gradient scenario whose admissible columns the restricted
  substrate contained in full (clinical ⊂ research ⊂ the old substrate), and the
  reported optima reproduce exactly on the corrected substrate.
- **What is retained:** the restricted basis stays reproducible via
  `run_fisher_phase1 --restrict-columns-to-feasibility`, the historical Phase-1
  and Phase-2 outputs are kept unchanged, and the gradient, trust-floor and
  Rician masks all keep their evaluation-time role.
- **What changes in interpretation:** no Phase-2 result may be read as a
  statement about intrinsic model identifiability over the stored acquisition
  domain. The recovered region is where cell volume is most strongly encoded, so
  that distinction is not academic; measuring it is a separate, unrun analysis.

### 2026-09-05-s0-marginalization — S0 promoted to a nuisance parameter

- **Previously:** section 2 defined the Fisher matrix on the normalized signal
  only. `S0` appeared nowhere, so every CRLB implicitly assumed the amplitude
  was known exactly. `fisher_phase01_framework.md` listed this as the first of
  its "Plan items needing a decision", and the specification sat unadopted in
  `archive/marginal_s0_fisher_crlb_implementation_plan.md`.
- **Now:** new **section 2.7**. Wherever a CRLB is computed, both the fixed-`S0`
  bound and the `S0`-marginalized Schur-complement bound are reported, together
  with the gap between them as its own quantity. Phase 4 (H4) and Phase 5 (5.2)
  are bound to the same implementation.
- **Why:** approved by the user. No real acquisition knows `S0` exactly, and
  Phase 5 would otherwise divide `--fit-s0` estimator RMSE by a bound those
  fitters structurally cannot reach.
- **Supersedes:** the archived handoff is un-archived to
  `marginal_s0_fisher_crlb_implementation_plan.md` and classified CURRENT with a
  per-section status header; its Fisher half is live, its fitting half remains a
  plan, its v2 coordinate material is superseded.

### 2026-09-05-t-epi — EPI readout time set to 30 ms

- **Previously:** `noise_model.t_epi_ms` was `0.0`, and section 2.6 gave the TE
  model without stating a value.
- **Now:** `30.0` ms, stated in section 2.6 with its physical basis.
  `scripts/analyze_fisher_feasibility.py` now takes its `T2` and `t_epi`
  defaults *from* the pre-registration instead of holding a second copy, so
  amending the pre-registration cannot leave the tool on a stale value.
- **Why:** `t_epi = 0` prices no readout at all, making long-`Delta` columns look
  cheaper than they are and biasing a protocol optimizer toward long diffusion
  times for a non-physical reason — the exact failure the TE-coupled noise model
  exists to prevent. 30 ms is the mid-range of the real 20-40 ms window.
- **Measured effect:** the *any-entry* column mask is unchanged (9,999 clinical /
  24,081 research; it is gradient-bound), but the *per-entry* survivor
  distribution — the quantity that actually enters a Fisher sum — falls
  materially: clinical median 3,985 → 3,372 and research median 5,449 → 4,391.

### 2026-09-05-relative-truncation-bias — truncation bias reported relative to |J|

- **Previously:** section 1.4 asked for Richardson extrapolation and the
  executed framework reported only an absolute RMS (0.01578 `rho`, 0.01554 `V`).
- **Now:** section 1.4 requires the bias relative to `|J|`, and its square, on
  the same cells the `beta` audit uses.
- **Why:** `beta` is a squared *relative* quantity, so an absolute RMS and a
  `beta` are not comparable, and comparing them is the entire purpose of
  computing both.

### 2026-09-05-varj-normalization — Var(J_hat) normalization defect fixed

- **Previously:** section 2.5 wrote `Var(J_hat) = [Var(S+) + Var(S-) - 2Cov]/h^2`
  without stating that `Var(S+-)` is the variance of the entry *mean*.
  `scripts/run_fisher_phase1.py` accumulated the two endpoint terms as raw
  `signal_variance/h^2` while dividing only the covariance term by
  `n_ensembles`, diverging from the module's own audited
  `madi.fisher_crlb.derivative_variance`.
- **Now:** section 2.5 states the `1/n_ensembles` explicitly, the script uses a
  single consistent endpoint scale, and a regression test pins the streaming
  path to the audited helper.
- **Why:** the defect inflated every reported `Var(J_hat)`, `SNR_partial`, and
  `beta` by `n_ensembles (1 - r/n_ensembles)/(1 - r)`, which grows with the
  common-random-number correlation `r` and so damaged the *best*-correlated axis
  worst. It is the reason the first Phase-1 execution appeared to show `k_io`
  derivative noise 25-48x that of `rho`. Derivative fields `J` were never
  affected; only the variance, SNR, and `beta` diagnostics were.

### 2026-09-05-protocol-sweep — Phase 2 reframed from three protocols to a sweep

- **Previously:** section 0.5 pre-registered "protocols to evaluate: the MADI II
  timing, the MADI III `(7, 25)` ms timing, the Jackson thesis protocol, plus
  declared multi-`Delta` candidates", and Phase 2 items 2.1 and 2.3 evaluated
  "the three published single-`Delta` protocols". The Jackson entry was blocked
  on `delta`/`Delta` values not recoverable from the thesis text or this
  repository.
- **Now:** Phase 2 sweeps the library's own feasible `(delta, Delta)` and
  `b`-subset space and compares single-`Delta` against multi-`Delta` arms at
  matched total measurement count. The three named protocols move to
  `reference_protocols` in the pre-registration as manuscript context. The
  Jackson entry keeps `requires_source_timing_confirmation` and is explicitly
  deprioritized.
- **Why:** the project goal is an optimal realistic acquisition, not reproduction
  of historical acquisitions, and approximate placement of published protocols
  is enough for the manuscript paragraph that wants them.
- **Specification gaps closed at adoption** (the reframing is sound but is not
  runnable as a one-line substitution): a declared evaluation node set, because
  an unrestricted `max` over `k_io` makes the minimax criterion degenerate given
  the Phase-1 `k_io` sensitivity result; a relative-CRLB floor for the linear
  `k_io` parameter; a `b`-subset selection rule, because the subset space is not
  enumerable; and the multi-`Delta` cardinality with its search strategy. All
  four are recorded in the pre-registration.

### 2026-09-05-drop-k3-k4 — the `k = 3` and `k = 4` stencils are dropped

- **Previously:** section 0.5 pre-registered `k = 1..4` for `rho` and `k = 1..2`
  for `V`, set purely by mask-band geometry, and Phase 1 computed and stored all
  seven derivative fields.
- **Now:** `k = 1..2` on both axes. `k = 2` exists only to feed the Richardson
  truncation-bias estimate; **no Fisher matrix is built from it**. A Phase-1
  re-run no longer writes `J_rho_k3` or `J_rho_k4`.
- **Why:** they exist at only 4,641 and 510 centres, and Phase 1 settled the
  question they were pre-registered to answer. Truncation bias already dominates
  Monte-Carlo noise by 17x (`rho`) and 55x (`V`) at `k = 1`, and each doubling of
  `h` quadruples the truncation bias, so a wider stencil is the wrong direction.
- **Evidence retained:** the existing `k = 3` / `k = 4` fields and the
  Phase-0/1 framework's report of them are kept unchanged. They *are* the
  evidence for this choice and are not deleted.

### 2026-09-05-withdraw-kio-restriction — no `k_io` ceiling in the minimax

- **Previously:** the Phase-2 evaluation node set was the retained `(rho, V)`
  mask nodes crossed with `k_io` in `[1, 30] s^-1`, on the grounds that the
  Phase-1 `k_io` sensitivity collapse makes an unrestricted max degenerate.
- **Now:** the full interior `k_io` grid, 1 to 125 s^-1. No ceiling.
- **Why:** user decision. The intent is to read the unrestricted result with the
  `k_io` limitation in mind rather than have the tooling hide it. The cap was an
  analyst's mitigation, not a measurement.
- **What replaces it:** the degeneracy is measured instead of prevented. Every
  sweep reports the argmax parameter per protocol and its distribution across
  the sweep, the spread of scores, the fraction of protocols within 5% and 1% of
  the best, and a `(log rho, log V)`-only minimax explicitly labelled a
  diagnostic contrast rather than a criterion.

### 2026-09-05-n0-eff-sweep — the amplitude reference quality is swept

- **Previously:** `n0_eff = 4` was a fixed assumption in the Phase-0/1 worked
  example of section 2.7.
- **Now:** swept over `{0, 1, 2, 4, 8}` plus the known-amplitude limit at every
  reported Phase-2 result, where `0` denotes no usable `b ~ 0` reference at all.
- **Why:** `n0_eff` governs the entire fixed-versus-fitted-`S0` trade and was an
  assumption rather than a measurement. At `n0_eff = 4` the penalty was already
  under 0.4%, so the whole informative range lies *below* the value that was
  assumed.
- **Limit of what it establishes:** this is a **variance** statement only. It
  says nothing about a *biased* amplitude reference, which is the Jackson-thesis
  situation (lowest shell `b = 50`, not 0, giving a systematic error of roughly
  `1 - exp(-50 D)` on every point). Averaging reduces noise; it cannot reduce
  bias. That distinction belongs to Phase 4 hypothesis H4 and is flagged for it,
  not resolved in Phase 2.

### 2026-09-05-phase2-averaging-aware-mask — Rician validity at the acquisition's own averaging

- **Previously:** section 2.6 gave the Rician filter as `S/sigma >= 3` without
  saying which `sigma`, and the pre-registered noise model carries
  `averages_per_column = 1`.
- **Now:** `sigma_c = sigma_1/sqrt(n_c)`, the noise the acquisition actually
  achieves on that column after its own averaging. The trust floor is unchanged.
- **Why:** the two pre-registered statements are inconsistent. The sweep declares
  that a budget of `N` images is split across the selected columns; the mask
  declares one average per column. Measured at execution, the one-average
  reading leaves the median node with 2-3 usable columns out of 24, so no
  three-parameter Fisher matrix exists anywhere on the grid and Phase 2 cannot
  run at all. `averages_per_column = 1` is the Phase-1 screen for which columns
  are *ever* usable; it is not the basis for evaluating a named acquisition.
- **Consequence, stated because it corrects an earlier claim:** `N` is no longer
  a pure scale factor. The Fisher matrix is still linear in `N`, but the mask
  depends on `n_c = N/n_selected`, so `N` changes which columns are usable and
  can change the ranking. The `2026-09-05-protocol-sweep` amendment asserted
  that the ranking was `N`-independent; **that assertion was wrong** and is
  corrected here. Results are reported at a declared `N` with a sensitivity
  check, not claimed budget-independent.

### 2026-09-05-phase2-node-aggregation — the node aggregate is a median, not a mean

- **Previously:** the evaluation node set was "weighted uniformly" and the
  criterion was "max over parameters", with the order of operations unstated.
  Read literally as a uniform mean over nodes, then a max over parameters.
- **Now:** per parameter, the **median** over evaluation nodes, where a node at
  which `F` is not positive definite contributes `+inf`; then the max over
  parameters. The **identifiable node fraction** is reported as a primary
  quantity for every protocol.
- **Why:** measured at execution. With the Monte-Carlo debias applied, *no*
  candidate acquisition is identifiable at every evaluation node, so a uniform
  mean is `+inf` for every protocol and ranks nothing. The median is the
  uniformly weighted order statistic that survives an infinite tail: it is
  finite exactly when a protocol identifies more than half the nodes.
- **What was rejected and why:** scoring on the mean over *only* the identifiable
  nodes. That is gameable in the worst possible direction — a protocol is
  rewarded for identifying fewer nodes — so it is reported alongside but is not
  the criterion.
- **Also strengthened:** a node counts as identifiable only when the entire
  inverse diagonal of `F` is positive, not merely when Sylvester's three leading
  minors are. The two are equivalent in exact arithmetic; in floating point a
  near-singular matrix can pass the leading-minor test and still return a
  non-positive cofactor ratio, which would surface as a NaN CRLB instead of as
  an unidentified node.

### 2026-09-05-complete-artifact — the production library is complete

- **Previously:** this plan was written while the build was running, and the
  Phase-0/1 framework ran against a 368-group development artifact.
- **Now:** `data/libraries/madi_dense_universal_remediated.npz` contains all 369
  canonical groups plus the free-water atom; every tool reports
  `grid_complete: true` with zero missing groups. The 368-group artifact is
  retained under its own name as the development input.
- **Why:** shard 45 landed and merged. Results and the full shard-45 diff are in
  `fisher_phase01_framework.md`.

