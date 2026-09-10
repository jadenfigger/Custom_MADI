# Fisher / CRLB Phase 3

Executed 2026-09-09. Classification: CURRENT (see [`INDEX.md`](INDEX.md)).
Specification: [`fisher_crlb_analysis_plan.md`](fisher_crlb_analysis_plan.md) §6,
as amended `2026-09-09-phase3-domain-decision` and
`2026-09-09-phase3-profiled-companion`; pre-registered in the `phase3` block of
`madi/fisher_crlb_preregistration.json`.

## Verdict

**PHASE 3 EXECUTED. THE HYPERBOLA HYPOTHESIS IS CONFIRMED IN THE PLANE AND IS
NOT THE WHOLE ANSWER.** Plan §2.4 predicted that *"the sloppy eigenvector lies
close to `(1, -1)`, meaning `rho` and `V` trade off against each other along a
hyperbola while `v_i` remains well determined."* Measured over the whole stored
acquisition domain, that is right about `rho` and `V` and wrong about which
direction the three-parameter problem sees least. Five results carry beyond this
phase.

1. **In the `(log rho, log V)` plane the degeneracy *is* the constant-`v_i`
   hyperbola.** With `k_io` profiled out, the least-determined direction of the
   `(log rho, log V)` block sits **2.95 degrees** from `(1, -1)/sqrt(2)` at the
   median node; **79%** of nodes are within 10 degrees and **68%** within 5. A
   uniformly random direction in the plane would give a median of 45 degrees.
   This is plan §2.4, confirmed, and it is the argument of
   [`fig3_3`](provenance/figures/fisher_phase3/fig3_3_sloppy_direction_field_full_stored_domain_kio10.png).
2. **But the sloppy eigenvector of the full three-parameter matrix is not the
   hyperbola — it is `k_io`.** The smallest-eigenvalue eigenvector of `D F D`
   sits **43.8 degrees** from the constant-`v_i` direction, with only 4.8% of
   nodes within 10 degrees, because it points largely out of the plane: the
   median `|k_io|` component of that unit eigenvector is **0.683**. The single
   worst-determined combination in this model is exchange, not the density /
   volume trade-off. **The two statements are not in tension; they are answers to
   different questions,** and reporting only the first would have overstated the
   hypothesis while reporting only the second would have buried it.
3. **The plane result is robust to every conditional assumption that was
   varied.** Across nine declared domains — the full stored grid, the same grid
   under uniform weighting, the same grid with the trust floor applied, the
   research (300 mT/m) and clinical (80 mT/m) gradient scenarios, and the four
   executed Phase-2 optimal acquisitions — the median profiled angle moves only
   between **2.63 and 5.11 degrees**. Plan §6 nominated the angle as *the*
   weighting-robust quantity; that is now measured rather than asserted.
4. **The one condition that does move it is amplitude.** Marginalizing over a
   completely unknown `S0` rotates the profiled sloppy direction from 2.95 to
   **15.6 degrees** on the full domain and drops the within-10-degree share from
   79% to 35%. A real `b ~ 0` reference buys that back in proportion to its size,
   which is why the Phase-2 optimal 16-column acquisitions at `n0_eff = 4` barely
   move (4.19 → 4.57 degrees) while the 29,880-column full domain barely
   recovers.
5. **The Monte-Carlo debias again decides an identifiability verdict without
   moving a direction.** Subtracting `Var(J_hat)` changes the Fisher diagonal by
   0.13-2.9%, takes the positive-definite node fraction from **1.0000 to 0.6775**,
   and rotates the sloppy eigenvector by a median of **0.14 degrees**. Phase 2
   found the same sub-1% correction flipping the single-`Delta` verdict; Phase 3
   shows it flips *whether a bound exists* while leaving *which direction is
   degenerate* essentially untouched.

**Both the model-layer and the conditional-layer results are reported, by user
decision of 2026-09-09**, and neither is nominated as the single answer. Plan §6
left that choice to the project owner explicitly; the choice made was **both,
contrasted**, and for the conditional layer **both readings** — scenario-wide and
the executed Phase-2 optima. The gradient scenarios are reported separately and
never pooled, as everywhere else in this work.

**Validation that carries the most weight:** Phase 3 re-forms the four executed
Phase-2 optimal arms by a completely different accumulation path — one streaming
pass over all 1,245 timing pairs with per-domain re-weighting, rather than Phase
2's per-arm sum — and reproduces **72 of 72 compared quantities at a maximum
relative difference of exactly 0.0** across all six `S0` regimes.

---

## 1. What Phase 3 computes, and the conventions it had to fix

### 1.1 The two instruments, and why there are two

Plan §3.1 asks for the eigendecomposition of the non-dimensionalized Fisher
matrix `F_tilde = D F D`, `D = diag(1, 1, k_io_ref)` (§2.3), and §3.2 for the
angle between its sloppy eigenvector and the constant-`v_i` direction. Plan §2.4
states the hypothesis, and it states it **about the `(log rho, log V)` plane**:
"`rho` and `V` trade off against each other along a hyperbola while `v_i`
remains well determined."

Those are not the same object. A three-parameter sloppy eigenvector need not lie
in the `(log rho, log V)` plane at all, and this one mostly does not. Reporting
only the 3x3 angle would answer §3.2 literally while answering §2.4's hypothesis
falsely, and reporting only an in-plane quantity would answer §2.4 while quietly
dropping §3.1's pre-registered instrument. **Both are therefore computed and
reported side by side**, with the pre-registered 3x3 form named as the
pre-registered one:

| Instrument | What it is | Implementation |
|---|---|---|
| **3x3 spectrum** (pre-registered, §3.1/§3.2) | eigen-spectrum of `D F D`; the sloppy eigenvector is the least-determined direction of the *three-parameter* problem | `madi.fisher_crlb.fisher_spectrum` |
| in-plane split of that eigenvector | how much of it lies in the plane, and the angle of what does | `madi.fisher_crlb.in_plane_direction_diagnostics` |
| **`k_io`-profiled 2x2 block** (companion) | `[[F_rr, F_rV], [F_rV, F_VV]] - outer([F_rk, F_Vk]) / F_kk`, the precision of `(log rho, log V)` with `k_io` estimated jointly | `madi.fisher_crlb.rho_V_profiled_spectrum` |

The profiled block is the instrument §2.4's hypothesis is actually about, and it
is the one the `(rho, V)`-plane figure of §3.3 needs, since a segment drawn in
that plane can only carry an in-plane direction. It has one further property
worth stating: **both of its axes are log parameters, so it is completely
independent of the `k_io_ref` convention.** Rescaling the `k_io` axis leaves its
angle numerically unchanged, which is pinned by
`test_the_hyperbola_angle_does_not_depend_on_the_k_io_reference_scale`. The
pre-registered 3x3 angle is *not* invariant that way, which is the second reason
both are reported.

Note that in two dimensions the profiled block's "stiff angle to `(1, 1)`" is
**identically equal** to its sloppy angle to `(1, -1)`, because the eigenvectors
are orthogonal and so are the two references. It is reported as an arithmetic
consistency check and is explicitly labelled as not a second test. The 3x3 stiff
angle *is* independent, and is reported: median **24.97 degrees** from
`(1, 1, 0)/sqrt(2)`, 66% of nodes within 30 degrees. So `v_i` is the
best-determined pair combination, but the stiff direction of the full problem is
not exactly the `v_i`-changing direction either — it carries a `k_io` component
too.

### 1.2 Conventions settled at execution

Four implementation decisions the plan left open. None changes the scientific
question; each is recorded here and in the pre-registration's `phase3` block.

**(a) `k_io_ref = max(k_io_node, 5 s^-1)`.** §2.3 specifies
`D = diag(1, 1, k_io_ref)` without fixing `k_io_ref`. The pre-registered
relative-CRLB floor (`relative_scale_for_k_io.k_io_floor_s^-1 = 5.0`) is reused,
so Phase-3 spectra and Phase-2 relative CRLBs are non-dimensionalized on one
convention and a reader can carry a number from one record to the other. The
choice affects only the third component of a 3-vector; every in-plane angle and
the whole profiled companion are invariant to it.

**(b) An angle is acute, in `[0, 90]`.** An eigenvector has no sign, so the
angle between two *directions* is taken through `arccos|cos|`. The null this is
read against is stated with every reported distribution: a uniformly random
direction gives a median of **60 degrees** in three dimensions and **45 degrees**
in two.

**(c) The sloppy direction is reported even where `F` is not positive
definite,** and the condition number is not. The debias can push a weakly
determined node indefinite; the least-determined direction still exists there,
and suppressing those nodes would hide exactly the degeneracy being measured.
`lambda_1 / lambda_3` is left NaN wherever `lambda_3 <= 0`, because a ratio
across zero is not a conditioning statement. Positive-definite fractions are
reported as primary quantities alongside, as in Phase 2.

**(d) The degeneracy flag is `lambda_2 / lambda_3`, not a span share.** When the
smallest two eigenvalues are close the sloppy eigenvector is a near-arbitrary
choice inside a plane and its angle cannot be read as a direction. The first
implementation of this record used `(lambda_2 - lambda_3) / (lambda_1 - lambda_3)`
and flagged 98.8% of nodes as "not isolated" — which was an artifact, not a
finding: this spectrum has one dominant stiff direction (`lambda_1` median
1.6e5 against `lambda_2` median 4.4e3), so that ratio is small even when
`lambda_2` and `lambda_3` are an order of magnitude apart. The direct ratio
`lambda_2 / lambda_3` has a median of **6.68** and flags **6.2%** of
positive-definite nodes as genuinely near-degenerate. The span share is retained
under an honest name (`sloppy_span_share`) as a spectrum-shape statistic.

### 1.3 What was NOT changed

- **No derivative field was regenerated.** Phase 3 reads the same unrestricted
  Phase-1 substrate and Phase-2 cache Phase 2 read — all 31,125 stored columns —
  and every declared domain is an evaluation-time re-weighting of it. This is the
  property the 2026-09-06 remediation bought, exercised for the first time.
- **The Monte-Carlo debias rule is Phase 2's**, endpoint-only
  `Var(J_hat) = (Var(S-) + Var(S+)) / (n_ensembles h^2)`, applied to the diagonal
  only. Dropping the positive CRN covariance overstates `Var(J_hat)`, so every
  matrix here is a lower bound and every CRLB conservative; the size of that
  conservatism is measured in [`fisher_phase2.md`](fisher_phase2.md) §2 and is
  unchanged.
- **`b = 0` is excluded from every domain.** `S(0) = 1` exactly for every stored
  entry, so `J` is identically zero there. Mathematically forced (domain audit
  R12), not an assumption.
- **No `k_io` ceiling.** The withdrawn restriction stays withdrawn; the
  `k_io > 30` region is reported by stratification, never by exclusion.
- **Phase 3 refuses to run on a restricted substrate.** The run aborts with the
  domain banner if the Phase-1 manifest does not declare the complete stored
  grid, because "the geometry of the model's own degeneracy" is not answerable on
  a scanner-masked substrate.

---

## 2. The declared domains

The user decision of 2026-09-09 was **both layers, contrasted**, and for the
conditional layer **both readings**. Nine domains are evaluated in one streaming
pass. The first, fourth and fifth differ **only** by the gradient ceiling, so
their contrast isolates the hardware condition; the second and third isolate the
weighting and the trust-floor assumption on identical columns.

| Domain | Layer | Columns | Gradient ceiling | Trust floor | Rician | Weighting |
|---|---|---:|---|---|---|---|
| `full_stored_domain` | model, unconditional | 29,880 (1,245 pairs) | none | not applied | not applied | TE/`T2` |
| `full_stored_domain_uniform_weighting` | model, contrast | 29,880 | none | not applied | not applied | `Sigma = I` |
| `full_stored_domain_trust_floor` | model, contrast | 29,880 | none | `S/S0 >= 0.015` | not applied | TE/`T2` |
| `scenario_wide_research` | conditional scenario | 22,836 (1,161 pairs) | 300 mT/m | not applied | not applied | TE/`T2` |
| `scenario_wide_clinical` | conditional scenario | 8,754 (917 pairs) | 80 mT/m | not applied | not applied | TE/`T2` |
| `phase2_optimum_research_size8_m1` | conditional acquisition | 8 at `(10, 43)` ms | 300 mT/m | applied | at `n_c = 16` | TE/`T2` |
| `phase2_optimum_research_size8_m2` | conditional acquisition | 16 at `(4, 44)+(10, 10)` ms | 300 mT/m | applied | at `n_c = 8` | TE/`T2` |
| `phase2_optimum_clinical_size8_m1` | conditional acquisition | 8 at `(29, 41)` ms | 80 mT/m | applied | at `n_c = 16` | TE/`T2` |
| `phase2_optimum_clinical_size8_m2` | conditional acquisition | 16 at `(11, 80)+(24, 24)` ms | 80 mT/m | applied | at `n_c = 8` | TE/`T2` |

**Why the scenario-wide arms carry no trust floor and no Rician mask.** A
scenario-wide reading names a scanner class, not a protocol, so it has no
averaging and the Rician condition `S >= 3 sigma_1/sqrt(n_c)` cannot be evaluated
at a declared `n_c`; and leaving the trust floor off is what makes the contrast
against `full_stored_domain` a clean single-variable comparison. Both conditions
are **annotated instead of applied**, which is exactly plan §2.8's rule: per node,
the share of the domain's undebiased non-dimensionalized Fisher trace contributed
by cells that satisfy each condition. On `full_stored_domain` the median node
draws **99.3%** of its information from above the trust floor, **69.2%** from
columns Rician-valid at one average, and **95.4%** from columns Rician-valid at 16
averages. The trust floor's effect on the *matrix*, not just the trace, is
measured directly by `full_stored_domain_trust_floor`.

**Noise model and scale.** `SNR = 50` at `b = 0`, `T2 = 80 ms`, `t_epi = 30 ms`,
all pre-registered. The five non-acquisition domains are evaluated at **one
average per column**; the Phase-2 arms carry their own matched-budget factor
`N / n_selected` at `N = 128`. Eigenvalues therefore scale linearly with total
averaging and are **not** comparable between a 29,880-column and a 16-column
domain; every ratio, every condition number and every angle is invariant to that
scale and **is** comparable. This is stated because the full-domain minimax
relative CRLB of 0.035 is not a protocol recommendation — it is what 29,880
images would buy.

**Evaluation nodes.** The same 11,417 as Phase 2: 233 retained `(rho, V)` mask
pairs times 49 interior `k_io` values, every node carrying all three `k = 1`
central stencils. 38.8% sit at `k_io > 30 s^-1`. The same 136 of 369 `(rho, V)`
pairs are lost to incomplete stencils at the mask-band edges, and the same
Phase-4 caveat applies.

---

## 3. Results

### 3.1 Spectra and condition numbers (plan item 3.1)

`full_stored_domain`, known amplitude, at one average per column and SNR 50.
Positive-definite node fraction **0.6775**.

| Quantity | min | q05 | median | q95 | max |
|---|---:|---:|---:|---:|---:|
| `lambda_1` (stiff) | 483 | 3.51e3 | **1.616e5** | 3.90e6 | 1.79e7 |
| `lambda_2` | -12.0 | 6.72 | 4.37e3 | 5.74e4 | 1.69e5 |
| `lambda_3` (sloppy) | -280 | -115 | 406 | 1.93e4 | 4.73e4 |
| `lambda_1/lambda_3` (PD nodes) | 50.6 | 98.0 | **281** | 2.06e3 | 2.23e6 |
| `lambda_2/lambda_3` (PD nodes) | 1.02 | 1.86 | **6.68** | 48.4 | 4.18e4 |
| profiled `(rho, V)` condition number | 1.20 | 9.30 | **22.5** | 98.8 | 9.21e4 |

`lambda_3` is negative at the low q05 because the Monte-Carlo debias can drive a
weakly determined node indefinite; those nodes are reported, not repaired
(§1.2c). **6.2%** of positive-definite nodes have `lambda_2/lambda_3 < 2` and so
carry a sloppy direction that is not uniquely defined; their angles are included
in every distribution below and are a known contributor to its tail.

Condition number across the declared domains, median over positive-definite
nodes:

| Domain | condition number median | q95 |
|---|---:|---:|
| `full_stored_domain` | 281 | 2.06e3 |
| `full_stored_domain_uniform_weighting` | 322 | 2.16e3 |
| `full_stored_domain_trust_floor` | 316 | 3.25e3 |
| `scenario_wide_research` | 522 | 4.73e3 |
| `scenario_wide_clinical` | **2,273** | 7.04e4 |
| `phase2_optimum_research_size8_m2` | 635 | 7.84e3 |
| `phase2_optimum_clinical_size8_m1` | 6,362 | 1.67e5 |

**The gradient ceiling costs about an order of magnitude in conditioning and
changes the degenerate direction hardly at all.** Clinical gradients are 8.1x
worse conditioned than the model's own domain (2,273 against 281), while the
profiled sloppy angle moves from 2.95 to 2.70 degrees. That separation — the
hardware condition governs *how badly* the problem is conditioned, the model
governs *which direction* is degenerate — is the clearest statement Phase 3
makes about the substrate/conditional boundary of §2.8.

### 3.2 The hyperbola hypothesis (plan item 3.2)

Angles to the constant-`v_i` direction. `3D` is the pre-registered
eigendecomposition of `D F D`; `profiled` is the `k_io`-eliminated `(log rho,
log V)` companion. Random-direction medians are 60 and 45 degrees respectively.

| Domain | 3D median | 3D <10° | `\|k_io\|` share of 3D sloppy | profiled median | profiled <10° | profiled <5° |
|---|---:|---:|---:|---:|---:|---:|
| `full_stored_domain` | 43.78 | 0.048 | 0.683 | **2.95** | 0.790 | 0.681 |
| `full_stored_domain_uniform_weighting` | 41.01 | 0.108 | 0.650 | **2.85** | 0.783 | 0.706 |
| `full_stored_domain_trust_floor` | 40.31 | 0.047 | 0.640 | **2.63** | 0.869 | 0.753 |
| `scenario_wide_research` | 40.73 | 0.140 | 0.646 | **2.76** | 0.796 | 0.702 |
| `scenario_wide_clinical` | 33.22 | 0.245 | 0.545 | **2.70** | 0.798 | 0.652 |
| `phase2_optimum_research_size8_m1` | 37.40 | 0.162 | 0.605 | **4.19** | 0.796 | 0.577 |
| `phase2_optimum_research_size8_m2` | 49.23 | 0.040 | 0.732 | **5.11** | 0.789 | 0.490 |
| `phase2_optimum_clinical_size8_m1` | 34.65 | 0.216 | 0.566 | **4.13** | 0.784 | 0.572 |
| `phase2_optimum_clinical_size8_m2` | 40.05 | 0.204 | 0.635 | **3.36** | 0.783 | 0.639 |

Two readings, and they must be kept apart.

**The profiled column is the answer to plan §2.4.** Its median never leaves the
2.63-5.11 degree band across a 3,700-fold range of column count (29,880 down to
8), two gradient ceilings, two weightings and a trust-floor switch, and the
within-10-degree share never leaves 0.78-0.87. The hypothesis holds, and it holds
as a property of the model rather than of any acquisition.

**The 3D column is not a refutation of that; it is a different fact.** The
three-parameter sloppy eigenvector points mostly along `k_io` — median `|k_io|`
component 0.545-0.732 — so its angle to a direction lying *in* the plane is large
almost by construction. Projected into the plane, that same eigenvector still
sits a median of **5.40 degrees** from the hyperbola on `full_stored_domain`, with
67% of nodes within 10 degrees: the part of the 3D sloppy direction that lives in
the plane agrees with the profiled instrument. The two are consistent.

The figures are
[`fig3_3`](provenance/figures/fisher_phase3/fig3_3_sloppy_direction_field_full_stored_domain_kio10.png)
(the segments, at `k_io = 10 s^-1`),
[`fig3_2`](provenance/figures/fisher_phase3/fig3_2_angle_distributions.png) (the
distribution under every domain), and
[`fig3_4`](provenance/figures/fisher_phase3/fig3_4_out_of_plane_full_stored_domain.png)
(the `|k_io|` share, which is result 2 of the verdict drawn on the plane).

### 3.3 Where in the plane it holds, and where it does not (plan item 3.3)

The structural figure is
[`fig3_3_sloppy_direction_field_full_stored_domain_kio10.png`](provenance/figures/fisher_phase3/fig3_3_sloppy_direction_field_full_stored_domain_kio10.png),
with the maps in
[`fig3_1_spectrum_maps_full_stored_domain.png`](provenance/figures/fisher_phase3/fig3_1_spectrum_maps_full_stored_domain.png)
and the per-`(rho, V)` medians as
[`table3_1_rho_V_medians_full_stored_domain.csv`](provenance/figures/fisher_phase3/table3_1_rho_V_medians_full_stored_domain.csv).

The mask band `0.40 <= rho*V*1e-6 <= 0.99` is a diagonal strip in
`(log rho, log V)`, so the figures are drawn in a **rigid 45-degree rotation** of
that plane — along-band `u = log10(rho/V)/sqrt(2)`, across-band
`w = log10(rho V)/sqrt(2)`, labelled by `v_i` — cut into three strips. A rotation
is orthogonal, so every drawn angle is the true angle and the panels stay at
equal aspect; the constant-`v_i` direction becomes horizontal, so the hypothesis
reads directly off the page. Lines of constant `rho` are drawn and labelled as
the second reference family.

The structure is strongly ordered by density, and it is not what a uniform
result would look like:

| `rho` (cells/uL) | nodes | positive definite | profiled angle median | profiled <10° | `\|k_io\|` share | condition number median |
|---|---:|---:|---:|---:|---:|---:|
| 1e4 - 1e5 | 3,724 | **0.155** | 9.73° | 0.506 | 0.704 | 1,518 |
| 1e5 - 1e6 | 3,822 | 0.860 | 1.78° | 0.860 | 0.752 | 358 |
| 1e6 - 1e7 | 3,871 | **1.000** | 2.65° | 0.995 | 0.365 | 188 |

**The low-density corner is where everything fails at once**: only 15.5% of its
nodes admit a bound at all, its conditioning is 8x worse than the high-density
corner, and its hyperbola alignment is the weakest. **16 of the 233 retained
`(rho, V)` pairs carry no identifiable node at any `k_io`**, and all 16 sit at
`rho` between 1.12e4 and 2.68e4 cells/uL with `V` between 18.9 and 56.9 pL — the
low-density, large-cell corner. That is the region the unrealistic-volume
pathology of Phase 4 runs into — an exponential-like branch running to about
180 pL/cell in the OHSU library, against a reachable `V_max` of
`0.99 / (rho_min * 1e-6) = 99 pL` in this one — so **Phase 4 should read this
table before H1**: the
Fisher matrix says the model carries essentially no joint information there, in
advance of any fit.

At the other end, `rho > 1e6` is fully identifiable, well conditioned, and its
sloppy direction has swung *into* the plane (`|k_io|` share 0.365 against 0.70 at
low density): at high cell density the residual degeneracy really is the
density/volume trade-off, and `k_io` is comparatively well determined.

The worst profiled angles are not scattered either. The five largest (75-80
degrees) all sit at `k_io` between 85 and 125 s^-1 and `v_i` between 0.834 and
0.843 — the top edge of the mask band at high exchange, where Phase 1 established
that no column reaches the top 5% of `|dS/dk_io|`. **13.5%** of nodes have a
profiled angle above 30 degrees.

### 3.4 The `k_io > 30` region

Stratified as Phase 2 stratifies, `full_stored_domain`:

| region | nodes | positive definite | 3D angle median | `\|k_io\|` share | profiled angle median | profiled <10° |
|---|---:|---:|---:|---:|---:|---:|
| `k_io <= 30` | 6,990 | 0.733 | 48.46° | 0.741 | 2.81° | 0.876 |
| `k_io > 30` | 4,427 | 0.591 | 31.38° | 0.520 | 3.16° | 0.654 |

The coarse region is less identifiable, as Phase 1 and Phase 2 both predicted,
but its **in-plane** degeneracy geometry is barely different — 3.16 against 2.81
degrees. The `k_io` sensitivity collapse costs `k_io` precision and node
coverage; it does not change what `rho` and `V` trade off against.

### 3.5 The amplitude regimes (plan §2.7)

Every CRLB is reported twice, as the plan requires, and Phase 3 extends the rule
to the geometry: the spectrum and the angles are computed under the
known-amplitude bound, the unknown-amplitude bound (`n0_eff = 0`), and the
pre-registered worked reference `n0_eff = 4`.

| Domain | regime | `lambda / F_s0s0` | PD fraction | profiled angle median | profiled <10° |
|---|---|---:|---:|---:|---:|
| `full_stored_domain` | known amplitude | — | 0.6775 | 2.95° | 0.790 |
| | `n0_eff = 4` | 0.0235 | 0.6420 | 13.08° | 0.413 |
| | unknown amplitude | 0 | 0.6275 | 15.61° | 0.353 |
| `scenario_wide_clinical` | known amplitude | — | 0.6936 | 2.70° | 0.798 |
| | `n0_eff = 4` | 0.0235 | 0.6525 | 5.32° | 0.671 |
| | unknown amplitude | 0 | 0.6280 | 6.33° | 0.588 |
| `phase2_optimum_research_size8_m2` | known amplitude | — | 0.7316 | 5.11° | 0.789 |
| | `n0_eff = 4` | 0.939 | 0.7272 | 5.04° | 0.758 |
| | unknown amplitude | 0 | 0.5503 | 21.99° | 0.253 |

**Amplitude uncertainty rotates the degeneracy, and the amount is governed by
one ratio.** Marginalizing `S0` subtracts a rank-one outer product from `F`, so
it does not merely inflate the CRLB — it moves which direction is least
determined, toward whatever combination of `(rho, V, k_io)` most resembles an
overall scale change. On the full domain that costs 12.7 degrees of alignment.

The `lambda / F_s0s0` column explains why `n0_eff = 4` behaves so differently
across domains, and it is a proportion statement rather than an anomaly:
`F_s0s0` grows with the number of tissue columns while the prior `lambda` does
not, so four `b ~ 0` averages constrain a 16-column acquisition strongly
(ratio 0.94, essentially the known-amplitude answer) and a 29,880-column one
hardly at all (ratio 0.023, essentially the unknown-amplitude answer).

`full_stored_domain_uniform_weighting` reports only the two scale-free regimes.
Its `Sigma = I` is a diagnostic re-weighting and carries no physical noise level,
so a finite amplitude prior could not be placed on the same scale as its
`F_s0s0`; the finite regimes are declared unavailable rather than computed on a
fabricated convention.

### 3.6 The Monte-Carlo debias, and what it does and does not move

| Domain | mean debias / diagonal (`rho`, `V`, `k_io`) | PD undebiased | PD debiased | sloppy rotation median | q95 | max |
|---|---|---:|---:|---:|---:|---:|
| `full_stored_domain` | 0.43%, 1.62%, 2.86% | 1.0000 | 0.6775 | 0.144° | 3.40° | 50.4° |
| `scenario_wide_research` | 0.31%, 1.21%, 2.03% | 1.0000 | 0.6908 | 0.086° | 2.77° | 64.6° |
| `scenario_wide_clinical` | 0.13%, 0.52%, 0.91% | 1.0000 | 0.6936 | 0.054° | 2.29° | 79.1° |

The same picture Phase 2 reported, seen from the geometry side. A correction of
under 3% on the diagonal removes a bound at a third of the grid, because the
debiased matrix is that close to singular; **and yet the direction it identifies
as sloppy barely moves** — a median rotation of a tenth of a degree, though up to
79 degrees at individual near-singular nodes. The profiled angle median moves
2.83 → 2.95 degrees on the full domain.

One number is worth stating plainly for the manuscript. **An undebiased
finite-difference Fisher analysis of this library would report every one of the
11,417 nodes as identifiable.** The library says 68%.

### 3.7 The Phase-2 reproduction cross-check

Phase 3 forms its Fisher matrices by a different route from Phase 2: one
streaming pass over all 1,245 timing pairs computing unweighted per-column
contributions once, then re-weighting them per declared domain. Phase 2 sums each
arm's own columns directly. The four executed Phase-2 optimal arms are therefore
re-derived here as an independent check.

| Arm | quantities compared | max relative difference |
|---|---:|---:|
| `phase2_optimum_research_size8_m1` | 6 | **0.0** |
| `phase2_optimum_research_size8_m2` | 30 | **0.0** |
| `phase2_optimum_clinical_size8_m1` | 6 | **0.0** |
| `phase2_optimum_clinical_size8_m2` | 30 | **0.0** |

Exactly zero, not a tolerance. The `m1` arms compare 6 quantities rather than 30
because their minimax and their per-parameter medians are `+inf` under every
regime, which is itself Phase 2's headline reproduced: **no single-`Delta`
acquisition has a finite minimax.** Non-finite pairs are skipped rather than
compared, so the comparison count is a fingerprint of that result.

---

## 4. What Phase 3 answers that earlier phases left open

- **`fisher_phase2.md` §3.5** deferred: *"Whether the worst nodes lie along
  constant-`v_i` hyperbolae is a Phase-3 question and is deliberately not
  answered here."* Answered: the degeneracy direction in the plane **is** the
  constant-`v_i` hyperbola almost everywhere, and Phase 2's observation that the
  worst `log rho` and `log V` `kappa` nodes coincide is exactly what a shared
  hyperbolic degeneracy predicts.
- **`fisher_domain_audit.md` §4** stated that the recovered short-`delta`,
  high-`b` corner made a model-level identifiability question askable and that
  the audit did not answer it. Phase 3 is the first analysis to use it: the
  `full_stored_domain` arm is the model-layer result, and its contrast against
  the two scenario arms measures what the ceiling costs — an order of magnitude
  in conditioning, essentially nothing in direction.
- **Plan §2.4's historical aside** is supported. If the original OHSU library's
  1,300 `rho*V` hyperbolae were laid out for computational convenience, they were
  an accidental empirical map of the model's own degeneracy: the direction the
  data cannot resolve sits a median of 3 degrees from those hyperbolae.

---

## 5. Limitations

Stated here rather than left implicit.

- **A spectrum is relative to a column set and a weighting, always.** Plan §6 is
  explicit about this and it is not removed by reporting nine domains; it is
  bounded by them. What is established is that the *angle* is stable across the
  ones varied, not that it is stable across every conceivable weighting.
- **The `Sigma` varied here is TE/`T2`-versus-uniform only.** A weighting derived
  from a different `T2`, a different `t_epi`, or a per-column SNR model was not
  swept. The two tried are not a trivial contrast — the TE-weighted `u_c` spans a
  factor of `exp(2 (TE_max - TE_min)/T2) = exp(2.7) ~ 15` between the shortest
  stored timing (`delta = Delta = 1 ms`, TE 32 ms) and the longest
  (`delta = 30`, `Delta = 80`, TE 140 ms), against a flat 1 — but they are two
  points, not a sweep.
- **6.2% of positive-definite nodes have `lambda_2/lambda_3 < 2`**, where the
  sloppy eigenvector is not uniquely defined. Their angles are in every reported
  distribution. They are a known contributor to the tail and were not excluded,
  because excluding them would select on the outcome.
- **The debias is endpoint-only outside the 8 diagnostic timing pairs**, exactly
  as in Phase 2, so it overstates `Var(J_hat)` and understates identifiability.
  The 0.6775 positive-definite fraction is a lower bound; Phase 2 calibrated the
  gap at about 6 percentage points at `(20, 50)` ms.
- **Truncation bias, not Monte-Carlo noise, remains the dominant systematic**
  (Phase 1: ~1.4% relative on `rho`, ~1.8% on `V` at `k = 1`). Its effect on an
  *angle* was not propagated. The angle differences this record treats as
  meaningful are the 2.6-to-5.1-degree spread across domains and the 2.95-to-15.6
  amplitude effect; a systematic of a few percent in `J` is not plausibly
  responsible for the latter, but the former is within reach of it and should not
  be over-read.
- **W4 has still not run.** Every debias here rests on `signal_variance` being
  correct in absolute terms. Phase 3 raises the stakes no further than Phase 2
  did: the debias flips a bound, not a direction.
- **136 of 369 `(rho, V)` pairs carry no Fisher matrix at all**, the mask-band
  edges, unchanged from Phase 2 and still exactly where Phase 4 will look.
- **Eigenvalues are not comparable between domains**; only ratios, condition
  numbers and angles are. Domains differ by up to 3,700x in column count and the
  five model/scenario arms carry no budget at all.
- **The `k_io`-profiled companion assumes `F_kk > 0`.** Where the debias drives
  `F_kk` non-positive there is no profiled block; those nodes are NaN in the
  angle and are excluded from its distributions by non-finiteness, which is a
  mild selection whose direction is not established.
- **Phase 4 material is out of scope and not pre-empted.** The low-density
  finding of §3.3 is offered to H1 as a prediction to test, not as a test.

---

## 6. Open decisions, for the user

1. **Whether the reported manuscript result is the full stored domain or a
   scenario.** Both are computed and neither is nominated, per the 2026-09-09
   decision. The manuscript will eventually have to pick a headline, and the
   honest framing is that §3.2's profiled angle is a model statement while
   §3.1's condition numbers are conditional. This is the same standing question
   as the clinical/research choice deferred on 2026-09-05.
2. **Whether to sweep the column weighting further.** Two weightings were tried.
   A `T2` sweep (say 60-100 ms) would turn "robust to the weightings tried" into
   "robust across the physiological `T2` range", at about two minutes of compute
   per point.
3. **Whether the amplitude rotation of §3.5 deserves its own analysis.** It is
   the largest effect Phase 3 found on the degeneracy direction, it is larger
   than every hardware effect measured, and it bears directly on Phase 4 H4 and
   Phase 5.2. It was measured here, not explained.
4. **Whether Phase 4 should use the §3.3 low-density table as an H1 prior.** The
   16 `(rho, V)` pairs with no identifiable node sit at the low-density,
   large-cell corner the unrealistic-volume blow-ups run toward. Deciding whether that is a prediction Phase
   4 tests or a confound Phase 4 must control for is a scientific choice.

---

## 7. Reproduction

*A fresh agent should follow this section and not reconstruct the run from git
history.* About 2 minutes of compute and 15 MiB of output, on top of the existing
Phase-1 fields and Phase-2 cache; the figures take a few seconds more. Nothing is
regenerated — Phase 3 is a re-evaluation of the existing unrestricted substrate.

```bash
conda activate mri
LIB=data/libraries/madi_dense_universal_remediated.npz
OUT=/home/jaden/madi_fisher_runs/full_domain

PYTHONPATH=. python -m scripts.run_fisher_phase3 \
    --artifact "$LIB" --phase1 "$OUT/phase1" --cache-dir "$OUT/cache" \
    --output-dir "$OUT/phase3" \
    --phase2-report "$OUT/phase2_N128/phase2_report.json"

PYTHONPATH=. python -m scripts.plot_fisher_phase3 \
    --run-dir "$OUT/phase3" --output-dir docs/provenance/figures/fisher_phase3
```

A healthy run prints `COLUMN DOMAIN COMPLETE — all 31125 stored (delta, Delta, b)
columns` and `GRID COMPLETE`, reports nine declared domains over 11,417 nodes,
and ends with four Phase-2 reproduction lines at `max relative difference
0.000e+00`. **If the Phase-1 manifest declares anything but the complete stored
grid the run aborts**, by design: the model-layer question is not answerable on a
scanner-masked substrate.

`--node-stride` and `--pair-limit` are smoke-only, are recorded in the manifest,
and disable the Phase-2 cross-check, since subsampling nodes changes a median over
nodes and the Phase-2 quantities would no longer be comparable.
`--phase2-report` is optional; without it the four declared-acquisition domains
and the cross-check are simply absent. `--domain` and `--k-io` on the plotter
choose which declaration the structural figures are drawn for; the record's
figures use `full_stored_domain` at `k_io = 10 s^-1`.

**Outputs**, in `$OUT/phase3`:

| File | Contents |
|---|---|
| `phase3_report.json` | the full record: every declared domain, all six `S0` regimes, spectra, angles, condition numbers, annotations, the debias contrast, the cross-check |
| `summary.txt` | the printed digest |
| `evaluation_nodes.npy`, `evaluation_node_labels.npy` | `(rho_index, V_index, k_io_index)` and `(rho, V, k_io)` per row, shared by every map |
| `maps_<domain>.<quantity>.npy` | 18 per-node maps per domain under the known-amplitude bound: eigenvalues/eigenvectors, condition number, both angles, in-plane and `k_io` shares, the profiled block's vectors/angles/eigenvalues/condition, positive-definiteness, relative CRLB and `kappa` |

Figures and the accompanying table are committed under
[`provenance/figures/fisher_phase3/`](provenance/figures/fisher_phase3/), because
the 8.6 GiB substrate they derive from is not in the repository and they are the
surviving record of this run.

**Implementation.** All Phase-3 arithmetic is in `madi/fisher_crlb.py`
(`nondimensionalized_fisher`, `fisher_spectrum`, `direction_angle_deg`,
`in_plane_direction_diagnostics`, `rho_V_profiled_block`,
`rho_V_profiled_spectrum`, `degeneracy_geometry`, plus the
`CONSTANT_VI_DIRECTION` / `VI_CHANGING_DIRECTION` constants).
`scripts/run_fisher_phase3.py` is orchestration: it imports `build_node_table`,
`pair_contributions` and `_summary` from `scripts/run_fisher_phase2.py` rather
than re-deriving them, which is the pattern
[`fisher_phase2.md`](fisher_phase2.md) §1.2 requires.
`scripts/plot_fisher_phase3.py` computes no Fisher quantity at all; it reads the
stored maps.

**Tests.** Nine added to `tests/physics_audit/test_fisher_crlb.py`, which is now
33 tests:

1. `test_fisher_spectrum_matches_the_single_node_reference` — the batched
   spectrum equals `fisher_diagnostics` matrix by matrix, eigenvalues and
   eigenvectors up to sign, in descending order.
2. `test_a_constructed_constant_vi_degeneracy_is_recovered_exactly` — the
   positive control: a matrix built blind along `(1, -1, 0)` reports a zero angle
   and a zero smallest eigenvalue.
3. `test_the_direction_angle_is_acute_and_sign_free` — an eigenvector's sign
   cannot change its reported angle.
4. `test_the_profiled_block_inverts_the_rho_V_block_of_F_inverse` — the Schur
   complement really is the joint `(log rho, log V)` precision.
5. `test_the_hyperbola_angle_does_not_depend_on_the_k_io_reference_scale` — the
   reported hypothesis test is not an artifact of the `D` convention.
6. `test_an_indefinite_debiased_node_is_reported_rather_than_repaired` — a
   direction at a non-PD node, and no condition number across zero.
7. `test_a_near_degenerate_sloppy_pair_is_flagged_by_the_eigenvalue_ratio` —
   including that the span share is deliberately not the degeneracy flag.
8. `test_phase3_accumulation_matches_the_audited_fisher_primitives` — the
   streaming pass equals `fisher_matrix` + `derivative_variance` column by
   column, which is the `Var(J_hat)`-defect guard applied to the new path.
9. `test_phase3_domains_apply_masks_without_changing_the_shared_substrate` — a
   domain is exactly the sum of its columns, a mask subtracts exactly the masked
   column, and the annotation reports exactly what the condition would remove.

The full suite is **122 passed, 4 skipped, 1 xfailed**, plus the one
**pre-existing, unrelated** failure
`tests/physics_audit/test_gpu_golden.py::test_cpu_golden_hash_and_deterministic_reference_replay`
(`KeyError: 'realised_vi_spatial_se'`, a geometry-stats schema drift in the
stored `cpu_gpu_golden_v1.npz` fixture). It is recorded as pre-existing in
[`fisher_phase2.md`](fisher_phase2.md) §1.2 and
[`fisher_domain_audit.md`](fisher_domain_audit.md) §7b, and was not touched.

---

## 8. Amendment log

This record is updated in place. Entries record what changed, what it replaced,
and why.

### 2026-09-09-initial — first execution of Phase 3

- **Previously:** Phase 3 was unrun. `fisher_crlb_analysis_plan.md` §6 specified
  items 3.1-3.3 and left the reported analysis domain explicitly to the project
  owner; `fisher_phase2.md` §7 recorded eigendecomposition, sloppy directions and
  the hyperbola hypothesis as out of scope and deliberately not pre-empted.
- **Now:** this record, `scripts/run_fisher_phase3.py`,
  `scripts/plot_fisher_phase3.py`, the degeneracy-geometry block of
  `madi/fisher_crlb.py`, nine tests, and the run at
  `/home/jaden/madi_fisher_runs/full_domain/phase3`.
- **User decision recorded:** the reported result is **both layers, contrasted**
  — the full stored domain as the model-layer answer, both declared gradient
  scenarios beside it, and for the conditional layer **both readings**,
  scenario-wide and the executed Phase-2 optima. Neither layer is nominated as
  the single answer and the two scenarios are never pooled.
- **Why the profiled 2x2 companion was added:** plan §3.1 pre-registers the 3x3
  eigendecomposition and plan §2.4 states the hypothesis about the `(log rho,
  log V)` plane. Those are different objects here, because the three-parameter
  sloppy direction is mostly `k_io`. Reporting only the pre-registered instrument
  would have answered §3.2 literally and §2.4 falsely.
- **Not changed:** the substrate, the Phase-1 fields, the Phase-2 cache, the
  Phase-2 sweep, the debias rule, the noise model, the evaluation node set, the
  withdrawn `k_io` ceiling, and every executed Phase-0/1 and Phase-2 number.

### 2026-09-09-degeneracy-flag — the near-degeneracy statistic was corrected before publication

- **Previously:** the first implementation reported
  `sloppy_separation = (lambda_2 - lambda_3) / (lambda_1 - lambda_3)` as the
  measure of whether the sloppy eigenvector is uniquely defined, and flagged
  98.8% of nodes as "not isolated".
- **Now:** the flag is `lambda_2 / lambda_3` on positive-definite nodes, with a
  threshold of 2; it flags 6.2%. The span share is retained under the name
  `sloppy_span_share` and described as a spectrum-shape statistic.
- **Why:** the span-share form is small whenever one stiff direction dominates,
  which it does here (`lambda_1` median 1.6e5 against `lambda_2` median 4.4e3),
  so it was reporting the spectrum's shape as if it were a degeneracy of the
  sloppy pair. No published number depended on it; it was corrected before this
  record was written.
