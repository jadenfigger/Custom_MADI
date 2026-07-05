# Reorg plan (2026-07-03)

Cleanup pass to separate the core MADI-GPU tool from side projects, legacy
code, and generated output that had accumulated in the repo. Executed on
branch `re-organizing`, committed as `83438ae "reorganizing"`.

## Removed

| What | Why |
|---|---|
| `madi_viewer_app/` (PyQt5 GUI, 20 files) | Confirmed disposable, despite being the more actively-developed of the two viewers by commit history — no longer used. |
| `scripts/madi_viewer.py` | Older/simpler viewer, also confirmed disposable. Neither viewer is used going forward. |
| `analysis/parameter_sweep_visualization.py` | Hardcoded path to one `.npz`, no CLI args — one-off, not reusable. |
| `--dwi15/--dwi25/--dwi30/--dwi40` flags in `fit_data.py` | Fully redundant with `--input Δ:path` (no bval/bvec), which reaches the same `LEGACY_SHELLS` fallback. No functionality lost. |
| `docs/thoughts.txt`, `docs/terminal_commands_history.txt` | Empty / stale personal scratch log referencing the deleted viewer. |
| `PyQt5` from `requirements.txt` | Only needed by the removed viewer app. |
| `figures/`, `error_landscape_3d_out/` — untracked from git (kept on disk) | Regeneratable output, not source. Added to `.gitignore` along with `out_singleshell/`. |

## Kept, with rationale

| What | Why |
|---|---|
| `madi/fitters.py` (CPU) alongside `fitters_gpu.py` | Genuine fallback, actually used — not legacy. |
| All fitting modes (map/bayes/amico/fits0/averages0/average-across-delta-s0/rician) | All actively used and compared against each other. |
| `analysis/plot_error_landscape.py` + `view_error_landscape_3d.py` (both) | Both explicitly wanted despite the 3D viewer being a superset/more recently touched — kept as distinct tools. |
| `scripts/edema_figures/`, `build_cohort_manifest.py`, `run_cohort_manifest.py` | Initially flagged as a separable side project (hardcoded subject IDs, external data paths), but `.claude/settings.local.json` showed a recent live cohort-manifest run against real edema data — this is active work, not legacy. |
| `scripts/_make_synth_dwi.py`, `_sanity_fitters.py` | Legitimate dev-time test helpers, not abandoned. |
| Rest of `madi/` core (`ensemble.py`, `walker_gpu.py`, `signal.py`, `library.py`, `config.py`) | Core simulation/fitting pipeline, no dead code found. |
| `docs/madi_checklist.txt`, `fitting_methods.md`, `sol_package_guide.md` | Scientific-correctness audit, math reference, and Sol supercomputer setup guide — all still load-bearing. |

## Also done

- Rewrote `README.md`'s project structure section (was stale, missing `scripts/`/`analysis/`/`docs/` layout entirely).
- Fixed a stale comment in `scripts/edema_figures/config.py` referencing the deleted viewer.

## Scientific validity audit (separate from the file reorg)

Ran a two-part audit of `madi/` against the source papers (Springer et al.,
*NMR in Biomedicine* 2023, Papers I & II + SI) using the pre-existing
checklist at `docs/madi_checklist.txt`. Findings, all reviewed and accepted
as known/intentional:

- Random-walk physics core (geometry, membrane crossing, k_io↔p_p
  translation, step mechanics) — confirmed correct against the papers,
  file:line, no issues.
- `madi/config.py`'s acquisition constants currently reflect the paper's
  clinical protocol rather than this project's 9.4T preclinical protocol —
  **known/expected**, not a bug (multiple libraries are built per
  acquisition setup as needed).
- No acquisition-agnostic raw displacement (`d_rms`) library is retained —
  the walk kernel bakes in δ/Δ at simulation time (`walker_gpu.py`'s
  `pfg1_steps`/`pfg2_steps`), so a new Δ requires resimulating rather than
  reprocessing saved data. **Known limitation**, consistent with current
  practice of resimulating per acquisition setup.
- `signal.py` computes `mean(cos φ)` per axis rather than literally
  `|⟨exp(iφ)⟩|` — equivalent under ensemble symmetry, unguarded numerically.
  **Accepted.**
- `v_i↔(α*, ⟨A/V⟩)` calibration table built from one finite multi-cell
  realization rather than the paper's 5×10⁶-ensemble cubic fit; domain
  padding smaller than the paper's own heuristic (compensated by a 1%
  escape tolerance); FWHM reporting and the `[H₂O_i]·V·k_io` metabolic
  product are not implemented. **All accepted as known limitations.**

No forbidden physics (T1/T2, anisotropy, bi-exponential fitting, distinct
intra/extracellular D) found anywhere in `madi/`.
