"""Fisher information, eigen-spectrum and CRLB at a library node.

What this adds to the slice
---------------------------
The chi2 slice is the EXACT confidence region on the sampled manifold: every
entry whose prediction agrees with the reference to within the noise.  The
Fisher matrix is that region's LOCAL QUADRATIC approximation.  Expanding the
prediction around the reference, `s(theta) ~ s_ref + J dtheta`, gives

    chi2(theta)  ~  dtheta^T F dtheta,      F = (J/sigma)^T (J/sigma)

so `{chi2 <= T}` is, to second order, the ellipsoid `{dtheta^T F dtheta <= T}`.
Where the two agree the linearisation holds; where the survivors curve away
along the constant-`v_i` hyperbola and the ellipse does not, it has broken
down.  Showing both on one plot is the point of this module.

What is computed here and what is not
-------------------------------------
Only the **Jacobian** is built here, by central differences across neighbouring
library entries, exactly as `scripts/run_fisher_phase1.py` does it: the same
canonical nominal grid, the same log-coordinate denominators, the same
pre-registered stencil half-widths.  Everything downstream -- the Fisher
matrix, its Monte-Carlo debias, the eigen-spectrum, CRLB, the condition number,
the constant-`v_i` angles, the S0 marginalisation -- is `madi.fisher_crlb`,
imported and called, never reimplemented.  A second copy of that arithmetic is
exactly what `docs/INDEX.md` classifies a notebook as SCRATCH for.

Honest limits, all surfaced in the UI
-------------------------------------
* A central stencil needs both neighbours to exist.  At the `v_i` band edges
  and at `k_io = 0 / 130` they do not, so 233 of 369 `(rho, V)` nodes and 49 of
  51 `k_io` values carry one: 11,417 of 18,819 entries (60.7%).  Nodes without
  one report no Fisher matrix rather than a one-sided substitute.
* The derivative is taken with respect to the NOMINAL grid coordinate.  The
  realised finite-geometry `rho`/`V` differ from it by up to 0.69% / 0.93%.
* The Monte-Carlo debias is conservative unless the common-random-number
  covariance is available, and that needs `ensemble_means_subset`, stored for
  only 8 `(delta, Delta)` pairs x 25 b = 200 columns.  Debiasing is therefore
  off by default and says how many of the chosen columns it could correct.
* A debiased Fisher matrix can come out indefinite.  `fisher_crlb` reports that
  rather than repairing it, and so does this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from madi import fisher_crlb as fc

from .columns import N_ENSEMBLES, LibraryLabels, memmap_member, read_small_member

# Axis order matches fc.PARAMETER_ORDER == ("log_rho", "log_V", "k_io").
AXES = ("rho", "V", "k_io")
AXIS_LABELS = {"rho": "log rho", "V": "log V", "k_io": "k_io"}

# Pre-registered half-widths (madi/fisher_crlb_preregistration.json).
DEFAULT_WIDTH = 1
RICHARDSON_WIDTHS = (1, 2)


@dataclass(frozen=True)
class NodeGrid:
    """The canonical `(rho, V, k_io)` grid, and which library row is each node.

    Nodes are indexed `(ir, iv, ik)` into the canonical axes, which is what a
    finite-difference stencil steps through.  `rhos`/`volumes`/`kios` are the
    NOMINAL axes from `fc.canonical_grid()`, so the stencil denominators match
    Phase 1's exactly.
    """

    rhos: np.ndarray
    volumes: np.ndarray
    kios: np.ndarray
    retained: frozenset
    lookup: dict
    node_of_row: dict
    log_rhos: np.ndarray = field(repr=False, default=None)
    log_volumes: np.ndarray = field(repr=False, default=None)

    @property
    def n_kio(self) -> int:
        return len(self.kios)

    def row(self, node) -> int | None:
        return self.lookup.get(tuple(int(i) for i in node))

    def nearest_node(self, rho: float, V: float, kio: float):
        """Canonical node nearest a parameter triple, in log rho / log V / k_io."""
        ir = int(np.argmin(np.abs(self.log_rhos - np.log(max(rho, 1e-300)))))
        iv = int(np.argmin(np.abs(self.log_volumes - np.log(max(V, 1e-300)))))
        ik = int(np.argmin(np.abs(self.kios - kio)))
        return ir, iv, ik


def build_node_grid(labels: LibraryLabels) -> NodeGrid:
    """Map every cellular library row onto its canonical grid node.

    Uses `fc.build_grid_manifest`, the same nominal-coordinate matching the
    Phase-0/1 validator uses, so a row that this call places at `(ir, iv, ik)`
    is the row Phase 1 would have differenced there.
    """
    rhos, volumes, kios, retained = fc.canonical_grid()
    manifest = fc.build_grid_manifest(labels.nominal_rhos, labels.nominal_Vs,
                                      labels.is_free_water)
    lookup: dict = {}
    node_of_row: dict = {}
    for (ir, iv), entries in manifest.group_entries.items():
        # k_io ascends inside a group, but sort rather than trust the order.
        for ik, entry in enumerate(entries[np.argsort(labels.kios[entries])]):
            lookup[(int(ir), int(iv), int(ik))] = int(entry)
            node_of_row[int(entry)] = (int(ir), int(iv), int(ik))
    return NodeGrid(rhos=rhos, volumes=volumes, kios=kios,
                    retained=frozenset(retained), lookup=lookup,
                    node_of_row=node_of_row,
                    log_rhos=np.log(rhos), log_volumes=np.log(volumes))


def stencil(grid: NodeGrid, node, axis: str, width: int = DEFAULT_WIDTH):
    """`(minus_row, plus_row, denominator)` of a central stencil, or None.

    The denominator is the step in the parameter's own coordinate: a difference
    of logs for `rho` and `V`, a plain difference for `k_io`.  Returns None
    when either neighbour is outside the retained grid, which is how band edges
    and the `k_io` endpoints report themselves.
    """
    ir, iv, ik = (int(i) for i in node)
    if axis == "rho":
        if (ir - width, iv) not in grid.retained or (ir + width, iv) not in grid.retained:
            return None
        minus, plus = grid.row((ir - width, iv, ik)), grid.row((ir + width, iv, ik))
        denominator = float(grid.log_rhos[ir + width] - grid.log_rhos[ir - width])
    elif axis == "V":
        if (ir, iv - width) not in grid.retained or (ir, iv + width) not in grid.retained:
            return None
        minus, plus = grid.row((ir, iv - width, ik)), grid.row((ir, iv + width, ik))
        denominator = float(grid.log_volumes[iv + width] - grid.log_volumes[iv - width])
    elif axis == "k_io":
        if ik - width < 0 or ik + width >= grid.n_kio:
            return None
        minus, plus = grid.row((ir, iv, ik - width)), grid.row((ir, iv, ik + width))
        denominator = float(grid.kios[ik + width] - grid.kios[ik - width])
    else:
        raise ValueError(f"unknown axis {axis!r}")
    if minus is None or plus is None or denominator <= 0.0:
        return None
    return minus, plus, denominator


def jacobian(grid: NodeGrid, block: np.ndarray, node,
             width: int = DEFAULT_WIDTH) -> tuple[np.ndarray | None, list[str]]:
    """`dS/dtheta` at one node, as (n_columns, 3), or None with the missing axes.

    ``block`` is the FULL (n_entries, n_columns) signal block, because the
    neighbours are ordinary library rows and are already in memory.
    """
    columns = np.zeros((block.shape[1], len(AXES)))
    missing = []
    for index, axis in enumerate(AXES):
        found = stencil(grid, node, axis, width)
        if found is None:
            missing.append(axis)
            continue
        minus, plus, denominator = found
        columns[:, index] = (block[plus] - block[minus]) / denominator
    if missing:
        return None, missing
    return columns, []


def jacobian_batch(grid: NodeGrid, block: np.ndarray,
                   width: int = DEFAULT_WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Jacobians at every node that has a complete central stencil.

    Returns `(rows, J)` with `J` of shape `(n_nodes, n_columns, 3)`.  Vectorised
    over nodes: the three differences are gathered as whole-array index
    operations, so the whole grid costs about as much as a few hundred single
    nodes.
    """
    rows, minus_index, plus_index, denominators = [], [], [], []
    for node, row in grid.lookup.items():
        found = [stencil(grid, node, axis, width) for axis in AXES]
        if any(item is None for item in found):
            continue
        rows.append(row)
        minus_index.append([item[0] for item in found])
        plus_index.append([item[1] for item in found])
        denominators.append([item[2] for item in found])
    if not rows:
        return np.empty(0, dtype=int), np.empty((0, block.shape[1], len(AXES)))
    minus_index = np.asarray(minus_index)
    plus_index = np.asarray(plus_index)
    denominators = np.asarray(denominators)
    # block[index] is (n_nodes, 3, n_columns); move the parameter axis last.
    difference = block[plus_index] - block[minus_index]
    J = np.transpose(difference / denominators[:, :, None], (0, 2, 1))
    return np.asarray(rows, dtype=int), J


def richardson(J_fine: np.ndarray, J_coarse: np.ndarray) -> np.ndarray:
    """`(4 J_h - J_2h) / 3`, Phase 1's truncation-cancelling combination."""
    return (4.0 * np.asarray(J_fine) - np.asarray(J_coarse)) / 3.0


def truncation_estimate(J_fine: np.ndarray, J_coarse: np.ndarray) -> float:
    """Relative size of the k=1 vs k=2 disagreement, as one number.

    For a second-order central difference the leading error scales with h^2, so
    `(J_h - J_2h) / 3` estimates the k=1 truncation bias.  Reported relative to
    the Jacobian's own magnitude, which is what Phase 1 calls `r_trunc`.
    """
    fine, coarse = np.asarray(J_fine, dtype=float), np.asarray(J_coarse, dtype=float)
    bias = (fine - coarse) / 3.0
    scale = np.sqrt(np.mean(fine ** 2))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.sqrt(np.mean(bias ** 2)) / scale)


# ---------------------------------------------------------------------------
# Monte-Carlo derivative variance, with the CRN covariance where it exists
# ---------------------------------------------------------------------------

@dataclass
class CrnSubset:
    """Where the stored per-ensemble means cover the chosen columns.

    The builder kept `ensemble_means_subset` for 8 `(delta, Delta)` pairs x 25
    b-values only.  Common random numbers make a difference of neighbouring
    entries far less noisy than `var_minus + var_plus` suggests, so without the
    covariance term the debias is conservative -- it subtracts too much and can
    push a weakly determined node indefinite.
    """

    position: np.ndarray            # per chosen column: index into 200, or -1
    ensemble: np.ndarray | None     # memmap (n_entries, n_ensembles, 200)
    n_ensembles: int = N_ENSEMBLES

    @property
    def covered(self) -> np.ndarray:
        return self.position >= 0

    @property
    def n_covered(self) -> int:
        return int(self.covered.sum())


def build_crn_subset(library_path, labels: LibraryLabels, columns) -> CrnSubset:
    """Match chosen columns against the stored per-ensemble diagnostic columns."""
    columns = np.asarray(columns, dtype=np.int64)
    position = np.full(columns.shape, -1, dtype=np.int64)
    try:
        subset_deltas = read_small_member(library_path, "ensemble_subset_pair_deltas")
        subset_Deltas = read_small_member(library_path, "ensemble_subset_pair_Deltas")
        subset_b = read_small_member(library_path, "ensemble_subset_b_values")
        subset_n_b = int(read_small_member(library_path, "ensemble_subset_n_b"))
        ensemble = memmap_member(library_path, "ensemble_means_subset")
    except (KeyError, ValueError, OSError):
        return CrnSubset(position=position, ensemble=None)

    pairs = {(float(d), float(D)): index
             for index, (d, D) in enumerate(zip(subset_deltas, subset_Deltas))}
    b_index = {float(b): index for index, b in enumerate(subset_b)}
    for slot, column in enumerate(columns):
        delta, Delta, b = labels.column_triple(int(column))
        pair = pairs.get((float(delta), float(Delta)))
        b_slot = b_index.get(float(b))
        if pair is not None and b_slot is not None:
            position[slot] = pair * subset_n_b + b_slot
    return CrnSubset(position=position, ensemble=ensemble,
                     n_ensembles=int(ensemble.shape[1]))


def derivative_variance(grid: NodeGrid, variance_block: np.ndarray, node,
                        width: int = DEFAULT_WIDTH,
                        crn: CrnSubset | None = None) -> np.ndarray | None:
    """`Var(J_hat)` at one node, (n_columns, 3), using fc.derivative_variance.

    Columns the CRN subset covers get the covariance-corrected variance; the
    rest get the conservative `(var_minus + var_plus) / n_ensembles` form.
    """
    answer = np.zeros((variance_block.shape[1], len(AXES)))
    for index, axis in enumerate(AXES):
        found = stencil(grid, node, axis, width)
        if found is None:
            return None
        minus, plus, denominator = found
        plain = fc.derivative_variance(variance_block[minus], variance_block[plus],
                                       None, None, denominator, N_ENSEMBLES)
        if crn is not None and crn.ensemble is not None and crn.n_covered:
            covered = crn.covered
            slots = crn.position[covered]
            corrected = fc.derivative_variance(
                variance_block[minus][covered], variance_block[plus][covered],
                np.asarray(crn.ensemble[minus])[:, slots],
                np.asarray(crn.ensemble[plus])[:, slots],
                denominator, crn.n_ensembles,
            )
            plain = plain.copy()
            plain[covered] = corrected
        answer[:, index] = plain
    return answer


# ---------------------------------------------------------------------------
# One node's Fisher report
# ---------------------------------------------------------------------------

def fisher_report(J: np.ndarray, signal: np.ndarray, sigma, kio_ref: float,
                  variance: np.ndarray | None = None,
                  s0_mode: str = "fixed") -> dict:
    """Fisher matrix, spectrum, CRLB and geometry at one node.

    ``s0_mode`` follows the app's slice convention: "fixed" treats S0 as known
    (the library curve is the model), "free" marginalises it out with
    `fc.amplitude_marginal_fisher`, which is the same amplitude convention the
    free-S0 matcher uses.
    """
    J = np.asarray(J, dtype=float)
    if s0_mode == "free":
        marginal = fc.amplitude_marginal_fisher(J, np.asarray(signal, dtype=float),
                                                sigma, variance=variance)
        F = marginal["F_marginal_s0"]
        amplitude = marginal
    else:
        F = fc.fisher_matrix(J, sigma, variance)
        amplitude = None

    packed = fc.pack_fisher(F)[None, :]
    diagnostics = fc.fisher_diagnostics(F, float(kio_ref))
    geometry = fc.degeneracy_geometry(packed, float(kio_ref))
    inverse_diagonal, determinant, positive = fc.packed_inverse_diagonal(packed)
    profiled_block, profiled_valid = fc.rho_V_profiled_block(packed)
    return {
        "F": F,
        "packed": packed,
        "crlb": diagnostics["crlb"],
        "kappa": diagnostics["kappa"],
        "invertible": bool(diagnostics["invertible"]),
        "positive_definite": bool(positive[0]),
        "determinant": float(determinant[0]),
        "crlb_log_vi": float(fc.directional_crlb(packed)[0]),
        "eigenvalues": geometry["eigenvalues"][0],
        "eigenvectors": geometry["eigenvectors"][0],
        "condition_number": float(geometry["condition_number"][0]),
        "sloppy_vector": geometry["sloppy_vector"][0],
        "sloppy_angle_deg": float(geometry["sloppy_angle_deg"][0]),
        "sloppy_kio_fraction": float(geometry["sloppy_in_plane_fraction"][0]),
        "profiled_block": profiled_block[0],
        "profiled_valid": bool(profiled_valid[0]),
        "profiled_angle_deg": float(geometry["profiled_sloppy_angle_deg"][0]),
        "profiled_condition_number": float(geometry["profiled_condition_number"][0]),
        "amplitude": amplitude,
    }


def batch_quantities(J: np.ndarray, rows: np.ndarray, sigma, kios: np.ndarray,
                     kio_floor: float = 1.0) -> dict:
    """CRLB / conditioning / angle for a whole batch of nodes, for the hues.

    `sigma` may be a scalar or one value per column.  Only fixed-S0 is offered
    here: the free-S0 correction needs each node's own signal vector, which
    would turn a vectorised pass into a Python loop for no gain in the hue.
    """
    J = np.asarray(J, dtype=float)
    sigma_array = np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[1],))
    scaled = J / sigma_array[None, :, None]
    F = np.einsum("nci,ncj->nij", scaled, scaled)
    packed = fc.pack_fisher(F)
    reference_scale = np.maximum(np.asarray(kios, dtype=float)[rows], kio_floor)
    spectrum = fc.fisher_spectrum(packed, reference_scale)
    inverse_diagonal, _, positive = fc.packed_inverse_diagonal(packed)
    profiled = fc.rho_V_profiled_spectrum(packed)
    with np.errstate(invalid="ignore"):
        crlb = np.sqrt(inverse_diagonal)
    return {
        "rows": rows,
        "packed": packed,
        "crlb_log_rho": crlb[:, 0],
        "crlb_log_V": crlb[:, 1],
        "crlb_kio": crlb[:, 2],
        "crlb_log_vi": fc.directional_crlb(packed),
        "condition_number": spectrum["condition_number"],
        "smallest_eigenvalue": spectrum["eigenvalues"][:, 2],
        "positive_definite": positive,
        "profiled_angle_deg": profiled["sloppy_angle_deg"],
    }


# ---------------------------------------------------------------------------
# The ellipse: what the Fisher matrix predicts the slice should look like
# ---------------------------------------------------------------------------

def confidence_ellipse(report: dict, threshold: float, mode: str = "profiled",
                       n_points: int = 181) -> np.ndarray | None:
    """`{dtheta^T F_2 dtheta <= threshold}` in the `(log rho, log V)` plane.

    ``mode``:
      "profiled" - `k_io` estimated jointly (marginal).  The 2x2 precision is
          `fc.rho_V_profiled_block`, whose inverse is the `(log rho, log V)`
          block of `F^-1`.  This is the Phase-3 convention and the honest one
          when `k_io` is not known.
      "conditional" - `k_io` known exactly: invert the plain `(log rho, log V)`
          block of `F`.  Always the smaller ellipse.

    Returns (n_points, 2) offsets in (log rho, log V), or None when the matrix
    does not admit one.  Using the slice's own chi2 threshold makes this the
    quadratic prediction of exactly the region the slice draws.
    """
    if mode == "profiled":
        precision = np.asarray(report["profiled_block"], dtype=float)
        if not report["profiled_valid"] or not np.all(np.isfinite(precision)):
            return None
    else:
        precision = np.asarray(report["F"], dtype=float)[:2, :2]

    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        return None
    # Semi-axis along eigenvector i is sqrt(threshold / lambda_i).
    radii = np.sqrt(float(threshold) / eigenvalues)
    angle = np.linspace(0.0, 2.0 * np.pi, int(n_points))
    unit = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return (unit * radii[None, :]) @ eigenvectors.T


def eigenvector_segments(report: dict, threshold: float,
                         mode: str = "profiled") -> dict | None:
    """Sloppy and stiff axes of the same ellipse, as end-to-end offsets."""
    if mode == "profiled":
        precision = np.asarray(report["profiled_block"], dtype=float)
        if not report["profiled_valid"] or not np.all(np.isfinite(precision)):
            return None
    else:
        precision = np.asarray(report["F"], dtype=float)[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        return None
    order = np.argsort(eigenvalues)          # ascending: sloppiest first
    answer = {}
    for name, index in (("sloppy", order[0]), ("stiff", order[-1])):
        radius = np.sqrt(float(threshold) / eigenvalues[index])
        answer[name] = eigenvectors[:, index] * radius
    return answer


def crlb_versus_measured_count(J: np.ndarray, signal: np.ndarray, sigma,
                               kio_ref: float, variance: np.ndarray | None = None,
                               s0_mode: str = "fixed",
                               reduced_threshold: float = 1.0) -> list[dict]:
    """CRLB after each additional measured column, in the slice's column order.

    `predicted_rho_width` converts the bound into the same "max/min" units the
    empirical width curve uses: a `+/- sqrt(T) * CRLB` interval in `log rho` is
    a ratio of `exp(2 sqrt(T) CRLB)`, so the two curves can be read on one axis.
    """
    answer = []
    for k in range(1, J.shape[0] + 1):
        threshold = float(reduced_threshold) * k
        try:
            report = fisher_report(J[:k], np.asarray(signal)[:k], sigma,
                                   kio_ref,
                                   None if variance is None else variance[:k],
                                   s0_mode=s0_mode)
        except np.linalg.LinAlgError:
            report = None
        record = {"n_measured": k, "crlb_log_rho": float("nan"),
                  "crlb_log_V": float("nan"), "crlb_log_vi": float("nan"),
                  "predicted_rho_width": float("nan"),
                  "predicted_V_width": float("nan")}
        if report is not None and report["positive_definite"]:
            record["crlb_log_rho"] = float(report["crlb"][0])
            record["crlb_log_V"] = float(report["crlb"][1])
            record["crlb_log_vi"] = report["crlb_log_vi"]
            scale = 2.0 * np.sqrt(max(threshold, 0.0))
            # A barely determined node can give a bound so wide that exp()
            # overflows.  The grid only spans three decades in rho, so anything
            # past that is "unbounded" either way; cap it rather than return inf.
            def width(crlb):
                return float(np.exp(min(scale * float(crlb), 700.0)))
            record["predicted_rho_width"] = width(report["crlb"][0])
            record["predicted_V_width"] = width(report["crlb"][1])
        answer.append(record)
    return answer


def coverage(grid: NodeGrid, width: int = DEFAULT_WIDTH) -> dict:
    """How many nodes carry a complete central stencil, and why the rest do not."""
    complete = 0
    per_axis = {axis: 0 for axis in AXES}
    for node in grid.lookup:
        missing = [axis for axis in AXES if stencil(grid, node, axis, width) is None]
        if not missing:
            complete += 1
        for axis in missing:
            per_axis[axis] += 1
    total = len(grid.lookup)
    return {"complete": complete, "total": total,
            "fraction": complete / total if total else float("nan"),
            "missing_per_axis": per_axis, "width": int(width)}
