"""Collapse a group of acquisition columns into one plot axis.

Why groups
----------
A single column is one number per entry: S/S0 at one (delta, Delta, b).  Two or
three of those make a scatter, but each axis then carries only one b-value's
worth of information.  A GROUP is a set of columns collapsed to one number, so
an axis can be "the whole decay curve at Delta = 20 ms, summarised".

Two ways to summarise, both display-only -- neither changes which entries
survive a slice:

``mean``
    A_g = mean over the group's columns of S/S0.  A linear projection onto
    w = (1/n)(1, ..., 1), still in S/S0 units.  Works for any set of columns.

``adc``
    The ordinary least-squares slope of ln S against b inside the group,
    negated, i.e. the apparent diffusion coefficient the group's points would
    give a mono-exponential fit.  Only meaningful within ONE (delta, Delta)
    pair -- b is the regressor, so mixing diffusion times would regress across
    two different curves -- and needs at least two distinct b-values.

A group holding a single column collapses (under ``mean``) to that column's
value exactly, so the older one-column-per-axis behaviour is a special case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# b is stored in s/mm^2.  1 s/mm^2 = 1e-3 ms/um^2, so a slope taken against the
# stored b is in um^2/ms once multiplied by 1/1e-3 = 1e3.
B_S_MM2_TO_MS_UM2 = 1e-3

COLLAPSE_METHODS = {"mean": "Mean signal", "adc": "ADC"}


@dataclass(frozen=True)
class AxisGroup:
    """One plot axis: a set of columns plus how they were chosen.

    ``kind`` is "pair" for one (delta, Delta) with a set of b-values, or "any"
    for an arbitrary set of columns.  Only "pair" groups can produce an ADC.
    """

    columns: np.ndarray
    kind: str
    delta: float | None = None
    Delta: float | None = None
    b_values: np.ndarray | None = None

    @property
    def n_columns(self) -> int:
        return int(len(self.columns))

    @property
    def supports_adc(self) -> bool:
        """ADC needs one diffusion time and at least two distinct b-values."""
        if self.kind != "pair" or self.b_values is None:
            return False
        return int(np.unique(self.b_values).size) >= 2

    def adc_refusal(self) -> str:
        """Why ADC is unavailable, in words, for the UI to show."""
        if self.kind != "pair":
            return "an 'any columns' group spans several diffusion times"
        if self.b_values is None or np.unique(self.b_values).size < 2:
            return "a single b-value cannot give a slope"
        return ""

    def _b_text(self) -> str:
        if self.b_values is None or len(self.b_values) == 0:
            return f"{self.n_columns} columns"
        if len(self.b_values) == 1:
            return f"b={self.b_values[0]:g}"
        return f"b={min(self.b_values):g}-{max(self.b_values):g}"

    def label(self, method: str) -> str:
        """Axis title, e.g. 'mean S/S0, d=4 D=20, b=1000-4000'."""
        if self.kind != "pair":
            where = f"{self.n_columns} columns"
        else:
            where = f"d={self.delta:g} D={self.Delta:g}, {self._b_text()}"
        if method == "adc":
            return f"ADC (um^2/ms), {where}"
        return f"mean S/S0, {where}"


def pair_group(labels, delta: float, Delta: float, b_values) -> AxisGroup:
    """A group of b-values at one stored (delta, Delta)."""
    chosen = np.sort(np.asarray(b_values, dtype=float))
    if chosen.size == 0:
        chosen = np.asarray(labels.b_values, dtype=float)
    columns = np.array([labels.column_index(delta, Delta, b) for b in chosen],
                       dtype=np.int64)
    return AxisGroup(columns=columns, kind="pair", delta=float(delta),
                     Delta=float(Delta), b_values=chosen)


def any_group(columns) -> AxisGroup:
    """An arbitrary set of columns, possibly spanning several pairs."""
    return AxisGroup(columns=np.asarray(sorted(set(int(c) for c in columns)),
                                        dtype=np.int64), kind="any")


def collapse_mean(block: np.ndarray) -> np.ndarray:
    """Mean of the group's columns, per entry.  (n_entries, n_cols) -> (n_entries,)."""
    return np.asarray(block, dtype=float).mean(axis=1)


def collapse_adc(block: np.ndarray, b_values) -> tuple[np.ndarray, np.ndarray]:
    """Apparent diffusion coefficient per entry, in um^2/ms.

    Closed-form least-squares slope of ln S against b, vectorised over entries:

        D = - sum_b (b - b_mean)(ln S_b - mean ln S) / sum_b (b - b_mean)^2

    Since the b deviations sum to zero, centring ln S is unnecessary and the
    numerator is just the dot product with the centred b.

    Signals are NOT clipped.  An entry with S <= 0 anywhere in the group (which
    happens at high b, where the Monte-Carlo mean can go slightly negative) has
    no logarithm, so it is reported as invalid and the caller drops it.

    Returns
    -------
    values : (n_entries,) float, NaN where invalid
    valid : (n_entries,) bool
    """
    block = np.asarray(block, dtype=float)
    b = np.asarray(b_values, dtype=float)
    if block.shape[1] != b.size:
        raise ValueError(f"block has {block.shape[1]} columns but {b.size} b-values")
    if np.unique(b).size < 2:
        raise ValueError("ADC needs at least two distinct b-values")

    valid = np.all(block > 0.0, axis=1) & np.all(np.isfinite(block), axis=1)
    centred_b = b - b.mean()
    denominator = float(np.sum(centred_b ** 2))

    values = np.full(block.shape[0], np.nan)
    if np.any(valid):
        log_signal = np.log(block[valid])
        slope = (log_signal @ centred_b) / denominator
        # slope is per (s/mm^2); dividing by the unit conversion puts it in um^2/ms.
        values[valid] = -slope / B_S_MM2_TO_MS_UM2
    return values, valid


def collapse(block: np.ndarray, group: AxisGroup,
             method: str) -> tuple[np.ndarray, np.ndarray]:
    """Collapse one group's block by ``method``; returns (values, valid mask)."""
    if method == "adc":
        if not group.supports_adc:
            raise ValueError(f"ADC unavailable: {group.adc_refusal()}")
        return collapse_adc(block, group.b_values)
    values = collapse_mean(block)
    return values, np.isfinite(values)


def collapse_reference(reference_by_column: dict[int, float], group: AxisGroup,
                       method: str) -> float | None:
    """Collapse a reference point given as {column index: S/S0}.

    Returns None when the reference does not cover every column of the group --
    a pasted reference only covers the measured columns, which need not include
    an axis group's columns.
    """
    try:
        values = np.array([reference_by_column[int(column)]
                           for column in group.columns], dtype=float)
    except KeyError:
        return None
    collapsed, valid = collapse(values[None, :], group, method)
    return float(collapsed[0]) if bool(valid[0]) else None
