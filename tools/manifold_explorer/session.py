"""Versioned exploration state and strict, bounded JSON/CSV interchange."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import zipfile

import numpy as np

SCHEMA = "madi-manifold-session-v1"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_COLUMNS = 2048

# Each binding is (component id, property). These are also the complete undo
# state; callbacks and browser history consume this single source of truth.
BINDINGS = [(name, "data" if name in {"measured-columns", "reference-row"} else "value")
            for name in (
                "measured-columns", "reference-row", "pick-delta", "pick-Delta", "pick-b",
                "collapse-method", "groups-as-measured", "reference-mode", "ref-rho",
                "ref-V", "ref-kio", "ref-paste", "sigma-exponent", "reduced-threshold",
                "s0-mode", "slice-options", "vi-min", "vi-max", "rho-max", "colour-by",
                "colour-scale", "fisher-width", "ellipse-mode", "fisher-options",
                "widths-toggle", "point-budget", "rank-options", "view", "inspect-row")]
BINDINGS += [({"type": kind, "index": index}, "value") for index in range(3)
             for kind in ("group-mode", "group-delta", "group-Delta", "group-b", "group-any")]


def key(component):
    return (f"{component['type']}:{component['index']}"
            if isinstance(component, dict) else component)


def identity(data):
    """Portable identity from row/acquisition labels and source vector checksum.

    Uses the archive's stored CRC, never a scan of the multi-GB signal matrix.
    It detects accidental library mismatches, not adversarial file tampering.
    """
    digest = hashlib.sha256()
    labels = data.labels
    for name in ("rhos", "Vs", "kios", "nominal_rhos", "nominal_Vs", "vis",
                 "pair_deltas", "pair_Deltas", "b_values", "is_free_water"):
        digest.update(np.ascontiguousarray(getattr(labels, name)).tobytes())
    with zipfile.ZipFile(data.library_path) as archive:
        members = {info.filename: (info.CRC, info.file_size) for info in archive.infolist()
                   if info.filename in {"vectors.npy", "signal_variance.npy"}}
    digest.update(json.dumps(members, sort_keys=True).encode())
    return {"name": data.library_path.name, "fingerprint": digest.hexdigest(),
            "n_entries": labels.n_entries, "n_columns": labels.n_columns}


def finite_number(value, name, minimum=None, maximum=None, nullable=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its allowed range [{minimum}, {maximum}].")


def column_list(value, labels, name):
    if not isinstance(value, list) or len(value) > MAX_COLUMNS:
        raise ValueError(f"{name} must contain at most {MAX_COLUMNS} columns.")
    if any(type(c) is not int or not 0 <= c < labels.n_columns for c in value):
        raise ValueError(f"{name} contains an invalid column id.")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate columns.")


def validate_controls(controls, labels):
    if not isinstance(controls, dict) or set(controls) != {key(c) for c, _ in BINDINGS}:
        raise ValueError("Session controls are missing or unsupported; use a v1 session from this explorer.")
    column_list(controls["measured-columns"], labels, "Measured columns")
    row = controls["reference-row"]
    if row is not None and (type(row) is not int or not 0 <= row < labels.n_entries):
        raise ValueError("Reference row is outside this library.")
    inspected = controls["inspect-row"]
    if inspected is not None and (type(inspected) is not int or not 0 <= inspected < labels.n_entries):
        raise ValueError("Inspected row must be an integer inside this library.")
    enums = {
        "collapse-method": {"mean", "adc"}, "reference-mode": {"entry", "paste"},
        "s0-mode": {"fixed", "free"}, "colour-scale": {"linear", "log"},
        "fisher-width": {"1", "2", "r"}, "ellipse-mode": {"profiled", "conditional"},
        "view": {"slice", "fisher", "widths", "inspect"},
        "colour-by": {"rho", "V", "k_io", "vi", "rmse", "chi", "crlb_log_rho",
                      "crlb_log_vi", "kappa", "lambda3", "profiled_angle"},
    }
    for name, allowed in enums.items():
        if not isinstance(controls[name], str) or controls[name] not in allowed:
            raise ValueError(f"Unsupported {name}.")
    for name, allowed in {"slice-options": {"variance", "free_water"},
                          "groups-as-measured": {"on"}, "widths-toggle": {"on"},
                          "rank-options": {"on"},
                          "fisher-options": {"debias", "axes", "overlay"}}.items():
        value = controls[name]
        if not isinstance(value, list) or any(not isinstance(v, str) or v not in allowed for v in value):
            raise ValueError(f"Unsupported {name}.")
    for name, low, high in (("sigma-exponent", -6, 0), ("reduced-threshold", 0, 10),
                            ("vi-min", 0, 1), ("vi-max", 0, 1),
                            ("point-budget", 100, 1000000)):
        finite_number(controls[name], name, low, high)
    if controls["vi-min"] > controls["vi-max"]:
        raise ValueError("v_i minimum exceeds its maximum.")
    for name in ("ref-rho", "ref-V", "ref-kio", "rho-max"):
        finite_number(controls[name], name, 0, nullable=True)
    if controls["rho-max"] == 0:
        raise ValueError("Maximum rho must be positive or empty.")
    if not isinstance(controls["ref-paste"], str) or len(controls["ref-paste"]) > 100000:
        raise ValueError("Pasted signal text is too large or invalid.")
    for prefix in ("pick", "group:0", "group:1", "group:2"):
        def field(suffix):
            return f"pick-{suffix}" if prefix == "pick" else f"group-{suffix}:{prefix[-1]}"
        delta, Delta = controls[field("delta")], controls[field("Delta")]
        finite_number(delta, "delta")
        finite_number(Delta, "Delta")
        try:
            labels.pair_index(delta, Delta)
            values = controls[field("b")]
            if not isinstance(values, list) or len(values) > labels.n_b:
                raise ValueError("Invalid b-value list.")
            for b in values:
                finite_number(b, "b")
                labels.b_index(b)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if prefix != "pick":
            mode = controls[field("mode")]
            if mode not in ("off", "pair", "any"):
                raise ValueError("Invalid display group mode.")
            column_list(controls[field("any")], labels, "Display group")
    if controls["reference-mode"] == "paste" and controls["measured-columns"]:
        parse_signal(controls["ref-paste"], len(controls["measured-columns"]))
    return controls


def parse_signal(text, count):
    try:
        values = np.asarray([float(v) for v in text.replace(",", " ").split()])
    except (ValueError, AttributeError) as exc:
        raise ValueError("Paste finite numeric S/S0 values separated by commas or spaces.") from exc
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Provide exactly {count} finite S/S0 values in measured-column order.")
    return values


def session_document(controls, library):
    return {"schema": SCHEMA, "library": library, "controls": controls}


def read_session(payload, library, labels):
    try:
        record = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("The session is not valid JSON.") from exc
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("Unsupported session format.")
    source = record.get("library")
    if not isinstance(source, dict) or source.get("fingerprint") != library["fingerprint"]:
        raise ValueError("This session belongs to a different library; row and column ids cannot be reused.")
    return validate_controls(record.get("controls"), labels)


def decode_upload(contents):
    if not isinstance(contents, str) or len(contents) > 4 * MAX_UPLOAD_BYTES // 3 + 1024:
        raise ValueError("Upload exceeds the 2 MiB limit.")
    try:
        header, encoded = contents.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("Expected a base64 upload.")
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid upload encoding.") from exc
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the 2 MiB limit.")
    return payload


def read_signal_csv(payload, labels):
    """Import column_id,signal or delta,Delta,b,signal; preserve file order."""
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        fields = set(reader.fieldnames or [])
        if "signal" not in fields or not ({"column_id"} <= fields or {"delta", "Delta", "b"} <= fields):
            raise ValueError("CSV needs column_id,signal or delta,Delta,b,signal headers.")
        columns, values = [], []
        for record in reader:
            column = (int(record["column_id"]) if "column_id" in fields else
                      labels.column_index(*(float(record[k]) for k in ("delta", "Delta", "b"))))
            columns.append(column)
            values.append(float(record["signal"]))
            if len(columns) > MAX_COLUMNS:
                raise ValueError(f"Signal CSV exceeds {MAX_COLUMNS} columns.")
        column_list(columns, labels, "Signal CSV")
        if not columns or not np.all(np.isfinite(values)):
            raise ValueError("Signal CSV must contain finite signal values.")
        return columns, " ".join(format(v, ".17g") for v in values)
    except (UnicodeError, KeyError, TypeError, csv.Error) as exc:
        raise ValueError("Invalid signal CSV. Check headers, numeric values and acquisition coordinates.") from exc
