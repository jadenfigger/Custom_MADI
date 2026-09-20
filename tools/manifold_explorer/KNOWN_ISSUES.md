# Known issues in existing code that this tool deliberately avoids

Neither issue is fixed here — existing code is left untouched by decision. This
file records what they are, what they cost, and the fix that was proposed, so
the decision is recoverable later. The explorer routes around both: it never
calls `load_library_meta`, never reads `entry_metadata_json`, and reads only
the small label members plus the columns it actually needs.

Measured on the production artifact
`data/libraries/madi_dense_universal_remediated.npz` (15.24 GB, 18,820 x 31,125),
on a machine with 15.4 GB RAM (about 1.9 GB free at the time) and an NVMe disk
delivering 1.53 GB/s sequential.

---

## 1. `mmap_mode="r"` is silently ignored for `.npz`, so a "lazy handle" is a full read

**Where:** `analysis/plot_signal_decay_identifiability.py:96` and `:122`

```python
data = np.load(path, mmap_mode="r")          # line 96
...
vectors = data["vectors"]                    # line 122: "lazy memmap handle — not yet read"
signal = np.asarray(vectors[:, cols])        # line 123: "only these 13 columns are pulled from disk"
```

**What goes wrong.** `np.load` only honours `mmap_mode` for a bare `.npy`. For a
zip archive it returns an `NpzFile` and the argument is dropped without a
warning — visible in `numpy.lib._npyio_impl.load`, where the `if mmap_mode:`
branch that calls `format.open_memmap` is reached only after the zip branch has
already returned. So `data["vectors"]` is not a handle: it decodes the entire
`vectors.npy` member into a new in-memory array, and the column selection on
the next line happens afterwards, on that full array.

**Measured cost.** The `vectors` member is 4.69 GB (float64, 18,820 x 31,125).
The script therefore needs 4.69 GB of RAM at that line to obtain 13 columns
(1.9 MB). On this machine, with ~1.9 GB free, that allocation will page heavily
or fail outright. The comments at lines 122–123 state the opposite of what
happens, which is the part most likely to mislead a future reader.

**Proposed fix.** Replace those two lines with the explorer's column reader,
which memory-maps the member in place (legal: every member of the production
artifact is `ZIP_STORED`) and pulls only the requested columns:

```python
from tools.manifold_explorer.columns import load_labels, open_reader
labels = load_labels(path)
signal = open_reader(path).read(cols)        # (18820, len(cols)), same values
```

Identical returned values and identical function signature; peak memory drops
from 4.69 GB to the size of the requested block.

---

## 2. `load_library_meta` reads 5.3 GB of arrays to record three `.shape` tuples

**Where:** `madi/library.py:1096-1098` (inside `load_library_meta`)

```python
meta['signal_imag_shape'] = tuple(data['signal_imag'].shape)
meta['signal_variance_shape'] = tuple(data['signal_variance'].shape)
meta['ensemble_means_subset_shape'] = tuple(data['ensemble_means_subset'].shape)
```

**What goes wrong.** `data` is an `NpzFile`, and every `data[key]` subscript
decodes that whole member from the archive — the module's own comment in
`load_library` (line 981) says as much. Three subscripts taken purely for their
`.shape` therefore read `signal_imag` (2.34 GB), `signal_variance` (2.34 GB) and
`ensemble_means_subset` (0.60 GB): **5.28 GB of I/O per call**. Peak RSS stays
low only because each array is freed immediately after its shape is taken, so
this shows up as latency, not as an out-of-memory failure.

**Measured cost.** 5.3 s per call on this machine (5.28 GB at ~1.5 GB/s, warm or
cold), against ~0.05 s for the same information read from the NPY headers. Every
caller pays it: `scripts/fit_data.py`, `scripts/edema_summary/prepare_protocol_slim_library.py`,
and anything else that just wants `delta_pairs` / `b_values` / `n_b`.

**Proposed fix.** Use the header reader that already exists in this repository,
`madi.fisher_crlb.npz_array_info` (`madi/fisher_crlb.py:107`), which parses the
NPY header of a member and returns `(shape, fortran_order, dtype)` without
touching the data:

```python
from .fisher_crlb import npz_array_info
meta['signal_imag_shape'] = npz_array_info(path, 'signal_imag')[0]
meta['signal_variance_shape'] = npz_array_info(path, 'signal_variance')[0]
meta['ensemble_means_subset_shape'] = npz_array_info(path, 'ensemble_means_subset')[0]
```

Same dict, same values, same types; ~100x faster. (Note the import direction:
`fisher_crlb` already imports from `library`, so the helper would have to move
into `library` or into a shared low-level module rather than being imported
back the other way.)

---

## 3. Not a defect, but worth knowing: `entry_metadata_json` is 5.27 GB

`entry_metadata_json` is a `(18820,) <U69952` array — 279,808 bytes of UCS-4 per
entry, **5.27 GB, or 35% of the whole 15.24 GB file**. `load_library()` reads it
in full to populate `LibraryEntry.metadata`, which is why loading the library
whole is so expensive. The explorer never touches it; all row labels come from
the small `kios` / `rhos` / `Vs` / `vis` / `nominal_*` / `weights` /
`is_free_water` arrays, which total about 1.3 MB.
---

## 4. Plotly 6 quirks this tool had to work around

Not defects in project code, but they cost real debugging time and will bite
again if someone edits `app.py`.

**Numpy `customdata` never reaches click events.** Plotly 6 serialises numpy
arrays as `{"dtype": ..., "bdata": <base64>}`. plotly.js decodes that for
coordinates, but it does not expand it into per-point `customdata`, so a click
arrives with `customdata: null` and the "click a point to re-centre the slice"
callback silently does nothing. Fixed by `_row_ids()`, which hands Plotly a
plain list of ints. Anything added to `customdata` must go through it.

**An empty `options` list nulls a Dropdown's value.** A `dcc.Dropdown` served
with `value="k_io"` but `options=[]`, intending a callback to fill the options,
renders blank and reports `None` to every callback that reads it. This bit
twice: once on the per-slot `Delta` dropdown and once on `colour-by`. Serve the
full option list in the layout and use callbacks only to change `disabled`.

**A near-collinear 3-D scatter disappears at the default camera.** With grouped
axes, three mean-S/S0 values at nearby diffusion times are almost perfectly
correlated, so all 18,819 points lie close to the cube's main diagonal. The
default Plotly camera (`eye = 1.25, 1.25, 1.25`) looks straight down that
diagonal and foreshortens the whole manifold into a streak a few pixels long,
which reads as an empty plot. Diagnosed by rendering the identical data to a
standalone HTML file, where the streak is visible. The scene now sets an
explicit off-diagonal camera and larger faint markers. If a future axis choice
looks empty in 3-D, check the camera before suspecting the data.
