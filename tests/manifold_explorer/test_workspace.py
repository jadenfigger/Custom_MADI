"""Scientific interchange and asynchronous workspace regression checks."""

from __future__ import annotations

import base64
import io
import json
import threading
import time

import numpy as np
import pytest

pytest.importorskip("dash", minversion="2.17")

from tools.manifold_explorer import analysis, axes, session, slice as slicing
from tools.manifold_explorer.app import create_app
from tools.manifold_explorer.runtime import ArrayCache, Jobs
from .test_manifold_explorer import make_tiny_library


def _components(root):
    if isinstance(root, (list, tuple)):
        for item in root:
            yield from _components(item)
    elif hasattr(root, "to_plotly_json"):
        yield root
        yield from _components(getattr(root, "children", None))


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "tiny.npz"
    vectors = make_tiny_library(path)
    app, data = create_app(path, prefer_cache=False, cache_mb=1)
    controls = {}
    wanted = {session.key(component): prop for component, prop in session.BINDINGS}
    for component in _components(app.layout()):
        key = session.key(component.id) if hasattr(component, "id") else None
        if key in wanted:
            controls[key] = getattr(component, wanted[key], None)
    controls["reference-row"] = int(data.eligible(.4, .99)[0])
    yield app, data, controls, vectors
    app.explorer_jobs.close()
    data.reader.close()


def _analysis(data, controls, **changes):
    kwargs = dict(measured=controls["measured-columns"],
        groups=[axes.pair_group(data.labels, 4, 20, []), axes.pair_group(data.labels, 4, 50, [])],
        method="mean", reference_row=controls["reference-row"], reference_mode="entry", pasted="",
        sigma=.02, reduced_threshold=1, s0_mode="fixed", options=["variance"], vi_min=.4, vi_max=.99)
    kwargs.update(changes)
    return analysis.compute(data, **kwargs)


def _callback(app, name, input_values, state_values=(), changed=None):
    key = next(key for key in app.callback_map if name in key)
    callback = app.callback_map[key]
    outputs = callback["output"]
    def output(out):
        return {"id": out.component_id, "property": out.component_property}
    payload = {"output": key,
        "outputs": [output(out) for out in outputs] if isinstance(outputs, list) else output(outputs),
        "inputs": [dict(spec, value=value) for spec, value in zip(callback["inputs"], input_values)],
        "state": [dict(spec, value=value) for spec, value in zip(callback["state"], state_values)],
        "changedPropIds": [changed or callback["inputs"][0]["id"] + "." + callback["inputs"][0]["property"]]}
    response = app.server.test_client().post("/_dash-update-component", json=payload)
    assert response.status_code == 200, response.data.decode()
    return response.get_json()["response"]


def _completed(app, data, controls):
    request = _callback(app, "analysis-request.data", [controls, 0], ["test-browser", None])["analysis-request"]["data"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status, result = app.explorer_jobs.poll("test-browser", request["token"])
        if status not in {"queued", "running"}:
            assert status == "complete", result
            assert "error" not in result, result
            break
        time.sleep(.01)
    else:
        pytest.fail("Analysis did not complete")
    polled = _callback(app, "analysis-complete.data", [request, 0], ["test-browser", None])
    assert polled["job-status"]["children"].startswith("Ready")
    assert polled["job-poll"]["disabled"]
    return request, polled["analysis-complete"]["data"], result


def test_session_roundtrip_and_library_identity(workspace):
    _, data, controls, _ = workspace
    library = session.identity(data)
    encoded = json.dumps(session.session_document(controls, library))
    assert session.read_session(encoded, library, data.labels) == controls
    foreign = dict(library, fingerprint="different")
    with pytest.raises(ValueError, match="different library"):
        session.read_session(encoded, foreign, data.labels)


@pytest.mark.parametrize("name,value", [
    ("sigma-exponent", float("nan")), ("measured-columns", [-1]),
    ("measured-columns", [0, 0]), ("reference-row", 100000),
    ("colour-by", []), ("vi-min", 1.2), ("group-b:0", [999]),
    ("point-budget", "all"), ("rho-max", 0), ("inspect-row", 1.5),
])
def test_session_rejects_invalid_controls(workspace, name, value):
    _, data, controls, _ = workspace
    controls[name] = value
    with pytest.raises(ValueError):
        session.validate_controls(controls, data.labels)


def test_signal_csv_roundtrip_preserves_acquisition_order(workspace):
    _, data, controls, _ = workspace
    snapshot = _analysis(data, controls, measured=[4, 2, 0])
    payload = analysis.export_reference_csv(data, snapshot).encode()
    columns, pasted = session.read_signal_csv(payload, data.labels)
    assert columns == [4, 2, 0]
    np.testing.assert_array_equal(session.parse_signal(pasted, 3), snapshot.result.reference_signal)
    by_coordinates = b"delta,Delta,b,signal\n4,50,500,0.9\n4,20,1000,0.8\n"
    assert session.read_signal_csv(by_coordinates, data.labels)[0] == [6, 2]


@pytest.mark.parametrize("payload", [b"column_id,signal\n1,nan", b"column_id,signal\n1,1\n1,2",
                                     b"column_id,signal\n900,1", b"signal\n1", b"column_id,signal\n"])
def test_signal_csv_rejects_ambiguous_or_nonfinite_data(workspace, payload):
    with pytest.raises(ValueError):
        session.read_signal_csv(payload, workspace[1].labels)


def test_bounded_upload():
    assert session.decode_upload("data:application/json;base64,e30=") == b"{}"
    with pytest.raises(ValueError):
        session.decode_upload("data:text/plain;base64,!")
    with pytest.raises(ValueError, match="2 MiB"):
        session.decode_upload("x" * (session.MAX_UPLOAD_BYTES * 2))


def test_numpy_export_is_complete_and_pickle_free(workspace):
    _, data, controls, vectors = workspace
    snapshot = _analysis(data, controls)
    document = session.session_document(controls, session.identity(data))
    with np.load(io.BytesIO(analysis.export_npz(data, snapshot, document)), allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["signal"], vectors[exported["row"]][:, exported["measured_columns"]])
        np.testing.assert_array_equal(exported["display"], snapshot.display)
        np.testing.assert_array_equal(exported["survives"], snapshot.result.survivors)
        assert json.loads(str(exported["session_json"])) == document
        assert all(exported[name].dtype != object for name in exported.files)
    assert len(analysis.export_csv(data, snapshot).splitlines()) == snapshot.result.n_survivors + 1
    assert len(analysis.export_csv(data, snapshot, False).splitlines()) == len(snapshot.result.eligible_rows) + 1


@pytest.mark.parametrize("mode", ["fixed", "free"])
def test_inspector_contributions_match_slice(workspace, mode):
    _, data, controls, _ = workspace
    snapshot = _analysis(data, controls, s0_mode=mode)
    for row in snapshot.result.eligible_rows[::9]:
        point = analysis.inspect(snapshot, row)
        assert point["chi2"] == pytest.approx(np.sum(point["standardized"] ** 2), abs=1e-9)
    with pytest.raises(ValueError, match="outside"):
        analysis.inspect(snapshot, -1)


@pytest.mark.parametrize("mode", ["fixed", "free"])
@pytest.mark.parametrize("use_variance", [False, True])
def test_linear_time_widths_match_independent_prefix_slices(workspace, mode, use_variance):
    _, data, controls, _ = workspace
    snapshot = _analysis(data, controls, s0_mode=mode, measured=list(range(15)))
    result = snapshot.result
    variance = snapshot.variance if use_variance else None
    curve = slicing.widths_versus_measured_count(snapshot.block, result.reference_signal,
        data.labels, result.eligible_rows, sigma_measurement=.037,
        variance_block=variance, s0_mode=mode, reduced_threshold=.8)
    for k, record in enumerate(curve, start=1):
        expected = slicing.slice_manifold(snapshot.block[:, :k], result.reference_signal[:k],
            data.labels, result.eligible_rows, .8 * k, .037,
            None if variance is None else variance[:, :k], mode)
        assert record["n_survivors"] == expected.n_survivors
        assert record["rho_log_width"] == pytest.approx(expected.parameter_stats["rho"]["log_width"])
        assert record["kio_std"] == pytest.approx(expected.parameter_stats["k_io"]["std"])


def test_acquisition_ranking_matches_direct_spread(workspace):
    _, data, controls, _ = workspace
    snapshot = _analysis(data, controls, measured=[0], sigma=1, options=[])
    ranking = analysis.rank_acquisitions(data, snapshot, list(range(5)))
    assert {r["column_id"] for r in ranking} == {1, 2, 3, 4}
    scores = [r["spread_noise"] for r in ranking]
    assert scores == sorted(scores, reverse=True)
    for record in ranking:
        expected = np.std(data.block([record["column_id"]])[snapshot.result.survivor_rows])
        assert record["spread_noise"] == pytest.approx(expected)


def test_plot_sampling_keeps_reference_and_both_populations():
    rows = np.arange(10000)
    survivors = rows % 7 == 0
    sample = analysis.sample_indices(rows, survivors, 100, reference_row=9001)
    assert len(sample) == 100
    assert 9001 in sample
    assert survivors[sample].any() and (~survivors[sample]).any()
    np.testing.assert_array_equal(sample, analysis.sample_indices(rows, survivors, 100, 9001))


def test_empty_filter_and_nonfinite_reference_are_actionable(workspace):
    _, data, controls, _ = workspace
    with pytest.raises(ValueError, match="No entries"):
        _analysis(data, controls, rho_max=1)
    with pytest.raises(ValueError, match="finite"):
        _analysis(data, controls, reference_mode="paste", pasted="nan " * 5)
    with pytest.raises(ValueError, match="No cellular"):
        data.nearest_entry(1, 1, 1, np.array([0]))


def test_cache_evicts_by_bytes_and_does_not_retain_oversized_arrays():
    cache = ArrayCache(32)
    first = cache.get("one", lambda: np.arange(4, dtype=np.float64))
    assert not first.flags.writeable
    cache.get("two", lambda: np.arange(4, dtype=np.float64))
    assert cache.bytes == 32
    assert cache.get("one", lambda: np.zeros(4))[0] == 0
    oversized = cache.get("huge", lambda: np.zeros(100))
    assert len(oversized) == 100 and cache.bytes == 32
    assert "huge" not in cache._items
    disabled = ArrayCache(0)
    disabled.get("empty", lambda: np.empty(0))
    assert not disabled._items
    for index in range(150):
        cache.get(index, lambda: np.empty(0))
    assert len(cache._items) <= 128


def test_latest_job_wins_and_browsers_are_isolated():
    jobs = Jobs(workers=1, max_sessions=3)
    entered, release = threading.Event(), threading.Event()
    def slow(checkpoint):
        entered.set()
        assert release.wait(3)
        checkpoint()
        return "obsolete"
    try:
        old = jobs.submit("browser-a", slow)
        assert entered.wait(3)
        new = jobs.submit("browser-a", lambda check: "new")
        other = jobs.submit("browser-b", lambda check: "other")
        release.set()
        jobs._jobs["browser-b"].future.result(timeout=3)
        assert jobs.poll("browser-a", old)[0] == "expired"
        assert jobs.poll("browser-a", new) == ("complete", "new")
        assert jobs.poll("browser-b", other) == ("complete", "other")
        assert jobs.poll("browser-b", new)[0] == "expired"
    finally:
        release.set()
        jobs.close()


def test_dash_background_result_export_and_error_recovery(workspace):
    app, data, controls, _ = workspace
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/assets/workspace.js").status_code == 200
    request, completed, result = _completed(app, data, controls)
    for format in ("survivors", "all", "reference", "npz", "html"):
        response = _callback(app, "result-download.data", [1],
            [format, "test-browser", request, completed])
        assert response["result-download"]["data"]["content"]
    controls["sigma-exponent"] = None
    invalid = _callback(app, "analysis-request.data", [controls, 0], ["test-browser", request])["analysis-request"]["data"]
    assert "error" in invalid
    assert app.explorer_jobs.poll("test-browser", request["token"])[0] == "expired"
    rejected = _callback(app, "result-download.data", [2], ["npz", "test-browser", invalid, completed])
    assert "result-download" not in rejected
    controls["sigma-exponent"] = -2
    _completed(app, data, controls)


def test_dash_session_import_is_atomic_and_signal_import_uses_file_order(workspace):
    app, data, controls, _ = workspace
    payload = json.dumps(session.session_document(controls, session.identity(data))).encode()
    upload = "data:application/json;base64," + base64.b64encode(payload).decode()
    imported = _callback(app, "session-download.data", [0, upload, None], [controls], "load-session.contents")
    assert imported["session-restore"]["data"]["controls"] == controls
    invalid = _callback(app, "session-download.data", [0, "bad", None], [controls], "load-session.contents")
    assert "session-restore" not in invalid
    csv = "data:text/csv;base64," + base64.b64encode(b"column_id,signal\n4,.8\n0,1\n").decode()
    imported = _callback(app, "session-download.data", [0, None, csv], [controls], "load-signal.contents")
    restored = imported["session-restore"]["data"]["controls"]
    assert restored["measured-columns"] == [4, 0]
    assert restored["reference-mode"] == "paste"
