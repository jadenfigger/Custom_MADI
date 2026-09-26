"""Run on Windows: python -m pytest analysis/adaptive_protocol/test_workflow.py"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from .core import Evaluator, aggregate, canonical, diagnostics, noise, quantile
from .export import append_table
from .search import Design, ScoreStore
from madi.fisher_crlb import pack_fisher, stored_column_domain, require_columns


@pytest.fixture
def config():
    return json.loads(Path(__file__).with_name("config.json").read_text())


class ToySource:
    fingerprint = "synthetic-fixture-only"
    def __init__(self):
        self.columns = np.array([[4, 15, 500], [4, 20, 1000], [4, 40, 2000], [4, 15, 0]], float)
        self.gradient = np.zeros(4)
        self.kref = np.array([5., 20., 40.])
        self.signal = np.ones((4, 3))*.8
        self.j = np.broadcast_to(np.vstack([np.eye(3), np.zeros(3)])[:, None, :], (4,3,3)).copy()
    def get_columns(self, ids):
        J = self.j[ids]; S = self.signal[ids]
        return (pack_fisher(J[..., :, None]*J[..., None, :]),
                np.concatenate([J*S[..., None], S[..., None]**2], axis=-1),
                np.zeros_like(J), S)
    def specification(self, protocol):
        return [[*self.columns[c],n] for c,n in protocol]


def test_canonical_repetitions():
    assert canonical([(2,0),(1,2),(0,1),(1,3)]) == ((0,1),(1,5))
    for bad in (-1,.1,float("nan"),True):
        with pytest.raises(ValueError): canonical([(0,bad)])


def test_only_adaptive_common_noise(config):
    source = ToySource()
    te, sigma = noise([(0,2),(2,3)], source.columns, config["noise"])
    assert te == pytest.approx(.058)
    assert sigma == pytest.approx(np.exp(.058/.040)/50)
    assert noise([(0,200),(2,300)], source.columns, config["noise"])[0] == te
    with pytest.raises(ValueError): noise([(0,1)],source.columns,dict(config["noise"],T2_s=.08))


def test_complete_FIM_repetition_and_S0(config):
    evaluator=Evaluator(ToySource(),config)
    one=evaluator.evaluate([(c,1) for c in range(4)],True)
    two=evaluator.evaluate([(c,2) for c in range(4)],True)
    for label in one["models"]:
        a,b=one["models"][label],two["models"][label]
        assert a["positive"].all()
        np.testing.assert_allclose(b["F_native"],2*a["F_native"])
        np.testing.assert_allclose(b["Finv_native"],a["Finv_native"]/2)
    assert np.all(one["models"]["marginal_S0"]["crlb_sd"] >= one["models"]["fixed_S0"]["crlb_sd"])
    # Each single column is singular; only complete assembly identifies tissue.
    assert not evaluator.evaluate([(0,4)])["models"]["fixed_S0"]["positive"].any()


def test_missing_amplitude_cannot_be_fixed_S0(config):
    source=ToySource(); source.signal[:]=0
    answer=Evaluator(source,config).evaluate([(c,1) for c in range(3)],True)
    assert answer["models"]["fixed_S0"]["positive"].all()
    assert not answer["models"]["marginal_S0"]["positive"].any()
    assert np.isinf(answer["models"]["marginal_S0"]["crlb_sd"]).all()


def test_singular_indefinite_and_scaled_coordinates():
    F=np.array([np.diag([1.,2.,3.]),np.diag([1.,2.,0.]),np.diag([1.,2.,-1.])])
    d=diagnostics(pack_fisher(F),np.array([5.,20.,40.]),1e-12,np.ones(3),True)
    np.testing.assert_array_equal(d["positive"],[True,False,False])
    assert np.isnan(d["Finv_native"][1:]).all()
    assert np.isinf(d["relative_crlb_sd"][1:]).all()
    assert d["scaled_trace_crlb"][0] == pytest.approx(1+.5+1/75)
    assert d["native_trace_crlb"][0] == pytest.approx(1+.5+1/3)


def test_invalid_nodes_not_removed_from_objective(config):
    # Exactly half of an even domain valid is insufficient: median includes +inf.
    F=np.array([np.eye(3),np.eye(3),np.zeros((3,3)),np.zeros((3,3))])
    d=diagnostics(pack_fisher(F),np.ones(4)*5,1e-12,np.ones(4))
    a=aggregate(d,np.ones(4)*5,config["evaluation"])
    assert a["objective_value"] == np.inf and not a["feasible"]
    assert a["identifiable_fraction"] == .5
    assert quantile(np.array([1,2,np.inf,np.inf]),.5) == np.inf
    assert quantile(np.array([1,2,3,np.inf]),.5) == 2.5


def test_masks_count_budget_and_are_evaluation_only(config):
    source=ToySource(); source.signal[0]=.001
    config["evaluation"]["trust_floor"]=.01
    answer=Evaluator(source,config).evaluate([(c,1) for c in range(4)])
    assert np.all(answer["used_measurements"]==3)
    assert sum(n for _,n in answer["protocol"]) == 4
    assert len(source.columns)==4


def test_candidate_cache_settings_and_rerun(tmp_path, config):
    source=ToySource(); evaluator=Evaluator(source,config)
    class Valid:
        budget=4
        def reason(self,p): return ""
    path=tmp_path/"scores.sqlite"
    a=ScoreStore(path,evaluator,"same_run")
    protocol=[(c,1) for c in range(4)]
    result=a.evaluate(protocol,Valid(),"test")
    a.close()
    b=ScoreStore(path,evaluator,"same_run")
    repeated=b.evaluate(protocol,Valid(),"test")
    assert b.fresh==0 and b.cache_hits==1 and result["score"]==repeated["score"]
    assert b.db.execute("SELECT count(*) FROM history").fetchone()[0]==1
    b.close()
    changed=copy.deepcopy(config); changed["noise"]["SNR_ref"]=40
    assert evaluator.key(protocol) != Evaluator(source,changed).key(protocol)
    changed=copy.deepcopy(config); changed["design"]["G_max_T_m"]=.3
    assert evaluator.key(protocol) != Evaluator(source,changed).key(protocol)


def test_csv_safe_upsert(tmp_path):
    path=tmp_path/"test.csv"
    rows=[{"id":"same", "score":np.inf, "name":"=literal", "json":"[1,2]"}]
    assert append_table(path,rows,["id"])==1
    assert append_table(path,rows,["id"])==1
    assert append_table(path,[dict(rows[0],score=2)],["id"])==1
    with path.open(encoding="utf-8-sig",newline="") as f:
        saved=list(csv.DictReader(f))
    assert saved[0]["score"]=="2" and saved[0]["json"]=="[1,2]"


def test_column_identity_guard():
    domain=stored_column_domain(5)
    np.testing.assert_array_equal(require_columns(domain,[4,0],"test"),[4,0])


def test_figures_only_uses_saved_settings(monkeypatch, config):
    import sys
    import tempfile
    from . import run, figures
    output_parent=Path(__file__).parent/"outputs"
    output_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_parent) as directory:
        output=Path(directory).resolve()
        saved=copy.deepcopy(config); saved["output"]=str(output)
        edited=copy.deepcopy(saved); edited["noise"]["SNR_ref"]=40
        path=output/"config.json"; path.write_text(json.dumps(edited))
        (output/"run_report.json").write_text(json.dumps({"config":saved,"run_id":"recorded"}))
        captured=[]
        monkeypatch.setattr(figures,"make_figures",lambda folder,cfg,rid: captured.append((cfg,rid)))
        monkeypatch.setattr(sys,"argv",["run","--config",str(path),"--figures-only"])
        run.main()
        assert captured==[(saved,"recorded")]
