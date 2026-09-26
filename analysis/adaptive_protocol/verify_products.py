"""Read-only checks of the published tables, source immutability and workbook."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import REPO, PARAMS
from .export import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=REPO/"analysis/adaptive_protocol/outputs/windows_run")
    parser.add_argument("--check-sources",action="store_true")
    args=parser.parse_args(); out=args.output
    report=json.loads((out/"run_report.json").read_text())
    summary=pd.read_csv(out/"protocol_summary.csv")
    nodes=pd.read_csv(out/"node_metrics.csv",low_memory=False)
    columns=pd.read_csv(out/"acquisition_columns.csv")
    history=pd.read_csv(out/"optimization_history.csv")
    assert not summary.duplicated(["evaluation_key","amplitude_model"]).any()
    assert not nodes.duplicated(["evaluation_key","amplitude_model","node_index"]).any()
    assert not columns.duplicated(["evaluation_key","full_column"]).any()
    assert not history.duplicated(["run_id","evaluation_id"]).any()
    errors=[]
    for _, row in summary.iterrows():
        cols=columns[columns.evaluation_key==row.evaluation_key]
        n=nodes[(nodes.evaluation_key==row.evaluation_key)&(nodes.amplitude_model==row.amplitude_model)]
        assert cols.averages.sum()==row.total_measurements
        te=(cols.delta_ms+cols.Delta_ms).max()/1000+.014
        assert np.isclose(te,row.TE_A_s)
        assert np.allclose(cols.TE_A_s,te)
        assert np.allclose(cols.sigma_single,np.exp((te-row.TE_ref_s)/.040)/row.SNR_ref)
        assert len(n)==row.node_count and n.identifiable.sum()==row.identifiable_count
        relative=n[[f"relative_crlb_sd_{p}" for p in PARAMS]].replace({"+inf":np.inf}).astype(float)
        assert np.isclose(np.median(relative,axis=0).max(),row.robust_score)
        invalid=n[~n.identifiable]
        assert invalid.invalidity_reason.notna().all()
        assert invalid[[f"Finv_native_{a}__{b}" for a in PARAMS for b in PARAMS]].isna().all().all()
        valid=n[n.identifiable].iloc[::31]
        F=valid[[f"F_scaled_{a}__{b}" for a in PARAMS for b in PARAMS]].to_numpy(float).reshape(-1,3,3)
        C=valid[[f"Finv_scaled_{a}__{b}" for a in PARAMS for b in PARAMS]].to_numpy(float).reshape(-1,3,3)
        if len(valid):
            error=float(np.max(abs(F@C-np.eye(3)))); errors.append(error)
            assert error<2e-6,error
    source_checked=0
    if args.check_sources:
        for source in report["source"]["sources"]:
            p=Path(source["path"]); stat=p.stat()
            assert stat.st_size==source["bytes"] and stat.st_mtime_ns==source["mtime_ns"],p
            if "sha256" in source:
                assert hashlib.sha256(p.read_bytes()).hexdigest()==source["sha256"],p
            source_checked+=1
    figures=json.loads((out/"figure_manifest.json").read_text())["files"]
    assert figures and all((out/p).stat().st_size>0 for p in figures)
    for meta_path in (out/"figures").rglob("*.metadata.json"):
        meta=json.loads(meta_path.read_text())
        assert all(meta[k]==v for k,v in report["config"]["noise"].items())
    workbook=out/"protocol_statistics.xlsx"
    workbook_check="not_present_yet"
    if workbook.exists():
        import openpyxl
        w=openpyxl.load_workbook(workbook,read_only=True,data_only=True)
        for sheet in w:
            if sheet.max_row is None or sheet.max_column is None:
                # Native XLSX export legitimately omits optional dimension hints.
                sheet.calculate_dimension(force=True)
            frame=pd.read_csv(out/f"{sheet.title}.csv",low_memory=False)
            assert sheet.max_row==len(frame)+1,(sheet.title,sheet.max_row,len(frame))
            assert sheet.max_column==len(frame.columns)
        top=w["top_protocols"]
        headers=list(next(top.values)); first=list(top.values)[1]
        first_budget=report["config"]["budgets"][0]
        assert first[headers.index("budget")]==first_budget
        assert isinstance(first[headers.index("objective")],float)
        assert np.isclose(first[headers.index("objective")],report["budgets"][str(first_budget)]["objective_value"])
        w.close(); workbook_check="passed"
    audit=dict(status="passed",summary_rows=len(summary),node_rows=len(nodes),history_rows=len(history),
               unique_keys=True,budget_and_common_TE=True,robust_scores_recomputed=True,
               invalid_covariance_blank=True,max_sampled_scaled_inverse_residual=max(errors),
               source_files_unchanged=source_checked,figure_files=len(figures),workbook=workbook_check)
    write_json(out/"product_validation.json",audit)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
