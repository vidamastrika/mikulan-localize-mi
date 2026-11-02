#!/usr/bin/env python3
"""Describe single-reference localization variation across six references.

Sample terminal inputs (from the repository root):
    python cross_subject_warp/scripts/08_reference_sensitivity.py --phase audit
    python cross_subject_warp/scripts/08_reference_sensitivity.py
    python cross_subject_warp/scripts/08_reference_sensitivity.py --output-root cross_subject_warp/outputs/08_reference_sensitivity_review

The evaluation reads completed CSVs only. It does not fit registrations,
reconstruct sources, or alter existing outputs. The six references change both
reference anatomy and reference-selected parameters, so these summaries are
descriptive and do not isolate an anatomy effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "primary_root": "cross_subject_warp/outputs/06_evaluate_localization_mapped",
    "sensitivity_root": "cross_subject_warp/outputs/06_evaluate_localization_qa_rt015",
    "no_warp_root": "cross_subject_no_warp/outputs/06_evaluate_localization",
    "historical": "cross_subject_warp/source/07_historical_baseline/13_selected_configuration_results_long.csv",
    "output_root": "cross_subject_warp/outputs/08_reference_sensitivity",
}
METHODS = 12
TEST_RUNS = 61
REFERENCES = 6
EXPECTED = METHODS * TEST_RUNS * REFERENCES
KEY = ["method", "test_subject", "run", "reference_subject", "registration_mode", "qa_policy"]
GROUP = ["qa_policy", "peak_scenario", "registration_mode", "method"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean_mm=np.nan, median_mm=np.nan, std_mm=np.nan,
                    q25_mm=np.nan, q75_mm=np.nan, iqr_mm=np.nan,
                    min_mm=np.nan, max_mm=np.nan)
    q25, q75 = x.quantile([.25, .75]).tolist()
    return dict(n=int(len(x)), mean_mm=float(x.mean()), median_mm=float(x.median()),
                std_mm=float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                q25_mm=float(q25), q75_mm=float(q75), iqr_mm=float(q75-q25),
                min_mm=float(x.min()), max_mm=float(x.max()))


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def read_reference_file(path: Path, peak_scenario: str, qa_policy: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = set(KEY[:-1] + ["strategy", "condition", "status", "evaluation_status",
                               "localization_error_mm", "localization_error_valid", "mapping_status",
                               "reconstruction_status"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    d = df.loc[(df.strategy == "Single-reference") & (df.condition == "cross_reference")].copy()
    if len(d) != 2 * EXPECTED:
        raise ValueError(f"{path}: expected {2*EXPECTED} mapped single-reference rows; got {len(d)}")
    d["qa_policy"] = qa_policy
    d["peak_scenario"] = peak_scenario
    d["input_csv"] = str(path)
    d["input_csv_sha256"] = sha256(path)
    d["localization_error_mm"] = pd.to_numeric(d.localization_error_mm, errors="coerce")
    d["localization_error_valid"] = truthy(d.localization_error_valid)
    if d[KEY].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError(f"{path}: null or duplicate reference/mode/policy keys")
    if not set(d.registration_mode.unique()).issubset({"affine", "syn"}):
        raise ValueError(f"{path}: unexpected registration modes")
    if d.method.nunique() != METHODS or d[["test_subject", "run"]].drop_duplicates().shape[0] != TEST_RUNS:
        raise ValueError(f"{path}: expected 12 methods and 61 test runs")
    if (d.reference_subject == d.test_subject).any():
        raise ValueError(f"{path}: self-reference rows found")
    expected_pairs = set()
    for mode, group in d.groupby("registration_mode"):
        if len(group) != EXPECTED:
            raise ValueError(f"{path}: {mode} expected {EXPECTED} rows; got {len(group)}")
        per_run = group.groupby(["method", "test_subject", "run"]).reference_subject.agg(["size", "nunique"])
        if len(per_run) != METHODS * TEST_RUNS or not (per_run["size"].eq(REFERENCES) & per_run["nunique"].eq(REFERENCES)).all():
            raise ValueError(f"{path}: {mode} does not have six distinct references for each method/test/run")
        expected_pairs.update(zip(group.reference_subject, group.test_subject))
        valid = d.loc[group.index, "localization_error_valid"]
        errors = d.loc[group.index, "localization_error_mm"]
        if (valid & errors.isna()).any() or ((~valid) & errors.notna()).any():
            raise ValueError(f"{path}: metric-valid flags and error values disagree")
    if len(expected_pairs) != 42:
        raise ValueError(f"{path}: expected 42 directed non-self subject pairs; got {len(expected_pairs)}")
    return d


def summarize_reference(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in d.groupby(GROUP + ["reference_subject"], sort=True, dropna=False):
        row = dict(zip(GROUP + ["reference_subject"], keys))
        valid = g.localization_error_valid
        row.update(scheduled=len(g), reconstruction_pass=int(g.reconstruction_status.eq("PASS").sum()),
                   mapped_valid=int(valid.sum()), mapped_invalid=int((~valid).sum()),
                   mapping_coverage=float(valid.mean()), **stats(g.loc[valid, "localization_error_mm"]))
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_per_run(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = GROUP + ["test_subject", "run"]
    for keys, g in d.groupby(by, sort=True, dropna=False):
        row = dict(zip(by, keys))
        valid = g.localization_error_valid
        s = stats(g.loc[valid, "localization_error_mm"])
        row.update(scheduled_references=REFERENCES, reference_count=g.reference_subject.nunique(),
                   reconstruction_pass=int(g.reconstruction_status.eq("PASS").sum()),
                   mapped_valid=int(valid.sum()), mapped_invalid=int((~valid).sum()),
                   mapping_coverage=float(valid.mean()),
                   range_mm=s["max_mm"]-s["min_mm"] if s["n"] else np.nan,
                   cv=s["std_mm"]/s["mean_mm"] if s["n"] > 1 and s["mean_mm"] != 0 else np.nan,
                   **s)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sensitivity(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in per_run.groupby(GROUP, sort=True, dropna=False):
        row = dict(zip(GROUP, keys))
        row.update(test_runs=len(g), scheduled=int(g.scheduled_references.sum()),
                   reconstruction_pass=int(g.reconstruction_pass.sum()),
                   mapped_valid=int(g.mapped_valid.sum()), mapped_invalid=int(g.mapped_invalid.sum()),
                   mapping_coverage=float(g.mapped_valid.sum()/g.scheduled_references.sum()),
                   complete_six_valid_runs=int(g.mapped_valid.eq(REFERENCES).sum()),
                   incomplete_valid_runs=int(g.mapped_valid.lt(REFERENCES).sum()),
                   median_across_reference_sd_mm=float(g.std_mm.median()) if g.std_mm.notna().any() else np.nan,
                   mean_across_reference_sd_mm=float(g.std_mm.mean()) if g.std_mm.notna().any() else np.nan,
                   median_across_reference_range_mm=float(g.range_mm.median()) if g.range_mm.notna().any() else np.nan,
                   **{f"test_run_median_{k}": v for k, v in stats(g.median_mm).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def pairing_matrix(d: pd.DataFrame) -> pd.DataFrame:
    by = GROUP + ["test_subject", "reference_subject"]
    rows = []
    for keys, g in d.groupby(by, sort=True, dropna=False):
        row = dict(zip(by, keys))
        valid = g.localization_error_valid
        row.update(scheduled_runs=len(g), reconstruction_pass=int(g.reconstruction_status.eq("PASS").sum()),
                   mapped_valid=int(valid.sum()), mapped_invalid=int((~valid).sum()),
                   mapping_coverage=float(valid.mean()), **stats(g.loc[valid, "localization_error_mm"]))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_peak_scenarios(d: pd.DataFrame) -> pd.DataFrame:
    match_keys = ["method", "test_subject", "run", "reference_subject", "registration_mode", "qa_policy"]
    a = d.loc[d.peak_scenario.eq("primary"), match_keys + ["localization_error_mm", "localization_error_valid", "reconstruction_status"]].rename(columns={
        "localization_error_mm": "primary_error_mm", "localization_error_valid": "primary_valid", "reconstruction_status": "primary_reconstruction_status"})
    b = d.loc[d.peak_scenario.eq("peak_sensitivity"), match_keys + ["localization_error_mm", "localization_error_valid", "reconstruction_status"]].rename(columns={
        "localization_error_mm": "sensitivity_error_mm", "localization_error_valid": "sensitivity_valid", "reconstruction_status": "sensitivity_reconstruction_status"})
    out = a.merge(b, on=match_keys, how="outer", validate="one_to_one", indicator=True)
    if not out._merge.eq("both").all():
        raise ValueError("Primary and peak-sensitivity observation keys differ")
    out["valid_both"] = truthy(out.primary_valid) & truthy(out.sensitivity_valid)
    out["sensitivity_minus_primary_mm"] = np.nan
    ok = out.valid_both
    out.loc[ok, "sensitivity_minus_primary_mm"] = out.loc[ok, "sensitivity_error_mm"] - out.loc[ok, "primary_error_mm"]
    out["status_changed"] = out.primary_reconstruction_status.ne(out.sensitivity_reconstruction_status)
    return out.drop(columns="_merge")


def within_context(no_warp_root: Path, historical_path: Path) -> pd.DataFrame:
    rows = []
    for filename, scenario in [("03_primary_localization.csv", "primary"), ("04_sensitivity_localization.csv", "peak_sensitivity")]:
        path = no_warp_root / filename
        d = pd.read_csv(path, low_memory=False)
        d = d.loc[d.condition.eq("within"), ["strategy", "test_subject", "run", "method", "status", "localization_error_mm"]].copy()
        d["context_source"] = "no_warp_within"
        d["scenario"] = scenario
        rows.append(d)
    historical = pd.read_csv(historical_path, low_memory=False).rename(columns={"subject": "test_subject"})
    historical = historical[["test_subject", "run", "method", "status", "localization_error_mm"]].copy()
    historical["strategy"] = "Historical-selected"
    historical["context_source"] = "historical_within"
    historical["scenario"] = "primary"
    rows.append(historical)
    d = pd.concat(rows, ignore_index=True, sort=False)
    d["localization_error_mm"] = pd.to_numeric(d.localization_error_mm, errors="coerce")
    d["error_valid"] = d.status.eq("PASS") & d.localization_error_mm.notna()
    return d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-root", default=DEFAULTS["primary_root"])
    p.add_argument("--sensitivity-root", default=DEFAULTS["sensitivity_root"])
    p.add_argument("--no-warp-root", default=DEFAULTS["no_warp_root"])
    p.add_argument("--historical", default=DEFAULTS["historical"])
    p.add_argument("--output-root", default=DEFAULTS["output_root"])
    p.add_argument("--phase", choices=["audit", "evaluate"], default="evaluate")
    a = p.parse_args(argv)
    primary_root = (ROOT / a.primary_root).resolve()
    sensitivity_root = (ROOT / a.sensitivity_root).resolve()
    no_warp_root = (ROOT / a.no_warp_root).resolve()
    historical = (ROOT / a.historical).resolve()
    output = (ROOT / a.output_root).resolve()
    primary_paths = [primary_root / "03_primary_localization.csv", primary_root / "04_sensitivity_localization.csv"]
    rt015_paths = [sensitivity_root / "03_primary_localization.csv", sensitivity_root / "04_sensitivity_localization.csv"]
    context_paths = [no_warp_root / "03_primary_localization.csv", no_warp_root / "04_sensitivity_localization.csv", historical]
    for path in primary_paths + rt015_paths + context_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if a.phase == "evaluate" and output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")

    pieces = []
    labels = [(primary_paths[0], "primary", "primary_rt010"),
              (primary_paths[1], "peak_sensitivity", "primary_rt010"),
              (rt015_paths[0], "primary", "pair_rt015_source_rt010"),
              (rt015_paths[1], "peak_sensitivity", "pair_rt015_source_rt010")]
    for path, peak, policy in labels:
        pieces.append(read_reference_file(path, peak, policy))
    all_rows = pd.concat(pieces, ignore_index=True, sort=False)
    # Distinct policies/scenarios must each retain the full scheduled denominator.
    schedule = all_rows.groupby(["qa_policy", "peak_scenario", "registration_mode"]).size()
    if len(schedule) != 8 or not schedule.eq(EXPECTED).all():
        raise ValueError(f"Unexpected schedule across policy/scenario/mode axes: {schedule.to_dict()}")
    if a.phase == "audit":
        print("AUDIT PASSED: 4,392 single-reference jobs x 2 modes x 2 peak scenarios x 2 QA policies")
        print("No localization summary files were written.")
        return 0

    ref_summary = summarize_reference(all_rows)
    per_run = sensitivity_per_run(all_rows)
    method_summary = summarize_sensitivity(per_run)
    pairing = pairing_matrix(all_rows)
    variant = compare_peak_scenarios(all_rows)
    baseline = within_context(no_warp_root, historical)
    report = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {str(x): sha256(x) for x in primary_paths + rt015_paths + context_paths},
        "scheduled_single_reference_jobs_per_scenario": EXPECTED,
        "reference_count_per_method_test_run": REFERENCES,
        "directed_nonself_subject_pairs": 42,
        "row_counts": {"all_reference_localizations": len(all_rows), "reference_method_summary": len(ref_summary),
                       "test_run_sensitivity": len(per_run), "method_sensitivity_summary": len(method_summary),
                       "reference_pairing_matrix": len(pairing), "primary_peak_sensitivity_pairs": len(variant)},
        "rules": {
            "valid_metric": "localization_error_valid from Script 06, with finite localization_error_mm",
            "primary_rt010_and_pair_rt015_source_rt010_are_separate": True,
            "registration_modes_are_separate": True,
            "peak_scenarios_are_matched_by_exact_method_test_run_reference_mode_policy_key": True,
            "invalid_rows_remain_in_scheduled_denominators": True,
            "reference_anatomy_and_reference_selected_parameters_both_vary": True,
            "interpretation": "descriptive reference-choice sensitivity, not an isolated anatomy effect or accuracy proof",
            "registration_fitting_executed": False,
            "inverse_reconstruction_executed": False,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "01_reference_run_level.csv": all_rows,
        "02_reference_method_summary.csv": ref_summary,
        "03_test_run_sensitivity.csv": per_run,
        "04_method_sensitivity_summary.csv": method_summary,
        "05_reference_pairing_matrix.csv": pairing,
        "06_primary_vs_sensitivity.csv": variant,
        "07_within_baseline_context.csv": baseline,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    (output / "08_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reference sensitivity tables written: {output}")
    print(f"Rows: {len(all_rows)}; matched primary/peak-sensitivity cases: {len(variant)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
