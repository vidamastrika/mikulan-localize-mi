#!/usr/bin/env python3
"""Compare completed historical, no-warp, LOSO-warp and single-reference results.

Example (from the repository root):
    python cross_subject_warp/scripts/07_compare_strategies.py

Use --output cross_subject_warp/outputs/07_compare_strategies_review for
a review run. Existing output folders are refused. This stage reads completed
CSVs only; it does not reconstruct sources or fit registrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "historical": "cross_subject_warp/source/07_historical_baseline/13_selected_configuration_results_long.csv",
    "no_warp": "cross_subject_no_warp/outputs/06_evaluate_localization",
    "warp_primary": "cross_subject_warp/outputs/06_evaluate_localization_mapped",
    "warp_sensitivity": "cross_subject_warp/outputs/06_evaluate_localization_qa_rt015",
    "output": "cross_subject_warp/outputs/07_compare_strategies",
}
PAIR_KEYS = ["scenario", "strategy", "method", "reference_subject", "test_subject", "run", "condition"]
BASE_KEYS = ["test_subject", "run", "method"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_eval(folder: Path, scenario: str, source: str, policy: str = "not_applicable") -> pd.DataFrame:
    parts = []
    for artifact, peak_scenario in [("03_primary_localization.csv", "primary"),
                                    ("04_sensitivity_localization.csv", "peak_sensitivity")]:
        path = folder / artifact
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        frame["peak_scenario"] = peak_scenario
        frame["input_scenario"] = scenario
        frame["comparison_source"] = source
        if "qa_policy" not in frame:
            frame["qa_policy"] = policy
        parts.append(frame)
    return pd.concat(parts, ignore_index=True, sort=False)


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    require_columns(df, ["localization_error_mm", "localization_error_valid", "status"], "localization rows")
    rows = []
    for key, g in df.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        valid = g["localization_error_valid"].fillna(False).astype(bool) & pd.to_numeric(g["localization_error_mm"], errors="coerce").notna()
        err = pd.to_numeric(g.loc[valid, "localization_error_mm"], errors="coerce")
        row = dict(zip(keys, key))
        row.update(n_rows=len(g), n_valid=int(valid.sum()), n_invalid=int((~valid).sum()),
                   median_error_mm=float(err.median()) if len(err) else None,
                   mean_error_mm=float(err.mean()) if len(err) else None,
                   n_reconstruction_pass=int(g.get("reconstruction_status", g["status"]).eq("PASS").sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def matched_delta(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], left_name: str, right_name: str) -> pd.DataFrame:
    require_columns(left, keys + ["localization_error_mm", "localization_error_valid"], left_name)
    require_columns(right, keys + ["localization_error_mm", "localization_error_valid"], right_name)
    a = left[keys + ["localization_error_mm", "localization_error_valid"]].rename(columns={
        "localization_error_mm": f"{left_name}_error_mm", "localization_error_valid": f"{left_name}_valid"})
    b = right[keys + ["localization_error_mm", "localization_error_valid"]].rename(columns={
        "localization_error_mm": f"{right_name}_error_mm", "localization_error_valid": f"{right_name}_valid"})
    out = a.merge(b, on=keys, how="outer", validate="one_to_one", indicator=True)
    av = out[f"{left_name}_valid"].fillna(False).astype(bool)
    bv = out[f"{right_name}_valid"].fillna(False).astype(bool)
    out["matched_valid"] = out["_merge"].eq("both") & av & bv
    out[f"{right_name}_minus_{left_name}_mm"] = pd.NA
    valid = out["matched_valid"]
    out.loc[valid, f"{right_name}_minus_{left_name}_mm"] = (
        pd.to_numeric(out.loc[valid, f"{right_name}_error_mm"]) - pd.to_numeric(out.loc[valid, f"{left_name}_error_mm"]))
    out["comparison"] = f"{right_name} minus {left_name}"
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("historical", "no_warp", "warp_primary", "warp_sensitivity", "output"):
        p.add_argument("--" + key.replace("_", "-"), default=DEFAULTS[key])
    a = p.parse_args(argv)
    historical_path = (ROOT / a.historical).resolve()
    no_warp_root = (ROOT / a.no_warp).resolve()
    primary_root = (ROOT / a.warp_primary).resolve()
    sensitivity_root = (ROOT / a.warp_sensitivity).resolve()
    output = (ROOT / a.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for path in (historical_path, no_warp_root / "03_primary_localization.csv", primary_root / "03_primary_localization.csv", sensitivity_root / "03_primary_localization.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)

    hist = pd.read_csv(historical_path, low_memory=False)
    require_columns(hist, ["subject", "run", "method", "status", "localization_error_mm"], "historical baseline")
    hist = hist.rename(columns={"subject": "test_subject"})
    hist["strategy"] = "Historical-selected"
    hist["scenario"] = "primary"
    hist["peak_scenario"] = "primary"
    hist["comparison_source"] = "historical_baseline"
    hist["qa_policy"] = "not_applicable"
    hist["localization_error_valid"] = hist["status"].eq("PASS") & pd.to_numeric(hist["localization_error_mm"], errors="coerce").notna()
    hist["condition"] = "historical_within"
    hist["reference_subject"] = pd.NA
    hist["registration_mode"] = "none"

    nw = read_eval(no_warp_root, "no_warp", "no_warp")
    wp = read_eval(primary_root, "primary_rt010", "warp_primary", "primary_rt010")
    ws = read_eval(sensitivity_root, "pair_rt015_source_rt010", "warp_sensitivity", "pair_rt015_source_rt010")
    # The unregistered cross-subject head separation is deliberately retained as a
    # diagnostic in source files, but is not a localization-error comparison.
    for frame in (nw, wp, ws):
        require_columns(frame, ["strategy", "scenario", "test_subject", "reference_subject", "run", "method", "condition", "localization_error_mm", "localization_error_valid"], "evaluation input")
        frame["registration_mode"] = frame.get("registration_mode", "none")
    observations = pd.concat([hist, nw, wp, ws], ignore_index=True, sort=False)
    observations["metric_interpretation"] = observations.get("interpretation", pd.Series(index=observations.index, dtype=object))
    observations.loc[observations["comparison_source"].eq("no_warp") & observations["condition"].ne("within"), "localization_error_valid"] = False
    observations.loc[observations["comparison_source"].eq("no_warp") & observations["condition"].ne("within"), "metric_interpretation"] = "unregistered head-frame diagnostic; excluded from anatomical localization error"

    # Explicit matched comparisons preserve scenario, QA policy, and registration mode.
    historical_within = nw[(nw.comparison_source == "no_warp") & (nw.condition == "within") & (nw.peak_scenario == "primary")]
    hmatch = matched_delta(hist, historical_within, BASE_KEYS, "historical", "no_warp_within")
    hmatch["comparison_scope"] = "historical selected baseline vs no-warp within-subject"

    nw_within = nw[(nw.comparison_source == "no_warp") & (nw.condition == "within")]
    warp_within = observations[(observations.comparison_source.isin(["warp_primary", "warp_sensitivity"])) & (observations.condition == "within")]
    within_checks = []
    for (peak_scenario, qa_policy), group in warp_within.groupby(["peak_scenario", "qa_policy"], dropna=False):
        left = nw_within[nw_within.peak_scenario.eq(peak_scenario)]
        right = group
        # Within rows are unique by method/test/run; retain labels so no policies are pooled.
        match = matched_delta(left, right, BASE_KEYS, "no_warp_within", "warp_within")
        match["peak_scenario"] = peak_scenario
        match["qa_policy"] = qa_policy
        within_checks.append(match)
    within_checks = pd.concat(within_checks, ignore_index=True) if within_checks else pd.DataFrame()

    # LOSO vs single-reference is matched only where the reference subject is identical.
    cross = observations[(observations.condition == "cross_reference") & observations.comparison_source.isin(["warp_primary", "warp_sensitivity"])]
    loso = cross[cross.strategy.eq("LOSO")]
    single = cross[cross.strategy.eq("Single-reference")]
    # Strategy identifies the side of the comparison, so it cannot be part of
    # the join key. Match only the shared scientific observation dimensions.
    strategy_match_keys = [key for key in PAIR_KEYS if key != "strategy"]
    sr_match = matched_delta(loso, single, strategy_match_keys + ["registration_mode", "qa_policy", "peak_scenario"], "loso", "single_reference")
    sr_match["comparison_scope"] = "LOSO vs single-reference at identical test, run, method, reference, registration, QA and peak scenario"

    # Summary rows are grouped on all design axes; no averaging across modes/policies.
    summary_keys = ["comparison_source", "strategy", "scenario", "peak_scenario", "qa_policy", "condition", "registration_mode", "method"]
    summary = summarize(observations, summary_keys)
    output.mkdir(parents=True)
    files = {
        "01_localization_observations.csv": observations,
        "02_descriptive_summary.csv": summary,
        "03_historical_vs_no_warp_within.csv": hmatch,
        "04_no_warp_within_regression.csv": within_checks,
        "05_loso_vs_single_reference_matched.csv": sr_match,
    }
    for name, frame in files.items():
        frame.to_csv(output / name, index=False)
    manifest = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(path): sha256(path) for path in [historical_path,
            no_warp_root / "03_primary_localization.csv", no_warp_root / "04_sensitivity_localization.csv",
            primary_root / "03_primary_localization.csv", primary_root / "04_sensitivity_localization.csv",
            sensitivity_root / "03_primary_localization.csv", sensitivity_root / "04_sensitivity_localization.csv"]},
        "rows": {name: len(frame) for name, frame in files.items()},
        "rules": {
            "unregistered_cross_subject_head_separation_is_not_localization_error": True,
            "primary_rt010_failures_remain_invalid": True,
            "rt015_is_separate_sensitivity_policy": True,
            "affine_syn_and_peak_scenarios_are_not_pooled": True,
            "loso_single_reference_match_requires_identical_reference": True,
            "registration_fitting_executed": False,
            "inverse_reconstruction_executed": False,
        },
    }
    (output / "06_comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Comparison tables written: {output}")
    print(f"Observation rows: {len(observations)}; LOSO/single-reference matched rows: {len(sr_match)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
