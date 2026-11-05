#!/usr/bin/env python3
# ============================================================
# Script 01: Prepare fsaverage Experiment
# ============================================================
# Purpose: Freeze schedules and parameter selections; audit all five anatomical
#   conditions and fsaverage resources. No reconstruction or registration.
# Inputs: no-warp preparation/selection/evaluation outputs; historical selected
#   baseline; mapped affine/SyN outputs; fsaverage BEM/source/fiducials.
# Outputs: cross_subject_fsaverage/outputs/01_prepare_experiment/ with frozen
#   configuration copies, schedule, audits, provenance manifest, and README.
# Data: seven Localize-MI subjects, 61 EEG runs, 12 inverse methods.
# Parameters/defaults: --output-root defaults to the numbered folder above;
#   --fsaverage-root defaults to ~/mne_data/MNE-fsaverage-data/fsaverage.
# Reused calculations: historical selections and results are read, never fit.
# Expected cases: 732 baseline; 732 fsaverage LOSO; 4,392 fsaverage Single
#   Reference; 5,124 new fsaverage evaluations planned, not guaranteed PASS.
# Example from repository root with the existing virtual environment active:
#   .venv/bin/python cross_subject_fsaverage/scripts/01_prepare_experiment.py \
#       --output-root cross_subject_fsaverage/outputs/01_prepare_experiment
# ============================================================
"""Audit and freeze the fsaverage experiment inputs without running models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["MNE", "dSPM", "sLORETA", "eLORETA", "Continuous-ECD", "RAP-MUSIC",
           "LCMV", "MxNE", "irMxNE", "Gamma-MAP", "Champagne", "SESAME"]
METHOD_META = {
    "MNE": ("Minimum-norm", 1994), "dSPM": ("Minimum-norm", 2002),
    "sLORETA": ("Minimum-norm", 2002), "eLORETA": ("Minimum-norm", 2011),
    "Continuous-ECD": ("Dipole / Beamformer", 1990),
    "RAP-MUSIC": ("Dipole / Beamformer", 2001), "LCMV": ("Dipole / Beamformer", 1997),
    "MxNE": ("Sparse", 2011), "irMxNE": ("Sparse", 2018),
    "Gamma-MAP": ("Bayesian", 2008), "Champagne": ("Bayesian", 2016),
    "SESAME": ("SESAME", 2020),
}
NW = Path("cross_subject_no_warp")
WARP = Path("cross_subject_warp")
INPUTS = {
    "loso_folds": NW / "outputs/01_prepare_experiment/01_loso_folds.csv",
    "test_runs": NW / "outputs/01_prepare_experiment/03_single_reference_cases.csv",
    "loso_selection": NW / "outputs/02_select_loso_parameters/01_loso_selected_parameters.csv",
    "loso_selection_manifest": NW / "outputs/02_select_loso_parameters/07_selection_manifest.json",
    "single_selection": NW / "outputs/04_single_reference_parameters/01_single_reference_selected_parameters.csv",
    "single_selection_manifest": NW / "outputs/04_single_reference_parameters/07_selection_manifest.json",
    "single_schedule": NW / "outputs/04_single_reference_parameters/09_reference_test_schedule.csv",
    "baseline_definitions": NW / "source/07_historical_baseline/01_selected_configuration_definitions.csv",
    "baseline_results": NW / "source/07_historical_baseline/13_selected_configuration_results_long.csv",
    "no_warp_primary": NW / "outputs/06_evaluate_localization/03_primary_localization.csv",
    "no_warp_manifest": NW / "outputs/06_evaluate_localization/07_evaluation_manifest.json",
    "no_warp_sensitivity": NW / "outputs/06_evaluate_localization/04_sensitivity_localization.csv",
    "warp_loso_selection": WARP / "outputs/02_select_loso_parameters/01_loso_selected_parameters.csv",
    "warp_single_selection": WARP / "outputs/04_single_reference_parameters/01_single_reference_selected_parameters.csv",
    "warp_mapped_primary": WARP / "outputs/06_evaluate_localization_mapped/03_primary_localization.csv",
    "warp_mapped_manifest": WARP / "outputs/06_evaluate_localization_mapped/07_evaluation_manifest.json",
    "warp_mapped_sensitivity": WARP / "outputs/06_evaluate_localization_mapped/04_sensitivity_localization.csv",
    "warp_pair_qa": WARP / "outputs/06_evaluate_localization_mapped/09_registration_readiness.csv",
}
REQUIRED = set(INPUTS) - {"no_warp_sensitivity", "warp_loso_selection", "warp_single_selection",
                          "warp_mapped_primary", "warp_mapped_manifest", "warp_mapped_sensitivity", "warp_pair_qa"}
SNAPSHOT = ["loso_folds", "test_runs", "loso_selection", "loso_selection_manifest",
            "single_selection", "single_selection_manifest", "single_schedule", "baseline_definitions"]
FS_FILES = {
    "source_space": "bem/fsaverage-ico-5-src.fif",
    "bem_solution": "bem/fsaverage-5120-5120-5120-bem-sol.fif",
    "fiducials": "bem/fsaverage-fiducials.fif",
    "head_surface": "bem/outer_skin.surf",
    "built_in_transform": "bem/fsaverage-trans.fif",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(name: str, paths: dict[str, Path]) -> pd.DataFrame:
    return pd.read_csv(paths[name], low_memory=False, dtype={"run": str})


def exact_keys(frame: pd.DataFrame, columns: list[str], expected: set[tuple], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame.duplicated(columns).any():
        raise ValueError(f"{label}: duplicate keys {columns}")
    actual = set(map(tuple, frame[columns].itertuples(index=False, name=None)))
    if actual != expected or len(frame) != len(expected):
        raise ValueError(f"{label}: expected {len(expected)} exact keys; found {len(frame)}; "
                         f"missing {len(expected-actual)}, extra {len(actual-expected)}")


def check_status(frame: pd.DataFrame, label: str) -> dict[str, int]:
    if "status" not in frame or not frame.status.isin(["PASS", "FAIL"]).all():
        raise ValueError(f"{label}: missing or invalid PASS/FAIL status")
    return {"PASS": int(frame.status.eq("PASS").sum()), "FAIL": int(frame.status.eq("FAIL").sum())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path("cross_subject_fsaverage/outputs/01_prepare_experiment"),
                        help="New output directory; existing directories are never overwritten")
    parser.add_argument("--fsaverage-root", type=Path,
                        default=Path.home() / "mne_data/MNE-fsaverage-data/fsaverage",
                        help="Local fsaverage subject directory (no download is attempted)")
    args = parser.parse_args(argv)
    out = (ROOT / args.output_root).resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {out}")
    paths = {name: ROOT / relative for name, relative in INPUTS.items()}
    inventory = []
    for name, path in paths.items():
        present = path.is_file()
        inventory.append(dict(name=name, role="required" if name in REQUIRED else "optional",
                              condition="baseline" if name.startswith("baseline") else
                              "no_warp" if name.startswith("no_warp") else
                              "affine_and_syn" if name.startswith("warp") else "schedule_or_selection",
                              path=str(path), present=present,
                              size_bytes=path.stat().st_size if present else None,
                              sha256=digest(path) if present else None))
    missing_required = [r["name"] for r in inventory if r["role"] == "required" and not r["present"]]
    if missing_required:
        raise FileNotFoundError(f"Required inputs missing: {missing_required}. No output written.")

    runs = read_csv("test_runs", paths)
    run_keys = set(map(tuple, runs[["test_subject", "run"]].itertuples(index=False, name=None)))
    if len(runs) != 61 or len(run_keys) != 61 or set(runs.test_subject) != {f"sub-{i:02d}" for i in range(1, 8)}:
        raise ValueError("Expected 61 unique runs across sub-01 through sub-07")
    folds = read_csv("loso_folds", paths)
    if len(folds) != 7 or set(folds.test_subject) != set(runs.test_subject):
        raise ValueError("Expected seven LOSO folds matching the run subjects")
    loso = read_csv("loso_selection", paths)
    single = read_csv("single_selection", paths)
    methods = set(METHODS)
    exact_keys(loso, ["method", "test_subject"],
               {(m, s) for m in methods for s in runs.test_subject.unique()}, "LOSO selections")
    exact_keys(single, ["method", "reference_subject"],
               {(m, s) for m in methods for s in runs.test_subject.unique()}, "Single Reference selections")
    for label, frame in [("LOSO", loso), ("Single Reference", single)]:
        if frame.configuration_key.isna().any() or frame.configuration_parameters.isna().any():
            raise ValueError(f"{label}: missing configuration")
        if "selection_status" in frame and not frame.selection_status.eq("SELECTED").all():
            raise ValueError(f"{label}: non-selected parameter record")
        for value in frame.configuration_parameters:
            if not isinstance(json.loads(value), dict):
                raise ValueError(f"{label}: configuration_parameters must be JSON objects")
    schedule = read_csv("single_schedule", paths)
    expected_schedule = {(ref, test, run) for test, run in run_keys
                         for ref in runs.test_subject.unique() if ref != test}
    exact_keys(schedule, ["reference_subject", "test_subject", "run"], expected_schedule,
               "Single Reference schedule")
    if not set(map(tuple, runs[["test_subject", "run", "reference_subject"]].itertuples(index=False, name=None))).issubset(
            {(test, run, ref) for ref, test, run in expected_schedule}):
        raise ValueError("LOSO reference identities are absent from the Single Reference schedule")

    baseline = read_csv("baseline_results", paths).rename(columns={"subject": "test_subject"})
    base_keys = {(m, test, run) for m in methods for test, run in run_keys}
    exact_keys(baseline, ["method", "test_subject", "run"], base_keys, "Historical baseline")
    base_status = check_status(baseline, "Historical baseline")
    if (baseline.loc[baseline.status.eq("PASS"), "localization_error_mm"].isna().any()):
        raise ValueError("Historical baseline has PASS without original error")
    definitions = read_csv("baseline_definitions", paths)
    exact_keys(definitions, ["method"], {(m,) for m in methods}, "Historical definitions")
    if not baseline[["method", "configuration_key"]].merge(
            definitions[["method", "configuration_key"]], on=["method", "configuration_key"],
            how="left", indicator=True)._merge.eq("both").all():
        raise ValueError("Historical run configurations do not match selected definitions")

    no_warp = read_csv("no_warp_primary", paths)
    check_status(no_warp, "No-warp primary")
    nw_counts = {}
    for strategy, expected in [("LOSO", 732), ("Single-reference", 4392)]:
        subset = no_warp[(no_warp.strategy == strategy) & (no_warp.condition == "single_reference_no_warp")]
        keys = ["method", "test_subject", "run"] + (["reference_subject"] if strategy != "LOSO" else [])
        expected_keys = base_keys if strategy == "LOSO" else {
            (m, test, run, ref) for m in methods for ref, test, run in expected_schedule}
        exact_keys(subset, keys, expected_keys, f"No-warp {strategy}")
        if len(subset) != expected:
            raise ValueError(f"No-warp {strategy}: expected {expected} rows")
        nw_counts[strategy] = check_status(subset, f"No-warp {strategy}")

    warp_summary = {"available": False, "configuration_keys_match_no_warp": None,
                    "primary_status_by_strategy_mode": [], "comparability": "pending later coordinate and case QA"}
    warp_names = ["warp_loso_selection", "warp_single_selection", "warp_mapped_primary", "warp_mapped_manifest"]
    if all(paths[name].is_file() for name in warp_names):
        key_match = True
        for name, original, keys in [("warp_loso_selection", loso, ["method", "test_subject"]),
                                     ("warp_single_selection", single, ["method", "reference_subject"])]:
            other = read_csv(name, paths)
            exact_keys(other, keys, set(map(tuple, original[keys].itertuples(index=False, name=None))), name)
            joined = original[keys + ["configuration_key"]].merge(
                other[keys + ["configuration_key"]], on=keys, suffixes=("_no_warp", "_warp"), validate="one_to_one")
            key_match &= joined.configuration_key_no_warp.eq(joined.configuration_key_warp).all()
        mapped = read_csv("warp_mapped_primary", paths)
        for strategy in ["LOSO", "Single-reference"]:
            for mode in ["affine", "syn"]:
                part = mapped[(mapped.strategy == strategy) & (mapped.registration_mode == mode) &
                              (mapped.condition == "cross_reference")]
                keys = ["method", "test_subject", "run"] + (["reference_subject"] if strategy != "LOSO" else [])
                expected_keys = base_keys if strategy == "LOSO" else {
                    (m, test, run, ref) for m in methods for ref, test, run in expected_schedule}
                exact_keys(part, keys, expected_keys, f"Mapped {strategy}/{mode}")
                warp_summary["primary_status_by_strategy_mode"].append(
                    dict(strategy=strategy, mode=mode, scheduled=len(part), **check_status(part, f"Mapped {strategy}/{mode}")))
        warp_summary.update(available=True, configuration_keys_match_no_warp=bool(key_match))
        if not key_match:
            warp_summary["comparability"] = "FAIL: selected configurations differ; do not label directly matched"

    fsroot = args.fsaverage_root.expanduser().resolve()
    resources = []
    for name, relative in FS_FILES.items():
        path = fsroot / relative
        resources.append(dict(name=name, path=str(path), present=path.is_file(),
                              sha256=digest(path) if path.is_file() else None))
    warnings = [f"Missing fsaverage resource: {r['path']}" for r in resources if not r["present"]]
    warnings += [f"Optional comparison input missing: {r['name']} ({r['path']})" for r in inventory
                 if r["role"] == "optional" and not r["present"]]
    warnings.append("Historical baseline snapshot has original errors but no peak coordinates; transformed baseline error unavailable until a documented peak artifact is located.")
    warnings.append("No-warp raw head separation is not anatomical localization error; common-space comparison requires validated mapping.")
    if warp_summary["available"] and not warp_summary["configuration_keys_match_no_warp"]:
        warnings.append("Warp parameter selections differ from no-warp selections; direct matched comparison is invalid.")

    planned = pd.concat([
        runs.assign(strategy="LOSO", reference_subject=runs.reference_subject)[["strategy", "reference_subject", "test_subject", "run"]],
        schedule.assign(strategy="Single-reference")[["strategy", "reference_subject", "test_subject", "run"]],
    ], ignore_index=True).merge(pd.DataFrame({"method": METHODS}), how="cross")
    if len(planned) != 5124 or planned.duplicated(["strategy", "method", "reference_subject", "test_subject", "run"]).any():
        raise ValueError("Expected 5,124 unique new fsaverage jobs")
    planned["model_subject"] = "fsaverage"
    planned["parameter_source"] = planned.strategy.map({"LOSO": "frozen LOSO selection by test subject",
                                                           "Single-reference": "frozen selection by reference subject"})
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), script="01_prepare_experiment.py",
                    counts=dict(test_runs=61, methods=12, loso_selection_records=84,
                                single_selection_records=84, baseline_results=732,
                                fsaverage_loso_planned=732, fsaverage_single_planned=4392,
                                fsaverage_total_planned=5124), baseline_status=base_status,
                    no_warp_status=nw_counts, warp=warp_summary, fsaverage_resources=resources,
                    input_inventory=inventory, warnings=warnings,
                    coordinate_policy="No registration or localization error computed in Script 01; later comparisons require a validated common anatomical frame.",
                    old_results_reconstructed=False, fsaverage_forward_computed=False,
                    software_versions={package: importlib.metadata.version(package) if importlib.metadata.packages_distributions().get(package) else None
                                       for package in ["pandas", "numpy", "mne"]})
    out.mkdir(parents=True, exist_ok=False)
    (out / "source_configurations").mkdir()
    for name in SNAPSHOT:
        shutil.copy2(paths[name], out / "source_configurations" / f"{name}{paths[name].suffix}")
    pd.DataFrame(inventory).to_csv(out / "01_input_inventory.csv", index=False)
    planned.to_csv(out / "02_fsaverage_reconstruction_plan.csv", index=False)
    pd.DataFrame([dict(method=m, family=METHOD_META[m][0], publication_year=METHOD_META[m][1])
                  for m in METHODS]).to_csv(out / "03_method_metadata.csv", index=False)
    pd.DataFrame(warp_summary["primary_status_by_strategy_mode"]).to_csv(out / "04_warp_coverage.csv", index=False)
    (out / "05_experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.txt").write_text("Script 01 input audit and frozen selections. No reconstruction or registration was run.\n"
                                    "Read 05_experiment_manifest.json warnings before Script 02.\n")
    print(f"Prepared {out}; planned 732 LOSO + 4,392 Single Reference fsaverage jobs.")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
