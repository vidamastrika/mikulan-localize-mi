#!/usr/bin/env python3
"""Pilot inverse-parameter sensitivity on four representative Localize-MI runs.

The published full grid is 3 methods × 4 EGI montages × 10 loose values ×
10 depth values × 4 SNR values = 4,800 solutions per run (Mikulan et al.,
Scientific Data 2020, doi:10.1038/s41597-020-0467-x). This pilot tests one
factor at a time around the validated Script 05 baseline, on all good EEG
channels only. It must not be described as the published full grid.

Run from the repository root:
    python scripts/08_parameter_pilot.py

Default output: outputs/parameter_pilot/pilot_results.csv
The table is saved after each solution. Use --resume after interruption.

Create figures from the parameter combination using script 04
python scripts/04_reproduce_method.py \
    --subject sub-05 \
    --run run-06 \
    --method MNE \
    --loose 1.0 \
    --depth 0.1 \
    --snr 1 \
    --output-root outputs/parameter_pilot/figures/baseline \
    --quiet

python scripts/04_reproduce_method.py \
    --subject sub-05 \
    --run run-06 \
    --method MNE \
    --loose 1.0 \
    --depth 1.0 \
    --snr 1 \
    --output-root outputs/parameter_pilot/figures/depth_1 \
    --quiet

python scripts/04_reproduce_method.py \
    --subject sub-07 \
    --run run-05 \
    --method dSPM \
    --loose 1.0 \
    --depth 0.1 \
    --snr 2 \
    --output-root outputs/parameter_pilot/figures/snr_2 \
    --quiet

You can see the parameter combinations in the output CSV, and the figures will show the same solution.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from localize_mi import load_run  # noqa: E402
from localize_mi.inverse import run_inverse  # noqa: E402
from localize_mi.metrics import (  # noqa: E402
    calculate_localization_metrics,
    load_stimulation_info,
    load_surface_transform,
)


DEFAULT_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)
PAPER_METHODS = ("MNE", "dSPM", "eLORETA")
FIELDNAMES = (
    "subject", "run", "method", "varied_parameter", "loose", "depth",
    "snr", "lambda2", "montage", "n_good_channels", "target_tmin_s",
    "target_tmax_s", "covariance_tmin_s", "covariance_tmax_s",
    "covariance_method_selected", "stimulation_pair", "hemisphere",
    "peak_vertex", "peak_time_ms", "localization_distance_mm",
    "nearest_source_distance_mm", "geometric_excess_mm",
    "estimated_x_m", "estimated_y_m", "estimated_z_m", "known_x_m",
    "known_y_m", "known_z_m", "baseline_distance_mm",
    "change_from_baseline_mm", "status", "error_type", "error_message",
    "runtime_seconds",
)


def settings():
    """Eight unique configurations, including the Script 05 baseline."""
    result = [("baseline", 1.0, 0.1, 1.0)]
    result += [("loose", v, 0.1, 1.0) for v in (0.1, 0.5)]
    result += [("depth", 1.0, v, 1.0) for v in (0.5, 1.0)]
    result += [("snr", 1.0, 0.1, v) for v in (2.0, 3.0, 4.0)]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data/Localize-MI")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "outputs/parameter_pilot/pilot_results.csv")
    parser.add_argument("--baseline-results", type=Path,
                        default=PROJECT / "outputs/tables/inverse_method_results.csv",
                        help="Script 05 table for paired baseline differences.")
    parser.add_argument("--case", nargs=2, metavar=("SUBJECT", "RUN"),
                        action="append", help="Repeat to override the four default runs.")
    parser.add_argument("--method", action="append", choices=(*PAPER_METHODS, "sLORETA"),
                        help="Repeat for multiple methods; default: the three paper methods.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing CSV, skipping completed configurations.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing CSV after explicit request.")
    parser.add_argument("--quiet", action="store_true", help="Suppress MNE progress logs.")
    return parser.parse_args()


def load_baselines(path, cases, methods):
    """Require matching validated baseline rows before computing differences."""
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"Script 05 baseline table not found: {path}")
    table = pd.read_csv(path)
    required = {"subject", "run", "method", "status", "loose", "depth", "snr",
                "target_tmin_s", "target_tmax_s", "localization_distance_mm"}
    if required - set(table.columns):
        raise ValueError(f"Script 05 table lacks columns: {sorted(required - set(table.columns))}")
    baseline = {}
    for subject, run in cases:
        for method in methods:
            selected = table.loc[(table.subject == subject) & (table.run == run)
                                 & (table.method == method) & (table.status == "PASS")
                                 & (table.loose == 1.0) & (table.depth == 0.1)
                                 & (table.snr == 1.0)
                                 & (table.target_tmin_s == -0.002)
                                 & (table.target_tmax_s == 0.002)]
            if len(selected) != 1:
                raise ValueError(f"Expected one Script 05 baseline: {subject} {run} {method}; "
                                 f"found {len(selected)}")
            baseline[(subject, run, method)] = float(selected.iloc[0].localization_distance_mm)
    return baseline


def key(row):
    return (str(row["subject"]), str(row["run"]), str(row["method"]),
            float(row["loose"]), float(row["depth"]), float(row["snr"]))


def save_rows(path, rows):
    """Atomically replace the CSV after each solution to support resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    cases = list(dict.fromkeys(tuple(case) for case in (args.case or DEFAULT_CASES)))
    methods = list(dict.fromkeys(args.method or PAPER_METHODS))
    combinations = [(subject, run, method, varied, loose, depth, snr)
                    for (subject, run), method, (varied, loose, depth, snr)
                    in itertools.product(cases, methods, settings())]
    if not dataset.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset}")
    baseline = load_baselines(args.baseline_results.expanduser().resolve(), cases, methods)
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"Output exists: {output}. Use --resume or --overwrite.")
    rows = []
    if args.resume and output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDNAMES):
                raise ValueError("Existing CSV columns do not match this pilot version.")
            rows = list(reader)
        actual = {key(row) for row in rows}
        if len(actual) != len(rows):
            raise ValueError("Existing CSV contains duplicate configurations.")
        expected = {(s, r, m, l, d, snr) for s, r, m, _, l, d, snr in combinations}
        if actual - expected:
            raise ValueError("Existing CSV contains combinations outside this selection."
                             " Use a different --output path.")
        # An interrupted pilot can retry failed solutions without creating
        # duplicate rows; successful results are preserved as checkpoints.
        rows = [row for row in rows if row["status"] == "PASS"]
    completed = {key(row) for row in rows}
    print(f"Parameter pilot: {len(cases)} runs × {len(methods)} methods × "
          f"{len(settings())} settings = {len(combinations)} solutions")
    print(f"All-good-channel montage; target window -2 to +2 ms; output: {output}", flush=True)

    for subject, run in cases:
        pending = [item for item in combinations if item[0:2] == (subject, run)
                   and (item[0], item[1], item[2], item[4], item[5], item[6])
                   not in completed]
        if not pending:
            continue
        loaded = load_run(dataset=dataset, subject=subject, run=run, task="seegstim")
        electrodes = dataset / "derivatives/epochs" / subject / "ieeg" / (
            f"{subject}_task-seegstim_space-surface_electrodes.tsv")
        transform_path = dataset / "derivatives/sourcemodelling" / subject / "xfm" / (
            f"{subject}_from-head_to-surface.h5")
        stimulation = load_stimulation_info(loaded.metadata["Description"], electrodes)
        transform = load_surface_transform(transform_path)

        for _, _, method, varied, loose, depth, snr in pending:
            start = time.monotonic()
            row = dict.fromkeys(FIELDNAMES, "")
            row.update(subject=subject, run=run, method=method,
                       varied_parameter=varied, loose=loose, depth=depth,
                       snr=snr, lambda2=1.0 / snr**2, montage="all_good",
                       n_good_channels=len(loaded.epochs.ch_names) - len(loaded.epochs.info["bads"]),
                       target_tmin_s=-0.002, target_tmax_s=0.002,
                       covariance_tmin_s=-0.250, covariance_tmax_s=-0.050,
                       stimulation_pair=stimulation.pair, hemisphere=stimulation.hemisphere,
                       baseline_distance_mm=baseline[(subject, run, method)])
            try:
                result = run_inverse(epochs=loaded.epochs, forward=loaded.forward,
                                     method=method, covariance_tmin=-0.250,
                                     covariance_tmax=-0.050, target_tmin=-0.002,
                                     target_tmax=0.002, loose=loose, depth=depth,
                                     snr=snr, covariance_method="auto",
                                     verbose=not args.quiet)
                metric = calculate_localization_metrics(
                    result.source_estimate, loaded.forward, transform, stimulation)
                distance = float(metric.localization_distance_mm)
                if varied == "baseline" and abs(distance - baseline[(subject, run, method)]) > 0.05:
                    raise ValueError(
                        f"Baseline changed by {distance - baseline[(subject, run, method)]:+.3f} mm "
                        "relative to Script 05. Check environment and pipeline before "
                        "interpreting the pilot."
                    )
                row.update(covariance_method_selected=str(result.covariance.get("method", "auto")),
                           peak_vertex=int(metric.peak_vertex),
                           peak_time_ms=float(metric.peak_time * 1000),
                           localization_distance_mm=distance,
                           nearest_source_distance_mm=float(metric.nearest_source_distance_mm),
                           geometric_excess_mm=distance - float(metric.nearest_source_distance_mm),
                           estimated_x_m=float(metric.peak_coordinate[0]),
                           estimated_y_m=float(metric.peak_coordinate[1]),
                           estimated_z_m=float(metric.peak_coordinate[2]),
                           known_x_m=float(metric.stimulation_midpoint[0]),
                           known_y_m=float(metric.stimulation_midpoint[1]),
                           known_z_m=float(metric.stimulation_midpoint[2]),
                           change_from_baseline_mm=distance - baseline[(subject, run, method)],
                           status="PASS")
            except (OSError, ValueError, RuntimeError, TypeError, KeyError, MemoryError) as exc:
                row.update(status="FAIL", error_type=type(exc).__name__, error_message=str(exc))
            row["runtime_seconds"] = round(time.monotonic() - start, 3)
            rows.append(row)
            save_rows(output, rows)
            if row["status"] == "PASS":
                print(f"{subject} {run} {method:8} {varied:8} "
                      f"loose={loose:.1f} depth={depth:.1f} SNR={snr:.0f} "
                      f"{distance:.2f} mm (change {row['change_from_baseline_mm']:+.2f})", flush=True)
            else:
                print(f"{subject} {run} {method} {varied} FAIL: {row['error_message']}", flush=True)

    n_failed = sum(row["status"] != "PASS" for row in rows)
    print(f"Saved {len(rows)} rows; failures: {n_failed}; output: {output}")
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
