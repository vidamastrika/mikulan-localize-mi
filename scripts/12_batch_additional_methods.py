#!/usr/bin/env python3
"""Run the pilot-selected additional inverse methods across Localize-MI.

This is the all-run companion to Script 11.  It deliberately imports the
finalized Script 11 implementation so the pilot and batch calculations use
identical ECD-grid, LCMV, MxNE, irMxNE, metric, and figure code.

Selected configuration
----------------------
* all released runs and all good EEG channels
* target window: -2 to +2 ms
* ECD-grid: maximum whitened goodness of fit
* LCMV: regularization 0.10 and shrunk data covariance from -5 to +5 ms
* MxNE and irMxNE: shared alpha 40, loose 1.0, and depth 0.1
* irMxNE: 10 iterations

Loose and depth are shared by MxNE and irMxNE to keep their comparison even.
The loose and depth values also match the baseline used by the basic inverse
methods. They are not parameters of ECD-grid or LCMV. LCMV regularization 0.10
was retained from the balanced four-run parameter pilot.

The CSV is atomically checkpointed after every solution.  ``--resume`` skips
matching successful rows and retries matching failed rows.  ``--overwrite``
starts the selected output again.  Diagnostic figures are disabled by default
and, when requested, are limited to four representative runs unless explicit
``--figure-case`` selections are supplied.

Examples
--------
Run a one-run validation first::

    python scripts/12_batch_additional_methods.py \
        --case sub-01 run-01 \
        --output outputs/additional_methods_comparable/test_sub01_run01.csv \
        --overwrite --quiet

Run all 61 released runs::

    python scripts/12_batch_additional_methods.py --quiet

Resume an interrupted batch::

    python scripts/12_batch_additional_methods.py --resume --quiet

Default output
--------------
outputs/additional_methods_comparable/additional_method_results.csv
outputs/additional_methods_comparable/additional_method_manifest.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import mne
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT11 = PROJECT / "scripts/11_additional_method_pilot.py"
DEFAULT_OUTPUT = (
    PROJECT / "outputs/additional_methods_comparable/additional_method_results.csv"
)
DEFAULT_FIGURE_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)


def load_script11():
    """Load Script 11 without duplicating its scientific implementation."""
    if not SCRIPT11.is_file():
        raise FileNotFoundError(
            f"Finalized Script 11 was not found: {SCRIPT11}"
        )
    specification = importlib.util.spec_from_file_location(
        "localize_mi_script11", SCRIPT11
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load Script 11: {SCRIPT11}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


pilot = load_script11()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT / "data/Localize-MI",
        help="Localize-MI BIDS root.",
    )
    parser.add_argument("--task", default="seegstim")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest", type=Path,
        help="Default: additional_method_manifest.json beside the CSV.",
    )
    parser.add_argument(
        "--baseline-results", type=Path,
        default=PROJECT / "outputs/tables/inverse_method_results.csv",
        help="Script 05 table used for descriptive reference comparisons.",
    )
    parser.add_argument(
        "--case", nargs=2, metavar=("SUBJECT", "RUN"), action="append",
        help="Repeat to process explicit cases instead of discovering all runs.",
    )
    parser.add_argument(
        "--subject", action="append",
        help="Repeat to restrict automatically discovered runs by participant.",
    )
    parser.add_argument(
        "--run", action="append",
        help="Repeat to restrict automatically discovered runs by run label.",
    )
    parser.add_argument(
        "--method", type=pilot.normalize_method, action="append",
        help="Repeat to select methods; default: all four additional methods.",
    )
    parser.add_argument("--max-runs", type=int)

    parser.add_argument(
        "--loose", type=float, default=1.0,
        help=("Shared MxNE/irMxNE loose value, matching the basic-method "
              "baseline. Default: 1.0."),
    )
    parser.add_argument("--depth", type=float, default=0.1)
    parser.add_argument("--lcmv-reg", type=float, default=0.10)
    parser.add_argument("--lcmv-data-tmin", type=float, default=-0.005)
    parser.add_argument("--lcmv-data-tmax", type=float, default=0.005)
    parser.add_argument(
        "--lcmv-data-covariance-method", default="shrunk",
        choices=("empirical", "shrunk", "diagonal_fixed", "auto"),
    )
    parser.add_argument(
        "--mxne-alpha", type=pilot.parse_alpha, default=40.0,
        help="Sparse regularization in [0, 100), or 'sure'. Default: 40.",
    )
    parser.add_argument("--irmxne-iterations", type=int, default=10)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--save-figures", action="store_true",
        help="Save figures only for representative or --figure-case runs.",
    )
    parser.add_argument(
        "--figure-case", nargs=2, metavar=("SUBJECT", "RUN"), action="append",
        help="Repeat to choose runs receiving figures.",
    )
    parser.add_argument(
        "--figures-dir", type=Path,
        help="Default: selected_run_figures beside the CSV.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("--max-runs must be at least 1.")
    if not 0 <= args.loose <= 1:
        raise ValueError("--loose must be between 0 and 1.")
    if not 0 <= args.depth <= 1:
        raise ValueError("--depth must be between 0 and 1.")
    if args.lcmv_reg < 0:
        raise ValueError("--lcmv-reg must be non-negative.")
    if args.lcmv_data_tmin >= args.lcmv_data_tmax:
        raise ValueError(
            "--lcmv-data-tmin must be earlier than --lcmv-data-tmax."
        )
    if args.irmxne_iterations <= 1:
        raise ValueError("--irmxne-iterations must be greater than 1.")


def natural_key(text):
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(text))
    )


def discover_runs(dataset, task, subjects=None, runs=None):
    """Find released EEG epoch arrays and return natural-sorted cases."""
    root = dataset / "derivatives/epochs"
    if not root.is_dir():
        raise NotADirectoryError(f"Epoch derivatives not found: {root}")
    pattern = re.compile(
        rf"^(sub-\d+)_task-{re.escape(task)}_(run-\d+)_epochs\.npy$"
    )
    selected_subjects = set(subjects or ())
    selected_runs = set(runs or ())
    cases = set()
    for path in root.glob("sub-*/eeg/*_epochs.npy"):
        match = pattern.match(path.name)
        if match is None:
            continue
        subject, run = match.groups()
        if selected_subjects and subject not in selected_subjects:
            continue
        if selected_runs and run not in selected_runs:
            continue
        cases.add((subject, run))
    return sorted(cases, key=lambda item: (natural_key(item[0]), natural_key(item[1])))


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def configuration(args, dataset, methods, cases):
    return {
        "design": "shared sparse parameters matching basic-method baseline",
        "selection_pilot": "Script 14 balanced four-run pilot",
        "dataset": str(dataset),
        "task": args.task,
        "methods": list(methods),
        "cases": [list(case) for case in cases],
        "montage": "all_good",
        "target_tmin_s": pilot.TARGET_TMIN,
        "target_tmax_s": pilot.TARGET_TMAX,
        "noise_covariance_tmin_s": pilot.COVARIANCE_TMIN,
        "noise_covariance_tmax_s": pilot.COVARIANCE_TMAX,
        "loose": args.loose,
        "depth": args.depth,
        "lcmv_reg": args.lcmv_reg,
        "lcmv_data_tmin_s": args.lcmv_data_tmin,
        "lcmv_data_tmax_s": args.lcmv_data_tmax,
        "lcmv_data_covariance_method": args.lcmv_data_covariance_method,
        "lcmv_pick_ori": "max-power",
        "lcmv_weight_norm": "unit-noise-gain-invariant",
        "mxne_alpha": args.mxne_alpha,
        "irmxne_iterations": args.irmxne_iterations,
    }


def manifest_payload(config, output, rows):
    return {
        "script": Path(__file__).name,
        "scientific_implementation": SCRIPT11.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_csv": str(output),
        "configuration": config,
        "software": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rows": len(rows),
        "passes": sum(str(row.get("status")) == "PASS" for row in rows),
        "failures": sum(str(row.get("status")) == "FAIL" for row in rows),
    }


def load_rows(output, resume, overwrite):
    if not output.exists() or overwrite:
        return []
    if not resume:
        raise FileExistsError(
            f"Output exists: {output}. Use --resume or --overwrite."
        )
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(pilot.FIELDNAMES):
            raise ValueError(
                "Existing CSV columns do not match the finalized Script 11."
            )
        return list(reader)


def same_number(observed, expected):
    try:
        return bool(np.isclose(float(observed), float(expected)))
    except (TypeError, ValueError):
        return False


def row_matches(row, subject, run, method, args):
    """Only skip a PASS produced with this exact baseline configuration."""
    if (
        str(row.get("status")) != "PASS"
        or str(row.get("subject")) != subject
        or str(row.get("run")) != run
        or str(row.get("method")) != method
        or str(row.get("montage")) != "all_good"
    ):
        return False
    common = {
        "target_tmin_s": pilot.TARGET_TMIN,
        "target_tmax_s": pilot.TARGET_TMAX,
        "covariance_tmin_s": pilot.COVARIANCE_TMIN,
        "covariance_tmax_s": pilot.COVARIANCE_TMAX,
    }
    if method in ("MxNE", "irMxNE"):
        common.update(
            loose=args.loose,
            depth=args.depth,
            mxne_alpha=args.mxne_alpha,
            mxne_iterations=(1 if method == "MxNE" else args.irmxne_iterations),
        )
    elif method == "LCMV":
        common.update(
            lcmv_reg=args.lcmv_reg,
            lcmv_data_tmin_s=args.lcmv_data_tmin,
            lcmv_data_tmax_s=args.lcmv_data_tmax,
        )
        if str(row.get("data_covariance_method")) != args.lcmv_data_covariance_method:
            return False
    return all(same_number(row.get(column), value) for column, value in common.items())


def remove_solution(rows, subject, run, method):
    """Remove a prior matching solution before retrying or replacing it."""
    return [
        row for row in rows
        if pilot.row_key(row) != (subject, run, method)
    ]


def base_row(subject, run, method, args, loaded=None, stimulation=None,
             noise_covariance=None, reference=None):
    row = dict.fromkeys(pilot.FIELDNAMES, "")
    row.update(
        subject=subject,
        run=run,
        method=method,
        montage="all_good",
        target_tmin_s=pilot.TARGET_TMIN,
        target_tmax_s=pilot.TARGET_TMAX,
        covariance_tmin_s=pilot.COVARIANCE_TMIN,
        covariance_tmax_s=pilot.COVARIANCE_TMAX,
    )
    if loaded is not None:
        row.update(
            n_good_channels=(
                len(loaded.epochs.ch_names) - len(loaded.epochs.info["bads"])
            ),
            epochs=len(loaded.epochs),
            sampling_frequency_hz=float(loaded.epochs.info["sfreq"]),
            evaluated_sources=pilot.count_forward_sources(loaded.forward),
        )
    if stimulation is not None:
        row.update(
            stimulation_pair=stimulation.pair,
            hemisphere=stimulation.hemisphere,
        )
    if noise_covariance is not None:
        row["covariance_method_selected"] = str(
            noise_covariance.get("method", "auto")
        )
    if reference is not None:
        row.update(
            reference_method=reference[0],
            reference_distance_mm=reference[1],
        )
    if method == "LCMV":
        row.update(
            data_covariance_method=args.lcmv_data_covariance_method,
            lcmv_data_tmin_s=args.lcmv_data_tmin,
            lcmv_data_tmax_s=args.lcmv_data_tmax,
            lcmv_reg=args.lcmv_reg,
            lcmv_pick_ori="max-power",
            lcmv_weight_norm="unit-noise-gain-invariant",
        )
    elif method in ("MxNE", "irMxNE"):
        row.update(
            loose=args.loose,
            depth=args.depth,
            mxne_alpha=args.mxne_alpha,
            mxne_iterations=(1 if method == "MxNE" else args.irmxne_iterations),
        )
    elif method == "ECD-grid":
        row["ecd_selection_criterion"] = "maximum whitened GOF"
    return row


def record_failure(rows, output, manifest, config, row, exc, started):
    row.update(
        status="FAIL",
        error_type=type(exc).__name__,
        error_message=str(exc),
        runtime_seconds=round(time.monotonic() - started, 3),
    )
    rows = remove_solution(rows, row["subject"], row["run"], row["method"])
    rows.append(row)
    pilot.save_rows(output, rows)
    atomic_json(manifest, manifest_payload(config, output, rows))
    return rows


def main():
    args = parse_args()
    validate_args(args)
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest else output.with_name(
            "additional_method_manifest.json"
            if output == DEFAULT_OUTPUT.resolve()
            else f"{output.stem}_manifest.json"
        )
    )
    figure_root = (
        args.figures_dir.expanduser().resolve()
        if args.figures_dir else output.parent / "selected_run_figures"
    )
    methods = list(dict.fromkeys(args.method or pilot.METHODS))

    if not dataset.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset}")
    if args.case:
        cases = list(dict.fromkeys(tuple(case) for case in args.case))
        if args.subject or args.run:
            raise ValueError("Do not combine --case with --subject or --run.")
    else:
        cases = discover_runs(dataset, args.task, args.subject, args.run)
    if args.max_runs is not None:
        cases = cases[:args.max_runs]
    if not cases:
        raise FileNotFoundError("No matching Localize-MI runs were found.")

    figure_cases = set(
        tuple(case) for case in (args.figure_case or DEFAULT_FIGURE_CASES)
    )
    combinations = [
        (subject, run, method)
        for subject, run in cases for method in methods
    ]
    selected_keys = set(combinations)
    rows = load_rows(output, args.resume, args.overwrite)
    outside = {
        pilot.row_key(row) for row in rows
        if pilot.row_key(row) not in selected_keys
    }
    if outside:
        raise ValueError(
            "Existing CSV contains rows outside the current selection; "
            "use the same selection or a different --output path."
        )
    keys = [pilot.row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Existing CSV contains duplicate method solutions.")

    config = configuration(args, dataset, methods, cases)
    references = pilot.load_references(
        args.baseline_results.expanduser().resolve(), cases
    )
    atomic_json(manifest, manifest_payload(config, output, rows))
    total = len(combinations)
    successful = {
        (subject, run, method)
        for subject, run, method in combinations
        if any(
            row_matches(row, subject, run, method, args) for row in rows
        )
    }

    print("LOCALIZE-MI ADDITIONAL-METHOD BATCH")
    print("-----------------------------------")
    print(f"Runs             : {len(cases)}")
    print(f"Methods          : {', '.join(methods)}")
    print(f"Solutions        : {total}")
    print(f"Already complete : {len(successful)}")
    print("Montage          : all good EEG channels")
    print("Target           : -2 to +2 ms")
    print(
        "LCMV covariance  : "
        f"{args.lcmv_data_tmin * 1000:g} to "
        f"{args.lcmv_data_tmax * 1000:g} ms"
    )
    print(f"LCMV reg         : {args.lcmv_reg:g}")
    print(f"Sparse alpha     : {args.mxne_alpha}")
    print(f"Sparse loose     : {args.loose:g} (shared)")
    print(f"Sparse depth     : {args.depth:g} (shared)")
    print(f"irMxNE iterations: {args.irmxne_iterations}")
    print(f"Output           : {output}", flush=True)

    completed_count = len(successful)
    new_failures = 0
    for run_index, (subject, run) in enumerate(cases, start=1):
        pending = [
            method for method in methods
            if (subject, run, method) not in successful
        ]
        if not pending:
            print(
                f"[{run_index:02d}/{len(cases):02d}] "
                f"{subject} {run}: already complete",
                flush=True,
            )
            continue
        print(
            f"[{run_index:02d}/{len(cases):02d}] {subject} {run}",
            flush=True,
        )

        preparation_start = time.monotonic()
        try:
            loaded = pilot.load_run(
                dataset=dataset, subject=subject, run=run, task=args.task
            )
            electrodes = (
                dataset / "derivatives/epochs" / subject / "ieeg"
                / f"{subject}_task-{args.task}_space-surface_electrodes.tsv"
            )
            transform_file = (
                dataset / "derivatives/sourcemodelling" / subject / "xfm"
                / f"{subject}_from-head_to-surface.h5"
            )
            stimulation = pilot.load_stimulation_info(
                loaded.metadata["Description"], electrodes
            )
            transform = pilot.load_surface_transform(transform_file)
            referenced, noise_covariance, evoked = pilot.common_preprocessing(
                loaded.epochs, args.quiet
            )
            data_covariance = None
            if "LCMV" in pending:
                data_covariance = pilot.compute_data_covariance(
                    referenced,
                    args.lcmv_data_covariance_method,
                    args.lcmv_data_tmin,
                    args.lcmv_data_tmax,
                    args.quiet,
                )
        except Exception as exc:
            for method in pending:
                row = base_row(subject, run, method, args)
                rows = record_failure(
                    rows, output, manifest, config, row, exc, preparation_start
                )
                completed_count += 1
                new_failures += 1
                print(f"  {method:8} FAIL during preparation: {exc}", flush=True)
            continue

        channel_count = len(loaded.epochs.ch_names) - len(
            loaded.epochs.info["bads"]
        )
        reference = references.get((subject, run))
        for method in pending:
            started = time.monotonic()
            row = base_row(
                subject, run, method, args, loaded, stimulation,
                noise_covariance, reference,
            )
            try:
                residual = None
                if method == "ECD-grid":
                    source_estimate, diagnostics = pilot.run_ecd_grid(
                        evoked, loaded.forward, noise_covariance,
                        stimulation.hemisphere,
                    )
                    row.update(
                        active_sources=1,
                        ecd_goodness_of_fit_percent=diagnostics["gof_percent"],
                    )
                elif method == "LCMV":
                    source_estimate = pilot.run_lcmv(
                        evoked, loaded.forward, noise_covariance,
                        data_covariance, args.lcmv_reg, args.quiet,
                    )
                    row["lcmv_data_covariance_samples"] = (
                        data_covariance.get("nfree", "")
                    )
                else:
                    iterations = (
                        1 if method == "MxNE" else args.irmxne_iterations
                    )
                    source_estimate, residual = pilot.run_mxne(
                        evoked, loaded.forward, noise_covariance,
                        args.mxne_alpha, args.loose, args.depth,
                        iterations, args.quiet,
                    )
                    row.update(
                        active_sources=pilot.count_active_sources(source_estimate),
                        explained_variance_percent=pilot.explained_variance(
                            evoked, residual
                        ),
                    )
                    if int(row["active_sources"]) == 0:
                        raise RuntimeError(
                            f"{method} returned no active cortical sources "
                            f"with alpha={args.mxne_alpha}."
                        )

                metric = pilot.calculate_localization_metrics(
                    source_estimate, loaded.forward, transform, stimulation
                )
                distance = float(metric.localization_distance_mm)
                row.update(
                    peak_vertex=int(metric.peak_vertex),
                    peak_time_s=float(metric.peak_time),
                    peak_time_ms=float(metric.peak_time * 1000),
                    localization_distance_mm=distance,
                    nearest_source_distance_mm=float(
                        metric.nearest_source_distance_mm
                    ),
                    geometric_excess_mm=(
                        distance - float(metric.nearest_source_distance_mm)
                    ),
                    estimated_x_m=float(metric.peak_coordinate[0]),
                    estimated_y_m=float(metric.peak_coordinate[1]),
                    estimated_z_m=float(metric.peak_coordinate[2]),
                    known_x_m=float(metric.stimulation_midpoint[0]),
                    known_y_m=float(metric.stimulation_midpoint[1]),
                    known_z_m=float(metric.stimulation_midpoint[2]),
                    status="PASS",
                    runtime_seconds=round(time.monotonic() - started, 3),
                )
                if row["reference_distance_mm"] != "":
                    row["change_from_reference_mm"] = (
                        distance - float(row["reference_distance_mm"])
                    )
                if args.save_figures and (subject, run) in figure_cases:
                    pilot.save_figures(
                        figure_root, subject, run, method, evoked,
                        source_estimate, metric, loaded.forward, transform,
                        stimulation, channel_count, args.loose, args.depth,
                        pilot.figure_parameter_context(method, args),
                    )
                rows = remove_solution(rows, subject, run, method)
                rows.append(row)
                pilot.save_rows(output, rows)
                atomic_json(
                    manifest, manifest_payload(config, output, rows)
                )
                reference_text = ""
                if row["change_from_reference_mm"] != "":
                    reference_text = (
                        f"; vs {row['reference_method']} "
                        f"{float(row['change_from_reference_mm']):+.2f} mm"
                    )
                print(
                    f"  {method:8}: {distance:.2f} mm{reference_text}",
                    flush=True,
                )
            except Exception as exc:
                rows = record_failure(
                    rows, output, manifest, config, row, exc, started
                )
                new_failures += 1
                print(f"  {method:8} FAIL: {exc}", flush=True)
            completed_count += 1
            if completed_count % 20 == 0 or completed_count == total:
                print(
                    f"Checkpoint: {completed_count}/{total}; "
                    f"new failures: {new_failures}",
                    flush=True,
                )

    failures = sum(str(row.get("status")) == "FAIL" for row in rows)
    passes = sum(str(row.get("status")) == "PASS" for row in rows)
    print("\nADDITIONAL-METHOD BATCH COMPLETE")
    print("--------------------------------")
    print(f"Rows      : {len(rows)}")
    print(f"Passes    : {passes}")
    print(f"Failures  : {failures}")
    print(f"CSV       : {output}")
    print(f"Manifest  : {manifest}")
    if args.save_figures:
        print(f"Figures   : {figure_root}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
