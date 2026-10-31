#!/usr/bin/env python3
"""
12_additional_methods_all_runs_REVISED.py

Run the FINAL BASELINE configuration of one or more additional EEG source-
localization methods across all released Localize-MI runs.

This is the all-run companion to:
    scripts/11_additional_method_pilot_REVISED.py

IMPORTANT DESIGN CHOICE
-----------------------
Script 12 does NOT re-implement the inverse methods. It imports Script 11 so
that the pilot and all-run experiments use exactly the same scientific code.
Only the run discovery, checkpointing, batch output, and QC figures are added.

BASELINE PARAMETERS
-------------------
Shared:
    montage                  = all good EEG channels
    noise covariance window  = -250 to -50 ms
    noise covariance method  = auto (MNE chooses estimator by CV)
    target window            = -2 to +2 ms

Continuous-ECD:
    position/orientation     = unconstrained continuous dipole fitting
    ecd_min_dist             = 5 mm
    primary selection        = maximum GOF in target window
    BEM                      = subject-specific 3-layer, ico=4
    conductivity             = brain 0.3 / skull 0.006 / scalp 0.3 S/m

MxNE:
    alpha                    = 40
    loose                    = 1.0
    depth                    = 0.1
    n_mxne_iter              = 1
    primary selection        = maximum source amplitude

RAP-MUSIC:
    n_dipoles                = 1
    primary selection        = maximum GOF in target window

LCMV:
    reg                      = 0.05
    data covariance          = shrunk, -5 to +5 ms
    pick_ori                 = max-power
    weight_norm              = unit-noise-gain-invariant
    primary selection        = maximum source amplitude

These are BASELINE values, not optimized values. Parameter exploration belongs
in Script 14.

TERMINAL EXAMPLES
-----------------
Recommended: run ONE METHOD AT A TIME.

Continuous ECD, all released runs:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method ecd --overwrite

MxNE, all released runs:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method mxne --overwrite

RAP-MUSIC, all released runs:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method rap_music --overwrite

LCMV, all released runs:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method lcmv --overwrite

Resume an interrupted MxNE batch:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method mxne --resume

Quick one-run validation before a full batch:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method mxne --case sub-01 run-01 --overwrite

Run only one participant:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method lcmv --subject sub-07 --overwrite

Run all four methods together (supported, but one-by-one is easier to manage):
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method all --overwrite

Quiet MNE logging:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method mxne --overwrite --quiet

Disable automatic QC figures:
    python scripts/12_additional_methods_all_runs_REVISED.py \
        --method mxne --overwrite --no-figures

DEFAULT OUTPUT ROOT
-------------------
    outputs/12_additional_methods_all_runs/

When one method is selected, the main CSV is method-specific, e.g.:
    12_continuous_ecd_all_runs.csv
    12_mxne_all_runs.csv
    12_rap_music_all_runs.csv
    12_lcmv_all_runs.csv

This prevents one method from overwriting another when methods are run
separately.

QC figures are saved to:
    outputs/12_additional_methods_all_runs/figures/

The two QC figures are intentionally simple:
    1. localization-error distribution (histogram)
    2. localization error sorted across runs

Detailed comparative/statistical figures belong in Script 13 (summary), not
here.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT11 = PROJECT / "scripts" / "11_additional_method_pilot.py"
DEFAULT_DATASET = PROJECT / "data" / "Localize-MI"
DEFAULT_OUTPUT_ROOT = PROJECT / "outputs" / "12_additional_methods_all_runs"


def load_script11():
    """Import the finalized Script 11 implementation."""
    if not SCRIPT11.is_file():
        raise FileNotFoundError(
            f"Required Script 11 was not found: {SCRIPT11}\n"
            "Keep the revised Script 11 in scripts/."
        )
    spec = importlib.util.spec_from_file_location("localize_mi_script11", SCRIPT11)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {SCRIPT11}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pilot = load_script11()


def natural_key(text):
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(text))
    )


def discover_runs(dataset, task="seegstim", subjects=None, runs=None):
    """Discover released Localize-MI EEG epoch arrays."""
    root = dataset / "derivatives" / "epochs"
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

    return sorted(
        cases,
        key=lambda item: (natural_key(item[0]), natural_key(item[1])),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", default="seegstim")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument(
        "--method",
        type=pilot.normalize_method,
        action="append",
        help="Repeat if needed. Choices: ecd, mxne, rap_music, lcmv, all.",
    )
    parser.add_argument(
        "--case", nargs=2, metavar=("SUBJECT", "RUN"), action="append"
    )
    parser.add_argument("--subject", action="append")
    parser.add_argument("--run", action="append")
    parser.add_argument("--max-runs", type=int)

    # Baseline parameters: intentionally identical to revised Script 11.
    parser.add_argument("--ecd-min-dist", type=float, default=5.0)
    parser.add_argument("--mxne-alpha", type=pilot.parse_alpha, default=40.0)
    parser.add_argument("--mxne-loose", type=float, default=1.0)
    parser.add_argument("--mxne-depth", type=float, default=0.1)
    parser.add_argument("--rap-music-n-dipoles", type=int, default=1)
    parser.add_argument("--lcmv-reg", type=float, default=0.05)
    parser.add_argument("--lcmv-data-tmin", type=float, default=-0.005)
    parser.add_argument("--lcmv-data-tmax", type=float, default=0.005)
    parser.add_argument(
        "--lcmv-data-covariance-method",
        default="shrunk",
        choices=("empirical", "shrunk", "diagonal_fixed", "auto"),
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")
    if args.case and (args.subject or args.run):
        raise ValueError("Do not combine --case with --subject or --run.")
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("--max-runs must be at least 1.")
    if args.ecd_min_dist < 0:
        raise ValueError("--ecd-min-dist must be non-negative.")
    if not 0 <= args.mxne_loose <= 1:
        raise ValueError("--mxne-loose must be between 0 and 1.")
    if not 0 <= args.mxne_depth <= 1:
        raise ValueError("--mxne-depth must be between 0 and 1.")
    if args.rap_music_n_dipoles < 1:
        raise ValueError("--rap-music-n-dipoles must be at least 1.")
    if args.lcmv_reg < 0:
        raise ValueError("--lcmv-reg must be non-negative.")
    if args.lcmv_data_tmin >= args.lcmv_data_tmax:
        raise ValueError("LCMV data covariance tmin must be earlier than tmax.")


def method_slug(method):
    return {
        "Continuous-ECD": "continuous_ecd",
        "MxNE": "mxne",
        "RAP-MUSIC": "rap_music",
        "LCMV": "lcmv",
    }[method]


def output_file_for(output_root, methods):
    if len(methods) == 1:
        name = f"12_{method_slug(methods[0])}_all_runs.csv"
    else:
        name = "12_all_methods_all_runs.csv"
    return output_root / name


def read_existing_rows(output_file, resume, overwrite, expected_keys):
    if overwrite or not output_file.exists():
        return []
    if not resume:
        raise FileExistsError(
            f"Output already exists: {output_file}\n"
            "Use --resume or --overwrite."
        )

    with output_file.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(pilot.FIELDNAMES):
            raise ValueError(
                "Existing CSV columns do not match revised Script 11."
            )
        rows = list(reader)

    keys = [pilot.row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Existing CSV contains duplicate method-run rows.")
    unexpected = set(keys) - set(expected_keys)
    if unexpected:
        raise ValueError(
            "Existing CSV contains rows outside the current selection. "
            "Use --overwrite or a different selection."
        )

    # PASS rows are complete; failed rows are retried on resume.
    return [row for row in rows if str(row.get("status")) == "PASS"]


def base_row(subject, run, method, loaded, epochs, noise_covariance, evoked,
             stimulation, ground_truth_mri):
    row = dict.fromkeys(pilot.FIELDNAMES, "")
    row.update(
        subject=subject,
        run=run,
        method=method,
        montage="all_good",
        epochs=len(loaded.epochs),
        all_channels=len(loaded.epochs.ch_names),
        bad_channels=len(loaded.epochs.info["bads"]),
        good_channels=len(evoked.ch_names),
        sampling_frequency_hz=float(epochs.info["sfreq"]),
        covariance_method=str(noise_covariance.get("method", "auto")),
        covariance_tmin_s=pilot.COVARIANCE_TMIN,
        covariance_tmax_s=pilot.COVARIANCE_TMAX,
        target_tmin_s=pilot.TARGET_TMIN,
        target_tmax_s=pilot.TARGET_TMAX,
        target_samples=len(evoked.times),
        stimulation_pair=stimulation.pair,
        hemisphere=stimulation.hemisphere,
        ground_truth_x_mm=float(ground_truth_mri[0] * 1000),
        ground_truth_y_mm=float(ground_truth_mri[1] * 1000),
        ground_truth_z_mm=float(ground_truth_mri[2] * 1000),
    )
    return row


def make_qc_figures(output_file, figure_dir):
    """Create lightweight QC figures; comparative analysis stays in Script 13."""
    table = pd.read_csv(output_file)
    table = table.loc[table["status"] == "PASS"].copy()
    table["selected_localization_error_mm"] = pd.to_numeric(
        table["selected_localization_error_mm"], errors="coerce"
    )
    table = table.dropna(subset=["selected_localization_error_mm"])
    if table.empty:
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Figure 1: error distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in table.groupby("method", sort=False):
        ax.hist(
            group["selected_localization_error_mm"].to_numpy(),
            bins="auto",
            alpha=0.45,
            label=method,
        )
    ax.set_xlabel("Localization error (mm)")
    ax.set_ylabel("Number of runs")
    ax.set_title("Baseline localization-error distribution")
    if table["method"].nunique() > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / f"{output_file.stem}_error_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # Figure 2: sorted run errors, useful for spotting outliers.
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in table.groupby("method", sort=False):
        values = np.sort(group["selected_localization_error_mm"].to_numpy())
        ax.plot(np.arange(1, len(values) + 1), values, marker="o",
                markersize=3, linewidth=1, label=method)
    ax.set_xlabel("Runs sorted by localization error")
    ax.set_ylabel("Localization error (mm)")
    ax.set_title("Baseline localization error across runs")
    if table["method"].nunique() > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / f"{output_file.stem}_sorted_errors.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved


def main():
    args = parse_args()
    validate_args(args)

    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    methods = pilot.resolve_methods(args.method)

    if args.case:
        cases = list(dict.fromkeys(tuple(case) for case in args.case))
    else:
        cases = discover_runs(dataset, args.task, args.subject, args.run)
    if args.max_runs is not None:
        cases = cases[:args.max_runs]
    if not cases:
        raise FileNotFoundError("No matching Localize-MI runs were found.")

    combinations = [
        (subject, run, method)
        for subject, run in cases
        for method in methods
    ]
    output_file = output_file_for(output_root, methods)
    rows = read_existing_rows(
        output_file, args.resume, args.overwrite, combinations
    )
    completed = {pilot.row_key(row) for row in rows}

    ecd_timepoint_dir = output_root / "ecd_timepoints"
    bem_dir = output_root / "bem"
    figure_dir = output_root / "figures"

    print("=" * 78)
    print("SCRIPT 12 — ADDITIONAL METHODS: ALL-RUN BASELINE")
    print("=" * 78)
    print(f"Runs     : {len(cases)}")
    print(f"Methods  : {', '.join(methods)}")
    print(f"Solutions: {len(combinations)}")
    print("Montage  : all good EEG channels")
    print("Target   : -2 to +2 ms")
    print(f"Output   : {output_file}")
    print("Baseline parameters:")
    if "Continuous-ECD" in methods:
        print(f"  Continuous-ECD: min_dist={args.ecd_min_dist:g} mm; max GOF")
    if "MxNE" in methods:
        print(
            f"  MxNE: alpha={args.mxne_alpha}; loose={args.mxne_loose:g}; "
            f"depth={args.mxne_depth:g}; n_mxne_iter=1"
        )
    if "RAP-MUSIC" in methods:
        print(f"  RAP-MUSIC: n_dipoles={args.rap_music_n_dipoles}; max GOF")
    if "LCMV" in methods:
        print(
            f"  LCMV: reg={args.lcmv_reg:g}; data_cov="
            f"{args.lcmv_data_covariance_method} "
            f"[{args.lcmv_data_tmin*1000:g}, {args.lcmv_data_tmax*1000:g}] ms; "
            "max-power; unit-noise-gain-invariant"
        )

    for run_index, (subject, run) in enumerate(cases, start=1):
        pending = [
            method for method in methods
            if (subject, run, method) not in completed
        ]
        if not pending:
            print(
                f"[{run_index:02d}/{len(cases):02d}] {subject} {run}: already complete",
                flush=True,
            )
            continue

        print(
            f"\n[{run_index:02d}/{len(cases):02d}] {subject} {run}",
            flush=True,
        )

        # If common preparation fails, record a failure for each pending method.
        try:
            loaded = pilot.load_run(
                dataset=dataset,
                subject=subject,
                run=run,
                task=args.task,
            )
            # Keep the original Epochs object for bookkeeping.  Script 11's
            # common_preprocessing() now selects good EEG channels explicitly,
            # so the actual inverse input matches Scripts 14 and 15.
            epochs = loaded.epochs.copy()

            stimulation, ground_truth_mri = pilot.get_stimulation_and_ground_truth(
                dataset, subject, loaded
            )
            transform_file = (
                dataset / "derivatives" / "sourcemodelling" / subject / "xfm"
                / f"{subject}_from-head_to-surface.h5"
            )
            surface_transform = pilot.load_surface_transform(transform_file)
            referenced, noise_covariance, evoked = pilot.common_preprocessing(
                epochs, args.quiet
            )

            data_covariance = None
            if "LCMV" in pending:
                data_covariance = pilot.compute_lcmv_data_covariance(
                    referenced,
                    args.lcmv_data_covariance_method,
                    args.lcmv_data_tmin,
                    args.lcmv_data_tmax,
                    args.quiet,
                )

            bem = None
            bem_file = None
            head_to_mri = None
            if "Continuous-ECD" in pending:
                bem, bem_file = pilot.get_or_build_bem(dataset, subject, bem_dir)

            if "Continuous-ECD" in pending or "RAP-MUSIC" in pending:
                head_to_mri = pilot.load_head_to_mri_transform(dataset, subject)

        except Exception as exc:
            for method in pending:
                row = dict.fromkeys(pilot.FIELDNAMES, "")
                row.update(
                    subject=subject,
                    run=run,
                    method=method,
                    montage="all_good",
                    status="FAIL",
                    error_type=type(exc).__name__,
                    error_message=f"Preparation failed: {exc}",
                )
                rows.append(row)
            pilot.save_rows(output_file, rows)
            print(f"  PREPARATION FAIL: {type(exc).__name__}: {exc}", flush=True)
            continue

        print(
            f"  epochs={len(loaded.epochs)}; "
            f"good channels={len(evoked.ch_names)}/{len(loaded.epochs.ch_names)}; "
            f"covariance={noise_covariance.get('method', 'auto')}",
            flush=True,
        )

        for method in pending:
            started = time.monotonic()
            row = base_row(
                subject, run, method, loaded, epochs, noise_covariance,
                evoked, stimulation, ground_truth_mri
            )

            try:
                if method == "Continuous-ECD":
                    diagnostics = pilot.run_continuous_ecd(
                        evoked,
                        noise_covariance,
                        bem,
                        head_to_mri,
                        ground_truth_mri,
                        args.ecd_min_dist,
                        args.quiet,
                    )
                    selected = diagnostics["selected"]
                    minimum_error = diagnostics["minimum_error"]
                    row.update(
                        selection_criterion="maximum_GOF",
                        selected_time_ms=float(selected["time_ms"]),
                        selected_gof_percent=float(selected["gof_percent"]),
                        selected_x_mm=float(selected["x_mm"]),
                        selected_y_mm=float(selected["y_mm"]),
                        selected_z_mm=float(selected["z_mm"]),
                        selected_localization_error_mm=float(
                            selected["localization_distance_mm"]
                        ),
                        minimum_observed_error_mm=float(
                            minimum_error["localization_distance_mm"]
                        ),
                        minimum_error_time_ms=float(minimum_error["time_ms"]),
                        gof_at_minimum_error_percent=float(
                            minimum_error["gof_percent"]
                        ),
                        ecd_min_dist_mm=args.ecd_min_dist,
                        bem_layers=3,
                        bem_ico=pilot.BEM_ICO,
                        conductivity_brain=pilot.BEM_CONDUCTIVITY[0],
                        conductivity_skull=pilot.BEM_CONDUCTIVITY[1],
                        conductivity_scalp=pilot.BEM_CONDUCTIVITY[2],
                        bem_file=str(bem_file),
                    )
                    ecd_timepoint_dir.mkdir(parents=True, exist_ok=True)
                    diagnostics["timepoint_table"].to_csv(
                        ecd_timepoint_dir
                        / f"{subject}_{run}_continuous_ecd_timepoints.csv",
                        index=False,
                    )

                elif method == "MxNE":
                    source_estimate, residual = pilot.run_mxne(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.mxne_alpha,
                        args.mxne_loose,
                        args.mxne_depth,
                        args.quiet,
                    )
                    active_sources = pilot.count_active_sources(source_estimate)
                    if active_sources == 0:
                        raise RuntimeError("MxNE returned no active cortical sources.")
                    metric = pilot.calculate_localization_metrics(
                        source_estimate,
                        loaded.forward,
                        surface_transform,
                        stimulation,
                    )
                    pilot.fill_grid_metric_fields(row, metric)
                    row.update(
                        mxne_alpha=args.mxne_alpha,
                        mxne_loose=args.mxne_loose,
                        mxne_depth=args.mxne_depth,
                        mxne_iterations=1,
                        mxne_active_sources=active_sources,
                        mxne_explained_variance_percent=pilot.explained_variance(
                            evoked, residual
                        ),
                    )

                elif method == "RAP-MUSIC":
                    dipoles, residual = pilot.run_rap_music(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.rap_music_n_dipoles,
                        args.quiet,
                    )
                    selected = pilot.select_rap_music_dipole(
                        dipoles, ground_truth_mri, head_to_mri
                    )
                    coordinate = selected["coordinate_m"]
                    row.update(
                        selection_criterion="maximum_GOF",
                        selected_time_ms=selected["time_ms"],
                        selected_gof_percent=selected["gof_percent"],
                        selected_x_mm=float(coordinate[0] * 1000),
                        selected_y_mm=float(coordinate[1] * 1000),
                        selected_z_mm=float(coordinate[2] * 1000),
                        selected_localization_error_mm=selected["distance_mm"],
                        rap_music_n_dipoles=args.rap_music_n_dipoles,
                        rap_music_gof_percent=selected["gof_percent"],
                    )

                elif method == "LCMV":
                    source_estimate = pilot.run_lcmv(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        data_covariance,
                        args.lcmv_reg,
                        args.quiet,
                    )
                    metric = pilot.calculate_localization_metrics(
                        source_estimate,
                        loaded.forward,
                        surface_transform,
                        stimulation,
                    )
                    pilot.fill_grid_metric_fields(row, metric)
                    row.update(
                        lcmv_reg=args.lcmv_reg,
                        lcmv_data_covariance_method=args.lcmv_data_covariance_method,
                        lcmv_data_tmin_s=args.lcmv_data_tmin,
                        lcmv_data_tmax_s=args.lcmv_data_tmax,
                        lcmv_data_covariance_samples=data_covariance.get("nfree", ""),
                        lcmv_pick_ori="max-power",
                        lcmv_weight_norm="unit-noise-gain-invariant",
                    )
                else:
                    raise RuntimeError(f"Unhandled method: {method}")

                row["status"] = "PASS"

            except Exception as exc:
                row.update(
                    status="FAIL",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            row["runtime_seconds"] = round(time.monotonic() - started, 3)
            # Remove an old failed row for the same key if this is a resume retry.
            rows = [r for r in rows if pilot.row_key(r) != (subject, run, method)]
            rows.append(row)
            pilot.save_rows(output_file, rows)

            if row["status"] == "PASS":
                print(
                    f"  {method:14s} PASS | "
                    f"error={float(row['selected_localization_error_mm']):.2f} mm | "
                    f"time={float(row['selected_time_ms']):+.3f} ms",
                    flush=True,
                )
                completed.add((subject, run, method))
            else:
                print(
                    f"  {method:14s} FAIL | {row['error_type']}: "
                    f"{row['error_message']}",
                    flush=True,
                )

    failures = sum(str(row.get("status")) != "PASS" for row in rows)
    passes = sum(str(row.get("status")) == "PASS" for row in rows)

    figure_paths = []
    if not args.no_figures and output_file.exists():
        figure_paths = make_qc_figures(output_file, figure_dir)

    print("\n" + "=" * 78)
    print("SCRIPT 12 COMPLETE")
    print("=" * 78)
    print(f"Rows     : {len(rows)}")
    print(f"Passes   : {passes}")
    print(f"Failures : {failures}")
    print(f"CSV      : {output_file}")
    if figure_paths:
        print("QC figures:")
        for path in figure_paths:
            print(f"  {path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
