#!/usr/bin/env python3
"""
20_batch_next_methods.py

Run the FINAL BASELINE configurations of the four Script-19 methods across
all released Localize-MI runs.

This is the all-run companion to:
    scripts/19_next_method_pilot.py

IMPORTANT DESIGN CHOICE
-----------------------
Script 20 does NOT re-implement irMxNE, Gamma-MAP, SESAME, or Champagne.
It imports Script 19 so the pilot and all-run experiments use exactly the
same scientific code, compatibility fixes, channel handling, metrics, and
baseline parameters.

BASELINE PARAMETERS
-------------------
Shared:
    montage                  = all good EEG channels
    noise covariance window  = -250 to -50 ms
    noise covariance method  = auto (MNE chooses estimator by CV)
    target window            = -2 to +2 ms

irMxNE:
    alpha                    = 40
    loose                    = 1.0
    depth                    = 0.1
    n_mxne_iter              = 10
    random_state             = 0

Gamma-MAP:
    alpha                    = 0.3
    loose                    = 1.0
    depth                    = 0.1
    xyz_same_gamma           = True
    update_mode              = 1
    maxit                    = 10000
    tol                      = 1e-6

SESAME (Pascarella-style noise parameterization):
    n_parts                  = 100
    max_n_dips               = 1
    lam                      = 0.25
    hyper_q                  = True
    random_seed              = 0
    fs                       = 0.15
    noise_std                = fs * peak(abs(target EEG))
    noise covariance         = NOT supplied to SESAME

    Pascarella et al. (2023) varied fs = 0.07, 0.13, 0.20, 0.27, 0.33
    for SSM/SESAME and did not use the prestimulus covariance for SSM.
    The paper text available to us identifies fs as the SNR/noise-variance
    control but does not expose the original code formula; this Script-20
    implementation therefore records explicitly that fs is operationalized as
    a fraction of the target-window peak absolute EEG amplitude.

Champagne:
    solver                   = Solver('champagne')
    alpha                    = 'auto'
    forward                  = restricted to exact good-channel Evoked order
    SSP                      = real MNE projector through Script-19 wrapper
    noise covariance         = supplied when supported by installed API

These are BASELINE values, not optimized values. Parameter-sensitivity
experiments belong in later scripts.

TERMINAL EXAMPLES
-----------------
Recommended: run ONE METHOD AT A TIME.

irMxNE, all released runs:
    python scripts/20_batch_next_methods.py --method irmxne --overwrite

Gamma-MAP, all released runs:
    python scripts/20_batch_next_methods.py --method gamma_map --overwrite

SESAME, all released runs:
    python scripts/20_batch_next_methods.py --method sesame --overwrite

SESAME with the suggested fs=0.20 baseline:
    python scripts/20_batch_next_methods.py \
        --method sesame --sesame-fs 0.20 --overwrite

SESAME targeted fs test on selected runs:
    python scripts/20_batch_next_methods.py \
        --method sesame \
        --case sub-02 run-02 \
        --case sub-06 run-01 \
        --sesame-fs 0.13 \
        --overwrite

Champagne, all released runs:
    python scripts/20_batch_next_methods.py --method champagne --overwrite

Resume an interrupted Gamma-MAP batch:
    python scripts/20_batch_next_methods.py --method gamma_map --resume

Quick one-run validation:
    python scripts/20_batch_next_methods.py \
        --method champagne --case sub-01 run-01 --overwrite

Run one participant only:
    python scripts/20_batch_next_methods.py \
        --method sesame --subject sub-07 --overwrite

Run all four together (supported, but not recommended because Gamma-MAP is
much slower than the others):
    python scripts/20_batch_next_methods.py --method all --overwrite

Quiet MNE logging:
    python scripts/20_batch_next_methods.py --method irmxne --overwrite --quiet

Disable QC figures:
    python scripts/20_batch_next_methods.py \
        --method gamma_map --overwrite --no-figures

DEFAULT OUTPUT ROOT
-------------------
    outputs/20_next_methods_all_runs/

Method-specific CSVs:
    20_irmxne_all_runs.csv
    20_gamma_map_all_runs.csv
    20_sesame_all_runs.csv
    20_champagne_all_runs.csv

If multiple methods are intentionally run in one command:
    20_next_methods_all_runs.csv

QC figures are lightweight only. Cross-method comparison belongs in the next
summary script.
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
SCRIPT19 = PROJECT / "scripts" / "19_next_method_pilot.py"
DEFAULT_DATASET = PROJECT / "data" / "Localize-MI"
DEFAULT_OUTPUT_ROOT = PROJECT / "outputs" / "20_next_methods_all_runs"


def load_script19():
    """Import the finalized Script 19 implementation."""
    if not SCRIPT19.is_file():
        raise FileNotFoundError(
            f"Required Script 19 was not found: {SCRIPT19}\n"
            "Keep the finalized file as scripts/19_next_method_pilot.py."
        )

    spec = importlib.util.spec_from_file_location(
        "localize_mi_script19_for_script20", SCRIPT19
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {SCRIPT19}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pilot = load_script19()


def run_sesame_with_fs(
    evoked,
    forward,
    n_parts,
    max_n_dips,
    lam,
    hyper_q,
    random_seed,
    fs,
    quiet,
):
    """Run SESAME using an explicit fs-scaled noise standard deviation.

    For this Pascarella-style Script-20 branch, SESAME is not given the
    prestimulus noise covariance. We operationalize fs as:

        noise_std = fs * max(abs(evoked.data))

    over the already-created -2 to +2 ms target Evoked. Pascarella et al.
    (2023) report fs values 0.07, 0.13, 0.20, 0.27, 0.33 and state that
    SSM/SESAME does not use the covariance used by the other inverse methods.
    The accessible paper text does not expose their original code formula, so
    this implementation records the chosen operationalization explicitly.
    """
    prepare_sesame = pilot.require_sesame()

    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("--sesame-fs must be a positive finite value.")

    signal_peak = float(np.max(np.abs(evoked.data)))
    if not np.isfinite(signal_peak) or signal_peak <= 0:
        raise RuntimeError(
            "Cannot derive SESAME noise_std because target EEG peak is "
            f"invalid: {signal_peak!r}"
        )

    noise_std = fs * signal_peak
    np.random.seed(int(random_seed))

    sesame = prepare_sesame(
        forward,
        evoked,
        n_parts=int(n_parts),
        noise_std=float(noise_std),
        top_min=float(evoked.times[0]),
        top_max=float(evoked.times[-1]),
        dip_mom_std=None,
        hyper_q=bool(hyper_q),
        noise_cov=None,
        subsample=None,
        max_n_dips=int(max_n_dips),
        lam=float(lam),
        verbose=not quiet,
    )

    import sesameeg.sesame as sesame_module

    original_particle = sesame_module.Particle

    def _compatible_particle(
        n_verts_value,
        lam_value,
        dip_mom_std=None,
        prior_locs=None,
        fixed_ori=None,
        hyper_q=None,
    ):
        if prior_locs is None:
            prior_locs = sesame.prior_locs
        if fixed_ori is None:
            fixed_ori = sesame.fixed_ori
        if hyper_q is None:
            hyper_q = sesame.hyper_q
        return original_particle(
            n_verts_value,
            lam_value,
            dip_mom_std=dip_mom_std,
            prior_locs=prior_locs,
            fixed_ori=fixed_ori,
            hyper_q=hyper_q,
        )

    sesame_module.Particle = _compatible_particle
    try:
        sesame.apply_sesame()
    finally:
        sesame_module.Particle = original_particle

    if not sesame.est_n_dips:
        raise RuntimeError("SESAME did not report an estimated source count.")

    estimated_n_dips = int(sesame.est_n_dips[-1])
    if estimated_n_dips < 1:
        raise RuntimeError(
            "SESAME selected a zero-source model for this target window. "
            "The result is retained as a FAIL because no localization "
            "coordinate can be compared with the stimulation midpoint."
        )

    source_estimate = sesame.compute_stc()
    return sesame, source_estimate, noise_std, signal_peak


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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", default="seegstim")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument(
        "--method",
        type=pilot.normalize_method,
        action="append",
        help=(
            "Repeat if needed. Choices: irmxne, gamma_map, sesame, "
            "champagne, all. Default: all."
        ),
    )
    parser.add_argument(
        "--case",
        nargs=2,
        metavar=("SUBJECT", "RUN"),
        action="append",
    )
    parser.add_argument("--subject", action="append")
    parser.add_argument("--run", action="append")
    parser.add_argument("--max-runs", type=int)

    # Baseline parameters: intentionally identical to finalized Script 19.
    parser.add_argument(
        "--irmxne-alpha", type=float, default=pilot.IRMXNE_ALPHA
    )
    parser.add_argument(
        "--irmxne-loose", type=float, default=pilot.IRMXNE_LOOSE
    )
    parser.add_argument(
        "--irmxne-depth", type=float, default=pilot.IRMXNE_DEPTH
    )
    parser.add_argument(
        "--irmxne-iterations", type=int, default=pilot.IRMXNE_ITERATIONS
    )

    parser.add_argument(
        "--gamma-alpha", type=float, default=pilot.GAMMA_ALPHA
    )
    parser.add_argument(
        "--gamma-loose", type=float, default=pilot.GAMMA_LOOSE
    )
    parser.add_argument(
        "--gamma-depth", type=float, default=pilot.GAMMA_DEPTH
    )

    parser.add_argument(
        "--sesame-n-parts", type=int, default=pilot.SESAME_N_PARTS
    )
    parser.add_argument(
        "--sesame-max-n-dips", type=int, default=pilot.SESAME_MAX_N_DIPS
    )
    parser.add_argument(
        "--sesame-lam", type=float, default=pilot.SESAME_LAM
    )
    parser.add_argument(
        "--sesame-hyper-q",
        action=argparse.BooleanOptionalAction,
        default=pilot.SESAME_HYPER_Q,
        help=(
            "Enable/disable SESAME hyper_q. Use --no-sesame-hyper-q to "
            "test hyper_q=False. Default follows Script 19 (True)."
        ),
    )
    parser.add_argument(
        "--sesame-random-seed", type=int, default=pilot.SESAME_RANDOM_SEED
    )
    parser.add_argument(
        "--sesame-fs",
        type=float,
        default=0.20,
        help=(
            "SESAME noise fraction. Script 20 operationalizes this as "
            "noise_std = fs * peak(abs(target EEG)). Pascarella et al. "
            "tested 0.07, 0.13, 0.20, 0.27, 0.33. Default: 0.20."
        ),
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

    if not np.isfinite(args.sesame_fs) or args.sesame_fs <= 0:
        raise ValueError("--sesame-fs must be a positive finite value.")

    # Reuse Script 19's scientific validation by giving it the attributes it
    # expects. Script 20 does not expose --check-dependencies.
    args.check_dependencies = False
    pilot.validate_args(args)


def output_file_for(output_root, methods):
    """Use method-specific files for the recommended one-method batches."""
    if len(methods) == 1:
        stems = {
            "irMxNE": "20_irmxne_all_runs.csv",
            "Gamma-MAP": "20_gamma_map_all_runs.csv",
            "SESAME": "20_sesame_all_runs.csv",
            "Champagne": "20_champagne_all_runs.csv",
        }
        return output_root / stems[methods[0]]

    return output_root / "20_next_methods_all_runs.csv"


def read_existing_rows(path, resume, overwrite, expected_keys):
    if resume and path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(pilot.FIELDNAMES):
                raise ValueError(
                    "Existing CSV columns do not match finalized Script 19."
                )
            rows = list(reader)

        keys = [pilot.row_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Existing CSV contains duplicate method solutions.")

        # Method-specific output files can be resumed with a smaller explicit
        # subset (--case/--subject) while retaining other rows from the same
        # method. Reject only rows from methods not requested in this file.
        requested_methods = {key[2] for key in expected_keys}
        unexpected_methods = {
            row["method"] for row in rows if row["method"] not in requested_methods
        }
        if unexpected_methods:
            raise ValueError(
                "Existing CSV contains methods outside the current selection: "
                f"{sorted(unexpected_methods)}"
            )

        expected = set(expected_keys)
        # PASS rows are complete. Failed rows in the CURRENT selection are
        # removed so they are retried. Failed rows outside a smaller explicit
        # resume selection are preserved.
        return [
            row
            for row in rows
            if row.get("status") == "PASS" or pilot.row_key(row) not in expected
        ]

    if overwrite:
        return []

    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Use --resume to continue or --overwrite to replace it."
        )

    return []


def save_rows(path, rows):
    pilot.save_rows(path, rows)


def make_qc_figures(output_file, figure_dir):
    """Create lightweight QC figures; comparative analysis comes later."""
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
    ax.set_title("Script 20 baseline localization-error distribution")
    if table["method"].nunique() > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / f"{output_file.stem}_error_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in table.groupby("method", sort=False):
        values = np.sort(group["selected_localization_error_mm"].to_numpy())
        ax.plot(
            np.arange(1, len(values) + 1),
            values,
            marker="o",
            markersize=3,
            linewidth=1,
            label=method,
        )
    ax.set_xlabel("Runs sorted by localization error")
    ax.set_ylabel("Localization error (mm)")
    ax.set_title("Script 20 baseline localization error across runs")
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
        cases = cases[: args.max_runs]
    if not cases:
        raise FileNotFoundError("No matching Localize-MI runs were found.")

    combinations = [
        (subject, run, method)
        for subject, run in cases
        for method in methods
    ]

    output_file = output_file_for(output_root, methods)
    rows = read_existing_rows(
        output_file,
        args.resume,
        args.overwrite,
        combinations,
    )
    completed = {pilot.row_key(row) for row in rows}

    figure_dir = output_root / "figures"

    print("=" * 78)
    print("SCRIPT 20 — NEXT METHODS: ALL-RUN BASELINE")
    print("=" * 78)
    print(f"Runs     : {len(cases)}")
    print(f"Methods  : {', '.join(methods)}")
    print(f"Solutions: {len(combinations)}")
    print("Montage  : all good EEG channels")
    print("Target   : -2 to +2 ms")
    print("Noise cov: -250 to -50 ms, auto (used by non-SESAME methods)")
    print(f"Output   : {output_file}")
    print("Baseline parameters:")

    if "irMxNE" in methods:
        print(
            f"  irMxNE     : alpha={args.irmxne_alpha:g}; "
            f"loose={args.irmxne_loose:g}; depth={args.irmxne_depth:g}; "
            f"n_mxne_iter={args.irmxne_iterations}"
        )
    if "Gamma-MAP" in methods:
        print(
            f"  Gamma-MAP  : alpha={args.gamma_alpha:g}; "
            f"loose={args.gamma_loose:g}; depth={args.gamma_depth:g}; "
            f"xyz_same_gamma={pilot.GAMMA_XYZ_SAME_GAMMA}; "
            f"update_mode={pilot.GAMMA_UPDATE_MODE}"
        )
    if "SESAME" in methods:
        print(
            f"  SESAME     : n_parts={args.sesame_n_parts}; "
            f"max_n_dips={args.sesame_max_n_dips}; "
            f"lam={args.sesame_lam:g}; hyper_q={args.sesame_hyper_q}; "
            f"fs={args.sesame_fs:g}; noise_std=fs*peak(|target EEG|); "
            "noise_cov=None"
        )
    if "Champagne" in methods:
        print("  Champagne  : Solver('champagne'); alpha='auto'; package defaults")

    for run_index, (subject, run) in enumerate(cases, start=1):
        pending = [
            method
            for method in methods
            if (subject, run, method) not in completed
        ]

        if not pending:
            print(
                f"[{run_index:02d}/{len(cases):02d}] "
                f"{subject} {run}: already complete",
                flush=True,
            )
            continue

        print(
            f"\n[{run_index:02d}/{len(cases):02d}] {subject} {run}",
            flush=True,
        )

        try:
            loaded = pilot.pilot.load_run(
                dataset=dataset,
                subject=subject,
                run=run,
                task=args.task,
            )
            epochs = loaded.epochs.copy()

            stimulation, ground_truth_mri = (
                pilot.pilot.get_stimulation_and_ground_truth(
                    dataset,
                    subject,
                    loaded,
                )
            )

            transform_file = (
                dataset
                / "derivatives"
                / "sourcemodelling"
                / subject
                / "xfm"
                / f"{subject}_from-head_to-surface.h5"
            )
            surface_transform = pilot.pilot.load_surface_transform(
                transform_file
            )

            referenced, noise_covariance, evoked = (
                pilot.pilot.common_preprocessing(
                    epochs,
                    args.quiet,
                )
            )

        except Exception as exc:
            for method in pending:
                row = dict.fromkeys(pilot.FIELDNAMES, "")
                row.update(
                    subject=subject,
                    run=run,
                    method=method,
                    montage="all_good",
                    mne_version=pilot.package_version("mne"),
                    sesameeg_version=pilot.package_version("sesameeg"),
                    invertmeeg_version=pilot.package_version("invertmeeg"),
                    status="FAIL",
                    error_type=type(exc).__name__,
                    error_message=f"Preparation failed: {exc}",
                )
                rows = [
                    existing
                    for existing in rows
                    if pilot.row_key(existing) != pilot.row_key(row)
                ]
                rows.append(row)
            save_rows(output_file, rows)
            print(
                f"  PREPARATION FAIL: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        print(
            f"  epochs={len(loaded.epochs)}; "
            f"good channels={len(evoked.ch_names)}/"
            f"{len(loaded.epochs.ch_names)}; "
            f"covariance={noise_covariance.get('method', 'auto')}",
            flush=True,
        )

        for method in pending:
            started = time.monotonic()
            row = pilot.base_row(
                subject,
                run,
                method,
                loaded,
                epochs,
                noise_covariance,
                evoked,
                stimulation,
                ground_truth_mri,
            )

            try:
                if method == "irMxNE":
                    source_estimate, residual = pilot.run_irmxne(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.irmxne_alpha,
                        args.irmxne_loose,
                        args.irmxne_depth,
                        args.irmxne_iterations,
                        args.quiet,
                    )

                    active_sources = pilot.count_active_sources(source_estimate)
                    if active_sources == 0:
                        raise RuntimeError(
                            "irMxNE returned no active cortical sources."
                        )

                    pilot.fill_standard_grid_metric(
                        row,
                        source_estimate,
                        loaded,
                        surface_transform,
                        stimulation,
                    )

                    row.update(
                        irmxne_alpha=args.irmxne_alpha,
                        irmxne_loose=args.irmxne_loose,
                        irmxne_depth=args.irmxne_depth,
                        irmxne_iterations=args.irmxne_iterations,
                        irmxne_random_state=pilot.IRMXNE_RANDOM_STATE,
                        irmxne_active_sources=active_sources,
                        irmxne_explained_variance_percent=(
                            pilot.pilot.explained_variance(evoked, residual)
                        ),
                    )

                elif method == "Gamma-MAP":
                    source_estimate, residual = pilot.run_gamma_map(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.gamma_alpha,
                        args.gamma_loose,
                        args.gamma_depth,
                        args.quiet,
                    )

                    active_sources = pilot.count_active_sources(source_estimate)
                    if active_sources == 0:
                        raise RuntimeError(
                            "Gamma-MAP returned no active cortical sources."
                        )

                    pilot.fill_standard_grid_metric(
                        row,
                        source_estimate,
                        loaded,
                        surface_transform,
                        stimulation,
                    )

                    row.update(
                        gamma_alpha=args.gamma_alpha,
                        gamma_loose=args.gamma_loose,
                        gamma_depth=args.gamma_depth,
                        gamma_xyz_same_gamma=pilot.GAMMA_XYZ_SAME_GAMMA,
                        gamma_update_mode=pilot.GAMMA_UPDATE_MODE,
                        gamma_maxit=pilot.GAMMA_MAXIT,
                        gamma_tol=pilot.GAMMA_TOL,
                        gamma_active_sources=active_sources,
                        gamma_explained_variance_percent=(
                            pilot.pilot.explained_variance(evoked, residual)
                        ),
                    )

                elif method == "SESAME":
                    sesame, source_estimate, sesame_noise_std, signal_peak = (
                        run_sesame_with_fs(
                            evoked,
                            loaded.forward,
                            args.sesame_n_parts,
                            args.sesame_max_n_dips,
                            args.sesame_lam,
                            args.sesame_hyper_q,
                            args.sesame_random_seed,
                            args.sesame_fs,
                            args.quiet,
                        )
                    )

                    pilot.fill_sesame_metric(
                        row,
                        source_estimate,
                        loaded,
                        surface_transform,
                        stimulation,
                    )

                    gof = sesame.goodness_of_fit()
                    dispersion = sesame.source_dispersion()

                    row.update(
                        covariance_method="not_used_by_SESAME_fs",
                        sesame_n_parts=args.sesame_n_parts,
                        sesame_max_n_dips=args.sesame_max_n_dips,
                        sesame_lam=args.sesame_lam,
                        sesame_hyper_q=args.sesame_hyper_q,
                        sesame_random_seed=args.sesame_random_seed,
                        sesame_noise_std_mode=(
                            f"fs={args.sesame_fs:g}; "
                            f"noise_std={sesame_noise_std:.12e}; "
                            f"target_peak={signal_peak:.12e}"
                        ),
                        sesame_dip_mom_std_mode="estimated_by_SESAME",
                        sesame_noise_covariance_supplied=False,
                        sesame_estimated_n_dips=int(sesame.est_n_dips[-1]),
                        sesame_gof_percent=float(gof) * 100.0,
                        sesame_source_dispersion_mm=float(dispersion),
                    )

                elif method == "Champagne":
                    source_estimate, signature, covariance_supplied = (
                        pilot.run_champagne(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            args.quiet,
                        )
                    )

                    pilot.fill_standard_grid_metric(
                        row,
                        source_estimate,
                        loaded,
                        surface_transform,
                        stimulation,
                    )

                    row.update(
                        champagne_solver_id="champagne",
                        champagne_alpha_mode=pilot.CHAMPAGNE_ALPHA,
                        champagne_noise_covariance_supplied=(
                            covariance_supplied
                        ),
                        champagne_make_operator_signature=signature,
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

            row["runtime_seconds"] = round(
                time.monotonic() - started,
                3,
            )

            rows = [
                existing
                for existing in rows
                if pilot.row_key(existing) != pilot.row_key(row)
            ]
            rows.append(row)
            save_rows(output_file, rows)
            if row["status"] == "PASS":
                completed.add((subject, run, method))

            if row["status"] == "PASS":
                print(
                    f"  {method:10s} PASS | "
                    f"error={float(row['selected_localization_error_mm']):.2f} mm | "
                    f"time={pilot.format_result_time(row)} | "
                    f"runtime={float(row['runtime_seconds']):.1f} s",
                    flush=True,
                )
            else:
                print(
                    f"  {method:10s} FAIL | "
                    f"{row['error_type']}: {row['error_message']}",
                    flush=True,
                )

    save_rows(output_file, rows)

    if not args.no_figures:
        saved_figures = make_qc_figures(output_file, figure_dir)
    else:
        saved_figures = []

    table = pd.DataFrame(rows)
    passes = int((table["status"] == "PASS").sum()) if not table.empty else 0
    failures = int((table["status"] != "PASS").sum()) if not table.empty else 0

    print("\n" + "=" * 78)
    print("SCRIPT 20 COMPLETE")
    print("=" * 78)
    print(f"Rows     : {len(rows)}")
    print(f"Passes   : {passes}")
    print(f"Failures : {failures}")
    print(f"Output   : {output_file}")

    if saved_figures:
        print("QC figures:")
        for path in saved_figures:
            print(f"  {path}")

    passed = table.loc[table["status"] == "PASS"].copy()
    if not passed.empty:
        passed["selected_localization_error_mm"] = pd.to_numeric(
            passed["selected_localization_error_mm"], errors="coerce"
        )
        passed = passed.dropna(subset=["selected_localization_error_mm"])

    if not passed.empty:
        summary = (
            passed.groupby("method", sort=False)["selected_localization_error_mm"]
            .agg(
                runs="count",
                mean_mm="mean",
                median_mm="median",
                minimum_mm="min",
                maximum_mm="max",
            )
            .reset_index()
        )
        for column in (
            "mean_mm",
            "median_mm",
            "minimum_mm",
            "maximum_mm",
        ):
            summary[column] = summary[column].round(2)

        print("\nBASELINE SUMMARY (DESCRIPTIVE ONLY)")
        print(summary.to_string(index=False))

    # Keep failed rows in the CSV and exit normally so long batches can be
    # inspected/resumed without losing successful work.


if __name__ == "__main__":
    main()
