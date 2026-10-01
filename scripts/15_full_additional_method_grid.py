#!/usr/bin/env python3
"""
15_full_additional_method_grid.py

Run the full Localize-MI parameter grid for the four additional inverse
methods after the Script-14 screening pilot.

This script is intended to be run ONE METHOD AT A TIME.

METHODS
-------
Continuous-ECD
MxNE
RAP-MUSIC
LCMV

SCIENTIFIC DESIGN
-----------------

Shared:
    Runs:
        all 61 released Localize-MI runs when --all-runs is supplied

    Montages:
        all_good, 128, 64, 32

        The reduced montages reuse Script 09's fixed greedy spatial channel
        selection and bad-channel replacement procedure. They are NOT claimed
        to reproduce unpublished manufacturer/author channel lists.

    EEG reference:
        average reference is applied before reduced-montage channel picking,
        matching Script 09.

    Target window:
        -2 to +2 ms

    Noise covariance:
        -250 to -50 ms, method="auto"

    Evaluation:
        localization error is calculated against the known Localize-MI
        stimulation midpoint.

Continuous-ECD:
    min_dist (mm):
        1, 2.5, 5, 7.5, 10, 15

    Primary solution:
        maximum goodness-of-fit (GOF) across target-window samples

    Position:
        continuous, not restricted to the cortical source grid

    BEM:
        participant-specific 3-layer BEM reconstructed from the released
        Localize-MI GIFTI surfaces

    IMPORTANT:
        fitted MNE Dipole positions are in HEAD coordinates and are transformed
        into the released surface/MRI coordinate frame before localization
        error is calculated.

MxNE:
    alpha:
        30, 40, 50

    loose:
        0.1, 0.2, ..., 1.0

    depth:
        0.1, 0.2, ..., 1.0

    n_mxne_iter:
        1

    This is a FULL FACTORIAL grid:
        3 alpha x 10 loose x 10 depth = 300 configurations per montage.

RAP-MUSIC:
    n_dipoles:
        FIXED at 1

    This is deliberate. Localize-MI provides one known stimulation location,
    so RAP-MUSIC is kept as a one-source model rather than artificially asking
    it to find additional sources for which no ground truth exists.

    The only experimental factor in Script 15 is therefore montage.

    IMPORTANT:
        fitted RAP-MUSIC Dipole positions are in HEAD coordinates and are
        transformed to the surface/MRI frame before error calculation.

LCMV:
    regularization:
        0, 0.01, 0.05, 0.10, 0.20

    symmetric data-covariance half-window:
        2, 5, 10 ms
        i.e. [-2,+2], [-5,+5], [-10,+10] ms

    data-covariance estimator:
        empirical, shrunk, diagonal_fixed

    pick_ori:
        max-power (fixed)

    reduce_rank:
        True (fixed)

    weight_norm:
        one FIXED value for the full experiment.
        Default: unit-noise-gain-invariant

        Script 14 can be used to justify a different fixed normalization.
        Do NOT turn this into another full-grid dimension unless explicitly
        planned.

DEFAULT FULL-GRID COUNTS
------------------------
Across 61 runs and four montages:

    Continuous-ECD:
        6 x 4 x 61 = 1,464 solutions

    MxNE:
        3 x 10 x 10 x 4 x 61 = 73,200 solutions

    RAP-MUSIC:
        1 x 4 x 61 = 244 solutions

    LCMV:
        5 x 3 x 3 x 4 x 61 = 10,980 solutions

    TOTAL:
        85,888 solutions

OUTPUT
------
Default root:

    outputs/15_full_additional_method_grid/

Each method gets its own directory:

    continuous_ecd/
        grid_results.csv
        grid_manifest.json
        channel_manifests/

    mxne/
        grid_results.csv
        grid_manifest.json
        channel_manifests/

    rap_music/
        grid_results.csv
        grid_manifest.json
        channel_manifests/

    lcmv/
        grid_results.csv
        grid_manifest.json
        channel_manifests/

Shared cached BEM solutions are stored in:

    outputs/15_full_additional_method_grid/bem/

No large summary figures are made here. Script 15 is the computation script.
Script 16 should create the marginal-effect, interaction, participant, montage,
and lowest-median-observed-configuration summaries and figures.

TERMINAL EXAMPLES
-----------------

1) Inspect counts before computing:

    python scripts/15_full_additional_method_grid.py \
        --method rap_music --all-runs --dry-run

    python scripts/15_full_additional_method_grid.py \
        --method ecd --all-runs --dry-run

    python scripts/15_full_additional_method_grid.py \
        --method mxne --all-runs --dry-run

    python scripts/15_full_additional_method_grid.py \
        --method lcmv --all-runs --dry-run


2) One-run baseline validation first:

Continuous-ECD baseline:
    python scripts/15_full_additional_method_grid.py \
        --method ecd \
        --case sub-01 run-01 \
        --montage all_good \
        --ecd-min-dist 5 \
        --overwrite

MxNE baseline:
    python scripts/15_full_additional_method_grid.py \
        --method mxne \
        --case sub-01 run-01 \
        --montage all_good \
        --mxne-alpha 40 \
        --mxne-loose 1.0 \
        --mxne-depth 0.1 \
        --overwrite

RAP-MUSIC baseline:
    python scripts/15_full_additional_method_grid.py \
        --method rap_music \
        --case sub-01 run-01 \
        --montage all_good \
        --overwrite

LCMV baseline:
    python scripts/15_full_additional_method_grid.py \
        --method lcmv \
        --case sub-01 run-01 \
        --montage all_good \
        --lcmv-reg 0.05 \
        --lcmv-window-ms 5 \
        --lcmv-covariance-method shrunk \
        --overwrite


3) Run the complete grids, one method at a time:

RAP-MUSIC (smallest):
    python scripts/15_full_additional_method_grid.py \
        --method rap_music --all-runs --overwrite

Continuous-ECD:
    python scripts/15_full_additional_method_grid.py \
        --method ecd --all-runs --overwrite

LCMV:
    python scripts/15_full_additional_method_grid.py \
        --method lcmv --all-runs --overwrite

MxNE (largest):
    python scripts/15_full_additional_method_grid.py \
        --method mxne --all-runs --overwrite


4) Resume an interrupted run:

    python scripts/15_full_additional_method_grid.py \
        --method mxne --all-runs --resume


5) Quiet MNE output:

    python scripts/15_full_additional_method_grid.py \
        --method rap_music --all-runs --overwrite --quiet


6) Restrict a full-grid run for debugging:

    python scripts/15_full_additional_method_grid.py \
        --method lcmv \
        --case sub-01 run-01 \
        --montage all_good \
        --lcmv-reg 0.05 \
        --lcmv-window-ms 5 \
        --lcmv-covariance-method shrunk \
        --overwrite --quiet

IMPORTANT EXECUTION RULE
------------------------
Do not run two processes that write to the same method-specific output
directory at the same time.

"Best" configurations found later are descriptive lowest-error /
lowest-median observed configurations on the data used to summarize them.
They are not independently validated optima.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import warnings

import mne
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT09 = PROJECT / "scripts" / "09_full_parameter_grid.py"
SCRIPT11 = PROJECT / "scripts" / "11_additional_method_pilot.py"

DEFAULT_OUTPUT_ROOT = (
    PROJECT / "outputs" / "15_full_additional_method_grid"
)

DEFAULT_PILOT_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)

MONTAGES = ("all_good", "128", "64", "32")

ECD_MIN_DISTANCES = (1.0, 2.5, 5.0, 7.5, 10.0, 15.0)

MXNE_ALPHAS = (30.0, 40.0, 50.0)
MXNE_LOOSE = tuple(round(i / 10, 1) for i in range(1, 11))
MXNE_DEPTH = tuple(round(i / 10, 1) for i in range(1, 11))

RAP_MUSIC_N_DIPOLES = 1

LCMV_REGULARIZATION = (0.0, 0.01, 0.05, 0.10, 0.20)
LCMV_WINDOW_MS = (2.0, 5.0, 10.0)
LCMV_COVARIANCE_METHODS = (
    "empirical",
    "shrunk",
    "diagonal_fixed",
)
DEFAULT_LCMV_WEIGHT_NORM = "unit-noise-gain-invariant"

TARGET = (-0.002, 0.002)
NOISE = (-0.250, -0.050)

METHOD_NAMES = {
    "ecd": "Continuous-ECD",
    "mxne": "MxNE",
    "rap_music": "RAP-MUSIC",
    "lcmv": "LCMV",
}

METHOD_SLUGS = {
    "Continuous-ECD": "continuous_ecd",
    "MxNE": "mxne",
    "RAP-MUSIC": "rap_music",
    "LCMV": "lcmv",
}

FIELDS = (
    "subject",
    "run",
    "method",
    "configuration_id",
    "montage",
    "montage_definition",
    "channels",
    "replacement_count",
    "epochs",
    "sampling_frequency_hz",

    "covariance_method",
    "covariance_tmin_s",
    "covariance_tmax_s",
    "target_tmin_s",
    "target_tmax_s",
    "target_samples",

    "stimulation_pair",
    "hemisphere",
    "ground_truth_x_mm",
    "ground_truth_y_mm",
    "ground_truth_z_mm",

    # Continuous-ECD
    "ecd_min_dist_mm",
    "ecd_selected_gof_percent",
    "ecd_minimum_observed_error_mm",
    "ecd_minimum_error_time_ms",
    "ecd_gof_at_minimum_error_percent",
    "bem_file",

    # MxNE
    "mxne_alpha",
    "mxne_loose",
    "mxne_depth",
    "mxne_iterations",
    "mxne_active_sources",
    "mxne_explained_variance_percent",

    # RAP-MUSIC
    "rap_music_n_dipoles",
    "rap_music_gof_percent",

    # LCMV
    "lcmv_reg",
    "lcmv_data_covariance_method",
    "lcmv_window_half_ms",
    "lcmv_data_tmin_s",
    "lcmv_data_tmax_s",
    "lcmv_data_covariance_samples",
    "lcmv_pick_ori",
    "lcmv_weight_norm",
    "lcmv_reduce_rank",

    # Common localization result
    "selection_criterion",
    "selected_time_ms",
    "selected_x_mm",
    "selected_y_mm",
    "selected_z_mm",
    "localization_distance_mm",

    # Grid-based geometry diagnostics
    "peak_vertex",
    "nearest_source_distance_mm",
    "geometric_excess_mm",

    # Baseline check
    "script12_baseline_mm",
    "change_from_script12_baseline_mm",

    "status",
    "error_type",
    "error_message",
    "runtime_seconds",
)


def load_python_module(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)

    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


grid09 = load_python_module(SCRIPT09, "localize_mi_script09")
pilot = load_python_module(SCRIPT11, "localize_mi_script11")


def normalize_method(value):
    key = str(value).strip().replace("-", "_").lower()

    aliases = {
        "ecd": "Continuous-ECD",
        "continuous_ecd": "Continuous-ECD",
        "continuousecd": "Continuous-ECD",
        "mxne": "MxNE",
        "rap_music": "RAP-MUSIC",
        "rapmusic": "RAP-MUSIC",
        "music": "RAP-MUSIC",
        "lcmv": "LCMV",
    }

    if key not in aliases:
        raise argparse.ArgumentTypeError(
            "Choose --method ecd, mxne, rap_music, or lcmv."
        )

    return aliases[key]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT / "data" / "Localize-MI",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--method",
        required=True,
        type=normalize_method,
    )

    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        metavar=("SUBJECT", "RUN"),
    )

    parser.add_argument(
        "--all-runs",
        action="store_true",
    )

    parser.add_argument(
        "--montage",
        choices=MONTAGES,
        action="append",
    )

    parser.add_argument(
        "--template-subject",
        default="sub-01",
    )

    parser.add_argument(
        "--template-run",
        default="run-01",
    )

    # ECD restrictions
    parser.add_argument(
        "--ecd-min-dist",
        type=float,
        action="append",
    )

    # MxNE restrictions
    parser.add_argument(
        "--mxne-alpha",
        type=float,
        action="append",
    )
    parser.add_argument(
        "--mxne-loose",
        type=float,
        action="append",
    )
    parser.add_argument(
        "--mxne-depth",
        type=float,
        action="append",
    )

    # LCMV restrictions
    parser.add_argument(
        "--lcmv-reg",
        type=float,
        action="append",
    )
    parser.add_argument(
        "--lcmv-window-ms",
        type=float,
        action="append",
    )
    parser.add_argument(
        "--lcmv-covariance-method",
        action="append",
        choices=LCMV_COVARIANCE_METHODS,
    )
    parser.add_argument(
        "--lcmv-weight-norm",
        default=DEFAULT_LCMV_WEIGHT_NORM,
        choices=(
            "unit-noise-gain",
            "unit-noise-gain-invariant",
            "nai",
            "none",
        ),
        help=(
            "One fixed normalization for the entire LCMV full grid. "
            "'none' passes weight_norm=None."
        ),
    )

    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT / "outputs" / "12_additional_methods_all_runs",
        help=(
            "Script-12 result directory used only to validate the all-good "
            "baseline configuration. If the relevant file is absent, the "
            "grid still runs but no Script-12 comparison is recorded."
        ),
    )

    parser.add_argument(
        "--baseline-tolerance-mm",
        type=float,
        default=0.05,
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def ordered_unique(values):
    return list(dict.fromkeys(values))


def choose_numeric(raw, allowed, name):
    if raw is None:
        return list(allowed)

    result = ordered_unique(float(value) for value in raw)

    for value in result:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value: {value}")

        if not any(np.isclose(value, allowed_value) for allowed_value in allowed):
            raise ValueError(
                f"{name}={value} is outside the planned grid {allowed}."
            )

    canonical = []
    for value in result:
        canonical.append(
            next(
                float(allowed_value)
                for allowed_value in allowed
                if np.isclose(value, allowed_value)
            )
        )

    return ordered_unique(canonical)


def choose_strings(raw, allowed, name):
    if raw is None:
        return list(allowed)

    result = ordered_unique(raw)

    invalid = [value for value in result if value not in allowed]
    if invalid:
        raise ValueError(
            f"{name} must be chosen from {allowed}; got {invalid}"
        )

    return result


def discover_cases(dataset):
    return grid09.discover_cases(dataset)


def configuration_grid(method, args):
    if method == "Continuous-ECD":
        distances = choose_numeric(
            args.ecd_min_dist,
            ECD_MIN_DISTANCES,
            "ECD min_dist",
        )

        return [
            {
                "configuration_id": f"ecd_min_dist_{value:g}",
                "ecd_min_dist_mm": value,
            }
            for value in distances
        ]

    if method == "MxNE":
        alphas = choose_numeric(
            args.mxne_alpha,
            MXNE_ALPHAS,
            "MxNE alpha",
        )
        loose_values = choose_numeric(
            args.mxne_loose,
            MXNE_LOOSE,
            "MxNE loose",
        )
        depth_values = choose_numeric(
            args.mxne_depth,
            MXNE_DEPTH,
            "MxNE depth",
        )

        rows = []
        for alpha in alphas:
            for loose in loose_values:
                for depth in depth_values:
                    rows.append(
                        {
                            "configuration_id": (
                                f"mxne_a{alpha:g}"
                                f"_l{loose:.1f}"
                                f"_d{depth:.1f}"
                            ),
                            "mxne_alpha": alpha,
                            "mxne_loose": loose,
                            "mxne_depth": depth,
                        }
                    )
        return rows

    if method == "RAP-MUSIC":
        return [
            {
                "configuration_id": "rap_music_n1",
                "rap_music_n_dipoles": 1,
            }
        ]

    if method == "LCMV":
        regs = choose_numeric(
            args.lcmv_reg,
            LCMV_REGULARIZATION,
            "LCMV reg",
        )
        windows = choose_numeric(
            args.lcmv_window_ms,
            LCMV_WINDOW_MS,
            "LCMV window",
        )
        covariances = choose_strings(
            args.lcmv_covariance_method,
            LCMV_COVARIANCE_METHODS,
            "LCMV covariance method",
        )

        rows = []
        for reg in regs:
            for window_ms in windows:
                for covariance_method in covariances:
                    rows.append(
                        {
                            "configuration_id": (
                                f"lcmv_r{reg:g}"
                                f"_w{window_ms:g}"
                                f"_{covariance_method}"
                            ),
                            "lcmv_reg": reg,
                            "lcmv_window_half_ms": window_ms,
                            "lcmv_data_covariance_method": covariance_method,
                            "lcmv_weight_norm": (
                                None
                                if args.lcmv_weight_norm == "none"
                                else args.lcmv_weight_norm
                            ),
                        }
                    )
        return rows

    raise RuntimeError(method)


def result_key_from_values(subject, run, montage, configuration_id):
    return (
        str(subject),
        str(run),
        str(montage),
        str(configuration_id),
    )


def result_key_from_row(row):
    return result_key_from_values(
        row["subject"],
        row["run"],
        row["montage"],
        row["configuration_id"],
    )


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def rewrite_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)


def append_row(path, row):
    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def load_script12_baseline(baseline_root, method):
    file_map = {
        "Continuous-ECD": "12_continuous_ecd_all_runs.csv",
        "MxNE": "12_mxne_all_runs.csv",
        "RAP-MUSIC": "12_rap_music_all_runs.csv",
        "LCMV": "12_lcmv_all_runs.csv",
    }

    path = baseline_root / file_map[method]

    if not path.is_file():
        return {}, None

    table = pd.read_csv(path)

    required = {
        "subject",
        "run",
        "status",
        "selected_localization_error_mm",
    }

    if required - set(table.columns):
        raise ValueError(
            f"Script-12 baseline file lacks columns "
            f"{sorted(required - set(table.columns))}: {path}"
        )

    table = table.loc[
        table["status"].astype(str).str.upper().eq("PASS")
    ].copy()

    result = {}

    for record in table.itertuples(index=False):
        key = (str(record.subject), str(record.run))

        if key in result:
            raise ValueError(
                f"Duplicate Script-12 baseline for {key}: {path}"
            )

        result[key] = float(
            record.selected_localization_error_mm
        )

    return result, path


def is_script12_baseline(method, montage, config):
    if montage != "all_good":
        return False

    if method == "Continuous-ECD":
        return np.isclose(
            config["ecd_min_dist_mm"],
            5.0,
        )

    if method == "MxNE":
        return (
            np.isclose(config["mxne_alpha"], 40.0)
            and np.isclose(config["mxne_loose"], 1.0)
            and np.isclose(config["mxne_depth"], 0.1)
        )

    if method == "RAP-MUSIC":
        return True

    if method == "LCMV":
        return (
            np.isclose(config["lcmv_reg"], 0.05)
            and np.isclose(
                config["lcmv_window_half_ms"],
                5.0,
            )
            and config["lcmv_data_covariance_method"] == "shrunk"
            and config["lcmv_weight_norm"]
            == "unit-noise-gain-invariant"
        )

    return False


def count_active_sources(source_estimate):
    return int(
        sum(
            len(vertices)
            for vertices in source_estimate.vertices
        )
    )


def explained_variance(evoked, residual):
    names = [
        name
        for name in evoked.ch_names
        if name not in evoked.info["bads"]
    ]

    observed = evoked.copy().pick(names).data
    remaining = residual.copy().pick(names).data

    denominator = float(np.sum(observed ** 2))

    if denominator == 0:
        return np.nan

    return 100.0 * (
        1.0
        - float(np.sum(remaining ** 2)) / denominator
    )


def compute_lcmv_data_covariance(
    epochs,
    method,
    tmin,
    tmax,
    quiet,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Epochs are not baseline corrected.*",
            category=RuntimeWarning,
        )

        return mne.compute_covariance(
            epochs,
            tmin=tmin,
            tmax=tmax,
            method=method,
            verbose=not quiet,
        )


def run_lcmv(
    evoked,
    forward,
    noise_covariance,
    data_covariance,
    reg,
    weight_norm,
    quiet,
):
    filters = mne.beamformer.make_lcmv(
        evoked.info,
        forward,
        data_covariance,
        reg=float(reg),
        noise_cov=noise_covariance,
        pick_ori="max-power",
        weight_norm=weight_norm,
        rank=None,
        reduce_rank=True,
        verbose=not quiet,
    )

    source_estimate = mne.beamformer.apply_lcmv(
        evoked,
        filters,
        verbose=not quiet,
    )

    data = np.asarray(source_estimate.data)

    if np.iscomplexobj(data):
        imaginary_scale = float(
            np.max(np.abs(data.imag))
        )
        real_scale = float(
            np.max(np.abs(data.real))
        )
        tolerance = max(
            1e-12,
            real_scale * 1e-8,
        )

        if imaginary_scale > tolerance:
            raise RuntimeError(
                "LCMV returned materially complex source data: "
                f"imag={imaginary_scale:.3e}, "
                f"real={real_scale:.3e}"
            )

        source_estimate = source_estimate.copy()
        source_estimate._data = data.real

    return source_estimate


def fill_grid_metric(row, metric):
    distance = float(metric.localization_distance_mm)

    row.update(
        selection_criterion="maximum_source_amplitude",
        selected_time_ms=float(metric.peak_time * 1000),
        selected_x_mm=float(metric.peak_coordinate[0] * 1000),
        selected_y_mm=float(metric.peak_coordinate[1] * 1000),
        selected_z_mm=float(metric.peak_coordinate[2] * 1000),
        localization_distance_mm=distance,
        peak_vertex=int(metric.peak_vertex),
        nearest_source_distance_mm=float(
            metric.nearest_source_distance_mm
        ),
        geometric_excess_mm=(
            distance
            - float(metric.nearest_source_distance_mm)
        ),
    )


def base_row(
    subject,
    run,
    method,
    montage,
    configuration_id,
    channels,
    replacements,
    loaded,
    evoked,
    noise_covariance,
    stimulation,
    ground_truth_mri,
):
    row = dict.fromkeys(FIELDS, "")

    row.update(
        subject=subject,
        run=run,
        method=method,
        configuration_id=configuration_id,
        montage=montage,
        montage_definition=(
            "all_good"
            if montage == "all_good"
            else "fixed_greedy_spatial_v1"
        ),
        channels=len(channels),
        replacement_count=len(replacements),
        epochs=len(loaded.epochs),
        sampling_frequency_hz=float(
            loaded.epochs.info["sfreq"]
        ),
        covariance_method=str(
            noise_covariance.get("method", "auto")
        ),
        covariance_tmin_s=NOISE[0],
        covariance_tmax_s=NOISE[1],
        target_tmin_s=TARGET[0],
        target_tmax_s=TARGET[1],
        target_samples=len(evoked.times),
        stimulation_pair=stimulation.pair,
        hemisphere=stimulation.hemisphere,
        ground_truth_x_mm=float(
            ground_truth_mri[0] * 1000
        ),
        ground_truth_y_mm=float(
            ground_truth_mri[1] * 1000
        ),
        ground_truth_z_mm=float(
            ground_truth_mri[2] * 1000
        ),
    )

    return row


def main():
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError(
            "Choose --resume or --overwrite, not both."
        )

    if args.case and args.all_runs:
        raise ValueError(
            "Choose explicit --case values OR --all-runs, not both."
        )

    if args.baseline_tolerance_mm < 0:
        raise ValueError(
            "--baseline-tolerance-mm must be non-negative."
        )

    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    method = args.method
    method_slug = METHOD_SLUGS[method]
    method_output = output_root / method_slug

    if not dataset.is_dir():
        raise NotADirectoryError(dataset)

    if args.all_runs:
        cases = discover_cases(dataset)
    elif args.case:
        cases = ordered_unique(
            tuple(item)
            for item in args.case
        )
    else:
        cases = list(DEFAULT_PILOT_CASES)

    montages = choose_strings(
        args.montage,
        MONTAGES,
        "montage",
    )

    configurations = configuration_grid(
        method,
        args,
    )

    montages.sort(key=MONTAGES.index)

    # Put the Script-12 baseline first when it is part of the selected grid.
    configurations.sort(
        key=lambda config: (
            not is_script12_baseline(
                method,
                "all_good",
                config,
            ),
            config["configuration_id"],
        )
    )

    total = (
        len(cases)
        * len(montages)
        * len(configurations)
    )

    print("=" * 78)
    print("SCRIPT 15 — FULL ADDITIONAL-METHOD PARAMETER GRID")
    print("=" * 78)
    print(f"Method         : {method}")
    print(f"Runs           : {len(cases)}")
    print(f"Montages       : {', '.join(montages)}")
    print(f"Configurations : {len(configurations)}")
    print(f"Solutions      : {total:,}")
    print(f"Output         : {method_output}")

    if method == "RAP-MUSIC":
        print("RAP-MUSIC     : n_dipoles=1 FIXED")

    if method == "LCMV":
        print(
            "LCMV weight norm: "
            + (
                "None"
                if configurations[0]["lcmv_weight_norm"] is None
                else str(
                    configurations[0]["lcmv_weight_norm"]
                )
            )
        )

    if args.dry_run:
        print("\nConfiguration IDs:")
        for config in configurations:
            print(f"  {config['configuration_id']}")
        return

    if method_output.exists():
        if args.overwrite:
            shutil.rmtree(method_output)
        elif not args.resume:
            raise FileExistsError(
                f"Output exists: {method_output}\n"
                "Use --resume or --overwrite."
            )

    method_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    channels_dir = (
        method_output / "channel_manifests"
    )
    channels_dir.mkdir(exist_ok=True)

    results_path = (
        method_output / "grid_results.csv"
    )
    manifest_path = (
        method_output / "grid_manifest.json"
    )

    template = pilot.load_run(
        dataset=dataset,
        subject=args.template_subject,
        run=args.template_run,
        task="seegstim",
    )

    (
        template_names,
        template_xyz,
        nominal_sets,
    ) = grid09.fixed_spatial_subsets(
        template.forward
    )

    template_payload = {
        "names": template_names,
        "coordinates_m": template_xyz.tolist(),
        "nominal": nominal_sets,
    }

    template_digest = hashlib.sha256(
        json.dumps(
            template_payload,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    configuration_payload = {
        "version": 1,
        "dataset": str(dataset),
        "method": method,
        "cases": [list(case) for case in cases],
        "montages": montages,
        "configurations": configurations,
        "target_s": list(TARGET),
        "noise_covariance_s": list(NOISE),
        "rap_music_n_dipoles": RAP_MUSIC_N_DIPOLES,
        "lcmv_pick_ori": "max-power",
        "lcmv_reduce_rank": True,
        "template_subject": args.template_subject,
        "template_run": args.template_run,
        "template_sha256": template_digest,
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
    }

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"--resume requires {manifest_path}"
            )

        observed_manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if observed_manifest != configuration_payload:
            raise ValueError(
                "Existing grid manifest differs from the current "
                "arguments. Use the original arguments or a new "
                "output root."
            )
    else:
        atomic_json(
            manifest_path,
            configuration_payload,
        )

        atomic_json(
            method_output
            / "nominal_channel_sets.json",
            {
                "description": (
                    "Script 09 greedy spatial montage selection; "
                    "not claimed to be exact author/manufacturer "
                    "channel lists."
                ),
                "reference": (
                    f"{args.template_subject} "
                    f"{args.template_run}"
                ),
                "sha256": template_digest,
                "sets": nominal_sets,
            },
        )

    existing_rows = []

    if args.resume and results_path.is_file():
        with results_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames != list(FIELDS):
                raise ValueError(
                    "Existing Script-15 CSV header does not match "
                    "this script."
                )

            existing_rows = list(reader)

        # Preserve successful rows and retry failures.
        successful_rows = [
            row
            for row in existing_rows
            if str(row.get("status")).upper()
            == "PASS"
        ]

        keys = [
            result_key_from_row(row)
            for row in successful_rows
        ]

        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate PASS rows found in existing results."
            )

        rewrite_rows(
            results_path,
            successful_rows,
        )

        existing_rows = successful_rows

    elif not results_path.exists():
        rewrite_rows(
            results_path,
            [],
        )

    completed = {
        result_key_from_row(row)
        for row in existing_rows
    }

    print(
        f"Already complete: {len(completed):,}/{total:,}",
        flush=True,
    )

    baseline_root = (
        args.baseline_root.expanduser().resolve()
    )

    (
        script12_baselines,
        baseline_file,
    ) = load_script12_baseline(
        baseline_root,
        method,
    )

    if baseline_file is None:
        print(
            "WARNING: Script-12 baseline file not found; "
            "baseline consistency values will be blank.",
            flush=True,
        )
    else:
        print(
            f"Script-12 baseline: {baseline_file}",
            flush=True,
        )

    bem_dir = output_root / "bem"

    new_failures = 0
    new_passes = 0

    for run_index, (subject, run) in enumerate(
        cases,
        start=1,
    ):
        run_has_pending = any(
            result_key_from_values(
                subject,
                run,
                montage,
                config["configuration_id"],
            )
            not in completed
            for montage in montages
            for config in configurations
        )

        if not run_has_pending:
            print(
                f"[{run_index:02d}/{len(cases):02d}] "
                f"{subject} {run}: already complete",
                flush=True,
            )
            continue

        print(
            f"\n[{run_index:02d}/{len(cases):02d}] "
            f"{subject} {run}",
            flush=True,
        )

        loaded = pilot.load_run(
            dataset=dataset,
            subject=subject,
            run=run,
            task="seegstim",
        )

        if set(loaded.epochs.ch_names) != set(
            template_names
        ):
            raise ValueError(
                f"EEG channel names differ from the Script-09 "
                f"template for {subject} {run}."
            )

        (
            stimulation,
            ground_truth_mri,
        ) = pilot.get_stimulation_and_ground_truth(
            dataset,
            subject,
            loaded,
        )

        surface_transform_path = (
            dataset
            / "derivatives"
            / "sourcemodelling"
            / subject
            / "xfm"
            / f"{subject}_from-head_to-surface.h5"
        )

        surface_transform = (
            pilot.load_surface_transform(
                surface_transform_path
            )
        )

        head_to_mri = None
        bem = None
        bem_file = None

        if method in (
            "Continuous-ECD",
            "RAP-MUSIC",
        ):
            head_to_mri = (
                pilot.load_head_to_mri_transform(
                    dataset,
                    subject,
                )
            )

        if method == "Continuous-ECD":
            bem, bem_file = pilot.get_or_build_bem(
                dataset,
                subject,
                bem_dir,
            )

        # Canonical additional-method preprocessing: physically keep good EEG
        # channels first, then apply the average reference across that good-channel
        # set. Reduced montages are picked only after this reference is formed.
        good_epochs = loaded.epochs.copy()
        good_epochs.pick("eeg", exclude="bads")
        good_names = list(good_epochs.ch_names)

        # Match the reduced-montage design: reference across all good channels
        # BEFORE selecting 128/64/32 subsets. For all_good, this object is used
        # unchanged, matching Script 11/12 common_preprocessing().
        referenced = pilot.apply_average_reference(
            good_epochs
        )

        for montage in montages:
            montage_has_pending = any(
                result_key_from_values(
                    subject,
                    run,
                    montage,
                    config["configuration_id"],
                )
                not in completed
                for config in configurations
            )

            if not montage_has_pending:
                continue

            if montage == "all_good":
                actual_channels = good_names
                replacements = []
            else:
                (
                    actual_channels,
                    replacements,
                ) = grid09.replace_bad_nominal(
                    nominal_sets[montage],
                    template_names,
                    template_xyz,
                    good_names,
                )

            # Apply the average reference before montage selection, then
            # physically keep only the channels used by this montage.
            #
            # This is especially important for Continuous-ECD / RAP-MUSIC:
            # mne.fit_dipole can be numerically sensitive to whether unused
            # bad channels remain in Info even when they are excluded from
            # the actual fit.  All montages therefore enter the inverse
            # method with exactly their selected good channels and no bads.
            if montage == "all_good":
                montage_epochs = referenced.copy()
            else:
                montage_epochs = (
                    referenced.copy().pick(
                        actual_channels
                    )
                )

            if montage_epochs.info["bads"]:
                raise AssertionError(
                    "Selected montage still contains bad channels."
                )

            channel_payload = {
                "subject": subject,
                "run": run,
                "montage": montage,
                "nominal": (
                    good_names
                    if montage == "all_good"
                    else nominal_sets[montage]
                ),
                "actual": actual_channels,
                "replacements": replacements,
                "reference": (
                    "average reference applied before reduced "
                    "montage pick"
                ),
                "template_sha256": template_digest,
            }

            channel_path = (
                channels_dir
                / f"{subject}_{run}_{montage}.json"
            )

            if (
                args.resume
                and channel_path.is_file()
            ):
                existing_channel_payload = json.loads(
                    channel_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    existing_channel_payload
                    != channel_payload
                ):
                    raise ValueError(
                        f"Montage changed on resume: "
                        f"{channel_path}"
                    )

            atomic_json(
                channel_path,
                channel_payload,
            )

            print(
                f"  {montage:8s}: "
                f"{len(actual_channels)} channels; "
                f"{len(replacements)} replacements",
                flush=True,
            )

            try:
                noise_covariance = (
                    pilot.estimate_noise_covariance(
                        montage_epochs,
                        tmin=NOISE[0],
                        tmax=NOISE[1],
                        method="auto",
                        verbose=not args.quiet,
                    )
                )

                evoked = pilot.create_target_evoked(
                    montage_epochs,
                    tmin=TARGET[0],
                    tmax=TARGET[1],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Shared covariance/evoked preparation failed "
                    f"for {subject} {run} {montage}"
                ) from exc

            lcmv_covariance_cache = {}

            for config in configurations:
                key = result_key_from_values(
                    subject,
                    run,
                    montage,
                    config["configuration_id"],
                )

                if key in completed:
                    continue

                started = time.monotonic()

                row = base_row(
                    subject=subject,
                    run=run,
                    method=method,
                    montage=montage,
                    configuration_id=config[
                        "configuration_id"
                    ],
                    channels=actual_channels,
                    replacements=replacements,
                    loaded=loaded,
                    evoked=evoked,
                    noise_covariance=noise_covariance,
                    stimulation=stimulation,
                    ground_truth_mri=(
                        ground_truth_mri
                    ),
                )

                try:
                    if method == "Continuous-ECD":
                        min_dist = float(
                            config["ecd_min_dist_mm"]
                        )

                        diagnostics = (
                            pilot.run_continuous_ecd(
                                evoked,
                                noise_covariance,
                                bem,
                                head_to_mri,
                                ground_truth_mri,
                                min_dist,
                                args.quiet,
                            )
                        )

                        selected = diagnostics["selected"]
                        minimum_error = diagnostics[
                            "minimum_error"
                        ]

                        distance = float(
                            selected[
                                "localization_distance_mm"
                            ]
                        )

                        row.update(
                            ecd_min_dist_mm=min_dist,
                            ecd_selected_gof_percent=float(
                                selected["gof_percent"]
                            ),
                            ecd_minimum_observed_error_mm=float(
                                minimum_error[
                                    "localization_distance_mm"
                                ]
                            ),
                            ecd_minimum_error_time_ms=float(
                                minimum_error["time_ms"]
                            ),
                            ecd_gof_at_minimum_error_percent=float(
                                minimum_error[
                                    "gof_percent"
                                ]
                            ),
                            bem_file=str(bem_file),
                            selection_criterion="maximum_GOF",
                            selected_time_ms=float(
                                selected["time_ms"]
                            ),
                            selected_x_mm=float(
                                selected["x_mm"]
                            ),
                            selected_y_mm=float(
                                selected["y_mm"]
                            ),
                            selected_z_mm=float(
                                selected["z_mm"]
                            ),
                            localization_distance_mm=distance,
                        )

                    elif method == "MxNE":
                        alpha = float(
                            config["mxne_alpha"]
                        )
                        loose = float(
                            config["mxne_loose"]
                        )
                        depth = float(
                            config["mxne_depth"]
                        )

                        (
                            source_estimate,
                            residual,
                        ) = pilot.run_mxne(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            alpha,
                            loose,
                            depth,
                            args.quiet,
                        )

                        active_sources = (
                            count_active_sources(
                                source_estimate
                            )
                        )

                        if active_sources == 0:
                            raise RuntimeError(
                                "MxNE returned no active sources."
                            )

                        metric = (
                            pilot.calculate_localization_metrics(
                                source_estimate,
                                loaded.forward,
                                surface_transform,
                                stimulation,
                            )
                        )

                        fill_grid_metric(
                            row,
                            metric,
                        )

                        distance = float(
                            metric.localization_distance_mm
                        )

                        row.update(
                            mxne_alpha=alpha,
                            mxne_loose=loose,
                            mxne_depth=depth,
                            mxne_iterations=1,
                            mxne_active_sources=(
                                active_sources
                            ),
                            mxne_explained_variance_percent=(
                                explained_variance(
                                    evoked,
                                    residual,
                                )
                            ),
                        )

                    elif method == "RAP-MUSIC":
                        (
                            dipoles,
                            residual,
                        ) = pilot.run_rap_music(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            RAP_MUSIC_N_DIPOLES,
                            args.quiet,
                        )

                        selected = (
                            pilot.select_rap_music_dipole(
                                dipoles,
                                ground_truth_mri,
                                head_to_mri,
                            )
                        )

                        coordinate_mri = np.asarray(
                            selected["coordinate_m"],
                            dtype=float,
                        )

                        distance = float(
                            selected["distance_mm"]
                        )

                        row.update(
                            rap_music_n_dipoles=1,
                            rap_music_gof_percent=float(
                                selected["gof_percent"]
                            ),
                            selection_criterion="maximum_GOF",
                            selected_time_ms=float(
                                selected["time_ms"]
                            ),
                            selected_x_mm=float(
                                coordinate_mri[0] * 1000
                            ),
                            selected_y_mm=float(
                                coordinate_mri[1] * 1000
                            ),
                            selected_z_mm=float(
                                coordinate_mri[2] * 1000
                            ),
                            localization_distance_mm=distance,
                        )

                    elif method == "LCMV":
                        reg = float(
                            config["lcmv_reg"]
                        )
                        window_ms = float(
                            config[
                                "lcmv_window_half_ms"
                            ]
                        )
                        covariance_method = str(
                            config[
                                "lcmv_data_covariance_method"
                            ]
                        )
                        weight_norm = config[
                            "lcmv_weight_norm"
                        ]

                        covariance_key = (
                            window_ms,
                            covariance_method,
                        )

                        if (
                            covariance_key
                            not in lcmv_covariance_cache
                        ):
                            half_window_s = (
                                window_ms / 1000.0
                            )

                            lcmv_covariance_cache[
                                covariance_key
                            ] = (
                                compute_lcmv_data_covariance(
                                    montage_epochs,
                                    covariance_method,
                                    -half_window_s,
                                    half_window_s,
                                    args.quiet,
                                )
                            )

                        data_covariance = (
                            lcmv_covariance_cache[
                                covariance_key
                            ]
                        )

                        source_estimate = run_lcmv(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            data_covariance,
                            reg,
                            weight_norm,
                            args.quiet,
                        )

                        metric = (
                            pilot.calculate_localization_metrics(
                                source_estimate,
                                loaded.forward,
                                surface_transform,
                                stimulation,
                            )
                        )

                        fill_grid_metric(
                            row,
                            metric,
                        )

                        distance = float(
                            metric.localization_distance_mm
                        )

                        row.update(
                            lcmv_reg=reg,
                            lcmv_data_covariance_method=(
                                covariance_method
                            ),
                            lcmv_window_half_ms=window_ms,
                            lcmv_data_tmin_s=(
                                -window_ms / 1000.0
                            ),
                            lcmv_data_tmax_s=(
                                window_ms / 1000.0
                            ),
                            lcmv_data_covariance_samples=(
                                data_covariance.get(
                                    "nfree",
                                    "",
                                )
                            ),
                            lcmv_pick_ori="max-power",
                            lcmv_weight_norm=(
                                "None"
                                if weight_norm is None
                                else weight_norm
                            ),
                            lcmv_reduce_rank=True,
                        )

                    else:
                        raise RuntimeError(
                            f"Unhandled method: {method}"
                        )

                    if is_script12_baseline(
                        method,
                        montage,
                        config,
                    ):
                        baseline = (
                            script12_baselines.get(
                                (subject, run)
                            )
                        )

                        if baseline is not None:
                            change = (
                                distance - baseline
                            )

                            row.update(
                                script12_baseline_mm=(
                                    baseline
                                ),
                                change_from_script12_baseline_mm=(
                                    change
                                ),
                            )

                            if (
                                abs(change)
                                > args.baseline_tolerance_mm
                            ):
                                raise ValueError(
                                    "Script-12 baseline mismatch: "
                                    f"Script15={distance:.4f} mm, "
                                    f"Script12={baseline:.4f} mm, "
                                    f"difference={change:+.4f} mm."
                                )

                    row.update(
                        status="PASS",
                        runtime_seconds=round(
                            time.monotonic() - started,
                            3,
                        ),
                    )

                    append_row(
                        results_path,
                        row,
                    )

                    completed.add(key)
                    new_passes += 1

                    print(
                        f"    {config['configuration_id']}: "
                        f"{distance:.2f} mm",
                        flush=True,
                    )

                except Exception as exc:
                    row.update(
                        status="FAIL",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        runtime_seconds=round(
                            time.monotonic() - started,
                            3,
                        ),
                    )

                    append_row(
                        results_path,
                        row,
                    )

                    completed.add(key)
                    new_failures += 1

                    print(
                        f"    {config['configuration_id']} "
                        f"FAIL: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

    print("\n" + "=" * 78)
    print("SCRIPT 15 COMPLETE")
    print("=" * 78)
    print(f"Method       : {method}")
    print(f"New PASS     : {new_passes}")
    print(f"New FAIL     : {new_failures}")
    print(f"Results      : {results_path}")
    print(f"Manifest     : {manifest_path}")

    if new_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
