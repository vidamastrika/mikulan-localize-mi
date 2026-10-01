#!/usr/bin/env python3
"""
14_additional_method_parameter_pilot.py

One-factor-at-a-time (OFAT) parameter pilot for the four additional
Localize-MI source-localization methods.

PURPOSE
-------
Script 11 established that each additional method can run on the four
representative cases. Script 12 ran one baseline configuration across all
released runs. Script 13 summarized the matched baseline results.

Script 14 now asks:
    Which METHOD-SPECIFIC parameters appear influential enough to include
    in the full Script 15 experiment?

This is an exploratory parameter-sensitivity pilot. It is NOT a validated
optimization. Each tested configuration is evaluated on the same four
representative runs, and any "lower-error configuration" found here is only
an observed pilot result.

STANDARD PILOT RUNS
-------------------
    sub-01 run-01  : easy / relatively accurate case
    sub-07 run-07  : relatively good/generalization case
    sub-05 run-06  : severe standard-method outlier
    sub-07 run-05  : intermediate/challenging case

SHARED DATA SETTINGS
--------------------
    montage             : all good EEG channels
    noise covariance    : -250 to -50 ms, method="auto"
    target window       : -2 to +2 ms

BASELINE + TESTED PARAMETERS
----------------------------

1) Continuous-ECD
   Baseline:
       min_dist = 5 mm

   OFAT sensitivity:
       min_dist = 1, 2.5, 5, 10, 15 mm

   Fixed:
       position/orientation = continuous/free
       selection            = maximum GOF
       BEM                  = participant-specific 3-layer BEM
       conductivity         = 0.3 / 0.006 / 0.3 S/m
       BEM ico              = 4

   Rationale:
       The pilot ECD fits often approached the inner-skull distance
       constraint, so min_dist is the first ECD-specific parameter tested.

2) MxNE
   Baseline:
       alpha = 40
       loose = 1.0
       depth = 0.1
       n_mxne_iter = 1

   OFAT sensitivity:
       alpha = 20, 30, 40, 50, 60
       loose = 0.2, 0.5, 1.0
       depth = 0.0, 0.1, 0.5, 0.8

   Only ONE factor changes from the baseline at a time.

3) RAP-MUSIC
   Baseline:
       n_dipoles = 1

   Sensitivity:
       n_dipoles = 1

   IMPORTANT:
       Localize-MI has one stimulation midpoint as ground truth, therefore
       n_dipoles=1 remains the scientifically closest baseline model.
       n_dipoles > 1 is included only as a model-order sensitivity check.

       For n_dipoles > 1, localization error is calculated for the FIRST
       recursively selected RAP-MUSIC source. We do NOT select whichever
       returned source happens to be closest to ground truth.

4) LCMV
   Baseline:
       reg                    = 0.05
       data covariance        = shrunk
       data covariance window = -5 to +5 ms
       pick_ori               = max-power
       weight_norm            = unit-noise-gain-invariant

   OFAT sensitivity:
       reg = 0.00, 0.01, 0.05, 0.10, 0.20
       data covariance window:
           -2 to +2 ms
           -5 to +5 ms
           -10 to +10 ms
       covariance estimator:
           shrunk
           empirical
           diagonal_fixed
       weight_norm:
           unit-noise-gain-invariant
           unit-noise-gain
           nai
           None

   pick_ori="max-power" and reduce_rank=True remain fixed.

COORDINATE FRAME
----------------
MNE Dipole positions from Continuous-ECD and RAP-MUSIC are in HEAD
coordinates. Before localization error is calculated they are transformed
to the released participant surface/MRI coordinate frame, matching the
Localize-MI stimulation ground truth.

TERMINAL EXAMPLES
-----------------

Inspect planned configurations without running:

    python scripts/14_additional_method_parameter_pilot.py \
        --method ecd --dry-run

Run Continuous-ECD parameter pilot:

    python scripts/14_additional_method_parameter_pilot.py \
        --method ecd --overwrite

Run MxNE parameter pilot:

    python scripts/14_additional_method_parameter_pilot.py \
        --method mxne --overwrite

Run RAP-MUSIC parameter pilot:

    python scripts/14_additional_method_parameter_pilot.py \
        --method rap_music --overwrite

Run LCMV parameter pilot:

    python scripts/14_additional_method_parameter_pilot.py \
        --method lcmv --overwrite

Run all four methods:

    python scripts/14_additional_method_parameter_pilot.py \
        --method all --overwrite

Resume an interrupted MxNE pilot:

    python scripts/14_additional_method_parameter_pilot.py \
        --method mxne --resume

Run only one case while debugging:

    python scripts/14_additional_method_parameter_pilot.py \
        --method lcmv \
        --case sub-05 run-06 \
        --overwrite

Suppress most MNE progress output:

    python scripts/14_additional_method_parameter_pilot.py \
        --method lcmv --overwrite --quiet

DEFAULT OUTPUT ROOT
-------------------
    outputs/14_additional_method_parameter_pilot/

Each method has an independent result file, so methods can be run one by one:

    14_continuous_ecd_parameter_pilot.csv
    14_mxne_parameter_pilot.csv
    14_rap_music_parameter_pilot.csv
    14_lcmv_parameter_pilot.csv

Per-method summaries:

    14_continuous_ecd_parameter_summary.csv
    14_mxne_parameter_summary.csv
    14_rap_music_parameter_summary.csv
    14_lcmv_parameter_summary.csv

Figures:

    figures/14_<method>_localization_error.png
    figures/14_<method>_change_from_baseline.png

The result CSV is checkpointed after every configuration.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import mne
from mne.transforms import apply_trans
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT11 = PROJECT / "scripts" / "11_additional_method_pilot.py"

DEFAULT_OUTPUT_ROOT = (
    PROJECT / "outputs" / "14_additional_method_parameter_pilot"
)

DEFAULT_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)

METHODS = (
    "Continuous-ECD",
    "MxNE",
    "RAP-MUSIC",
    "LCMV",
)

FIELDNAMES = (
    "subject", "run", "method",
    "configuration_id", "configuration_label",
    "varied_parameter", "is_baseline",
    "montage",
    "epochs", "all_channels", "bad_channels", "good_channels",
    "sampling_frequency_hz",
    "covariance_method",
    "covariance_tmin_s", "covariance_tmax_s",
    "target_tmin_s", "target_tmax_s", "target_samples",
    "stimulation_pair", "hemisphere",
    "ground_truth_x_mm", "ground_truth_y_mm", "ground_truth_z_mm",

    # ECD
    "ecd_min_dist_mm",

    # MxNE
    "mxne_alpha", "mxne_loose", "mxne_depth", "mxne_iterations",
    "mxne_active_sources", "mxne_explained_variance_percent",

    # RAP-MUSIC
    "rap_music_n_dipoles",
    "rap_music_returned_sources",
    "rap_music_selected_source_number",
    "rap_music_gof_percent",
    "rap_music_explained_variance_percent",

    # LCMV
    "lcmv_reg",
    "lcmv_data_covariance_method",
    "lcmv_data_tmin_s", "lcmv_data_tmax_s",
    "lcmv_data_covariance_samples",
    "lcmv_pick_ori", "lcmv_weight_norm",

    # Common localization result
    "selection_criterion",
    "selected_time_ms", "selected_gof_percent",
    "selected_x_mm", "selected_y_mm", "selected_z_mm",
    "selected_localization_error_mm",
    "peak_vertex",
    "nearest_source_distance_mm", "geometric_excess_mm",

    # Paired comparison to the method's own pilot baseline
    "baseline_error_mm", "change_from_baseline_mm",

    "status", "error_type", "error_message", "runtime_seconds",
)


def load_script11():
    if not SCRIPT11.is_file():
        raise FileNotFoundError(
            f"Final Script 11 not found: {SCRIPT11}"
        )

    specification = importlib.util.spec_from_file_location(
        "localize_mi_script11",
        SCRIPT11,
    )

    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {SCRIPT11}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


pilot = load_script11()


def normalize_method(value):
    key = str(value).strip().replace("-", "").replace("_", "").lower()

    mapping = {
        "ecd": "Continuous-ECD",
        "continuousecd": "Continuous-ECD",
        "dipole": "Continuous-ECD",
        "mxne": "MxNE",
        "rapmusic": "RAP-MUSIC",
        "music": "RAP-MUSIC",
        "lcmv": "LCMV",
    }

    if key == "all":
        return "ALL"

    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unknown method {value!r}; choose ecd, mxne, rap_music, "
            "lcmv, or all."
        )

    return mapping[key]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

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
        "--case",
        nargs=2,
        metavar=("SUBJECT", "RUN"),
        action="append",
        help="Repeat to override the four standard pilot runs.",
    )

    parser.add_argument(
        "--method",
        type=normalize_method,
        action="append",
        help="Repeat for multiple methods; default: all four.",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip per-method summary figures.",
    )
    parser.add_argument("--dpi", type=int, default=180)

    return parser.parse_args()


def selected_methods(values):
    if not values or "ALL" in values:
        return list(METHODS)
    return list(dict.fromkeys(values))


def config(
    configuration_id,
    label,
    varied_parameter,
    *,
    ecd_min_dist_mm="",
    mxne_alpha="",
    mxne_loose="",
    mxne_depth="",
    rap_music_n_dipoles="",
    lcmv_reg="",
    lcmv_data_tmin_s="",
    lcmv_data_tmax_s="",
    lcmv_data_covariance_method="",
    lcmv_weight_norm="",
):
    return {
        "configuration_id": configuration_id,
        "configuration_label": label,
        "varied_parameter": varied_parameter,
        "is_baseline": configuration_id == "baseline",
        "ecd_min_dist_mm": ecd_min_dist_mm,
        "mxne_alpha": mxne_alpha,
        "mxne_loose": mxne_loose,
        "mxne_depth": mxne_depth,
        "rap_music_n_dipoles": rap_music_n_dipoles,
        "lcmv_reg": lcmv_reg,
        "lcmv_data_tmin_s": lcmv_data_tmin_s,
        "lcmv_data_tmax_s": lcmv_data_tmax_s,
        "lcmv_data_covariance_method": lcmv_data_covariance_method,
        "lcmv_weight_norm": lcmv_weight_norm,
    }


def ecd_settings():
    return [
        config(
            "baseline", "baseline: min_dist=5 mm", "baseline",
            ecd_min_dist_mm=5.0,
        ),
        config(
            "min_dist_1", "min_dist=1 mm", "min_dist",
            ecd_min_dist_mm=1.0,
        ),
        config(
            "min_dist_2p5", "min_dist=2.5 mm", "min_dist",
            ecd_min_dist_mm=2.5,
        ),
        config(
            "min_dist_10", "min_dist=10 mm", "min_dist",
            ecd_min_dist_mm=10.0,
        ),
        config(
            "min_dist_15", "min_dist=15 mm", "min_dist",
            ecd_min_dist_mm=15.0,
        ),
    ]


def mxne_settings():
    baseline = dict(
        mxne_alpha=40.0,
        mxne_loose=1.0,
        mxne_depth=0.1,
    )

    settings = [
        config(
            "baseline",
            "baseline: alpha=40, loose=1.0, depth=0.1",
            "baseline",
            **baseline,
        )
    ]

    for alpha in (20.0, 30.0, 50.0, 60.0):
        settings.append(
            config(
                f"alpha_{int(alpha)}",
                f"alpha={alpha:g}",
                "alpha",
                mxne_alpha=alpha,
                mxne_loose=baseline["mxne_loose"],
                mxne_depth=baseline["mxne_depth"],
            )
        )

    for loose in (0.2, 0.5):
        settings.append(
            config(
                f"loose_{str(loose).replace('.', 'p')}",
                f"loose={loose:g}",
                "loose",
                mxne_alpha=baseline["mxne_alpha"],
                mxne_loose=loose,
                mxne_depth=baseline["mxne_depth"],
            )
        )

    for depth in (0.0, 0.5, 0.8):
        settings.append(
            config(
                f"depth_{str(depth).replace('.', 'p')}",
                f"depth={depth:g}",
                "depth",
                mxne_alpha=baseline["mxne_alpha"],
                mxne_loose=baseline["mxne_loose"],
                mxne_depth=depth,
            )
        )

    return settings


def rap_music_settings():
    return [
        config(
            "baseline", "baseline: n_dipoles=1", "baseline",
            rap_music_n_dipoles=1,
        ),
        config(
            "n_dipoles_2", "n_dipoles=2", "n_dipoles",
            rap_music_n_dipoles=2,
        ),
        config(
            "n_dipoles_3", "n_dipoles=3", "n_dipoles",
            rap_music_n_dipoles=3,
        ),
    ]


def lcmv_settings():
    baseline = dict(
        lcmv_reg=0.05,
        lcmv_data_tmin_s=-0.005,
        lcmv_data_tmax_s=0.005,
        lcmv_data_covariance_method="shrunk",
        lcmv_weight_norm="unit-noise-gain-invariant",
    )

    settings = [
        config(
            "baseline",
            (
                "baseline: reg=.05, shrunk, -5..+5 ms, "
                "unit-noise-gain-invariant"
            ),
            "baseline",
            **baseline,
        )
    ]

    for reg in (0.0, 0.01, 0.10, 0.20):
        settings.append(
            config(
                f"reg_{str(reg).replace('.', 'p')}",
                f"reg={reg:g}",
                "reg",
                lcmv_reg=reg,
                lcmv_data_tmin_s=baseline["lcmv_data_tmin_s"],
                lcmv_data_tmax_s=baseline["lcmv_data_tmax_s"],
                lcmv_data_covariance_method=baseline[
                    "lcmv_data_covariance_method"
                ],
                lcmv_weight_norm=baseline["lcmv_weight_norm"],
            )
        )

    for tmin, tmax, cid, label in (
        (-0.002, 0.002, "window_2ms", "data covariance=-2..+2 ms"),
        (-0.010, 0.010, "window_10ms", "data covariance=-10..+10 ms"),
    ):
        settings.append(
            config(
                cid,
                label,
                "data_covariance_window",
                lcmv_reg=baseline["lcmv_reg"],
                lcmv_data_tmin_s=tmin,
                lcmv_data_tmax_s=tmax,
                lcmv_data_covariance_method=baseline[
                    "lcmv_data_covariance_method"
                ],
                lcmv_weight_norm=baseline["lcmv_weight_norm"],
            )
        )

    for method in ("empirical", "diagonal_fixed"):
        settings.append(
            config(
                f"cov_{method}",
                f"data covariance method={method}",
                "data_covariance_method",
                lcmv_reg=baseline["lcmv_reg"],
                lcmv_data_tmin_s=baseline["lcmv_data_tmin_s"],
                lcmv_data_tmax_s=baseline["lcmv_data_tmax_s"],
                lcmv_data_covariance_method=method,
                lcmv_weight_norm=baseline["lcmv_weight_norm"],
            )
        )

    for weight_norm, cid, label in (
        (
            "unit-noise-gain",
            "norm_unit_noise_gain",
            "weight_norm=unit-noise-gain",
        ),
        ("nai", "norm_nai", "weight_norm=nai"),
        (None, "norm_none", "weight_norm=None"),
    ):
        settings.append(
            config(
                cid,
                label,
                "weight_norm",
                lcmv_reg=baseline["lcmv_reg"],
                lcmv_data_tmin_s=baseline["lcmv_data_tmin_s"],
                lcmv_data_tmax_s=baseline["lcmv_data_tmax_s"],
                lcmv_data_covariance_method=baseline[
                    "lcmv_data_covariance_method"
                ],
                lcmv_weight_norm=weight_norm,
            )
        )

    return settings


def settings_for(method):
    if method == "Continuous-ECD":
        return ecd_settings()
    if method == "MxNE":
        return mxne_settings()
    if method == "RAP-MUSIC":
        return rap_music_settings()
    if method == "LCMV":
        return lcmv_settings()
    raise ValueError(method)


def method_slug(method):
    return {
        "Continuous-ECD": "continuous_ecd",
        "MxNE": "mxne",
        "RAP-MUSIC": "rap_music",
        "LCMV": "lcmv",
    }[method]


def result_path(output_root, method):
    return (
        output_root
        / f"14_{method_slug(method)}_parameter_pilot.csv"
    )


def summary_path(output_root, method):
    return (
        output_root
        / f"14_{method_slug(method)}_parameter_summary.csv"
    )


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)


def key(row):
    return (
        str(row["subject"]),
        str(row["run"]),
        str(row["method"]),
        str(row["configuration_id"]),
    )


def load_rows(path, resume, overwrite):
    if not path.exists() or overwrite:
        return []

    if not resume:
        raise FileExistsError(
            f"Output exists: {path}. Use --resume or --overwrite."
        )

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDNAMES):
            raise ValueError(
                f"Existing CSV columns do not match Script 14: {path}"
            )
        rows = list(reader)

    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate configurations in {path}")

    # Retry failures on resume.
    return [
        row for row in rows
        if str(row.get("status")) == "PASS"
    ]


def run_continuous_ecd(
    evoked,
    noise_covariance,
    bem,
    head_to_mri,
    ground_truth_mri,
    min_dist_mm,
    quiet,
):
    dipole, residual = mne.fit_dipole(
        evoked,
        noise_covariance,
        bem,
        head_to_mri,
        min_dist=float(min_dist_mm),
        n_jobs=None,
        verbose=not quiet,
    )

    records = []

    for index in range(len(dipole.times)):
        coordinate_head = np.asarray(
            dipole.pos[index],
            dtype=float,
        )
        coordinate_mri = np.asarray(
            apply_trans(head_to_mri, coordinate_head),
            dtype=float,
        )

        records.append(
            {
                "time_ms": float(dipole.times[index] * 1000),
                "gof_percent": float(dipole.gof[index]),
                "coordinate_mri": coordinate_mri,
                "distance_mm": float(
                    np.linalg.norm(
                        coordinate_mri - ground_truth_mri
                    )
                    * 1000
                ),
            }
        )

    if not records:
        raise RuntimeError("Continuous-ECD returned no fitted samples.")

    selected = max(
        records,
        key=lambda item: item["gof_percent"],
    )

    return selected, residual


def run_mxne(
    evoked,
    forward,
    noise_covariance,
    alpha,
    loose,
    depth,
    quiet,
):
    return mne.inverse_sparse.mixed_norm(
        evoked,
        forward,
        noise_covariance,
        alpha=float(alpha),
        loose=float(loose),
        depth=float(depth),
        n_mxne_iter=1,
        return_residual=True,
        return_as_dipoles=False,
        rank=None,
        random_state=0,
        verbose=not quiet,
    )


def run_rap_music(
    evoked,
    forward,
    noise_covariance,
    n_dipoles,
    quiet,
):
    return mne.beamformer.rap_music(
        evoked,
        forward,
        noise_covariance,
        n_dipoles=int(n_dipoles),
        return_residual=True,
        verbose=not quiet,
    )


def select_first_rap_music_source(
    dipoles,
    ground_truth_mri,
    head_to_mri,
):
    if not dipoles:
        raise RuntimeError("RAP-MUSIC returned no sources.")

    # Deliberately use source #1, not the source closest to ground truth.
    dipole = dipoles[0]

    if len(dipole.times) == 0:
        raise RuntimeError("First RAP-MUSIC source has no time samples.")

    index = int(np.argmax(dipole.gof))

    coordinate_head = np.asarray(
        dipole.pos[index],
        dtype=float,
    )
    coordinate_mri = np.asarray(
        apply_trans(head_to_mri, coordinate_head),
        dtype=float,
    )

    return {
        "time_ms": float(dipole.times[index] * 1000),
        "gof_percent": float(dipole.gof[index]),
        "coordinate_mri": coordinate_mri,
        "distance_mm": float(
            np.linalg.norm(
                coordinate_mri - ground_truth_mri
            )
            * 1000
        ),
    }


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
            tmin=float(tmin),
            tmax=float(tmax),
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
        imaginary_scale = float(np.max(np.abs(data.imag)))
        real_scale = float(np.max(np.abs(data.real)))
        tolerance = max(1e-12, real_scale * 1e-8)

        if imaginary_scale > tolerance:
            raise RuntimeError(
                "LCMV returned materially complex data: "
                f"imag={imaginary_scale:.3e}, "
                f"real={real_scale:.3e}"
            )

        source_estimate = source_estimate.copy()
        source_estimate._data = data.real

    return source_estimate


def fill_metric(row, metric):
    distance = float(metric.localization_distance_mm)

    row.update(
        selection_criterion="maximum_source_amplitude",
        selected_time_ms=float(metric.peak_time * 1000),
        selected_x_mm=float(metric.peak_coordinate[0] * 1000),
        selected_y_mm=float(metric.peak_coordinate[1] * 1000),
        selected_z_mm=float(metric.peak_coordinate[2] * 1000),
        selected_localization_error_mm=distance,
        peak_vertex=int(metric.peak_vertex),
        nearest_source_distance_mm=float(
            metric.nearest_source_distance_mm
        ),
        geometric_excess_mm=(
            distance
            - float(metric.nearest_source_distance_mm)
        ),
    )


def add_baseline_differences(rows):
    baseline = {}

    for row in rows:
        if (
            str(row.get("status")) == "PASS"
            and str(row.get("configuration_id")) == "baseline"
        ):
            baseline[
                (
                    str(row["subject"]),
                    str(row["run"]),
                    str(row["method"]),
                )
            ] = float(row["selected_localization_error_mm"])

    for row in rows:
        if str(row.get("status")) != "PASS":
            row["baseline_error_mm"] = ""
            row["change_from_baseline_mm"] = ""
            continue

        lookup = (
            str(row["subject"]),
            str(row["run"]),
            str(row["method"]),
        )

        if lookup not in baseline:
            row["baseline_error_mm"] = ""
            row["change_from_baseline_mm"] = ""
            continue

        value = float(row["selected_localization_error_mm"])
        base = baseline[lookup]

        row["baseline_error_mm"] = base
        row["change_from_baseline_mm"] = value - base

    return rows


def build_summary(rows, settings):
    table = pd.DataFrame(rows)

    config_order = [
        item["configuration_id"]
        for item in settings
    ]

    output = []

    for item in settings:
        cid = item["configuration_id"]

        subset = table.loc[
            (table["configuration_id"] == cid)
            & (table["status"] == "PASS")
        ].copy()

        failures = int(
            (
                (table["configuration_id"] == cid)
                & (table["status"] != "PASS")
            ).sum()
        )

        if subset.empty:
            output.append(
                {
                    "configuration_id": cid,
                    "configuration_label": item["configuration_label"],
                    "varied_parameter": item["varied_parameter"],
                    "is_baseline": item["is_baseline"],
                    "successful_runs": 0,
                    "failed_runs": failures,
                }
            )
            continue

        values = subset[
            "selected_localization_error_mm"
        ].astype(float)

        changes = pd.to_numeric(
            subset["change_from_baseline_mm"],
            errors="coerce",
        ).dropna()

        q25, median, q75 = np.quantile(
            values,
            [0.25, 0.5, 0.75],
        )

        output.append(
            {
                "configuration_id": cid,
                "configuration_label": item["configuration_label"],
                "varied_parameter": item["varied_parameter"],
                "is_baseline": item["is_baseline"],
                "successful_runs": len(values),
                "failed_runs": failures,
                "mean_error_mm": float(values.mean()),
                "standard_deviation_mm": (
                    float(values.std(ddof=1))
                    if len(values) > 1
                    else np.nan
                ),
                "median_error_mm": float(median),
                "q25_mm": float(q25),
                "q75_mm": float(q75),
                "minimum_mm": float(values.min()),
                "maximum_mm": float(values.max()),
                "mean_change_from_baseline_mm": (
                    float(changes.mean())
                    if len(changes)
                    else np.nan
                ),
                "median_change_from_baseline_mm": (
                    float(changes.median())
                    if len(changes)
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(output)

    if not summary.empty:
        summary["configuration_id"] = pd.Categorical(
            summary["configuration_id"],
            categories=config_order,
            ordered=True,
        )
        summary = summary.sort_values(
            "configuration_id"
        ).reset_index(drop=True)
        summary["configuration_id"] = (
            summary["configuration_id"].astype(str)
        )

    return summary


def save_figures(rows, settings, method, output_root, dpi):
    table = pd.DataFrame(rows)
    table = table.loc[table["status"] == "PASS"].copy()

    if table.empty:
        return

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    ids = [
        item["configuration_id"]
        for item in settings
    ]
    labels = [
        item["configuration_label"]
        for item in settings
    ]

    data = []
    for cid in ids:
        values = pd.to_numeric(
            table.loc[
                table["configuration_id"] == cid,
                "selected_localization_error_mm",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        data.append(values)

    valid_positions = [
        index + 1
        for index, values in enumerate(data)
        if len(values)
    ]
    valid_data = [
        values for values in data if len(values)
    ]

    if valid_data:
        figure, axis = plt.subplots(
            figsize=(max(10, 0.95 * len(ids)), 6)
        )

        axis.boxplot(
            valid_data,
            positions=valid_positions,
            widths=0.45,
            showfliers=True,
        )

        for position, values in zip(
            valid_positions,
            valid_data,
        ):
            axis.scatter(
                np.full(len(values), position, dtype=float),
                values,
                s=24,
                zorder=3,
            )

        axis.set_xticks(
            np.arange(1, len(ids) + 1),
            labels,
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("Localization error (mm)")
        axis.set_title(
            f"{method}: parameter-pilot localization error"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(
            figure_dir
            / f"14_{method_slug(method)}_localization_error.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)

    change_data = []
    change_positions = []
    change_labels = []

    for index, (cid, label) in enumerate(
        zip(ids, labels),
        start=1,
    ):
        if cid == "baseline":
            continue

        values = pd.to_numeric(
            table.loc[
                table["configuration_id"] == cid,
                "change_from_baseline_mm",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        if not len(values):
            continue

        change_data.append(values)
        change_positions.append(len(change_positions) + 1)
        change_labels.append(label)

    if change_data:
        figure, axis = plt.subplots(
            figsize=(max(10, 0.95 * len(change_data)), 6)
        )

        axis.boxplot(
            change_data,
            positions=change_positions,
            widths=0.45,
            showfliers=True,
        )

        for position, values in zip(
            change_positions,
            change_data,
        ):
            axis.scatter(
                np.full(len(values), position, dtype=float),
                values,
                s=24,
                zorder=3,
            )

        axis.axhline(0, linewidth=1)
        axis.set_xticks(
            change_positions,
            change_labels,
            rotation=35,
            ha="right",
        )
        axis.set_ylabel(
            "Change from baseline localization error (mm)"
        )
        axis.set_title(
            f"{method}: paired change from baseline"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(
            figure_dir
            / f"14_{method_slug(method)}_change_from_baseline.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def print_dry_run(methods, cases):
    print("=" * 78)
    print("SCRIPT 14 DRY RUN")
    print("=" * 78)
    print("Cases:")
    for subject, run in cases:
        print(f"  {subject} {run}")

    for method in methods:
        settings = settings_for(method)
        print(
            f"\n{method}: {len(settings)} configurations "
            f"x {len(cases)} runs = "
            f"{len(settings) * len(cases)} solutions"
        )
        for item in settings:
            print(
                f"  {item['configuration_id']:22s} "
                f"{item['configuration_label']}"
            )


def main():
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError(
            "Choose --resume or --overwrite, not both."
        )

    if args.dpi < 50:
        raise ValueError("--dpi must be at least 50.")

    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset.is_dir():
        raise NotADirectoryError(
            f"Dataset not found: {dataset}"
        )

    cases = list(
        dict.fromkeys(
            tuple(case)
            for case in (args.case or DEFAULT_CASES)
        )
    )
    methods = selected_methods(args.method)

    if args.dry_run:
        print_dry_run(methods, cases)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    bem_dir = output_root / "bem"

    for method in methods:
        settings = settings_for(method)
        output_file = result_path(output_root, method)
        method_summary_file = summary_path(
            output_root,
            method,
        )

        if args.overwrite:
            if output_file.exists():
                output_file.unlink()
            if method_summary_file.exists():
                method_summary_file.unlink()

        rows = load_rows(
            output_file,
            args.resume,
            args.overwrite,
        )

        expected = {
            (
                subject,
                run,
                method,
                item["configuration_id"],
            )
            for subject, run in cases
            for item in settings
        }

        outside = {
            key(row) for row in rows
            if key(row) not in expected
        }

        if outside:
            raise ValueError(
                f"{output_file} contains rows outside the "
                "current case/configuration selection. "
                "Use another output root or --overwrite."
            )

        completed = {
            key(row)
            for row in rows
            if str(row.get("status")) == "PASS"
        }

        total = len(cases) * len(settings)

        print("\n" + "=" * 78)
        print(f"SCRIPT 14 — {method} PARAMETER PILOT")
        print("=" * 78)
        print(
            f"Cases          : {len(cases)}\n"
            f"Configurations : {len(settings)}\n"
            f"Solutions      : {total}\n"
            f"Output         : {output_file}"
        )

        for run_index, (subject, run) in enumerate(
            cases,
            start=1,
        ):
            pending = [
                item for item in settings
                if (
                    subject,
                    run,
                    method,
                    item["configuration_id"],
                ) not in completed
            ]

            if not pending:
                print(
                    f"[{run_index}/{len(cases)}] "
                    f"{subject} {run}: already complete",
                    flush=True,
                )
                continue

            print(
                f"\n[{run_index}/{len(cases)}] "
                f"{subject} {run}",
                flush=True,
            )

            loaded = pilot.load_run(
                dataset=dataset,
                subject=subject,
                run=run,
                task="seegstim",
            )

            epochs = loaded.epochs.copy()
            n_epochs = len(epochs)
            n_all_channels = len(epochs.ch_names)
            n_bad_channels = len(epochs.info["bads"])

            epochs.pick("eeg", exclude="bads")
            n_good_channels = len(epochs.ch_names)

            stimulation, ground_truth_mri = (
                pilot.get_stimulation_and_ground_truth(
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

            surface_transform = (
                pilot.load_surface_transform(
                    transform_file
                )
            )

            referenced, noise_covariance, evoked = (
                pilot.common_preprocessing(
                    epochs,
                    args.quiet,
                )
            )

            head_to_mri = None
            bem = None

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
                bem, _ = pilot.get_or_build_bem(
                    dataset,
                    subject,
                    bem_dir,
                )

            covariance_cache = {}

            for setting_index, item in enumerate(
                pending,
                start=1,
            ):
                started = time.monotonic()

                row = dict.fromkeys(
                    FIELDNAMES,
                    "",
                )

                row.update(
                    subject=subject,
                    run=run,
                    method=method,
                    configuration_id=item[
                        "configuration_id"
                    ],
                    configuration_label=item[
                        "configuration_label"
                    ],
                    varied_parameter=item[
                        "varied_parameter"
                    ],
                    is_baseline=item["is_baseline"],
                    montage="all_good",
                    epochs=n_epochs,
                    all_channels=n_all_channels,
                    bad_channels=n_bad_channels,
                    good_channels=n_good_channels,
                    sampling_frequency_hz=float(
                        epochs.info["sfreq"]
                    ),
                    covariance_method=str(
                        noise_covariance.get(
                            "method",
                            "auto",
                        )
                    ),
                    covariance_tmin_s=(
                        pilot.COVARIANCE_TMIN
                    ),
                    covariance_tmax_s=(
                        pilot.COVARIANCE_TMAX
                    ),
                    target_tmin_s=pilot.TARGET_TMIN,
                    target_tmax_s=pilot.TARGET_TMAX,
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

                try:
                    if method == "Continuous-ECD":
                        min_dist = float(
                            item["ecd_min_dist_mm"]
                        )

                        selected, residual = (
                            run_continuous_ecd(
                                evoked,
                                noise_covariance,
                                bem,
                                head_to_mri,
                                ground_truth_mri,
                                min_dist,
                                args.quiet,
                            )
                        )

                        coordinate = selected[
                            "coordinate_mri"
                        ]

                        row.update(
                            ecd_min_dist_mm=min_dist,
                            selection_criterion=(
                                "maximum_GOF"
                            ),
                            selected_time_ms=selected[
                                "time_ms"
                            ],
                            selected_gof_percent=selected[
                                "gof_percent"
                            ],
                            selected_x_mm=float(
                                coordinate[0] * 1000
                            ),
                            selected_y_mm=float(
                                coordinate[1] * 1000
                            ),
                            selected_z_mm=float(
                                coordinate[2] * 1000
                            ),
                            selected_localization_error_mm=(
                                selected["distance_mm"]
                            ),
                        )

                    elif method == "MxNE":
                        alpha = float(item["mxne_alpha"])
                        loose = float(item["mxne_loose"])
                        depth = float(item["mxne_depth"])

                        source_estimate, residual = run_mxne(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            alpha,
                            loose,
                            depth,
                            args.quiet,
                        )

                        active_sources = (
                            pilot.count_active_sources(
                                source_estimate
                            )
                        )

                        if active_sources == 0:
                            raise RuntimeError(
                                "MxNE returned no active "
                                "cortical sources."
                            )

                        metric = (
                            pilot.calculate_localization_metrics(
                                source_estimate,
                                loaded.forward,
                                surface_transform,
                                stimulation,
                            )
                        )

                        fill_metric(row, metric)

                        row.update(
                            mxne_alpha=alpha,
                            mxne_loose=loose,
                            mxne_depth=depth,
                            mxne_iterations=1,
                            mxne_active_sources=(
                                active_sources
                            ),
                            mxne_explained_variance_percent=(
                                pilot.explained_variance(
                                    evoked,
                                    residual,
                                )
                            ),
                        )

                    elif method == "RAP-MUSIC":
                        n_dipoles = int(
                            item["rap_music_n_dipoles"]
                        )

                        dipoles, residual = run_rap_music(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            n_dipoles,
                            args.quiet,
                        )

                        selected = (
                            select_first_rap_music_source(
                                dipoles,
                                ground_truth_mri,
                                head_to_mri,
                            )
                        )

                        coordinate = selected[
                            "coordinate_mri"
                        ]

                        row.update(
                            rap_music_n_dipoles=n_dipoles,
                            rap_music_returned_sources=(
                                len(dipoles)
                            ),
                            rap_music_selected_source_number=1,
                            rap_music_gof_percent=selected[
                                "gof_percent"
                            ],
                            rap_music_explained_variance_percent=(
                                pilot.explained_variance(
                                    evoked,
                                    residual,
                                )
                            ),
                            selection_criterion=(
                                "first_recursive_source"
                            ),
                            selected_time_ms=selected[
                                "time_ms"
                            ],
                            selected_gof_percent=selected[
                                "gof_percent"
                            ],
                            selected_x_mm=float(
                                coordinate[0] * 1000
                            ),
                            selected_y_mm=float(
                                coordinate[1] * 1000
                            ),
                            selected_z_mm=float(
                                coordinate[2] * 1000
                            ),
                            selected_localization_error_mm=(
                                selected["distance_mm"]
                            ),
                        )

                    elif method == "LCMV":
                        reg = float(
                            item["lcmv_reg"]
                        )
                        tmin = float(
                            item["lcmv_data_tmin_s"]
                        )
                        tmax = float(
                            item["lcmv_data_tmax_s"]
                        )
                        covariance_method = str(
                            item[
                                "lcmv_data_covariance_method"
                            ]
                        )
                        weight_norm = item[
                            "lcmv_weight_norm"
                        ]

                        cache_key = (
                            covariance_method,
                            tmin,
                            tmax,
                        )

                        if cache_key not in covariance_cache:
                            covariance_cache[
                                cache_key
                            ] = compute_lcmv_data_covariance(
                                referenced,
                                covariance_method,
                                tmin,
                                tmax,
                                args.quiet,
                            )

                        data_covariance = (
                            covariance_cache[cache_key]
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

                        fill_metric(row, metric)

                        row.update(
                            lcmv_reg=reg,
                            lcmv_data_covariance_method=(
                                covariance_method
                            ),
                            lcmv_data_tmin_s=tmin,
                            lcmv_data_tmax_s=tmax,
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
                        )

                    else:
                        raise RuntimeError(
                            f"Unhandled method: {method}"
                        )

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

                # Replace a prior row with the same key if needed.
                rows = [
                    existing
                    for existing in rows
                    if key(existing) != key(row)
                ]
                rows.append(row)

                rows = add_baseline_differences(
                    rows
                )
                save_rows(output_file, rows)

                if row["status"] == "PASS":
                    print(
                        f"  "
                        f"{item['configuration_id']:22s} "
                        f"error="
                        f"{float(row['selected_localization_error_mm']):7.2f} mm",
                        flush=True,
                    )
                else:
                    print(
                        f"  "
                        f"{item['configuration_id']:22s} "
                        f"FAIL: {row['error_type']}: "
                        f"{row['error_message']}",
                        flush=True,
                    )

        rows = add_baseline_differences(rows)
        save_rows(output_file, rows)

        summary = build_summary(
            rows,
            settings,
        )
        summary.to_csv(
            method_summary_file,
            index=False,
        )

        if not args.no_figures:
            save_figures(
                rows,
                settings,
                method,
                output_root,
                args.dpi,
            )

        failures = sum(
            str(row.get("status")) != "PASS"
            for row in rows
        )

        print("\nSUMMARY")
        if not summary.empty:
            columns = [
                "configuration_id",
                "successful_runs",
                "failed_runs",
                "mean_error_mm",
                "median_error_mm",
                "median_change_from_baseline_mm",
            ]
            existing = [
                column
                for column in columns
                if column in summary.columns
            ]
            print(
                summary[existing].to_string(
                    index=False,
                    float_format=lambda value: (
                        f"{value:.2f}"
                    ),
                )
            )

        print(
            f"\nRows     : {len(rows)}\n"
            f"Failures : {failures}\n"
            f"Results  : {output_file}\n"
            f"Summary  : {method_summary_file}"
        )


if __name__ == "__main__":
    main()
