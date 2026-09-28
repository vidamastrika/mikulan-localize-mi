#!/usr/bin/env python3
"""
11_additional_method_pilot_REVISED.py

Pilot additional EEG source-localization methods on the standard four
representative Localize-MI runs.

Methods
-------
Continuous-ECD
    Continuous equivalent-current-dipole fitting with mne.fit_dipole().
    The dipole position and orientation are unconstrained. One dipole is
    fitted at every sample in the -2 to +2 ms target window. The PRIMARY
    solution is selected using maximum goodness-of-fit (GOF), without using
    the known stimulation location. The minimum observed localization error
    is saved separately as a diagnostic only.

MxNE
    Mixed-norm sparse source estimate using mne.inverse_sparse.mixed_norm().

RAP-MUSIC
    Recursively applied and projected MUSIC using mne.beamformer.rap_music().
    The baseline pilot uses one dipole because the stimulation ground truth
    corresponds to one stimulation midpoint.

LCMV
    Linearly constrained minimum-variance beamformer using a max-power
    orientation and unit-noise-gain-invariant normalization.

Standard pilot runs
-------------------
    sub-01 run-01  : good/easy case
    sub-07 run-07  : another relatively good case
    sub-05 run-06  : severe standard-method outlier
    sub-07 run-05  : intermediate/challenging case

IMPORTANT
---------
This is a BASELINE PILOT, not the parameter-grid experiment. Method-specific
parameter exploration belongs in Script 14.

Run from the repository root.

TERMINAL EXAMPLES
-----------------

Run all four methods on all four standard pilot runs:

    python scripts/11_additional_method_pilot_REVISED.py --method all --overwrite

Run Continuous ECD only:

    python scripts/11_additional_method_pilot_REVISED.py --method ecd --overwrite

Run MxNE only:

    python scripts/11_additional_method_pilot_REVISED.py --method mxne --overwrite

Run RAP-MUSIC only:

    python scripts/11_additional_method_pilot_REVISED.py --method rap_music --overwrite

Run LCMV only:

    python scripts/11_additional_method_pilot_REVISED.py --method lcmv --overwrite

Run two selected methods:

    python scripts/11_additional_method_pilot_REVISED.py \
        --method mxne --method lcmv --overwrite

Run one method on one case:

    python scripts/11_additional_method_pilot_REVISED.py \
        --method ecd --case sub-01 run-01 --overwrite

Continue an interrupted run:

    python scripts/11_additional_method_pilot_REVISED.py --method all --resume

Quiet MNE logging:

    python scripts/11_additional_method_pilot_REVISED.py \
        --method all --overwrite --quiet

Default output root
-------------------
    outputs/11_additional_method_pilot/

Main table:
    outputs/11_additional_method_pilot/
        11_additional_method_pilot_results.csv

Continuous-ECD time-point tables:
    outputs/11_additional_method_pilot/ecd_timepoints/
        sub-XX_run-XX_continuous_ecd_timepoints.csv

Cached participant BEM solutions:
    outputs/11_additional_method_pilot/bem/
        sub-XX-bem-sol.fif
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
import warnings

import h5py
import mne
import nibabel as nib
import numpy as np
import pandas as pd

from mne.bem import _surfaces_to_bem
from mne.io.constants import FIFF
from mne.transforms import Transform


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from localize_mi import load_run  # noqa: E402
from localize_mi.inverse import (  # noqa: E402
    apply_average_reference,
    create_target_evoked,
    estimate_noise_covariance,
)
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

METHODS = ("Continuous-ECD", "MxNE", "RAP-MUSIC", "LCMV")

TARGET_TMIN = -0.002
TARGET_TMAX = 0.002
COVARIANCE_TMIN = -0.250
COVARIANCE_TMAX = -0.050

BEM_CONDUCTIVITY = (0.3, 0.006, 0.3)
BEM_ICO = 4

FIELDNAMES = (
    # Identification / status
    "subject", "run", "method", "montage", "status",
    "error_type", "error_message",

    # Data
    "epochs", "all_channels", "bad_channels", "good_channels",
    "sampling_frequency_hz",

    # Shared preprocessing
    "covariance_method",
    "covariance_tmin_s", "covariance_tmax_s",
    "target_tmin_s", "target_tmax_s", "target_samples",

    # Ground truth
    "stimulation_pair", "hemisphere",
    "ground_truth_x_mm", "ground_truth_y_mm", "ground_truth_z_mm",

    # Common localization result
    "selection_criterion",
    "selected_time_ms",
    "selected_x_mm", "selected_y_mm", "selected_z_mm",
    "selected_localization_error_mm",

    # Continuous ECD
    "selected_gof_percent",
    "minimum_observed_error_mm",
    "minimum_error_time_ms",
    "gof_at_minimum_error_percent",
    "ecd_min_dist_mm",
    "bem_layers", "bem_ico",
    "conductivity_brain", "conductivity_skull", "conductivity_scalp",
    "bem_file",

    # MxNE
    "mxne_alpha", "mxne_loose", "mxne_depth",
    "mxne_iterations", "mxne_active_sources",
    "mxne_explained_variance_percent",

    # RAP-MUSIC
    "rap_music_n_dipoles",
    "rap_music_gof_percent",

    # LCMV
    "lcmv_reg",
    "lcmv_data_covariance_method",
    "lcmv_data_tmin_s", "lcmv_data_tmax_s",
    "lcmv_data_covariance_samples",
    "lcmv_pick_ori", "lcmv_weight_norm",

    # Cortical-grid diagnostics for grid-based methods
    "peak_vertex",
    "nearest_source_distance_mm",
    "geometric_excess_mm",

    # Runtime
    "runtime_seconds",
)


def normalize_method(value):
    """Normalize terminal spellings to stable CSV method names."""
    key = str(value).strip().replace("-", "").replace("_", "").lower()

    mapping = {
        "ecd": "Continuous-ECD",
        "continuousecd": "Continuous-ECD",
        "dipole": "Continuous-ECD",
        "fitdipole": "Continuous-ECD",
        "mxne": "MxNE",
        "rapmusic": "RAP-MUSIC",
        "music": "RAP-MUSIC",
        "lcmv": "LCMV",
    }

    if key == "all":
        return "ALL"

    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unknown method {value!r}. Choose ecd, mxne, rap_music, "
            "lcmv, or all."
        )

    return mapping[key]


def parse_alpha(value):
    """Accept SURE or a numeric MxNE alpha in [0, 100)."""
    text = str(value).strip().lower()
    if text == "sure":
        return "sure"

    alpha = float(text)
    if not 0 <= alpha < 100:
        raise argparse.ArgumentTypeError(
            "MxNE alpha must be in [0, 100), or 'sure'."
        )
    return alpha


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT / "data" / "Localize-MI",
        help="Localize-MI BIDS root.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT / "outputs" / "11_additional_method_pilot",
        help="Script 11 output directory.",
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
        help=(
            "Method to run. Repeat for multiple methods. "
            "Choices: ecd, mxne, rap_music, lcmv, all. "
            "Default: all."
        ),
    )

    # Continuous ECD baseline parameter
    parser.add_argument(
        "--ecd-min-dist",
        type=float,
        default=5.0,
        help="Minimum ECD distance from inner skull in mm. Default: 5.",
    )

    # MxNE baseline parameters
    parser.add_argument(
        "--mxne-alpha",
        type=parse_alpha,
        default=40.0,
        help="MxNE alpha in [0,100), or SURE. Default: 40.",
    )
    parser.add_argument(
        "--mxne-loose",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mxne-depth",
        type=float,
        default=0.1,
    )

    # RAP-MUSIC baseline parameter
    parser.add_argument(
        "--rap-music-n-dipoles",
        type=int,
        default=1,
        help="Number of RAP-MUSIC dipoles. Default: 1.",
    )

    # LCMV baseline parameters
    parser.add_argument(
        "--lcmv-reg",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lcmv-data-tmin",
        type=float,
        default=-0.005,
    )
    parser.add_argument(
        "--lcmv-data-tmax",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--lcmv-data-covariance-method",
        default="shrunk",
        choices=("empirical", "shrunk", "diagonal_fixed", "auto"),
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def validate_args(args):
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")

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
        raise ValueError(
            "--lcmv-data-tmin must be earlier than --lcmv-data-tmax."
        )


def resolve_methods(values):
    """Resolve --method values, including --method all."""
    if not values or "ALL" in values:
        return list(METHODS)

    return list(dict.fromkeys(values))


def save_rows(path, rows):
    """Atomically checkpoint the main result table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)


def row_key(row):
    return (
        str(row["subject"]),
        str(row["run"]),
        str(row["method"]),
    )


def common_preprocessing(epochs, quiet):
    """Shared average reference, noise covariance and target Evoked."""
    referenced = apply_average_reference(epochs)

    noise_covariance = estimate_noise_covariance(
        referenced,
        tmin=COVARIANCE_TMIN,
        tmax=COVARIANCE_TMAX,
        method="auto",
        verbose=not quiet,
    )

    evoked = create_target_evoked(
        referenced,
        tmin=TARGET_TMIN,
        tmax=TARGET_TMAX,
    )

    return referenced, noise_covariance, evoked


def compute_lcmv_data_covariance(
    epochs,
    method,
    tmin,
    tmax,
    quiet,
):
    """Estimate LCMV data covariance from the requested interval."""
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


def load_gifti_bem_surface(path):
    """Load one released Localize-MI GIFTI BEM surface in millimetres."""
    image = nib.load(path)

    if len(image.darrays) < 2:
        raise ValueError(
            f"GIFTI surface lacks vertices/triangles: {path}"
        )

    rr = np.asarray(image.darrays[0].data, dtype=float)
    tris = np.asarray(image.darrays[1].data, dtype=int)

    if rr.ndim != 2 or rr.shape[1] != 3:
        raise ValueError(
            f"Unexpected surface vertex shape {rr.shape}: {path}"
        )

    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(
            f"Unexpected surface triangle shape {tris.shape}: {path}"
        )

    return {"rr": rr, "tris": tris}


def build_bem_solution(dataset, subject):
    """Reconstruct the released subject-specific 3-layer BEM."""
    anat_dir = (
        dataset
        / "derivatives"
        / "sourcemodelling"
        / subject
        / "anat"
    )

    surface_paths = [
        anat_dir / f"{subject}_inner_skull.surf.gii",
        anat_dir / f"{subject}_outer_skull.surf.gii",
        anat_dir / f"{subject}_outer_skin.surf.gii",
    ]

    for path in surface_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    surfaces = [
        load_gifti_bem_surface(path)
        for path in surface_paths
    ]

    ids = [
        FIFF.FIFFV_BEM_SURF_ID_BRAIN,
        FIFF.FIFFV_BEM_SURF_ID_SKULL,
        FIFF.FIFFV_BEM_SURF_ID_HEAD,
    ]

    bem_surfaces = _surfaces_to_bem(
        surfaces,
        ids,
        BEM_CONDUCTIVITY,
        ico=BEM_ICO,
        rescale=True,
    )

    return mne.make_bem_solution(bem_surfaces)


def get_or_build_bem(dataset, subject, bem_dir):
    """Cache one reconstructed BEM solution per participant."""
    bem_dir.mkdir(parents=True, exist_ok=True)
    path = bem_dir / f"{subject}-bem-sol.fif"

    if path.is_file():
        return mne.read_bem_solution(path), path

    print(f"  Building BEM for {subject}...", flush=True)

    bem = build_bem_solution(dataset, subject)

    mne.write_bem_solution(
        path,
        bem,
        overwrite=True,
    )

    return bem, path


def load_head_to_mri_transform(dataset, subject):
    """Load Localize-MI's released head -> surface/MRI transform."""
    path = (
        dataset
        / "derivatives"
        / "sourcemodelling"
        / subject
        / "xfm"
        / f"{subject}_from-head_to-surface.h5"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as file:
        matrix = np.asarray(file["trans"][()], dtype=float)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 transform, found {matrix.shape}: {path}"
        )

    return Transform(
        fro="head",
        to="mri",
        trans=matrix,
    )


def get_stimulation_and_ground_truth(dataset, subject, loaded):
    """Load stimulation metadata and midpoint in surface/MRI coordinates."""
    electrodes = (
        dataset
        / "derivatives"
        / "epochs"
        / subject
        / "ieeg"
        / f"{subject}_task-seegstim_space-surface_electrodes.tsv"
    )

    stimulation = load_stimulation_info(
        loaded.metadata["Description"],
        electrodes,
    )

    possible_names = (
        "midpoint",
        "midpoint_m",
        "coordinate",
        "coordinates",
        "position",
        "position_m",
        "known_location",
        "known_location_m",
    )

    ground_truth = None

    for name in possible_names:
        if hasattr(stimulation, name):
            value = np.asarray(
                getattr(stimulation, name),
                dtype=float,
            ).squeeze()

            if value.shape == (3,):
                ground_truth = value
                break

    if ground_truth is None:
        raise AttributeError(
            "Could not identify stimulation midpoint."
        )

    # Released stimulation coordinates are expected in metres in the helper,
    # but keep this defensive conversion for direct numeric fields in mm.
    if np.max(np.abs(ground_truth)) > 1:
        ground_truth = ground_truth / 1000.0

    return stimulation, ground_truth


def explained_variance(evoked, residual):
    """Sensor-space variance explained by a sparse fit."""
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


def count_active_sources(source_estimate):
    return int(
        sum(len(vertices) for vertices in source_estimate.vertices)
    )


def run_continuous_ecd(
    evoked,
    noise_covariance,
    bem,
    trans,
    ground_truth_mri,
    min_dist_mm,
    quiet,
):
    """
    Fit one continuous ECD per target sample.

    PRIMARY selection:
        maximum GOF, which does not use ground truth.

    DIAGNOSTIC:
        minimum localization error across the target window.
        This uses ground truth and must never be used as the primary selection.
    """
    dipole, residual = mne.fit_dipole(
        evoked,
        noise_covariance,
        bem,
        trans,
        min_dist=min_dist_mm,
        n_jobs=None,
        verbose=not quiet,
    )

    rows = []

    for index in range(len(dipole.times)):
        coordinate = np.asarray(
            dipole.pos[index],
            dtype=float,
        )

        distance_mm = float(
            np.linalg.norm(
                coordinate - ground_truth_mri
            )
            * 1000.0
        )

        rows.append(
            {
                "time_s": float(dipole.times[index]),
                "time_ms": float(dipole.times[index] * 1000),
                "gof_percent": float(dipole.gof[index]),
                "x_m": float(coordinate[0]),
                "y_m": float(coordinate[1]),
                "z_m": float(coordinate[2]),
                "x_mm": float(coordinate[0] * 1000),
                "y_mm": float(coordinate[1] * 1000),
                "z_mm": float(coordinate[2] * 1000),
                "amplitude_Am": float(dipole.amplitude[index]),
                "orientation_x": float(dipole.ori[index, 0]),
                "orientation_y": float(dipole.ori[index, 1]),
                "orientation_z": float(dipole.ori[index, 2]),
                "localization_distance_mm": distance_mm,
            }
        )

    table = pd.DataFrame(rows)

    selected = table.loc[
        table["gof_percent"].idxmax()
    ]

    minimum_error = table.loc[
        table["localization_distance_mm"].idxmin()
    ]

    diagnostics = {
        "selected": selected,
        "minimum_error": minimum_error,
        "timepoint_table": table,
        "residual": residual,
    }

    return diagnostics


def run_mxne(
    evoked,
    forward,
    noise_covariance,
    alpha,
    loose,
    depth,
    quiet,
):
    """Run baseline MxNE with one mixed-norm iteration."""
    return mne.inverse_sparse.mixed_norm(
        evoked,
        forward,
        noise_covariance,
        alpha=alpha,
        loose=loose,
        depth=depth,
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
    """Run RAP-MUSIC and return fitted dipoles plus residual."""
    return mne.beamformer.rap_music(
        evoked,
        forward,
        noise_covariance,
        n_dipoles=n_dipoles,
        return_residual=True,
        verbose=not quiet,
    )


def select_rap_music_dipole(
    dipoles,
    ground_truth_mri,
):
    """
    Select the RAP-MUSIC dipole for the one-source baseline.

    With n_dipoles=1 there is one fitted dipole object. Within its time
    course, choose maximum GOF without using ground truth.
    """
    if len(dipoles) == 0:
        raise RuntimeError("RAP-MUSIC returned no dipoles.")

    if len(dipoles) != 1:
        raise RuntimeError(
            "Script 11 baseline expects one RAP-MUSIC dipole. "
            "Use --rap-music-n-dipoles 1; multi-dipole behaviour belongs "
            "in the parameter pilot."
        )

    dipole = dipoles[0]

    if len(dipole.times) == 0:
        raise RuntimeError(
            "RAP-MUSIC returned an empty dipole time course."
        )

    index = int(np.argmax(dipole.gof))

    coordinate = np.asarray(
        dipole.pos[index],
        dtype=float,
    )

    distance_mm = float(
        np.linalg.norm(
            coordinate - ground_truth_mri
        )
        * 1000.0
    )

    return {
        "time_ms": float(dipole.times[index] * 1000),
        "gof_percent": float(dipole.gof[index]),
        "coordinate_m": coordinate,
        "distance_mm": distance_mm,
    }


def run_lcmv(
    evoked,
    forward,
    noise_covariance,
    data_covariance,
    reg,
    quiet,
):
    """Create and apply one scalar max-power LCMV beamformer."""
    filters = mne.beamformer.make_lcmv(
        evoked.info,
        forward,
        data_covariance,
        reg=reg,
        noise_cov=noise_covariance,
        pick_ori="max-power",
        weight_norm="unit-noise-gain-invariant",
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
                "LCMV returned a materially complex source estimate: "
                f"imag={imaginary_scale:.3e}, "
                f"real={real_scale:.3e}."
            )

        source_estimate = source_estimate.copy()
        source_estimate._data = data.real

    return source_estimate


def fill_grid_metric_fields(row, metric):
    """Store the common localization metric for grid-based methods."""
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


def main():
    args = parse_args()
    validate_args(args)

    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    output_file = (
        output_root
        / "11_additional_method_pilot_results.csv"
    )

    ecd_timepoint_dir = output_root / "ecd_timepoints"
    bem_dir = output_root / "bem"

    cases = list(
        dict.fromkeys(
            tuple(case)
            for case in (args.case or DEFAULT_CASES)
        )
    )

    methods = resolve_methods(args.method)

    combinations = [
        (subject, run, method)
        for subject, run in cases
        for method in methods
    ]

    if not dataset.is_dir():
        raise NotADirectoryError(
            f"Dataset directory not found: {dataset}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not (
        args.resume or args.overwrite
    ):
        raise FileExistsError(
            f"Output exists: {output_file}\n"
            "Use --resume or --overwrite."
        )

    rows = []

    if args.resume and output_file.exists():
        with output_file.open(
            encoding="utf-8",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)

            if reader.fieldnames != list(FIELDNAMES):
                raise ValueError(
                    "Existing CSV columns do not match revised Script 11."
                )

            rows = list(reader)

        keys = [row_key(row) for row in rows]

        if len(keys) != len(set(keys)):
            raise ValueError(
                "Existing CSV contains duplicate method solutions."
            )

        expected = set(combinations)

        unexpected = set(keys) - expected

        if unexpected:
            raise ValueError(
                "Existing CSV contains solutions outside the current "
                "case/method selection. Use another output root or "
                "--overwrite."
            )

        # Retry failed rows when resuming.
        rows = [
            row
            for row in rows
            if row["status"] == "PASS"
        ]

    elif args.overwrite:
        rows = []

    completed = {
        row_key(row)
        for row in rows
    }

    print("=" * 78)
    print("SCRIPT 11 — ADDITIONAL METHOD PILOT (REVISED)")
    print("=" * 78)
    print(
        f"Cases   : {len(cases)}\n"
        f"Methods : {', '.join(methods)}\n"
        f"Total   : {len(combinations)} method-run combinations\n"
        f"Output  : {output_file}"
    )

    for subject, run in cases:
        pending = [
            method
            for s, r, method in combinations
            if (s, r) == (subject, run)
            and (s, r, method) not in completed
        ]

        if not pending:
            continue

        print("\n" + "=" * 78)
        print(f"{subject} / {run}")
        print("=" * 78)

        loaded = load_run(
            dataset=dataset,
            subject=subject,
            run=run,
            task="seegstim",
        )

        epochs = loaded.epochs.copy()

        n_epochs = len(epochs)
        n_all_channels = len(epochs.ch_names)
        n_bad_channels = len(epochs.info["bads"])

        # Keep only good EEG channels so every method sees the same sensors.
        epochs.pick("eeg", exclude="bads")
        n_good_channels = len(epochs.ch_names)

        stimulation, ground_truth_mri = (
            get_stimulation_and_ground_truth(
                dataset,
                subject,
                loaded,
            )
        )

        surface_transform_file = (
            dataset
            / "derivatives"
            / "sourcemodelling"
            / subject
            / "xfm"
            / f"{subject}_from-head_to-surface.h5"
        )

        surface_transform = load_surface_transform(
            surface_transform_file
        )

        referenced, noise_covariance, evoked = (
            common_preprocessing(
                epochs,
                args.quiet,
            )
        )

        print(
            f"Epochs={n_epochs}; "
            f"channels={n_good_channels}/{n_all_channels}; "
            f"bad={n_bad_channels}; "
            f"covariance={noise_covariance.get('method', 'auto')}",
            flush=True,
        )

        data_covariance = None

        if "LCMV" in pending:
            data_covariance = compute_lcmv_data_covariance(
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
            bem, bem_file = get_or_build_bem(
                dataset,
                subject,
                bem_dir,
            )

            head_to_mri = load_head_to_mri_transform(
                dataset,
                subject,
            )

        for method in pending:
            start = time.monotonic()

            row = dict.fromkeys(FIELDNAMES, "")

            row.update(
                subject=subject,
                run=run,
                method=method,
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
                covariance_tmin_s=COVARIANCE_TMIN,
                covariance_tmax_s=COVARIANCE_TMAX,
                target_tmin_s=TARGET_TMIN,
                target_tmax_s=TARGET_TMAX,
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
                    diagnostics = run_continuous_ecd(
                        evoked,
                        noise_covariance,
                        bem,
                        head_to_mri,
                        ground_truth_mri,
                        args.ecd_min_dist,
                        args.quiet,
                    )

                    selected = diagnostics["selected"]
                    minimum_error = diagnostics[
                        "minimum_error"
                    ]

                    row.update(
                        selection_criterion="maximum_GOF",
                        selected_time_ms=float(
                            selected["time_ms"]
                        ),
                        selected_gof_percent=float(
                            selected["gof_percent"]
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
                        selected_localization_error_mm=float(
                            selected[
                                "localization_distance_mm"
                            ]
                        ),
                        minimum_observed_error_mm=float(
                            minimum_error[
                                "localization_distance_mm"
                            ]
                        ),
                        minimum_error_time_ms=float(
                            minimum_error["time_ms"]
                        ),
                        gof_at_minimum_error_percent=float(
                            minimum_error["gof_percent"]
                        ),
                        ecd_min_dist_mm=args.ecd_min_dist,
                        bem_layers=3,
                        bem_ico=BEM_ICO,
                        conductivity_brain=BEM_CONDUCTIVITY[0],
                        conductivity_skull=BEM_CONDUCTIVITY[1],
                        conductivity_scalp=BEM_CONDUCTIVITY[2],
                        bem_file=str(bem_file),
                    )

                    ecd_timepoint_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    timepoint_file = (
                        ecd_timepoint_dir
                        / f"{subject}_{run}_continuous_ecd_timepoints.csv"
                    )

                    diagnostics[
                        "timepoint_table"
                    ].to_csv(
                        timepoint_file,
                        index=False,
                    )

                elif method == "MxNE":
                    source_estimate, residual = run_mxne(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.mxne_alpha,
                        args.mxne_loose,
                        args.mxne_depth,
                        args.quiet,
                    )

                    active_sources = count_active_sources(
                        source_estimate
                    )

                    if active_sources == 0:
                        raise RuntimeError(
                            "MxNE returned no active cortical sources."
                        )

                    metric = calculate_localization_metrics(
                        source_estimate,
                        loaded.forward,
                        surface_transform,
                        stimulation,
                    )

                    fill_grid_metric_fields(
                        row,
                        metric,
                    )

                    row.update(
                        mxne_alpha=args.mxne_alpha,
                        mxne_loose=args.mxne_loose,
                        mxne_depth=args.mxne_depth,
                        mxne_iterations=1,
                        mxne_active_sources=active_sources,
                        mxne_explained_variance_percent=(
                            explained_variance(
                                evoked,
                                residual,
                            )
                        ),
                    )

                elif method == "RAP-MUSIC":
                    dipoles, residual = run_rap_music(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.rap_music_n_dipoles,
                        args.quiet,
                    )

                    selected = select_rap_music_dipole(
                        dipoles,
                        ground_truth_mri,
                    )

                    coordinate = selected[
                        "coordinate_m"
                    ]

                    row.update(
                        selection_criterion="maximum_GOF",
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
                        selected_localization_error_mm=selected[
                            "distance_mm"
                        ],
                        rap_music_n_dipoles=(
                            args.rap_music_n_dipoles
                        ),
                        rap_music_gof_percent=selected[
                            "gof_percent"
                        ],
                    )

                elif method == "LCMV":
                    source_estimate = run_lcmv(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        data_covariance,
                        args.lcmv_reg,
                        args.quiet,
                    )

                    metric = calculate_localization_metrics(
                        source_estimate,
                        loaded.forward,
                        surface_transform,
                        stimulation,
                    )

                    fill_grid_metric_fields(
                        row,
                        metric,
                    )

                    row.update(
                        lcmv_reg=args.lcmv_reg,
                        lcmv_data_covariance_method=(
                            args.lcmv_data_covariance_method
                        ),
                        lcmv_data_tmin_s=(
                            args.lcmv_data_tmin
                        ),
                        lcmv_data_tmax_s=(
                            args.lcmv_data_tmax
                        ),
                        lcmv_data_covariance_samples=(
                            data_covariance.get(
                                "nfree",
                                "",
                            )
                        ),
                        lcmv_pick_ori="max-power",
                        lcmv_weight_norm=(
                            "unit-noise-gain-invariant"
                        ),
                    )

                else:
                    raise RuntimeError(
                        f"Unhandled method: {method}"
                    )

                row["status"] = "PASS"

            except (
                OSError,
                ValueError,
                RuntimeError,
                TypeError,
                KeyError,
                IndexError,
                np.linalg.LinAlgError,
                MemoryError,
            ) as exc:
                row.update(
                    status="FAIL",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            row["runtime_seconds"] = round(
                time.monotonic() - start,
                3,
            )

            rows.append(row)
            save_rows(output_file, rows)

            if row["status"] == "PASS":
                print(
                    f"  {method:14s} PASS | "
                    f"error="
                    f"{float(row['selected_localization_error_mm']):.2f} mm | "
                    f"time={float(row['selected_time_ms']):+.3f} ms",
                    flush=True,
                )
            else:
                print(
                    f"  {method:14s} FAIL | "
                    f"{row['error_type']}: "
                    f"{row['error_message']}",
                    flush=True,
                )

    failures = sum(
        row["status"] != "PASS"
        for row in rows
    )

    print("\n" + "=" * 78)
    print("SCRIPT 11 COMPLETE")
    print("=" * 78)
    print(
        f"Rows     : {len(rows)}\n"
        f"Failures : {failures}\n"
        f"Output   : {output_file}"
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
