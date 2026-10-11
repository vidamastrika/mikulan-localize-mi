#!/usr/bin/env python3
"""
22_next_method_parameter_pilot.py

One-factor-at-a-time (OFAT) parameter-sensitivity pilot for the four methods
introduced in Scripts 19-20:

    irMxNE
    Gamma-MAP
    SESAME
    Champagne

PURPOSE
-------
Script 19 established that the four methods can run on the standard pilot
cases. Script 20 established finalized all-run baselines. Script 21 summarizes
all twelve methods.

Script 22 now asks:

    Which method-specific parameters appear influential enough to include
    in the full parameter experiment?

This is an exploratory pilot, NOT an optimization. Only one factor changes
from the finalized baseline at a time.

STANDARD PILOT RUNS
-------------------
    sub-01 run-01
    sub-07 run-07
    sub-05 run-06
    sub-07 run-05

SHARED SETTINGS
---------------
    montage       : all good EEG channels
    target window : -2 to +2 ms

For irMxNE, Gamma-MAP, and Champagne:
    noise covariance : -250 to -50 ms, method="auto"

For SESAME:
    no prestimulus covariance is supplied
    noise_std = fs * max(abs(target Evoked data))
    This is the same operationalization used in the finalized Script 20.

FINAL BASELINES + PILOT LEVELS
------------------------------

1) irMxNE

Baseline:
    alpha        = 40
    loose        = 1.0
    depth        = 0.1
    n_mxne_iter  = 10

OFAT levels:
    alpha        = 20, 40, 60
    loose        = 0.4, 0.7, 1.0
    depth        = 0.1, 0.4, 0.7
    n_mxne_iter  = 5, 10, 15

Unique configurations:
    baseline + 2 + 2 + 2 + 2 = 9


2) Gamma-MAP

Baseline:
    alpha           = 0.3
    loose           = 1.0
    depth           = 0.1
    xyz_same_gamma  = True
    update_mode     = 1
    maxit           = 10000
    tol             = 1e-6

OFAT levels:
    alpha           = 0.1, 0.3, 0.5
    loose           = 0.4, 0.7, 1.0
    depth           = 0.1, 0.4, 0.7
    xyz_same_gamma  = True, False

Unique configurations:
    baseline + 2 + 2 + 2 + 1 = 8


3) SESAME

Baseline:
    fs          = 0.15
    n_parts     = 100
    lam         = 0.25
    max_n_dips  = 1
    hyper_q     = True
    random_seed = 0

OFAT levels:
    fs = 0.10, 0.15, 0.20

All other SESAME parameters stay fixed.

Unique configurations:
    3


4) Champagne

Baseline:
    alpha                  = "auto"
    prune                  = True
    convergence_criterion  = 1e-8
    max_iter               = 2000
    pruning_thresh         = 0.001

OFAT levels:
    convergence_criterion = 1e-6, 1e-8, 1e-10
    prune                 = True, False

alpha remains "auto".

Unique configurations:
    baseline + 2 + 1 = 4


TOTAL WITH FOUR PILOT RUNS
--------------------------
    irMxNE     : 9 x 4 = 36
    Gamma-MAP  : 8 x 4 = 32
    SESAME     : 3 x 4 = 12
    Champagne  : 4 x 4 = 16

    Total = 96 solutions


TERMINAL EXAMPLES
-----------------

Inspect planned configurations:

    python scripts/22_next_method_parameter_pilot.py --method all --dry-run

Run one method:

    python scripts/22_next_method_parameter_pilot.py \
        --method irmxne --overwrite

    python scripts/22_next_method_parameter_pilot.py \
        --method gamma_map --overwrite

    python scripts/22_next_method_parameter_pilot.py \
        --method sesame --overwrite

    python scripts/22_next_method_parameter_pilot.py \
        --method champagne --overwrite

Run all four:

    python scripts/22_next_method_parameter_pilot.py \
        --method all --overwrite

Resume:

    python scripts/22_next_method_parameter_pilot.py \
        --method gamma_map --resume

Run one debugging case:

    python scripts/22_next_method_parameter_pilot.py \
        --method sesame \
        --case sub-01 run-01 \
        --overwrite

Quiet output:

    python scripts/22_next_method_parameter_pilot.py \
        --method all --overwrite --quiet


DEFAULT OUTPUT
--------------
    outputs/22_next_method_parameter_pilot/

Result CSVs:
    22_irmxne_parameter_pilot.csv
    22_gamma_map_parameter_pilot.csv
    22_sesame_parameter_pilot.csv
    22_champagne_parameter_pilot.csv

Summary CSVs:
    22_irmxne_parameter_summary.csv
    22_gamma_map_parameter_summary.csv
    22_sesame_parameter_summary.csv
    22_champagne_parameter_summary.csv

Figures:
    figures/22_<method>_localization_error.png
    figures/22_<method>_change_from_baseline.png

The result CSV is checkpointed after every configuration.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT19 = PROJECT / "scripts" / "19_next_method_pilot.py"

DEFAULT_OUTPUT_ROOT = (
    PROJECT / "outputs" / "22_next_method_parameter_pilot"
)

DEFAULT_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)

METHODS = (
    "irMxNE",
    "Gamma-MAP",
    "SESAME",
    "Champagne",
)

# Finalized Script-20 baselines.
IRMXNE_BASELINE = dict(
    alpha=40.0,
    loose=1.0,
    depth=0.1,
    iterations=10,
)

GAMMA_BASELINE = dict(
    alpha=0.3,
    loose=1.0,
    depth=0.1,
    xyz_same_gamma=True,
)

SESAME_BASELINE = dict(
    fs=0.15,
    n_parts=100,
    max_n_dips=1,
    lam=0.25,
    hyper_q=True,
    random_seed=0,
)

CHAMPAGNE_BASELINE = dict(
    alpha="auto",
    prune=True,
    convergence_criterion=1e-8,
    max_iter=2000,
    pruning_thresh=0.001,
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

    # irMxNE
    "irmxne_alpha", "irmxne_loose", "irmxne_depth",
    "irmxne_iterations", "irmxne_random_state",
    "irmxne_active_sources", "irmxne_explained_variance_percent",

    # Gamma-MAP
    "gamma_alpha", "gamma_loose", "gamma_depth",
    "gamma_xyz_same_gamma",
    "gamma_update_mode", "gamma_maxit", "gamma_tol",
    "gamma_active_sources", "gamma_explained_variance_percent",

    # SESAME
    "sesame_fs",
    "sesame_n_parts", "sesame_max_n_dips", "sesame_lam",
    "sesame_hyper_q", "sesame_random_seed",
    "sesame_noise_std", "sesame_target_peak",
    "sesame_noise_covariance_supplied",
    "sesame_estimated_n_dips",
    "sesame_gof_percent", "sesame_source_dispersion_mm",

    # Champagne
    "champagne_alpha",
    "champagne_prune",
    "champagne_convergence_criterion",
    "champagne_max_iter",
    "champagne_pruning_thresh",
    "champagne_noise_covariance_supplied",
    "champagne_make_operator_signature",

    # Common result
    "selection_criterion",
    "selected_time_ms",
    "selected_x_mm", "selected_y_mm", "selected_z_mm",
    "selected_localization_error_mm",
    "peak_vertex",
    "nearest_source_distance_mm", "geometric_excess_mm",

    # Comparison with this method's own baseline on the same run
    "baseline_error_mm",
    "change_from_baseline_mm",

    "status", "error_type", "error_message",
    "runtime_seconds",
)


def load_script19():
    if not SCRIPT19.is_file():
        raise FileNotFoundError(
            f"Final Script 19 not found: {SCRIPT19}\n"
            "Keep it as scripts/19_next_method_pilot.py."
        )

    spec = importlib.util.spec_from_file_location(
        "localize_mi_script19_for_script22",
        SCRIPT19,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {SCRIPT19}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pilot = load_script19()


def normalize_method(value):
    key = (
        str(value)
        .strip()
        .replace("-", "")
        .replace("_", "")
        .lower()
    )

    mapping = {
        "irmxne": "irMxNE",
        "gamma": "Gamma-MAP",
        "gammamap": "Gamma-MAP",
        "sesame": "SESAME",
        "champagne": "Champagne",
        "champ": "Champagne",
    }

    if key == "all":
        return "ALL"

    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unknown method {value!r}; choose irmxne, gamma_map, "
            "sesame, champagne, or all."
        )

    return mapping[key]


def selected_methods(values):
    if not values or "ALL" in values:
        return list(METHODS)
    return list(dict.fromkeys(values))


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
        "--case",
        nargs=2,
        metavar=("SUBJECT", "RUN"),
        action="append",
        help="Repeat to override the four standard pilot cases.",
    )

    parser.add_argument(
        "--method",
        type=normalize_method,
        action="append",
        help=(
            "Repeat for multiple methods. Choices: irmxne, gamma_map, "
            "sesame, champagne, all. Default: all."
        ),
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)

    return parser.parse_args()


def config(
    configuration_id,
    configuration_label,
    varied_parameter,
    is_baseline=False,
    **kwargs,
):
    row = dict(
        configuration_id=configuration_id,
        configuration_label=configuration_label,
        varied_parameter=varied_parameter,
        is_baseline=bool(is_baseline),
    )
    row.update(kwargs)
    return row


def irmxne_settings():
    b = IRMXNE_BASELINE

    settings = [
        config(
            "baseline",
            "baseline: alpha=40, loose=1.0, depth=0.1, iterations=10",
            "baseline",
            True,
            irmxne_alpha=b["alpha"],
            irmxne_loose=b["loose"],
            irmxne_depth=b["depth"],
            irmxne_iterations=b["iterations"],
        )
    ]

    for value in (20.0, 60.0):
        settings.append(
            config(
                f"alpha_{value:g}",
                f"alpha={value:g}",
                "alpha",
                irmxne_alpha=value,
                irmxne_loose=b["loose"],
                irmxne_depth=b["depth"],
                irmxne_iterations=b["iterations"],
            )
        )

    for value in (0.4, 0.7):
        settings.append(
            config(
                f"loose_{value:g}",
                f"loose={value:g}",
                "loose",
                irmxne_alpha=b["alpha"],
                irmxne_loose=value,
                irmxne_depth=b["depth"],
                irmxne_iterations=b["iterations"],
            )
        )

    for value in (0.4, 0.7):
        settings.append(
            config(
                f"depth_{value:g}",
                f"depth={value:g}",
                "depth",
                irmxne_alpha=b["alpha"],
                irmxne_loose=b["loose"],
                irmxne_depth=value,
                irmxne_iterations=b["iterations"],
            )
        )

    for value in (5, 15):
        settings.append(
            config(
                f"iterations_{value}",
                f"n_mxne_iter={value}",
                "n_mxne_iter",
                irmxne_alpha=b["alpha"],
                irmxne_loose=b["loose"],
                irmxne_depth=b["depth"],
                irmxne_iterations=value,
            )
        )

    return settings


def gamma_settings():
    b = GAMMA_BASELINE

    settings = [
        config(
            "baseline",
            "baseline: alpha=0.3, loose=1.0, depth=0.1, xyz_same_gamma=True",
            "baseline",
            True,
            gamma_alpha=b["alpha"],
            gamma_loose=b["loose"],
            gamma_depth=b["depth"],
            gamma_xyz_same_gamma=b["xyz_same_gamma"],
        )
    ]

    for value in (0.1, 0.5):
        settings.append(
            config(
                f"alpha_{value:g}",
                f"alpha={value:g}",
                "alpha",
                gamma_alpha=value,
                gamma_loose=b["loose"],
                gamma_depth=b["depth"],
                gamma_xyz_same_gamma=b["xyz_same_gamma"],
            )
        )

    for value in (0.4, 0.7):
        settings.append(
            config(
                f"loose_{value:g}",
                f"loose={value:g}",
                "loose",
                gamma_alpha=b["alpha"],
                gamma_loose=value,
                gamma_depth=b["depth"],
                gamma_xyz_same_gamma=b["xyz_same_gamma"],
            )
        )

    for value in (0.4, 0.7):
        settings.append(
            config(
                f"depth_{value:g}",
                f"depth={value:g}",
                "depth",
                gamma_alpha=b["alpha"],
                gamma_loose=b["loose"],
                gamma_depth=value,
                gamma_xyz_same_gamma=b["xyz_same_gamma"],
            )
        )

    settings.append(
        config(
            "xyz_same_gamma_false",
            "xyz_same_gamma=False",
            "xyz_same_gamma",
            gamma_alpha=b["alpha"],
            gamma_loose=b["loose"],
            gamma_depth=b["depth"],
            gamma_xyz_same_gamma=False,
        )
    )

    return settings


def sesame_settings():
    b = SESAME_BASELINE

    settings = []

    for fs in (0.10, 0.15, 0.20):
        settings.append(
            config(
                f"fs_{fs:.2f}".replace(".", "p"),
                f"fs={fs:.2f}",
                "fs" if not np.isclose(fs, b["fs"]) else "baseline",
                bool(np.isclose(fs, b["fs"])),
                sesame_fs=fs,
                sesame_n_parts=b["n_parts"],
                sesame_max_n_dips=b["max_n_dips"],
                sesame_lam=b["lam"],
                sesame_hyper_q=b["hyper_q"],
                sesame_random_seed=b["random_seed"],
            )
        )

    return settings


def champagne_settings():
    b = CHAMPAGNE_BASELINE

    settings = [
        config(
            "baseline",
            "baseline: alpha=auto, prune=True, convergence=1e-8",
            "baseline",
            True,
            champagne_alpha=b["alpha"],
            champagne_prune=b["prune"],
            champagne_convergence_criterion=b["convergence_criterion"],
            champagne_max_iter=b["max_iter"],
            champagne_pruning_thresh=b["pruning_thresh"],
        )
    ]

    for value in (1e-6, 1e-10):
        settings.append(
            config(
                f"convergence_{value:.0e}",
                f"convergence_criterion={value:.0e}",
                "convergence_criterion",
                champagne_alpha=b["alpha"],
                champagne_prune=b["prune"],
                champagne_convergence_criterion=value,
                champagne_max_iter=b["max_iter"],
                champagne_pruning_thresh=b["pruning_thresh"],
            )
        )

    settings.append(
        config(
            "prune_false",
            "prune=False",
            "prune",
            champagne_alpha=b["alpha"],
            champagne_prune=False,
            champagne_convergence_criterion=b["convergence_criterion"],
            champagne_max_iter=b["max_iter"],
            champagne_pruning_thresh=b["pruning_thresh"],
        )
    )

    return settings


def settings_for(method):
    if method == "irMxNE":
        return irmxne_settings()
    if method == "Gamma-MAP":
        return gamma_settings()
    if method == "SESAME":
        return sesame_settings()
    if method == "Champagne":
        return champagne_settings()
    raise ValueError(method)


def method_slug(method):
    return {
        "irMxNE": "irmxne",
        "Gamma-MAP": "gamma_map",
        "SESAME": "sesame",
        "Champagne": "champagne",
    }[method]


def result_path(output_root, method):
    return (
        output_root
        / f"22_{method_slug(method)}_parameter_pilot.csv"
    )


def summary_path(output_root, method):
    return (
        output_root
        / f"22_{method_slug(method)}_parameter_summary.csv"
    )


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDNAMES,
        )
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

    with path.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames != list(FIELDNAMES):
            raise ValueError(
                f"Existing CSV columns do not match Script 22: {path}"
            )

        rows = list(reader)

    keys = [key(row) for row in rows]

    if len(keys) != len(set(keys)):
        raise ValueError(
            f"Duplicate configurations in {path}"
        )

    # Retry FAIL rows on resume.
    return [
        row
        for row in rows
        if str(row.get("status")) == "PASS"
    ]


def count_active_sources(source_estimate):
    return int(
        sum(
            len(vertices)
            for vertices in source_estimate.vertices
        )
    )


def run_gamma_map_variable_xyz(
    evoked,
    forward,
    noise_covariance,
    alpha,
    loose,
    depth,
    xyz_same_gamma,
    quiet,
):
    return mne.inverse_sparse.gamma_map(
        evoked,
        forward,
        noise_covariance,
        alpha=float(alpha),
        loose=float(loose),
        depth=float(depth),
        xyz_same_gamma=bool(xyz_same_gamma),
        maxit=pilot.GAMMA_MAXIT,
        tol=pilot.GAMMA_TOL,
        update_mode=pilot.GAMMA_UPDATE_MODE,
        return_residual=True,
        return_as_dipoles=False,
        rank=None,
        verbose=not quiet,
    )


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
    """
    Same SESAME fs operationalization used for the finalized Script 20:

        noise_std = fs * max(abs(target Evoked data))

    No prestimulus noise covariance is passed to SESAME.
    """
    prepare_sesame = pilot.require_sesame()

    fs = float(fs)

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "SESAME fs must be a positive finite value."
        )

    signal_peak = float(
        np.max(np.abs(evoked.data))
    )

    if (
        not np.isfinite(signal_peak)
        or signal_peak <= 0
    ):
        raise RuntimeError(
            "Cannot derive SESAME noise_std because "
            f"target EEG peak is invalid: {signal_peak!r}"
        )

    noise_std = fs * signal_peak

    np.random.seed(
        int(random_seed)
    )

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

    # Compatibility for SESAMEEG 0.0.3 particle replacement.
    import sesameeg.sesame as sesame_module

    original_particle = sesame_module.Particle

    def compatible_particle(
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

    sesame_module.Particle = compatible_particle

    try:
        sesame.apply_sesame()
    finally:
        sesame_module.Particle = original_particle

    if not sesame.est_n_dips:
        raise RuntimeError(
            "SESAME did not report an estimated source count."
        )

    estimated_n_dips = int(
        sesame.est_n_dips[-1]
    )

    if estimated_n_dips < 1:
        raise RuntimeError(
            "SESAME selected a zero-source model for this target window."
        )

    source_estimate = sesame.compute_stc()

    return (
        sesame,
        source_estimate,
        noise_std,
        signal_peak,
    )


def ensure_champagne_projector_compatibility():
    """
    invertmeeg may call mne.make_projector(..., verbose=...).
    MNE 1.12 keeps the implementation internally and its exact signature can
    differ, so expose a small compatibility wrapper.
    """
    if hasattr(mne, "make_projector"):
        return

    from mne._fiff.proj import (
        make_projector as internal_make_projector,
    )

    def make_projector_compat(*args, **kwargs):
        kwargs.pop("verbose", None)
        return internal_make_projector(
            *args,
            **kwargs,
        )

    mne.make_projector = make_projector_compat


def run_champagne_parameterized(
    evoked,
    forward,
    noise_covariance,
    alpha,
    prune,
    convergence_criterion,
    max_iter,
    pruning_thresh,
    quiet,
):
    """
    Champagne parameter runner using the same channel-alignment logic as
    Script 19, while varying only the selected OFAT parameter.
    """
    ensure_champagne_projector_compatibility()

    forward = mne.pick_channels_forward(
        forward,
        include=list(evoked.ch_names),
        exclude=[],
        ordered=True,
        copy=True,
        verbose=not quiet,
    )

    if list(forward.ch_names) != list(evoked.ch_names):
        raise RuntimeError(
            "Champagne forward channels do not exactly match Evoked channels."
        )

    covariance_names = list(
        getattr(
            noise_covariance,
            "ch_names",
            [],
        )
    )

    if (
        covariance_names
        and covariance_names
        != list(evoked.ch_names)
    ):
        raise RuntimeError(
            "Champagne covariance channels do not exactly "
            "match Evoked channels."
        )

    Solver = pilot.require_champagne_solver()
    solver = Solver("champagne")

    make_operator = solver.make_inverse_operator
    signature = inspect.signature(
        make_operator
    )
    parameters = signature.parameters

    requested = {
        "alpha": alpha,
        "prune": bool(prune),
        "convergence_criterion": float(
            convergence_criterion
        ),
        "max_iter": int(max_iter),
        "pruning_thresh": float(
            pruning_thresh
        ),
    }

    kwargs = {}

    for name, value in requested.items():
        if name in parameters:
            kwargs[name] = value
        elif name in (
            "prune",
            "convergence_criterion",
        ):
            raise RuntimeError(
                f"Installed invertmeeg Champagne API does not expose "
                f"required pilot parameter {name!r}. "
                f"Signature: {signature}"
            )

    covariance_supplied = False

    if "noise_cov" in parameters:
        kwargs["noise_cov"] = (
            noise_covariance
        )
        covariance_supplied = True

    if "mne_obj" in parameters:
        kwargs["mne_obj"] = evoked
    elif "evoked" in parameters:
        kwargs["evoked"] = evoked

    make_operator(
        forward,
        **kwargs,
    )

    source_estimate = (
        solver.apply_inverse_operator(
            evoked
        )
    )

    if isinstance(
        source_estimate,
        (list, tuple),
    ):
        if len(source_estimate) != 1:
            raise RuntimeError(
                "Champagne returned multiple SourceEstimates."
            )

        source_estimate = (
            source_estimate[0]
        )

    if not hasattr(
        source_estimate,
        "data",
    ):
        raise TypeError(
            "Champagne did not return an "
            "MNE SourceEstimate-compatible object."
        )

    return (
        source_estimate,
        str(signature),
        covariance_supplied,
    )


def fill_common_grid_metric(
    row,
    source_estimate,
    loaded,
    surface_transform,
    stimulation,
):
    metric = (
        pilot.pilot.calculate_localization_metrics(
            source_estimate,
            loaded.forward,
            surface_transform,
            stimulation,
        )
    )

    pilot.pilot.fill_grid_metric_fields(
        row,
        metric,
    )

    return metric


def fill_sesame_metric(
    row,
    source_estimate,
    loaded,
    surface_transform,
    stimulation,
):
    metric = fill_common_grid_metric(
        row,
        source_estimate,
        loaded,
        surface_transform,
        stimulation,
    )

    row["selection_criterion"] = (
        "maximum_posterior_location_probability"
    )
    row["selected_time_ms"] = ""

    return metric


def base_row(
    subject,
    run,
    method,
    item,
    loaded,
    epochs,
    noise_covariance,
    evoked,
    stimulation,
    ground_truth_mri,
):
    row = dict.fromkeys(
        FIELDNAMES,
        "",
    )

    n_all_channels = len(
        epochs.ch_names
    )
    n_bad_channels = len(
        epochs.info["bads"]
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
        is_baseline=item[
            "is_baseline"
        ],
        montage="all_good",
        epochs=len(loaded.epochs),
        all_channels=n_all_channels,
        bad_channels=n_bad_channels,
        good_channels=len(evoked.ch_names),
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
            pilot.pilot.COVARIANCE_TMIN
        ),
        covariance_tmax_s=(
            pilot.pilot.COVARIANCE_TMAX
        ),
        target_tmin_s=(
            pilot.pilot.TARGET_TMIN
        ),
        target_tmax_s=(
            pilot.pilot.TARGET_TMAX
        ),
        target_samples=len(
            evoked.times
        ),
        stimulation_pair=(
            stimulation.pair
        ),
        hemisphere=(
            stimulation.hemisphere
        ),
        ground_truth_x_mm=float(
            ground_truth_mri[0]
            * 1000
        ),
        ground_truth_y_mm=float(
            ground_truth_mri[1]
            * 1000
        ),
        ground_truth_z_mm=float(
            ground_truth_mri[2]
            * 1000
        ),
    )

    return row


def apply_baseline_differences(rows):
    """
    Fill baseline_error_mm and change_from_baseline_mm within each subject/run.
    """
    grouped = {}

    for row in rows:
        if row.get("status") != "PASS":
            continue

        grouped.setdefault(
            (
                row["subject"],
                row["run"],
                row["method"],
            ),
            [],
        ).append(row)

    for group_rows in grouped.values():
        baseline_rows = [
            row
            for row in group_rows
            if str(
                row.get("is_baseline")
            ).lower()
            in ("true", "1")
        ]

        if len(baseline_rows) != 1:
            continue

        baseline_error = float(
            baseline_rows[0][
                "selected_localization_error_mm"
            ]
        )

        for row in group_rows:
            error = float(
                row[
                    "selected_localization_error_mm"
                ]
            )

            row[
                "baseline_error_mm"
            ] = baseline_error

            row[
                "change_from_baseline_mm"
            ] = (
                error
                - baseline_error
            )


def make_summary(
    rows,
    path,
):
    table = pd.DataFrame(
        rows
    )

    if table.empty:
        pd.DataFrame().to_csv(
            path,
            index=False,
        )
        return pd.DataFrame()

    passed = table.loc[
        table["status"] == "PASS"
    ].copy()

    if passed.empty:
        pd.DataFrame().to_csv(
            path,
            index=False,
        )
        return pd.DataFrame()

    passed[
        "selected_localization_error_mm"
    ] = pd.to_numeric(
        passed[
            "selected_localization_error_mm"
        ],
        errors="coerce",
    )

    passed[
        "change_from_baseline_mm"
    ] = pd.to_numeric(
        passed[
            "change_from_baseline_mm"
        ],
        errors="coerce",
    )

    grouped = (
        passed.groupby(
            [
                "configuration_id",
                "configuration_label",
                "varied_parameter",
                "is_baseline",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            runs=(
                "selected_localization_error_mm",
                "count",
            ),
            mean_mm=(
                "selected_localization_error_mm",
                "mean",
            ),
            standard_deviation_mm=(
                "selected_localization_error_mm",
                "std",
            ),
            median_mm=(
                "selected_localization_error_mm",
                "median",
            ),
            minimum_mm=(
                "selected_localization_error_mm",
                "min",
            ),
            maximum_mm=(
                "selected_localization_error_mm",
                "max",
            ),
            mean_change_from_baseline_mm=(
                "change_from_baseline_mm",
                "mean",
            ),
            median_change_from_baseline_mm=(
                "change_from_baseline_mm",
                "median",
            ),
        )
    )

    attempted = (
        table.groupby(
            "configuration_id"
        )
        .size()
        .rename(
            "attempted_runs"
        )
    )

    failures = (
        table.assign(
            failed=(
                table["status"]
                != "PASS"
            )
        )
        .groupby(
            "configuration_id"
        )["failed"]
        .sum()
        .rename(
            "failed_runs"
        )
    )

    grouped = grouped.merge(
        attempted,
        on="configuration_id",
        how="left",
    ).merge(
        failures,
        on="configuration_id",
        how="left",
    )

    grouped.to_csv(
        path,
        index=False,
    )

    return grouped


def make_figures(
    rows,
    method,
    figure_dir,
    dpi,
):
    table = pd.DataFrame(
        rows
    )

    if table.empty:
        return

    table = table.loc[
        table["status"] == "PASS"
    ].copy()

    if table.empty:
        return

    table[
        "selected_localization_error_mm"
    ] = pd.to_numeric(
        table[
            "selected_localization_error_mm"
        ],
        errors="coerce",
    )

    table[
        "change_from_baseline_mm"
    ] = pd.to_numeric(
        table[
            "change_from_baseline_mm"
        ],
        errors="coerce",
    )

    settings = settings_for(
        method
    )

    order = [
        item["configuration_id"]
        for item in settings
    ]

    labels = {
        item["configuration_id"]:
        item["configuration_label"]
        for item in settings
    }

    data = []

    used_labels = []

    for configuration_id in order:
        values = table.loc[
            table["configuration_id"]
            == configuration_id,
            "selected_localization_error_mm",
        ].dropna().to_numpy(
            dtype=float
        )

        if len(values):
            data.append(values)
            used_labels.append(
                labels[
                    configuration_id
                ]
            )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if data:
        fig, ax = plt.subplots(
            figsize=(
                max(
                    10,
                    1.15 * len(data),
                ),
                6,
            )
        )

        ax.boxplot(
            data,
            showfliers=True,
        )

        ax.set_xticks(
            np.arange(
                1,
                len(data) + 1,
            ),
            used_labels,
            rotation=35,
            ha="right",
        )

        ax.set_ylabel(
            "Localization error (mm)"
        )

        ax.set_title(
            f"{method}: parameter-pilot localization error"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            figure_dir
            / (
                f"22_{method_slug(method)}"
                "_localization_error.png"
            ),
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(fig)

    change_data = []
    change_labels = []

    for configuration_id in order:
        subset = table.loc[
            (
                table[
                    "configuration_id"
                ]
                == configuration_id
            )
            & (
                table[
                    "is_baseline"
                ].astype(str)
                .str.lower()
                .isin(
                    ["false", "0"]
                )
            ),
            "change_from_baseline_mm",
        ].dropna().to_numpy(
            dtype=float
        )

        if len(subset):
            change_data.append(
                subset
            )
            change_labels.append(
                labels[
                    configuration_id
                ]
            )

    if change_data:
        fig, ax = plt.subplots(
            figsize=(
                max(
                    10,
                    1.15
                    * len(change_data),
                ),
                6,
            )
        )

        ax.boxplot(
            change_data,
            showfliers=True,
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.set_xticks(
            np.arange(
                1,
                len(change_data) + 1,
            ),
            change_labels,
            rotation=35,
            ha="right",
        )

        ax.set_ylabel(
            "Change from baseline localization error (mm)"
        )

        ax.set_title(
            f"{method}: paired change from baseline"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            figure_dir
            / (
                f"22_{method_slug(method)}"
                "_change_from_baseline.png"
            ),
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(fig)


def print_dry_run(
    methods,
    cases,
):
    print("=" * 78)
    print("SCRIPT 22 DRY RUN")
    print("=" * 78)

    print("Cases:")

    for subject, run in cases:
        print(
            f"  {subject} {run}"
        )

    total = 0

    for method in methods:
        settings = settings_for(
            method
        )

        solutions = (
            len(settings)
            * len(cases)
        )

        total += solutions

        print(
            f"\n{method}: "
            f"{len(settings)} configurations "
            f"x {len(cases)} runs = "
            f"{solutions} solutions"
        )

        for item in settings:
            marker = (
                " [BASELINE]"
                if item["is_baseline"]
                else ""
            )

            print(
                f"  "
                f"{item['configuration_id']:25s} "
                f"{item['configuration_label']}"
                f"{marker}"
            )

    print(
        f"\nTOTAL: {total} solutions"
    )


def main():
    args = parse_args()

    if (
        args.resume
        and args.overwrite
    ):
        raise ValueError(
            "Choose --resume or --overwrite, not both."
        )

    if args.dpi < 50:
        raise ValueError(
            "--dpi must be at least 50."
        )

    dataset = (
        args.dataset
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not dataset.is_dir():
        raise NotADirectoryError(
            f"Dataset not found: {dataset}"
        )

    cases = list(
        dict.fromkeys(
            tuple(case)
            for case in (
                args.case
                or DEFAULT_CASES
            )
        )
    )

    methods = selected_methods(
        args.method
    )

    if args.dry_run:
        print_dry_run(
            methods,
            cases,
        )
        return

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir = (
        output_root
        / "figures"
    )

    for method in methods:
        settings = settings_for(
            method
        )

        output_file = result_path(
            output_root,
            method,
        )

        method_summary_file = (
            summary_path(
                output_root,
                method,
            )
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
                item[
                    "configuration_id"
                ],
            )
            for subject, run in cases
            for item in settings
        }

        outside = {
            key(row)
            for row in rows
            if key(row)
            not in expected
        }

        if outside:
            raise ValueError(
                f"{output_file} contains rows outside "
                "the current case/configuration selection. "
                "Use another output root or --overwrite."
            )

        completed = {
            key(row)
            for row in rows
            if row.get("status")
            == "PASS"
        }

        total = (
            len(cases)
            * len(settings)
        )

        print(
            "\n"
            + "=" * 78
        )
        print(
            f"SCRIPT 22 — {method} PARAMETER PILOT"
        )
        print(
            "=" * 78
        )
        print(
            f"Cases          : {len(cases)}\n"
            f"Configurations : {len(settings)}\n"
            f"Solutions      : {total}\n"
            f"Output         : {output_file}"
        )

        for run_index, (
            subject,
            run,
        ) in enumerate(
            cases,
            start=1,
        ):
            pending = [
                item
                for item in settings
                if (
                    subject,
                    run,
                    method,
                    item[
                        "configuration_id"
                    ],
                )
                not in completed
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

            loaded = (
                pilot.pilot.load_run(
                    dataset=dataset,
                    subject=subject,
                    run=run,
                    task="seegstim",
                )
            )

            epochs = (
                loaded.epochs.copy()
            )

            (
                stimulation,
                ground_truth_mri,
            ) = (
                pilot.pilot
                .get_stimulation_and_ground_truth(
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
                / (
                    f"{subject}"
                    "_from-head_to-surface.h5"
                )
            )

            surface_transform = (
                pilot.pilot
                .load_surface_transform(
                    transform_file
                )
            )

            (
                referenced,
                noise_covariance,
                evoked,
            ) = (
                pilot.pilot
                .common_preprocessing(
                    epochs,
                    args.quiet,
                )
            )

            print(
                f"  epochs={len(loaded.epochs)}; "
                f"good channels={len(evoked.ch_names)}/"
                f"{len(epochs.ch_names)}; "
                f"covariance="
                f"{noise_covariance.get('method', 'auto')}",
                flush=True,
            )

            for setting_index, item in enumerate(
                pending,
                start=1,
            ):
                started = (
                    time.monotonic()
                )

                row = base_row(
                    subject,
                    run,
                    method,
                    item,
                    loaded,
                    epochs,
                    noise_covariance,
                    evoked,
                    stimulation,
                    ground_truth_mri,
                )

                try:
                    if method == "irMxNE":
                        alpha = float(
                            item[
                                "irmxne_alpha"
                            ]
                        )
                        loose = float(
                            item[
                                "irmxne_loose"
                            ]
                        )
                        depth = float(
                            item[
                                "irmxne_depth"
                            ]
                        )
                        iterations = int(
                            item[
                                "irmxne_iterations"
                            ]
                        )

                        (
                            source_estimate,
                            residual,
                        ) = pilot.run_irmxne(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            alpha,
                            loose,
                            depth,
                            iterations,
                            args.quiet,
                        )

                        active_sources = (
                            count_active_sources(
                                source_estimate
                            )
                        )

                        if active_sources == 0:
                            raise RuntimeError(
                                "irMxNE returned no "
                                "active cortical sources."
                            )

                        fill_common_grid_metric(
                            row,
                            source_estimate,
                            loaded,
                            surface_transform,
                            stimulation,
                        )

                        row.update(
                            irmxne_alpha=alpha,
                            irmxne_loose=loose,
                            irmxne_depth=depth,
                            irmxne_iterations=iterations,
                            irmxne_random_state=(
                                pilot.IRMXNE_RANDOM_STATE
                            ),
                            irmxne_active_sources=(
                                active_sources
                            ),
                            irmxne_explained_variance_percent=(
                                pilot.pilot
                                .explained_variance(
                                    evoked,
                                    residual,
                                )
                            ),
                        )

                    elif method == "Gamma-MAP":
                        alpha = float(
                            item[
                                "gamma_alpha"
                            ]
                        )
                        loose = float(
                            item[
                                "gamma_loose"
                            ]
                        )
                        depth = float(
                            item[
                                "gamma_depth"
                            ]
                        )
                        xyz_same_gamma = bool(
                            item[
                                "gamma_xyz_same_gamma"
                            ]
                        )

                        (
                            source_estimate,
                            residual,
                        ) = run_gamma_map_variable_xyz(
                            evoked,
                            loaded.forward,
                            noise_covariance,
                            alpha,
                            loose,
                            depth,
                            xyz_same_gamma,
                            args.quiet,
                        )

                        active_sources = (
                            count_active_sources(
                                source_estimate
                            )
                        )

                        if active_sources == 0:
                            raise RuntimeError(
                                "Gamma-MAP returned no "
                                "active cortical sources."
                            )

                        fill_common_grid_metric(
                            row,
                            source_estimate,
                            loaded,
                            surface_transform,
                            stimulation,
                        )

                        row.update(
                            gamma_alpha=alpha,
                            gamma_loose=loose,
                            gamma_depth=depth,
                            gamma_xyz_same_gamma=(
                                xyz_same_gamma
                            ),
                            gamma_update_mode=(
                                pilot.GAMMA_UPDATE_MODE
                            ),
                            gamma_maxit=(
                                pilot.GAMMA_MAXIT
                            ),
                            gamma_tol=(
                                pilot.GAMMA_TOL
                            ),
                            gamma_active_sources=(
                                active_sources
                            ),
                            gamma_explained_variance_percent=(
                                pilot.pilot
                                .explained_variance(
                                    evoked,
                                    residual,
                                )
                            ),
                        )

                    elif method == "SESAME":
                        fs = float(
                            item[
                                "sesame_fs"
                            ]
                        )

                        (
                            sesame,
                            source_estimate,
                            sesame_noise_std,
                            signal_peak,
                        ) = run_sesame_with_fs(
                            evoked,
                            loaded.forward,
                            int(
                                item[
                                    "sesame_n_parts"
                                ]
                            ),
                            int(
                                item[
                                    "sesame_max_n_dips"
                                ]
                            ),
                            float(
                                item[
                                    "sesame_lam"
                                ]
                            ),
                            bool(
                                item[
                                    "sesame_hyper_q"
                                ]
                            ),
                            int(
                                item[
                                    "sesame_random_seed"
                                ]
                            ),
                            fs,
                            args.quiet,
                        )

                        fill_sesame_metric(
                            row,
                            source_estimate,
                            loaded,
                            surface_transform,
                            stimulation,
                        )

                        gof = (
                            sesame.goodness_of_fit()
                        )
                        dispersion = (
                            sesame.source_dispersion()
                        )

                        row.update(
                            covariance_method=(
                                "not_used_by_SESAME_fs"
                            ),
                            sesame_fs=fs,
                            sesame_n_parts=item[
                                "sesame_n_parts"
                            ],
                            sesame_max_n_dips=item[
                                "sesame_max_n_dips"
                            ],
                            sesame_lam=item[
                                "sesame_lam"
                            ],
                            sesame_hyper_q=item[
                                "sesame_hyper_q"
                            ],
                            sesame_random_seed=item[
                                "sesame_random_seed"
                            ],
                            sesame_noise_std=(
                                sesame_noise_std
                            ),
                            sesame_target_peak=(
                                signal_peak
                            ),
                            sesame_noise_covariance_supplied=False,
                            sesame_estimated_n_dips=int(
                                sesame.est_n_dips[-1]
                            ),
                            sesame_gof_percent=float(
                                gof
                            )
                            * 100.0,
                            sesame_source_dispersion_mm=float(
                                dispersion
                            ),
                        )

                    elif method == "Champagne":
                        (
                            source_estimate,
                            signature,
                            covariance_supplied,
                        ) = (
                            run_champagne_parameterized(
                                evoked,
                                loaded.forward,
                                noise_covariance,
                                item[
                                    "champagne_alpha"
                                ],
                                item[
                                    "champagne_prune"
                                ],
                                item[
                                    "champagne_convergence_criterion"
                                ],
                                item[
                                    "champagne_max_iter"
                                ],
                                item[
                                    "champagne_pruning_thresh"
                                ],
                                args.quiet,
                            )
                        )

                        fill_common_grid_metric(
                            row,
                            source_estimate,
                            loaded,
                            surface_transform,
                            stimulation,
                        )

                        row.update(
                            champagne_alpha=item[
                                "champagne_alpha"
                            ],
                            champagne_prune=item[
                                "champagne_prune"
                            ],
                            champagne_convergence_criterion=item[
                                "champagne_convergence_criterion"
                            ],
                            champagne_max_iter=item[
                                "champagne_max_iter"
                            ],
                            champagne_pruning_thresh=item[
                                "champagne_pruning_thresh"
                            ],
                            champagne_noise_covariance_supplied=(
                                covariance_supplied
                            ),
                            champagne_make_operator_signature=(
                                signature
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
                        error_type=(
                            type(exc).__name__
                        ),
                        error_message=str(
                            exc
                        ),
                    )

                row[
                    "runtime_seconds"
                ] = round(
                    time.monotonic()
                    - started,
                    3,
                )

                rows = [
                    existing
                    for existing in rows
                    if key(existing)
                    != key(row)
                ]

                rows.append(
                    row
                )

                apply_baseline_differences(
                    rows
                )

                save_rows(
                    output_file,
                    rows,
                )

                if (
                    row["status"]
                    == "PASS"
                ):
                    completed.add(
                        key(row)
                    )

                    print(
                        f"    "
                        f"[{setting_index}/{len(pending)}] "
                        f"{item['configuration_label']}: "
                        f"PASS | "
                        f"{float(row['selected_localization_error_mm']):.2f} mm",
                        flush=True,
                    )

                else:
                    print(
                        f"    "
                        f"[{setting_index}/{len(pending)}] "
                        f"{item['configuration_label']}: "
                        f"FAIL | "
                        f"{row['error_type']}: "
                        f"{row['error_message']}",
                        flush=True,
                    )

        apply_baseline_differences(
            rows
        )

        save_rows(
            output_file,
            rows,
        )

        summary = make_summary(
            rows,
            method_summary_file,
        )

        make_figures(
            rows,
            method,
            figure_dir,
            args.dpi,
        )

        table = pd.DataFrame(
            rows
        )

        passes = int(
            (
                table["status"]
                == "PASS"
            ).sum()
        )

        failures = int(
            (
                table["status"]
                != "PASS"
            ).sum()
        )

        print(
            "\n"
            + "-" * 78
        )
        print(
            f"{method} COMPLETE"
        )
        print(
            "-" * 78
        )
        print(
            f"Rows     : {len(rows)}"
        )
        print(
            f"Passes   : {passes}"
        )
        print(
            f"Failures : {failures}"
        )
        print(
            f"Results  : {output_file}"
        )
        print(
            f"Summary  : {method_summary_file}"
        )

        if not summary.empty:
            columns = [
                "configuration_label",
                "runs",
                "median_mm",
                "mean_mm",
                "median_change_from_baseline_mm",
                "failed_runs",
            ]

            available = [
                column
                for column in columns
                if column
                in summary.columns
            ]

            print(
                "\nPILOT SUMMARY"
            )

            print(
                summary[
                    available
                ].to_string(
                    index=False,
                    float_format=(
                        lambda value:
                        f"{value:.2f}"
                    ),
                )
            )


if __name__ == "__main__":
    main()
