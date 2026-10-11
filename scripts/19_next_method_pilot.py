#!/usr/bin/env python3
"""
19_next_method_pilot.py

Baseline pilot for the next four Localize-MI EEG source-localization methods:

    irMxNE
    Gamma-MAP
    SESAME
    Champagne

This script deliberately mirrors Script 11. It reuses Script 11's data loading,
preprocessing, ground-truth handling, and localization metric so that the new
methods enter the experiment through the same numerical path as the previous
methods wherever their APIs permit it.

PURPOSE
-------
This is a BASELINE PILOT only. It answers two questions:

    1. Can each new method run reliably on the four standard representative
       Localize-MI cases?
    2. Does each method produce a localization result that can be evaluated by
       the same Localize-MI localization-distance metric used previously?

Parameter sensitivity belongs in a later script. Baselines here are chosen to
be scientifically connected to the previous pipeline, not optimized on these
four runs.

STANDARD PILOT RUNS
-------------------
    sub-01 run-01  : good/easy case
    sub-07 run-07  : another relatively good case
    sub-05 run-06  : severe standard-method outlier
    sub-07 run-05  : intermediate/challenging case

SHARED DATA SETTINGS
--------------------
    montage              : all good EEG channels
    EEG reference        : average reference
    noise covariance     : -250 to -50 ms, method="auto"
    target window        : -2 to +2 ms
    forward/BEM/source   : released Localize-MI participant-specific model
    evaluation           : distance to known stimulation midpoint

BASELINE PARAMETERS
-------------------
1) irMxNE
    Implementation       : mne.inverse_sparse.mixed_norm
    alpha                : 40
    loose                : 1.0
    depth                : 0.1
    n_mxne_iter          : 10
    random_state         : 0

    Connection to previous MxNE:
        Script 11 used alpha=40, loose=1.0, depth=0.1, n_mxne_iter=1.
        Therefore the irMxNE baseline changes only n_mxne_iter from 1 to 10.

2) Gamma-MAP
    Implementation       : mne.inverse_sparse.gamma_map
    alpha                : 0.5
    loose                : 1.0
    depth                : 0.1
    xyz_same_gamma       : True
    update_mode          : 1 (MacKay)
    maxit                : 10000
    tol                  : 1e-6

    Connection to previous methods:
        loose=1.0 and depth=0.1 match the previous MxNE baseline.
        alpha=0.5 is the conventional value used in MNE's Gamma-MAP example;
        Gamma-MAP alpha is NOT numerically equivalent to MxNE alpha.

3) SESAME
    Implementation       : sesameeg.mne.prepare_sesame
    n_parts              : 100
    max_n_dips           : 1
    lam                  : 0.25
    hyper_q              : True
    noise_std            : estimated by SESAME (None)
    dip_mom_std          : estimated by SESAME (None)
    noise_covariance     : same Script-11 covariance supplied for prewhitening
    random_seed          : 0 (best-effort reproducibility)

    Connection to previous methods:
        max_n_dips=1 matches the one-known-source design used for RAP-MUSIC and
        the one Localize-MI stimulation midpoint. The same target window and
        pre-stimulus covariance are used.

4) Champagne
    Implementation       : invertmeeg Solver("champagne")
    alpha                : "auto"
    remaining settings   : invertmeeg Champagne defaults

    Connection to previous methods:
        Champagne has a different parameterization, so no artificial mapping
        from MxNE/MNE alpha, loose, or depth is imposed. It receives the same
        good-channel, average-referenced target Evoked and the same forward
        model. Its implementation version is recorded in the CSV.

    IMPORTANT:
        The current public invertmeeg API exposes Champagne through a unified
        Solver interface. This pilot deliberately treats the package default
        Champagne formulation as the baseline. Champagne update rules and
        implementation-specific regularization choices belong in the later
        parameter-sensitivity pilot.

SELECTION / LOCALIZATION RULE
-----------------------------
irMxNE, Gamma-MAP, Champagne:
    The returned SourceEstimate is evaluated with the same Script-11 helper,
    i.e. the cortical grid point/time with maximum source amplitude is used.

SESAME:
    SESAME estimates source locations jointly over the target interval. The
    posterior probability map from compute_stc() is evaluated spatially, and
    the selected location is the maximum posterior-probability grid location.
    selected_time_ms is therefore left blank for SESAME because a peak time is
    not the scientific selection criterion.

OPTIONAL DEPENDENCIES
---------------------
irMxNE and Gamma-MAP are provided by MNE-Python.
SESAME requires:

    pip install sesameeg

Champagne requires:

    pip install invertmeeg

The installed versions are recorded in the output table.

TERMINAL EXAMPLES
-----------------
Run all four methods on all four standard pilot cases:

    python scripts/19_next_method_pilot.py --method all --overwrite

Run one method only:

    python scripts/19_next_method_pilot.py --method irmxne --overwrite
    python scripts/19_next_method_pilot.py --method gamma_map --overwrite
    python scripts/19_next_method_pilot.py --method sesame --overwrite
    python scripts/19_next_method_pilot.py --method champagne --overwrite

Run one method on one case:

    python scripts/19_next_method_pilot.py \
        --method sesame --case sub-01 run-01 --overwrite

Resume an interrupted pilot:

    python scripts/19_next_method_pilot.py --method all --resume

Check optional packages without running localization:

    python scripts/19_next_method_pilot.py --check-dependencies

Quiet MNE / package logging where supported:

    python scripts/19_next_method_pilot.py --method all --overwrite --quiet

DEFAULT OUTPUT
--------------
    outputs/19_next_method_pilot/
        19_next_method_pilot_results.csv

No parameter-search figure is created here. This script is intentionally the
new-method analogue of Script 11; the all-run baseline and matched summary will
be separate scripts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import time
import warnings

import mne
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT11 = PROJECT / "scripts" / "11_additional_method_pilot.py"

DEFAULT_OUTPUT_ROOT = PROJECT / "outputs" / "19_next_method_pilot"

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

# Baselines deliberately linked to the preceding pipeline.
IRMXNE_ALPHA = 40.0
IRMXNE_LOOSE = 1.0
IRMXNE_DEPTH = 0.1
IRMXNE_ITERATIONS = 10
IRMXNE_RANDOM_STATE = 0

GAMMA_ALPHA = 0.5
GAMMA_LOOSE = 1.0
GAMMA_DEPTH = 0.1
GAMMA_XYZ_SAME_GAMMA = True
GAMMA_UPDATE_MODE = 1
GAMMA_MAXIT = 10000
GAMMA_TOL = 1e-6

SESAME_N_PARTS = 100
SESAME_MAX_N_DIPS = 1
SESAME_LAM = 0.25
SESAME_HYPER_Q = True
SESAME_RANDOM_SEED = 0

CHAMPAGNE_ALPHA = "auto"


FIELDNAMES = (
    # Identification / status
    "subject", "run", "method", "montage", "status",
    "error_type", "error_message",

    # Software provenance
    "mne_version", "sesameeg_version", "invertmeeg_version",

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
    "peak_vertex",
    "nearest_source_distance_mm", "geometric_excess_mm",

    # irMxNE
    "irmxne_alpha", "irmxne_loose", "irmxne_depth",
    "irmxne_iterations", "irmxne_random_state",
    "irmxne_active_sources", "irmxne_explained_variance_percent",

    # Gamma-MAP
    "gamma_alpha", "gamma_loose", "gamma_depth",
    "gamma_xyz_same_gamma", "gamma_update_mode",
    "gamma_maxit", "gamma_tol",
    "gamma_active_sources", "gamma_explained_variance_percent",

    # SESAME
    "sesame_n_parts", "sesame_max_n_dips", "sesame_lam",
    "sesame_hyper_q", "sesame_random_seed",
    "sesame_noise_std_mode", "sesame_dip_mom_std_mode",
    "sesame_noise_covariance_supplied",
    "sesame_estimated_n_dips",
    "sesame_gof_percent", "sesame_source_dispersion_mm",

    # Champagne
    "champagne_solver_id", "champagne_alpha_mode",
    "champagne_noise_covariance_supplied",
    "champagne_make_operator_signature",

    # Runtime
    "runtime_seconds",
)


def load_script11():
    """Import the finalized Script 11 so scientific helpers are reused."""
    if not SCRIPT11.is_file():
        raise FileNotFoundError(
            f"Required Script 11 was not found: {SCRIPT11}\n"
            "Keep the finalized file as scripts/11_additional_method_pilot.py."
        )

    specification = importlib.util.spec_from_file_location(
        "localize_mi_script11_for_script19",
        SCRIPT11,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {SCRIPT11}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


pilot = load_script11()


def package_version(distribution_name):
    """Return installed package version, or an empty string when unavailable."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def normalize_method(value):
    key = str(value).strip().replace("-", "").replace("_", "").lower()

    mapping = {
        "irmxne": "irMxNE",
        "iterativemxne": "irMxNE",
        "iterativereweightedmxne": "irMxNE",
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
            f"Unknown method {value!r}. Choose irmxne, gamma_map, sesame, "
            "champagne, or all."
        )

    return mapping[key]


def resolve_methods(values):
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

    # Baseline values are exposed mainly for one-run debugging. The canonical
    # pilot should be run with defaults; sensitivity experiments come later.
    parser.add_argument("--irmxne-alpha", type=float, default=IRMXNE_ALPHA)
    parser.add_argument("--irmxne-loose", type=float, default=IRMXNE_LOOSE)
    parser.add_argument("--irmxne-depth", type=float, default=IRMXNE_DEPTH)
    parser.add_argument(
        "--irmxne-iterations", type=int, default=IRMXNE_ITERATIONS
    )

    parser.add_argument("--gamma-alpha", type=float, default=GAMMA_ALPHA)
    parser.add_argument("--gamma-loose", type=float, default=GAMMA_LOOSE)
    parser.add_argument("--gamma-depth", type=float, default=GAMMA_DEPTH)

    parser.add_argument("--sesame-n-parts", type=int, default=SESAME_N_PARTS)
    parser.add_argument(
        "--sesame-max-n-dips", type=int, default=SESAME_MAX_N_DIPS
    )
    parser.add_argument("--sesame-lam", type=float, default=SESAME_LAM)
    parser.add_argument(
        "--sesame-random-seed", type=int, default=SESAME_RANDOM_SEED
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Print required/optional package availability and exit.",
    )

    return parser.parse_args()


def validate_args(args):
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")

    if not 0 <= args.irmxne_alpha < 100:
        raise ValueError("--irmxne-alpha must be in [0, 100).")
    if not 0 <= args.irmxne_loose <= 1:
        raise ValueError("--irmxne-loose must be between 0 and 1.")
    if not 0 <= args.irmxne_depth <= 1:
        raise ValueError("--irmxne-depth must be between 0 and 1.")
    if args.irmxne_iterations <= 1:
        raise ValueError(
            "irMxNE requires --irmxne-iterations > 1. "
            "Use Script 11 for ordinary MxNE (n_mxne_iter=1)."
        )

    if args.gamma_alpha <= 0:
        raise ValueError("--gamma-alpha must be positive.")
    if not 0 <= args.gamma_loose <= 1:
        raise ValueError("--gamma-loose must be between 0 and 1.")
    if not 0 <= args.gamma_depth <= 1:
        raise ValueError("--gamma-depth must be between 0 and 1.")

    if args.sesame_n_parts < 2:
        raise ValueError("--sesame-n-parts must be at least 2.")
    if args.sesame_max_n_dips < 1:
        raise ValueError("--sesame-max-n-dips must be at least 1.")
    if args.sesame_lam <= 0:
        raise ValueError("--sesame-lam must be positive.")


def dependency_report():
    records = [
        ("mne", package_version("mne"), True),
        ("sesameeg", package_version("sesameeg"), False),
        ("pyvista", package_version("pyvista"), False),
        ("invertmeeg", package_version("invertmeeg"), False),
    ]

    print("SCRIPT 19 DEPENDENCY CHECK")
    print("=" * 78)
    for name, version, required in records:
        status = version if version else "NOT INSTALLED"
        role = "required" if required else "optional / method-specific"
        print(f"{name:12s}: {status:20s} ({role})")

    if not package_version("sesameeg"):
        print("\nSESAME install:   pip install sesameeg")
    elif not package_version("pyvista"):
        print("\nSESAMEEG is installed but PyVista is missing.")
        print("SESAME fix:       pip install pyvista")
    if not package_version("invertmeeg"):
        print("Champagne install: pip install invertmeeg")


def require_sesame():
    """Import SESAME with compatibility support for current NumPy."""
    # SESAMEEG 0.0.3 still calls np.in1d internally. NumPy removed that alias;
    # np.isin is the supported replacement with equivalent membership semantics
    # for SESAME's usage here.
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    try:
        from sesameeg.mne import prepare_sesame
    except ModuleNotFoundError as exc:
        if exc.name == "pyvista":
            raise ImportError(
                "SESAMEEG is installed, but its visualization module imports "
                "PyVista at package import time and 'pyvista' is missing. "
                "Install it in the same environment with: pip install pyvista"
            ) from exc
        if package_version("sesameeg"):
            raise ImportError(
                "SESAMEEG is installed, but one of its import-time dependencies "
                f"is missing: {exc.name!r}. Install that dependency in the same "
                "environment, then retry."
            ) from exc
        raise ImportError(
            "SESAME requires the 'sesameeg' package. Install it in the same "
            "environment with: pip install sesameeg"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "SESAMEEG could not be imported. Verify the installation with: "
            "python -c \"import sesameeg; print(sesameeg.__version__)\""
        ) from exc
    return prepare_sesame


def ensure_invertmeeg_mne_projector_compatibility(quiet=False):
    """Restore the projector symbol expected by some invertmeeg releases.

    MNE 1.12 no longer exposes make_projector at the mne top level, while
    some invertmeeg code still looks for mne.make_projector.

    This wrapper forwards calls to MNE's current internal implementation,
    but removes the unsupported 'verbose' keyword that invertmeeg may pass.
    """
    if hasattr(mne, "make_projector"):
        return False

    try:
        from mne._fiff.proj import make_projector as _mne_make_projector
    except ImportError as exc:
        raise ImportError(
            "Could not locate MNE's make_projector implementation required "
            "for invertmeeg compatibility."
        ) from exc

    def _make_projector_compat(*args, **kwargs):
        # invertmeeg may pass verbose=..., but MNE 1.12's internal
        # make_projector() does not accept that keyword.
        kwargs.pop("verbose", None)

        return _mne_make_projector(*args, **kwargs)

    mne.make_projector = _make_projector_compat

    if not quiet:
        print(
            "    Compatibility: exposed wrapped "
            "mne._fiff.proj.make_projector as mne.make_projector "
            "for invertmeeg.",
            flush=True,
        )

    return True


def require_champagne_solver():
    try:
        from invert import Solver
    except ImportError as exc:
        raise ImportError(
            "Champagne requires the 'invertmeeg' package (import name: "
            "'invert'). Install it in the same environment with: "
            "pip install invertmeeg"
        ) from exc
    return Solver


def save_rows(path, rows):
    """Atomically checkpoint the main pilot table."""
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


def read_existing_rows(path, resume, overwrite, expected_keys):
    if overwrite or not path.exists():
        return []

    if not resume:
        raise FileExistsError(
            f"Output exists: {path}\nUse --resume or --overwrite."
        )

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDNAMES):
            raise ValueError(
                "Existing CSV columns do not match Script 19. "
                "Use --overwrite after confirming that the old file is no "
                "longer needed."
            )
        rows = list(reader)

    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Existing Script-19 CSV contains duplicate rows.")

    # --resume may be used to ADD another method/case to the same pilot CSV.
    # Preserve rows outside the current selection. Within the current selection,
    # keep PASS rows and remove FAIL rows so they are retried.
    expected_keys = set(expected_keys)
    kept = []
    for row in rows:
        key = row_key(row)
        if key not in expected_keys:
            kept.append(row)
        elif str(row.get("status")) == "PASS":
            kept.append(row)
    return kept


def base_row(
    subject,
    run,
    method,
    loaded,
    epochs,
    noise_covariance,
    evoked,
    stimulation,
    ground_truth_mri,
):
    row = dict.fromkeys(FIELDNAMES, "")
    row.update(
        subject=subject,
        run=run,
        method=method,
        montage="all_good",
        mne_version=package_version("mne"),
        sesameeg_version=package_version("sesameeg"),
        invertmeeg_version=package_version("invertmeeg"),
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
        ground_truth_x_mm=float(ground_truth_mri[0] * 1000.0),
        ground_truth_y_mm=float(ground_truth_mri[1] * 1000.0),
        ground_truth_z_mm=float(ground_truth_mri[2] * 1000.0),
    )
    return row


def count_active_sources(source_estimate):
    """Number of source-space vertices present in a sparse SourceEstimate."""
    return int(sum(len(vertices) for vertices in source_estimate.vertices))


def run_irmxne(
    evoked,
    forward,
    noise_covariance,
    alpha,
    loose,
    depth,
    iterations,
    quiet,
):
    """Run irMxNE, changing only n_mxne_iter relative to Script-11 MxNE."""
    return mne.inverse_sparse.mixed_norm(
        evoked,
        forward,
        noise_covariance,
        alpha=float(alpha),
        loose=float(loose),
        depth=float(depth),
        n_mxne_iter=int(iterations),
        return_residual=True,
        return_as_dipoles=False,
        rank=None,
        random_state=IRMXNE_RANDOM_STATE,
        verbose=not quiet,
    )


def run_gamma_map(
    evoked,
    forward,
    noise_covariance,
    alpha,
    loose,
    depth,
    quiet,
):
    """Run the MNE-Python Gamma-MAP hierarchical Bayesian solver."""
    return mne.inverse_sparse.gamma_map(
        evoked,
        forward,
        noise_covariance,
        alpha=float(alpha),
        loose=float(loose),
        depth=float(depth),
        xyz_same_gamma=GAMMA_XYZ_SAME_GAMMA,
        maxit=GAMMA_MAXIT,
        tol=GAMMA_TOL,
        update_mode=GAMMA_UPDATE_MODE,
        return_residual=True,
        return_as_dipoles=False,
        rank=None,
        verbose=not quiet,
    )


def run_sesame(
    evoked,
    forward,
    noise_covariance,
    n_parts,
    max_n_dips,
    lam,
    random_seed,
    quiet,
):
    """Run SESAME over exactly the Script-19 target Evoked interval."""
    prepare_sesame = require_sesame()

    # SESAMEEG 0.0.3 uses NumPy-based stochastic sampling. This sets the global
    # seed immediately before constructing the sampler. It is a best-effort
    # reproducibility measure and is recorded in the result table.
    np.random.seed(int(random_seed))

    sesame = prepare_sesame(
        forward,
        evoked,
        n_parts=int(n_parts),
        noise_std=None,
        top_min=float(evoked.times[0]),
        top_max=float(evoked.times[-1]),
        dip_mom_std=None,
        hyper_q=SESAME_HYPER_Q,
        noise_cov=noise_covariance,
        subsample=None,
        max_n_dips=int(max_n_dips),
        lam=float(lam),
        verbose=not quiet,
    )

    # SESAMEEG 0.0.3 has a bug in apply_sesame() when max_n_dips is smaller
    # than a Poisson-drawn initial particle count. Its replacement Particle is
    # created without prior_locs/fixed_ori/hyper_q, although the original
    # posterior particles were created with those values. With max_n_dips=1,
    # this path is expected to be reached occasionally. Patch only the Particle
    # symbol used by apply_sesame(), supplying the state that SESAME omitted,
    # then restore the original class immediately afterwards.
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
    return sesame, source_estimate


def run_champagne(evoked, forward, noise_covariance, quiet):
    """
    Run invertmeeg's baseline Champagne solver.

    The public package API is intentionally treated as authoritative rather
    than emulating Champagne inside this project. alpha='auto' is requested.
    If the installed make_inverse_operator explicitly exposes a noise_cov
    parameter, the same Script-11 covariance is supplied; otherwise no hidden
    argument is forced and this fact is recorded for interpretation.
    """
    # invertmeeg versions that call mne.make_projector are incompatible with
    # MNE 1.12's public namespace. Restore the current MNE implementation
    # before constructing the solver so it does not silently use identity.
    ensure_invertmeeg_mne_projector_compatibility(quiet=quiet)

    # invertmeeg expects the forward channels and covariance channels to be
    # identical. Our shared preprocessing physically removes bad EEG channels
    # before constructing the Evoked and noise covariance, whereas the released
    # Localize-MI forward still contains all EEG channels. Restrict a COPY of
    # the forward to the exact Evoked channel names/order before Champagne.
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
            "Champagne channel alignment failed: picked forward channels do "
            "not exactly match evoked.ch_names."
        )

    covariance_names = list(getattr(noise_covariance, "ch_names", []))
    if covariance_names and covariance_names != list(evoked.ch_names):
        missing = [ch for ch in evoked.ch_names if ch not in covariance_names]
        extra = [ch for ch in covariance_names if ch not in evoked.ch_names]
        raise RuntimeError(
            "Champagne covariance channel alignment failed before invertmeeg: "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )

    if not quiet:
        print(
            f"    Champagne aligned forward: {len(forward.ch_names)} channels "
            "matching Evoked/noise covariance.",
            flush=True,
        )

    Solver = require_champagne_solver()
    solver = Solver("champagne")

    make_operator = solver.make_inverse_operator
    signature = inspect.signature(make_operator)
    parameters = signature.parameters

    kwargs = {}
    if "alpha" in parameters:
        kwargs["alpha"] = CHAMPAGNE_ALPHA

    covariance_supplied = False
    if "noise_cov" in parameters:
        kwargs["noise_cov"] = noise_covariance
        covariance_supplied = True

    # Some data-adaptive implementations may explicitly request the data when
    # building the operator. Only supply it if the installed API names it.
    if "mne_obj" in parameters:
        kwargs["mne_obj"] = evoked
    elif "evoked" in parameters:
        kwargs["evoked"] = evoked

    if not quiet:
        print(
            "    Champagne make_inverse_operator signature: "
            f"{signature}",
            flush=True,
        )

    make_operator(forward, **kwargs)
    source_estimate = solver.apply_inverse_operator(evoked)

    # Defensive support for APIs that return a one-element sequence.
    if isinstance(source_estimate, (list, tuple)):
        if len(source_estimate) != 1:
            raise RuntimeError(
                "Champagne returned multiple SourceEstimates. Script 19 "
                "expects one baseline estimate per run."
            )
        source_estimate = source_estimate[0]

    if not hasattr(source_estimate, "data"):
        raise TypeError(
            "Champagne did not return an MNE SourceEstimate-compatible object."
        )

    return source_estimate, str(signature), covariance_supplied


def fill_standard_grid_metric(row, source_estimate, loaded, surface_transform, stimulation):
    """Use the exact Script-11 localization metric for grid-based estimates."""
    metric = pilot.calculate_localization_metrics(
        source_estimate,
        loaded.forward,
        surface_transform,
        stimulation,
    )
    pilot.fill_grid_metric_fields(row, metric)
    return metric


def fill_sesame_metric(row, source_estimate, loaded, surface_transform, stimulation):
    """Evaluate SESAME's posterior location map with the same spatial metric."""
    metric = pilot.calculate_localization_metrics(
        source_estimate,
        loaded.forward,
        surface_transform,
        stimulation,
    )
    pilot.fill_grid_metric_fields(row, metric)

    # The SESAME STC is a posterior location map, not a physiological source
    # time course. Preserve the spatial peak but do not report its synthetic STC
    # time index as a source peak time.
    row["selection_criterion"] = "maximum_posterior_location_probability"
    row["selected_time_ms"] = ""
    return metric


def format_result_time(row):
    value = row.get("selected_time_ms", "")
    if value in ("", None):
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):+.3f} ms"
    except (TypeError, ValueError):
        return str(value)


def main():
    args = parse_args()
    validate_args(args)

    if args.check_dependencies:
        dependency_report()
        return

    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_file = output_root / "19_next_method_pilot_results.csv"

    if not dataset.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset}")

    cases = list(
        dict.fromkeys(tuple(case) for case in (args.case or DEFAULT_CASES))
    )
    methods = resolve_methods(args.method)

    # Fail early only for the optional dependency that is actually requested.
    if "SESAME" in methods:
        require_sesame()
    if "Champagne" in methods:
        require_champagne_solver()

    combinations = [
        (subject, run, method)
        for subject, run in cases
        for method in methods
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_existing_rows(
        output_file,
        args.resume,
        args.overwrite,
        combinations,
    )
    completed = {row_key(row) for row in rows}

    print("=" * 78)
    print("SCRIPT 19 — NEXT METHODS: BASELINE PILOT")
    print("=" * 78)
    print(f"Cases     : {len(cases)}")
    print(f"Methods   : {', '.join(methods)}")
    print(f"Solutions : {len(combinations)}")
    print("Montage   : all good EEG channels")
    print("Target    : -2 to +2 ms")
    print("Noise cov : -250 to -50 ms, auto")
    print(f"Output    : {output_file}")
    print("Baseline parameters:")
    if "irMxNE" in methods:
        print(
            f"  irMxNE    : alpha={args.irmxne_alpha:g}; "
            f"loose={args.irmxne_loose:g}; depth={args.irmxne_depth:g}; "
            f"n_mxne_iter={args.irmxne_iterations}"
        )
    if "Gamma-MAP" in methods:
        print(
            f"  Gamma-MAP : alpha={args.gamma_alpha:g}; "
            f"loose={args.gamma_loose:g}; depth={args.gamma_depth:g}; "
            "xyz_same_gamma=True; update_mode=1"
        )
    if "SESAME" in methods:
        print(
            f"  SESAME    : n_parts={args.sesame_n_parts}; "
            f"max_n_dips={args.sesame_max_n_dips}; "
            f"lam={args.sesame_lam:g}; hyper_q=True"
        )
    if "Champagne" in methods:
        print("  Champagne : Solver('champagne'); alpha='auto'; package defaults")

    for case_index, (subject, run) in enumerate(cases, start=1):
        pending = [
            method
            for method in methods
            if (subject, run, method) not in completed
        ]

        if not pending:
            print(
                f"[{case_index:02d}/{len(cases):02d}] {subject} {run}: "
                "already complete",
                flush=True,
            )
            continue

        print(
            f"\n[{case_index:02d}/{len(cases):02d}] {subject} {run}",
            flush=True,
        )

        # Common preparation: intentionally the same path used by Script 11.
        try:
            loaded = pilot.load_run(
                dataset=dataset,
                subject=subject,
                run=run,
                task="seegstim",
            )
            epochs = loaded.epochs.copy()

            stimulation, ground_truth_mri = pilot.get_stimulation_and_ground_truth(
                dataset,
                subject,
                loaded,
            )

            transform_file = (
                dataset
                / "derivatives"
                / "sourcemodelling"
                / subject
                / "xfm"
                / f"{subject}_from-head_to-surface.h5"
            )
            surface_transform = pilot.load_surface_transform(transform_file)

            referenced, noise_covariance, evoked = pilot.common_preprocessing(
                epochs,
                args.quiet,
            )

        except Exception as exc:
            for method in pending:
                row = dict.fromkeys(FIELDNAMES, "")
                row.update(
                    subject=subject,
                    run=run,
                    method=method,
                    montage="all_good",
                    mne_version=package_version("mne"),
                    sesameeg_version=package_version("sesameeg"),
                    invertmeeg_version=package_version("invertmeeg"),
                    status="FAIL",
                    error_type=type(exc).__name__,
                    error_message=f"Preparation failed: {exc}",
                )
                rows = [existing for existing in rows if row_key(existing) != row_key(row)]
                rows.append(row)

            save_rows(output_file, rows)
            print(
                f"  PREPARATION FAIL: {type(exc).__name__}: {exc}",
                flush=True,
            )
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
                    source_estimate, residual = run_irmxne(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.irmxne_alpha,
                        args.irmxne_loose,
                        args.irmxne_depth,
                        args.irmxne_iterations,
                        args.quiet,
                    )

                    active_sources = count_active_sources(source_estimate)
                    if active_sources == 0:
                        raise RuntimeError("irMxNE returned no active cortical sources.")

                    fill_standard_grid_metric(
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
                        irmxne_random_state=IRMXNE_RANDOM_STATE,
                        irmxne_active_sources=active_sources,
                        irmxne_explained_variance_percent=pilot.explained_variance(
                            evoked,
                            residual,
                        ),
                    )

                elif method == "Gamma-MAP":
                    source_estimate, residual = run_gamma_map(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.gamma_alpha,
                        args.gamma_loose,
                        args.gamma_depth,
                        args.quiet,
                    )

                    active_sources = count_active_sources(source_estimate)
                    if active_sources == 0:
                        raise RuntimeError("Gamma-MAP returned no active cortical sources.")

                    fill_standard_grid_metric(
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
                        gamma_xyz_same_gamma=GAMMA_XYZ_SAME_GAMMA,
                        gamma_update_mode=GAMMA_UPDATE_MODE,
                        gamma_maxit=GAMMA_MAXIT,
                        gamma_tol=GAMMA_TOL,
                        gamma_active_sources=active_sources,
                        gamma_explained_variance_percent=pilot.explained_variance(
                            evoked,
                            residual,
                        ),
                    )

                elif method == "SESAME":
                    sesame, source_estimate = run_sesame(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.sesame_n_parts,
                        args.sesame_max_n_dips,
                        args.sesame_lam,
                        args.sesame_random_seed,
                        args.quiet,
                    )

                    fill_sesame_metric(
                        row,
                        source_estimate,
                        loaded,
                        surface_transform,
                        stimulation,
                    )

                    gof = sesame.goodness_of_fit()
                    dispersion = sesame.source_dispersion()

                    row.update(
                        sesame_n_parts=args.sesame_n_parts,
                        sesame_max_n_dips=args.sesame_max_n_dips,
                        sesame_lam=args.sesame_lam,
                        sesame_hyper_q=SESAME_HYPER_Q,
                        sesame_random_seed=args.sesame_random_seed,
                        sesame_noise_std_mode="estimated_by_SESAME",
                        sesame_dip_mom_std_mode="estimated_by_SESAME",
                        sesame_noise_covariance_supplied=True,
                        sesame_estimated_n_dips=int(sesame.est_n_dips[-1]),
                        sesame_gof_percent=float(gof) * 100.0,
                        sesame_source_dispersion_mm=float(dispersion),
                    )

                elif method == "Champagne":
                    source_estimate, signature, covariance_supplied = run_champagne(
                        evoked,
                        loaded.forward,
                        noise_covariance,
                        args.quiet,
                    )

                    fill_standard_grid_metric(
                        row,
                        source_estimate,
                        loaded,
                        surface_transform,
                        stimulation,
                    )

                    row.update(
                        champagne_solver_id="champagne",
                        champagne_alpha_mode=CHAMPAGNE_ALPHA,
                        champagne_noise_covariance_supplied=covariance_supplied,
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

            row["runtime_seconds"] = round(time.monotonic() - started, 3)

            # Replace an old failed row for this key when --resume retries it.
            rows = [existing for existing in rows if row_key(existing) != row_key(row)]
            rows.append(row)
            save_rows(output_file, rows)

            if row["status"] == "PASS":
                print(
                    f"  {method:10s} PASS | "
                    f"error={float(row['selected_localization_error_mm']):.2f} mm | "
                    f"time={format_result_time(row)}",
                    flush=True,
                )
                completed.add((subject, run, method))
            else:
                print(
                    f"  {method:10s} FAIL | "
                    f"{row['error_type']}: {row['error_message']}",
                    flush=True,
                )

    failures = sum(str(row.get("status")) != "PASS" for row in rows)
    passes = sum(str(row.get("status")) == "PASS" for row in rows)

    print("\n" + "=" * 78)
    print("SCRIPT 19 COMPLETE")
    print("=" * 78)
    print(f"Rows     : {len(rows)}")
    print(f"Passes   : {passes}")
    print(f"Failures : {failures}")
    print(f"Output   : {output_file}")

    if output_file.is_file() and passes:
        table = pd.read_csv(output_file)
        passed = table.loc[table["status"].astype(str).eq("PASS")].copy()
        passed["selected_localization_error_mm"] = pd.to_numeric(
            passed["selected_localization_error_mm"],
            errors="coerce",
        )
        summary = (
            passed.groupby("method", as_index=False)
            .agg(
                runs=("selected_localization_error_mm", "count"),
                mean_mm=("selected_localization_error_mm", "mean"),
                median_mm=("selected_localization_error_mm", "median"),
                minimum_mm=("selected_localization_error_mm", "min"),
                maximum_mm=("selected_localization_error_mm", "max"),
            )
        )
        print("\nPILOT SUMMARY (DESCRIPTIVE ONLY)")
        print(
            summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.2f}",
            )
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
