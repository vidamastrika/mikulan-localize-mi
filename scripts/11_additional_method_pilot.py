#!/usr/bin/env python3
"""Pilot four additional source-localization methods on representative runs.

Methods
-------
ECD-grid
    A single equivalent-current-dipole fit constrained to the participant's
    existing cortical source grid. The sensor data and lead field are whitened
    with the same pre-stimulation noise covariance used by the other methods.
    This is deliberately named ECD-grid: it is not an unconstrained continuous
    ``mne.fit_dipole`` fit, which would require a separately saved BEM solution.
LCMV
    Linearly constrained minimum-variance beamformer.
MxNE
    Mixed-norm sparse estimate with one MxNE iteration.
irMxNE
    Iteratively reweighted mixed-norm estimate.

The script follows Script 08's pilot conventions: four representative runs,
all good EEG channels, a -2 to +2 ms target window, atomic CSV checkpoints,
``--resume`` and ``--overwrite`` support, and optional diagnostic figures.

Run from the repository root:

    python scripts/11_additional_method_pilot.py --quiet

Save figures as well:

    python scripts/11_additional_method_pilot.py --save-figures --quiet

Default output:
    outputs/additional_method_pilot/additional_method_pilot_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
import warnings

import mne
import numpy as np
import pandas as pd


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
from localize_mi.plotting import (  # noqa: E402
    create_localization_figure,
    create_peak_time_course_figure,
    create_target_evoked_figure,
)


DEFAULT_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)
METHODS = ("ECD-grid", "LCMV", "MxNE", "irMxNE")
REFERENCE_METHODS = ("MNE", "dSPM", "sLORETA", "eLORETA")

TARGET_TMIN = -0.002
TARGET_TMAX = 0.002
COVARIANCE_TMIN = -0.250
COVARIANCE_TMAX = -0.050

FIELDNAMES = (
    "subject", "run", "method", "status", "error_type", "error_message",
    "montage", "n_good_channels", "epochs", "sampling_frequency_hz",
    "target_tmin_s", "target_tmax_s", "covariance_tmin_s",
    "covariance_tmax_s", "covariance_method_selected",
    "data_covariance_method", "lcmv_data_tmin_s", "lcmv_data_tmax_s",
    "lcmv_data_covariance_samples",
    "loose", "depth", "lcmv_reg",
    "lcmv_pick_ori", "lcmv_weight_norm", "mxne_alpha",
    "mxne_iterations", "evaluated_sources", "active_sources",
    "ecd_selection_criterion",
    "ecd_goodness_of_fit_percent", "explained_variance_percent",
    "stimulation_pair", "hemisphere", "peak_vertex", "peak_time_s",
    "peak_time_ms", "localization_distance_mm",
    "nearest_source_distance_mm", "geometric_excess_mm",
    "estimated_x_m", "estimated_y_m", "estimated_z_m", "known_x_m",
    "known_y_m", "known_z_m", "reference_method",
    "reference_distance_mm", "change_from_reference_mm", "runtime_seconds",
)


def parse_alpha(value):
    """Accept ``sure`` or an MxNE alpha in the interval [0, 100)."""
    text = str(value).strip().lower()
    if text == "sure":
        return "sure"
    alpha = float(text)
    if not 0 <= alpha < 100:
        raise argparse.ArgumentTypeError("MxNE alpha must be in [0, 100), or 'sure'.")
    return alpha


def normalize_method(value):
    """Normalize command-line spellings without changing CSV method names."""
    key = str(value).replace("-", "").replace("_", "").lower()
    mapping = {
        "ecd": "ECD-grid",
        "ecdgrid": "ECD-grid",
        "lcmv": "LCMV",
        "mxne": "MxNE",
        "irmxne": "irMxNE",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unknown method {value!r}; choose ECD-grid, LCMV, MxNE, or irMxNE."
        )
    return mapping[key]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT / "data/Localize-MI",
        help="Localize-MI BIDS root.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(PROJECT / "outputs/additional_method_pilot/"
                 "additional_method_pilot_results.csv"),
    )
    parser.add_argument(
        "--baseline-results", type=Path,
        default=PROJECT / "outputs/tables/inverse_method_results.csv",
        help="Script 05 table used only for a descriptive reference comparison.",
    )
    parser.add_argument(
        "--case", nargs=2, metavar=("SUBJECT", "RUN"), action="append",
        help="Repeat to override the four representative runs.",
    )
    parser.add_argument(
        "--method", type=normalize_method, action="append",
        help="Repeat to select methods; default: all four methods.",
    )
    parser.add_argument("--loose", type=float, default=1.0)
    parser.add_argument("--depth", type=float, default=0.1)
    parser.add_argument("--lcmv-reg", type=float, default=0.05)
    parser.add_argument(
        "--lcmv-data-tmin", type=float, default=-0.005,
        help="Beginning of the LCMV data-covariance interval in seconds.",
    )
    parser.add_argument(
        "--lcmv-data-tmax", type=float, default=0.005,
        help="End of the LCMV data-covariance interval in seconds.",
    )
    parser.add_argument(
        "--lcmv-data-covariance-method", default="shrunk",
        choices=("empirical", "shrunk", "diagonal_fixed", "auto"),
    )
    parser.add_argument(
        "--mxne-alpha", type=parse_alpha, default=40.0,
        help=("Sparse regularization in [0, 100), or 'sure'. Default: 40, "
              "selected after the four-run pilot because SURE returned an "
              "empty solution for sub-05 run-06."),
    )
    parser.add_argument("--irmxne-iterations", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument(
        "--figures-dir", type=Path,
        help="Default: a figures directory beside the output CSV.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")
    if not 0 <= args.loose <= 1:
        raise ValueError("--loose must be between 0 and 1.")
    if not 0 <= args.depth <= 1:
        raise ValueError("--depth must be between 0 and 1.")
    if args.lcmv_reg < 0:
        raise ValueError("--lcmv-reg must be non-negative.")
    if args.lcmv_data_tmin >= args.lcmv_data_tmax:
        raise ValueError("--lcmv-data-tmin must be earlier than --lcmv-data-tmax.")
    if args.irmxne_iterations <= 1:
        raise ValueError("--irmxne-iterations must be greater than 1.")


def load_references(path, cases):
    """Load the best standard-method result per run as context, not a baseline."""
    if not path.is_file():
        print(f"WARNING: reference table not found: {path}", flush=True)
        return {}
    table = pd.read_csv(path)
    required = {
        "subject", "run", "method", "status", "localization_distance_mm"
    }
    missing = required - set(table.columns)
    if missing:
        print(f"WARNING: reference table lacks {sorted(missing)}", flush=True)
        return {}
    table = table.loc[
        table["status"].eq("PASS")
        & table["method"].isin(REFERENCE_METHODS)
    ].copy()
    references = {}
    for subject, run in cases:
        subset = table.loc[
            table["subject"].eq(subject) & table["run"].eq(run)
        ]
        if subset.empty:
            continue
        best = subset.loc[subset["localization_distance_mm"].idxmin()]
        references[(subject, run)] = (
            str(best["method"]), float(best["localization_distance_mm"])
        )
    return references


def save_rows(path, rows):
    """Atomically checkpoint the complete table after every solution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def row_key(row):
    return str(row["subject"]), str(row["run"]), str(row["method"])


def common_preprocessing(epochs, quiet):
    """Perform the shared referencing, covariance and averaging once per run."""
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


def compute_data_covariance(epochs, method, tmin, tmax, quiet):
    """Estimate LCMV data covariance across trials in the target interval."""
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


def run_lcmv(evoked, forward, noise_covariance, data_covariance, reg, quiet):
    """Create and apply one scalar, maximum-power LCMV beamformer."""
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
        imaginary_scale = float(np.max(np.abs(data.imag)))
        real_scale = float(np.max(np.abs(data.real)))
        tolerance = max(1e-12, real_scale * 1e-8)
        if imaginary_scale > tolerance:
            raise RuntimeError(
                "LCMV returned a materially complex source estimate: "
                f"maximum imaginary magnitude {imaginary_scale:.3e}, "
                f"maximum real magnitude {real_scale:.3e}."
            )
        source_estimate = source_estimate.copy()
        source_estimate._data = data.real
    return source_estimate


def run_mxne(
    evoked, forward, noise_covariance, alpha, loose, depth,
    iterations, quiet,
):
    """Run MxNE or irMxNE and return the source estimate and residual."""
    return mne.inverse_sparse.mixed_norm(
        evoked,
        forward,
        noise_covariance,
        alpha=alpha,
        loose=loose,
        depth=depth,
        n_mxne_iter=iterations,
        return_residual=True,
        return_as_dipoles=False,
        rank=None,
        random_state=0,
        verbose=not quiet,
    )


def covariance_matrix_for_names(covariance, names):
    """Return covariance data in exactly the requested channel order."""
    indices = {name: index for index, name in enumerate(covariance.ch_names)}
    missing = [name for name in names if name not in indices]
    if missing:
        raise ValueError(f"Noise covariance lacks channels: {missing}")
    selected = [indices[name] for name in names]
    return np.asarray(covariance.data)[np.ix_(selected, selected)]


def symmetric_whitener(covariance):
    """Construct a rank-aware symmetric whitener from a covariance matrix."""
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    maximum = float(np.max(values))
    if maximum <= 0:
        raise ValueError("Noise covariance is not positive semidefinite.")
    keep = values > maximum * 1e-12
    if not np.any(keep):
        raise ValueError("Noise covariance has zero numerical rank.")
    return (vectors[:, keep] / np.sqrt(values[keep])).T


def make_single_vertex_stc(forward, hemisphere, vertex, time_course, evoked):
    """Represent a fixed-location ECD-grid fit as a sparse SourceEstimate."""
    empty = np.array([], dtype=int)
    vertices = ([np.array([vertex], dtype=int), empty]
                if hemisphere == "lh"
                else [empty, np.array([vertex], dtype=int)])
    subject = forward["src"][0].get("subject_his_id", None)
    return mne.SourceEstimate(
        np.asarray(time_course, dtype=float)[np.newaxis, :],
        vertices=vertices,
        tmin=float(evoked.times[0]),
        tstep=1.0 / float(evoked.info["sfreq"]),
        subject=subject,
    )


def run_ecd_grid(evoked, forward, noise_covariance, hemisphere):
    """Fit one whitened dipole at every cortical vertex and select maximum GOF.

    The search is restricted to the known stimulation hemisphere, matching the
    hemisphere-restricted peak selection used for the distributed methods.
    """
    good_names = [
        name for name in evoked.ch_names if name not in evoked.info["bads"]
    ]
    if not good_names:
        raise ValueError("No good EEG channels remain for ECD-grid.")

    oriented = mne.convert_forward_solution(
        forward,
        surf_ori=True,
        force_fixed=False,
        use_cps=True,
        copy=True,
        verbose=False,
    )
    row_lookup = {
        name: index for index, name in enumerate(oriented["sol"]["row_names"])
    }
    missing = [name for name in good_names if name not in row_lookup]
    if missing:
        raise ValueError(f"Forward solution lacks channels: {missing}")
    row_indices = [row_lookup[name] for name in good_names]
    leadfield = np.asarray(oriented["sol"]["data"])[row_indices]
    data = evoked.copy().pick(good_names).data

    # ``apply_average_reference`` has already projected the EEG data. Apply
    # the corresponding average-reference matrix to the forward lead field
    # and covariance so all three quantities remain in the same subspace.
    channel_count = len(good_names)
    projector = (
        np.eye(channel_count)
        - np.ones((channel_count, channel_count), dtype=float) / channel_count
    )
    covariance = covariance_matrix_for_names(noise_covariance, good_names)
    covariance = projector @ covariance @ projector.T
    whitener = symmetric_whitener(covariance)
    whitened_data = whitener @ projector @ data
    whitened_leadfield = whitener @ projector @ leadfield

    source_vertices = [np.asarray(src["vertno"], dtype=int)
                       for src in oriented["src"]]
    source_count = sum(len(vertices) for vertices in source_vertices)
    if source_count == 0 or whitened_leadfield.shape[1] % source_count:
        raise ValueError("Forward lead-field dimensions do not match its sources.")
    orientations = whitened_leadfield.shape[1] // source_count
    hemi_index = 0 if hemisphere == "lh" else 1
    global_offset = 0 if hemi_index == 0 else len(source_vertices[0])

    total_power = np.sum(whitened_data ** 2, axis=0)
    total_power = np.maximum(total_power, np.finfo(float).eps)
    best_gof = -np.inf
    best_amplitude = -np.inf
    best_vertex = None
    best_time_course = None
    best_time_index = None

    for local_index, vertex in enumerate(source_vertices[hemi_index]):
        global_index = global_offset + local_index
        first = global_index * orientations
        gain = whitened_leadfield[:, first:first + orientations]
        moment = np.linalg.pinv(gain, rcond=1e-12) @ whitened_data
        predicted = gain @ moment
        residual_power = np.sum((whitened_data - predicted) ** 2, axis=0)
        goodness = np.clip(1.0 - residual_power / total_power, 0.0, 1.0)
        amplitudes = np.linalg.norm(moment, axis=0)
        time_index = int(np.argmax(goodness))
        gof = float(goodness[time_index])
        amplitude = float(amplitudes[time_index])
        if (gof > best_gof) or (np.isclose(gof, best_gof) and
                               amplitude > best_amplitude):
            best_gof = gof
            best_amplitude = amplitude
            best_vertex = int(vertex)
            best_time_course = amplitudes
            best_time_index = time_index

    if best_vertex is None:
        raise RuntimeError("ECD-grid did not produce a valid cortical fit.")
    source_estimate = make_single_vertex_stc(
        oriented, hemisphere, best_vertex, best_time_course, evoked
    )
    diagnostics = {
        "gof_percent": best_gof * 100.0,
        "selection_time_s": float(evoked.times[best_time_index]),
    }
    return source_estimate, diagnostics


def explained_variance(evoked, residual):
    """Calculate sensor-space variance explained by a fitted model."""
    names = [name for name in evoked.ch_names if name not in evoked.info["bads"]]
    observed = evoked.copy().pick(names).data
    remaining = residual.copy().pick(names).data
    denominator = float(np.sum(observed ** 2))
    if denominator == 0:
        return np.nan
    return 100.0 * (1.0 - float(np.sum(remaining ** 2)) / denominator)


def count_active_sources(source_estimate):
    return int(sum(len(vertices) for vertices in source_estimate.vertices))


def count_forward_sources(forward):
    """Count cortical vertices evaluated by a distributed method."""
    return int(sum(len(source_space["vertno"]) for source_space in forward["src"]))


def signed_milliseconds(seconds):
    """Format a covariance limit as signed milliseconds."""
    value = float(seconds) * 1000
    if np.isclose(value, 0):
        return "0"
    return f"{value:+g}".replace("-", "−")


def figure_parameter_context(method, args):
    """Return method-specific settings for figure subtitles."""
    if method == "ECD-grid":
        return "ECD criterion: maximum whitened GOF"
    if method == "LCMV":
        return (
            f"Reg: {args.lcmv_reg:g} | "
            f"Data covariance: {args.lcmv_data_covariance_method}\n"
            "Covariance window: "
            f"{signed_milliseconds(args.lcmv_data_tmin)} to "
            f"{signed_milliseconds(args.lcmv_data_tmax)} ms | "
            "Orientation: max-power | Weight norm: unit-noise-gain-invariant"
        )
    iterations = 1 if method == "MxNE" else args.irmxne_iterations
    alpha = args.mxne_alpha if isinstance(args.mxne_alpha, str) else f"{args.mxne_alpha:g}"
    return f"Alpha: {alpha} | MxNE iterations: {iterations}"


def save_figures(
    root, subject, run, method, evoked, source_estimate, metric,
    forward, transform, stimulation, channel_count, loose, depth,
    additional_context,
):
    directory = root / subject / run / method
    directory.mkdir(parents=True, exist_ok=True)
    shared = dict(
        subject=subject,
        run=run,
        stimulation_pair=stimulation.pair,
        montage="all_good",
        channel_count=channel_count,
        target_tmin=TARGET_TMIN,
        target_tmax=TARGET_TMAX,
        additional_context=additional_context,
    )
    sparse_parameters = (
        dict(loose=loose, depth=depth) if method in ("MxNE", "irMxNE")
        else {}
    )
    create_target_evoked_figure(
        evoked=evoked,
        method=method,
        output_file=directory / "01_evoked_target_window.png",
        covariance_tmin=COVARIANCE_TMIN,
        covariance_tmax=COVARIANCE_TMAX,
        **shared,
        **sparse_parameters,
    )
    create_peak_time_course_figure(
        source_estimate=source_estimate,
        peak_row=metric.peak_row,
        peak_time=metric.peak_time,
        method=method,
        output_file=directory / "02_peak_source_time_course.png",
        **shared,
        **sparse_parameters,
    )
    create_localization_figure(
        forward=forward,
        transform=transform,
        peak_coordinate=metric.peak_coordinate,
        stimulation_midpoint=metric.stimulation_midpoint,
        localization_distance_mm=metric.localization_distance_mm,
        nearest_source_coordinate=metric.nearest_source_coordinate,
        nearest_source_distance_mm=metric.nearest_source_distance_mm,
        method=method,
        output_file=directory / "03_localization_comparison.png",
        **shared,
        **sparse_parameters,
    )


def main():
    args = parse_args()
    validate_args(args)
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    figure_root = (args.figures_dir.expanduser().resolve()
                   if args.figures_dir else output.parent / "figures")
    cases = list(dict.fromkeys(tuple(case) for case in (args.case or DEFAULT_CASES)))
    methods = list(dict.fromkeys(args.method or METHODS))
    combinations = [(subject, run, method)
                    for subject, run in cases for method in methods]

    if not dataset.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset}")
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"Output exists: {output}. Use --resume or --overwrite.")

    references = load_references(
        args.baseline_results.expanduser().resolve(), cases
    )
    rows = []
    if args.resume and output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDNAMES):
                raise ValueError("Existing CSV columns do not match Script 11.")
            rows = list(reader)
        existing_keys = [row_key(row) for row in rows]
        if len(existing_keys) != len(set(existing_keys)):
            raise ValueError("Existing CSV contains duplicate method solutions.")
        expected = set(combinations)
        if set(existing_keys) - expected:
            raise ValueError(
                "Existing CSV contains solutions outside this selection; "
                "use a different --output path."
            )
        rows = [row for row in rows if row["status"] == "PASS"]

    completed = {row_key(row) for row in rows}
    print(
        f"Additional-method pilot: {len(cases)} runs × {len(methods)} methods "
        f"= {len(combinations)} solutions"
    )
    print(
        "Methods: " + ", ".join(methods)
        + f"\nAll good channels; target -2 to +2 ms; output: {output}",
        flush=True,
    )

    for subject, run in cases:
        pending = [method for s, r, method in combinations
                   if (s, r) == (subject, run)
                   and (s, r, method) not in completed]
        if not pending:
            continue

        loaded = load_run(
            dataset=dataset, subject=subject, run=run, task="seegstim"
        )
        electrodes = (
            dataset / "derivatives/epochs" / subject / "ieeg"
            / f"{subject}_task-seegstim_space-surface_electrodes.tsv"
        )
        transform_file = (
            dataset / "derivatives/sourcemodelling" / subject / "xfm"
            / f"{subject}_from-head_to-surface.h5"
        )
        stimulation = load_stimulation_info(
            loaded.metadata["Description"], electrodes
        )
        transform = load_surface_transform(transform_file)
        referenced, noise_covariance, evoked = common_preprocessing(
            loaded.epochs, args.quiet
        )
        data_covariance = None
        if "LCMV" in pending:
            data_covariance = compute_data_covariance(
                referenced,
                args.lcmv_data_covariance_method,
                args.lcmv_data_tmin,
                args.lcmv_data_tmax,
                args.quiet,
            )
        channel_count = len(loaded.epochs.ch_names) - len(
            loaded.epochs.info["bads"]
        )

        for method in pending:
            start = time.monotonic()
            row = dict.fromkeys(FIELDNAMES, "")
            row.update(
                subject=subject,
                run=run,
                method=method,
                montage="all_good",
                n_good_channels=channel_count,
                epochs=len(loaded.epochs),
                sampling_frequency_hz=float(loaded.epochs.info["sfreq"]),
                evaluated_sources=count_forward_sources(loaded.forward),
                target_tmin_s=TARGET_TMIN,
                target_tmax_s=TARGET_TMAX,
                covariance_tmin_s=COVARIANCE_TMIN,
                covariance_tmax_s=COVARIANCE_TMAX,
                covariance_method_selected=str(
                    noise_covariance.get("method", "auto")
                ),
                stimulation_pair=stimulation.pair,
                hemisphere=stimulation.hemisphere,
            )
            if (subject, run) in references:
                reference_method, reference_distance = references[(subject, run)]
                row.update(
                    reference_method=reference_method,
                    reference_distance_mm=reference_distance,
                )

            try:
                residual = None
                if method == "ECD-grid":
                    source_estimate, diagnostics = run_ecd_grid(
                        evoked, loaded.forward, noise_covariance,
                        stimulation.hemisphere,
                    )
                    row.update(
                        active_sources=1,
                        ecd_selection_criterion="maximum whitened GOF",
                        ecd_goodness_of_fit_percent=diagnostics["gof_percent"],
                    )
                elif method == "LCMV":
                    source_estimate = run_lcmv(
                        evoked, loaded.forward, noise_covariance,
                        data_covariance, args.lcmv_reg, args.quiet,
                    )
                    row.update(
                        data_covariance_method=args.lcmv_data_covariance_method,
                        lcmv_data_tmin_s=args.lcmv_data_tmin,
                        lcmv_data_tmax_s=args.lcmv_data_tmax,
                        lcmv_data_covariance_samples=data_covariance.get(
                            "nfree", ""
                        ),
                        lcmv_reg=args.lcmv_reg,
                        lcmv_pick_ori="max-power",
                        lcmv_weight_norm="unit-noise-gain-invariant",
                    )
                else:
                    iterations = 1 if method == "MxNE" else args.irmxne_iterations
                    source_estimate, residual = run_mxne(
                        evoked, loaded.forward, noise_covariance,
                        args.mxne_alpha, args.loose, args.depth,
                        iterations, args.quiet,
                    )
                    row.update(
                        loose=args.loose,
                        depth=args.depth,
                        mxne_alpha=args.mxne_alpha,
                        mxne_iterations=iterations,
                        active_sources=count_active_sources(source_estimate),
                        explained_variance_percent=explained_variance(
                            evoked, residual
                        ),
                    )
                    if int(row["active_sources"]) == 0:
                        raise RuntimeError(
                            f"{method} returned no active cortical sources with "
                            f"alpha={args.mxne_alpha}; try a lower fixed --mxne-alpha."
                        )

                metric = calculate_localization_metrics(
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
                )
                if row["reference_distance_mm"] != "":
                    row["change_from_reference_mm"] = (
                        distance - float(row["reference_distance_mm"])
                    )

                if args.save_figures:
                    save_figures(
                        figure_root, subject, run, method, evoked,
                        source_estimate, metric, loaded.forward, transform,
                        stimulation, channel_count, args.loose, args.depth,
                        figure_parameter_context(method, args),
                    )
            except (
                OSError, ValueError, RuntimeError, TypeError, KeyError,
                IndexError, np.linalg.LinAlgError, MemoryError,
            ) as exc:
                row.update(
                    status="FAIL",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            row["runtime_seconds"] = round(time.monotonic() - start, 3)
            rows.append(row)
            save_rows(output, rows)
            if row["status"] == "PASS":
                reference_text = ""
                if row["change_from_reference_mm"] != "":
                    reference_text = (
                        f"; vs {row['reference_method']} "
                        f"{float(row['change_from_reference_mm']):+.2f} mm"
                    )
                print(
                    f"{subject} {run} {method:8}: "
                    f"{float(row['localization_distance_mm']):.2f} mm"
                    f"{reference_text}",
                    flush=True,
                )
            else:
                print(
                    f"{subject} {run} {method} FAIL: {row['error_message']}",
                    flush=True,
                )

    failures = sum(row["status"] != "PASS" for row in rows)
    print(
        f"Saved {len(rows)} rows; failures: {failures}; output: {output}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
