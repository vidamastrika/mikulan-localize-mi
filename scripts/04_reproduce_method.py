#!/usr/bin/env python3
"""
Run one inverse method on one Localize-MI EEG run.

Purpose
-------
This script provides a single-run example for MNE, dSPM, sLORETA, or
eLORETA. It is used to inspect and verify one method before applying it
to the complete dataset.

All methods use the same:

1. Participant-specific EEG geometry.
2. Average EEG reference.
3. Noise-covariance interval.
4. Target time interval.
5. Forward model.
6. Orientation, depth, and SNR settings.

This makes differences between results more likely to come from the
inverse methods themselves rather than inconsistent preprocessing.

Default configuration
---------------------
The default settings reproduce the authors' demonstration:

- Covariance interval: -250 to -50 ms
- Target interval: -2 to +2 ms
- Loose orientation: 1.0
- Depth weighting: 0.1
- SNR: 1.0
- Method: eLORETA

Outputs
-------
Results are saved under:

    outputs/source_localization/<subject>/<run>/<method>/

The output files are:

1. source_localization_summary.json
2. 01_evoked_target_window.png
3. 02_peak_source_time_course.png
4. 03_localization_comparison.png

No dataset files are modified.

Examples
--------
Run eLORETA:

    python scripts/04_reproduce_method.py \
        --subject sub-01 \
        --run run-01 \
        --method eLORETA

Run sLORETA:

    python scripts/04_reproduce_method.py \
        --subject sub-01 \
        --run run-01 \
        --method sLORETA
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import mne


# Locate the repository and reusable source package automatically.
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SOURCE_DIR))

from localize_mi import load_run  # noqa: E402
from localize_mi.inverse import (  # noqa: E402
    normalize_method_name,
    run_inverse,
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


DEFAULT_DATASET = PROJECT_DIR / "data" / "Localize-MI"

DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "source_localization"
)


def parse_arguments():
    """
    Read the run, inverse method, parameters, and output location.

    Returns
    -------
    argparse.Namespace
        Values selected through the command line.
    """

    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Participant ID, for example sub-01.",
    )

    parser.add_argument(
        "--task",
        default="seegstim",
        help="Task name. The released task is seegstim.",
    )

    parser.add_argument(
        "--run",
        default="run-01",
        help="Run ID, for example run-01.",
    )

    parser.add_argument(
        "--method",
        default="eLORETA",
        help="MNE, dSPM, sLORETA, or eLORETA.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the Localize-MI dataset.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for saved results.",
    )

    parser.add_argument(
        "--covariance-tmin",
        type=float,
        default=-0.250,
        help=(
            "Beginning of the covariance interval "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--covariance-tmax",
        type=float,
        default=-0.050,
        help=(
            "End of the covariance interval "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--target-tmin",
        type=float,
        default=-0.002,
        help=(
            "Beginning of the localized interval "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--target-tmax",
        type=float,
        default=0.002,
        help=(
            "End of the localized interval "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--loose",
        type=float,
        default=1.0,
        help=(
            "Source-orientation freedom between 0 and 1."
        ),
    )

    parser.add_argument(
        "--depth",
        type=float,
        default=0.1,
        help=(
            "Depth compensation between 0 and 1."
        ),
    )

    parser.add_argument(
        "--snr",
        type=float,
        default=1.0,
        help="Assumed signal-to-noise ratio.",
    )

    parser.add_argument(
        "--covariance-method",
        default="auto",
        help=(
            "Covariance estimator. The default follows "
            "the authors' example."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide detailed MNE calculation messages.",
    )

    return parser.parse_args()


def require_files(paths):
    """
    Check that the SEEG coordinate and transform files exist.

    Parameters
    ----------
    paths : list of pathlib.Path
        Files required for evaluating localization.

    Returns
    -------
    None
        Nothing is returned when all files exist.

    Raises
    ------
    FileNotFoundError
        Raised with a list of missing files.
    """

    missing = [
        path
        for path in paths
        if not path.is_file()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{formatted}"
        )


def covariance_name(covariance, requested_method):
    """
    Report the covariance estimator selected by MNE.

    When ``auto`` is requested, MNE compares multiple estimators. The
    returned name indicates which estimator was selected.

    Parameters
    ----------
    covariance : mne.Covariance
        Estimated covariance.
    requested_method : str
        Method requested through the command line.

    Returns
    -------
    str
        Selected or requested covariance-method name.
    """

    selected = covariance.get(
        "method",
        requested_method,
    )

    return str(selected)


def main():
    """
    Run one inverse method and save its measurements and figures.

    Returns
    -------
    None
        Results are printed and saved to the output directory.
    """

    args = parse_arguments()

    dataset = (
        args.dataset
        .expanduser()
        .resolve()
    )

    method = normalize_method_name(
        args.method
    )

    output_directory = (
        args.output_root
        .expanduser()
        .resolve()
        / args.subject
        / args.run
        / method
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    task_basename = (
        f"{args.subject}_task-{args.task}"
    )

    seeg_electrode_file = (
        dataset
        / "derivatives"
        / "epochs"
        / args.subject
        / "ieeg"
        / (
            f"{task_basename}"
            "_space-surface_electrodes.tsv"
        )
    )

    transform_file = (
        dataset
        / "derivatives"
        / "sourcemodelling"
        / args.subject
        / "xfm"
        / (
            f"{args.subject}"
            "_from-head_to-surface.h5"
        )
    )

    require_files([
        seeg_electrode_file,
        transform_file,
    ])

    # Load the EEG and matching participant-specific forward model.
    loaded = load_run(
        dataset=dataset,
        subject=args.subject,
        task=args.task,
        run=args.run,
    )

    # Read the known stimulation pair and its surface-space midpoint.
    stimulation = load_stimulation_info(
        description=loaded.metadata["Description"],
        electrode_file=seeg_electrode_file,
    )

    transform = load_surface_transform(
        transform_file
    )

    # Run the same shared workflow regardless of inverse method.
    inverse_result = run_inverse(
        epochs=loaded.epochs,
        forward=loaded.forward,
        method=method,
        covariance_tmin=args.covariance_tmin,
        covariance_tmax=args.covariance_tmax,
        target_tmin=args.target_tmin,
        target_tmax=args.target_tmax,
        loose=args.loose,
        depth=args.depth,
        snr=args.snr,
        covariance_method=args.covariance_method,
        verbose=not args.quiet,
    )

    # Compare the strongest estimated source with the known stimulation
    # midpoint and with the nearest available cortical source.
    metrics = calculate_localization_metrics(
        source_estimate=(
            inverse_result.source_estimate
        ),
        forward=loaded.forward,
        transform=transform,
        stimulation=stimulation,
    )

    covariance_rank = mne.compute_rank(
        inverse_result.covariance,
        info=inverse_result.epochs.info,
        rank=None,
        verbose=False,
    )

    eeg_rank = int(
        covariance_rank.get("eeg", 0)
    )

    selected_covariance = covariance_name(
        inverse_result.covariance,
        args.covariance_method,
    )

    # The geometric excess is the localization error remaining after
    # subtracting the closest distance representable by the source grid.
    geometric_excess_mm = float(
        metrics.localization_distance_mm
        - metrics.nearest_source_distance_mm
    )

    target_figure = (
        output_directory
        / "01_evoked_target_window.png"
    )

    peak_figure = (
        output_directory
        / "02_peak_source_time_course.png"
    )

    localization_figure = (
        output_directory
        / "03_localization_comparison.png"
    )

    summary_file = (
        output_directory
        / "source_localization_summary.json"
    )

    create_target_evoked_figure(
        evoked=inverse_result.evoked,
        method=method,
        output_file=target_figure,
    )

    create_peak_time_course_figure(
        source_estimate=(
            inverse_result.source_estimate
        ),
        peak_row=metrics.peak_row,
        peak_time=metrics.peak_time,
        method=method,
        output_file=peak_figure,
    )

    create_localization_figure(
        forward=loaded.forward,
        transform=transform,
        peak_coordinate=metrics.peak_coordinate,
        stimulation_midpoint=(
            metrics.stimulation_midpoint
        ),
        localization_distance_mm=(
            metrics.localization_distance_mm
        ),
        nearest_source_coordinate=(
            metrics.nearest_source_coordinate
        ),
        nearest_source_distance_mm=(
            metrics.nearest_source_distance_mm
        ),
        method=method,
        output_file=localization_figure,
    )

    summary = {
        "dataset": str(dataset),
        "subject": args.subject,
        "task": args.task,
        "run": args.run,
        "epochs": len(loaded.epochs),
        "good_channels": (
            len(loaded.epochs.ch_names)
            - len(loaded.epochs.info["bads"])
        ),
        "bad_channels": len(
            loaded.epochs.info["bads"]
        ),
        "sampling_frequency_hz": float(
            loaded.epochs.info["sfreq"]
        ),
        "average_reference": True,
        "covariance_tmin_s": (
            inverse_result.covariance_tmin
        ),
        "covariance_tmax_s": (
            inverse_result.covariance_tmax
        ),
        "covariance_requested_method": (
            args.covariance_method
        ),
        "covariance_selected_method": (
            selected_covariance
        ),
        "covariance_nfree": int(
            inverse_result.covariance["nfree"]
        ),
        "covariance_eeg_rank": eeg_rank,
        "target_tmin_s": (
            inverse_result.target_tmin
        ),
        "target_tmax_s": (
            inverse_result.target_tmax
        ),
        "inverse_method": method,
        "loose": inverse_result.loose,
        "depth": inverse_result.depth,
        "snr": inverse_result.snr,
        "lambda2": inverse_result.lambda2,
        "stimulation_pair": stimulation.pair,
        "stimulation_contacts": (
            stimulation.contacts
        ),
        "hemisphere": stimulation.hemisphere,
        "peak_vertex": metrics.peak_vertex,
        "peak_time_s": metrics.peak_time,
        "peak_coordinate_m": (
            metrics.peak_coordinate.tolist()
        ),
        "stimulation_midpoint_m": (
            metrics.stimulation_midpoint.tolist()
        ),
        "localization_distance_mm": (
            metrics.localization_distance_mm
        ),
        "nearest_source_vertex": (
            metrics.nearest_source_vertex
        ),
        "nearest_source_coordinate_m": (
            metrics.nearest_source_coordinate.tolist()
        ),
        "nearest_source_distance_mm": (
            metrics.nearest_source_distance_mm
        ),
        "geometric_excess_mm": (
            geometric_excess_mm
        ),
    }

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nSOURCE-LOCALIZATION RESULT")
    print("--------------------------")
    print(f"Subject             : {args.subject}")
    print(f"Run                 : {args.run}")
    print(f"Method              : {method}")
    print(f"Epochs              : {len(loaded.epochs)}")
    print(
        f"Good channels       : "
        f"{summary['good_channels']}"
    )
    print(
        f"Bad channels        : "
        f"{summary['bad_channels']}"
    )
    print(
        f"Covariance window   : "
        f"{inverse_result.covariance_tmin:.3f} to "
        f"{inverse_result.covariance_tmax:.3f} s"
    )
    print(
        f"Covariance estimator: "
        f"{selected_covariance}"
    )
    print(
        f"Covariance rank     : "
        f"{eeg_rank}"
    )
    print(
        f"Target window       : "
        f"{inverse_result.target_tmin * 1000:.1f} to "
        f"{inverse_result.target_tmax * 1000:.1f} ms"
    )
    print(f"Loose               : {inverse_result.loose}")
    print(f"Depth               : {inverse_result.depth}")
    print(f"SNR                 : {inverse_result.snr}")
    print(f"Lambda²             : {inverse_result.lambda2}")
    print(f"Stimulation pair    : {stimulation.pair}")
    print(f"Hemisphere          : {stimulation.hemisphere}")
    print(f"Peak vertex         : {metrics.peak_vertex}")
    print(
        f"Peak time           : "
        f"{metrics.peak_time * 1000:.3f} ms"
    )
    print(
        "Estimated coordinate: "
        f"[{metrics.peak_coordinate[0]:.5f}, "
        f"{metrics.peak_coordinate[1]:.5f}, "
        f"{metrics.peak_coordinate[2]:.5f}] m"
    )
    print(
        "Known midpoint      : "
        f"[{metrics.stimulation_midpoint[0]:.5f}, "
        f"{metrics.stimulation_midpoint[1]:.5f}, "
        f"{metrics.stimulation_midpoint[2]:.5f}] m"
    )
    print(
        f"Localization distance: "
        f"{metrics.localization_distance_mm:.2f} mm"
    )
    print(
        f"Nearest source limit: "
        f"{metrics.nearest_source_distance_mm:.2f} mm"
    )
    print(
        f"Geometric excess    : "
        f"{geometric_excess_mm:.2f} mm"
    )
    print(f"Summary             : {summary_file}")
    print(f"Target EEG figure   : {target_figure}")
    print(f"Peak time course    : {peak_figure}")
    print(
        f"Localization figure : "
        f"{localization_figure}"
    )


if __name__ == "__main__":
    main()