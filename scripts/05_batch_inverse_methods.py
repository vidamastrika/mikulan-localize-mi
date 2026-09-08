#!/usr/bin/env python3
"""
Run all supported inverse methods across the Localize-MI dataset.

Purpose
-------
This script applies MNE, dSPM, sLORETA, and eLORETA to every released
EEG run. It collects the main localization measurements in one table so
that the methods can later be compared across the complete dataset.

For each run, the script:

1. Loads the released EEG epochs and participant-specific forward model.
2. Applies the average EEG reference.
3. estimates the pre-stimulation noise covariance.
4. Creates one inverse operator.
5. Applies each requested inverse method.
6. Compares each source estimate with the known stimulation location.
7. Saves one result row for every run and method.

The covariance and inverse operator are shared across methods within the
same run because their input data and modelling parameters are identical.

Output
------
The main output is:

    outputs/tables/inverse_method_results.csv

One row represents one run-method combination. The table is saved after
every method, so completed work is retained if the process is interrupted.

Generated figures are intentionally omitted because a complete batch
would otherwise produce hundreds of files. Script 04 remains available
for detailed figures from selected runs.
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd


# Locate the repository and make the local package importable when this
# script is started directly from VS Code or the terminal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


from localize_mi.inverse import (  # noqa: E402
    SUPPORTED_METHODS,
    apply_average_reference,
    apply_inverse_method,
    create_inverse_operator,
    create_target_evoked,
    estimate_noise_covariance,
    normalize_method_name,
)
from localize_mi.io import load_run  # noqa: E402
from localize_mi.metrics import (  # noqa: E402
    calculate_localization_metrics,
    load_stimulation_info,
    load_surface_transform,
)


# These columns give the output table a stable order. This makes the CSV
# easier to inspect and prevents columns from moving when failures occur.
RESULT_COLUMNS = [
    "subject",
    "task",
    "run",
    "method",
    "status",
    "error_type",
    "error_message",
    "epochs",
    "all_channels",
    "good_channels",
    "bad_channels",
    "sampling_frequency_hz",
    "covariance_tmin_s",
    "covariance_tmax_s",
    "covariance_method_requested",
    "covariance_method_selected",
    "covariance_rank",
    "covariance_samples",
    "target_tmin_s",
    "target_tmax_s",
    "loose",
    "depth",
    "snr",
    "lambda2",
    "stimulation_description",
    "stimulation_pair",
    "contact_1",
    "contact_2",
    "hemisphere",
    "peak_vertex",
    "peak_time_s",
    "peak_time_ms",
    "estimated_x_m",
    "estimated_y_m",
    "estimated_z_m",
    "known_x_m",
    "known_y_m",
    "known_z_m",
    "localization_distance_mm",
    "nearest_source_vertex",
    "nearest_source_x_m",
    "nearest_source_y_m",
    "nearest_source_z_m",
    "nearest_source_distance_mm",
    "geometric_excess_mm",
    "runtime_seconds",
]


# Read command-line options and return the selected analysis settings.
#
# Output:
#   argparse.Namespace containing paths, methods, modelling parameters,
#   and options controlling which runs should be processed.
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple inverse methods across all Localize-MI runs."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "Localize-MI",
        help=(
            "Path to the Localize-MI dataset. "
            "Default: data/Localize-MI"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "tables"
            / "inverse_method_results.csv"
        ),
        help=(
            "CSV file used to store the batch results."
        ),
    )

    parser.add_argument(
        "--task",
        default="seegstim",
        help="BIDS task name. Default: seegstim",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_METHODS.values()),
        help=(
            "Inverse methods to run. "
            "Default: MNE dSPM sLORETA eLORETA"
        ),
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help=(
            "Optional participant list, for example "
            "--subjects sub-01 sub-02."
        ),
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help=(
            "Optional run list, for example "
            "--runs run-01 run-02."
        ),
    )

    parser.add_argument(
        "--covariance-tmin",
        type=float,
        default=-0.250,
        help="Beginning of covariance interval in seconds.",
    )

    parser.add_argument(
        "--covariance-tmax",
        type=float,
        default=-0.050,
        help="End of covariance interval in seconds.",
    )

    parser.add_argument(
        "--covariance-method",
        default="auto",
        help=(
            "Covariance estimator passed to MNE. Default: auto"
        ),
    )

    parser.add_argument(
        "--target-tmin",
        type=float,
        default=-0.002,
        help="Beginning of localization interval in seconds.",
    )

    parser.add_argument(
        "--target-tmax",
        type=float,
        default=0.002,
        help="End of localization interval in seconds.",
    )

    parser.add_argument(
        "--loose",
        type=float,
        default=1.0,
        help="Source-orientation freedom. Default: 1.0",
    )

    parser.add_argument(
        "--depth",
        type=float,
        default=0.1,
        help="Depth compensation. Default: 0.1",
    )

    parser.add_argument(
        "--snr",
        type=float,
        default=1.0,
        help="Assumed signal-to-noise ratio. Default: 1.0",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep the existing CSV and skip successful results "
            "that use the same parameters."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing output table and start again."
        ),
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Process only the first N discovered runs. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed MNE calculation messages.",
    )

    arguments = parser.parse_args()

    if arguments.resume and arguments.overwrite:
        parser.error(
            "--resume and --overwrite cannot be used together."
        )

    if arguments.max_runs is not None and arguments.max_runs < 1:
        parser.error(
            "--max-runs must be at least 1."
        )

    arguments.methods = [
        normalize_method_name(method)
        for method in arguments.methods
    ]

    # Remove repeated method names while preserving their order.
    arguments.methods = list(
        dict.fromkeys(arguments.methods)
    )

    return arguments


# Find every released epoch array and extract its participant and run.
#
# Output:
#   A sorted list of (subject, run) pairs.
def discover_runs(
    dataset,
    task,
    selected_subjects=None,
    selected_runs=None,
):
    epochs_root = (
        dataset
        / "derivatives"
        / "epochs"
    )

    if not epochs_root.is_dir():
        raise NotADirectoryError(
            f"Epoch derivatives directory was not found: {epochs_root}"
        )

    pattern = re.compile(
        rf"^(sub-\d+)_task-{re.escape(task)}_"
        rf"(run-\d+)_epochs\.npy$"
    )

    discovered = []

    for array_file in epochs_root.glob(
        "sub-*/eeg/*_epochs.npy"
    ):
        match = pattern.match(array_file.name)

        if match is None:
            continue

        subject, run = match.groups()

        if (
            selected_subjects is not None
            and subject not in selected_subjects
        ):
            continue

        if (
            selected_runs is not None
            and run not in selected_runs
        ):
            continue

        discovered.append(
            (subject, run)
        )

    return sorted(
        set(discovered)
    )


# Return the files containing the stimulation coordinates and transform.
#
# Output:
#   Two Paths: the SEEG electrode table and surface transformation.
def get_stimulation_paths(
    dataset,
    subject,
    task,
):
    electrode_file = (
        dataset
        / "derivatives"
        / "epochs"
        / subject
        / "ieeg"
        / (
            f"{subject}_task-{task}_"
            "space-surface_electrodes.tsv"
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

    return (
        electrode_file,
        transform_file,
    )


# Return a plain text form of the covariance estimator selected by MNE.
#
# Output:
#   A name such as "shrunk", "empirical", or "unknown".
def get_selected_covariance_method(covariance):
    selected = covariance.get(
        "method",
        "unknown",
    )

    if isinstance(selected, (list, tuple)):
        return ",".join(
            str(value)
            for value in selected
        )

    return str(selected)


# Calculate the rank of the covariance used by the inverse model.
#
# Output:
#   Total usable rank across the channel types, normally EEG rank.
def get_covariance_rank(
    covariance,
    info,
):
    rank_by_type = mne.compute_rank(
        covariance,
        info=info,
        verbose=False,
    )

    return int(
        sum(rank_by_type.values())
    )


# Create the fields shared by every method applied to the same run.
#
# Output:
#   A dictionary containing the run identity, data counts, parameters,
#   and known stimulation information.
def create_common_row(
    loaded_run,
    stimulation,
    arguments,
    covariance=None,
    covariance_rank=None,
):
    epochs = loaded_run.epochs
    all_channels = len(epochs.ch_names)
    bad_channels = len(epochs.info["bads"])
    good_channels = all_channels - bad_channels

    description = loaded_run.metadata.get(
        "Description",
        "",
    )

    covariance_selected = ""

    covariance_samples = np.nan

    if covariance is not None:
        covariance_selected = (
            get_selected_covariance_method(
                covariance
            )
        )

        covariance_samples = covariance.get(
            "nfree",
            np.nan,
        )

    common = {
        "subject": loaded_run.subject,
        "task": loaded_run.task,
        "run": loaded_run.run,
        "method": "",
        "status": "",
        "error_type": "",
        "error_message": "",
        "epochs": len(epochs),
        "all_channels": all_channels,
        "good_channels": good_channels,
        "bad_channels": bad_channels,
        "sampling_frequency_hz": float(
            epochs.info["sfreq"]
        ),
        "covariance_tmin_s": float(
            arguments.covariance_tmin
        ),
        "covariance_tmax_s": float(
            arguments.covariance_tmax
        ),
        "covariance_method_requested": str(
            arguments.covariance_method
        ),
        "covariance_method_selected": (
            covariance_selected
        ),
        "covariance_rank": (
            covariance_rank
            if covariance_rank is not None
            else np.nan
        ),
        "covariance_samples": covariance_samples,
        "target_tmin_s": float(
            arguments.target_tmin
        ),
        "target_tmax_s": float(
            arguments.target_tmax
        ),
        "loose": float(arguments.loose),
        "depth": float(arguments.depth),
        "snr": float(arguments.snr),
        "lambda2": 1.0 / float(arguments.snr) ** 2,
        "stimulation_description": description,
        "stimulation_pair": (
            stimulation.pair
            if stimulation is not None
            else ""
        ),
        "contact_1": (
            stimulation.contacts[0]
            if stimulation is not None
            else ""
        ),
        "contact_2": (
            stimulation.contacts[1]
            if stimulation is not None
            else ""
        ),
        "hemisphere": (
            stimulation.hemisphere
            if stimulation is not None
            else ""
        ),
        "peak_vertex": np.nan,
        "peak_time_s": np.nan,
        "peak_time_ms": np.nan,
        "estimated_x_m": np.nan,
        "estimated_y_m": np.nan,
        "estimated_z_m": np.nan,
        "known_x_m": (
            float(stimulation.midpoint[0])
            if stimulation is not None
            else np.nan
        ),
        "known_y_m": (
            float(stimulation.midpoint[1])
            if stimulation is not None
            else np.nan
        ),
        "known_z_m": (
            float(stimulation.midpoint[2])
            if stimulation is not None
            else np.nan
        ),
        "localization_distance_mm": np.nan,
        "nearest_source_vertex": np.nan,
        "nearest_source_x_m": np.nan,
        "nearest_source_y_m": np.nan,
        "nearest_source_z_m": np.nan,
        "nearest_source_distance_mm": np.nan,
        "geometric_excess_mm": np.nan,
        "runtime_seconds": np.nan,
    }

    return common


# Convert successful localization measurements into one table row.
#
# Output:
#   A complete dictionary ready to append to the results CSV.
def create_success_row(
    common,
    method,
    inverse_result,
    metrics,
    runtime_seconds,
):
    row = common.copy()

    row.update({
        "method": method,
        "status": "PASS",
        "peak_vertex": metrics.peak_vertex,
        "peak_time_s": metrics.peak_time,
        "peak_time_ms": (
            metrics.peak_time * 1000
        ),
        "estimated_x_m": float(
            metrics.peak_coordinate[0]
        ),
        "estimated_y_m": float(
            metrics.peak_coordinate[1]
        ),
        "estimated_z_m": float(
            metrics.peak_coordinate[2]
        ),
        "localization_distance_mm": float(
            metrics.localization_distance_mm
        ),
        "nearest_source_vertex": (
            metrics.nearest_source_vertex
        ),
        "nearest_source_x_m": float(
            metrics.nearest_source_coordinate[0]
        ),
        "nearest_source_y_m": float(
            metrics.nearest_source_coordinate[1]
        ),
        "nearest_source_z_m": float(
            metrics.nearest_source_coordinate[2]
        ),
        "nearest_source_distance_mm": float(
            metrics.nearest_source_distance_mm
        ),
        "geometric_excess_mm": float(
            metrics.localization_distance_mm
            - metrics.nearest_source_distance_mm
        ),
        "lambda2": float(
            inverse_result[2]
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
    })

    return row


# Create an output row when a run or method cannot be completed.
#
# Output:
#   A row marked FAIL with the exception name and readable message.
def create_failure_row(
    common,
    method,
    error,
    runtime_seconds,
):
    row = common.copy()

    row.update({
        "method": method,
        "status": "FAIL",
        "error_type": type(error).__name__,
        "error_message": str(error).replace(
            "\n",
            " ",
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
    })

    return row


# Save the complete table through a temporary file and then replace the
# destination. This reduces the chance of leaving a partially written CSV.
#
# Output:
#   The updated CSV at the requested output path.
def save_results(
    rows,
    output_file,
):
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pd.DataFrame(
        rows,
        columns=RESULT_COLUMNS,
    )

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    table.to_csv(
        temporary_file,
        index=False,
    )

    temporary_file.replace(
        output_file
    )


# Load an existing table when --resume is used.
#
# Output:
#   Existing rows and the successful analyses that can be skipped.
def load_existing_results(
    output_file,
    resume,
    overwrite,
):
    if not output_file.exists():
        return [], set()

    if overwrite:
        return [], set()

    if not resume:
        raise FileExistsError(
            f"Output table already exists: {output_file}\n"
            "Use --resume to continue it or --overwrite to replace it."
        )

    table = pd.read_csv(
        output_file
    )

    rows = table.to_dict(
        orient="records"
    )

    return rows, set()


# Decide whether an existing successful result used the current settings.
#
# Output:
#   True when the result can safely be skipped during --resume.
def result_matches_settings(
    row,
    subject,
    run,
    method,
    arguments,
):
    if str(row.get("status")) != "PASS":
        return False

    if str(row.get("subject")) != subject:
        return False

    if str(row.get("run")) != run:
        return False

    if str(row.get("method")) != method:
        return False

    expected_values = {
        "covariance_tmin_s": arguments.covariance_tmin,
        "covariance_tmax_s": arguments.covariance_tmax,
        "target_tmin_s": arguments.target_tmin,
        "target_tmax_s": arguments.target_tmax,
        "loose": arguments.loose,
        "depth": arguments.depth,
        "snr": arguments.snr,
    }

    for column, expected in expected_values.items():
        try:
            observed = float(
                row.get(column)
            )
        except (TypeError, ValueError):
            return False

        if not np.isclose(
            observed,
            float(expected),
        ):
            return False

    requested_method = str(
        row.get(
            "covariance_method_requested",
            "",
        )
    )

    if requested_method != str(
        arguments.covariance_method
    ):
        return False

    return True


# Remove older rows for one run-method combination before saving a new
# result. This prevents duplicate rows after retrying a failed analysis.
#
# Output:
#   The result list without the older matching row.
def remove_previous_result(
    rows,
    subject,
    run,
    method,
):
    return [
        row
        for row in rows
        if not (
            str(row.get("subject")) == subject
            and str(row.get("run")) == run
            and str(row.get("method")) == method
        )
    ]


# Print the final progress summary after the batch ends.
#
# Output:
#   A concise terminal summary of successful and failed analyses.
def print_summary(
    rows,
    output_file,
    expected_methods,
    discovered_runs,
):
    selected_rows = [
        row
        for row in rows
        if (
            str(row.get("method"))
            in expected_methods
            and (
                str(row.get("subject")),
                str(row.get("run")),
            )
            in discovered_runs
        )
    ]

    passed = sum(
        str(row.get("status")) == "PASS"
        for row in selected_rows
    )

    failed = sum(
        str(row.get("status")) == "FAIL"
        for row in selected_rows
    )

    print()
    print("BATCH INVERSE-METHOD SUMMARY")
    print("----------------------------")
    print(
        f"Runs selected       : {len(discovered_runs)}"
    )
    print(
        f"Methods selected    : {len(expected_methods)}"
    )
    print(
        "Expected results    : "
        f"{len(discovered_runs) * len(expected_methods)}"
    )
    print(
        f"Successful results  : {passed}"
    )
    print(
        f"Failed results      : {failed}"
    )
    print(
        f"Saved table         : {output_file}"
    )


# Run all selected methods for every discovered dataset run.
#
# Output:
#   An incrementally saved CSV containing successes and failures.
def main():
    arguments = parse_arguments()

    dataset = (
        arguments.dataset
        .expanduser()
        .resolve()
    )

    output_file = (
        arguments.output
        .expanduser()
        .resolve()
    )

    discovered_runs = discover_runs(
        dataset=dataset,
        task=arguments.task,
        selected_subjects=arguments.subjects,
        selected_runs=arguments.runs,
    )

    if arguments.max_runs is not None:
        discovered_runs = discovered_runs[
            :arguments.max_runs
        ]

    if not discovered_runs:
        raise FileNotFoundError(
            "No matching Localize-MI epoch arrays were found."
        )

    rows, _ = load_existing_results(
        output_file=output_file,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
    )

    total_runs = len(discovered_runs)

    print("LOCALIZE-MI BATCH INVERSE METHODS")
    print("---------------------------------")
    print(f"Dataset          : {dataset}")
    print(f"Runs discovered  : {total_runs}")
    print(
        "Methods          : "
        + ", ".join(arguments.methods)
    )
    print(f"Output           : {output_file}")
    print()

    for run_index, (subject, run) in enumerate(
        discovered_runs,
        start=1,
    ):
        pending_methods = [
            method
            for method in arguments.methods
            if not any(
                result_matches_settings(
                    row=existing_row,
                    subject=subject,
                    run=run,
                    method=method,
                    arguments=arguments,
                )
                for existing_row in rows
            )
        ]

        if not pending_methods:
            print(
                f"[{run_index:02d}/{total_runs:02d}] "
                f"{subject} {run}: already complete"
            )
            continue

        print(
            f"[{run_index:02d}/{total_runs:02d}] "
            f"{subject} {run}"
        )

        loaded_run = None
        referenced_epochs = None
        covariance = None
        inverse_operator = None
        evoked = None
        stimulation = None
        transform = None

        preparation_start = time.perf_counter()

        try:
            loaded_run = load_run(
                dataset=dataset,
                subject=subject,
                task=arguments.task,
                run=run,
            )

            (
                electrode_file,
                transform_file,
            ) = get_stimulation_paths(
                dataset=dataset,
                subject=subject,
                task=arguments.task,
            )

            description = (
                loaded_run.metadata.get(
                    "Description",
                    "",
                )
            )

            stimulation = load_stimulation_info(
                description=description,
                electrode_file=electrode_file,
            )

            transform = load_surface_transform(
                transform_file
            )

            referenced_epochs = apply_average_reference(
                loaded_run.epochs
            )

            covariance = estimate_noise_covariance(
                epochs=referenced_epochs,
                tmin=arguments.covariance_tmin,
                tmax=arguments.covariance_tmax,
                method=arguments.covariance_method,
                verbose=arguments.verbose,
            )

            covariance_rank = get_covariance_rank(
                covariance=covariance,
                info=referenced_epochs.info,
            )

            inverse_operator = create_inverse_operator(
                epochs=referenced_epochs,
                forward=loaded_run.forward,
                covariance=covariance,
                loose=arguments.loose,
                depth=arguments.depth,
                verbose=arguments.verbose,
            )

            evoked = create_target_evoked(
                epochs=referenced_epochs,
                tmin=arguments.target_tmin,
                tmax=arguments.target_tmax,
            )

            common = create_common_row(
                loaded_run=loaded_run,
                stimulation=stimulation,
                arguments=arguments,
                covariance=covariance,
                covariance_rank=covariance_rank,
            )

        except Exception as error:
            preparation_runtime = (
                time.perf_counter()
                - preparation_start
            )

            print(
                "    Preparation failed: "
                f"{type(error).__name__}: {error}"
            )

            if arguments.verbose:
                traceback.print_exc()

            # Loading may fail before a LocalizeMIRun exists. Create a
            # minimal object-like row so the failure is still recorded.
            if loaded_run is None:
                common = {
                    column: np.nan
                    for column in RESULT_COLUMNS
                }

                common.update({
                    "subject": subject,
                    "task": arguments.task,
                    "run": run,
                    "covariance_tmin_s": (
                        arguments.covariance_tmin
                    ),
                    "covariance_tmax_s": (
                        arguments.covariance_tmax
                    ),
                    "covariance_method_requested": (
                        arguments.covariance_method
                    ),
                    "target_tmin_s": (
                        arguments.target_tmin
                    ),
                    "target_tmax_s": (
                        arguments.target_tmax
                    ),
                    "loose": arguments.loose,
                    "depth": arguments.depth,
                    "snr": arguments.snr,
                    "lambda2": (
                        1.0 / arguments.snr**2
                    ),
                })

            else:
                common = create_common_row(
                    loaded_run=loaded_run,
                    stimulation=stimulation,
                    arguments=arguments,
                )

            for method in pending_methods:
                rows = remove_previous_result(
                    rows=rows,
                    subject=subject,
                    run=run,
                    method=method,
                )

                rows.append(
                    create_failure_row(
                        common=common,
                        method=method,
                        error=error,
                        runtime_seconds=preparation_runtime,
                    )
                )

            save_results(
                rows=rows,
                output_file=output_file,
            )

            del loaded_run
            gc.collect()
            continue

        for method in pending_methods:
            method_start = time.perf_counter()

            print(
                f"    {method:<8}",
                end="",
                flush=True,
            )

            try:
                inverse_result = apply_inverse_method(
                    evoked=evoked,
                    inverse_operator=inverse_operator,
                    method=method,
                    snr=arguments.snr,
                    verbose=arguments.verbose,
                )

                source_estimate = inverse_result[0]

                metrics = calculate_localization_metrics(
                    source_estimate=source_estimate,
                    forward=loaded_run.forward,
                    transform=transform,
                    stimulation=stimulation,
                )

                runtime_seconds = (
                    time.perf_counter()
                    - method_start
                )

                row = create_success_row(
                    common=common,
                    method=method,
                    inverse_result=inverse_result,
                    metrics=metrics,
                    runtime_seconds=runtime_seconds,
                )

                print(
                    " PASS "
                    f"({metrics.localization_distance_mm:.2f} mm)"
                )

            except Exception as error:
                runtime_seconds = (
                    time.perf_counter()
                    - method_start
                )

                row = create_failure_row(
                    common=common,
                    method=method,
                    error=error,
                    runtime_seconds=runtime_seconds,
                )

                print(
                    " FAIL "
                    f"({type(error).__name__}: {error})"
                )

                if arguments.verbose:
                    traceback.print_exc()

            rows = remove_previous_result(
                rows=rows,
                subject=subject,
                run=run,
                method=method,
            )

            rows.append(row)

            save_results(
                rows=rows,
                output_file=output_file,
            )

            if "source_estimate" in locals():
                del source_estimate

            if "metrics" in locals():
                del metrics

            if "inverse_result" in locals():
                del inverse_result

            gc.collect()

        del loaded_run
        del referenced_epochs
        del covariance
        del inverse_operator
        del evoked
        del stimulation
        del transform

        gc.collect()

    print_summary(
        rows=rows,
        output_file=output_file,
        expected_methods=set(arguments.methods),
        discovered_runs=set(discovered_runs),
    )


if __name__ == "__main__":
    main()