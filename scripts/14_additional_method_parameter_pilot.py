#!/usr/bin/env python3
"""One-factor-at-a-time parameter pilot for the additional inverse methods.

This script follows the fixed-configuration all-run analysis in Scripts 12–13.
It changes one parameter at a time around the Script 12 baseline so that each
observed localization change has a clear interpretation.  It is intentionally
not the full interaction grid; that belongs in Script 15.

Default pilot
-------------
* four representative runs used by Scripts 08 and 11
* all good EEG channels
* target window: -2 to +2 ms
* ECD-grid: one fixed reference configuration
* LCMV: reg = 0.01, 0.05, 0.10 and covariance half-window = 2, 5, 10 ms
* MxNE/irMxNE: alpha = 20, 40, 60; loose = 0.1, 0.5, 1.0;
  depth = 0.1, 0.5, 1.0
* irMxNE only: iterations = 5, 10, 20

Only one parameter differs from the baseline in each non-baseline solution.
This produces 22 configurations per run and 88 solutions for the four default
runs.  Results are atomically checkpointed after every solution.

Examples
--------
Run a one-case validation::

    python scripts/14_additional_method_parameter_pilot.py \
        --case sub-01 run-01 --overwrite --quiet

Run the complete four-run pilot::

    python scripts/14_additional_method_parameter_pilot.py --overwrite --quiet

Resume an interrupted pilot::

    python scripts/14_additional_method_parameter_pilot.py --resume --quiet

Default output
--------------
outputs/additional_method_parameter_pilot/pilot_results.csv
outputs/additional_method_parameter_pilot/pilot_manifest.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT11 = PROJECT / "scripts/11_additional_method_pilot.py"
DEFAULT_OUTPUT = (
    PROJECT / "outputs/additional_method_parameter_pilot/pilot_results.csv"
)
DEFAULT_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)
METHODS = ("ECD-grid", "LCMV", "MxNE", "irMxNE")

# The fixed configuration already validated in Scripts 11–13.
BASELINE = {
    "loose": 1.0,
    "depth": 0.1,
    "lcmv_reg": 0.05,
    "lcmv_data_tmin": -0.005,
    "lcmv_data_tmax": 0.005,
    "lcmv_covariance_method": "shrunk",
    "mxne_alpha": 40.0,
    "irmxne_iterations": 10,
}

EXTRA_FIELDS = (
    "configuration_id", "is_baseline", "varied_parameter",
    "varied_value", "lcmv_covariance_half_window_ms",
)


def load_script11():
    """Import the finalized implementation rather than duplicate algorithms."""
    if not SCRIPT11.is_file():
        raise FileNotFoundError(
            f"Finalized Script 11 was not found: {SCRIPT11}"
        )
    specification = importlib.util.spec_from_file_location(
        "localize_mi_script11_for_parameter_pilot", SCRIPT11
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load Script 11: {SCRIPT11}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


pilot = load_script11()
FIELDNAMES = EXTRA_FIELDS + tuple(pilot.FIELDNAMES)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT / "data/Localize-MI",
        help="Localize-MI BIDS root.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest", type=Path,
        help="Default: pilot_manifest.json beside the output CSV.",
    )
    parser.add_argument(
        "--baseline-results", type=Path,
        default=PROJECT / "outputs/tables/inverse_method_results.csv",
        help="Script 05 results used only for descriptive comparison.",
    )
    parser.add_argument(
        "--case", nargs=2, metavar=("SUBJECT", "RUN"), action="append",
        help="Repeat to replace the four default representative runs.",
    )
    parser.add_argument(
        "--method", type=pilot.normalize_method, action="append",
        help="Repeat to restrict the pilot to selected methods.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--fail-on-error", action="store_true",
        help=("Return a non-zero exit status when any tested parameter "
              "combination fails. By default failures are recorded because "
              "they are valid pilot outcomes."),
    )
    return parser.parse_args()


def validate_args(args):
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")


def configuration(method, identifier, varied_parameter="baseline", **updates):
    values = dict(BASELINE)
    values.update(updates)
    if varied_parameter == "baseline":
        varied_value = ""
    elif varied_parameter == "lcmv_covariance_half_window_ms":
        varied_value = abs(float(values["lcmv_data_tmax"])) * 1000
    else:
        varied_value = values[varied_parameter]
    return {
        "method": method,
        "configuration_id": identifier,
        "is_baseline": varied_parameter == "baseline",
        "varied_parameter": varied_parameter,
        "varied_value": varied_value,
        **values,
    }


def build_configurations(methods):
    """Create unique OFAT configurations around the fixed baseline."""
    configs = []
    if "ECD-grid" in methods:
        configs.append(configuration("ECD-grid", "ecd_fixed"))

    if "LCMV" in methods:
        configs.append(configuration("LCMV", "lcmv_baseline"))
        for value in (0.01, 0.10):
            configs.append(configuration(
                "LCMV", f"lcmv_reg_{value:g}", "lcmv_reg",
                lcmv_reg=value,
            ))
        for half_window_ms in (2, 10):
            half_window_s = half_window_ms / 1000
            configs.append(configuration(
                "LCMV", f"lcmv_cov_pm{half_window_ms}ms",
                "lcmv_covariance_half_window_ms",
                lcmv_data_tmin=-half_window_s,
                lcmv_data_tmax=half_window_s,
            ))

    for method in ("MxNE", "irMxNE"):
        if method not in methods:
            continue
        prefix = method.lower()
        configs.append(configuration(method, f"{prefix}_baseline"))
        for value in (20.0, 60.0):
            configs.append(configuration(
                method, f"{prefix}_alpha_{value:g}", "mxne_alpha",
                mxne_alpha=value,
            ))
        for value in (0.1, 0.5):
            configs.append(configuration(
                method, f"{prefix}_loose_{value:g}", "loose", loose=value,
            ))
        for value in (0.5, 1.0):
            configs.append(configuration(
                method, f"{prefix}_depth_{value:g}", "depth", depth=value,
            ))
        if method == "irMxNE":
            for value in (5, 20):
                configs.append(configuration(
                    method, f"irmxne_iterations_{value}",
                    "irmxne_iterations", irmxne_iterations=value,
                ))

    identifiers = [item["configuration_id"] for item in configs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Internal error: duplicate configuration IDs.")
    return configs


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    temporary.replace(path)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def solution_key(row):
    return str(row["subject"]), str(row["run"]), str(row["configuration_id"])


def load_existing(path, resume, overwrite, expected_keys):
    if path.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"Output exists: {path}. Use --resume or --overwrite."
        )
    if overwrite or not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDNAMES):
            raise ValueError(
                "Existing CSV columns do not match Script 14; use a new output."
            )
        rows = list(reader)
    keys = [solution_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Existing CSV contains duplicate solutions.")
    unexpected = set(keys) - set(expected_keys)
    if unexpected:
        raise ValueError(
            "Existing CSV contains solutions outside the current selection; "
            "use a different --output path."
        )
    # Preserve successful checkpoints; failed configurations are retried.
    return [row for row in rows if row["status"] == "PASS"]


def same_number(left, right):
    try:
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left) == str(right)


def manifest_payload(cases, configs, output, rows):
    return {
        "script": "14_additional_method_parameter_pilot.py",
        "design": "one-factor-at-a-time",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_csv": str(output),
        "cases": [list(case) for case in cases],
        "baseline": BASELINE,
        "configurations": configs,
        "solutions_expected": len(cases) * len(configs),
        "rows_written": len(rows),
        "passes": sum(str(row.get("status")) == "PASS" for row in rows),
        "failures": sum(str(row.get("status")) != "PASS" for row in rows),
    }


def base_row(subject, run, config, loaded, stimulation, noise_covariance,
             references):
    row = dict.fromkeys(FIELDNAMES, "")
    channel_count = len(loaded.epochs.ch_names) - len(loaded.epochs.info["bads"])
    row.update(
        configuration_id=config["configuration_id"],
        is_baseline=config["is_baseline"],
        varied_parameter=config["varied_parameter"],
        varied_value=config["varied_value"],
        subject=subject,
        run=run,
        method=config["method"],
        montage="all_good",
        n_good_channels=channel_count,
        epochs=len(loaded.epochs),
        sampling_frequency_hz=float(loaded.epochs.info["sfreq"]),
        evaluated_sources=pilot.count_forward_sources(loaded.forward),
        target_tmin_s=pilot.TARGET_TMIN,
        target_tmax_s=pilot.TARGET_TMAX,
        covariance_tmin_s=pilot.COVARIANCE_TMIN,
        covariance_tmax_s=pilot.COVARIANCE_TMAX,
        covariance_method_selected=str(noise_covariance.get("method", "auto")),
        stimulation_pair=stimulation.pair,
        hemisphere=stimulation.hemisphere,
    )
    if config["method"] == "LCMV":
        row["lcmv_covariance_half_window_ms"] = (
            float(config["lcmv_data_tmax"]) * 1000
        )
    if (subject, run) in references:
        name, distance = references[(subject, run)]
        row["reference_method"] = name
        row["reference_distance_mm"] = distance
    return row


def main():
    args = parse_args()
    validate_args(args)
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve() if args.manifest
        else output.parent / "pilot_manifest.json"
    )
    if not dataset.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset}")

    cases = list(dict.fromkeys(tuple(case) for case in (args.case or DEFAULT_CASES)))
    methods = list(dict.fromkeys(args.method or METHODS))
    configs = build_configurations(methods)
    expected_keys = [
        (subject, run, config["configuration_id"])
        for subject, run in cases for config in configs
    ]
    rows = load_existing(
        output, args.resume, args.overwrite, expected_keys
    )
    completed = {solution_key(row) for row in rows}
    references = pilot.load_references(
        args.baseline_results.expanduser().resolve(), cases
    )

    print("LOCALIZE-MI ADDITIONAL-METHOD PARAMETER PILOT")
    print("---------------------------------------------")
    print(f"Design          : one factor at a time")
    print(f"Runs            : {len(cases)}")
    print(f"Configurations  : {len(configs)} per run")
    print(f"Solutions       : {len(expected_keys)}")
    print(f"Already complete: {len(rows)}")
    print(f"Methods         : {', '.join(methods)}")
    print(f"Output          : {output}", flush=True)

    for case_index, (subject, run) in enumerate(cases, start=1):
        pending = [
            config for config in configs
            if (subject, run, config["configuration_id"]) not in completed
        ]
        if not pending:
            continue
        print(f"[{case_index:02d}/{len(cases):02d}] {subject} {run}", flush=True)

        loaded = pilot.load_run(
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
        stimulation = pilot.load_stimulation_info(
            loaded.metadata["Description"], electrodes
        )
        transform = pilot.load_surface_transform(transform_file)
        referenced, noise_covariance, evoked = pilot.common_preprocessing(
            loaded.epochs, args.quiet
        )
        covariance_cache = {}

        for config in pending:
            started = time.monotonic()
            method = config["method"]
            row = base_row(
                subject, run, config, loaded, stimulation,
                noise_covariance, references,
            )
            try:
                residual = None
                if method == "ECD-grid":
                    source_estimate, diagnostics = pilot.run_ecd_grid(
                        evoked, loaded.forward, noise_covariance,
                        stimulation.hemisphere,
                    )
                    row.update(
                        active_sources=1,
                        ecd_selection_criterion="maximum whitened GOF",
                        ecd_goodness_of_fit_percent=diagnostics["gof_percent"],
                    )
                elif method == "LCMV":
                    covariance_key = (
                        config["lcmv_covariance_method"],
                        config["lcmv_data_tmin"], config["lcmv_data_tmax"],
                    )
                    if covariance_key not in covariance_cache:
                        covariance_cache[covariance_key] = (
                            pilot.compute_data_covariance(
                                referenced, config["lcmv_covariance_method"],
                                config["lcmv_data_tmin"],
                                config["lcmv_data_tmax"], args.quiet,
                            )
                        )
                    data_covariance = covariance_cache[covariance_key]
                    source_estimate = pilot.run_lcmv(
                        evoked, loaded.forward, noise_covariance,
                        data_covariance, config["lcmv_reg"], args.quiet,
                    )
                    row.update(
                        data_covariance_method=config["lcmv_covariance_method"],
                        lcmv_data_tmin_s=config["lcmv_data_tmin"],
                        lcmv_data_tmax_s=config["lcmv_data_tmax"],
                        lcmv_data_covariance_samples=data_covariance.get(
                            "nfree", ""
                        ),
                        lcmv_reg=config["lcmv_reg"],
                        lcmv_pick_ori="max-power",
                        lcmv_weight_norm="unit-noise-gain-invariant",
                    )
                else:
                    iterations = (
                        1 if method == "MxNE"
                        else int(config["irmxne_iterations"])
                    )
                    source_estimate, residual = pilot.run_mxne(
                        evoked, loaded.forward, noise_covariance,
                        config["mxne_alpha"], config["loose"],
                        config["depth"], iterations, args.quiet,
                    )
                    active = pilot.count_active_sources(source_estimate)
                    row.update(
                        loose=config["loose"],
                        depth=config["depth"],
                        mxne_alpha=config["mxne_alpha"],
                        mxne_iterations=iterations,
                        active_sources=active,
                        explained_variance_percent=pilot.explained_variance(
                            evoked, residual
                        ),
                    )
                    if active == 0:
                        raise RuntimeError(
                            f"{method} returned no active cortical sources."
                        )

                metric = pilot.calculate_localization_metrics(
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
            except (
                OSError, ValueError, RuntimeError, TypeError, KeyError,
                IndexError, np.linalg.LinAlgError, MemoryError,
            ) as exc:
                row.update(
                    status="FAIL",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            row["runtime_seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            atomic_csv(output, rows)
            atomic_json(
                manifest,
                manifest_payload(cases, configs, output, rows),
            )
            if row["status"] == "PASS":
                print(
                    f"  {config['configuration_id']:<28} "
                    f"{float(row['localization_distance_mm']):7.2f} mm",
                    flush=True,
                )
            else:
                print(
                    f"  {config['configuration_id']:<28} FAIL: "
                    f"{row['error_message']}", flush=True,
                )

    failures = sum(str(row.get("status")) != "PASS" for row in rows)
    print("\nPARAMETER PILOT COMPLETE")
    print("------------------------")
    print(f"Rows      : {len(rows)}")
    print(f"Passes    : {len(rows) - failures}")
    print(f"Failures  : {failures}")
    print(f"CSV       : {output}")
    print(f"Manifest  : {manifest}")
    if failures and args.fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
