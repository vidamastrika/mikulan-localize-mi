#!/usr/bin/env python3
"""Run a documented Localize-MI parameter grid with four inverse methods.

The three published methods (MNE, dSPM, eLORETA) and parameter values come
from Mikulan et al. (Scientific Data, 2020, doi:10.1038/s41597-020-0467-x).
sLORETA is our extension. The authors' itcfpy montage channel lists are not
publicly available: the 128/64/32 selections and bad-channel replacements
below are OUR reproducible geometric approximation, not exact author lists.

Start with a cheap baseline check:
    python scripts/09_full_parameter_grid.py --case sub-01 run-01 \
        --montage all_good --loose 1.0 --depth 0.1 --snr 1 --quiet

Then start a fresh, distinct output directory for the full pilot:
    python scripts/09_full_parameter_grid.py --quiet
    python scripts/09_full_parameter_grid.py --quiet --resume

One run at a time, the script iterates over all combinations of montages, methods,
    python scripts/09_full_parameter_grid.py \
        --case sub-01 run-01 \
        --output-dir outputs/grid_sub01_run01 \
        --quiet

All run in one command
    python scripts/09_full_parameter_grid.py \
        --all-runs \
        --output-dir outputs/grid_all_runs \
        --quiet

The default is the four Script 08 cases. Use --all-runs for all 61 runs,
with a DIFFERENT --output-dir. Each row is checkpointed in a CSV; a manifest
locks the parameters and reference geometry so an incompatible run cannot
silently resume it. Do not run two writers against the same output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import mne
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from localize_mi import load_run  # noqa: E402
from localize_mi.inverse import (  # noqa: E402
    apply_average_reference, apply_inverse_method, create_inverse_operator,
    create_target_evoked, estimate_noise_covariance,
)
from localize_mi.metrics import (  # noqa: E402
    calculate_localization_metrics, load_stimulation_info, load_surface_transform,
)

CASES = (("sub-01", "run-01"), ("sub-07", "run-07"),
         ("sub-05", "run-06"), ("sub-07", "run-05"))
METHODS = ("MNE", "dSPM", "eLORETA", "sLORETA")
MONTAGES = ("all_good", "128", "64", "32")
GRID = tuple(round(n / 10, 1) for n in range(1, 11))
TARGET = (-0.002, 0.002)
NOISE = (-0.250, -0.050)
FIELDS = (
    "subject", "run", "method", "method_origin", "montage",
    "montage_definition", "channels", "replacement_count", "loose", "depth",
    "snr", "lambda2", "target_tmin_s", "target_tmax_s", "covariance_tmin_s",
    "covariance_tmax_s", "covariance_method_selected", "stimulation_pair",
    "hemisphere", "peak_vertex", "peak_time_ms", "estimated_x_m",
    "estimated_y_m", "estimated_z_m", "known_x_m", "known_y_m", "known_z_m",
    "localization_distance_mm", "nearest_source_distance_mm",
    "geometric_excess_mm", "script05_baseline_mm", "change_from_baseline_mm",
    "status", "error_type", "error_message", "runtime_seconds",
)


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=PROJECT / "data/Localize-MI")
    p.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/full_parameter_grid")
    p.add_argument("--baseline-results", type=Path,
                   default=PROJECT / "outputs/tables/inverse_method_results.csv")
    p.add_argument("--template-subject", default="sub-01")
    p.add_argument("--template-run", default="run-01")
    p.add_argument("--case", nargs=2, action="append", metavar=("SUBJECT", "RUN"))
    p.add_argument("--all-runs", action="store_true")
    p.add_argument("--montage", choices=MONTAGES, action="append")
    p.add_argument("--method", choices=METHODS, action="append")
    p.add_argument("--loose", type=float, action="append")
    p.add_argument("--depth", type=float, action="append")
    p.add_argument("--snr", type=int, action="append")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Inspect selection without writing results.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def ordered_unique(items):
    return list(dict.fromkeys(items))


def choose_values(raw, allowed, name):
    if raw is None:
        return list(allowed)
    result = ordered_unique(raw)
    if any(x not in allowed for x in result) or any(
        isinstance(x, (int, float)) and not math.isfinite(x) for x in result
    ):
        raise ValueError(f"{name} must be chosen from {allowed}: received {result}")
    return result


def discover_cases(dataset):
    items = []
    for path in sorted((dataset / "derivatives/epochs").glob("sub-*/eeg/*_epochs.npy")):
        name = path.name
        if "_task-seegstim_" not in name:
            continue
        subject, tail = name.split("_task-seegstim_", 1)
        items.append((subject, tail.removesuffix("_epochs.npy")))
    if not items:
        raise FileNotFoundError("No seegstim epoch arrays found under dataset/derivatives/epochs")
    return ordered_unique(items)


def positions_from_forward(forward):
    names = list(forward["info"]["ch_names"])
    xyz = np.array([ch["loc"][:3] for ch in forward["info"]["chs"]], dtype=float)
    if xyz.shape != (len(names), 3) or not np.isfinite(xyz).all():
        raise ValueError("Reference forward model has invalid EEG positions")
    if len(set(names)) != len(names) or np.any(np.linalg.norm(xyz, axis=1) < 0.01):
        raise ValueError("Reference forward model has duplicate names or missing positions")
    return names, xyz


def fixed_spatial_subsets(forward):
    """Greedy farthest-point coverage of all 256 reference forward sensors.

    Anchor at the sensor with greatest Z; ties use lexicographic channel name.
    Each subsequent sensor maximizes distance to its closest selected sensor.
    The first 32, 64, and 128 picks form nested, fixed nominal montages.
    """
    names, xyz = positions_from_forward(forward)
    if len(names) != 256:
        raise ValueError(f"Expected 256 reference EEG sensors; got {len(names)}")
    ranks = sorted(range(len(names)), key=lambda i: names[i])
    start = sorted(ranks, key=lambda i: (-xyz[i, 2], names[i]))[0]
    picked = [start]
    minimum_sq = np.sum((xyz - xyz[start]) ** 2, axis=1)
    minimum_sq[start] = -1.0
    while len(picked) < 128:
        candidate = min(ranks, key=lambda i: (-minimum_sq[i], names[i]))
        picked.append(candidate)
        minimum_sq = np.minimum(minimum_sq, np.sum((xyz - xyz[candidate]) ** 2, axis=1))
        minimum_sq[picked] = -1.0
    return names, xyz, {str(n): [names[i] for i in picked[:n]] for n in (128, 64, 32)}


def replace_bad_nominal(nominal, template_names, template_xyz, good_names):
    """Replace a bad nominal channel with closest unused good channel.

    Good nominal channels are reserved first, so replacements never displace
    a requested healthy channel. Positions come from a fixed 256-channel
    reference, making the selection reproducible across runs.
    """
    available = set(good_names)
    index = {name: i for i, name in enumerate(template_names)}
    if len(available) < len(nominal):
        raise ValueError(f"Only {len(available)} good sensors for {len(nominal)}-channel montage")
    if set(nominal) - index.keys() or available - index.keys():
        raise ValueError("Run EEG channels differ from reference montage channel names")
    reserved = set(nominal) & available
    replacements = []
    selected = []
    for wanted in nominal:
        if wanted in available:
            selected.append(wanted)
            continue
        candidates = available - reserved
        if not candidates:
            raise ValueError(f"No usable replacement for {wanted}")
        old = template_xyz[index[wanted]]
        choice = min(candidates, key=lambda name: (
            float(np.linalg.norm(template_xyz[index[name]] - old)), name))
        distance_mm = float(np.linalg.norm(template_xyz[index[choice]] - old) * 1000)
        selected.append(choice)
        reserved.add(choice)
        replacements.append({"nominal": wanted, "actual": choice, "distance_mm": distance_mm})
    if len(selected) != len(set(selected)) or set(selected) - available:
        raise AssertionError("Reduced montage produced duplicate or bad EEG channels")
    return selected, replacements


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def result_key(row):
    return (row["subject"], row["run"], row["montage"], row["method"],
            round(float(row["loose"]), 1), round(float(row["depth"]), 1), int(row["snr"]))


def baseline_table(path, cases, methods):
    import pandas as pd

    table = pd.read_csv(path)
    required = {"subject", "run", "method", "status", "loose", "depth", "snr",
                "target_tmin_s", "target_tmax_s", "localization_distance_mm"}
    if required - set(table):
        raise ValueError(f"Script 05 table lacks {sorted(required - set(table))}")
    baseline = {}
    for subject, run in cases:
        for method in methods:
            rows = table.loc[(table.subject == subject) & (table.run == run)
                             & (table.method == method) & (table.status == "PASS")
                             & np.isclose(table.loose, 1.) & np.isclose(table.depth, .1)
                             & np.isclose(table.snr, 1.)
                             & np.isclose(table.target_tmin_s, -.002)
                             & np.isclose(table.target_tmax_s, .002)]
            if len(rows) != 1:
                raise ValueError(f"Expected one Script 05 baseline for {subject} {run} {method}; found {len(rows)}")
            baseline[(subject, run, method)] = float(rows.iloc[0].localization_distance_mm)
    return baseline


def make_row(subject, run, montage, method, loose, depth, snr, channels,
             replacements, stimulation, baseline):
    row = dict.fromkeys(FIELDS, "")
    row.update(subject=subject, run=run, montage=montage, method=method,
               method_origin="author" if method != "sLORETA" else "extension",
               montage_definition="all_good" if montage == "all_good" else "fixed_greedy_spatial_v1",
               channels=len(channels), replacement_count=len(replacements),
               loose=loose, depth=depth, snr=snr, lambda2=1. / snr**2,
               target_tmin_s=TARGET[0], target_tmax_s=TARGET[1],
               covariance_tmin_s=NOISE[0], covariance_tmax_s=NOISE[1],
               stimulation_pair=stimulation.pair, hemisphere=stimulation.hemisphere,
               script05_baseline_mm=baseline)
    return row


def append_result(csv_path, row):
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    args = arguments()
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both")
    if args.case and args.all_runs:
        raise ValueError("Choose --case or --all-runs, not both")
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not dataset.is_dir():
        raise NotADirectoryError(dataset)
    cases = (discover_cases(dataset) if args.all_runs else
             ordered_unique([tuple(case) for case in args.case] if args.case else CASES))
    methods = choose_values(args.method, METHODS, "method")
    montages = choose_values(args.montage, MONTAGES, "montage")
    loose_values = choose_values(args.loose, GRID, "loose")
    depth_values = choose_values(args.depth, GRID, "depth")
    snrs = choose_values(args.snr, (1, 2, 3, 4), "snr")
    # Baseline first, even for the full grid: it catches a changed pipeline.
    montages.sort(key=MONTAGES.index)
    loose_values.sort(key=lambda x: (x != 1., x))
    depth_values.sort(key=lambda x: (x != .1, x))
    snrs.sort()
    total = len(cases) * len(montages) * len(methods) * len(loose_values) * len(depth_values) * len(snrs)
    print(f"Selected {len(cases)} runs; {total:,} solutions", flush=True)
    if args.dry_run:
        print(f"Montages: {montages}; methods: {methods}; loose: {loose_values}; "
              f"depth: {depth_values}; SNR: {snrs}")
        return
    baseline_path = args.baseline_results.expanduser().resolve()
    baselines = baseline_table(baseline_path, cases, methods)
    template = load_run(dataset=dataset, subject=args.template_subject, run=args.template_run,
                        task="seegstim")
    template_names, template_xyz, nominal = fixed_spatial_subsets(template.forward)
    template_digest = hashlib.sha256(json.dumps({
        "names": template_names, "coordinates_m": template_xyz.tolist(), "nominal": nominal},
        sort_keys=True).encode("utf-8")).hexdigest()
    configuration = dict(version=1, dataset=str(dataset), baseline_results=str(baseline_path),
                         baseline_file_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                         mne_version=mne.__version__, numpy_version=np.__version__,
                         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         cases=cases, methods=methods, montages=montages, loose=loose_values,
                         depth=depth_values, snr=snrs, target=TARGET, covariance=NOISE,
                         template_subject=args.template_subject, template_run=args.template_run,
                         template_sha256=template_digest)
    manifest_path = output / "grid_manifest.json"
    results_path = output / "grid_results.csv"
    channels_dir = output / "channel_manifests"
    if args.resume:
        if not manifest_path.is_file() or not results_path.is_file():
            raise FileNotFoundError("--resume requires an existing manifest and results CSV")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != json.loads(json.dumps(configuration)):
            raise ValueError("Existing grid manifest differs: choose the original arguments or a new --output-dir")
    elif manifest_path.exists() or results_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists at {output}: choose --resume or --overwrite")
        # Files below are regenerated. Stale run channel manifests get replaced as visited.
    output.mkdir(parents=True, exist_ok=True)
    channels_dir.mkdir(exist_ok=True)
    if not args.resume:
        atomic_json(manifest_path, configuration)
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(FIELDS)
        atomic_json(output / "nominal_channel_sets.json", {
            "description": "Greedy spatial selection on reference forward sensor positions; NOT author channel lists",
            "reference": f"{args.template_subject} {args.template_run}",
            "sha256": template_digest, "sets": nominal,
        })

    done = set()
    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Existing CSV header does not match Script 09")
        for row in reader:
            key = result_key(row)
            if key in done:
                raise ValueError(f"Duplicate result in CSV: {key}")
            done.add(key)
    print(f"Output: {results_path} ({len(done):,} already checkpointed)", flush=True)

    n_fail = 0
    n_new = 0
    for subject, run in cases:
        pending = any((subject, run, montage, method, loose, depth, snr) not in done
                      for montage in montages for method in methods
                      for loose in loose_values for depth in depth_values for snr in snrs)
        if not pending:
            continue
        loaded = load_run(dataset=dataset, subject=subject, run=run, task="seegstim")
        if set(loaded.epochs.ch_names) != set(template_names):
            raise ValueError(f"EEG channel names differ from template for {subject} {run}")
        stimulation = load_stimulation_info(
            loaded.metadata["Description"], dataset / "derivatives/epochs" / subject / "ieeg" /
            f"{subject}_task-seegstim_space-surface_electrodes.tsv")
        transform = load_surface_transform(dataset / "derivatives/sourcemodelling" / subject / "xfm" /
                                           f"{subject}_from-head_to-surface.h5")
        good = [name for name in loaded.epochs.ch_names if name not in loaded.epochs.info["bads"]]
        # Referencing first matches the author's order and keeps Script 05's
        # all-good baseline unchanged. Never run run_inverse on picked epochs:
        # it would apply a second, different average reference.
        referenced = apply_average_reference(loaded.epochs)
        for montage in montages:
            if not any((subject, run, montage, method, loose, depth, snr) not in done
                       for method in methods for loose in loose_values
                       for depth in depth_values for snr in snrs):
                continue
            if montage == "all_good":
                actual, replacements = good, []
                sub_epochs = referenced
            else:
                actual, replacements = replace_bad_nominal(
                    nominal[montage], template_names, template_xyz, good)
                sub_epochs = referenced.copy().pick(actual)
                if sub_epochs.info["bads"]:
                    raise AssertionError("Picked reduced montage still has bad channels")
            run_montage_path = channels_dir / f"{subject}_{run}_{montage}.json"
            channel_data = dict(subject=subject, run=run, montage=montage,
                                nominal=good if montage == "all_good" else nominal[montage],
                                actual=actual, replacements=replacements, reference="all good channels before pick",
                                template_sha256=template_digest)
            if args.resume and run_montage_path.exists():
                if json.loads(run_montage_path.read_text(encoding="utf-8")) != channel_data:
                    raise ValueError(f"Montage changed on resume: {run_montage_path}")
            atomic_json(run_montage_path, channel_data)
            print(f"{subject} {run} {montage}: {len(actual)} channels; "
                  f"{len(replacements)} replacements", flush=True)
            try:
                cov = estimate_noise_covariance(sub_epochs, *NOISE, method="auto",
                                                verbose=not args.quiet)
                evoked = create_target_evoked(sub_epochs, *TARGET)
            except Exception as exc:
                raise RuntimeError(f"Covariance or evoked failed for {subject} {run} {montage}") from exc

            for loose in loose_values:
                for depth in depth_values:
                    remaining = [(method, snr) for method in methods for snr in snrs
                                 if (subject, run, montage, method, loose, depth, snr) not in done]
                    if not remaining:
                        continue
                    inverse_error = None
                    try:
                        inverse = create_inverse_operator(sub_epochs, loaded.forward, cov,
                                                          loose=loose, depth=depth,
                                                          verbose=not args.quiet)
                    except (ValueError, RuntimeError, OSError, TypeError, MemoryError) as exc:
                        inverse_error = exc
                    for method, snr in remaining:
                        started = time.monotonic()
                        row = make_row(subject, run, montage, method, loose, depth, snr,
                                       actual, replacements, stimulation,
                                       baselines[(subject, run, method)])
                        try:
                            if inverse_error is not None:
                                raise RuntimeError(f"inverse operator: {inverse_error}")
                            stc, _, _ = apply_inverse_method(evoked, inverse, method, snr,
                                                              verbose=not args.quiet)
                            metric = calculate_localization_metrics(stc, loaded.forward,
                                                                    transform, stimulation)
                            distance = float(metric.localization_distance_mm)
                            baseline = baselines[(subject, run, method)]
                            if montage == "all_good" and loose == 1. and depth == .1 and snr == 1:
                                if abs(distance - baseline) > .05:
                                    raise ValueError(f"Script 05 baseline mismatch: {distance:.3f} "
                                                     f"versus {baseline:.3f} mm")
                            row.update(covariance_method_selected=str(cov.get("method", "auto")),
                                       peak_vertex=int(metric.peak_vertex),
                                       peak_time_ms=float(metric.peak_time * 1000),
                                       estimated_x_m=float(metric.peak_coordinate[0]),
                                       estimated_y_m=float(metric.peak_coordinate[1]),
                                       estimated_z_m=float(metric.peak_coordinate[2]),
                                       known_x_m=float(metric.stimulation_midpoint[0]),
                                       known_y_m=float(metric.stimulation_midpoint[1]),
                                       known_z_m=float(metric.stimulation_midpoint[2]),
                                       localization_distance_mm=distance,
                                       nearest_source_distance_mm=float(metric.nearest_source_distance_mm),
                                       geometric_excess_mm=distance-float(metric.nearest_source_distance_mm),
                                       change_from_baseline_mm=distance-baseline, status="PASS")
                        except (ValueError, RuntimeError, OSError, TypeError, KeyError, MemoryError) as exc:
                            row.update(status="FAIL", error_type=type(exc).__name__, error_message=str(exc))
                            n_fail += 1
                            print(f"FAIL {subject} {run} {montage} {method} "
                                  f"l={loose} d={depth} s={snr}: {exc}", flush=True)
                        row["runtime_seconds"] = round(time.monotonic()-started, 3)
                        append_result(results_path, row)
                        done.add((subject, run, montage, method, loose, depth, snr))
                        n_new += 1
                        if row["status"] != "PASS" and montage == "all_good" and loose == 1. and depth == .1 and snr == 1:
                            raise RuntimeError("Script 05 baseline check failed. Review the CSV error before continuing.")
                    if n_new % 100 < len(remaining):
                        print(f"Checkpoint: {len(done):,}/{total:,}; failures this invocation: {n_fail}", flush=True)
    print(f"Finished: {len(done):,}/{total:,} rows, {n_new:,} new, {n_fail} new failures; {results_path}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
