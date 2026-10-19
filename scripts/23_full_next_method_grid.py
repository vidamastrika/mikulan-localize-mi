#!/usr/bin/env python3
"""
23_full_next_method_grid.py

Full Localize-MI parameter grid for irMxNE, Gamma-MAP, SESAME, and Champagne.
This is the full-grid counterpart to Script 22 and follows the execution
philosophy of Script 15: run ONE METHOD AT A TIME, checkpoint every solution,
retain failures, and analyze results in a later script.

===============================================================================
COMPLETE SCRIPT-23 PARAMETER DESIGN
===============================================================================

SHARED
------
Runs:
    all 61 released Localize-MI runs when --all-runs is supplied

Montages:
    all_good, 128, 64, 32

Reduced montages:
    Same fixed greedy spatial channel selection and bad-channel replacement
    procedure as Scripts 09 and 15.

Reference:
    Average reference across all good EEG channels BEFORE reduced-montage pick.

Target window:
    -2 to +2 ms

Noise covariance for irMxNE, Gamma-MAP, Champagne:
    -250 to -50 ms, method="auto"

SESAME:
    No prestimulus covariance is supplied to SESAME.
    Same finalized Script-20 operationalization:
        noise_std = fs * max(abs(target Evoked data))

Evaluation:
    Euclidean localization error to the known stimulation midpoint.

1) irMxNE
--------
EXPERIMENTAL:
    alpha = 25, 30, 35
    loose = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    depth = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

FIXED:
    n_mxne_iter = 10
    random_state = 0

Grid:
    3 x 10 x 10 = 300 method configurations per montage
    300 x 4 x 61 = 73,200 solutions

Note:
    loose and depth deliberately use the SAME 0.1 increments as the
    Script-15 full MxNE grid for direct comparability.

Note:
    alpha=60 is NOT included. Script 15's full MxNE experiment used
    alpha=30,40,50, so Script 23 retains that range for comparability.

2) Gamma-MAP
------------
EXPERIMENTAL:
    alpha = 0.1, 0.2, 0.3, 0.4, 0.5
    loose = 0.4, 0.7, 1.0
    xyz_same_gamma = True, False

FIXED:
    depth = 0.1
    update_mode = 1
    maxit = 10000
    tol = 1e-6

Grid:
    5 x 3 x 2 = 30 method configurations per montage
    30 x 4 x 61 = 7,320 solutions

Note:
    depth is fixed because Script 22 gave identical localization results for
    depth=0.1,0.4,0.7 on the pilot runs.

3) SESAME
---------
EXPERIMENTAL:
    fs = 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33

    step = 0.03

FIXED:
    n_parts = 100
    lam = 0.25
    max_n_dips = 1
    hyper_q = True
    random_seed = 0
    dip_mom_std = estimated by SESAME
    noise covariance supplied = False

Grid:
    10 method configurations per montage
    10 x 4 x 61 = 2,440 solutions

4) Champagne
------------
EXPERIMENTAL:
    pruning_thresh = 0.0001, 0.001, 0.01

FIXED:
    alpha = "auto"
    prune = True
    convergence_criterion = 1e-8
    max_iter = 2000

Grid:
    3 method configurations per montage
    3 x 4 x 61 = 732 solutions

Note:
    prune=False was screened in Script 22 but was extremely computationally
    expensive. Script 23 therefore keeps the package-default prune=True and
    studies sensitivity to the pruning threshold instead.

    convergence criterion is fixed because Script 22 gave identical results
    for 1e-6, 1e-8, and 1e-10.

TOTAL DEFAULT FULL GRID
-----------------------
    irMxNE     = 73,200
    Gamma-MAP  =  7,320
    SESAME     =  2,440
    Champagne  =    732
    TOTAL      = 83,692 solutions

===============================================================================
TERMINAL EXAMPLES
===============================================================================

1) Dry-run / inspect counts

    python scripts/23_full_next_method_grid.py \
        --method irmxne --all-runs --dry-run

    python scripts/23_full_next_method_grid.py \
        --method gamma_map --all-runs --dry-run

    python scripts/23_full_next_method_grid.py \
        --method sesame --all-runs --dry-run

    python scripts/23_full_next_method_grid.py \
        --method champagne --all-runs --dry-run

2) Validate finalized baseline on one run

irMxNE:
    python scripts/23_full_next_method_grid.py \
        --method irmxne \
        --case sub-01 run-01 \
        --montage all_good \
        --irmxne-alpha 40 \
        --irmxne-loose 1.0 \
        --irmxne-depth 0.1 \
        --overwrite

Gamma-MAP:
    python scripts/23_full_next_method_grid.py \
        --method gamma_map \
        --case sub-01 run-01 \
        --montage all_good \
        --gamma-alpha 0.3 \
        --gamma-loose 1.0 \
        --gamma-xyz-same-gamma true \
        --overwrite

SESAME:
    python scripts/23_full_next_method_grid.py \
        --method sesame \
        --case sub-01 run-01 \
        --montage all_good \
        --sesame-fs 0.15 \
        --overwrite

Champagne:
    python scripts/23_full_next_method_grid.py \
        --method champagne \
        --case sub-01 run-01 \
        --montage all_good \
        --champagne-pruning-thresh 0.001 \
        --overwrite

3) Run complete grids, one method at a time

    python scripts/23_full_next_method_grid.py \
        --method sesame --all-runs --overwrite

    python scripts/23_full_next_method_grid.py \
        --method champagne --all-runs --overwrite

    python scripts/23_full_next_method_grid.py \
        --method irmxne --all-runs --overwrite

    python scripts/23_full_next_method_grid.py \
        --method gamma_map --all-runs --overwrite

4) Resume

    python scripts/23_full_next_method_grid.py \
        --method gamma_map --all-runs --resume

5) Restrict values for debugging

Champagne:
    python scripts/23_full_next_method_grid.py \
        --method champagne \
        --case sub-01 run-01 \
        --montage all_good \
        --champagne-pruning-thresh 0.0001 0.001 0.01 \
        --overwrite --quiet

irMxNE:
    python scripts/23_full_next_method_grid.py \
        --method irmxne \
        --case sub-01 run-01 \
        --montage all_good \
        --irmxne-alpha 30 40 50 \
        --irmxne-loose 0.8 0.9 1.0 \
        --irmxne-depth 0.1 0.2 0.3 \
        --overwrite --quiet

Gamma-MAP:
    python scripts/23_full_next_method_grid.py \
        --method gamma_map \
        --case sub-01 run-01 \
        --montage all_good \
        --gamma-alpha 0.3 0.5 \
        --gamma-loose 0.7 1.0 \
        --gamma-xyz-same-gamma true false \
        --overwrite --quiet

OUTPUT
------
    outputs/23_full_next_method_grid/<method>/
        grid_results.csv
        grid_manifest.json
        nominal_channel_sets.json
        channel_manifests/

No large analysis figures are generated here. Script 23 is computation-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

import mne
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT09 = PROJECT / "scripts" / "09_full_parameter_grid.py"
SCRIPT19 = PROJECT / "scripts" / "19_next_method_pilot.py"
DEFAULT_OUTPUT_ROOT = PROJECT / "outputs" / "23_full_next_method_grid"

DEFAULT_PILOT_CASES = (
    ("sub-01", "run-01"),
    ("sub-07", "run-07"),
    ("sub-05", "run-06"),
    ("sub-07", "run-05"),
)
MONTAGES = ("all_good", "128", "64", "32")

IRMXNE_ALPHAS = (15.0,20.0, 25.0, 30.0, 35.0)
IRMXNE_LOOSE = tuple(round(i / 10, 1) for i in range(1, 11))
IRMXNE_DEPTH = tuple(round(i / 10, 1) for i in range(1, 11))
IRMXNE_ITERATIONS = 10

GAMMA_ALPHAS = (0.20, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.7)
GAMMA_LOOSE = (0.3, 0.5, 0.7)
GAMMA_XYZ = (True, False)
GAMMA_DEPTH = (0.5,)

SESAME_FS = (
    0.03, 0.06, 0.09,
    0.12, 0.15,
    0.18, 0.21, 0.24, 0.27, 0.30, 0.33,
)
SESAME_N_PARTS = 100
SESAME_MAX_N_DIPS = 1
SESAME_LAM = 0.25
SESAME_HYPER_Q = True
SESAME_RANDOM_SEED = 0

CHAMPAGNE_PRUNING_THRESHOLDS = (0.0001, 0.001, 0.005, 0.01, 0.05)
CHAMPAGNE_ALPHA = "auto"
CHAMPAGNE_PRUNE = True
CHAMPAGNE_CONVERGENCE = 1e-8
CHAMPAGNE_MAX_ITER = 2000

TARGET = (-0.002, 0.002)
NOISE = (-0.250, -0.050)

METHOD_SLUGS = {
    "irMxNE": "irmxne",
    "Gamma-MAP": "gamma_map",
    "SESAME": "sesame",
    "Champagne": "champagne",
}

FIELDS = (
    "subject", "run", "method", "configuration_id",
    "montage", "montage_definition", "channels", "replacement_count",
    "epochs", "sampling_frequency_hz",
    "covariance_method", "covariance_tmin_s", "covariance_tmax_s",
    "target_tmin_s", "target_tmax_s", "target_samples",
    "stimulation_pair", "hemisphere",
    "ground_truth_x_mm", "ground_truth_y_mm", "ground_truth_z_mm",
    "irmxne_alpha", "irmxne_loose", "irmxne_depth", "irmxne_iterations",
    "irmxne_random_state", "irmxne_active_sources",
    "irmxne_explained_variance_percent",
    "gamma_alpha", "gamma_loose", "gamma_depth", "gamma_xyz_same_gamma",
    "gamma_update_mode", "gamma_maxit", "gamma_tol", "gamma_active_sources",
    "gamma_explained_variance_percent",
    "sesame_fs", "sesame_n_parts", "sesame_max_n_dips", "sesame_lam",
    "sesame_hyper_q", "sesame_random_seed", "sesame_noise_std",
    "sesame_target_peak", "sesame_noise_covariance_supplied",
    "sesame_estimated_n_dips", "sesame_gof_percent",
    "sesame_source_dispersion_mm",
    "champagne_alpha", "champagne_prune", "champagne_convergence_criterion",
    "champagne_max_iter", "champagne_pruning_thresh",
    "champagne_noise_covariance_supplied", "champagne_make_operator_signature",
    "selection_criterion", "selected_time_ms",
    "selected_x_mm", "selected_y_mm", "selected_z_mm",
    "localization_distance_mm", "peak_vertex",
    "nearest_source_distance_mm", "geometric_excess_mm",
    "status", "error_type", "error_message", "runtime_seconds",
)


def import_script(path, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


grid09 = import_script(SCRIPT09, "localize_mi_script09_for_script23")
pilot = import_script(SCRIPT19, "localize_mi_script19_for_script23")


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_method(value):
    key = str(value).strip().replace("-", "").replace("_", "").lower()
    mapping = {
        "irmxne": "irMxNE",
        "gammamap": "Gamma-MAP",
        "gamma": "Gamma-MAP",
        "sesame": "SESAME",
        "champagne": "Champagne",
        "champ": "Champagne",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            "Choose irmxne, gamma_map, sesame, or champagne."
        )
    return mapping[key]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=PROJECT / "data" / "Localize-MI")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--method", required=True, type=parse_method)
    p.add_argument("--all-runs", action="store_true")
    p.add_argument("--case", nargs=2, metavar=("SUBJECT", "RUN"), action="append")
    p.add_argument("--montage", action="append", choices=MONTAGES)
    p.add_argument("--irmxne-alpha", nargs="+", type=float)
    p.add_argument("--irmxne-loose", nargs="+", type=float)
    p.add_argument("--irmxne-depth", nargs="+", type=float)
    p.add_argument("--gamma-alpha", nargs="+", type=float)
    p.add_argument("--gamma-loose", nargs="+", type=float)
    p.add_argument("--gamma-xyz-same-gamma", nargs="+", type=parse_bool)
    p.add_argument("--sesame-fs", nargs="+", type=float)
    p.add_argument("--champagne-pruning-thresh", nargs="+", type=float)
    p.add_argument("--template-subject", default="sub-01")
    p.add_argument("--template-run", default="run-01")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def ordered_unique(values):
    return list(dict.fromkeys(values))


def choose_numeric(raw, allowed, name):
    if raw is None:
        return list(allowed)
    result = ordered_unique(float(v) for v in raw)
    canonical = []
    for value in result:
        if not math.isfinite(value):
            raise ValueError(f"{name}: non-finite value {value}")
        matches = [a for a in allowed if np.isclose(value, a)]
        if not matches:
            raise ValueError(f"{name}={value} outside planned grid {allowed}")
        canonical.append(float(matches[0]))
    return ordered_unique(canonical)


def choose_bool(raw, allowed, name):
    if raw is None:
        return list(allowed)
    result = ordered_unique(bool(v) for v in raw)
    if any(v not in allowed for v in result):
        raise ValueError(f"{name} outside {allowed}")
    return result


def discover_cases(dataset):
    root = dataset / "derivatives" / "epochs"
    pattern = re.compile(r"^(sub-\d+)_task-seegstim_(run-\d+)_epochs\.npy$")
    cases = set()
    for path in root.glob("sub-*/eeg/*_epochs.npy"):
        match = pattern.match(path.name)
        if match:
            cases.add(match.groups())
    natural = lambda s: tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s))
    return sorted(cases, key=lambda x: (natural(x[0]), natural(x[1])))


def configurations(method, args):
    if method == "irMxNE":
        alphas = choose_numeric(args.irmxne_alpha, IRMXNE_ALPHAS, "irMxNE alpha")
        loose = choose_numeric(args.irmxne_loose, IRMXNE_LOOSE, "irMxNE loose")
        depth = choose_numeric(args.irmxne_depth, IRMXNE_DEPTH, "irMxNE depth")
        return [
            dict(configuration_id=f"irmxne_a{a:g}_l{l:.1f}_d{d:.1f}_i10",
                 irmxne_alpha=a, irmxne_loose=l, irmxne_depth=d,
                 irmxne_iterations=IRMXNE_ITERATIONS)
            for a in alphas for l in loose for d in depth
        ]
    if method == "Gamma-MAP":
        alphas = choose_numeric(args.gamma_alpha, GAMMA_ALPHAS, "Gamma alpha")
        loose = choose_numeric(args.gamma_loose, GAMMA_LOOSE, "Gamma loose")
        xyz = choose_bool(args.gamma_xyz_same_gamma, GAMMA_XYZ, "xyz_same_gamma")
        depths = GAMMA_DEPTH

        return [
            dict(
                configuration_id=f"gamma_a{a:.2f}_l{l:.1f}_d{d:.1f}_xyz{int(x)}",
                gamma_alpha=a,
                gamma_loose=l,
                gamma_depth=d,
                gamma_xyz_same_gamma=x,
            )
            for a in alphas
            for l in loose
            for d in depths
            for x in xyz
        ]
    if method == "SESAME":
        fs_values = choose_numeric(args.sesame_fs, SESAME_FS, "SESAME fs")
        return [dict(configuration_id=f"sesame_fs_{fs:.2f}".replace(".", "p"), sesame_fs=fs)
                for fs in fs_values]
    if method == "Champagne":
        values = choose_numeric(
            args.champagne_pruning_thresh,
            CHAMPAGNE_PRUNING_THRESHOLDS,
            "Champagne pruning_thresh",
        )
        return [
            dict(
                configuration_id=f"champagne_thresh_{v:g}",
                champagne_pruning_thresh=v,
            )
            for v in values
        ]
    raise ValueError(method)


def result_key(subject, run, montage, configuration_id):
    return str(subject), str(run), str(montage), str(configuration_id)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def rewrite_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def append_row(path, row):
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def ensure_projector_compatibility():
    if hasattr(mne, "make_projector"):
        return
    from mne._fiff.proj import make_projector as internal_make_projector
    def wrapper(*args, **kwargs):
        kwargs.pop("verbose", None)
        return internal_make_projector(*args, **kwargs)
    mne.make_projector = wrapper


def count_active_sources(stc):
    return int(sum(len(v) for v in stc.vertices))


def run_gamma(evoked, forward, cov, cfg, quiet):
    return mne.inverse_sparse.gamma_map(
        evoked, forward, cov,
        alpha=float(cfg["gamma_alpha"]),
        loose=float(cfg["gamma_loose"]),
        depth=float(cfg["gamma_depth"]),
        xyz_same_gamma=bool(cfg["gamma_xyz_same_gamma"]),
        maxit=2000, tol=1e-4,
        update_mode=1,
        return_residual=True, return_as_dipoles=False, rank=None,
        verbose=not quiet,
    )


def run_sesame(evoked, forward, fs, quiet):
    prepare_sesame = pilot.require_sesame()
    peak = float(np.max(np.abs(evoked.data)))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError(f"Invalid target EEG peak: {peak}")
    noise_std = float(fs) * peak
    np.random.seed(SESAME_RANDOM_SEED)
    sesame = prepare_sesame(
        forward, evoked,
        n_parts=SESAME_N_PARTS,
        noise_std=noise_std,
        top_min=float(evoked.times[0]), top_max=float(evoked.times[-1]),
        dip_mom_std=None, hyper_q=SESAME_HYPER_Q, noise_cov=None,
        subsample=None, max_n_dips=SESAME_MAX_N_DIPS, lam=SESAME_LAM,
        verbose=not quiet,
    )
    import sesameeg.sesame as sesame_module
    original_particle = sesame_module.Particle
    def compatible_particle(n_verts_value, lam_value, dip_mom_std=None,
                            prior_locs=None, fixed_ori=None, hyper_q=None):
        if prior_locs is None: prior_locs = sesame.prior_locs
        if fixed_ori is None: fixed_ori = sesame.fixed_ori
        if hyper_q is None: hyper_q = sesame.hyper_q
        return original_particle(n_verts_value, lam_value,
                                 dip_mom_std=dip_mom_std,
                                 prior_locs=prior_locs,
                                 fixed_ori=fixed_ori,
                                 hyper_q=hyper_q)
    sesame_module.Particle = compatible_particle
    try:
        sesame.apply_sesame()
    finally:
        sesame_module.Particle = original_particle
    if not sesame.est_n_dips or int(sesame.est_n_dips[-1]) < 1:
        raise RuntimeError("SESAME selected a zero-source model.")
    return sesame, sesame.compute_stc(), noise_std, peak


def run_champagne(evoked, forward, cov, pruning_thresh, quiet):
    ensure_projector_compatibility()
    forward = mne.pick_channels_forward(
        forward, include=list(evoked.ch_names), exclude=[], ordered=True,
        copy=True, verbose=not quiet,
    )
    Solver = pilot.require_champagne_solver()
    solver = Solver("champagne")
    make_operator = solver.make_inverse_operator
    signature = inspect.signature(make_operator)
    params = signature.parameters
    requested = dict(
        alpha=CHAMPAGNE_ALPHA,
        prune=CHAMPAGNE_PRUNE,
        convergence_criterion=CHAMPAGNE_CONVERGENCE,
        max_iter=CHAMPAGNE_MAX_ITER,
        pruning_thresh=float(pruning_thresh),
    )
    kwargs = {k: v for k, v in requested.items() if k in params}
    if "prune" not in params:
        raise RuntimeError(f"Installed Champagne API does not expose prune: {signature}")
    cov_supplied = False
    if "noise_cov" in params:
        kwargs["noise_cov"] = cov; cov_supplied = True
    if "mne_obj" in params: kwargs["mne_obj"] = evoked
    elif "evoked" in params: kwargs["evoked"] = evoked
    make_operator(forward, **kwargs)
    stc = solver.apply_inverse_operator(evoked)
    if isinstance(stc, (list, tuple)):
        if len(stc) != 1: raise RuntimeError("Champagne returned multiple STCs")
        stc = stc[0]
    return stc, str(signature), cov_supplied


def fill_metric(row, stc, loaded, surface_transform, stimulation, sesame=False):
    metric = pilot.pilot.calculate_localization_metrics(
        stc, loaded.forward, surface_transform, stimulation
    )
    row["selection_criterion"] = (
        "maximum_posterior_location_probability" if sesame
        else "maximum_source_amplitude"
    )
    distance = float(metric.localization_distance_mm)

    row["selected_time_ms"] = (
        ""
        if sesame
        else float(metric.peak_time * 1000)
    )
    row["selected_x_mm"] = float(metric.peak_coordinate[0] * 1000)
    row["selected_y_mm"] = float(metric.peak_coordinate[1] * 1000)
    row["selected_z_mm"] = float(metric.peak_coordinate[2] * 1000)
    row["localization_distance_mm"] = distance
    row["peak_vertex"] = int(metric.peak_vertex)
    row["nearest_source_distance_mm"] = float(
        metric.nearest_source_distance_mm
    )
    row["geometric_excess_mm"] = (
        distance - float(metric.nearest_source_distance_mm)
    )


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")
    if args.case and args.all_runs:
        raise ValueError("Choose --case OR --all-runs, not both.")

    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not dataset.is_dir(): raise NotADirectoryError(dataset)
    method = args.method
    method_dir = output_root / METHOD_SLUGS[method]
    cases = discover_cases(dataset) if args.all_runs else (
        ordered_unique(tuple(x) for x in args.case) if args.case else list(DEFAULT_PILOT_CASES)
    )
    montages = ordered_unique(args.montage) if args.montage else list(MONTAGES)
    cfgs = configurations(method, args)
    total = len(cases) * len(montages) * len(cfgs)

    print("=" * 78)
    print("SCRIPT 23 — FULL NEXT-METHOD PARAMETER GRID")
    print("=" * 78)
    print(f"Method         : {method}")
    print(f"Runs           : {len(cases)}")
    print(f"Montages       : {', '.join(montages)}")
    print(f"Configurations : {len(cfgs)}")
    print(f"Solutions      : {total:,}")
    print(f"Output         : {method_dir}")
    if args.dry_run:
        print("\nConfiguration IDs:")
        for c in cfgs: print(" ", c["configuration_id"])
        return

    if method_dir.exists():
        if args.overwrite: shutil.rmtree(method_dir)
        elif not args.resume: raise FileExistsError(f"Output exists: {method_dir}")
    method_dir.mkdir(parents=True, exist_ok=True)
    channels_dir = method_dir / "channel_manifests"; channels_dir.mkdir(exist_ok=True)
    results_path = method_dir / "grid_results.csv"
    manifest_path = method_dir / "grid_manifest.json"

    template = pilot.pilot.load_run(dataset=dataset, subject=args.template_subject,
                                    run=args.template_run, task="seegstim")
    template_names, template_xyz, nominal_sets = grid09.fixed_spatial_subsets(template.forward)
    digest = hashlib.sha256(json.dumps({"names": template_names,
                                        "coordinates_m": template_xyz.tolist(),
                                        "nominal": nominal_sets}, sort_keys=True).encode()).hexdigest()
    manifest = dict(version=1, dataset=str(dataset), method=method,
                    cases=[list(c) for c in cases], montages=montages,
                    configurations=cfgs, target_s=list(TARGET),
                    noise_covariance_s=list(NOISE), template_sha256=digest,
                    mne_version=mne.__version__, numpy_version=np.__version__)
    if args.resume:
        if not manifest_path.is_file(): raise FileNotFoundError(manifest_path)
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Manifest mismatch; use original args or new output root.")
    else:
        atomic_json(manifest_path, manifest)
        atomic_json(method_dir / "nominal_channel_sets.json",
                    {"sha256": digest, "sets": nominal_sets})

    rows = []
    if args.resume and results_path.is_file():
        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != list(FIELDS): raise ValueError("CSV header mismatch")
            rows = [r for r in reader if str(r.get("status")).upper() == "PASS"]
        rewrite_rows(results_path, rows)
    elif not results_path.exists():
        rewrite_rows(results_path, [])
    completed = {result_key(r["subject"], r["run"], r["montage"], r["configuration_id"])
                 for r in rows}

    for run_i, (subject, run) in enumerate(cases, 1):
        print(f"\n[{run_i:02d}/{len(cases):02d}] {subject} {run}", flush=True)
        loaded = pilot.pilot.load_run(dataset=dataset, subject=subject, run=run, task="seegstim")
        stimulation, ground_truth = pilot.pilot.get_stimulation_and_ground_truth(dataset, subject, loaded)
        transform_file = (dataset / "derivatives" / "sourcemodelling" / subject / "xfm" /
                          f"{subject}_from-head_to-surface.h5")
        surface_transform = pilot.pilot.load_surface_transform(transform_file)
        good_epochs = loaded.epochs.copy(); good_epochs.pick("eeg", exclude="bads")
        good_names = list(good_epochs.ch_names)
        referenced = pilot.pilot.apply_average_reference(good_epochs)

        for montage in montages:
            if montage == "all_good":
                actual_channels, replacements = good_names, []
                montage_epochs = referenced.copy()
            else:
                actual_channels, replacements = grid09.replace_bad_nominal(
                    nominal_sets[montage], template_names, template_xyz, good_names
                )
                montage_epochs = referenced.copy().pick(actual_channels)
            atomic_json(channels_dir / f"{subject}_{run}_{montage}.json",
                        dict(subject=subject, run=run, montage=montage,
                             actual=actual_channels, replacements=replacements,
                             template_sha256=digest))
            print(f"  {montage:8s}: {len(actual_channels)} channels; "
                  f"{len(replacements)} replacements", flush=True)

            try:
                cov = pilot.pilot.estimate_noise_covariance(
                    montage_epochs, tmin=NOISE[0], tmax=NOISE[1],
                    method="auto", verbose=not args.quiet)
                evoked = pilot.pilot.create_target_evoked(
                    montage_epochs, tmin=TARGET[0], tmax=TARGET[1])
            except Exception as exc:
                print(f"    preprocessing FAIL: {type(exc).__name__}: {exc}")
                continue

            for j, cfg in enumerate(cfgs, 1):
                k = result_key(subject, run, montage, cfg["configuration_id"])
                if k in completed: continue
                start = time.monotonic()
                row = dict.fromkeys(FIELDS, "")
                row.update(subject=subject, run=run, method=method,
                           configuration_id=cfg["configuration_id"], montage=montage,
                           montage_definition=("all_good" if montage == "all_good"
                                               else "fixed_greedy_spatial_v1"),
                           channels=len(actual_channels), replacement_count=len(replacements),
                           epochs=len(loaded.epochs),
                           sampling_frequency_hz=float(loaded.epochs.info["sfreq"]),
                           covariance_method=str(cov.get("method", "auto")),
                           covariance_tmin_s=NOISE[0], covariance_tmax_s=NOISE[1],
                           target_tmin_s=TARGET[0], target_tmax_s=TARGET[1],
                           target_samples=len(evoked.times),
                           stimulation_pair=stimulation.pair,
                           hemisphere=stimulation.hemisphere,
                           ground_truth_x_mm=float(ground_truth[0] * 1000),
                           ground_truth_y_mm=float(ground_truth[1] * 1000),
                           ground_truth_z_mm=float(ground_truth[2] * 1000))
                try:
                    if method == "irMxNE":
                        stc, residual = pilot.run_irmxne(
                            evoked, loaded.forward, cov, cfg["irmxne_alpha"],
                            cfg["irmxne_loose"], cfg["irmxne_depth"], IRMXNE_ITERATIONS,
                            args.quiet)
                        active = count_active_sources(stc)
                        if active == 0: raise RuntimeError("irMxNE returned no active sources")
                        fill_metric(row, stc, loaded, surface_transform, stimulation)
                        row.update(irmxne_alpha=cfg["irmxne_alpha"],
                                   irmxne_loose=cfg["irmxne_loose"],
                                   irmxne_depth=cfg["irmxne_depth"],
                                   irmxne_iterations=IRMXNE_ITERATIONS,
                                   irmxne_random_state=pilot.IRMXNE_RANDOM_STATE,
                                   irmxne_active_sources=active,
                                   irmxne_explained_variance_percent=pilot.pilot.explained_variance(evoked, residual))
                    elif method == "Gamma-MAP":
                        stc, residual = run_gamma(evoked, loaded.forward, cov, cfg, args.quiet)
                        active = count_active_sources(stc)
                        if active == 0: raise RuntimeError("Gamma-MAP returned no active sources")
                        fill_metric(row, stc, loaded, surface_transform, stimulation)
                        row.update(gamma_alpha=cfg["gamma_alpha"],
                                   gamma_loose=cfg["gamma_loose"], gamma_depth=cfg["gamma_depth"],
                                   gamma_xyz_same_gamma=cfg["gamma_xyz_same_gamma"],
                                   gamma_update_mode=1,
                                   gamma_maxit=2000, gamma_tol=1e-4,
                                   gamma_active_sources=active,
                                   gamma_explained_variance_percent=pilot.pilot.explained_variance(evoked, residual))
                    elif method == "SESAME":
                        sesame, stc, noise_std, peak = run_sesame(
                            evoked, loaded.forward, cfg["sesame_fs"], args.quiet)
                        fill_metric(row, stc, loaded, surface_transform, stimulation, sesame=True)
                        row.update(covariance_method="not_used_by_SESAME_fs",
                                   sesame_fs=cfg["sesame_fs"], sesame_n_parts=SESAME_N_PARTS,
                                   sesame_max_n_dips=SESAME_MAX_N_DIPS, sesame_lam=SESAME_LAM,
                                   sesame_hyper_q=SESAME_HYPER_Q,
                                   sesame_random_seed=SESAME_RANDOM_SEED,
                                   sesame_noise_std=noise_std, sesame_target_peak=peak,
                                   sesame_noise_covariance_supplied=False,
                                   sesame_estimated_n_dips=int(sesame.est_n_dips[-1]),
                                   sesame_gof_percent=float(sesame.goodness_of_fit()) * 100,
                                   sesame_source_dispersion_mm=float(sesame.source_dispersion()))
                    elif method == "Champagne":
                        stc, signature, cov_supplied = run_champagne(
                            evoked,
                            loaded.forward,
                            cov,
                            cfg["champagne_pruning_thresh"],
                            args.quiet,
                        )
                        fill_metric(row, stc, loaded, surface_transform, stimulation)
                        row.update(
                            champagne_alpha=CHAMPAGNE_ALPHA,
                            champagne_prune=CHAMPAGNE_PRUNE,
                            champagne_convergence_criterion=CHAMPAGNE_CONVERGENCE,
                            champagne_max_iter=CHAMPAGNE_MAX_ITER,
                            champagne_pruning_thresh=cfg["champagne_pruning_thresh"],
                            champagne_noise_covariance_supplied=cov_supplied,
                            champagne_make_operator_signature=signature,
                        )
                    row["status"] = "PASS"
                except Exception as exc:
                    row.update(status="FAIL", error_type=type(exc).__name__,
                               error_message=str(exc))
                row["runtime_seconds"] = round(time.monotonic() - start, 3)
                append_row(results_path, row)
                if row["status"] == "PASS": completed.add(k)
                if j == 1 or j == len(cfgs) or j % 10 == 0:
                    print(f"    [{j:03d}/{len(cfgs):03d}] "
                          f"{cfg['configuration_id']} {row['status']}", flush=True)

    print("\n" + "=" * 78)
    print("SCRIPT 23 COMPLETE")
    print("=" * 78)
    print(f"Method : {method}")
    print(f"Output : {results_path}")
    print("FAIL rows are intentionally retained for later sensitivity analysis.")


if __name__ == "__main__":
    main()
