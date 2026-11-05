#!/usr/bin/env python3
# ============================================================
# Script 02: Coregister EEG to fsaverage
# ============================================================
# Purpose: Fit and audit rigid test-head to fsaverage-MRI transforms from
#   subject-specific EEG sensor coordinates and the fsaverage scalp surface.
# Inputs: Script 01 manifest/plan; seven native forward FIFs; released EEG
#   electrode TSVs (audit only); fsaverage outer_skin.surf and trans file.
# Outputs: cross_subject_fsaverage/outputs/02_coregister_eeg/ with seven
#   candidate FIF transforms, per-electrode/subject diagnostics, and manifest.
# Data: seven Localize-MI subjects, 61 scheduled runs; one fit per subject.
# Defaults: 40 ICP iterations, 15% trimmed matches, median <= 10 mm and
#   95th-percentile <= 20 mm for numerical acceptance; no template scaling.
# Reused: Script 01 frozen plan; native forward sensor coordinates. No inverse.
# Expected downstream evaluations: 732 LOSO + 4,392 Single Reference (5,124).
# Example from repository root using the existing virtual environment:
#   .venv/bin/python cross_subject_fsaverage/scripts/02_coregister_eeg.py \
#       --output-root cross_subject_fsaverage/outputs/02_coregister_eeg
# ============================================================
"""Fit candidate rigid EEG-to-fsaverage transforms; require scientific review."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = [f"sub-{n:02d}" for n in range(1, 8)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proper rigid transform minimizing paired squared distances, in metres."""
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    rotation = vt.T @ d @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = b - rotation @ a
    return matrix


def icp(sensors: np.ndarray, scalp: np.ndarray, initial: np.ndarray,
        iterations: int, trim: float) -> tuple[np.ndarray, np.ndarray, int]:
    tree = cKDTree(scalp)
    matrix = initial.copy()
    previous = np.inf
    used = 0
    for iteration in range(iterations):
        transformed = apply(matrix, sensors)
        distance, index = tree.query(transformed)
        keep = np.argsort(distance)[:max(3, int(np.ceil(len(sensors) * (1-trim))))]
        update = rigid_fit(transformed[keep], scalp[index[keep]])
        candidate = update @ matrix
        new_distance = tree.query(apply(candidate, sensors))[0]
        objective = float(np.median(new_distance))
        matrix = candidate
        used = iteration + 1
        if abs(previous-objective) < 1e-7:
            break
        previous = objective
    return matrix, tree.query(apply(matrix, sensors))[0], used


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script01-root", type=Path,
                   default=Path("cross_subject_fsaverage/outputs/01_prepare_experiment"))
    p.add_argument("--output-root", type=Path,
                   default=Path("cross_subject_fsaverage/outputs/02_coregister_eeg"))
    p.add_argument("--dataset", type=Path,
                   default=Path.home() / "Documents/DATASETS SOURCE LOCALIZATION/Localize-MI")
    p.add_argument("--fsaverage-root", type=Path,
                   default=Path.home() / "mne_data/MNE-fsaverage-data/fsaverage")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--trim-fraction", type=float, default=0.15)
    p.add_argument("--max-median-mm", type=float, default=10.0)
    p.add_argument("--max-p95-mm", type=float, default=20.0)
    args = p.parse_args(argv)
    if args.iterations < 1 or not 0 <= args.trim_fraction < 0.5:
        raise ValueError("iterations must be positive and trim-fraction in [0, 0.5)")
    if args.max_median_mm <= 0 or args.max_p95_mm <= 0:
        raise ValueError("distance thresholds must be positive")
    script01 = (ROOT / args.script01_root).resolve()
    out = (ROOT / args.output_root).resolve()
    dataset = args.dataset.expanduser().resolve()
    fsroot = args.fsaverage_root.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing existing output directory: {out}")
    manifest_path = script01 / "05_experiment_manifest.json"
    plan_path = script01 / "02_fsaverage_reconstruction_plan.csv"
    scalp_path = fsroot / "bem/outer_skin.surf"
    built_in_path = fsroot / "bem/fsaverage-trans.fif"
    for path in [manifest_path, plan_path, scalp_path, built_in_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
    prep = json.loads(manifest_path.read_text())
    plan = pd.read_csv(plan_path)
    if prep["counts"]["fsaverage_total_planned"] != 5124 or len(plan) != 5124:
        raise ValueError("Script 01 plan does not contain 5,124 cases")
    if set(plan.test_subject) != set(SUBJECTS):
        raise ValueError("Script 01 plan subject set changed")
    import mne
    from mne.io.constants import FIFF
    scalp, _ = mne.read_surface(str(scalp_path), read_metadata=False)
    scalp = np.asarray(scalp, float)
    if scalp.ndim != 2 or scalp.shape[1] != 3 or not np.isfinite(scalp).all():
        raise ValueError("Invalid fsaverage scalp surface")
    # FreeSurfer .surf coordinates are millimetres; MNE transforms and EEG
    # forward sensor coordinates use metres.
    scalp_radius = float(np.percentile(np.linalg.norm(scalp, axis=1), 95))
    if not 50 < scalp_radius < 250:
        raise ValueError(f"Unexpected FreeSurfer scalp radius in mm: {scalp_radius:g}")
    scalp /= 1000.0
    built_in = mne.read_trans(str(built_in_path), return_all=False)
    frames = (int(built_in["from"]), int(built_in["to"]))
    matrix = np.asarray(built_in["trans"], float)
    if frames == (int(FIFF.FIFFV_COORD_HEAD), int(FIFF.FIFFV_COORD_MRI)):
        initial = matrix
    elif frames == (int(FIFF.FIFFV_COORD_MRI), int(FIFF.FIFFV_COORD_HEAD)):
        initial = np.linalg.inv(matrix)
    else:
        raise ValueError(f"Unexpected built-in fsaverage transform frames: {frames}")
    rows, electrode_rows, transforms = [], [], {}
    for subject in SUBJECTS:
        started = time.perf_counter()
        forward_path = dataset / "derivatives/sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif"
        tsv_path = dataset / "derivatives/epochs" / subject / "eeg" / f"{subject}_task-seegstim_electrodes.tsv"
        for path in [forward_path, tsv_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Required subject input missing: {path}")
        fwd = mne.read_forward_solution(str(forward_path), verbose=False)
        channels = fwd["info"]["chs"]
        if len(channels) != 256 or any(int(ch["coord_frame"]) != int(FIFF.FIFFV_COORD_HEAD) for ch in channels):
            raise ValueError(f"{subject}: expected 256 EEG sensors in native head frame")
        names = [ch["ch_name"] for ch in channels]
        sensors = np.asarray([ch["loc"][:3] for ch in channels], float)
        if len(set(names)) != 256 or not np.isfinite(sensors).all():
            raise ValueError(f"{subject}: invalid sensor names or coordinates")
        tsv = pd.read_csv(tsv_path, sep="\t").set_index("name")
        if set(tsv.index) != set(names):
            raise ValueError(f"{subject}: released electrode TSV channel names differ from forward")
        released = tsv.loc[names, ["x", "y", "z"]].to_numpy(float)
        tsv_rms = float(np.sqrt(np.mean(np.sum((sensors-released)**2, axis=1))) * 1000)
        matrix, distance, n_iter = icp(sensors, scalp, initial, args.iterations, args.trim_fraction)
        med, p95, maximum = (float(np.percentile(distance*1000, q)) for q in [50, 95, 100])
        rotation = matrix[:3, :3]
        orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        roundtrip = float(np.max(np.linalg.norm(apply(np.linalg.inv(matrix), apply(matrix, sensors))-sensors, axis=1)) * 1000)
        numerical_pass = (med <= args.max_median_mm and p95 <= args.max_p95_mm and
                          orthogonality < 1e-6 and abs(determinant-1) < 1e-6 and roundtrip < 1e-4)
        status = "PASS_NUMERICAL_REVIEW_REQUIRED" if numerical_pass else "FAIL_NUMERICAL"
        transforms[subject] = matrix
        rows.append(dict(subject=subject, status=status, n_sensors=len(sensors),
                         median_scalp_distance_mm=med, p95_scalp_distance_mm=p95,
                         max_scalp_distance_mm=maximum, tsv_vs_forward_rms_mm=tsv_rms,
                         determinant=determinant, rotation_orthogonality_error=orthogonality,
                         roundtrip_max_mm=roundtrip, iterations=n_iter,
                         translation_norm_mm=float(np.linalg.norm(matrix[:3, 3])*1000),
                         rotation_from_initial_deg=float(np.degrees(np.arccos(np.clip((np.trace(rotation @ initial[:3, :3].T)-1)/2, -1, 1)))),
                         runtime_seconds=time.perf_counter()-started,
                         native_forward_path=str(forward_path), native_forward_sha256=sha256(forward_path),
                         electrode_tsv_path=str(tsv_path), electrode_tsv_sha256=sha256(tsv_path)))
        mapped = apply(matrix, sensors)
        _, nearest_index = cKDTree(scalp).query(mapped)
        for name, original, mapped_point, nearest, dist in zip(names, sensors, mapped, scalp[nearest_index], distance):
            electrode_rows.append(dict(subject=subject, channel=name,
                native_head_x_m=float(original[0]), native_head_y_m=float(original[1]), native_head_z_m=float(original[2]),
                fsaverage_mri_x_m=float(mapped_point[0]), fsaverage_mri_y_m=float(mapped_point[1]), fsaverage_mri_z_m=float(mapped_point[2]),
                nearest_scalp_x_m=float(nearest[0]), nearest_scalp_y_m=float(nearest[1]), nearest_scalp_z_m=float(nearest[2]),
                scalp_distance_mm=float(dist*1000)))
    out.mkdir(parents=True, exist_ok=False)
    transdir = out / "transforms"
    transdir.mkdir()
    for subject, matrix in transforms.items():
        trans = mne.transforms.Transform("head", "mri", matrix)
        mne.write_trans(str(transdir / f"{subject}_head_to_fsaverage-trans.fif"), trans, overwrite=False)
    pd.DataFrame(rows).to_csv(out / "01_subject_coregistration_audit.csv", index=False)
    pd.DataFrame(electrode_rows).to_csv(out / "02_electrode_distances.csv", index=False)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), script="02_coregister_eeg.py",
                  script01_manifest_sha256=sha256(manifest_path), script01_plan_sha256=sha256(plan_path),
                  fsaverage_scalp_sha256=sha256(scalp_path), fsaverage_builtin_trans_sha256=sha256(built_in_path),
                  fsaverage_scalp_file_units="mm; converted to metres before fitting",
                  parameters=dict(iterations=args.iterations, trim_fraction=args.trim_fraction,
                                  max_median_mm=args.max_median_mm, max_p95_mm=args.max_p95_mm,
                                  scale_mode="none", initial_transform="fsaverage built-in transform, normalized to head-to-MRI"),
                  coordinate_frames=dict(input="native test model head, metres", output="fsaverage MRI surface RAS, metres"),
                  method="rigid trimmed nearest-scalp ICP, proper rotation, initialized by built-in fsaverage transform",
                  status_counts=pd.Series([r["status"] for r in rows]).value_counts().to_dict(),
                  notes=["Forward-file sensor coordinates are subject-specific; released EEG electrode TSV is identical across participants and is audit-only.",
                         "Numerical scalp fit does not validate anatomical correspondence or cortical source accuracy.",
                         "Transforms are candidates pending visual/independent review before Script 04.",
                         "No native-to-fsaverage anatomical registration or inverse reconstruction was performed."],
                  software_versions=dict(mne=mne.__version__, numpy=np.__version__, pandas=pd.__version__))
    (out / "03_coregistration_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "README.txt").write_text("Seven candidate native-head to fsaverage-MRI rigid transforms. Review 01_subject_coregistration_audit.csv and 02_electrode_distances.csv. PASS_NUMERICAL_REVIEW_REQUIRED is not scientific approval.\n")
    print(f"Wrote candidate transforms and diagnostics: {out}")
    print(pd.DataFrame(rows)[["subject", "status", "median_scalp_distance_mm", "p95_scalp_distance_mm", "tsv_vs_forward_rms_mm"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
