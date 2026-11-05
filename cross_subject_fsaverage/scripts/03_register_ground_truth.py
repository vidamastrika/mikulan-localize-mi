#!/usr/bin/env python3
# ============================================================
# Script 03: Register Ground Truth to fsaverage
# ============================================================
# Purpose: Register each native MRI to fsaverage and map saved stimulation
#   midpoints into fsaverage MRI coordinates. Does not fit EEG or inverses.
# Inputs: Script 01 plan; no-warp Script 06 coordinate audit; validated native
#   head-to-scanner bridges; seven native T1w MRIs; fsaverage T1.mgz.
# Outputs: cross_subject_fsaverage/outputs/03_register_ground_truth/ with MRI
#   registrations, 61 mapped points, QA table, input inventory, and manifest.
# Data: seven Localize-MI subjects and their 61 stimulation runs.
# Defaults: --phase audit (no registration); --phase register uses ANTs SyNQuick,
#   0.1 mm maximum point roundtrip, and the local registration environment.
# Reused: existing native coordinate bridges and original ground-truth audit;
#   native-to-native affine/SyN transforms are NOT compatible with fsaverage.
# Expected downstream cases: 732 LOSO + 4,392 Single Reference (5,124).
# Example from repository root, using the existing virtual environment:
#   .venv/bin/python cross_subject_fsaverage/scripts/03_register_ground_truth.py \
#       --phase audit --output-root cross_subject_fsaverage/outputs/03_register_ground_truth_audit
# Full registration is a separate explicit command with --phase register.
# ============================================================
"""Audit or register native ground-truth coordinates into fsaverage MRI space."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = [f"sub-{n:02d}" for n in range(1, 8)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def inside(image: nib.spatialimages.SpatialImage, points: np.ndarray) -> np.ndarray:
    vox = nib.affines.apply_affine(np.linalg.inv(image.affine), points)
    return np.isfinite(vox).all(axis=1) & np.all(vox >= 1, axis=1) & np.all(vox < np.asarray(image.shape[:3])-2, axis=1)


def point_mapper(path: Path):
    """Load the previously pinned ANTs RAS/LPS point-map helper by path."""
    spec = importlib.util.spec_from_file_location("fsaverage_ants_point_map", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.point_map


def run_registration(registration, fixed, moving, directory, subject, environment):
    """Give the shell wrapper whitespace-free aliases, preserving original MRIs."""
    with tempfile.TemporaryDirectory(prefix="fsaverage_ants_", dir="/tmp") as work:
        work = Path(work)
        (work / "fixed.nii.gz").symlink_to(fixed.resolve())
        (work / "moving.nii").symlink_to(moving.resolve())
        (work / "output").symlink_to(directory.resolve(), target_is_directory=True)
        command = [str(registration), "-d", "3", "-f", "fixed.nii.gz",
                   "-m", "moving.nii", "-o", f"output/{subject}_to_fsaverage_", "-t", "s"]
        result = subprocess.run(command, cwd=work, text=True, capture_output=True,
                                env=environment)
    (directory / "command.json").write_text(json.dumps({
        "argv": command, "working_directory": "temporary /tmp/fsaverage_ants_* directory",
        "fixed_original": str(fixed), "moving_original": str(moving),
        "output_original": str(directory), "returncode": result.returncode,
    }, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["audit", "register"], default="audit")
    p.add_argument("--output-root", type=Path,
                   default=Path("cross_subject_fsaverage/outputs/03_register_ground_truth"))
    p.add_argument("--script01-root", type=Path,
                   default=Path("cross_subject_fsaverage/outputs/01_prepare_experiment"))
    p.add_argument("--dataset", type=Path,
                   default=Path.home() / "Documents/DATASETS SOURCE LOCALIZATION/Localize-MI")
    p.add_argument("--fsaverage-root", type=Path,
                   default=Path.home() / "mne_data/MNE-fsaverage-data/fsaverage")
    p.add_argument("--ants-bin", type=Path,
                   default=Path.home() / "micromamba/envs/localizemi-registration/bin",
                   help="Directory containing antsRegistrationSyNQuick.sh and antsApplyTransformsToPoints")
    p.add_argument("--max-roundtrip-mm", type=float, default=0.1)
    args = p.parse_args(argv)
    out = (ROOT / args.output_root).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing existing output directory: {out}")
    if args.max_roundtrip_mm <= 0:
        raise ValueError("max-roundtrip-mm must be positive")
    script01 = (ROOT / args.script01_root).resolve()
    dataset = args.dataset.expanduser().resolve()
    fsroot = args.fsaverage_root.expanduser().resolve()
    antsbin = args.ants_bin.expanduser().resolve()
    plan_path = script01 / "02_fsaverage_reconstruction_plan.csv"
    gt_path = ROOT / "cross_subject_no_warp/outputs/06_evaluate_localization/02_coordinate_validation.csv"
    bridge_path = ROOT / "cross_subject_warp/outputs/06_coordinate_bridge/02_native_physical_bridges.json"
    map_code = ROOT / "cross_subject_warp/source/01_registration_resources/scripts/32_all_reference_localization_evaluation.py"
    fs_path = fsroot / "mri/T1.mgz"
    native_paths = {s: dataset / "derivatives/epochs" / s / "anat" / f"{s}_mod-T1w_maskface.nii" for s in SUBJECTS}
    paths = {"plan": plan_path, "ground_truth": gt_path, "native_bridge": bridge_path,
             "point_mapping_code": map_code, "fsaverage_T1": fs_path,
             **{f"native_T1_{s}": path for s, path in native_paths.items()}}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    plan = pd.read_csv(plan_path)
    if len(plan) != 5124 or set(plan.test_subject) != set(SUBJECTS):
        raise ValueError("Script 01 plan differs from expected 5,124 cases/seven subjects")
    ground = pd.read_csv(gt_path)
    expected = set(map(tuple, plan[["test_subject", "run"]].drop_duplicates().itertuples(index=False, name=None)))
    actual = set(map(tuple, ground[["test_subject", "run"]].itertuples(index=False, name=None)))
    if len(ground) != 61 or actual != expected or not ground.coordinate_audit.eq("PASS_NUMERICAL").all():
        raise ValueError("Historical ground-truth audit does not cover all 61 valid test runs")
    bridges = json.loads(bridge_path.read_text())
    if set(bridges) != set(SUBJECTS):
        raise ValueError("Native coordinate bridge does not cover seven subjects")
    fsimage = nib.load(str(fs_path))
    if fsimage.ndim != 3 or not isinstance(fsimage.header, nib.freesurfer.mghformat.MGHHeader):
        raise ValueError("fsaverage T1 must be a 3-D FreeSurfer MGH MRI (millimetres)")
    fs_surface_ras = np.asarray(fsimage.header.get_vox2ras_tkr(), float)
    fs_scanner_ras = np.asarray(fsimage.affine, float)
    surface_bridge = fs_surface_ras @ np.linalg.inv(fs_scanner_ras)
    inventory = []
    for name, path in paths.items():
        inventory.append(dict(name=name, path=str(path), sha256=sha(path), size_bytes=path.stat().st_size))
    for subject, path in native_paths.items():
        image = nib.load(str(path))
        if image.ndim != 3 or image.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"{subject}: native MRI is not 3-D millimetres")
        bridge = np.asarray(bridges[subject]["head_to_scanner_ras_mm"], float)
        back = np.asarray(bridges[subject]["scanner_ras_to_head_mm"], float)
        if bridge.shape != (4, 4) or not np.allclose(back @ bridge, np.eye(4), atol=1e-5):
            raise ValueError(f"{subject}: invalid native coordinate bridge")
        points = ground[ground.test_subject.eq(subject)][[f"gt_head_{a}_mm" for a in "xyz"]].to_numpy(float)
        if not inside(image, apply(bridge, points)).all():
            raise ValueError(f"{subject}: stimulation midpoint outside native MRI interior")
    registration_rows, mapped_rows = [], []
    if args.phase == "register":
        registration = antsbin / "antsRegistrationSyNQuick.sh"
        point_exe = antsbin / "antsApplyTransformsToPoints"
        for path in [registration, point_exe]:
            if not path.is_file():
                raise FileNotFoundError(f"ANTs executable missing: {path}")
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(inventory).to_csv(out / "01_input_inventory.csv", index=False)
    if args.phase == "register":
        mapper = point_mapper(map_code)
        fs_nifti = out / "fsaverage_T1_scanner_ras.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(fsimage.dataobj, dtype=np.float32), fsimage.affine), str(fs_nifti))
        environment = os.environ.copy()
        environment["PATH"] = str(antsbin) + os.pathsep + environment.get("PATH", "")
        for subject in SUBJECTS:
            started = time.perf_counter()
            directory = out / "registrations" / subject
            directory.mkdir(parents=True)
            prefix = directory / f"{subject}_to_fsaverage_"
            result = run_registration(registration, fs_nifti, native_paths[subject],
                                      directory, subject, environment)
            (directory / "registration.log").write_text(result.stdout + "\nSTDERR\n" + result.stderr)
            affine = Path(str(prefix) + "0GenericAffine.mat")
            warp = Path(str(prefix) + "1Warp.nii.gz")
            inverse_warp = Path(str(prefix) + "1InverseWarp.nii.gz")
            warped = Path(str(prefix) + "Warped.nii.gz")
            missing = [path.name for path in [affine, warp, inverse_warp, warped]
                       if not path.is_file() or path.stat().st_size == 0]
            status = "FAIL_REGISTRATION" if result.returncode or missing else "PENDING_POINT_QA"
            reason = (f"ANTs exit {result.returncode}; missing/empty outputs: {', '.join(missing) or 'none'}"
                      if status == "FAIL_REGISTRATION" else "")
            native = nib.load(str(native_paths[subject]))
            subset = ground[ground.test_subject.eq(subject)].copy()
            head = subset[[f"gt_head_{a}_mm" for a in "xyz"]].to_numpy(float)
            source = apply(bridges[subject]["head_to_scanner_ras_mm"], head)
            if status == "PENDING_POINT_QA":
                try:
                    # ANTs image moving->fixed transforms have reversed point direction.
                    mapped = mapper(str(point_exe), source, [f"[{affine},1]", str(inverse_warp)])
                    returned = mapper(str(point_exe), mapped, [str(warp), str(affine)])
                    errors = np.linalg.norm(returned-source, axis=1)
                    valid = inside(fsimage, mapped) & np.isfinite(errors) & (errors <= args.max_roundtrip_mm)
                    status = "PASS_NUMERICAL_REVIEW_REQUIRED" if valid.all() else "FAIL_POINT_QA"
                    reason = "" if valid.all() else f"{int((~valid).sum())} point(s) fail domain/roundtrip"
                    surface = apply(surface_bridge, mapped)
                    for record, source_point, mapped_point, surface_point, error, ok in zip(
                            subset.to_dict("records"), source, mapped, surface, errors, valid):
                        mapped_rows.append(dict(test_subject=subject, run=record["run"],
                            gt_stimulation_pair=record["gt_stimulation_pair"], gt_hemisphere=record["gt_hemisphere"],
                            native_scanner_ras_x_mm=source_point[0], native_scanner_ras_y_mm=source_point[1], native_scanner_ras_z_mm=source_point[2],
                            fsaverage_scanner_ras_x_mm=mapped_point[0], fsaverage_scanner_ras_y_mm=mapped_point[1], fsaverage_scanner_ras_z_mm=mapped_point[2],
                            fsaverage_surface_ras_x_mm=surface_point[0], fsaverage_surface_ras_y_mm=surface_point[1], fsaverage_surface_ras_z_mm=surface_point[2],
                            roundtrip_mm=float(error), point_status="PASS_NUMERICAL_REVIEW_REQUIRED" if ok else "FAIL_POINT_QA"))
                except Exception as exc:
                    status, reason = "FAIL_POINT_QA", f"{type(exc).__name__}: {exc}"
            registration_rows.append(dict(subject=subject, status=status, reason=reason,
                                          runtime_seconds=time.perf_counter()-started,
                                          native_mri_sha256=sha(native_paths[subject]),
                                          affine_sha256=sha(affine) if affine.is_file() else None,
                                          warp_sha256=sha(warp) if warp.is_file() else None,
                                          inverse_warp_sha256=sha(inverse_warp) if inverse_warp.is_file() else None,
                                          warped_image_sha256=sha(warped) if warped.is_file() else None))
    pd.DataFrame(registration_rows, columns=["subject", "status", "reason", "runtime_seconds", "native_mri_sha256", "affine_sha256", "warp_sha256", "inverse_warp_sha256", "warped_image_sha256"]).to_csv(out / "02_registration_status.csv", index=False)
    pd.DataFrame(mapped_rows).to_csv(out / "03_mapped_ground_truth.csv", index=False)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), phase=args.phase,
                  planned_evaluations=5124, test_runs=61, subjects=SUBJECTS,
                  input_inventory=inventory, fsaverage_scanner_to_surface_ras_mm=surface_bridge.tolist(),
                  point_transform_order="native moving->fsaverage fixed: inverse affine then inverse SyN field for ANTs points; reverse uses forward field then affine",
                  point_coordinates="scanner RAS mm externally; RAS/LPS conversion in pinned point_map helper",
                  max_roundtrip_mm=args.max_roundtrip_mm,
                  registration_fitting_executed=args.phase == "register",
                  inverse_reconstruction_executed=False,
                  review_required=True,
                  limitations=["Native-to-native prior affine/SyN transforms do not target fsaverage and were not reused.",
                               "Native maskface and intact fsaverage MRI may register imperfectly; inspect warped images and contacts before use.",
                               "Numerical point roundtrip does not establish anatomical accuracy.",
                               "This stage is independent of the sub-03 EEG scalp-fit failure; that failure remains in Script 02."])
    (out / "04_registration_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "README.txt").write_text("Review 02_registration_status.csv, mapped contacts, and warped MRI overlays before accepting any anatomical mapping. Original baseline and no-warp results are unchanged.\n")
    print(f"Script 03 {args.phase} outputs: {out}")
    if registration_rows:
        print(pd.DataFrame(registration_rows)[["subject", "status", "reason"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
