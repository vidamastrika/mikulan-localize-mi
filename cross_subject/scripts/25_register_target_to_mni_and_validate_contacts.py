#!/usr/bin/env python3
"""Script 25: MRI-based native sub-02 -> MNI152NLin2009aSym SyN registration.

Run at repository root. The MNI coordinates supplied by Localize-MI are *never*
used to optimize registration; they are held out for an independent comparison.
Native electrode coordinates are provisionally treated as NIfTI RAS physical mm.
ANTs uses ITK LPS physical mm, so point-coordinate sign conversion is explicit.

Files written only beneath cross_subject/outputs/25_native_to_mni_registration/.
Dependencies: numpy, nibabel, pandas, matplotlib, scipy; ANTs binaries from the
micromamba 'localizemi-registration' environment (or on PATH).

Example:
  micromamba run -n localizemi-registration .venv/bin/python \
    cross_subject/scripts/25_register_target_to_mni_and_validate_contacts.py \
    --subject sub-02

Existing complete registration is reused by default. For a second audit only:
  ... --audit-only
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

TEMPLATE_ID = 'MNI152NLin2009aSym'
FLIP = np.diag([-1., -1., 1.])


def find_binary(name: str, env: str) -> str:
    if (p := shutil.which(name)):
        return p
    candidate = Path.home()/'micromamba'/'envs'/env/'bin'/name
    if candidate.is_file():
        return str(candidate)
    mm = shutil.which('micromamba')
    if mm:
        result = subprocess.run([mm, 'run', '-n', env, 'sh', '-c', f'command -v {name}'],
                                capture_output=True, text=True)
        for line in reversed(result.stdout.splitlines()):
            if Path(line.strip()).is_file():
                return line.strip()
    raise RuntimeError(f'Cannot locate {name}. Run with micromamba run -n {env}, or set PATH.')


def transform_points(executable: str, pts_ras_mm: np.ndarray, transforms: list[str]) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts_ras_mm, dtype=float))
    if pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError('Point array must be finite N x 3 RAS coordinates (mm)')
    lps = pts @ FLIP.T
    with tempfile.TemporaryDirectory(prefix='localizemi_25_pts_') as temp:
        inp, out = Path(temp)/'input.csv', Path(temp)/'output.csv'
        pd.DataFrame({'x': lps[:, 0], 'y': lps[:, 1], 'z': lps[:, 2], 't': 0}).to_csv(
            inp, index=False, float_format='%.12f')
        command = [executable, '-d', '3', '-i', str(inp), '-o', str(out)]
        for transform in transforms:
            command.extend(['-t', transform])
        proc = subprocess.run(command, text=True, capture_output=True)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError('Point mapping failed\n' + proc.stdout + '\n' + proc.stderr)
        out_pts = pd.read_csv(out)[['x', 'y', 'z']].to_numpy(float) @ FLIP.T
        if out_pts.shape != pts.shape or not np.isfinite(out_pts).all():
            raise RuntimeError('Invalid ANTs transformed points')
        return out_pts


def paired_contacts(dataset: Path, subject: str):
    folder = dataset/'derivatives'/'epochs'/subject/'ieeg'
    def read(space):
        path = folder/f'{subject}_task-seegstim_space-{space}_electrodes.tsv'
        tab = pd.read_csv(path, sep='\t')
        if tab['name'].duplicated().any():
            raise ValueError(f'Duplicate contact names in {path}')
        return tab.set_index('name')[['x','y','z']].astype(float)*1000., path
    native, path_native = read('T1w')
    mni, path_mni = read(TEMPLATE_ID)
    common = native.index.intersection(mni.index, sort=False)
    if len(common) < 3:
        raise RuntimeError('Insufficient matched independent MNI contacts')
    return common.to_list(), native.loc[common].to_numpy(), mni.loc[common].to_numpy(), [str(path_native), str(path_mni)]


def image_metrics(ref, warped, mask_img):
    fixed = np.asarray(ref.dataobj, dtype=np.float32)
    moving = np.asarray(warped.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0.5
    if fixed.shape != moving.shape or mask.shape != fixed.shape:
        raise RuntimeError('Warped image, MNI template, and mask shapes differ')
    valid = mask & np.isfinite(fixed) & np.isfinite(moving) & (fixed > 0) & (moving > 0)
    if valid.sum() < 1000:
        raise RuntimeError('Too few voxels for image similarity')
    a, b = fixed[valid], moving[valid]
    correlation = float(np.corrcoef(a, b)[0, 1])
    a_norm = a / max(float(np.percentile(a, 99)), 1e-10)
    b_norm = b / max(float(np.percentile(b, 99)), 1e-10)
    return {'shared_positive_brain_mask_voxels': int(valid.sum()),
            'brain_mask_voxels': int(mask.sum()),
            'pearson_inside_shared_positive_brain_mask': correlation,
            'normalized_mae_inside_shared_positive_brain_mask': float(np.mean(np.abs(a_norm-b_norm))),
            'note': 'Image similarity is not anatomical landmark accuracy or brain Dice.'}


def draw_overlay(ref, warped, mask, contact_ras, names, errors, out):
    ref_data = np.asarray(ref.dataobj, dtype=np.float32)
    warped_data = np.asarray(warped.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask.dataobj) > 0
    ijk = nib.affines.apply_affine(np.linalg.inv(ref.affine), contact_ras)
    centre = np.round(np.median(ijk, axis=0)).astype(int)
    centre = np.clip(centre, 0, np.array(ref.shape)-1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, axis in enumerate([0, 1, 2]):
        idx = int(centre[axis]); a = np.take(ref_data, idx, axis=axis).T
        b = np.take(warped_data, idx, axis=axis).T
        m = np.take(mask_data, idx, axis=axis).T
        shared = np.maximum(a, b)
        limits = np.percentile(shared[m], [1, 99]) if m.any() else np.percentile(shared, [1, 99])
        lo, hi = float(limits[0]), float(limits[1]); hi = max(hi, lo + 1e-5)
        axes[0, i].imshow(a, cmap='gray', origin='lower', vmin=lo, vmax=hi)
        axes[0, i].set_title(f'MNI reference: axis {axis}, voxel {idx}')
        axes[1, i].imshow(a, cmap='Reds', origin='lower', alpha=.6, vmin=lo, vmax=hi)
        axes[1, i].imshow(b, cmap='Greens', origin='lower', alpha=.45, vmin=lo, vmax=hi)
        axes[1, i].set_title(f'MNI red + registered {"donor"} green, axis {axis}')
        remain = [j for j in range(3) if j != axis]
        for j, point in enumerate(ijk):
            # Show near-slice points only, with actual out-of-plane offset in mm.
            if abs(point[axis]-idx) <= 5:
                axes[1, i].scatter(point[remain[0]], point[remain[1]],
                                   s=45, marker='o', facecolor='none', edgecolor='cyan', linewidths=1.2)
        for row in axes[:, i]:
            row.set_xlabel(f'Voxel axis {remain[0]}'); row.set_ylabel(f'Voxel axis {remain[1]}')
    fig.suptitle('Native MRI → exact MNI template: qualitative QA (cyan = transformed contacts within 5 vox of slice)')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', default='sub-02')
    p.add_argument('--dataset-root', type=Path, default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--template-dir', type=Path, default=Path.home()/'.cache'/'templateflow'/f'tpl-{TEMPLATE_ID}')
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/25_native_to_mni_registration'))
    p.add_argument('--micromamba-env', default='localizemi-registration')
    p.add_argument('--audit-only', action='store_true', help='Skip ANTs registration; require completed outputs')
    p.add_argument('--force', action='store_true', help='Overwrite an existing completed registration')
    args = p.parse_args()
    if args.force and args.audit_only:
        p.error('--force and --audit-only cannot be combined')

    subject = args.subject
    native = args.dataset_root/'derivatives'/'epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
    template = args.template_dir/f'tpl-{TEMPLATE_ID}_res-1_T1w.nii.gz'
    mask = args.template_dir/f'tpl-{TEMPLATE_ID}_res-1_desc-brain_mask.nii.gz'
    for required in [native, template, mask]:
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f'Missing or empty input: {required}')
    reference_img, native_img, mask_img = nib.load(str(template)), nib.load(str(native)), nib.load(str(mask))
    if reference_img.shape != mask_img.shape or not np.allclose(reference_img.affine, mask_img.affine, atol=1e-4):
        raise RuntimeError('Brain mask is not on template grid')

    out = args.output_root/subject
    out.mkdir(parents=True, exist_ok=True)
    prefix = out/f'{subject}_to_{TEMPLATE_ID}_'
    warped = out/f'{subject}_in_{TEMPLATE_ID}.nii.gz'
    inverse_warped = out/f'{TEMPLATE_ID}_in_{subject}.nii.gz'
    affine = Path(str(prefix)+'0GenericAffine.mat')
    warp = Path(str(prefix)+'1Warp.nii.gz')
    inverse_warp = Path(str(prefix)+'1InverseWarp.nii.gz')
    log = out/'ants_registration.log'
    registration_exists = all(f.is_file() and f.stat().st_size > 0 for f in (warped, affine, warp, inverse_warp))
    runtime = None
    registration_executed = False
    if not args.audit_only and (args.force or not registration_exists):
        exe = find_binary('antsRegistration', args.micromamba_env)
        cmd = [exe, '--dimensionality', '3', '--float', '1', '--verbose', '1',
               '--collapse-output-transforms', '1',
               '--output', f'[{prefix},{warped},{inverse_warped}]',
               '--interpolation', 'Linear', '--use-histogram-matching', '0',
               '--winsorize-image-intensities', '[0.005,0.995]',
               '--initial-moving-transform', f'[{template},{native},1]',
               '--transform', 'Rigid[0.1]',
               '--metric', f'MI[{template},{native},1,32,Regular,0.25]',
               '--convergence', '[1000x500x250x0,1e-6,10]',
               '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox',
               '--transform', 'Affine[0.1]',
               '--metric', f'MI[{template},{native},1,32,Regular,0.25]',
               '--convergence', '[1000x500x250x0,1e-6,10]',
               '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox',
               '--transform', 'SyN[0.1,3,0]',
               '--metric', f'CC[{template},{native},1,4]',
               '--convergence', '[100x70x50x20,1e-6,10]',
               '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox']
        print(f'Running ANTs native {subject} → MNI. This may take 10–30+ minutes.', flush=True)
        print('Logging to:', log, flush=True)
        started = time.monotonic()
        with log.open('w') as fh:
            fh.write('COMMAND: '+subprocess.list2cmdline(cmd)+'\n'); fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        runtime = round(time.monotonic()-started, 3)
        if proc.returncode != 0:
            raise RuntimeError(f'ANTs failed (exit code {proc.returncode}); inspect {log}')
        registration_executed = True
    elif registration_exists:
        print('Reusing completed ANTs registration files.', flush=True)
    else:
        raise FileNotFoundError('Audit-only requested, but ANTs transform files are missing.')
    for needed in (warped, affine, warp, inverse_warp):
        if not needed.is_file() or needed.stat().st_size == 0:
            raise RuntimeError(f'Missing ANTs output: {needed}')

    point_exe = find_binary('antsApplyTransformsToPoints', args.micromamba_env)
    names, native_pts, author_mni_pts, contact_paths = paired_contacts(args.dataset_root, subject)
    # ANTs inverse image-resampling sequence is used for native->MNI POINTS.
    # ANTs image-resampling sequence [warp, affine] is used for MNI->native POINTS.
    native_to_mni = [f'[{affine},1]', str(inverse_warp)]
    mni_to_native = [str(warp), str(affine)]
    mapped = transform_points(point_exe, native_pts, native_to_mni)
    returned = transform_points(point_exe, mapped, mni_to_native)
    roundtrip_mm = np.linalg.norm(returned-native_pts, axis=1)
    reference_error_mm = np.linalg.norm(mapped-author_mni_pts, axis=1)
    if float(roundtrip_mm.max()) > 0.5:
        raise RuntimeError(f'Point mapping round trip exceeded 0.5 mm: {roundtrip_mm.max():.4f}')

    csv_path = out/'contact_native_to_mni_validation.csv'
    pd.DataFrame({
        'subject': subject, 'contact': names,
        'native_t1w_x_mm': native_pts[:, 0], 'native_t1w_y_mm': native_pts[:, 1], 'native_t1w_z_mm': native_pts[:, 2],
        'author_mni_x_mm': author_mni_pts[:, 0], 'author_mni_y_mm': author_mni_pts[:, 1], 'author_mni_z_mm': author_mni_pts[:, 2],
        'our_mni_x_mm': mapped[:, 0], 'our_mni_y_mm': mapped[:, 1], 'our_mni_z_mm': mapped[:, 2],
        'author_discrepancy_mm': reference_error_mm, 'roundtrip_error_mm': roundtrip_mm,
    }).to_csv(csv_path, index=False, float_format='%.7f')

    warped_img = nib.load(str(warped))
    if warped_img.shape != reference_img.shape or not np.allclose(warped_img.affine, reference_img.affine, atol=1e-3):
        raise RuntimeError('Warped donor MRI does not match template image grid')
    similarity = image_metrics(reference_img, warped_img, mask_img)
    fig_path = out/'native_to_mni_overlay.png'
    draw_overlay(reference_img, warped_img, mask_img, mapped, names, reference_error_mm, fig_path)
    report = {
        'status': 'EXPLORATORY_MNI_REGISTRATION_CONTACT_VALIDATION',
        'subject': subject, 'template_id': TEMPLATE_ID,
        'template': str(template), 'template_mask': str(mask), 'native_mri': str(native),
        'warped_native_mri': str(warped), 'registration_executed_this_run': registration_executed,
        'registration_runtime_seconds': runtime, 'registration_log': str(log),
        'registration_method': 'rigid + affine + SyN; full MRI MI/CC; no contact-coordinate fitting',
        'transform_file_affine': str(affine), 'transform_file_forward_warp': str(warp),
        'transform_file_inverse_warp': str(inverse_warp),
        'point_sequence_native_to_mni': native_to_mni,
        'point_sequence_mni_to_native': mni_to_native,
        'coordinate_assumption': 'Native T1w electrode coordinates are assumed to be NiBabel NIfTI physical RAS mm; transformed to ANTs LPS and back',
        'source_electrode_files': contact_paths, 'contact_count': len(names),
        'contact_discrepancy_to_author_mni_mm': {
            'median': float(np.median(reference_error_mm)), 'mean': float(np.mean(reference_error_mm)),
            'max': float(np.max(reference_error_mm)), 'per_contact': dict(zip(names, map(float, reference_error_mm)))},
        'point_roundtrip_error_mm': {'median': float(np.median(roundtrip_mm)), 'max': float(np.max(roundtrip_mm))},
        'mni_image_similarity': similarity,
        'outputs': {'csv': str(csv_path), 'figure': str(fig_path)},
        'limitations': [
            'Dataset MNI electrode coordinates represent author normalization; matching the same template does not reproduce their original transform.',
            'Do not use the author MNI contacts for fitting or tuning the registration.',
            'The RAS convention of Localize-MI native T1w contacts is still provisional.',
            'A small point roundtrip error validates implementation, not cortical correspondence.',
            'MNI distances are exploratory; localization errors must remain measured in target native coordinates.',
            'MNI brain surfaces for the eventual visualization must be derived or acquired in the matching template space.'
        ]}
    report_path = out/'native_to_mni_registration_audit.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print('\nRegistration/validation outputs:', out)
    print('Contact agreement vs original Localize-MI MNI TSV:')
    print(f'  median {np.median(reference_error_mm):.2f} mm, max {np.max(reference_error_mm):.2f} mm (NOT source-localization error)')
    print(f'  roundtrip max {np.max(roundtrip_mm):.4f} mm')
    print('  status:', report['status'])
    print('Please share:', report_path, 'and', fig_path)


if __name__ == '__main__':
    main()
