#!/usr/bin/env python3
"""Localize-MI cross-subject: ANTs rigid+affine+SyN pilot, MRI-only QA.

Does NOT map SESAME points, select transforms by localization error, or prove
brain-specific registration. Original affine experiment remains untouched.

Example:
 python cross_subject/scripts/19_run_syn_registration.py --donor sub-01 --target sub-02

Requires user .venv: numpy, scipy, nibabel, matplotlib; ANTs antsRegistration
and antsApplyTransforms found on PATH or through micromamba registration env.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes, label


def ants_executable(name, env_name):
    found = shutil.which(name)
    if found:
        return found
    mm = shutil.which('micromamba')
    if mm:
        result = subprocess.run([mm, 'run', '-n', env_name, 'which', name],
                                capture_output=True, text=True)
        if result.returncode == 0:
            candidate = result.stdout.strip().splitlines()[-1].strip()
            if Path(candidate).is_file():
                return candidate
    raise RuntimeError(f'{name} not found. Put ANTs bin on PATH or check micromamba env {env_name!r}.')


def run_and_log(cmd, log_path):
    print('Launching ANTs registration. This can take many minutes.', flush=True)
    print(' '.join(map(str, cmd)), flush=True)
    with open(log_path, 'w') as out:
        out.write('COMMAND: ' + ' '.join(map(str, cmd)) + '\n\n')
        out.flush()
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        ret = proc.wait()
    if ret:
        raise RuntimeError(f'ANTs registration exited with code {ret}. Check {log_path}')


def foreground(image):
    # This is a rough head/foreground segmentation, NOT a brain tissue mask.
    a = np.asarray(image, dtype=np.float32)
    pos = a[np.isfinite(a) & (a > 0)]
    if pos.size == 0:
        raise ValueError('MRI contains no positive finite intensities')
    threshold = 0.20 * float(np.percentile(pos, 99))
    mask = np.isfinite(a) & (a > threshold)
    lab, count = label(mask)
    if count:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == sizes.argmax()
    return binary_fill_holes(mask), threshold


def metrics(reference, other):
    ref, ref_thresh = foreground(reference)
    mov, mov_thresh = foreground(other)
    intr = ref & mov
    both = int(intr.sum())
    total = int(ref.sum()) + int(mov.sum())
    dice = float(2 * both / total) if total else None
    jacc = float(both / (int((ref | mov).sum()))) if (ref | mov).any() else None
    x = np.asarray(reference[intr], dtype=float)
    y = np.asarray(other[intr], dtype=float)
    if x.size > 10 and np.std(x) > 0 and np.std(y) > 0:
        x /= max(float(np.percentile(x, 99)), 1e-12)
        y /= max(float(np.percentile(y, 99)), 1e-12)
        corr = float(np.corrcoef(x, y)[0, 1])
        mae = float(np.mean(np.abs(x - y)))
    else:
        corr = mae = None
    return {
        'foreground_dice_NOT_brain_dice': dice,
        'foreground_jaccard': jacc,
        'intensity_pearson_inside_shared_foreground': corr,
        'normalized_intensity_mae_inside_shared_foreground': mae,
        'foreground_intersection_voxels': both,
        'reference_foreground_voxels': int(ref.sum()),
        'moving_foreground_voxels': int(mov.sum()),
        'threshold_ref': ref_thresh,
        'threshold_other': mov_thresh,
    }


def figure(reference, warped, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for i, axis in enumerate((0, 1, 2)):
        idx = reference.shape[axis] // 2
        r = np.take(reference, idx, axis=axis).T
        w = np.take(warped, idx, axis=axis).T
        normal = max(np.percentile(reference[reference > 0], 99), 1e-9)
        wnormal = max(np.percentile(warped[warped > 0], 99), 1e-9)
        rn, wn = np.clip(r / normal, 0, 1), np.clip(w / wnormal, 0, 1)
        ax[i, 0].imshow(rn, cmap='gray', origin='lower', vmin=0, vmax=1)
        ax[i, 1].imshow(wn, cmap='gray', origin='lower', vmin=0, vmax=1)
        rgb = np.stack([rn, wn, rn], axis=-1)  # magenta reference, green donor
        ax[i, 2].imshow(np.clip(rgb, 0, 1), origin='lower')
        for j in range(3):
            ax[i, j].set_title(f'Axis {axis}, slice {idx}' + (' — target' if j == 0 else ' — warped donor' if j == 1 else ' — overlay'))
            ax[i, j].set_axis_off()
    fig.suptitle('Localize-MI SyN MRI registration — target magenta, donor green', fontsize=14)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor', default='sub-01')
    parser.add_argument('--target', default='sub-02')
    parser.add_argument('--dataset-root', type=Path, default=Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    parser.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/19_syn_registration'))
    parser.add_argument('--micromamba-env', default='localizemi-registration')
    parser.add_argument('--skip-registration', action='store_true', help='Evaluate already-generated SyN images only')
    args = parser.parse_args()
    if args.donor == args.target:
        parser.error('Donor and target must differ')
    fixed = args.dataset_root / 'derivatives' / 'epochs' / args.target / 'anat' / f'{args.target}_mod-T1w_maskface.nii'
    moving = args.dataset_root / 'derivatives' / 'epochs' / args.donor / 'anat' / f'{args.donor}_mod-T1w_maskface.nii'
    for p in (fixed, moving):
        if not p.is_file():
            raise FileNotFoundError(p)
    out = args.output_root / f'{args.donor}_to_{args.target}'
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'{args.donor}_to_{args.target}_syn_'
    warped = out / 'donor_syn_in_target.nii.gz'
    inverse_warped = out / 'target_syn_in_donor.nii.gz'
    affine = Path(str(prefix) + '0GenericAffine.mat')
    warp = Path(str(prefix) + '1Warp.nii.gz')
    inverse_warp = Path(str(prefix) + '1InverseWarp.nii.gz')
    log = out / 'ants_registration.log'

    elapsed = None
    if not args.skip_registration:
        exe = ants_executable('antsRegistration', args.micromamba_env)
        cmd = [exe, '--dimensionality', '3', '--float', '1', '--verbose', '1',
               '--collapse-output-transforms', '1',
               '--output', f'[{prefix},{warped},{inverse_warped}]',
               '--interpolation', 'Linear',
               '--use-histogram-matching', '0',
               '--winsorize-image-intensities', '[0.005,0.995]',
               '--initial-moving-transform', f'[{fixed},{moving},1]',
               '--transform', 'Rigid[0.1]',
               '--metric', f'MI[{fixed},{moving},1,32,Regular,0.25]',
               '--convergence', '[1000x500x250x0,1e-6,10]',
               '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox',
               '--transform', 'Affine[0.1]',
               '--metric', f'MI[{fixed},{moving},1,32,Regular,0.25]',
               '--convergence', '[1000x500x250x0,1e-6,10]',
               '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox',
               '--transform', 'SyN[0.1,3,0]',
               '--metric', f'CC[{fixed},{moving},1,4]',
               '--convergence', '[100x70x50x20,1e-6,10]',
               '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox']
        t0 = time.monotonic()
        run_and_log(cmd, log)
        elapsed = round(time.monotonic() - t0, 2)
    for p in (warped, inverse_warped, affine, warp, inverse_warp):
        if not p.is_file():
            raise FileNotFoundError(f'Missing expected ANTs output: {p}')

    fixed_img, warped_img = nib.load(str(fixed)), nib.load(str(warped))
    reference = fixed_img.get_fdata(dtype=np.float32)
    registered = warped_img.get_fdata(dtype=np.float32)
    if reference.shape != registered.shape or not np.allclose(fixed_img.affine, warped_img.affine, atol=0.01):
        raise ValueError('Warped image does not match target image grid')
    syn_metrics = metrics(reference, registered)
    baseline_file = Path('cross_subject/outputs/09_anatomical_registration') / f'{args.donor}_to_{args.target}' / f'{args.donor}_in_{args.target}.nii.gz'
    affine_metrics = None
    if baseline_file.is_file():
        affine_img = nib.load(str(baseline_file))
        if affine_img.shape == fixed_img.shape and np.allclose(affine_img.affine, fixed_img.affine, atol=0.01):
            affine_metrics = metrics(reference, affine_img.get_fdata(dtype=np.float32))

    figure(reference, registered, out / 'syn_overlay.png')
    report = {
        'status': 'MRI_ONLY_QA_POINT_TRANSFORMS_NOT_VALIDATED',
        'donor': args.donor, 'target': args.target,
        'fixed_mri': str(fixed), 'moving_mri': str(moving),
        'warped_donor_mri': str(warped), 'inverse_warped_target_mri': str(inverse_warped),
        'forward_transform_sequence_for_image_resampling': [str(warp), str(affine)],
        'affine': str(affine), 'forward_deformation': str(warp), 'inverse_deformation': str(inverse_warp),
        'registration_runtime_seconds': elapsed,
        'registration_log': str(log),
        'syn_quality': syn_metrics,
        'previous_affine_quality_using_identical_measurement': affine_metrics,
        'notes': [
            'Foreground Dice is not brain Dice; mask includes non-brain structures.',
            'SyN point transforms must be separately validated; image resampling and point transforms use different coordinate mappings.',
            'NIfTI world coordinates are RAS, ANTs physical coordinates are LPS; verify convention of dataset electrode coordinates separately.',
            'Do NOT select transforms by SESAME localization error or use ground truth to tune registration.',
            'Nonlinear registration can improve intensity agreement without improving cortical correspondence; anatomical QA remains essential.'
        ]
    }
    destination = out / 'syn_registration_quality.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('\nSyN registration complete. MRI-only quality report:')
    print(json.dumps({'syn': syn_metrics, 'affine': affine_metrics}, indent=2))
    print(f'\nReport: {destination}\nOverlay: {out / "syn_overlay.png"}')
    print('IMPORTANT: Do not calculate SESAME localization error with SyN until point transforms are verified.')

if __name__ == '__main__':
    main()
