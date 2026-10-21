#!/usr/bin/env python3
"""Localize-MI Step 18: evaluate sub-01 -> sub-02 MRI affine registration.

Requires: numpy, scipy, nibabel, matplotlib (project .venv).
Uses already-produced ANTs warped donor MRI. Does NOT rerun registration,
read ground-truth source positions, or select transformations by localization error.

IMPORTANT: thresholded MRI foreground is *not* a brain segmentation. Reported Dice
is foreground-mask Dice, not brain-mask Dice or anatomical landmark accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_fill_holes, label


def largest_component(mask):
    labels, count = label(mask)
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == np.argmax(sizes)


def foreground(image, fraction):
    finite_positive = image[np.isfinite(image) & (image > 0)]
    if finite_positive.size == 0:
        raise ValueError('MRI has no positive finite voxels')
    threshold = fraction * float(np.percentile(finite_positive, 99))
    mask = np.isfinite(image) & (image > threshold)
    # Conservative large-connected-region mask; not a validated brain mask.
    return binary_fill_holes(largest_component(mask)), threshold


def norm_intensity(image, mask):
    valid = image[mask]
    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        raise ValueError('Not enough MRI intensity variation')
    return np.clip((image - low) / (high - low), 0, 1)


def measures(fixed, moving, fixed_mask, moving_mask):
    intersection = fixed_mask & moving_mask
    union = fixed_mask | moving_mask
    dice = (2 * np.count_nonzero(intersection) /
            (np.count_nonzero(fixed_mask) + np.count_nonzero(moving_mask)))
    jaccard = np.count_nonzero(intersection) / max(np.count_nonzero(union), 1)
    # Intensity comparison within both thresholded images; avoid empty-background bias.
    if np.count_nonzero(intersection) < 100:
        corr = None
        nmae = None
    else:
        a = norm_intensity(fixed, intersection)[intersection]
        b = norm_intensity(moving, intersection)[intersection]
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
        nmae = float(np.mean(np.abs(a - b)))
    return {
        'foreground_dice_NOT_brain_dice': float(dice),
        'foreground_jaccard': float(jaccard),
        'foreground_intersection_voxels': int(np.count_nonzero(intersection)),
        'fixed_foreground_voxels': int(np.count_nonzero(fixed_mask)),
        'moving_foreground_voxels': int(np.count_nonzero(moving_mask)),
        'intersection_normalized_intensity_pearson': corr,
        'intersection_normalized_intensity_mae': nmae,
    }


def plane(volume, axis, index):
    return np.rot90(np.take(volume, index, axis=axis))


def save_figure(fixed, before, after, mask, output):
    points = np.argwhere(mask)
    center = np.round(np.median(points, axis=0)).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(17.5, 12.4), constrained_layout=True)
    for row, (axis, plane_name) in enumerate(zip((0, 1, 2), ('Axis 0', 'Axis 1', 'Axis 2'))):
        at = int(center[axis])
        f = plane(fixed, axis, at)
        b = plane(before, axis, at)
        a = plane(after, axis, at)
        joint = mask & (after > 0)
        if np.count_nonzero(joint) > 100:
            fn = norm_intensity(fixed, joint)
            bn = norm_intensity(before, joint) if np.count_nonzero(before[joint] > 0) > 100 else before
            an = norm_intensity(after, joint)
        else:
            fn, bn, an = fixed, before, after
        plots = (
            (f, 'Target MRI (sub-02)'),
            (b, 'Donor before affine'),
            (a, 'Donor after affine'),
            (np.abs(plane(fn, axis, at) - plane(an, axis, at)), 'Normalized absolute difference'),
        )
        for col, (img, title) in enumerate(plots):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray' if col < 3 else 'magma', origin='lower', interpolation='nearest')
            if col in (1, 2):
                ax.contour(plane(mask, axis, at).astype(float),
                           levels=[0.5], colors=['cyan'], linewidths=0.7, alpha=0.75)
            ax.set_title(f'{plane_name}, voxel {at}\n{title}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Affine registration quality check | cyan = fixed-MRI foreground outline\n'
                 'Foreground contours are NOT brain segmentations', fontsize=13)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)


def save_overlay(fixed, moving, mask, output):
    center = np.round(np.median(np.argwhere(mask), axis=0)).astype(int)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, axis in zip(axs, (0, 1, 2)):
        cut = int(center[axis])
        f = plane(fixed, axis, cut)
        m = plane(moving, axis, cut)
        # RGB composite: target red/magenta, registered donor green.
        def scale(im):
            p = np.percentile(im[im > 0], 99) if np.any(im > 0) else 1
            return np.clip(im / max(p, 1e-10), 0, 1)
        rgb = np.stack([scale(f), scale(m), scale(f) * 0.2], axis=-1)
        ax.imshow(rgb, origin='lower', interpolation='nearest')
        ax.set_title(f'Axis {axis}, slice {cut}')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Target MRI = magenta | affine-warped donor = green | overlap ≈ pale', fontsize=12)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--target', default='sub-02')
    p.add_argument('--data-root', type=Path, default=Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    p.add_argument('--registration-root', type=Path, default=Path('cross_subject/outputs/09_anatomical_registration'))
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/18_registration_quality'))
    p.add_argument('--foreground-fraction', type=float, default=0.20,
                   help='Threshold as a fraction of the positive-intensity 99th percentile (default 0.20)')
    args = p.parse_args()
    if not (0.01 <= args.foreground_fraction <= 0.9):
        p.error('--foreground-fraction must be in [0.01, 0.9]')
    fixed_path = args.data_root / 'derivatives/epochs' / args.target / 'anat' / f'{args.target}_mod-T1w_maskface.nii'
    donor_path = args.data_root / 'derivatives/epochs' / args.donor / 'anat' / f'{args.donor}_mod-T1w_maskface.nii'
    registration = args.registration_root / f'{args.donor}_to_{args.target}'
    warped_path = registration / f'{args.donor}_in_{args.target}.nii.gz'
    for file in (fixed_path, donor_path, warped_path):
        if not file.is_file():
            raise FileNotFoundError(f'Missing required file: {file}')
    fixed_img = nib.load(str(fixed_path))
    donor_img = nib.load(str(donor_path))
    warped_img = nib.load(str(warped_path))
    fixed = np.nan_to_num(fixed_img.get_fdata(dtype=np.float32))
    # Resample outputs to target NIfTI grid if necessary; identity in world coords.
    reference = (fixed_img.shape, fixed_img.affine)
    if warped_img.shape != fixed_img.shape or not np.allclose(warped_img.affine, fixed_img.affine, atol=0.0001):
        warped_img = resample_from_to(warped_img, reference, order=1)
    warped = np.nan_to_num(warped_img.get_fdata(dtype=np.float32))
    before_img = resample_from_to(donor_img, reference, order=1)
    before = np.nan_to_num(before_img.get_fdata(dtype=np.float32))

    masks = {}
    thresholds = {}
    for name, image in [('fixed', fixed), ('before', before), ('after', warped)]:
        masks[name], thresholds[name] = foreground(image, args.foreground_fraction)
    before_stats = measures(fixed, before, masks['fixed'], masks['before'])
    after_stats = measures(fixed, warped, masks['fixed'], masks['after'])
    out = args.output_root / f'{args.donor}_to_{args.target}'
    out.mkdir(parents=True, exist_ok=True)
    quality = {
        'status': 'FOREGROUND_AND_INTENSITY_QA_ONLY',
        'donor': args.donor, 'target': args.target,
        'reference_mri': str(fixed_path), 'donor_mri': str(donor_path),
        'warped_donor_mri': str(warped_path),
        'target_image_shape': list(fixed_img.shape),
        'voxel_sizes_mm': [float(x) for x in fixed_img.header.get_zooms()[:3]],
        'foreground_method': 'largest connected component + fill holes; threshold fraction of positive 99th percentile',
        'foreground_threshold_fraction': args.foreground_fraction,
        'thresholds': thresholds,
        'before_affine_physical_grid_alignment': before_stats,
        'after_affine_registration': after_stats,
        'delta_after_minus_before': {
            'foreground_dice': after_stats['foreground_dice_NOT_brain_dice'] - before_stats['foreground_dice_NOT_brain_dice'],
            'intensity_pearson': (after_stats['intersection_normalized_intensity_pearson'] - before_stats['intersection_normalized_intensity_pearson'])
            if after_stats['intersection_normalized_intensity_pearson'] is not None and before_stats['intersection_normalized_intensity_pearson'] is not None else None,
        },
        'warnings': [
            'Dice is based on thresholded MRI foreground, NOT validated brain tissue segmentation.',
            'Intensity comparisons depend on modality, contrast, and masked face artifacts.',
            'Rigid+affine registration cannot guarantee accurate cortical correspondence.',
            'This QA does not establish the independent T1w RAS/LPS coordinate convention.',
            'No source coordinates or localization errors are used by this script.',
        ],
    }
    report = out / 'registration_quality.json'
    report.write_text(json.dumps(quality, indent=2))
    fig = out / 'registration_quality_slices.png'
    ov = out / 'registered_mri_overlay.png'
    save_figure(fixed, before, warped, masks['fixed'], fig)
    save_overlay(fixed, warped, masks['fixed'], ov)
    print(f'Report: {report}')
    print(f'Figures: {fig}; {ov}')
    for name, stats in [('Before affine', before_stats), ('After affine', after_stats)]:
        print(f'{name}: foreground Dice = {stats["foreground_dice_NOT_brain_dice"]:.4f}, '
              f'intensity Pearson = {stats["intersection_normalized_intensity_pearson"]}')
    print('CAUTION: foreground Dice is not brain-mask Dice. Registration remains provisional.')


if __name__ == '__main__':
    main()
