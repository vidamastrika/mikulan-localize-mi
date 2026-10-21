#!/usr/bin/env python3
"""Validate ANTs affine *point* transformation against a precomputed warped MRI.

No fitting, registration, SESAME rerun or localization-error calculation occurs.
The affine registration is assumed to be moving DONOR -> fixed TARGET.

Required: nibabel, numpy, scipy, pandas, matplotlib, h5py, ANTs' antsApplyTransformsToPoints.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

FLIP = np.diag([-1.0, -1.0, 1.0])


def apply_ants_points(points_ras_mm, transform, invert=False):
    """Supply ITK LPS physical points; receive output in RAS physical mm."""
    points = np.asarray(points_ras_mm, dtype=float).reshape(-1, 3)
    with tempfile.TemporaryDirectory(prefix='ants_points_') as d:
        inp, out = Path(d)/'in.csv', Path(d)/'out.csv'
        coords = points @ FLIP.T
        pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2],
                      't': np.zeros(len(coords))}).to_csv(inp, index=False, float_format='%.12f')
        spec = f'[{transform},1]' if invert else str(transform)
        command = ['antsApplyTransformsToPoints', '-d', '3', '-i', str(inp), '-o', str(out), '-t', spec]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{' '.join(command)}\n{completed.stderr}\n{completed.stdout}")
        result = pd.read_csv(out)
        return result[['x', 'y', 'z']].to_numpy(dtype=float) @ FLIP.T


def sample_mri(volume, voxel_coords):
    return map_coordinates(volume, np.asarray(voxel_coords, float).T, order=1,
                           mode='constant', cval=0.0, prefilter=False)


def world_to_voxel(img, xyz):
    return nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)


def image_consistency(fixed_warped, moving_img, target_world, mat, invert):
    moving_world = apply_ants_points(target_world, mat, invert=invert)
    voxel = world_to_voxel(moving_img, moving_world)
    valid = np.all((voxel >= 0) & (voxel <= np.array(moving_img.shape[:3])-1), axis=1)
    observed = sample_mri(fixed_warped['volume'], fixed_warped['voxels'])
    expected = sample_mri(np.asarray(moving_img.dataobj, dtype=np.float32), voxel)
    if valid.sum() == 0:
        return {'valid_fraction': 0.0, 'n_inside': 0, 'rmse': None, 'mae': None, 'correlation': None}
    diff = expected[valid] - observed[valid]
    x, y = expected[valid], observed[valid]
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
    return {'valid_fraction': float(valid.mean()), 'n_inside': int(valid.sum()),
            'rmse': float(np.sqrt(np.mean(diff**2))), 'mae': float(np.mean(np.abs(diff))),
            'correlation': correlation}


def load_contacts(root, sub):
    path = root/'derivatives'/'epochs'/sub/'ieeg'/f'{sub}_task-seegstim_space-T1w_electrodes.tsv'
    df = pd.read_csv(path, sep='\t')
    if not {'x', 'y', 'z'}.issubset(df.columns):
        raise ValueError(f'Missing coordinates: {path}')
    return df, df[['x', 'y', 'z']].to_numpy(float)*1000.


def surface_point_to_mri(root, sub, mm):
    path = root/'derivatives'/'sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
    with h5py.File(path) as f:
        mat = f['trans'][()]
    # Matrix maps EEG head to surface; prior Script13 confirmed inverse maps surface to T1w electrode frame.
    return nib.affines.apply_affine(np.linalg.inv(mat), np.array(mm, float)/1000.)*1000.


def draw_figure(target_img, transformed, donor_contacts, target_contacts, output, note):
    vol = np.asarray(target_img.dataobj, dtype=np.float32)
    warped_vox = world_to_voxel(target_img, transformed)
    target_vox = world_to_voxel(target_img, target_contacts)
    center = np.median(warped_vox, axis=0).astype(int)
    center = np.clip(center, [0, 0, 0], np.array(vol.shape[:3])-1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, fixed_axis in zip(axes, (0, 1, 2)):
        sl = int(center[fixed_axis])
        cut = [slice(None)]*3
        cut[fixed_axis] = sl
        img_slice = vol[tuple(cut)]
        remaining = [i for i in (0, 1, 2) if i != fixed_axis]
        ax.imshow(img_slice.T, cmap='gray', origin='lower')
        for coords, color, name in [(warped_vox, 'magenta', 'Donor contacts transformed'),
                                    (target_vox, 'lime', 'Target contacts')]:
            near = np.abs(coords[:, fixed_axis]-sl) <= 5
            if np.any(near):
                ax.scatter(coords[near, remaining[0]], coords[near, remaining[1]],
                           s=40, c=color, ec='black', lw=0.45, label=name)
        ax.set_xlim(0, vol.shape[remaining[0]]-1)
        ax.set_ylim(0, vol.shape[remaining[1]]-1)
        ax.set_title(f'Target MRI voxel axis {fixed_axis}={sl}\npoints within ±5 voxels')
        ax.set_xlabel(f'Voxel axis {remaining[0]}')
        ax.set_ylabel(f'Voxel axis {remaining[1]}')
        ax.legend(fontsize=8, loc='upper right') if ax.get_legend_handles_labels()[0] else None
    fig.suptitle('ANTs affine: donor contacts mapped into target MRI\n'+note, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--target', default='sub-02')
    p.add_argument('--data-root', type=Path,
                   default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--registration-root', type=Path, default=Path('cross_subject/outputs/09_anatomical_registration'))
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/16_ants_point_validation'))
    p.add_argument('--donor-surface-point-mm', type=float, nargs=3, metavar=('X','Y','Z'),
                   help='Optional SESAME location in DONOR native surface coordinates, in mm')
    p.add_argument('--samples', type=int, default=2500)
    args = p.parse_args()
    if shutil.which('antsApplyTransformsToPoints') is None:
        p.error('antsApplyTransformsToPoints not found; activate the ANTs/micromamba environment')
    reg = args.registration_root/f'{args.donor}_to_{args.target}'
    affine = reg/f'{args.donor}_to_{args.target}_0GenericAffine.mat'
    warped = reg/f'{args.donor}_in_{args.target}.nii.gz'
    donor_mri = args.data_root/'derivatives'/'epochs'/args.donor/'anat'/f'{args.donor}_mod-T1w_maskface.nii'
    target_mri = args.data_root/'derivatives'/'epochs'/args.target/'anat'/f'{args.target}_mod-T1w_maskface.nii'
    for f in [affine, warped, donor_mri, target_mri]:
        if not f.exists():
            p.error(f'Missing required file: {f}')
    donor_img, target_img, warped_img = map(lambda x: nib.load(str(x)), (donor_mri, target_mri, warped))
    if donor_img.ndim != 3 or target_img.ndim != 3 or warped_img.ndim != 3:
        p.error('Expected 3D MRI volumes')
    if target_img.shape != warped_img.shape or not np.allclose(target_img.affine, warped_img.affine, atol=1e-3):
        p.error('Warped donor is not on target image grid; abort comparison')
    wvol = np.asarray(warped_img.dataobj, dtype=np.float32)
    nonzero = wvol[wvol > 0]
    if len(nonzero) < args.samples:
        p.error('Warped donor image has insufficient nonzero voxels')
    # Select distributed tissue sites on the fixed grid (not source-localization outcomes).
    cutoff = float(np.quantile(nonzero, 0.25))
    candidates = np.column_stack(np.where(wvol > cutoff))
    rng = np.random.default_rng(20261021)
    fixed_vox = candidates[rng.choice(len(candidates), min(args.samples, len(candidates)), replace=False)]
    fixed_world = nib.affines.apply_affine(target_img.affine, fixed_vox)
    fixed_warped = {'volume': wvol, 'voxels': fixed_vox}
    checks = {
        'ANTs_affine_without_inverse_for_target_to_donor': image_consistency(fixed_warped, donor_img, fixed_world, affine, invert=False),
        'ANTs_affine_with_inverse_for_target_to_donor': image_consistency(fixed_warped, donor_img, fixed_world, affine, invert=True)
    }
    metrics = [(name, r['rmse']) for name, r in checks.items() if r['rmse'] is not None]
    if not metrics:
        p.error('No valid image consistency samples')
    best_key, _ = min(metrics, key=lambda x: x[1])
    target_to_donor_inverse = best_key.endswith('with_inverse_for_target_to_donor')
    better = checks[best_key]['rmse']
    other = [r['rmse'] for k, r in checks.items() if k != best_key][0]
    # Threshold is a safety check, not statistical proof. An exactly matching warped image is expected.
    decisively_better = other is not None and other > better*2 and other-better > 0.01
    donor_df, donor_contacts = load_contacts(args.data_root, args.donor)
    target_df, target_contacts = load_contacts(args.data_root, args.target)
    donor_to_target_inverse = not target_to_donor_inverse
    mapped = apply_ants_points(donor_contacts, affine, invert=donor_to_target_inverse)
    recovered = apply_ants_points(mapped, affine, invert=target_to_donor_inverse)
    delta = np.linalg.norm(recovered-donor_contacts, axis=1)
    output_dir = args.output_root/f'{args.donor}_to_{args.target}'
    output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({'name': donor_df['name']})
    for dim, lab in enumerate('xyz'):
        out[f'donor_{lab}_RAS_mm'] = donor_contacts[:, dim]
        out[f'target_{lab}_RAS_mm'] = mapped[:, dim]
        out[f'roundtrip_{lab}_RAS_mm'] = recovered[:, dim]
    out['roundtrip_error_mm'] = delta
    out.to_csv(output_dir/'transformed_donor_contacts.csv', index=False, float_format='%.9f')
    report = {
        'status': 'DIRECTION_SUPPORTED_BY_WARPED_IMAGE' if decisively_better and float(np.max(delta)) < .05 else 'REVIEW_REQUIRED',
        'caution': 'Registration quality and T1w-coordinate RAS convention remain separate questions. Do not use localization error to choose transform direction.',
        'donor': args.donor, 'target': args.target,
        'registration_affine': str(affine), 'warped_donor': str(warped),
        'reference_image_comparison': checks,
        'selected_target_to_donor_command_affine_inverse_flag': target_to_donor_inverse,
        'selected_donor_to_target_command_affine_inverse_flag': donor_to_target_inverse,
        'image_direction_decisively_better_heuristic': bool(decisively_better),
        'coordinate_handling': 'Input points assumed NIfTI RAS physical mm; flip X,Y to ITK LPS physical mm before antsApplyTransformsToPoints and flip back afterwards',
        'donor_electrode_count': len(donor_contacts), 'target_electrode_count': len(target_contacts),
        'roundtrip_max_error_mm': float(np.max(delta)),
        'roundtrip_median_error_mm': float(np.median(delta)),
    }
    if args.donor_surface_point_mm is not None:
        sesame_donor_surface = np.array(args.donor_surface_point_mm)
        sesame_donor_t1w = surface_point_to_mri(args.data_root, args.donor, sesame_donor_surface)
        sesame_target_t1w = apply_ants_points(sesame_donor_t1w[None, :], affine,
                                                invert=donor_to_target_inverse)[0]
        sesame_return = apply_ants_points(sesame_target_t1w[None, :], affine,
                                            invert=target_to_donor_inverse)[0]
        report['optional_source_point'] = {
            'donor_native_surface_mm': sesame_donor_surface.tolist(),
            'donor_T1w_candidate_RAS_mm': sesame_donor_t1w.tolist(),
            'target_T1w_candidate_RAS_mm': sesame_target_t1w.tolist(),
            'roundtrip_error_mm': float(np.linalg.norm(sesame_return-sesame_donor_t1w)),
            'note': 'Diagnostics only. No localization error computed.'}
    draw_figure(target_img, mapped, donor_contacts, target_contacts,
                output_dir/'donor_contacts_mapped_to_target_mri.png',
                f'provisional T1w RAS; direction image-supported={decisively_better}')
    (output_dir/'ants_point_transform_diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print(f'\nOutputs: {output_dir}')
    if not decisively_better:
        print('WARNING: image-based direction distinction not sufficiently strong; do not proceed to localization error.')


if __name__ == '__main__':
    main()
