#!/usr/bin/env python3
"""Localize-MI: visual QA of native pial surfaces/SEEG contacts versus T1w MRI.

Compares two *candidate* interpretations of the dataset's T1w table axes:
  RAS_same_axes: native surface -> inverse(head_to_surface) -> NIfTI RAS mm
  LPS_to_RAS_flip_XY: the same, then flip X and Y

This DOES NOT perform image registration, modify original data, or estimate
cross-subject localization errors. Visual anatomical plausibility is not a
validated coordinate-system proof. Uses voxel axes, not radiological labels.

From project root:
 python cross_subject/scripts/15_validate_surface_mri_alignment.py \\
      --donor sub-01 --target sub-02
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


def apply_affine(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    return nib.affines.apply_affine(matrix, xyz)


def validate_contacts(subject, root, inv_head_surface):
    d = root / 'derivatives' / 'epochs' / subject / 'ieeg'
    s = pd.read_csv(d / f'{subject}_task-seegstim_space-surface_electrodes.tsv', sep='\t')
    t = pd.read_csv(d / f'{subject}_task-seegstim_space-T1w_electrodes.tsv', sep='\t')
    both = s[['name', 'x', 'y', 'z']].merge(
        t[['name', 'x', 'y', 'z']], on='name', suffixes=('_surface', '_t1w'), validate='one_to_one'
    )
    if both.empty:
        raise ValueError(f'{subject}: no matching contacts in surface and T1w tables')
    src = both[['x_surface', 'y_surface', 'z_surface']].to_numpy(float) * 1000
    dest = both[['x_t1w', 'y_t1w', 'z_t1w']].to_numpy(float) * 1000
    mapped = apply_affine(src, inv_head_surface)
    discrepancy = np.linalg.norm(dest - mapped, axis=1)
    if np.max(discrepancy) > 0.1:
        raise ValueError(f'{subject}: surface->T1w electrode mismatch: max {np.max(discrepancy):.4f} mm')
    return dest, list(both['name']), discrepancy


def load_subject(root, subject, step):
    anat = root / 'derivatives' / 'epochs' / subject / 'anat'
    filename = anat / f'{subject}_mod-T1w_maskface.nii'
    if not filename.exists():
        filename = filename.with_suffix('.nii.gz')
    img = nib.load(str(filename))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f'Expected 3D MRI, got {data.shape}')
    xf = root / 'derivatives' / 'sourcemodelling' / subject / 'xfm' / f'{subject}_from-head_to-surface.h5'
    with h5py.File(xf, 'r') as f:
        matrix = np.asarray(f['trans'], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f'Unexpected transformation shape {matrix.shape}')
    inv_head_surface_mm = np.linalg.inv(matrix).copy()
    inv_head_surface_mm[:3, 3] *= 1000  # source coords in mm, HDF5 transform in m

    pts = {}
    for hemi in ('L', 'R'):
        f = root / 'derivatives' / 'sourcemodelling' / subject / 'anat' / f'{subject}_hemi-{hemi}_pial.surf.gii'
        gii = nib.load(str(f))
        # Anatomical vertex coordinates in these files are in millimetres.
        arr = np.asarray(gii.darrays[0].data, dtype=float)
        pts[hemi] = apply_affine(arr[::step], inv_head_surface_mm)
    contacts, names, residual = validate_contacts(subject, root, inv_head_surface_mm)
    return img, data, pts, contacts, names, residual, filename, xf


def point_diagnostic(pts_ras, image, data):
    vox = apply_affine(pts_ras, np.linalg.inv(image.affine))
    shp = np.array(data.shape)
    inside = np.all((vox >= 0) & (vox <= shp - 1), axis=1)
    signal = np.full(len(vox), np.nan)
    if inside.any():
        signal[inside] = map_coordinates(data, vox[inside].T, order=1, mode='constant', cval=np.nan)
    return vox, inside, signal


def make_figure(subject, name, data, surface_vox, contact_vox, contacts_inside, names, path, width):
    # Overview: brain-centre slice. Contact-centre: slice near SEEG coordinates.
    all_surface = np.concatenate(list(surface_vox.values()), axis=0)
    center = np.median(all_surface, axis=0)
    contact_center = np.median(contact_vox, axis=0)
    low, high = np.nanpercentile(data, [1, 99.7])
    if high <= low: high = low + 1
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=145)
    for row, (label, center_xyz) in enumerate((('Central anatomy', center), ('Electrode-centered', contact_center))):
        for slice_axis in range(3):
            ax = axes[row, slice_axis]
            idx = int(np.clip(np.rint(center_xyz[slice_axis]), 0, data.shape[slice_axis]-1))
            remaining = [dim for dim in range(3) if dim != slice_axis]
            sl = np.take(data, idx, axis=slice_axis)
            ax.imshow(sl, origin='lower', cmap='gray', vmin=low, vmax=high, interpolation='nearest')
            for hemi, color in [('L', '#33d1ff'), ('R', '#ffb347')]:
                v = surface_vox[hemi]
                keep = np.abs(v[:, slice_axis] - idx) <= width
                if keep.any():
                    ax.scatter(v[keep, remaining[1]], v[keep, remaining[0]],
                               s=0.75, c=color, alpha=0.55, linewidths=0, rasterized=True)
            ckeep = contacts_inside & (np.abs(contact_vox[:, slice_axis] - idx) <= 5)
            if ckeep.any():
                ax.scatter(contact_vox[ckeep, remaining[1]], contact_vox[ckeep, remaining[0]],
                           s=42, c='#fe3863', marker='o', edgecolors='black', linewidths=0.65, zorder=10)
                for i in np.flatnonzero(ckeep):
                    ax.annotate(names[i], (contact_vox[i, remaining[1]], contact_vox[i, remaining[0]]),
                                xytext=(3, 3), textcoords='offset points', fontsize=7, color='#ffff66')
            ax.set_xlim(0, data.shape[remaining[1]]-1)
            ax.set_ylim(0, data.shape[remaining[0]]-1)
            ax.set_title(f'{label}: voxel axis {slice_axis} = {idx}\n'
                         f'pial vertices ±{width:g} vox, contacts ±5 vox')
            ax.set_xlabel(f'Voxel axis {remaining[1]}')
            ax.set_ylabel(f'Voxel axis {remaining[0]}')
            ax.set_aspect('equal', adjustable='box')
    handles = [Line2D([0], [0], marker='.', markersize=13, ls='', color='#33d1ff', label='Left pial'),
               Line2D([0], [0], marker='.', markersize=13, ls='', color='#ffb347', label='Right pial'),
               Line2D([0], [0], marker='o', markersize=8, ls='', color='#fe3863', label='SEEG contacts')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f'{subject} — candidate {name} — surface and contacts over MRI\n'
                 'Native surface → inverse head-to-surface → proposed NIfTI RAS physical coordinates', fontsize=13)
    fig.subplots_adjust(top=0.87, bottom=0.08, hspace=0.35, wspace=0.25)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--target', default='sub-02')
    p.add_argument('--dataset', type=Path, default=Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI')
    p.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/15_surface_mri_alignment'))
    p.add_argument('--vertex-step', type=int, default=5, help='Use every Nth pial vertex (default 5)')
    p.add_argument('--slice-width', type=float, default=2.0, help='Slice plane thickness in MRI voxels')
    args = p.parse_args()
    if args.vertex_step < 1 or args.slice_width <= 0:
        p.error('--vertex-step must be >=1 and --slice-width > 0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'status': 'VISUAL_QA_ONLY', 'purpose': 'Compare anatomical alignment under two candidate coordinate conventions; no error calculation',
              'transform': 'inverse head-to-surface HDF5 (metres) applied to native surface coordinates (mm); optional XY flip before NiBabel world->voxel',
              'warning': 'Do not select candidate solely from positive MRI intensity or field-of-view percentages. Confirm anatomical contours and direction convention.',
              'subjects': {}}
    for sub in dict.fromkeys([args.donor, args.target]):
        img, data, vertices, contacts, names, residual, mri_file, xf = load_subject(args.dataset, sub, args.vertex_step)
        result = {'mri': str(mri_file), 'head_to_surface_file': str(xf), 'contact_count': len(names),
                  'contact_surface_to_t1w_max_discrepancy_mm': float(np.max(residual)), 'candidates': {}}
        for key, flip in [('RAS_same_axes', [1,1,1]), ('LPS_to_RAS_flip_XY', [-1,-1,1])]:
            vox, per_hemi = {}, {}
            for hemi, xyz in vertices.items():
                vx, inside, signal = point_diagnostic(xyz * flip, img, data)
                vox[hemi] = vx
                per_hemi[hemi] = {'sampled_vertices': len(xyz), 'inside_FOV': int(np.sum(inside)),
                                  'inside_FOV_fraction': float(np.mean(inside)),
                                  'positive_intensity_fraction_of_inside': float(np.mean(signal[inside] > 0)) if inside.any() else None}
            cvx, cinside, csig = point_diagnostic(contacts * flip, img, data)
            out = args.output_dir / f'{sub}_{key}_surface_mri_overlay.png'
            make_figure(sub, key, data, vox, cvx, cinside, names, out, args.slice_width)
            result['candidates'][key] = {'surface': per_hemi, 'contacts_inside_FOV': int(cinside.sum()),
                                         'contacts_positive_intensity': int(np.sum(csig[cinside] > 0)),
                                         'figure': str(out)}
            print(f'{sub} {key}: left {per_hemi["L"]["inside_FOV_fraction"]:.1%} right {per_hemi["R"]["inside_FOV_fraction"]:.1%} pial vertices in MRI FOV; contact {cinside.sum()}/{len(names)} in FOV')
            print(' Figure:', out)
        report['subjects'][sub] = result
    out = args.output_dir / f'{args.donor}_to_{args.target}_surface_mri_alignment.json'
    out.write_text(json.dumps(report, indent=2))
    print('Saved:', out)
    print('IMPORTANT: These figures are QA, not formal validation of the MRI world-coordinate frame.')


if __name__ == '__main__':
    main()
