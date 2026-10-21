#!/usr/bin/env python3
"""Diagnostic only: test whether Localize-MI T1w electrode coordinates use RAS or LPS.

This script does not perform registration or calculate localization errors.
Run with the project's .venv (nibabel, numpy, scipy, pandas, matplotlib).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


def evaluate(img, pts_mm):
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), pts_mm)
    shape = np.array(img.shape[:3])
    inside = np.all((vox >= 0) & (vox <= shape - 1), axis=1)
    data = np.asarray(img.dataobj, dtype=np.float32)
    # Evaluation only; 'positive intensity' is not a validated anatomical landmark.
    intens = np.full(len(vox), np.nan)
    if inside.any():
        intens[inside] = map_coordinates(data, vox[inside].T, order=1, mode='constant', cval=np.nan)
    return vox, inside, intens


def overlay(img, vox, inside, name, subject, outfile):
    data = np.asarray(img.dataobj, dtype=np.float32)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    clipped = np.clip(data, np.nanpercentile(data, 2), np.nanpercentile(data, 99.5))
    axes = [(0, 1, 2, 'Sagittal (i fixed)'), (1, 0, 2, 'Coronal (j fixed)'), (2, 0, 1, 'Axial (k fixed)')]
    for ax, (slice_axis, u, v, title) in zip(axs, axes):
        if inside.any():
            slice_no = int(np.round(np.median(vox[inside, slice_axis])))
        else:
            slice_no = img.shape[slice_axis] // 2
        slice_no = int(np.clip(slice_no, 0, img.shape[slice_axis] - 1))
        sl = np.take(clipped, slice_no, axis=slice_axis)
        # sl axis 0/1 remains in original order; imshow origin=lower uses x=axis1, y=axis0
        remaining = [a for a in range(3) if a != slice_axis]
        ax.imshow(sl, origin='lower', cmap='gray', interpolation='nearest')
        valid = inside & (np.abs(vox[:, slice_axis] - slice_no) <= 5)
        if valid.any():
            ax.scatter(vox[valid, remaining[1]], vox[valid, remaining[0]], s=32,
                       c='#ff7f0e', edgecolors='black', linewidths=.5)
        ax.set_title(f'{title}\nvoxel={slice_no}, contacts ±5 vox: {valid.sum()}')
        ax.set_xlabel(f'Voxel axis {remaining[1]}')
        ax.set_ylabel(f'Voxel axis {remaining[0]}')
    fig.suptitle(f'{subject}: T1w electrodes interpreted as {name} (diagnostic, not registration)')
    fig.tight_layout()
    fig.savefig(outfile, dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='sub-02')
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--dataset', type=Path, default=Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI')
    p.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/14_t1w_axes_validation'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report={'purpose':'Diagnostic RAS vs LPS candidate check; not validation of anatomical registration', 'subjects':{}}
    for subject in dict.fromkeys([args.donor, args.target]):
        mri=args.dataset / 'derivatives' / 'epochs' / subject / 'anat' / f'{subject}_mod-T1w_maskface.nii'
        path=args.dataset / 'derivatives' / 'epochs' / subject / 'ieeg' / f'{subject}_task-seegstim_space-T1w_electrodes.tsv'
        if not mri.is_file() or not path.is_file():
            raise FileNotFoundError(f'Missing T1w image or contact table: {mri}, {path}')
        image=nib.load(str(mri))
        tab=pd.read_csv(path, sep='\t')
        xyz=tab[['x','y','z']].astype(float).to_numpy()*1000.0
        if not np.isfinite(xyz).all():
            raise ValueError(f'Nonfinite T1w contacts: {path}')
        rec={'mri':str(mri),'electrodes':str(path),'contact_count':len(xyz),
             'image_affine_RAS_mm':image.affine.tolist(),'candidates':{}}
        for name,pts in [('RAS_same_axes',xyz),('LPS_to_RAS_flip_XY',xyz*np.array([-1,-1,1]))]:
            vox,inside,intens=evaluate(image,pts)
            out=args.output_dir / f'{subject}_{name}_contact_overlay.png'
            overlay(image,vox,inside,name,subject,out)
            sample=intens[np.isfinite(intens)]
            rec['candidates'][name]={
                'contacts_inside_FOV':int(inside.sum()),
                'contacts_positive_intensity':int((sample>0).sum()),
                'median_sampled_intensity':float(np.median(sample)) if len(sample) else None,
                'coordinate_range_mm':[pts.min(axis=0).tolist(),pts.max(axis=0).tolist()],
                'figure':str(out)
            }
            print(f'{subject} {name}: {inside.sum()}/{len(xyz)} in MRI FOV; {(sample>0).sum()}/{len(sample)} positive MRI voxels')
        report['subjects'][subject]=rec
    outfile=args.output_dir/f'{args.donor}_to_{args.target}_axes_diagnostic.json'
    outfile.write_text(json.dumps(report,indent=2))
    print('Report:',outfile)
    print('NOTE: FOV and positive intensity alone do not establish the correct coordinate frame. Inspect overlays and dataset provenance.')

if __name__=='__main__':
    main()
