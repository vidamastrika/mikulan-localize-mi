#!/usr/bin/env python3
"""Localize-MI cross-subject Step 21: post-hoc local MRI registration QA.

Compares existing affine-warped and SyN-warped donor images against the target
around the known target stimulation site, for evaluation ONLY. The source/ground
truth is never used to fit or select a transformation. Uses target MRI physical
RAS coordinates provisionally as in Scripts 17-20; this convention is not
independently established. Regional intensity agreement does not establish
correspondence of cortical landmarks or registration accuracy in millimetres.

Run at repository root with the project's Python .venv (no ANTs executable needed):
  python cross_subject/scripts/21_validate_local_anatomical_alignment.py \
      --donor sub-01 --target sub-02 --run run-01
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


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--target', default='sub-02')
    p.add_argument('--run', default='run-01')
    p.add_argument('--dataset-root', type=Path, default=Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/21_local_registration_validation'))
    p.add_argument('--radii-mm', type=float, nargs='+', default=[15., 25., 40.])
    p.add_argument('--display-half-width-mm', type=float, default=48.)
    return p.parse_args()


def read_image(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required image: {path}')
    img = nib.load(str(path))
    return img, np.asarray(img.dataobj, dtype=np.float32)


def align_grid(img, target):
    if img.shape[:3] == target.shape[:3] and np.allclose(img.affine, target.affine, atol=1e-3):
        return img
    return resample_from_to(img, (target.shape[:3], target.affine), order=1)


def local_scores(ref, other, region):
    mask = region & np.isfinite(ref) & np.isfinite(other) & (ref > 0) & (other > 0)
    n = int(np.count_nonzero(mask))
    result = {'region_voxels': int(np.count_nonzero(region)),
              'shared_positive_voxels': n, 'shared_positive_fraction': float(n/max(np.count_nonzero(region), 1))}
    if n < 100:
        result.update(pearson=None, normalized_mae=None)
        return result
    a, b = ref[mask].astype(np.float64), other[mask].astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-10 and np.std(b) > 1e-10 else None
    # Region-specific robust contrast normalization; NOT absolute MRI intensity fidelity.
    def normalise(x):
        low, high = np.percentile(x, [2, 98])
        return np.clip((x-low) / max(high-low, 1e-9), 0., 1.)
    result.update(pearson=corr, normalized_mae=float(np.mean(np.abs(normalise(a)-normalise(b)))))
    return result


def centred_roi(affine, shape, centre_ras, radius):
    # Compute a small bounding voxel box then Euclidean physical distance there;
    # this correctly handles nonstandard NIfTI axis order and obliquity.
    center_vox = nib.affines.apply_affine(np.linalg.inv(affine), centre_ras)
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
    margin = int(np.ceil(radius / float(np.min(voxel_sizes)))) + 3
    lo = np.maximum(np.floor(center_vox).astype(int) - margin, 0)
    hi = np.minimum(np.ceil(center_vox).astype(int) + margin + 1, shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    xyz = np.indices(tuple(hi-lo), dtype=np.float64).reshape(3, -1).T + lo
    world = nib.affines.apply_affine(affine, xyz)
    mask = (np.linalg.norm(world-centre_ras, axis=1) <= radius).reshape(tuple(hi-lo))
    return sl, mask


def plane_cut(arr, axis, idx, pixel_halfwidth, centre_vox):
    cut = np.take(arr, idx, axis=axis)
    remaining_axes = [i for i in range(3) if i != axis]
    # Explicit plotted dimensions: horizontal=remaining_axes[0], vertical=remaining_axes[1].
    # Transposition ensures x index corresponds to first remaining axis.
    cut = cut.T
    x_mid, y_mid = centre_vox[remaining_axes[0]], centre_vox[remaining_axes[1]]
    x1, x2 = max(0,int(np.floor(x_mid-pixel_halfwidth))), min(cut.shape[1],int(np.ceil(x_mid+pixel_halfwidth+1)))
    y1, y2 = max(0,int(np.floor(y_mid-pixel_halfwidth))), min(cut.shape[0],int(np.ceil(y_mid+pixel_halfwidth+1)))
    return cut[y1:y2, x1:x2], [x1,x2,y1,y2], (x_mid,y_mid)


def write_figure(ref, affine, syn, image_affine, centre_mm, half_width_mm, out_path):
    centre_vox = nib.affines.apply_affine(np.linalg.inv(image_affine), centre_mm)
    px = int(np.ceil(half_width_mm / float(np.min(np.linalg.norm(image_affine[:3,:3],axis=0)))))
    fig, axs = plt.subplots(3,4,figsize=(17,12), constrained_layout=True)
    rows = ('MRI axis 0','MRI axis 1','MRI axis 2')
    for axis, axrow in enumerate(axs):
        index = int(np.clip(round(centre_vox[axis]),0,ref.shape[axis]-1))
        a, ext, centre = plane_cut(ref,axis,index,px,centre_vox)
        b, _, _ = plane_cut(affine,axis,index,px,centre_vox)
        c, _, _ = plane_cut(syn,axis,index,px,centre_vox)
        # Same reference-driven display range in all grayscale panels.
        positive = a[a>0]
        vmin, vmax = (np.percentile(positive,[2,98]) if positive.size else (0,1))
        vmax = max(float(vmax),float(vmin)+1e-6)
        axrow[0].imshow(a,cmap='gray',vmin=vmin,vmax=vmax,origin='lower',extent=ext)
        axrow[1].imshow(b,cmap='gray',vmin=vmin,vmax=vmax,origin='lower',extent=ext)
        axrow[2].imshow(c,cmap='gray',vmin=vmin,vmax=vmax,origin='lower',extent=ext)
        # Red = target MRI, green = donor warped: normalized independently in this displayed ROI.
        def norm(x):
            return np.clip((x-vmin)/(vmax-vmin),0,1)
        rgb = np.zeros((*a.shape,3),dtype=float)
        rgb[...,0] = norm(a)
        rgb[...,1] = norm(c)
        axrow[3].imshow(rgb,origin='lower',extent=ext)
        for j,ax in enumerate(axrow):
            ax.plot(centre[0],centre[1],'+',color='yellow',markersize=10,markeredgewidth=1.5)
            ax.set_xlim(ext[0],ext[1]);ax.set_ylim(ext[2],ext[3]);ax.set_aspect('equal')
            ax.set_xlabel('Voxel index (image axis)')
            if j==0: ax.set_ylabel(f'{rows[axis]}, slice {index}\nVoxel index')
    for ax,title in zip(axs[0],('Target MRI','Affine-warped donor','SyN-warped donor','SyN overlay: target red / donor green')):
        ax.set_title(title,fontsize=11)
    fig.suptitle('Local anatomical alignment — fixed ROI around stimulation site (post hoc QA)\n'
                 'Yellow + = evaluation centre; images use voxel axes (not localization error)',fontsize=13)
    fig.savefig(out_path,dpi=180)
    plt.close(fig)


def main():
    args=get_args()
    repo=Path.cwd()
    pair=f'{args.donor}_to_{args.target}'
    aud_path=repo/'cross_subject/outputs/20_syn_point_validation'/f'{args.target}_with_{args.donor}_{args.run}'/'syn_paired_sesame_audit.json'
    if not aud_path.exists():
        raise FileNotFoundError(f'Run Script 20 first; audit missing: {aud_path}')
    audit=json.loads(aud_path.read_text())
    m=audit.get('metrics',{})
    if (m.get('donor'),m.get('target'),m.get('run')) != (args.donor,args.target,args.run):
        raise ValueError('Script 20 audit does not match requested donor/target/run')
    center=np.asarray(audit['source_coordinates_mm']['ground_truth_target_surface'],float)
    # Get the validated surface-to-T1w transformation indirectly from Script 20:
    # target_surface = target_T1w + head_to_surface_mm.
    # Script 20 has no target T1w GT entry, so read target head-to-surface H5.
    import h5py
    h5=args.dataset_root/'derivatives/sourcemodelling'/args.target/'xfm'/f'{args.target}_from-head_to-surface.h5'
    with h5py.File(h5,'r') as f:
        transform=np.asarray(f['trans'],float)
    if transform.shape!=(4,4):raise ValueError('Unexpected H5 transform')
    center_t1w=nib.affines.apply_affine(np.linalg.inv(transform),center/1000)*1000
    paths={
      'target':args.dataset_root/'derivatives/epochs'/args.target/'anat'/f'{args.target}_mod-T1w_maskface.nii',
      'affine':repo/'cross_subject/outputs/09_anatomical_registration'/pair/f'{args.donor}_in_{args.target}.nii.gz',
      'syn':repo/'cross_subject/outputs/19_syn_registration'/pair/'donor_syn_in_target.nii.gz',
    }
    target,ref=read_image(paths['target'])
    aff_img, _=read_image(paths['affine'])
    syn_img, _=read_image(paths['syn'])
    affine=np.asarray(align_grid(aff_img,target).dataobj,dtype=np.float32)
    syn=np.asarray(align_grid(syn_img,target).dataobj,dtype=np.float32)
    if not (np.isfinite(center_t1w).all() and np.isfinite(ref).all()):
        raise ValueError('Non-finite centre or MRI')
    centre_vox=nib.affines.apply_affine(np.linalg.inv(target.affine),center_t1w)
    if np.any(centre_vox<0) or np.any(centre_vox>=np.array(ref.shape)):
        raise ValueError(f'Evaluation centre outside image: {centre_vox}')
    scored={}
    for radius in args.radii_mm:
        if radius<=0:raise ValueError('Radii must be positive')
        sl,roi=centred_roi(target.affine,np.array(ref.shape),center_t1w,float(radius))
        a=local_scores(ref[sl],affine[sl],roi)
        b=local_scores(ref[sl],syn[sl],roi)
        scored[str(radius)]={'affine':a,'syn':b,
          'delta_pearson_syn_minus_affine': b['pearson']-a['pearson'] if a['pearson'] is not None and b['pearson'] is not None else None,
          'delta_normalized_mae_syn_minus_affine':b['normalized_mae']-a['normalized_mae'] if a['normalized_mae'] is not None and b['normalized_mae'] is not None else None}
    out=repo/args.output_root/pair/f'{args.target}_{args.run}'
    out.mkdir(parents=True,exist_ok=True)
    png=out/'local_affine_vs_syn_mri_overlay.png'
    write_figure(ref,affine,syn,target.affine,center_t1w,args.display_half_width_mm,png)
    report={
      'status':'POST_HOC_LOCAL_MRI_QA_NOT_ANATOMICAL_VALIDATION',
      'subjects':{'donor':args.donor,'target':args.target,'run':args.run},
      'centre':{'target_native_surface_mm':center.tolist(), 'target_T1w_candidate_RAS_mm':center_t1w.tolist(),'target_mri_voxel_indices':centre_vox.tolist()},
      'radii_mm':args.radii_mm,'metrics_by_radius_mm':scored,
      'files':{k:str(v) for k,v in paths.items()},
      'figure':str(png),
      'method':'Spherical Euclidean physical-space ROI; Pearson correlation and independently percentile-normalized MAE within shared positive-intensity voxels; no brain tissue segmentation',
      'limitations':[
       'The ROI is centred on known stimulation ground truth: valid only for post-hoc QA, never for fitting/tuning/choosing registrations.',
       'Local intensity similarity is not direct evidence of homologous cortical anatomy or local spatial error in mm.',
       'T1w electrode and surface points provisionally assumed to match NIfTI physical RAS, as in Scripts 17-20.',
       'A high local metric may reflect smooth structures or contrast, not accurate cortical folds.',
       'Use predefined registration protocol before multi-run assessment; do not choose transform with smallest SESAME error.'
      ]}
    target_json=out/'local_registration_quality.json'
    target_json.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {target_json}\nWrote {png}')
    for radius,r in scored.items():
      print(f"  ROI {radius} mm: affine r={r['affine']['pearson']}, SyN r={r['syn']['pearson']}; affine NMAE={r['affine']['normalized_mae']}, SyN NMAE={r['syn']['normalized_mae']}")
    print('CAUTION: post-hoc regional image similarity only; not registration accuracy.')


if __name__=='__main__':main()
