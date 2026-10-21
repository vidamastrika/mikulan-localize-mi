#!/usr/bin/env python3
"""Visual MRI registration quality check (does not validate surface->MRI transforms).

Run from the repository root in the existing Python .venv:
 python cross_subject/scripts/10_visual_check_registration.py

Takes the sub-01 MRI already resampled to the sub-02 voxel grid by ANTs.
Creates a 3-orientation target/donor overlay figure and intensity-based QC JSON.
Intensity statistics are diagnostics, not proof of anatomical correspondence.
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


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--target', default='sub-02')
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--registered', type=Path, default=Path('cross_subject/outputs/09_anatomical_registration/sub-01_to_sub-02/sub-01_in_sub-02.nii.gz'))
    p.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/10_registration_quality/sub-01_to_sub-02'))
    a=p.parse_args()
    fixed=a.dataset/'derivatives'/'epochs'/a.target/'anat'/f'{a.target}_mod-T1w_maskface.nii'
    if not fixed.is_file() or not a.registered.is_file():
        raise FileNotFoundError(f'Missing fixed MRI or registered donor MRI: {fixed}, {a.registered}')
    img_f=nib.load(str(fixed));img_m=nib.load(str(a.registered))
    if img_f.shape[:3] != img_m.shape[:3]:
        raise ValueError('Registered donor not on target image voxel grid (shape mismatch)')
    if not np.allclose(img_f.affine,img_m.affine,atol=1e-3):
        raise ValueError('Registered donor not on target image physical grid (affine mismatch)')
    x=np.asarray(img_f.dataobj,dtype=np.float32)
    y=np.asarray(img_m.dataobj,dtype=np.float32)
    valid=np.isfinite(x)&np.isfinite(y)
    fg=x[np.isfinite(x)&(x>0)]
    mg=y[np.isfinite(y)&(y>0)]
    if fg.size==0 or mg.size==0: raise ValueError('MRI images contain no positive tissue intensities')
    # Display normalization only. Both channels have separately estimated ranges.
    flo,fhi=np.percentile(fg,[5,99]);mlo,mhi=np.percentile(mg,[5,99])
    xn=np.clip((x-flo)/max(float(fhi-flo),1e-6),0,1)
    yn=np.clip((y-mlo)/max(float(mhi-mlo),1e-6),0,1)
    tissue=valid & (xn>.12) & (yn>.12)
    xv=xn[tissue].astype(float);yv=yn[tissue].astype(float)
    corr = float(np.corrcoef(xv,yv)[0,1]) if xv.size>10 and np.std(xv)>1e-8 and np.std(yv)>1e-8 else None
    idx=np.argwhere((xn>.25)&valid)
    center=np.median(idx,axis=0).astype(int) if len(idx) else np.array(x.shape)//2
    # imshow slice mapping matches in both images because voxel grids are identical.
    def section(arr,axis):
        sl=np.take(arr,int(center[axis]),axis=axis)
        return np.rot90(sl)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,3,figsize=(14,12),constrained_layout=True)
    for j,(axis,title) in enumerate([(0,'Sagittal'),(1,'Coronal'),(2,'Axial')]):
        xf=section(xn,axis);ym=section(yn,axis)
        axes[j,0].imshow(xf,cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[j,1].imshow(ym,cmap='gray',origin='lower',vmin=0,vmax=1)
        # Color overlay: target = magenta, donor = green. Similar anatomy => near white.
        overlay=np.zeros((*xf.shape,3),dtype=np.float32)
        overlay[...,0]=xf
        overlay[...,2]=xf
        overlay[...,1]=ym
        axes[j,2].imshow(overlay,origin='lower')
        for k in range(3):
            axes[j,k].set_xticks([]);axes[j,k].set_yticks([])
        axes[j,0].set_ylabel(f'{title} slice {center[axis]}',fontsize=11)
    for k,t in enumerate((f'Target {a.target}',f'Registered donor {a.donor}', 'Overlay: target magenta / donor green')):
        axes[0,k].set_title(t,fontsize=12)
    fig.suptitle('ANATOMICAL REGISTRATION VISUAL CHECK — not yet a validated source-space transform',fontsize=14)
    output=a.output_dir/'mri_registration_overlay.png'
    fig.savefig(output,dpi=165,bbox_inches='tight');plt.close(fig)
    report={
        'target':a.target,'donor':a.donor,'fixed_mri':str(fixed),'registered_donor_mri':str(a.registered),
        'shape':list(img_f.shape[:3]),'affine_grid_matches':True,
        'slice_voxel_indices':center.tolist(),
        'positive_tissue_overlap_voxels':int(xv.size),
        'intensity_normalized_pearson_on_joint_tissue':corr,
        'figure':str(output),
        'status':'VISUAL_REVIEW_REQUIRED',
        'warning':'MRI registration overlap alone does not verify native GIFTI surface -> NIfTI physical mapping; do not calculate source localization errors yet.'
    }
    rp=a.output_dir/'mri_registration_quality.json'
    rp.write_text(json.dumps(report,indent=2))
    print(f'MRI overlay: {output}')
    print(f'Quality report: {rp}')
    print(f'Image grid matches: yes | intensity correlation (diagnostic only): {corr}')
    print('Next: review overlay AND independently verify pial surface/SESAME coordinates in MRI space.')


if __name__=='__main__':main()
