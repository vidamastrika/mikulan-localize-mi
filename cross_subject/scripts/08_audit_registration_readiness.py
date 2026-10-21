#!/usr/bin/env python3
"""Read-only prerequisite audit for Localize-MI inter-subject MRI registration.

Run from repo root:
 python cross_subject/scripts/08_audit_registration_readiness.py --target sub-02 --donor sub-01

Does NOT estimate registration, alter MRI/forward models, or calculate localization errors.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np

def info_gifti(path):
    import nibabel as nib
    if not path.exists(): return {'exists':False, 'path':str(path)}
    im=nib.load(str(path)); pts=np.asarray(im.darrays[0].data,float)
    sc=0.001 if np.nanpercentile(np.linalg.norm(pts,axis=1),50)>2 else 1.
    pts=pts*sc
    return {'exists':True,'path':str(path),'vertices':len(pts),'unit_assumed':'mm' if sc==.001 else 'm',
            'bounds_m':{'min':pts.min(0).tolist(),'max':pts.max(0).tolist()},
            'centroid_m':pts.mean(0).tolist(),
            'coord_systems':[{'dataspace':str(getattr(d.coordsys,'dataspace',None)),
                              'xformspace':str(getattr(d.coordsys,'xformspace',None)),
                              'xform':np.asarray(d.coordsys.xform).tolist() if d.coordsys is not None else None}
                             for d in im.darrays[:1]]}

def info_nifti(path):
    import nibabel as nib
    if not path.exists(): return {'exists':False,'path':str(path)}
    im=nib.load(str(path)); aff=np.asarray(im.affine,float)
    shape=im.shape[:3]
    ijk=np.array([[i,j,k,1] for i in [0,shape[0]-1] for j in [0,shape[1]-1] for k in [0,shape[2]-1]],float)
    corners=(ijk@aff.T)[:,:3]/1000
    return {'exists':True,'path':str(path),'shape':list(im.shape),'affine_mm':aff.tolist(),
            'axis_codes':list(nib.aff2axcodes(aff)),
            'bounds_m':{'min':corners.min(0).tolist(),'max':corners.max(0).tolist()},
            'voxel_zooms_mm':[float(z) for z in im.header.get_zooms()[:3]],
            'sform_code':int(im.header['sform_code']), 'qform_code':int(im.header['qform_code'])}

def box_center(data):
    if not data.get('exists'): return None
    b=data['bounds_m'];return (np.asarray(b['min'])+np.asarray(b['max']))/2

def subject_audit(root, sub):
    anat=root/'derivatives'/'sourcemodelling'/sub/'anat'
    raw=root/sub/'anat'
    epoch=root/'derivatives'/'epochs'/sub/'anat'
    out={'subject':sub,'candidate_mris':{},'surfaces':{},'transforms':{}}
    for label,path in {
        'raw_T1w':raw/f'{sub}_T1w.nii',
        'raw_T1w_gz':raw/f'{sub}_T1w.nii.gz',
        'epochs_mod_T1w':epoch/f'{sub}_mod-T1w_maskface.nii',
        'epochs_mod_T1w_gz':epoch/f'{sub}_mod-T1w_maskface.nii.gz'
    }.items():out['candidate_mris'][label]=info_nifti(path)
    for label,path in {
        'left_pial':anat/f'{sub}_hemi-L_pial.surf.gii',
        'right_pial':anat/f'{sub}_hemi-R_pial.surf.gii',
        'outer_skin':anat/f'{sub}_outer_skin.surf.gii'
    }.items():out['surfaces'][label]=info_gifti(path)
    h5=root/'derivatives'/'sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
    out['transforms']['head_to_surface_path']=str(h5)
    if h5.exists():
        import h5py
        with h5py.File(h5,'r') as f:
            out['transforms']['head_to_surface_4x4']=np.asarray(f['trans'][()],float).tolist()
    else:out['transforms']['head_to_surface_missing']=True
    fwdpath=root/'derivatives'/'sourcemodelling'/sub/'fwd'/f'{sub}_fwd.fif'
    out['transforms']['forward_path']=str(fwdpath)
    if fwdpath.exists():
        import mne
        from mne.io.constants import FIFF
        fwd=mne.read_forward_solution(str(fwdpath),verbose='ERROR')
        t=fwd.get('mri_head_t')
        out['transforms']['forward_mri_head_t']=(dict(from_frame=int(t['from']),to_frame=int(t['to']),matrix=np.asarray(t['trans']).tolist()) if t else None)
        if t is not None:
            out['transforms']['forward_transform_frames_expected']=(int(t['from'])==int(FIFF.FIFFV_COORD_MRI) and int(t['to'])==int(FIFF.FIFFV_COORD_HEAD))
    # Geometric bounds alone are NOT a proof that GIFTI and MRI share a coordinate frame.
    for key,mri in out['candidate_mris'].items():
        if not mri.get('exists'):continue
        mc=box_center(mri)
        out.setdefault('bounding_box_center_offsets_mm',{})[key]={}
        for sk,surf in out['surfaces'].items():
            if surf.get('exists'):
                out['bounding_box_center_offsets_mm'][key][sk]=float(np.linalg.norm(mc-box_center(surf))*1000)
    return out

def json_numpy_default(value):
    """Convert NumPy scalars and arrays to standard JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',default='sub-02');p.add_argument('--donor',default='sub-01')
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--output-dir',type=Path,default=Path('cross_subject/outputs/08_registration_readiness'))
    args=p.parse_args()
    if args.target==args.donor:p.error('Target and donor must be different')
    root=args.dataset.expanduser().resolve()
    if not root.exists():p.error(f'Dataset not found: {root}')
    r={'target':args.target,'donor':args.donor,'dataset':str(root),
       'software':{'python':sys.version.split()[0],
                   'antsRegistration':shutil.which('antsRegistration'),
                   'antsApplyTransformsToPoints':shutil.which('antsApplyTransformsToPoints'),
                   'N4BiasFieldCorrection':shutil.which('N4BiasFieldCorrection')},
       'subjects':{s:subject_audit(root,s) for s in (args.target,args.donor)}}
    r['notes']=[
        'Diagnostic only; no registration computed and no localization errors should be inferred.',
        'NIfTI affine is in millimetres; GIFTI vertices were normalized to metres for bounding boxes.',
        'Surface-to-voxel-coordinate compatibility requires independent landmark or transform verification; similar bounds are not proof.',
        'Do not combine donor-space source, target-space truth and MNI coordinates directly.',
        'Target anatomy is allowed ONLY in post-hoc evaluation, not in cross-subject source reconstruction.'
    ]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=args.output_dir/f'{args.target}_with_{args.donor}_readiness.json'
    report.write_text(json.dumps(r, indent=2, default=json_numpy_default), encoding='utf-8')
    print(f'REGISTRATION READINESS AUDIT — {args.target} with donor {args.donor}')
    print(f'Report: {report}')
    for subj,data in r['subjects'].items():
        print(f'\n{subj}')
        for label,item in data['candidate_mris'].items():
            if item.get('exists'):print(f'  MRI {label}: {item["shape"]}, axes={item["axis_codes"]}')
        for label,item in data['surfaces'].items():
            print(f'  {label}: '+(f'{item["vertices"]} vertices; unit={item["unit_assumed"]}' if item.get('exists') else 'MISSING'))
        print('  forward MRI→HEAD:',bool(data['transforms'].get('forward_mri_head_t')))
    print('\nANTs binaries:',', '.join(f'{k}={v or "NOT FOUND"}' for k,v in r['software'].items() if k!='python'))
    print('\nNo registration or localization distance has been calculated.')
if __name__=='__main__':main()
