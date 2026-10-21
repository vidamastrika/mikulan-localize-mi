#!/usr/bin/env python3
"""Paired within-/cross-subject SESAME pilot comparison (affine, provisional).

Run from the mikulan-localize-mi repository root. Does NOT rerun SESAME,
registration, or choose any coordinate convention by localization error.

Reads the matching within-subject result from Script 23, the cross-subject
Script 05 result, and the Script 16 registration-direction diagnostic.
Recomputes donor->target point mapping with ANTs using its ITK LPS convention;
converts target T1w->native surface via the dataset head-to-surface matrix.

Outputs one row, detailed JSON provenance and MRI QA plot. Cross-subject error
is provisional until physical-axis assumptions and registration accuracy receive
independent anatomical validation. Tested for the existing sub-02/run-01 pilot.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLIP = np.diag([-1., -1., 1.])
XYZ = ('x','y','z')


def get_xyz(record, stem):
    a = np.array([record[f'{stem}_{q}_mm'] for q in XYZ], dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f'Non-finite {stem} coordinate')
    return a


def select_previous(path, target, run, params):
    df = pd.read_csv(path, low_memory=False)
    required = ['subject','run','method','status','selected_x_mm','selected_y_mm',
                'selected_z_mm','ground_truth_x_mm','ground_truth_y_mm',
                'ground_truth_z_mm','localization_distance_mm']
    pairs = {'sesame_fs':'fs','sesame_n_parts':'n_parts',
             'sesame_max_n_dips':'max_n_dips','sesame_lam':'lam',
             'sesame_random_seed':'seed'}
    needed = required+list(pairs)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing Script 23 CSV columns: {missing}')
    df = df.loc[(df.subject.astype(str)==target) &
                (df.run.astype(str).str.strip()==run) &
                (df.method.astype(str).str.lower()=='sesame') &
                (df.status.astype(str).str.upper()=='PASS')].copy()
    for col, key in pairs.items():
        df = df.loc[np.isclose(pd.to_numeric(df[col], errors='coerce'),
                               float(params[key]), atol=1e-7, rtol=0)]
    if 'montage' in df:
        df = df.loc[df.montage.astype(str).str.lower().isin(['all_good','all-good','all good'])]
    if 'sesame_hyper_q' in df:
        required_hq = bool(params['hyper_q'])
        df = df.loc[df.sesame_hyper_q.astype(str).str.lower().isin(['true','1','yes'])==required_hq]
    if len(df) != 1:
        raise ValueError(f'Expected exactly one matching within-subject SESAME row, got {len(df)}. '
                         'Do not select by localization error. Inspect run and parameter fields.')
    return df.iloc[0]


def h5_matrix(root, sub):
    path = root/'derivatives'/'sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
    with h5py.File(path) as f:
        trans = np.asarray(f['trans'], float)
    if trans.shape != (4,4) or not np.allclose(trans[3], [0,0,0,1]):
        raise ValueError(f'Unexpected head-to-surface transform in {path}')
    return trans, path


def transform_point(p_mm, mat):
    return nib.affines.apply_affine(mat, np.asarray(p_mm,float)/1000.)*1000.


def ants_map(point_ras_mm, affine, inverse):
    if not shutil.which('antsApplyTransformsToPoints'):
        raise RuntimeError('Cannot find antsApplyTransformsToPoints. Activate ANTs in PATH.')
    with tempfile.TemporaryDirectory(prefix='sesame_ants17_') as td:
        inp, out = Path(td)/'input.csv', Path(td)/'output.csv'
        itk_lps = FLIP @ np.asarray(point_ras_mm,float)
        pd.DataFrame([dict(x=itk_lps[0], y=itk_lps[1], z=itk_lps[2], t=0.)]).to_csv(
            inp, index=False, float_format='%.12f')
        flag = f'[{affine},1]' if inverse else str(affine)
        result = subprocess.run(['antsApplyTransformsToPoints','-d','3','-i',str(inp),
                                 '-o',str(out),'-t',flag],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f'ANTs point mapping failed: {result.stderr}\n{result.stdout}')
        mapped_lps = pd.read_csv(out)[['x','y','z']].to_numpy(float)
        if mapped_lps.shape != (1,3):
            raise ValueError('Unexpected transformed point output')
        return FLIP @ mapped_lps[0]


def electrode_midpoint(root, sub, a, b):
    path = root/'derivatives'/'epochs'/sub/'ieeg'/f'{sub}_task-seegstim_space-surface_electrodes.tsv'
    df = pd.read_csv(path,sep='\t')
    if df.name.duplicated().any():
        raise ValueError('Duplicate electrode names')
    points = []
    for name in (a,b):
        s = df.loc[df.name.astype(str)==name]
        if len(s)!=1: raise ValueError(f'{name!r} not uniquely found in {path}')
        points.append(s[list(XYZ)].to_numpy(float)[0]*1000.)
    return np.mean(points,axis=0),path


def qa_plot(img_path, gt_mm, within_mm, cross_mm, head_to_surface, output):
    img = nib.load(str(img_path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    points_surface = np.vstack([gt_mm,within_mm,cross_mm])
    points_world = nib.affines.apply_affine(np.linalg.inv(head_to_surface),points_surface/1000)*1000
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine),points_world)
    fig, axs = plt.subplots(1,3,figsize=(16,5))
    names = ['Ground truth','Within-subject','Cross-subject affine']
    cols = ['lime','#36aaff','magenta']
    for ax,axis in zip(axs,(0,1,2)):
        sl = int(np.clip(round(vox[0,axis]),0,vol.shape[axis]-1))
        vol_slice = np.take(vol,sl,axis=axis)
        rem = [d for d in range(3) if d!=axis]
        ax.imshow(vol_slice.T, origin='lower',cmap='gray')
        for p,label,color in zip(vox,names,cols):
            # Mark out-of-plane points explicitly; never pretend all are on the slice.
            outplane = abs(p[axis]-sl)
            ax.scatter(p[rem[0]],p[rem[1]],c=color,s=70,ec='black',
                       marker='o' if outplane<=5 else 'x',label=f'{label} (offset {outplane:.1f} vox)')
        ax.set_title(f'Target MRI axis {axis}, slice {sl}\n'
                     'o ≤5 vox from slice; x = projected from elsewhere')
        ax.set_xlim(0,vol.shape[rem[0]]-1);ax.set_ylim(0,vol.shape[rem[1]]-1)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('SESAME paired comparison — provisional RAS + affine registration\n'
                 'Target native points converted to T1w for MRI overlay',fontsize=12)
    fig.tight_layout(rect=[0,.08,1,.89]);fig.savefig(output,dpi=170);plt.close(fig)
    return vox.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',default='sub-02')
    p.add_argument('--donor',default='sub-01')
    p.add_argument('--run',default='run-01')
    p.add_argument('--contact-a',default="X'1")
    p.add_argument('--contact-b',default="X'2")
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--previous-results',type=Path,default=Path('outputs/23_full_next_method_grid/sesame/grid_results.csv'))
    p.add_argument('--cross-json',type=Path)
    p.add_argument('--diagnostic-json',type=Path)
    p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/17_paired_sesame_validation'))
    args = p.parse_args()
    if args.target==args.donor: p.error('Target and donor must differ')
    if args.run not in ('run-01',):
        p.error('This pilot defaults to run-01; specify contact pair and verify run metadata for other cases.')
    cross_path = args.cross_json or Path('cross_subject/outputs/05_sesame_pilot')/f'{args.target}_with_{args.donor}_{args.run}'/'sesame_pilot_result.json'
    diag_path = args.diagnostic_json or Path('cross_subject/outputs/16_ants_point_validation')/f'{args.donor}_to_{args.target}'/'ants_point_transform_diagnostic.json'
    for path in (args.previous_results,cross_path,diag_path):
        if not path.is_file(): p.error(f'Missing required file: {path}')
    cross = json.loads(cross_path.read_text())
    diag = json.loads(diag_path.read_text())
    if cross.get('status')!='PASS' or cross.get('target')!=args.target or cross.get('donor')!=args.donor:
        raise ValueError('Cross SESAME pilot is not a PASS for requested target/donor')
    if cross.get('ground_truth_used') is not False or cross.get('target_forward_used') is not False:
        raise ValueError('Cross SESAME reconstruction is not confirmed ground-truth/target-forward independent')
    if diag.get('status')!='DIRECTION_SUPPORTED_BY_WARPED_IMAGE' or not diag.get('image_direction_decisively_better_heuristic'):
        raise ValueError('Script 16 direction not supported by warped MRI; abort error calculation')
    if diag.get('donor')!=args.donor or diag.get('target')!=args.target:
        raise ValueError('Diagnostic donor/target mismatch')
    if diag.get('roundtrip_max_error_mm',float('inf'))>.05:
        raise ValueError('Script 16 electrode roundtrip too large')
    if diag.get('selected_donor_to_target_command_affine_inverse_flag') is not True:
        raise ValueError('Expected inverse affine for donor-to-target points; review before proceeding')
    params = cross['parameters']
    prev = select_previous(args.previous_results,args.target,args.run,params)
    within = get_xyz(prev,'selected')
    truth = get_xyz(prev,'ground_truth')
    baseline_error = float(np.linalg.norm(within-truth))
    reported_baseline = float(prev['localization_distance_mm'])
    if abs(baseline_error-reported_baseline)>.1:
        raise ValueError(f'Prior baseline recomputed {baseline_error:.5f} differs from CSV {reported_baseline:.5f}')
    gt_contacts,truth_path = electrode_midpoint(args.dataset,args.target,args.contact_a,args.contact_b)
    if np.linalg.norm(gt_contacts-truth)>.1:
        raise ValueError(f'Ground-truth mismatch {np.linalg.norm(gt_contacts-truth):.4f} mm. '
                         'Inspect stimulation pair; do not silently substitute contacts.')
    # Script 05 stores the selected SESAME forward-source coordinate in DONOR
    # HEAD space (the forward's src rr), not in donor native SURFACE space.
    # The donor H5 is HEAD -> SURFACE. Convert before comparing to Script 16,
    # which was explicitly given a donor native-surface point in millimetres.
    donor_head_point = np.asarray(cross['peak_coordinate_donor_src_rr_m'],float)*1000
    if donor_head_point.shape!=(3,) or not np.isfinite(donor_head_point).all():
        raise ValueError('Invalid donor HEAD-space source peak')
    donor_mat,donor_h5=h5_matrix(args.dataset,args.donor)
    target_mat,target_h5=h5_matrix(args.dataset,args.target)
    donor_point=transform_point(donor_head_point,donor_mat)  # head -> native surface
    donor_t1w=transform_point(donor_point,np.linalg.inv(donor_mat))
    prior_source=diag.get('optional_source_point')
    if not prior_source or np.linalg.norm(donor_point-np.array(prior_source['donor_native_surface_mm']))>.001:
        raise ValueError('Script 16 validated a DIFFERENT source point; re-run Script 16 with current donor source')
    affine=Path(diag['registration_affine'])
    if not affine.is_file(): raise FileNotFoundError(f'Affine file missing: {affine}')
    target_t1w=ants_map(donor_t1w,affine,inverse=True)
    check=ants_map(target_t1w,affine,inverse=False)
    roundtrip=float(np.linalg.norm(check-donor_t1w))
    if roundtrip>.05: raise ValueError(f'Cross SESAME point roundtrip error {roundtrip:.4f} mm')
    if np.linalg.norm(target_t1w-np.asarray(prior_source['target_T1w_candidate_RAS_mm'],float))>.05:
        raise ValueError('Mapped SESAME point does not match Script 16; investigate')
    cross_native=transform_point(target_t1w,target_mat)
    cross_error=float(np.linalg.norm(cross_native-truth))
    delta=cross_error-baseline_error
    output_dir=args.output_root/f'{args.target}_with_{args.donor}_{args.run}'
    output_dir.mkdir(parents=True,exist_ok=True)
    row={'target':args.target,'donor':args.donor,'run':args.run,'method':'SESAME',
         'contact_pair':f'{args.contact_a} - {args.contact_b}',
         'within_subject_error_mm':baseline_error,'cross_subject_affine_error_mm':cross_error,
         'cross_minus_within_mm':delta,'cross_vs_within_pct':100*delta/baseline_error,
         'ants_source_roundtrip_mm':roundtrip,'coordinate_assumption':'T1w points as NIfTI RAS',
         'registration':'rigid+affine; anatomical accuracy not independently validated',
         'status':'PROVISIONAL_AFFINE_NOT_PUBLICATION_VALIDATED'}
    pd.DataFrame([row]).to_csv(output_dir/'paired_sesame_errors.csv',index=False,float_format='%.9f')
    t_mri=args.dataset/'derivatives'/'epochs'/args.target/'anat'/f'{args.target}_mod-T1w_maskface.nii'
    voxel=qa_plot(t_mri,truth,within,cross_native,target_mat,output_dir/'paired_sources_target_mri.png')
    report={'status':row['status'],'metrics':row,'coordinate_chain':{
        'ground_truth_target_surface_mm':truth.tolist(),
        'ground_truth_contacts_surface_midpoint_mm':gt_contacts.tolist(),
        'within_subject_source_target_surface_mm':within.tolist(),
        'cross_subject_source_donor_head_mm':donor_head_point.tolist(),
        'cross_subject_source_donor_surface_mm':donor_point.tolist(),
        'cross_subject_source_donor_T1w_provisional_RAS_mm':donor_t1w.tolist(),
        'cross_subject_source_target_T1w_provisional_RAS_mm':target_t1w.tolist(),
        'cross_subject_source_target_surface_mm':cross_native.tolist(),
        'MRI_plot_point_voxels_ground_truth_within_cross':voxel},
        'parameters':params,
        'qa':{'within_csv_reported_error_mm':reported_baseline,
              'within_recomputed_error_mm':baseline_error,
              'ground_truth_contact_agreement_mm':float(np.linalg.norm(gt_contacts-truth)),
              'cross_ants_roundtrip_mm':roundtrip,
              'script16_roundtrip_max_error_mm':diag['roundtrip_max_error_mm'],
              'script16_direction_supported':True},
        'provenance':{'within_csv':str(args.previous_results),'cross_pilot':str(cross_path),
                      'script16_diagnostic':str(diag_path),'registration_affine':str(affine),
                      'ground_truth_electrodes':str(truth_path),'donor_h5':str(donor_h5),
                      'target_h5':str(target_h5),'target_mri':str(t_mri)},
        'limitations':['RAS versus LPS not independently proved using non-symmetric anatomical labels.',
                       'ANTs transform direction is verified but rigid+affine anatomical accuracy is not.',
                       'Mapped cross-subject error is preliminary and is not a registration-independent method accuracy.',
                       'Ground truth is evaluation-only, not used for SESAME fitting or transform selection.']}
    (output_dir/'paired_sesame_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row,indent=2))
    print('Outputs:',output_dir)

if __name__=='__main__':
    main()
