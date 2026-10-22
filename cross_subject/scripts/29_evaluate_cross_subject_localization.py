#!/usr/bin/env python3
"""Script 29 — Evaluate 61 frozen-SESAME cross-subject runs (read-only inputs).

FROZEN SESAME (selected from all-PASS 61/61 within-subject configuration):
    montage=all_good; fs=0.12; n_parts=100; max_n_dips=1;
    lam=0.25; hyper_q=True; random_seed=0; noise_covariance=None;
    target window=[-2,+2] ms; within-subject median=14.196784 mm.

TERMINOLOGY: test subject supplies EEG and stimulation reference; reference
subject supplies the forward/head model. Reconstruction already happened in
Script 27 and NEVER uses the test participant's MRI or ground truth.

Evaluation ONLY: mapped reference-model source -> test native surface (mm),
then Euclidean distance to test stimulation midpoint. Affine and SyN are
separate, provisional registration-dependent estimates. Test MRI and truth
are used only at evaluation. RAS assumption remains provisional.

Default requires exactly 61 matching within-subject grid rows and 61 scheduled
cross-subject jobs; missing/invalid jobs are output as explicit FAIL rows.
Registration point directions follow validated Scripts 16/20. Per-pair image
resampling direction is checked with sampled MRI intensity points, and the
source-point round-trip must be <=0.1 mm. Does NOT rerun SESAME/registration.

Run from the project root:
  micromamba run -n localizemi-registration .venv/bin/python \\
    cross_subject/scripts/29_evaluate_cross_subject_localization.py \\
    --within-grid /path/to/your/full_sesame_grid_results.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

FLIP = np.diag([-1., -1., 1.])
CFG = dict(fs=.12, n_parts=100, max_n_dips=1, lam=.25, hyper_q=True, seed=0)
OUTPUT_COLUMNS = [
    'test_subject','reference_subject','run','stimulation_pair','status','failure_reason',
    'within_status','within_subject_error_mm','cross_affine_error_mm','cross_syn_error_mm',
    'affine_minus_within_mm','syn_minus_within_mm',
    'ground_truth_x_mm','ground_truth_y_mm','ground_truth_z_mm',
    'within_x_mm','within_y_mm','within_z_mm',
    'reference_head_x_mm','reference_head_y_mm','reference_head_z_mm',
    'reference_surface_x_mm','reference_surface_y_mm','reference_surface_z_mm',
    'affine_test_x_mm','affine_test_y_mm','affine_test_z_mm',
    'syn_test_x_mm','syn_test_y_mm','syn_test_z_mm',
    'affine_roundtrip_mm','syn_roundtrip_mm','within_ground_truth_contact_discrepancy_mm',
    'reference_registration_image_rmse_affine','reference_registration_image_rmse_syn',
    'fs','montage','n_parts','max_n_dips','lam','hyper_q','seed',
    'source_json','affine_registration','syn_registration','coordinate_assumption'
]


def cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--batch-root', type=Path, default=Path('cross_subject/outputs/27_cross_subject_sesame_batch'))
    p.add_argument('--within-grid', type=Path, default=Path('outputs/23_full_next_method_grid/sesame/grid_results.csv'))
    p.add_argument('--affine-root', type=Path, default=Path('cross_subject/outputs/09_anatomical_registration'))
    p.add_argument('--syn-root', type=Path, default=Path('cross_subject/outputs/19_syn_registration'))
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/29_cross_subject_localization'))
    p.add_argument('--only-test', help='Evaluate only one test subject for troubleshooting')
    p.add_argument('--expect-runs', type=int, default=61)
    p.add_argument('--ants-bin', type=Path, help='ANTs antsApplyTransformsToPoints executable')
    p.add_argument('--image-qa-samples', type=int, default=200)
    p.add_argument('--max-roundtrip-mm', type=float, default=0.1)
    p.add_argument('--max-image-rmse', type=float, default=.015,
                   help='Tolerance for reproducing saved warped image; units are MRI intensity, not mm')
    return p.parse_args()


def require(p):
    if not Path(p).is_file():
        raise FileNotFoundError(f'Missing file: {p}')
    return Path(p)


def read_json(p):
    return json.loads(require(p).read_text())


def transform(m, xyz_mm):
    return nib.affines.apply_affine(m, np.asarray(xyz_mm,float)/1000.)*1000.


def h5(dataset, sub):
    path = dataset/'derivatives'/'sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
    with h5py.File(require(path), 'r') as f:
        m = np.asarray(f['trans'],float)
    if m.shape!=(4,4) or not np.allclose(m[3], [0,0,0,1]):
        raise ValueError(f'Invalid head->surface affine: {path}')
    return m


def exe_location(args):
    if args.ants_bin:
        return str(require(args.ants_bin))
    p=shutil.which('antsApplyTransformsToPoints')
    if p: return p
    p=Path.home()/'micromamba/envs/localizemi-registration/bin/antsApplyTransformsToPoints'
    if p.is_file(): return str(p)
    raise FileNotFoundError('Cannot locate antsApplyTransformsToPoints; run with micromamba or --ants-bin')


def point_map(exe, points_ras, terms):
    pts=np.asarray(points_ras,float).reshape(-1,3)
    if not np.isfinite(pts).all(): raise ValueError('Nonfinite input point(s)')
    lps=pts@FLIP.T
    with tempfile.TemporaryDirectory(prefix='localizemi29_') as td:
        inf,out=Path(td)/'input.csv',Path(td)/'output.csv'
        pd.DataFrame({'x':lps[:,0],'y':lps[:,1],'z':lps[:,2],'t':np.zeros(len(pts))}).to_csv(inf,index=False,float_format='%.12f')
        cmd=[exe,'-d','3','-i',str(inf),'-o',str(out)]
        for term in terms: cmd+=['-t',str(term)]
        res=subprocess.run(cmd,text=True,capture_output=True)
        if res.returncode or not out.is_file():
            raise RuntimeError('ANTs point-map failed: '+res.stdout[-600:]+res.stderr[-600:])
        mapped=pd.read_csv(out)[['x','y','z']].to_numpy(float)
    if mapped.shape!=pts.shape or not np.isfinite(mapped).all():
        raise ValueError('ANTs returned invalid points')
    return mapped@FLIP.T


def contact_midpoint(dataset, sub, pair):
    # The original grid contains the ground truth and the exact stimulation
    # pair. Its contact spelling uses abbreviated second contact (X'1-2).
    path=dataset/'derivatives'/'epochs'/sub/'ieeg'/f'{sub}_task-seegstim_space-surface_electrodes.tsv'
    tab=pd.read_csv(require(path),sep='\t')
    name_to_xyz={str(row['name']):row[['x','y','z']].to_numpy(float)*1000.
                 for _,row in tab.iterrows()}
    s=str(pair).replace(' ','')
    if '-' not in s: raise ValueError(f'Unrecognized stimulation pair: {s}')
    first,second=s.split('-',1)
    if second not in name_to_xyz:
        prefix=first.rstrip('0123456789')
        second=prefix+second
    if first not in name_to_xyz or second not in name_to_xyz:
        raise ValueError(f'Contacts {first}, {second} missing from {path}')
    return (name_to_xyz[first]+name_to_xyz[second])/2.


def within_grid_rows(path, expected):
    d=pd.read_csv(require(path),low_memory=False)
    must=['subject','run','method','configuration_id','montage','status','sesame_fs',
          'stimulation_pair','localization_distance_mm','selected_x_mm','selected_y_mm','selected_z_mm',
          'ground_truth_x_mm','ground_truth_y_mm','ground_truth_z_mm']
    missing=[k for k in must if k not in d]
    if missing: raise ValueError('Within grid missing columns: '+str(missing))
    d=d[(d.method=='SESAME')&(d.configuration_id=='sesame_fs_0p12')&
        (d.montage=='all_good')&np.isclose(pd.to_numeric(d.sesame_fs,errors='coerce'),.12)].copy()
    if len(d)!=expected or d[['subject','run']].duplicated().any():
        raise ValueError(f'Within-subject grid selection should have {expected} unique rows; found {len(d)}')
    if not d.status.eq('PASS').all(): raise ValueError('Selected within-subject grid contains failures')
    for column, value in [('sesame_n_parts',100),('sesame_max_n_dips',1),('sesame_lam',.25),('sesame_random_seed',0)]:
        if column in d and not np.allclose(pd.to_numeric(d[column],errors='coerce'),value):
            raise ValueError(f'Within baseline wrong {column}')
    if 'sesame_hyper_q' in d and not d.sesame_hyper_q.astype(str).str.lower().eq('true').all():
        raise ValueError('Wrong baseline sesame_hyper_q')
    return {(row['subject'],row['run']):row for _,row in d.iterrows()}


def regfiles(a,reference,test):
    pair=f'{reference}_to_{test}'
    ad=a.affine_root/pair
    sd=a.syn_root/pair
    base=f'{pair}_syn_'
    aff=ad/f'{pair}_0GenericAffine.mat'
    sa=sd/f'{base}0GenericAffine.mat'
    sw=sd/f'{base}1Warp.nii.gz'
    si=sd/f'{base}1InverseWarp.nii.gz'
    for p in [aff,sa,sw,si,ad/f'{reference}_in_{test}.nii.gz',sd/'donor_syn_in_target.nii.gz']:
        require(p)
    return {'aff':aff,'syn_aff':sa,'warp':sw,'invwarp':si,
            'aff_image':ad/f'{reference}_in_{test}.nii.gz','syn_image':sd/'donor_syn_in_target.nii.gz'}


def terms(f):
    return {
        'aff_t2r':[str(f['aff'])], 'aff_r2t':[f'[{f["aff"]},1]'],
        'syn_t2r':[str(f['warp']),str(f['syn_aff'])],
        'syn_r2t':[f'[{f["syn_aff"]},1]',str(f['invwarp'])]
    }


def image_qa(exe, fixed_path, moving_path, warped_path, t2r_terms, n=200):
    fixed=nib.load(str(require(fixed_path))); moving=nib.load(str(require(moving_path)))
    warped=nib.load(str(require(warped_path)))
    if fixed.shape!=warped.shape or not np.allclose(fixed.affine,warped.affine,atol=.02):
        raise ValueError('Warped image does not share target MRI grid')
    ar=np.asarray(warped.dataobj,dtype=np.float32)
    # deterministic uniform sample among foreground voxels (no ground truth)
    rng=np.random.default_rng(2901)
    nonzero=np.flatnonzero(ar>.1*np.percentile(ar[ar>0],99))
    if len(nonzero)<n: raise ValueError('Too little nonzero MRI foreground for direction QA')
    inds=nonzero[rng.choice(len(nonzero),size=n,replace=False)]
    vox=np.column_stack(np.unravel_index(inds,ar.shape))
    ras=nib.affines.apply_affine(fixed.affine,vox)
    mapped=point_map(exe,ras,t2r_terms)
    srcvox=nib.affines.apply_affine(np.linalg.inv(moving.affine),mapped)
    good=np.all((srcvox>=1)&(srcvox<np.asarray(moving.shape)-2),axis=1)
    if good.sum()<.75*n: raise ValueError(f'MRI QA point coverage too low ({int(good.sum())}/{n})')
    src=np.asarray(moving.dataobj,dtype=np.float32)
    predicted=map_coordinates(src,srcvox[good].T,order=1,mode='constant',cval=0)
    observed=ar.reshape(-1)[inds][good]
    rmse=float(np.sqrt(np.mean((predicted-observed)**2)))
    return {'rmse':rmse,'n_valid':int(good.sum()),'n_tested':n}


def pair_check(exe,a,reference,test,cache):
    key=(reference,test)
    if key in cache:return cache[key]
    f=regfiles(a,reference,test)
    t=terms(f)
    fixed=a.dataset/'derivatives'/'epochs'/test/'anat'/f'{test}_mod-T1w_maskface.nii'
    moving=a.dataset/'derivatives'/'epochs'/reference/'anat'/f'{reference}_mod-T1w_maskface.nii'
    for mode,warped in [('aff',f['aff_image']),('syn',f['syn_image'])]:
        qa=image_qa(exe,fixed,moving,warped,t[f'{mode}_t2r'],a.image_qa_samples)
        if qa['rmse']>a.max_image_rmse:
            raise ValueError(f'{mode} MRI point-direction reproduction RMSE {qa["rmse"]:.5g} exceeds {a.max_image_rmse}')
        f[f'{mode}_qa']=qa
    f['terms']=t
    cache[key]=f
    return f


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as handle:
        w=csv.DictWriter(handle,fieldnames=OUTPUT_COLUMNS,extrasaction='ignore');w.writeheader();w.writerows(rows)


def run():
    a=cli()
    if a.image_qa_samples<50 or a.max_roundtrip_mm<=0: raise ValueError('Invalid QA thresholds')
    manifest=pd.read_csv(require(a.batch_root/'experiment_schedule.csv'))
    if manifest[['target','donor','run']].duplicated().any():raise ValueError('Schedule has duplicate target/reference/run entries')
    if len(manifest)!=a.expect_runs:raise ValueError(f'Expected {a.expect_runs} scheduled jobs; got {len(manifest)}')
    keys=set(zip(manifest.target,manifest.run))
    if len(keys)!=a.expect_runs:raise ValueError('Schedule contains repeated EEG runs or multiple reference models')
    baseline=within_grid_rows(a.within_grid,a.expect_runs)
    if keys!=set(baseline):raise ValueError('Schedule and within-subject grid target/run sets differ')
    all_subjects=sorted(set(manifest.target))
    if len(all_subjects)!=7:raise ValueError('Expected seven test subjects')
    exe=exe_location(a)
    a.output_root.mkdir(parents=True,exist_ok=True)
    rows=[]; cache={}; details=[]
    for _,job in manifest.sort_values(['target','run']).iterrows():
        test,ref,run_id=job['target'],job['donor'],job['run']
        if a.only_test and test!=a.only_test:continue
        key=(test,run_id)
        base=baseline[key]
        row={k:'' for k in OUTPUT_COLUMNS}
        row.update(test_subject=test,reference_subject=ref,run=run_id,
                   stimulation_pair=base['stimulation_pair'],status='FAIL',within_status='PASS',
                   fs=.12,montage='all_good',n_parts=100,max_n_dips=1,lam=.25,hyper_q=True,seed=0,
                   coordinate_assumption='T1w coordinates assumed NIfTI RAS; provisional anatomical mapping',
                   source_json=str(job['result_json']))
        try:
            if test==ref:raise ValueError('Test subject equals reference subject')
            result=read_json(Path(str(job['result_json'])))
            if result.get('status')!='PASS' or result.get('target')!=test or result.get('donor')!=ref:
                raise ValueError('Source result status or subject identifiers do not match')
            if (result.get('ground_truth_used') is not False or result.get('target_forward_used') is not False
                    or result.get('target_electrode_tsv_used') is not False):
                raise ValueError('Reconstruction independence flags invalid')
            if any(result.get('parameters',{}).get(k)!=v for k,v in CFG.items()):
                raise ValueError('Source parameters are not the frozen fs=0.12 settings')
            if result.get('run')!=run_id:raise ValueError('Saved source run identifier mismatch')
            truth=np.array([base[k] for k in ['ground_truth_x_mm','ground_truth_y_mm','ground_truth_z_mm']],float)
            own=np.array([base[k] for k in ['selected_x_mm','selected_y_mm','selected_z_mm']],float)
            baseline_error=float(np.linalg.norm(truth-own))
            if not np.isfinite(baseline_error) or abs(baseline_error-float(base['localization_distance_mm']))>.1:
                raise ValueError('Within-subject grid error does not recompute')
            midpoint=contact_midpoint(a.dataset,test,base['stimulation_pair'])
            contact_delta=float(np.linalg.norm(midpoint-truth))
            if contact_delta>.1:raise ValueError(f'Stimulation contacts disagree with grid ground truth ({contact_delta:.3f} mm)')
            head=np.array(result['peak_coordinate_donor_src_rr_m'],float)*1000
            if head.shape!=(3,) or not np.isfinite(head).all():raise ValueError('Nonfinite source point')
            ref_h5=h5(a.dataset,ref); test_h5=h5(a.dataset,test)
            surface=transform(ref_h5,head)
            ref_t1w=transform(np.linalg.inv(ref_h5),surface)
            pair=pair_check(exe,a,ref,test,cache)
            affine_mri=point_map(exe,ref_t1w,pair['terms']['aff_r2t'])[0]
            syn_mri=point_map(exe,ref_t1w,pair['terms']['syn_r2t'])[0]
            affine_back=point_map(exe,affine_mri,pair['terms']['aff_t2r'])[0]
            syn_back=point_map(exe,syn_mri,pair['terms']['syn_t2r'])[0]
            affine_round=float(np.linalg.norm(affine_back-ref_t1w))
            syn_round=float(np.linalg.norm(syn_back-ref_t1w))
            if max(affine_round,syn_round)>a.max_roundtrip_mm:
                raise ValueError(f'Source round trip exceeded {a.max_roundtrip_mm} mm: affine={affine_round}, syn={syn_round}')
            affine_native=transform(test_h5,affine_mri)
            syn_native=transform(test_h5,syn_mri)
            affine_error=float(np.linalg.norm(affine_native-truth))
            syn_error=float(np.linalg.norm(syn_native-truth))
            row.update(status='PASS_PROVISIONAL',failure_reason='',
                       within_subject_error_mm=baseline_error,cross_affine_error_mm=affine_error,
                       cross_syn_error_mm=syn_error,affine_minus_within_mm=affine_error-baseline_error,
                       syn_minus_within_mm=syn_error-baseline_error,
                       ground_truth_x_mm=truth[0],ground_truth_y_mm=truth[1],ground_truth_z_mm=truth[2],
                       within_x_mm=own[0],within_y_mm=own[1],within_z_mm=own[2],
                       reference_head_x_mm=head[0],reference_head_y_mm=head[1],reference_head_z_mm=head[2],
                       reference_surface_x_mm=surface[0],reference_surface_y_mm=surface[1],reference_surface_z_mm=surface[2],
                       affine_test_x_mm=affine_native[0],affine_test_y_mm=affine_native[1],affine_test_z_mm=affine_native[2],
                       syn_test_x_mm=syn_native[0],syn_test_y_mm=syn_native[1],syn_test_z_mm=syn_native[2],
                       affine_roundtrip_mm=affine_round,syn_roundtrip_mm=syn_round,
                       within_ground_truth_contact_discrepancy_mm=contact_delta,
                       reference_registration_image_rmse_affine=pair['aff_qa']['rmse'],
                       reference_registration_image_rmse_syn=pair['syn_qa']['rmse'],
                       affine_registration=str(pair['aff']),syn_registration=str(pair['syn_aff']))
        except Exception as exc:
            row['failure_reason']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
        print(f'{test} <- {ref} {run_id}: {row["status"]}' + (f' | {row["failure_reason"]}' if row['failure_reason'] else ''),flush=True)
        write_csv(a.output_root/'cross_subject_localization_errors.csv',rows)
    statuses=Counter(x['status'] for x in rows)
    ok=[r for r in rows if r['status']=='PASS_PROVISIONAL']
    metrics={}
    for col in ('within_subject_error_mm','cross_affine_error_mm','cross_syn_error_mm',
                'affine_minus_within_mm','syn_minus_within_mm'):
        arr=np.asarray([r[col] for r in ok],float)
        metrics[col]={'n':int(len(arr)),'median':float(np.median(arr)) if len(arr) else None,
                      'mean':float(np.mean(arr)) if len(arr) else None,
                      'sd':float(np.std(arr,ddof=1)) if len(arr)>1 else None}
    summary={'status':'PROVISIONAL_REGISTRATION_DEPENDENT',
             'created_utc':datetime.now(timezone.utc).isoformat(),
             'scheduled':int(len(rows)),'pass':len(ok),'fail':len(rows)-len(ok),
             'statuses':dict(statuses),'metrics_complete_case_only':metrics,
             'registration_pair_image_qa':{f'{ref}_to_{test}':
                 {'affine':v['aff_qa'],'syn':v['syn_qa']} for (ref,test),v in cache.items()},
             'inputs':{'schedule':str(a.batch_root/'experiment_schedule.csv'),'within_grid':str(a.within_grid)},
             'limitations':['RAS interpretation of dataset native T1w still provisional',
                            'Image reproduction and point round-trip test implementation, not true cortical correspondence',
                            'Registration-dependent cross-subject errors are not intrinsic SESAME errors',
                            'SESAME within-subject tuning included all seven subjects (not nested held-out CV)',
                            'Complete-case stats exclude failed evaluations; preserve failed rows in CSV',
                            'Neither source localization errors nor stimulation sites were used to fit registrations']}
    (a.output_root/'localization_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    pd.DataFrame([{'test_subject':test,'reference_subject':ref, 'n':len(group),
                   'pass':sum(x['status']=='PASS_PROVISIONAL' for x in group),
                   'median_affine_mm':float(np.median([x['cross_affine_error_mm'] for x in group if x['status']=='PASS_PROVISIONAL'])) if any(x['status']=='PASS_PROVISIONAL' for x in group) else np.nan,
                   'median_syn_mm':float(np.median([x['cross_syn_error_mm'] for x in group if x['status']=='PASS_PROVISIONAL'])) if any(x['status']=='PASS_PROVISIONAL' for x in group) else np.nan}
                 for (test,ref), group in ((k,[r for r in rows if (r['test_subject'],r['reference_subject'])==k]) for k in sorted(set((r['test_subject'],r['reference_subject']) for r in rows)))])\
        .to_csv(a.output_root/'per_test_subject_summary.csv',index=False)
    print('\nCompleted:',len(ok),'/',len(rows),'provisional paired evaluations')
    print('Output:',a.output_root/'cross_subject_localization_errors.csv')
    print('Summary:',a.output_root/'localization_summary.json')
    if len(ok)!=len(rows):raise SystemExit(1)

if __name__=='__main__':run()
