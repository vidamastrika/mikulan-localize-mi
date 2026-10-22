#!/usr/bin/env python3
"""SCRIPT 28 — ONE-REFERENCE-PER-TEST SUBJECT MRI REGISTRATION BATCH.

FROZEN SESAME CONTEXT (RECONSTRUCTION IS **NOT** PERFORMED HERE)
----------------------------------------------------------------
Selected using within-subject 61/61 PASS grid, lowest median error:
    fs=0.12; montage=all good EEG channels; n_parts=100; max_n_dips=1;
    lam=0.25; hyper_q=True; random_seed=0; noise covariance=None;
    target window=-2 to +2 ms. Source reconstruction belongs to Script 27.

TERMINOLOGY: test subject supplies EEG and ground truth; reference subject
supplies the head/forward model. ANTs 'fixed' = test MRI, 'moving' =
reference MRI. This script registers images only, never reads stimulation
locations, SESAME results or localization errors, and never selects
transforms based on localization accuracy.

DEFAULT IS DRY RUN. With --execute, register reference->test rigid+affine
and rigid+affine+SyN. Reuse existing sub-01->sub-02 outputs when valid.
This batch is sequential to prevent large SyN memory contention. It writes
registration_schedule.csv and registration_status.csv under
cross_subject/outputs/28_batch_registration/. Per-pair transform files stay
in the original 09_anatomical_registration and 19_syn_registration trees,
compatible with Scripts 16–21. The image QA is a THRESHOLDED FOREGROUND
proxy, NOT a brain Dice score or validation of cortical correspondence.

Usage:
  python cross_subject/scripts/28_batch_reference_test_registration.py
  python cross_subject/scripts/28_batch_reference_test_registration.py --execute --resume
  python cross_subject/scripts/28_batch_reference_test_registration.py --execute --resume --only-test sub-03
  python cross_subject/scripts/28_batch_reference_test_registration.py --execute --resume --only-test sub-03 --skip-affine

ANTs requires antsRegistration on PATH. E.g. run through micromamba:
  micromamba run -n localizemi-registration .venv/bin/python ... --execute --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SUBJECTS = [f'sub-{i:02d}' for i in range(1, 8)]
PAIRS = [(test, f'sub-{(int(test[-2:])-2)%7+1:02d}') for test in SUBJECTS]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def image_ok(file, fixed):
    if not file.is_file() or file.stat().st_size < 1000:
        return False
    try:
        import nibabel as nib
        import numpy as np
        a = nib.load(str(file)); b = nib.load(str(fixed))
        return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=.02)
    except Exception:
        return False


def aff_paths(root, reference, test):
    d = root / f'{reference}_to_{test}'
    prefix = d / f'{reference}_to_{test}_'
    return dict(dir=d, prefix=prefix, warped=d / f'{reference}_in_{test}.nii.gz',
                affine=Path(f'{prefix}0GenericAffine.mat'), log=d/'ants_registration.log')


def syn_paths(root, reference, test):
    d = root / f'{reference}_to_{test}'
    prefix = d / f'{reference}_to_{test}_syn_'
    return dict(dir=d, prefix=prefix, warped=d/'donor_syn_in_target.nii.gz',
                inverse_warped=d/'target_syn_in_donor.nii.gz',
                affine=Path(f'{prefix}0GenericAffine.mat'),
                warp=Path(f'{prefix}1Warp.nii.gz'), inverse_warp=Path(f'{prefix}1InverseWarp.nii.gz'),
                quality=d/'syn_registration_quality.json', log=d/'ants_registration.log')


def is_aff_done(p, fixed):
    return (p['affine'].is_file() and p['affine'].stat().st_size > 50 and
            image_ok(p['warped'], fixed))


def is_syn_done(p, fixed):
    keys = ('affine', 'warp', 'inverse_warp', 'quality')
    if not all(p[k].is_file() and p[k].stat().st_size > 50 for k in keys):
        return False
    if not image_ok(p['warped'], fixed):
        return False
    try:
        d = json.loads(p['quality'].read_text())
        return d.get('status') == 'MRI_ONLY_QA_POINT_TRANSFORMS_NOT_VALIDATED'
    except Exception:
        return False


def ants_bin(name, env):
    p = shutil.which(name)
    if p:
        return p
    candidate = Path.home()/'micromamba'/'envs'/env/'bin'/name
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError(f'{name} unavailable. Run with: micromamba run -n {env} .venv/bin/python <script> --execute --resume')


def run_ants(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as f:
        f.write('COMMAND: ' + ' '.join(map(str, cmd)) + '\n\n')
        f.flush()
        proc = subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'ANTs exited {proc.returncode}; see {log}')


def run_affine(ants, fixed, moving, p):
    p['dir'].mkdir(parents=True, exist_ok=True)
    cmd = [ants, '--dimensionality','3','--float','1','--verbose','1',
           '--collapse-output-transforms','1',
           '--output',f"[{p['prefix']},{p['warped']}]",
           '--interpolation','Linear','--use-histogram-matching','0',
           '--winsorize-image-intensities','[0.005,0.995]',
           '--initial-moving-transform',f'[{fixed},{moving},1]']
    for method in ('Rigid', 'Affine'):
        cmd += ['--transform',f'{method}[0.1]',
                '--metric',f'MI[{fixed},{moving},1,32,Regular,0.25]',
                '--convergence','[1000x500x250x0,1e-6,10]',
                '--shrink-factors','8x4x2x1', '--smoothing-sigmas','3x2x1x0vox']
    run_ants(cmd, p['log'])
    if not is_aff_done(p, fixed):
        raise RuntimeError(f'Affine output incomplete or on wrong grid: {p["dir"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', type=Path, default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    parser.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/28_batch_registration'))
    parser.add_argument('--affine-root', type=Path, default=Path('cross_subject/outputs/09_anatomical_registration'))
    parser.add_argument('--syn-root', type=Path, default=Path('cross_subject/outputs/19_syn_registration'))
    parser.add_argument('--syn-script', type=Path, default=Path(__file__).with_name('19_run_syn_registration.py'))
    parser.add_argument('--affine-qa-script', type=Path, default=Path(__file__).with_name('18_evaluate_affine_registration_quality.py'))
    parser.add_argument('--micromamba-env', default='localizemi-registration')
    parser.add_argument('--execute', action='store_true', help='Without this flag only write schedule and existing-status report')
    parser.add_argument('--resume', action='store_true', help='Skip complete verified registration outputs')
    parser.add_argument('--only-test', choices=SUBJECTS)
    parser.add_argument('--skip-affine', action='store_true', help='Only SyN; affine output for missing pairs will remain unavailable')
    parser.add_argument('--force', action='store_true', help='Allow regeneration/overwrite of preexisting outputs')
    args = parser.parse_args()
    if args.resume and args.force:
        parser.error('--resume and --force are mutually exclusive')
    if args.execute and not args.syn_script.is_file():
        parser.error(f'Missing Script 19: {args.syn_script}')
    rows = []
    for test, reference in PAIRS:
        fixed = args.dataset_root/'derivatives'/'epochs'/test/'anat'/f'{test}_mod-T1w_maskface.nii'
        moving = args.dataset_root/'derivatives'/'epochs'/reference/'anat'/f'{reference}_mod-T1w_maskface.nii'
        if not fixed.is_file() or not moving.is_file():
            parser.error(f'Missing MRI(s) for {reference}->{test}: {moving}, {fixed}')
        ap = aff_paths(args.affine_root, reference, test)
        sp = syn_paths(args.syn_root, reference, test)
        rows.append({'test_subject':test,'reference_subject':reference,
                     'fixed_mri':str(fixed),'moving_mri':str(moving),
                     'affine_dir':str(ap['dir']),'syn_dir':str(sp['dir']),
                     'affine_already_complete':is_aff_done(ap,fixed),
                     'syn_already_complete':is_syn_done(sp,fixed),
                     'affine_status':'NOT_RUN', 'syn_status':'NOT_RUN',
                     'affine_seconds':'','syn_seconds':'', 'error':''})
    args.output_root.mkdir(parents=True, exist_ok=True)
    sched = args.output_root/'registration_schedule.csv'
    status = args.output_root/'registration_status.csv'
    write_csv(sched, rows, ['test_subject','reference_subject','fixed_mri','moving_mri','affine_dir','syn_dir','affine_already_complete','syn_already_complete'])
    columns = ['test_subject','reference_subject','affine_status','syn_status','affine_seconds','syn_seconds','error','affine_dir','syn_dir']
    if not args.execute:
        write_csv(status, rows, columns)
        print('DRY RUN — NO registrations executed. Schedule:',sched)
        for r in rows:
            print(f" {r['reference_subject']} -> {r['test_subject']}: affine={'ready' if r['affine_already_complete'] else 'missing'}, SyN={'ready' if r['syn_already_complete'] else 'missing'}")
        return
    ants = None
    for r in rows:
        test, reference = r['test_subject'], r['reference_subject']
        if args.only_test and test != args.only_test:
            r['affine_status']=r['syn_status']='FILTERED'
            continue
        fixed, moving = Path(r['fixed_mri']), Path(r['moving_mri'])
        ap, sp = aff_paths(args.affine_root,reference,test), syn_paths(args.syn_root,reference,test)
        print(f'\nREFERENCE {reference} -> TEST {test}',flush=True)
        for kind, p, done in [('affine',ap,is_aff_done(ap,fixed)), ('syn',sp,is_syn_done(sp,fixed))]:
            if kind == 'affine' and args.skip_affine:
                r[kind+'_status']='NOT_REQUESTED'
                continue
            existing = any(p['dir'].glob('*')) if p['dir'].exists() else False
            if done and args.resume:
                r[kind+'_status']='SKIPPED_COMPLETE'
                print(f'  {kind}: reused validated outputs',flush=True)
                continue
            if existing and not args.force and not done:
                r[kind+'_status']='INCOMPLETE_EXISTING'
                r['error'] += f'{kind}: existing partial/invalid outputs; inspect or explicitly use --force; '
                print('  WARNING: incomplete existing output; skipping overwrite',flush=True)
                continue
            if done and not args.force:
                r[kind+'_status']='EXISTS_USE_RESUME'
                r['error'] += f'{kind}: valid result already exists; use --resume; '
                continue
            t0=time.monotonic()
            try:
                if kind == 'affine':
                    ants = ants or ants_bin('antsRegistration',args.micromamba_env)
                    run_affine(ants,fixed,moving,p)
                    if args.affine_qa_script.is_file():
                        subprocess.run([sys.executable,str(args.affine_qa_script), '--donor',reference,'--target',test,
                                        '--data-root',str(args.dataset_root), '--registration-root',str(args.affine_root)],
                                       check=True)
                else:
                    cmd=[sys.executable,str(args.syn_script),'--donor',reference,'--target',test,
                         '--dataset-root',str(args.dataset_root),'--output-root',str(args.syn_root),
                         '--micromamba-env',args.micromamba_env]
                    subprocess.run(cmd,check=True)
                    if not is_syn_done(p,fixed):
                        raise RuntimeError('SyN output validation failed')
                r[kind+'_status']='PASS'
                print(f'  {kind}: PASS',flush=True)
            except Exception as e:
                r[kind+'_status']='FAIL'
                r['error'] += f'{kind}: {type(e).__name__}: {e}; '
                print(f'  {kind}: FAIL: {e}',flush=True)
            finally:
                r[kind+'_seconds']=round(time.monotonic()-t0,2)
                write_csv(status,rows,columns)
    write_csv(status,rows,columns)
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'executed':True,
             'registration_only':True,'reference_test_pairs':7,
             'statuses':[{k:r[k] for k in columns} for r in rows],
             'limitations':['No ground-truth or EEG used for registration.',
                            'Foreground overlap is NOT cortical/brain Dice.',
                            'SyN point mapping remains separately validated in the next evaluation.',
                            'Native T1w RAS frame convention remains provisional.']}
    (args.output_root/'batch_registration_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('\nRegistration batch finished. Status:',status)

if __name__ == '__main__':
    main()
