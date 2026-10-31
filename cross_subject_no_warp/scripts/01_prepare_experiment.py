#!/usr/bin/env python3
# SCRIPT 01 — Prepare LOSO single-reference experiment without brain warping
#
# PURPOSE
# Prepare the experiment schedule and audit existing coordinate transforms.
# This script does NOT select inverse-method parameters, run localization,
# calculate localization errors, or perform brain registration.
#
# TERMINAL COMMAND — run from the project root
# python cross_subject_no_warp/scripts/01_prepare_experiment.py
#
# INPUT DATA
# Dataset root:
#   ~/Documents/DATASETS SOURCE LOCALIZATION/Localize-MI/
#
# Existing schedule:
#   cross_subject/outputs/30_all_reference_sesame_batch/
#       experiment_schedule_all_references.csv
# Only participant/run identifiers are reused from this schedule.
# Previous localization errors and selected parameters are not used.
#
# For each participant, the coordinate audit reads:
#   derivatives/sourcemodelling/sub-XX/fwd/sub-XX_fwd.fif
#   derivatives/sourcemodelling/sub-XX/xfm/
#       sub-XX_from-head_to-surface.h5
#   derivatives/epochs/sub-XX/ieeg/
#       sub-XX_task-seegstim_space-surface_electrodes.tsv
#
# EXPERIMENT DESIGN
# Participants: sub-01 through sub-07; 61 EEG runs in total.
# LOSO: hold out one participant; the remaining six form the training set.
# Primary reference: sub-01.
# Fallback reference: sub-02, used only when sub-01 is the test participant.
#
# Two planned conditions per method and test run:
#   1. Within-subject: test EEG + test participant's head model.
#   2. Single-reference: test EEG + reference participant's head model.
# Both conditions must use the same training-only LOSO-selected parameters.
#
# METHODS
# MNE, dSPM, sLORETA, eLORETA, Continuous-ECD, RAP-MUSIC,
# LCMV, MxNE, irMxNE, Gamma-MAP, Champagne, SESAME.
#
# PLANNED RECONSTRUCTIONS
# 12 methods x 61 runs x 2 conditions = 1,464 fits.
# These fits are scheduled here, not executed.
#
# COORDINATE POLICY
# No inter-brain affine registration, SyN, scaling, or warping.
# Existing rigid within-participant head-to-surface transforms are allowed.
# Later evaluation will compare estimated and true locations in corresponding
# head frames, assuming corresponding origins and axes across participants.
#
# AUDIT SETTINGS
# Rigidity tolerance: 1e-5 for rotation orthogonality and determinant.
# H5 versus forward-transform comparison: atol=1e-5, rtol=0.
# Active source coordinates must be finite and within 1 metre of the origin.
# Coordinate round-trip error is reported in millimetres.
# PASS_NUMERICAL does not independently verify anatomical fiducials or
# stimulation-contact units.
#
# PARAMETERS NOT SELECTED HERE
# Montage, regularization, depth, loose orientation, and method-specific
# parameters remain pending Script 02.
# Script 02 must exclude the test participant BEFORE configuration filtering
# and ranking; full-dataset all-pass selections must not define eligibility.
#
# OUTPUT DIRECTORY
#   cross_subject_no_warp/outputs/01_prepare_experiment/
#
# OUTPUT FILES
#   01_loso_folds.csv
#   02_fold_membership.csv
#   03_single_reference_cases.csv
#   04_reconstruction_plan.csv
#   05_coordinate_audit.csv
#   06_coordinate_transforms.json
#   07_experiment_manifest.json
#   README.txt
#
# OPTIONAL COMMANDS
# Prepare the schedule without auditing dataset files:
# python cross_subject_no_warp/scripts/01_prepare_experiment.py --plan-only
#
# Regenerate existing preparation outputs:
# python cross_subject_no_warp/scripts/01_prepare_experiment.py --overwrite
#
# Override the dataset location:
# python cross_subject_no_warp/scripts/01_prepare_experiment.py \
#     --dataset "/path/to/Localize-MI"
#
# BEFORE LOCALIZATION
# Inspect the coordinate audit.
# Decide whether to retain the previous known-hemisphere peak restriction
# or use whole-brain peak selection; document this before Script 03.

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC',
         'LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
SUBJECTS=[f'sub-{i:02d}' for i in range(1,8)]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_schedule(path):
    d=pd.read_csv(path,dtype=str)
    if {'test_subject','reference_subject','run'}.issubset(d):
        d=d[['test_subject','reference_subject','run']]
    elif {'target','donor','run'}.issubset(d):
        d=d[['target','donor','run']].rename(columns={'target':'test_subject','donor':'reference_subject'})
    else:raise ValueError('Schedule needs test_subject/reference_subject/run or target/donor/run')
    if d.isna().any().any():raise ValueError('Missing schedule identifiers')
    for c in d:d[c]=d[c].str.strip()
    if d.duplicated().any():raise ValueError('Duplicate schedule rows')
    if len(d)!=366 or set(d.test_subject)!=set(SUBJECTS):raise ValueError('Expected 366 cases across sub-01 through sub-07')
    counts=d.groupby(['test_subject','run']).size()
    if len(counts)!=61 or not counts.eq(6).all():raise ValueError('Expected 61 runs, each with six references')
    for (test,run),g in d.groupby(['test_subject','run']):
        if set(g.reference_subject)!=set(SUBJECTS)-{test}:raise ValueError(f'Invalid reference set: {test}/{run}')
    return d.sort_values(['test_subject','run','reference_subject']).reset_index(drop=True)


def make_plan(schedule,primary,fallback):
    if primary==fallback or primary not in SUBJECTS or fallback not in SUBJECTS:
        raise ValueError('Primary and fallback must be different dataset participants')
    folds=[];cases=[];membership=[]
    for test in SUBJECTS:
        ref=fallback if test==primary else primary
        training=[s for s in SUBJECTS if s!=test]
        runs=sorted(schedule.loc[schedule.test_subject.eq(test),'run'].unique())
        folds.append(dict(fold_id=test,test_subject=test,reference_subject=ref,
                          training_subjects=';'.join(training),n_training_subjects=6,n_test_runs=len(runs)))
        membership.extend(dict(fold_id=test,subject=s,role='test' if s==test else 'training',
                               is_reference=s==ref) for s in SUBJECTS)
        for run in runs:cases.append(dict(fold_id=test,test_subject=test,reference_subject=ref,run=run))
    jobs=[]
    for case in cases:
        for method in METHODS:
            for condition in ['within','single_reference_no_warp']:
                model=case['test_subject'] if condition=='within' else case['reference_subject']
                jobs.append(dict(**case,method=method,condition=condition,model_subject=model,
                                 parameter_fold=case['fold_id'],parameter_status='PENDING_SCRIPT_02'))
    return pd.DataFrame(folds),pd.DataFrame(membership),pd.DataFrame(cases),pd.DataFrame(jobs)


def check_rigid(matrix,tol=1e-5):
    m=np.asarray(matrix,float)
    if m.shape!=(4,4) or not np.isfinite(m).all():raise ValueError('Transform must be a finite 4x4 matrix')
    if not np.allclose(m[3],[0,0,0,1],atol=tol,rtol=0):raise ValueError('Invalid homogeneous bottom row')
    r=m[:3,:3];orth=float(np.max(np.abs(r.T@r-np.eye(3))));det=float(np.linalg.det(r))
    if orth>tol or abs(det-1)>tol:raise ValueError(f'Not a proper rigid transform: orthogonality error={orth:g}, determinant={det:g}')
    return orth,det


def apply(matrix,points):
    points=np.asarray(points,float)
    return points@matrix[:3,:3].T+matrix[:3,3]


def audit_subject(dataset,subject):
    row=dict(subject=subject,status='FAIL',failure_reason='')
    model=dataset/'derivatives'/'sourcemodelling'/subject
    fwd_path=model/'fwd'/f'{subject}_fwd.fif'
    xfm_path=model/'xfm'/f'{subject}_from-head_to-surface.h5'
    contacts=dataset/'derivatives'/'epochs'/subject/'ieeg'/f'{subject}_task-seegstim_space-surface_electrodes.tsv'
    row.update(forward_path=str(fwd_path),head_to_surface_path=str(xfm_path),contacts_path=str(contacts))
    matrices=None
    try:
        for path in [fwd_path,xfm_path,contacts]:
            if not path.is_file():raise FileNotFoundError(path)
        import h5py
        import mne
        from mne.io.constants import FIFF
        with h5py.File(xfm_path,'r') as f:m=np.asarray(f['trans'],float)
        orth,det=check_rigid(m)
        inverse=np.linalg.inv(m)
        fwd=mne.read_forward_solution(str(fwd_path),verbose=False)
        t=fwd['mri_head_t'];fr=int(t['from']);to=int(t['to'])
        if (fr,to)==(int(FIFF.FIFFV_COORD_MRI),int(FIFF.FIFFV_COORD_HEAD)):
            fwd_surface_to_head=np.asarray(t['trans'],float)
        elif (fr,to)==(int(FIFF.FIFFV_COORD_HEAD),int(FIFF.FIFFV_COORD_MRI)):
            fwd_surface_to_head=np.linalg.inv(t['trans'])
        else:raise ValueError(f'Unexpected forward MRI/head transform frames {fr}->{to}')
        check_rigid(fwd_surface_to_head)
        difference=float(np.max(np.abs(inverse-fwd_surface_to_head)))
        row['h5_vs_forward_transform_max_abs']=difference
        if not np.allclose(inverse,fwd_surface_to_head,atol=1e-5,rtol=0):
            raise ValueError('H5 surface->head and forward MRI->head transforms differ; verify surface coordinate definition before proceeding')
        frames=[];active_points=[]
        for src in fwd['src']:
            frame=int(src['coord_frame']);frames.append(frame)
            xyz=np.asarray(src['rr'],float)[np.asarray(src['vertno'],int)]
            if frame==int(FIFF.FIFFV_COORD_MRI):xyz=apply(fwd_surface_to_head,xyz)
            elif frame!=int(FIFF.FIFFV_COORD_HEAD):raise ValueError(f'Unsupported source frame: {frame}')
            active_points.append(xyz)
        points=np.concatenate(active_points)
        if not len(points) or not np.isfinite(points).all():raise ValueError('No finite active source coordinates')
        max_radius=float(np.linalg.norm(points,axis=1).max())
        if max_radius>1:raise ValueError('Source coordinates exceed one metre; inspect units/frame')
        roundtrip=float(np.linalg.norm(apply(inverse,apply(m,points))-points,axis=1).max()*1000)
        contacts_table=pd.read_csv(contacts,sep='\t')
        if not {'name','x','y','z'}.issubset(contacts_table):raise ValueError('Contacts TSV lacks name/x/y/z')
        # Do not read error outcomes or use contact positions to select references.
        channel_names=fwd['info']['ch_names']
        if len(channel_names)!=len(set(channel_names)):raise ValueError('Duplicate forward channel names')
        row.update(status='PASS_NUMERICAL',rotation_orthogonality_error=orth,rotation_determinant=det,
                   roundtrip_max_mm=roundtrip,source_frame_codes=';'.join(map(str,frames)),
                   source_max_head_radius_m=max_radius,n_sources=len(points),n_forward_channels=len(channel_names),
                   h5_sha256=sha256(xfm_path),forward_sha256=sha256(fwd_path),contacts_sha256=sha256(contacts),
                   coordinate_convention='MNE head frame; verify dataset anatomical fiducial definition separately',
                   contact_units='metres assumed from existing Script 32; confirm metadata in evaluation')
        matrices=dict(head_to_surface_m=m.tolist(),surface_to_head_m=inverse.tolist(),
                      forward_source_frame_codes=frames,forward_channels=channel_names)
    except Exception as exc:row['failure_reason']=f'{type(exc).__name__}: {exc}'
    return row,matrices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--schedule',type=Path,default=Path('cross_subject/outputs/30_all_reference_sesame_batch/experiment_schedule_all_references.csv'))
    p.add_argument('--output-root',type=Path,default=Path('cross_subject_no_warp/outputs/01_prepare_experiment'))
    p.add_argument('--reference',default='sub-01',choices=SUBJECTS)
    p.add_argument('--fallback-reference',default='sub-02',choices=SUBJECTS)
    p.add_argument('--plan-only',action='store_true',help='Write plan without dataset/coordinate audit; not ready for reconstruction')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();a.dataset=a.dataset.expanduser();root=a.output_root
    if root.exists() and any(root.iterdir()) and not a.overwrite:raise FileExistsError('Output folder is nonempty; use --overwrite only to regenerate this preparation')
    schedule=read_schedule(a.schedule)
    folds,members,cases,jobs=make_plan(schedule,a.reference,a.fallback_reference)
    rows=[];transforms={}
    for subject in SUBJECTS:
        if a.plan_only:row=dict(subject=subject,status='NOT_CHECKED',failure_reason='Plan-only mode');matrix=None
        else:
            print(f'Auditing {subject} ...',flush=True)
            row,matrix=audit_subject(a.dataset,subject)
            print(f'  {row["status"]}: {row["failure_reason"]}',flush=True)
        rows.append(row)
        if matrix is not None:transforms[subject]=matrix
    audit=pd.DataFrame(rows)
    root.mkdir(parents=True,exist_ok=True)
    folds.to_csv(root/'01_loso_folds.csv',index=False)
    members.to_csv(root/'02_fold_membership.csv',index=False)
    cases.to_csv(root/'03_single_reference_cases.csv',index=False)
    jobs.to_csv(root/'04_reconstruction_plan.csv',index=False)
    audit.to_csv(root/'05_coordinate_audit.csv',index=False)
    (root/'06_coordinate_transforms.json').write_text(json.dumps(transforms,indent=2)+'\n')
    audited=bool(audit.status.eq('PASS_NUMERICAL').all())
    manifest=dict(version=1,experiment='LOSO single reference; no inter-brain registration or warping',
                  dataset=str(a.dataset.resolve()),schedule=str(a.schedule.resolve()),schedule_sha256=sha256(a.schedule),
                  preparation_script_sha256=sha256(__file__),methods=METHODS,subjects=SUBJECTS,
                  primary_reference=a.reference,fallback_reference=a.fallback_reference,
                  n_test_runs=len(cases),n_folds=len(folds),n_reconstruction_jobs=len(jobs),
                  rigid_coordinate_audit_passed=audited,parameters_selected=False,
                  brain_registration=False,warping=False,scaling=False,
                  evaluation_space='direct corresponding MNE head frames, metres converted to mm for distance',
                  parameter_selection='Script 02 must use training-subject grid rows only; same fold parameters for within and cross',
                  source_selection_policy='UNRESOLVED: verify prior hemisphere restriction before implementing Script 03',
                  readiness='PREPARATION_ONLY: numerical audit does not establish anatomical correspondence')
    (root/'07_experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'README.txt').write_text('''Script 01 prepares 7 LOSO folds, 61 single-reference cases, and 1464 intended fits (12 methods x 61 runs x 2 conditions).
The primary reference is sub-01 by default; sub-02 replaces it only for test sub-01. Overrides are recorded in the manifest.
Only subject/run identifiers are reused from the old schedule. Prior selected parameters and error values are not used.
All selection in Script 02 must exclude the held-out subject BEFORE checking configuration completeness, failures, or ranking. Do not prefilter candidates using the full-data all-pass selections.
Both conditions in a fold must use the same training-only selected parameters. New parameters require new fits unless all computational settings match a validated old reconstruction.
No inter-brain affine/SyN transforms, scaling or warping are loaded. A rigid within-subject head<->surface coordinate conversion is allowed.
Audit checks H5 rigidity, agreement with forward MRI/head transforms, active source frames/units, and coordinate roundtrip. PASS_NUMERICAL is not proof of correct anatomical fiducials or contact-space units.
Known stimulation locations will be converted from test-native surface to test-head using the inverse rigid transform at evaluation. Direct head-frame distances assume corresponding origins/axes across subjects.
Reference geometry is used for cross-subject localization; test anatomy is used only for the within baseline and evaluation. Do not pass ground-truth coordinates to the inverse solver.
IMPORTANT: prior scripts restrict peak selection to the known stimulation hemisphere. Decide and document whether to retain this informed restriction or use whole-brain peaks before Script 03. Existing hemisphere-restricted grid errors cannot silently serve as whole-brain LOSO-selection errors.
Plan-only outputs do not verify the dataset. A failed audit exits nonzero after saving its report. Inspect 05_coordinate_audit.csv before proceeding.
Next folders: outputs/02_select_loso_parameters, outputs/03_run_localization, outputs/04_evaluate_errors, outputs/05_compare_results.
''')
    print(f'Prepared {len(folds)} folds; {len(cases)} test runs; {len(jobs)} planned fits. Output: {root}')
    if a.plan_only:print('PLAN ONLY: coordinate audit not run.')
    elif not audited:raise SystemExit('Coordinate audit failed; inspect 05_coordinate_audit.csv. No solver was run.')
    else:print('Numerical coordinate audit passed. Parameters remain pending Script 02.')

if __name__=='__main__':main()
