#!/usr/bin/env python3
# RESEARCH OBJECTIVE
# Audit the completed template/fsaverage reconstruction and describe numerical
# separation from independently audited stimulation coordinates. This is NOT
# anatomically validated localization error: template HEAD and individual HEAD
# have no validated anatomical correspondence. All 7 subjects share one model.
# ASSUMPTIONS / SCOPE
# eN->EN remains ASSUMED_UNVERIFIED. No inverse, registration, anatomy fitting,
# montage optimization, parameter adjustment, retries or QA-threshold changes.
# Subject MRI/digitization are never introduced into reconstruction. Existing
# individual ground-truth coordinates are evaluation-only diagnostic references.
# INPUTS (read-only)
# Script 03 status/full plan/frozen manifest; every frozen input's SHA-256;
# audited no-warp 61-run coordinate CSV and its ground-truth dataset manifest;
# raw contact/HEAD-to-surface files are hash-checked only (not used for fitting);
# previous fsaverage registration-status CSV, used only to report its status.
# OUTPUTS (fresh directory strictly within this variant/outputs)
# 01_run_level_diagnostics.csv: all 5,124 jobs, diagnostic distance on PASS only;
# localization_error_mm always missing, localization_error_valid always False.
# 02_ground_truth_diagnostic_reference.csv: 61 original audited reference rows.
# 03_method_summary.csv; 04_subject_summary.csv; 05_reference_summary.csv;
# 06_equal_subject_summary.csv; 07_run_reference_summary.csv;
# 08_unsuccessful_jobs.csv; 09_input_integrity_inventory.csv;
# 10_anatomical_evaluation_readiness.csv; 11_sub03_summary.csv;
# 12_evaluation_manifest.json; README.md; exact script snapshot.
# DESIGN / VALIDATION
# 7 subjects, 61 runs, 12 methods; 732 LOSO + 4,392 Single Reference = 5,124.
# Require completed status ledger, exact frozen plan/config keys, finite PASS
# estimates, matching pair/hemisphere, and consistent template HEAD/fsaverage MRI
# coordinates. All failed jobs remain in denominators with missing distances.
# Single Reference's 6 references are repeated evaluations, not 366 unique runs.
# Report pooled jobs, per-subject, per-reference and equal-subject summaries;
# retain all subjects including sub-03, with an additional explicit sub-03 table.
# No special failed individual-coregistration label applies to new sub-03 jobs.
# Unregistered Euclidean diagnostic = ||template HEAD estimate - individual HEAD
# ground truth|| in mm. It cannot support claims of anatomical accuracy.
# PARAMETERS / DEPENDENCIES
# --reconstruction-root, --ground-truth, --ground-truth-manifest,
# --registration-status, --output-root override paths. No overwrite/resume.
# Python >=3.10, NumPy, pandas. No MNE/ANTs imports, EEG array loading or figures.
# All frozen files (including EEG arrays) are streamed for SHA verification only.
# A shared writer lock prevents concurrent changes by Script 03 during audit.
# COMPLETE COMMAND (repository root; user executes manually):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/04_evaluate_diagnostics.py
# Optional fresh output:
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/04_evaluate_diagnostics.py \
#   --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/04_evaluate_diagnostics_v2
# Exit 0: diagnostic audit complete, anatomical error evaluation still BLOCKED.
# Exit 1: input/provenance/completeness/coordinate validation failure; no fake PASS.
"""Audit completed template reconstructions; report diagnostic distances only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

VARIANT=Path(__file__).resolve().parents[1]
ROOT=VARIANT.parents[1]
KEY=['strategy','method','reference_subject','test_subject','run']
XYZ=[f'estimated_head_{a}_mm' for a in 'xyz']
MRI=[f'estimated_fsaverage_mri_{a}_mm' for a in 'xyz']
GT=[f'gt_head_{a}_mm' for a in 'xyz']
METHODS={'MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC',
         'LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME'}
SUBJECTS={f'sub-{i:02d}' for i in range(1,8)}


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def summarize(data,groups):
    rows=[]
    for values,g in data.groupby(groups,dropna=False,sort=True):
        if not isinstance(values,tuple):values=(values,)
        d=g.diagnostic_distance_mm.dropna()
        rows.append(dict(zip(groups,values))|dict(
            n_total=len(g),n_pass=int(g.status.eq('PASS').sum()),n_fail=int(g.status.eq('FAIL').sum()),
            n_distances=len(d),n_unique_runs=len(g[['test_subject','run']].drop_duplicates()),
            pass_rate=float(g.status.eq('PASS').mean()),
            diagnostic_mean_mm=d.mean(),diagnostic_median_mm=d.median(),
            diagnostic_q25_mm=d.quantile(.25),diagnostic_q75_mm=d.quantile(.75),
            diagnostic_maximum_mm=d.max()))
    return pd.DataFrame(rows)


def evaluate(args):
    reconstruction=args.reconstruction_root.expanduser().resolve()
    out=args.output_root.expanduser().resolve()
    require((VARIANT/'outputs').resolve() in out.parents,'Outputs must stay within this variant/outputs')
    require(VARIANT in reconstruction.parents,'Reconstruction must belong to this variant')
    require(not out.exists() and reconstruction not in out.parents and out not in reconstruction.parents,
            'Use a fresh output directory separate from reconstruction inputs')
    hashes={};inventory=[]
    def verify(path,role,expected=None):
        path=Path(path).expanduser().resolve();value=digest(path)
        require(expected is None or expected==value,f'Input SHA differs: {path}')
        if str(path) not in hashes:
            hashes[str(path)]=value
            inventory.append(dict(role=role,path=str(path),sha256=value,size_bytes=path.stat().st_size))
        return path
    ledger=verify(reconstruction/'01_execution_status.csv','reconstruction_result')
    plan_path=verify(reconstruction/'02_reconstruction_plan.csv','reconstruction_plan')
    frozen_path=verify(reconstruction/'06_frozen_run_manifest.json','frozen_run_manifest')
    frozen=json.loads(frozen_path.read_text())
    print('Verifying frozen reconstruction inputs (streaming SHA-256 only)...',flush=True)
    for path,expected in frozen['input_sha256'].items():verify(path,'frozen_reconstruction_input',expected)
    verify(Path(__file__),'evaluation_script')
    data=pd.read_csv(ledger,keep_default_na=False)
    plan=pd.read_csv(plan_path,keep_default_na=False)
    require(set(KEY+XYZ+MRI+['status','coordinate_frame','coordinate_units','stimulation_pair',
            'hemisphere','mapping_status','configuration_key','configuration_parameters'])<=set(data),
            'Result ledger lacks required identity/estimate fields')
    require(len(data)==len(plan)==5124 and not data.duplicated(KEY).any()
            and not plan.duplicated(KEY).any(),'Expected 5,124 unique planned/completed jobs')
    require(set(data.status)<={'PASS','FAIL'},'Reconstruction is incomplete or contains unknown statuses')
    require(set(data.test_subject)==SUBJECTS and set(data.method)==METHODS,'Subject/method coverage differs')
    require(data.groupby('strategy').size().to_dict()=={'LOSO':732,'Single-reference':4392},'Strategy counts differ')
    left=data.set_index(KEY);right=plan.set_index(KEY)
    require(set(left.index)==set(right.index),'Checkpoint identities differ from plan')
    right=right.loc[left.index]
    for col in ['configuration_key','parameter_owner','configuration_parameters','model_subject']:
        require(left[col].eq(right[col]).all(),f'Checkpoint changed frozen planned field: {col}')
    require(data.model_subject.eq('fsaverage').all() and data.mapping_status.eq('ASSUMED_UNVERIFIED').all(),
            'Anatomy or exploratory mapping policy changed')
    require(len(data[['test_subject','run']].drop_duplicates())==61,'Expected 61 subject/run pairs')
    single=data.loc[data.strategy.eq('Single-reference')]
    require(not single.reference_subject.eq(single.test_subject).any()
            and single.groupby(['method','test_subject','run']).reference_subject.nunique().eq(6).all(),
            'Single Reference exclusion/coverage differs')
    transform_paths=[Path(p) for p in frozen['input_sha256'] if Path(p).name=='07_coordinate_transform_provenance.json']
    require(len(transform_paths)==1,'Ambiguous template transform provenance')
    transform=np.asarray(json.loads(transform_paths[0].read_text())['head_to_fsaverage_mri'],float)
    require(transform.shape==(4,4) and np.isfinite(transform).all(),'Invalid template transform')
    passed=data.status.eq('PASS')
    require(data.loc[passed,'coordinate_frame'].eq('template_head').all()
            and data.loc[passed,'coordinate_units'].eq('mm').all(),'Unexpected estimate frame or units')
    head=data.loc[passed,XYZ].apply(pd.to_numeric,errors='raise').to_numpy(float)
    mapped=data.loc[passed,MRI].apply(pd.to_numeric,errors='raise').to_numpy(float)
    require(np.isfinite(head).all() and np.isfinite(mapped).all(),'Nonfinite PASS estimates')
    transform_error=np.linalg.norm(mapped-(head@transform[:3,:3].T+1000*transform[:3,3]),axis=1)
    require(not len(transform_error) or transform_error.max()<=1e-5,'HEAD/fsaverage MRI coordinate inconsistency')
    gt_path=verify(args.ground_truth,'diagnostic_reference_coordinates')
    gt_manifest=verify(args.ground_truth_manifest,'diagnostic_reference_origin')
    origin=json.loads(gt_manifest.read_text())
    require(Path(origin['ground_truth_dataset']).resolve()==Path(frozen['dataset']).resolve(),
            'Ground-truth dataset differs from reconstruction dataset')
    gt=pd.read_csv(gt_path,keep_default_na=False)
    required=['test_subject','run','coordinate_audit','gt_stimulation_pair','gt_hemisphere',
              'contact_sha256','transform_sha256']+GT
    require(set(required)<=set(gt) and len(gt)==61 and not gt.duplicated(['test_subject','run']).any(),
            'Expected 61 uniquely audited diagnostic reference rows')
    require(set(map(tuple,gt[['test_subject','run']].values))==set(map(tuple,data[['test_subject','run']].values)),
            'Ground-truth run coverage differs')
    require(gt.coordinate_audit.eq('PASS_NUMERICAL').all(),'Ground-truth coordinate audit failed')
    gt[GT]=gt[GT].apply(pd.to_numeric,errors='raise')
    require(np.isfinite(gt[GT].to_numpy(float)).all(),'Nonfinite ground truth')
    # Hash provenance of independently derived coordinates. These individual
    # reference files are never parsed into a forward or used to align sensors.
    dataset=Path(origin['ground_truth_dataset'])
    for row in gt.itertuples():
        verify(dataset/'derivatives/epochs'/row.test_subject/'ieeg'/
            f'{row.test_subject}_task-seegstim_space-surface_electrodes.tsv',
            'evaluation_only_contact_reference',row.contact_sha256)
        verify(dataset/'derivatives/sourcemodelling'/row.test_subject/'xfm'/
            f'{row.test_subject}_from-head_to-surface.h5',
            'evaluation_only_reference_transform_hash',row.transform_sha256)
    result=data.merge(gt[required],on=['test_subject','run'],how='left',validate='many_to_one')
    passed=result.status.eq('PASS')
    require(result.loc[passed,'stimulation_pair'].eq(result.loc[passed,'gt_stimulation_pair']).all(),
            'Ground-truth stimulation pair mismatch')
    require(result.loc[passed,'hemisphere'].eq(result.loc[passed,'gt_hemisphere']).all(),
            'Ground-truth hemisphere mismatch')
    result['diagnostic_distance_mm']=np.nan
    result.loc[passed,'diagnostic_distance_mm']=np.linalg.norm(
        result.loc[passed,XYZ].to_numpy(float)-result.loc[passed,GT].to_numpy(float),axis=1)
    result['localization_error_mm']=np.nan
    result['localization_error_valid']=False
    result['evaluation_status']=np.where(passed,'UNREGISTERED_HEAD_DIAGNOSTIC_ONLY','NO_ESTIMATE_FAIL')
    result['evaluation_cohort']='all_seven_subjects_template_geometry_exploratory'
    result['interpretation']='TEMPLATE_HEAD_VS_INDIVIDUAL_HEAD_UNREGISTERED_SEPARATION_NOT_ANATOMICAL_ERROR'
    method=summarize(result,['strategy','method'])
    subject=summarize(result,['strategy','method','test_subject'])
    reference=summarize(result.loc[result.strategy.eq('Single-reference')],['strategy','method','reference_subject'])
    equal=subject.groupby(['strategy','method'],as_index=False).agg(
        n_subjects=('test_subject','nunique'),
        n_subjects_with_distances=('diagnostic_median_mm','count'),
        mean_subject_median_diagnostic_mm=('diagnostic_median_mm','mean'),
        median_subject_median_diagnostic_mm=('diagnostic_median_mm','median'),
        mean_subject_pass_rate=('pass_rate','mean'))
    run=summarize(result,['strategy','method','test_subject','run'])
    run['aggregation_note']='Across available reference identities; repeated evaluations of the same EEG run, not independent subjects'
    unsuccessful=result.loc[~passed,KEY+['status','failure_reason','configuration_key','configuration_parameters']]
    registration=args.registration_status.expanduser().resolve()
    prior={}
    if registration.is_file():
        registration=verify(registration,'prior_registration_status_reference_only')
        status=pd.read_csv(registration,keep_default_na=False)
        require({'subject','status','reason'}<=set(status) and not status.subject.duplicated().any(),
                'Ambiguous prior registration-status report')
        prior={r.subject:r for r in status.itertuples()}
    readiness=pd.DataFrame([dict(test_subject=s,
        prior_registration_status=prior[s].status if s in prior else 'MISSING',
        prior_registration_reason=prior[s].reason if s in prior else 'No previous status row',
        anatomical_evaluation_status='BLOCKED_NO_VALIDATED_COMMON_FRAME_MAPPING',
        valid_anatomical_errors=0,
        diagnostic_reference_policy='individual HEAD coordinates only; no anatomy mapping applied',
        reconstruction_uses_individual_geometry=False) for s in sorted(SUBJECTS)])
    # Freeze a coherent input snapshot before publication. The caller holds a
    # shared writer lock throughout, and byte checks cover other input writers.
    for path,expected in hashes.items():
        require(Path(path).parent not in out.parents and Path(path).parent!=out,
                f'Output overlaps an input directory: {Path(path).parent}')
        require(digest(path)==expected,f'Input changed during evaluation: {path}')
    out.mkdir(parents=True,exist_ok=False)
    outputs={
        '01_run_level_diagnostics.csv':result,'02_ground_truth_diagnostic_reference.csv':gt,
        '03_method_summary.csv':method,'04_subject_summary.csv':subject,
        '05_reference_summary.csv':reference,'06_equal_subject_summary.csv':equal,
        '07_run_reference_summary.csv':run,'08_unsuccessful_jobs.csv':unsuccessful,
        '09_input_integrity_inventory.csv':pd.DataFrame(inventory),
        '10_anatomical_evaluation_readiness.csv':readiness,
        '11_sub03_summary.csv':subject.loc[subject.test_subject.eq('sub-03')]}
    for name,table in outputs.items():table.to_csv(out/name,index=False)
    shutil.copy2(Path(__file__),out/'script_snapshot.py.txt')
    counts={str(k):int(v) for k,v in result.status.value_counts().items()}
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='DIAGNOSTIC_EVALUATION_COMPLETE_ANATOMICAL_ERROR_BLOCKED',
        input_sha256=hashes,output_sha256={n:digest(out/n) for n in outputs},
        counts=dict(total=len(result),solver_status=counts,diagnostic_distances=int(passed.sum()),
                    valid_anatomical_errors=0,subjects=7,runs=61,methods=12),
        diagnostic_definition='Euclidean norm in mm: template HEAD estimate minus individual HEAD ground truth; unregistered',
        coordinate_consistency_maximum_mm=float(transform_error.max()) if len(transform_error) else None,
        mapping_status='ASSUMED_UNVERIFIED',registration_applied=False,
        inverse_reconstruction_executed=False,parameters_changed=False,original_files_changed=False,
        individual_ground_truth_used_for_evaluation_only=True,anatomical_localization_error_valid=False,
        limitations=['Template HEAD and individual HEAD have no validated anatomical correspondence.',
            'Raw diagnostic distances are not anatomically validated errors or a method-accuracy ranking.',
            'Template channel correspondence and original sensor/scalp mismatch remain unvalidated.',
            'Both anatomy and sensor geometry differ from the prior no-warp experiment.',
            'Single Reference repeats EEG runs across six parameter owners; report reference/subject summaries alongside pooled jobs.',
            'Equal-subject summaries use available distances and expose failures through pass rates.',
            'All seven subjects use common template geometry; old sub-03 individual-coregistration failure is not a new reconstruction exclusion.',
            'Known stimulation hemisphere informed cortical peak selection; dipoles used maximum GOF.',
            'Existing registration status is inventoried only; no mapping is accepted or optimized by this script.'])
    (out/'12_evaluation_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text(
        '# Template/fsaverage diagnostic evaluation\n\n'
        f'{len(result):,} jobs: {counts.get("PASS",0):,} PASS, {counts.get("FAIL",0)} FAIL. '
        f'{int(passed.sum()):,} diagnostic distances; **zero valid anatomical localization errors**.\n\n'
        'Distance means template HEAD minus individual HEAD Euclidean separation, without anatomical registration. '
        'Matching axis names do not establish anatomical correspondence. '
        '`localization_error_mm` remains empty and `localization_error_valid` remains False.\n\n'
        '01: full job-level diagnostic ledger; 02: audited reference coordinates; '
        '03: pooled method/strategy summaries; 04: participant summaries; '
        '05: Single Reference owner summaries; 06: equal-participant summaries; '
        '07: one summary per unique test run/method/strategy across reference owners; '
        '08: preserved failures; 09: SHA-256 input inventory; '
        '10: anatomical-evaluation blockers; 11: additional sub-03 summary; '
        '12: provenance and interpretation manifest.\n\n'
        'Failed jobs retain missing distances and count in pass-rate denominators. '
        'Sub-03 remains included in all-seven summaries. Six references are repeated evaluations of the same run. '
        'Do not compare these diagnostic values as anatomical errors against baseline/no-warp results.\n\n'
        'Next: review the diagnostic audit, resolve and independently validate the evaluation-only '
        'mapping of stimulation locations to fsaverage, then calculate anatomical errors in a separately labeled stage. '
        'No subject anatomy, digitization or ground-truth fitting is introduced into reconstruction.\n')
    print(f'{report["status"]}: {out}')
    print(f'{len(result)} jobs; {int(passed.sum())} diagnostic distances; {int((~passed).sum())} failures; anatomical errors unavailable.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reconstruction-root',type=Path,default=VARIANT/'outputs/03_reconstruct_assumed_v1')
    p.add_argument('--ground-truth',type=Path,default=ROOT/'cross_subject_no_warp/outputs/06_evaluate_localization/02_coordinate_validation.csv')
    p.add_argument('--ground-truth-manifest',type=Path,default=ROOT/'cross_subject_no_warp/outputs/06_evaluate_localization/07_evaluation_manifest.json')
    p.add_argument('--registration-status',type=Path,default=ROOT/'cross_subject_fsaverage/outputs/03_register_ground_truth/02_registration_status.csv')
    p.add_argument('--output-root',type=Path,default=VARIANT/'outputs/04_evaluate_diagnostics')
    args=p.parse_args(argv)
    # Never create or change the reconstruction lock file from evaluation.
    with (args.reconstruction_root.expanduser().resolve()/'.writer.lock').open('r') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Reconstruction writer is active; evaluate after it finishes')
        evaluate(args)


if __name__=='__main__':
    main()
