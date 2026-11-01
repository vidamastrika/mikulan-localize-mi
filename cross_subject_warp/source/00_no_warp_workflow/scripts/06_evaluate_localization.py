#!/usr/bin/env python3
"""Script 06 — no-warp reconstruction evaluation (primary + sensitivity).

RUN FROM PROJECT ROOT, WITH PROJECT .venv ACTIVATED:
  python cross_subject_no_warp/scripts/06_evaluate_localization.py --phase audit
  python cross_subject_no_warp/scripts/06_evaluate_localization.py --phase evaluate
  python cross_subject_no_warp/scripts/06_evaluate_localization.py --phase evaluate --overwrite

INPUTS (never modified):
  outputs/03_reference_based_final/01_execution_status.csv              LOSO primary
  outputs/05_single_reference_final/01_execution_status.csv            single-reference primary
  outputs/03_gamma_wholebrain_sensitivity/01_execution_status_wholebrain_variant.csv  LOSO variant
  outputs/05_wholebrain_hemisphere_sensitivity/04_execution_status_variant.csv       single-ref variant
  Dataset Localize-MI, source coordinates, contacts and rigid within-subject transforms.

METHOD:
  - Validate all keys and expected primary schedules (1464 LOSO, 4392 single-reference).
  - Prove variants are identical to primary except explicitly identified overrides;
    require PASS->PASS retained rows and FAIL->PASS only for override cases.
  - Derive stimulation midpoint (metres, surface coordinates) using the project's
    original localize_mi.metrics.load_stimulation_info and the same load_run loader.
  - Convert stimulation midpoint from test subject surface into test subject HEAD
    with inverse of test head->surface H5 transform. No intersubject transform.
  - Within-subject Euclidean localization error is calculated in the test head frame.
  - Cross-subject raw head-coordinate separation is a DIAGNOSTIC only, NOT a validated
    anatomical localization error: test and reference head frames have no proven
    inter-subject anatomical correspondence. No affine, SyN, or warping is applied.
  - Never replace failures with arbitrary large errors or omit them from failure rates.
  - Variant is a MIXED peak-selection policy, not a uniform whole-brain benchmark.

OUTPUTS: outputs/06_evaluate_localization/ (atomic directory replacement on --overwrite)
  01_input_validation.csv         input counts and override audit
  02_coordinate_validation.csv    per-run source and ground truth coordinate metadata
  03_primary_localization.csv     all primary rows, with valid within errors
  04_sensitivity_localization.csv both variant rows, with valid within errors
  05_method_summary.csv           PASS/FAIL and within median/IQR by strategy
  06_failure_analysis.csv         failures by strategy and variant
  07_evaluation_manifest.json     source file SHA256 + policy
  08_raw_head_diagnostic.csv      cross raw coordinate separation, explicitly NOT localization error
  README.txt                      limitations and interpretation

--phase audit validates inputs only (no dataset loading). --phase evaluate reads
actual ground-truth coordinates. --overwrite ONLY replaces this Script 06 output
folder after a successful complete evaluation; no other folder is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path('cross_subject_no_warp/outputs')
KEY = ['method','reference_subject','test_subject','run','condition']
CORE_KEYS = ['method','test_subject','run','condition']
METHODS = {'MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC','LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME'}
SUBJECTS = {f'sub-{i:02d}' for i in range(1,8)}
NUMERICAL = ['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def required(path):
    if not path.is_file(): raise FileNotFoundError(f'Missing input: {path}')
    return path


def read(path):
    return pd.read_csv(required(path),dtype=str,keep_default_na=False)


def same_value(a,b):
    if a==b:return True
    try:
        x=float(a);y=float(b)
        return bool(np.isfinite(x) and np.isfinite(y) and np.isclose(x,y,rtol=1e-12,atol=1e-9))
    except (ValueError,TypeError):return False


def validate_primary(df,strategy):
    req=set(KEY+['status','configuration_key','actual_channels_json','configuration_parameters']+NUMERICAL)
    missing=req-set(df.columns)
    if missing:raise ValueError(f'{strategy}: missing columns {sorted(missing)}')
    if df.duplicated(KEY).any():raise ValueError(f'{strategy}: duplicate job keys')
    expected=1464 if strategy=='LOSO' else 4392
    if len(df)!=expected:raise ValueError(f'{strategy}: expected {expected} rows, got {len(df)}')
    if set(df.method)!=METHODS:raise ValueError(f'{strategy}: method set mismatch')
    if set(df.test_subject)!=SUBJECTS:raise ValueError(f'{strategy}: test subject set mismatch')
    if not set(df.status)<= {'PASS','FAIL'}:raise ValueError(f'{strategy}: unknown statuses')
    if (df.reference_subject==df.test_subject).any():raise ValueError(f'{strategy}: reference==test')
    cross=df[df.condition=='single_reference_no_warp']
    if strategy=='LOSO':
        if len(cross)!=732 or len(df[df.condition=='within'])!=732:raise ValueError('LOSO condition imbalance')
        if df.groupby(CORE_KEYS).size().ne(1).any():raise ValueError('Duplicate LOSO core jobs')
    else:
        if len(cross)!=4392 or set(df.condition)!= {'single_reference_no_warp'}:
            raise ValueError('Single-reference should have only 4392 cross jobs')
    for subject,group in df.groupby('test_subject'):
        references=set(group.reference_subject)
        if strategy=='LOSO' and len(references)!=1:raise ValueError(f'{subject}: LOSO references incorrect')
        if strategy!='LOSO' and references!=SUBJECTS-{subject}:raise ValueError(f'{subject}: single-reference set incorrect')
    return dict(strategy=strategy,total=len(df),passed=int((df.status=='PASS').sum()),failed=int((df.status=='FAIL').sum()))


def validate_variant(primary,variant,strategy):
    if 'sensitivity_override' not in variant:raise ValueError(f'{strategy}: missing sensitivity_override')
    if variant.duplicated(KEY).any():raise ValueError(f'{strategy}: duplicate variant jobs')
    if len(variant)!=len(primary):raise ValueError(f'{strategy}: variant row count mismatch')
    p=primary.set_index(KEY).sort_index()
    v=variant.set_index(KEY).sort_index()
    if not p.index.equals(v.index):raise ValueError(f'{strategy}: variant jobs differ from primary')
    override=v.sensitivity_override.astype(str).str.lower().eq('true')
    if int(override.sum()) != (1 if strategy=='LOSO' else 3):raise ValueError(f'{strategy}: unexpected override count')
    if not (p.loc[override,'status']=='FAIL').all() or not (v.loc[override,'status']=='PASS').all():
        raise ValueError(f'{strategy}: overrides must be FAIL -> PASS')
    # Non-overridden rows MUST match in every original column, except empty metadata
    # is never silently normalized. Variant-only provenance columns are excluded.
    for col in p.columns:
        if col not in v:raise ValueError(f'{strategy}: variant missing original column {col}')
        a=p.loc[~override,col];b=v.loc[~override,col]
        bad=[i for i,(x,y) in enumerate(zip(a,b)) if not same_value(x,y)]
        if bad:raise ValueError(f'{strategy}: changed {len(bad)} untouched rows in {col}')
    # For overrides, validate immutable identifiers, configuration, geometry and EEG.
    immutable=['configuration_key','configuration_parameters','reference_subject','model_subject',
               'actual_channels_json','nominal_channels_json','eeg_data_sha256','epoch_times_sha256',
               'montage_template_subject','channels']
    for col in immutable:
        if col not in p.columns or col not in v.columns:continue
        for a,b in zip(p.loc[override,col],v.loc[override,col]):
            # The original FAIL may have no metadata; do not equate blank with verified.
            if a and b and not same_value(a,b):raise ValueError(f'{strategy}: override changed {col}')
    return dict(strategy=strategy,override_count=int(override.sum()),variant_pass=int((variant.status=='PASS').sum()),
                variant_fail=int((variant.status=='FAIL').sum()),untouched_records=int((~override).sum()))


def get_ground_truth(dataset,root,subjects_runs):
    sys.path.insert(0,str(root/'src'))
    from localize_mi import load_run
    from localize_mi.metrics import load_stimulation_info,load_surface_transform
    out=[];look={}
    for sub,run in sorted(subjects_runs):
        path=dataset/'derivatives'/'epochs'/sub/'ieeg'/f'{sub}_task-seegstim_space-surface_electrodes.tsv'
        xfm=dataset/'derivatives'/'sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
        row={'test_subject':sub,'run':run,'coordinate_audit':'FAIL','coordinate_audit_reason':''}
        try:
            data=load_run(dataset=dataset,subject=sub,run=run,task='seegstim')
            stim=load_stimulation_info(data.metadata['Description'],path)
            transform=np.asarray(load_surface_transform(xfm),float)
            if transform.shape!=(4,4):raise ValueError('Expected 4x4 head-to-surface matrix')
            if not np.allclose(transform[3],[0,0,0,1],atol=1e-6):raise ValueError('Invalid transform bottom row')
            r=transform[:3,:3]
            if not np.allclose(r.T@r,np.eye(3),atol=1e-5):raise ValueError('Non-rigid transform')
            mid=np.asarray(stim.midpoint,float).reshape(3)
            if not np.isfinite(mid).all() or np.linalg.norm(mid)>1:raise ValueError('Midpoint not finite or plausible metres')
            xyz=(np.linalg.inv(transform)@np.r_[mid,1])[:3]*1000
            if not np.isfinite(xyz).all() or np.linalg.norm(xyz)>1000:raise ValueError('Invalid head coordinate')
            row.update(coordinate_audit='PASS_NUMERICAL',gt_head_x_mm=xyz[0],gt_head_y_mm=xyz[1],
                       gt_head_z_mm=xyz[2],gt_surface_x_m=mid[0],gt_surface_y_m=mid[1],gt_surface_z_m=mid[2],
                       gt_hemisphere=stim.hemisphere,gt_stimulation_pair=stim.pair,
                       contact_sha256=sha(path),transform_sha256=sha(xfm))
            look[(sub,run)]=row
        except Exception as e:row['coordinate_audit_reason']=f'{type(e).__name__}: {e}'
        out.append(row)
    table=pd.DataFrame(out)
    errors=table[table.coordinate_audit!='PASS_NUMERICAL']
    if len(errors):
        raise ValueError(f'{len(errors)} test runs have invalid GT coordinate audit: '+
                         '; '.join(errors.test_subject+' '+errors.run+' '+errors.coordinate_audit_reason))
    return table,look


def evaluate_rows(df,gt,scenario,strategy):
    out=df.copy()
    out.insert(0,'strategy',strategy)
    out.insert(1,'scenario',scenario)
    coords=[];diag=[]
    for rec in out.to_dict('records'):
        g=gt[(rec['test_subject'],rec['run'])]
        valid=rec['status']=='PASS'
        xyz=np.array([float(rec[c]) for c in NUMERICAL]) if valid else np.full(3,np.nan)
        true=np.array([float(g['gt_head_'+a+'_mm']) for a in 'xyz'])
        if valid and not np.isfinite(xyz).all():raise ValueError('PASS missing finite peak coordinates')
        if valid and str(rec.get('coordinate_frame',''))!='model_head':raise ValueError('Unexpected coordinate frame')
        if valid and str(rec.get('coordinate_units',''))!='mm':raise ValueError('Unexpected coordinate units')
        if valid and rec.get('stimulation_pair') and str(rec['stimulation_pair'])!=str(g['gt_stimulation_pair']):
            raise ValueError('Stimulation pair mismatch '+str(rec))
        raw=float(np.linalg.norm(xyz-true)) if valid else np.nan
        iswithin=rec['condition']=='within'
        coords.append(dict(gt_head_x_mm=true[0],gt_head_y_mm=true[1],gt_head_z_mm=true[2],
                           gt_hemisphere=g['gt_hemisphere'],coordinate_validation='PASS_NUMERICAL' if valid else 'NO_ESTIMATE',
                           localization_error_mm=raw if iswithin else np.nan,
                           localization_error_valid=bool(valid and iswithin),
                           cross_anatomical_error_available=False if not iswithin else bool(valid),
                           raw_head_separation_mm=raw if not iswithin else np.nan,
                           interpretation='within_test_head' if iswithin else 'cross_unregistered_head_diagnostic_only'))
        if not iswithin:
            diag.append(dict(strategy=strategy,scenario=scenario,method=rec['method'],reference_subject=rec['reference_subject'],
                             test_subject=rec['test_subject'],run=rec['run'],status=rec['status'],
                             raw_head_separation_mm=raw,valid_localization_error=False))
    out=pd.concat([out.reset_index(drop=True),pd.DataFrame(coords)],axis=1)
    return out,pd.DataFrame(diag)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phase',choices=['audit','evaluate'],default='audit')
    ap.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    ap.add_argument('--project-root',type=Path,default=Path.cwd())
    ap.add_argument('--loso-primary',type=Path,default=BASE/'03_reference_based_final/01_execution_status.csv')
    ap.add_argument('--single-primary',type=Path,default=BASE/'05_single_reference_final/01_execution_status.csv')
    ap.add_argument('--loso-variant',type=Path,default=BASE/'03_gamma_wholebrain_sensitivity/01_execution_status_wholebrain_variant.csv')
    ap.add_argument('--single-variant',type=Path,default=BASE/'05_wholebrain_hemisphere_sensitivity/04_execution_status_variant.csv')
    ap.add_argument('--output-root',type=Path,default=BASE/'06_evaluate_localization')
    ap.add_argument('--overwrite',action='store_true',help='Replace ONLY Script 06 output on successful evaluation')
    args=ap.parse_args()
    paths={'loso_primary':args.loso_primary,'single_primary':args.single_primary,
           'loso_variant':args.loso_variant,'single_variant':args.single_variant}
    data={k:read(v) for k,v in paths.items()}
    checks=[validate_primary(data['loso_primary'],'LOSO'),validate_primary(data['single_primary'],'Single-reference'),
            validate_variant(data['loso_primary'],data['loso_variant'],'LOSO'),
            validate_variant(data['single_primary'],data['single_variant'],'Single-reference')]
    print('INPUT AUDIT PASSED')
    for c in checks:print(c)
    if args.phase=='audit':print('Audit-only: no output files created');return
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{args.output_root} already exists. Use --overwrite after inspection.')
    pairs=set(zip(data['single_primary'].test_subject,data['single_primary'].run))
    pairs.update(zip(data['loso_primary'].test_subject,data['loso_primary'].run))
    gt_table,gt=get_ground_truth(args.dataset.expanduser().resolve(),args.project_root.resolve(),pairs)
    analyses=[];diagnostics=[]
    for strategy,scn,label in [('LOSO','primary','loso_primary'),('LOSO','sensitivity','loso_variant'),
                                ('Single-reference','primary','single_primary'),('Single-reference','sensitivity','single_variant')]:
        rows,diag=evaluate_rows(data[label],gt,scn,strategy)
        analyses.append(rows);diagnostics.append(diag)
    primary=pd.concat([analyses[0],analyses[2]],ignore_index=True)
    sensitivity=pd.concat([analyses[1],analyses[3]],ignore_index=True)
    allrows=pd.concat(analyses,ignore_index=True)
    summ=[]
    for (scenario,strategy,method,cond),part in allrows.groupby(['scenario','strategy','method','condition'],sort=True):
        err=part.localization_error_mm.dropna()
        summ.append(dict(scenario=scenario,strategy=strategy,method=method,condition=cond,
                         scheduled=len(part),passed=int((part.status=='PASS').sum()),failed=int((part.status=='FAIL').sum()),
                         error_n=len(err),median_error_mm=err.median() if len(err) else np.nan,
                         q25_error_mm=err.quantile(.25) if len(err) else np.nan,
                         q75_error_mm=err.quantile(.75) if len(err) else np.nan,
                         cross_error_valid=False if cond!='within' else True))
    fail=allrows[allrows.status=='FAIL'][['scenario','strategy']+KEY+['failure_reason']].copy()
    manifest={'input_sha256':{k:sha(v) for k,v in paths.items()},'input_paths':{k:str(v) for k,v in paths.items()},
              'script_sha256':sha(__file__),'ground_truth_dataset':str(args.dataset),
              'cross_evaluation':'NOT anatomical localization error: no intersubject alignment',
              'within_evaluation':'Euclidean distance in test head frame, mm',
              'sensitivity_policy':'mixed peak-extraction rule; only explicit former failures overridden',
              'counts':checks}
    parent=args.output_root.parent;parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.06_evaluate_staging_',dir=parent))
    try:
        pd.DataFrame(checks).to_csv(temp/'01_input_validation.csv',index=False)
        gt_table.to_csv(temp/'02_coordinate_validation.csv',index=False)
        primary.to_csv(temp/'03_primary_localization.csv',index=False)
        sensitivity.to_csv(temp/'04_sensitivity_localization.csv',index=False)
        pd.DataFrame(summ).to_csv(temp/'05_method_summary.csv',index=False)
        fail.to_csv(temp/'06_failure_analysis.csv',index=False)
        pd.concat(diagnostics,ignore_index=True).to_csv(temp/'08_raw_head_diagnostic.csv',index=False)
        (temp/'07_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (temp/'README.txt').write_text(__doc__+'\n')
        if args.output_root.exists():
            # replace only AFTER complete staging; restore previous output on failed move.
            backup=parent/(args.output_root.name+'.previous_'+sha(__file__)[:10])
            if backup.exists():raise FileExistsError(f'Backup already exists; inspect first: {backup}')
            args.output_root.rename(backup)
            try:temp.rename(args.output_root)
            except Exception:
                backup.rename(args.output_root)
                raise
            print('Previous Script 06 output archived:',backup)
        else:temp.rename(args.output_root)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print('Evaluation written:',args.output_root)
    print('Cross-subject raw separations are NOT valid anatomical localization errors.')

if __name__=='__main__':main()
