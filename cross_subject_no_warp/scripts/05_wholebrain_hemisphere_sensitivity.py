#!/usr/bin/env python3
"""Script 05 supplementary analysis: unrestricted peak for hemisphere-only FAIL.

PURPOSE
  Preserve 05_single_reference_final as PRIMARY (4,378 PASS, 14 FAIL).
  Retry ONLY original FAIL cases with the exact failure:
      RuntimeError: Cannot find peaks with no vertices
  This suggests an empty requested hemisphere; it is verified by re-running
  the SAME solver and extracting a maximum-absolute whole-brain peak.
  Gamma-MAP 'No active dipoles found' and SESAME 'zero sources' stay FAIL.
  NO parameter tuning, channel-selection change, registration, or warping.

INPUTS (from project root)
  cross_subject_no_warp/outputs/05_single_reference_final/01_execution_status.csv
  cross_subject_no_warp/outputs/04_single_reference_parameters/
  cross_subject_no_warp/source/02_parameter_grids/
  cross_subject_no_warp/scripts/{02,03,04,05}_*.py and Localize-MI data.

OUTPUTS (separate, immutable primary dataset)
  cross_subject_no_warp/outputs/05_wholebrain_hemisphere_sensitivity/
    01_candidate_cases.csv              Exact failed candidates
    02_retry_status.csv                 Whole-brain rerun of these ONLY
    03_primary_all_failures.csv         All 14 original failures
    04_execution_status_variant.csv    4392 rows, substitute verified PASS only
    05_override_audit.csv               Original/replacement status and policy
    06_manifest.json                    Checksums and scientific limitations
    README.txt                          Primary-vs-sensitivity explanation

USAGE (activate .venv, run from project root)
  python cross_subject_no_warp/scripts/05_wholebrain_hemisphere_sensitivity.py --phase plan
  python cross_subject_no_warp/scripts/05_wholebrain_hemisphere_sensitivity.py --phase reconstruct --quiet
  python cross_subject_no_warp/scripts/05_wholebrain_hemisphere_sensitivity.py --phase merge

IMPORTANT
  A whole-brain PASS means a nonzero peak was located; not a scientific
  validation, not proof of optimization convergence or localization accuracy.
  Remaining FAIL rows are preserved. Script 06 must report two distinct sets:
  (A) strict primary evaluation; (B) mixed-peak-rule sensitivity evaluation.
  The variant may NEVER replace the original primary 05 final results.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('cross_subject_no_warp')
KEYS=['method','reference_subject','test_subject','run','condition']
ERROR='RuntimeError: Cannot find peaks with no vertices'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def atomic_csv(path,df):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f'Refuse to overwrite {path}')
    df.to_csv(path,index=False)

def write_json(path,data):
    if path.exists():raise FileExistsError(f'Refuse to overwrite {path}')
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')

def key(row):return tuple(row[k] for k in KEYS)

def load_primary(path):
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if len(df)!=4392 or df.duplicated(KEYS).any():raise ValueError('Expected 4392 unique primary jobs')
    if (df.status=='PASS').sum()!=4378 or (df.status=='FAIL').sum()!=14:
        raise ValueError('Unexpected primary PASS/FAIL count; inspect before continuing')
    if (df.reference_subject==df.test_subject).any():raise ValueError('Self-reference detected')
    return df

def candidates(df):
    # Exact error only. Never include Gamma-MAP no-active or SESAME zero-source failures.
    return df[(df.status=='FAIL') & (df.failure_reason==ERROR)].copy()

def peak_unrestricted(stc,fwd,requested,mne):
    from mne.io.constants import FIFF
    verts=[np.asarray(v,dtype=int) for v in stc.vertices]
    if len(verts)!=2 or sum(map(len,verts))!=stc.data.shape[0]:
        raise ValueError('Invalid cortical vertex arrangement')
    dat=np.asarray(stc.data)
    if dat.ndim!=2 or not dat.size or not np.isfinite(dat).all() or not np.any(np.abs(dat)>0):
        raise ValueError('Empty, invalid, or all-zero source estimate')
    ix,ti=np.unravel_index(np.abs(dat).argmax(),dat.shape)
    hi=0 if ix<len(verts[0]) else 1
    source=fwd['src'][hi]; idx=ix if hi==0 else ix-len(verts[0]);v=int(verts[hi][idx]);t=float(stc.times[ti]);found=['lh','rh'][hi]
    if requested not in ('lh','rh'):raise ValueError('Unknown ground-truth hemisphere')
    # A case only qualifies if the requested hemisphere was actually empty.
    if len(verts[0 if requested=='lh' else 1])!=0:
        raise ValueError('Requested hemisphere is not empty: not a hemisphere-only failure')
    if found==requested:raise ValueError('Unexpected same-hemisphere peak')
    xyz=np.asarray(source['rr'][v],float)
    frame=int(source['coord_frame'])
    if frame==int(FIFF.FIFFV_COORD_MRI):
        tr=fwd['mri_head_t']
        if int(tr['from'])==int(FIFF.FIFFV_COORD_HEAD):tr=mne.transforms.invert_transform(tr)
        if int(tr['from'])!=int(FIFF.FIFFV_COORD_MRI) or int(tr['to'])!=int(FIFF.FIFFV_COORD_HEAD):
            raise ValueError('Wrong MRI-to-head transform')
        xyz=mne.transforms.apply_trans(tr,xyz)
    elif frame!=int(FIFF.FIFFV_COORD_HEAD):raise ValueError('Unknown source frame')
    print(f'  Peak diagnostic: requested={requested} found={found} '
          f'vertices={[len(x) for x in verts]} vertex={v} time={t:.6f}',flush=True)
    return xyz,v,t

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase',choices=['plan','reconstruct','merge'],default='plan')
    p.add_argument('--primary',type=Path,default=ROOT/'outputs/05_single_reference_final/01_execution_status.csv')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/05_wholebrain_hemisphere_sensitivity')
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--bem-root',type=Path,default=Path('outputs/12_additional_methods_all_runs/bem'))
    p.add_argument('--selection-root',type=Path,default=ROOT/'outputs/04_single_reference_parameters')
    p.add_argument('--source-root',type=Path,default=ROOT/'source/02_parameter_grids')
    p.add_argument('--preparation-root',type=Path,default=ROOT/'outputs/01_prepare_experiment')
    p.add_argument('--code-source-root',type=Path,default=ROOT/'source/03_solver_code')
    p.add_argument('--script03',type=Path,default=ROOT/'scripts/03_run_localization.py')
    p.add_argument('--script02',type=Path,default=ROOT/'scripts/02_select_loso_parameters.py')
    p.add_argument('--script04',type=Path,default=ROOT/'scripts/04_select_single_reference_parameters.py')
    p.add_argument('--script05',type=Path,default=ROOT/'scripts/05_run_single_reference.py')
    p.add_argument('--quiet',action='store_true')
    a=p.parse_args();out=a.output_root
    df=load_primary(a.primary);target=candidates(df)
    if a.phase=='plan':
        print(f'Primary: {len(df)} rows; {(df.status=="PASS").sum()} PASS; {(df.status=="FAIL").sum()} FAIL')
        print(f'Hemisphere-only candidates: {len(target)}')
        print(target[KEYS+['failure_reason']].to_string(index=False))
        print('Other errors remain FAIL:',len(df[df.status=='FAIL'])-len(target))
        return
    if a.phase=='reconstruct':
        if not len(target):raise ValueError('No hemisphere-only candidates')
        if (out/'02_retry_status.csv').exists():raise FileExistsError('Retry output exists; use merge or a NEW output root')
        out.mkdir(parents=True,exist_ok=True)
        mod05=load_module(a.script05.resolve(),'sensitivity_script05')
        solver=mod05.load_module(a.script03.resolve())
        a.dataset=a.dataset.expanduser().resolve();a.project_root=Path.cwd().resolve()
        a.code_source_root=a.code_source_root.resolve();a.montage_template='reference'
        sel,schedule,doc,state=mod05.verify_selection(a,solver)
        configs,errors=mod05.resolve_configurations(sel,solver,a.source_root,doc)
        if errors:raise ValueError('Selection preflight failed')
        full=mod05.build_jobs(sel,schedule)
        lookup={key(j):j for j in full}
        if not set(map(tuple,target[KEYS].values)).issubset(lookup):raise ValueError('Candidate job not in Script05 schedule')
        solver.check_audit(a,a.source_root,doc)
        if not (a.code_source_root/'scripts/09_full_parameter_grid.py').is_file():
            raise FileNotFoundError('Frozen solver code unavailable')
        solver.peak_head=peak_unrestricted
        engine=solver.Engine(a)
        results=[]
        for case in target.to_dict('records'):
            k=key(case);job=lookup[k];config=configs[(job['method'],job['reference_subject'])]
            if config['configuration_key']!=case['configuration_key']:
                raise ValueError(f'Parameter key changed for {k}')
            start=time.perf_counter()
            r=dict(job,status='FAIL',failure_reason='',configuration_parameters=json.dumps(config['parameters'],sort_keys=True))
            try:
                r.update(engine.solve(job,config['parameters']))
                r.update(status='PASS',failure_reason='')
                r['hemisphere_constraint_applied']=False
                r['selection_criterion']='maximum_absolute_amplitude_whole_brain'
                r['peak_selection_policy']='whole_brain_absolute_peak'
            except Exception as exc:
                r['failure_reason']=f'{type(exc).__name__}: {exc}'
                (out/'retry_failures.log').open('a').write(json.dumps(job)+'\n'+traceback.format_exc()+'\n')
            r['runtime_seconds']=round(time.perf_counter()-start,5)
            r['source_primary_sha256']=sha(a.primary)
            r['source_script03_sha256']=sha(a.script03)
            r['source_script05_sha256']=sha(a.script05)
            results.append(r)
            print(f'{k}: {r["status"]} {r["failure_reason"]}',flush=True)
        atomic_csv(out/'01_candidate_cases.csv',target)
        atomic_csv(out/'02_retry_status.csv',pd.DataFrame(results))
        atomic_csv(out/'03_primary_all_failures.csv',df[df.status=='FAIL'])
        write_json(out/'reconstruction_provenance.json',{
            'policy':'WHOLE_BRAIN_HEMISPHERE_ONLY_SENSITIVITY',
            'primary_sha256':sha(a.primary),'script03_sha256':sha(a.script03),
            'script05_sha256':sha(a.script05),'script04_selected_sha256':sha(a.selection_root/'01_single_reference_selected_parameters.csv'),
            'retry_status_sha256':sha(out/'02_retry_status.csv'),
            'attempted':len(results),'passing':sum(r['status']=='PASS' for r in results),
            'limitations':'Different peak policy; no change to parameter/forward/EEG. Unregistered reference head coordinates.'})
        return
    # merge: make alternate copy; preserve every original failure except justified successful hemispheric reruns.
    rs=out/'02_retry_status.csv';proof=out/'reconstruction_provenance.json'
    if not rs.is_file() or not proof.is_file():raise FileNotFoundError('Run --phase reconstruct first')
    provenance=json.loads(proof.read_text())
    if provenance['primary_sha256']!=sha(a.primary) or provenance['retry_status_sha256']!=sha(rs):
        raise ValueError('Input CSV changed since reconstruction')
    retry=pd.read_csv(rs,dtype=str,keep_default_na=False)
    if retry.duplicated(KEYS).any() or set(map(tuple,retry[KEYS].values))!=set(map(tuple,target[KEYS].values)):
        raise ValueError('Retry does not match exact hemisphere-only candidates')
    old={key(r):r for r in df.to_dict('records')}
    updated=df.to_dict('records');audit=[];overrides={}
    for r in retry.to_dict('records'):
        k=key(r);prev=old[k];eligible=(r['status']=='PASS')
        if eligible:
            for fld in ('configuration_key','reference_subject','test_subject','run','method','condition','model_subject','configuration_parameters'):
                if prev.get(fld) not in ('',None) and r.get(fld)!=prev[fld]:raise ValueError(f'Conflicting {fld} for {k}')
            if r.get('montage_template_subject')!=r.get('reference_subject'):raise ValueError('Wrong montage template')
            if r.get('hemisphere') not in ('lh','rh'):raise ValueError('Missing true stimulation hemisphere')
            if r.get('peak_selection_policy')!='whole_brain_absolute_peak':raise ValueError('Wrong peak rule')
            if not r.get('eeg_data_sha256') or not r.get('actual_channels_json'):raise ValueError('Missing input metadata')
            overrides[k]=r
        audit.append({**dict(zip(KEYS,k)),'original_status':prev['status'],
                      'original_failure_reason':prev['failure_reason'],'retry_status':r['status'],
                      'retry_failure_reason':r.get('failure_reason',''),
                      'substituted_in_sensitivity':eligible,
                      'peak_selection_policy':'whole_brain_absolute_peak' if eligible else 'original_strict_policy'})
    for row in updated:
        k=key(row);row['sensitivity_override']='false';row['peak_selection_policy']='original_strict_peak_policy'
        if k in overrides:
            row.update({c:overrides[k].get(c,'') for c in df.columns if c in overrides[k]})
            row['sensitivity_override']='true';row['peak_selection_policy']='whole_brain_absolute_peak'
            row['hemisphere_constraint_applied']='False'
            row['selection_criterion']='maximum_absolute_amplitude_whole_brain'
    variant=pd.DataFrame(updated)
    if len(variant)!=4392 or variant.duplicated(KEYS).any():raise ValueError('Invalid variant size')
    if (variant.status=='PASS').sum()!=4378+len(overrides):raise ValueError('Incorrect PASS total')
    # Select retained PASS records using one consistent DataFrame.
    retained = variant.loc[
        (variant["sensitivity_override"].astype(str).str.lower() == "false")
        & (variant["status"] == "PASS")
    ]

    original_pass = df.loc[df["status"] == "PASS"]

    # Check that all original PASS records are retained.
    if set(map(key, original_pass.to_dict("records"))) != set(
        map(key, retained.to_dict("records"))
    ):
        raise ValueError("Original PASS records changed during merge")
        raise ValueError('Original PASS set changed')
    atomic_csv(out/'04_execution_status_variant.csv',variant)
    atomic_csv(out/'05_override_audit.csv',pd.DataFrame(audit))
    write_json(out/'06_manifest.json',{**provenance,'variant_sha256':sha(out/'04_execution_status_variant.csv'),
               'original_pass':4378,'original_fail':14,'substituted':len(overrides),
               'variant_pass':int((variant.status=='PASS').sum()),'variant_fail':int((variant.status=='FAIL').sum()),
               'note':'Sensitivity dataset contains mixed hemisphere peak-selection rules; original primary unchanged'})
    (out/'README.txt').write_text('PRIMARY: 05_single_reference_final/01_execution_status.csv, 4378 PASS, 14 FAIL.\n'
        'SENSITIVITY: 04_execution_status_variant.csv. Only hemisphere-empty candidates are rerun using a whole-brain peak.\n'
        'No active dipoles and zero-source cases remain FAIL; even a successful whole-brain result is not equivalent\n'
        'to a strict-hemisphere reconstruction. Not a cross-subject error metric or proof of convergence.\n')
    print('Variant saved:',out/'04_execution_status_variant.csv')
    print(f'Primary: 4378 PASS/14 FAIL; variant: {4378+len(overrides)} PASS/{14-len(overrides)} FAIL')

if __name__=='__main__':main()
