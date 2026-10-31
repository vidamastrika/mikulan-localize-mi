#!/usr/bin/env python3
"""Script 05 final audit — single-reference, no-warp reconstruction.

PURPOSE
  Read the immutable Script 05 execution CSV, validate the complete 4,392-job
  schedule, check all 84 reference-method selections from Script 04, inspect
  each PASS reconstruction's channel lists and metadata, and preserve failures.
  Create a separate verified output directory; NEVER alter the original CSV.
  This is a RECORD/PROVENANCE audit, not numerical solver reproduction or
  proof that unregistered head coordinates permit localization-error scoring.

DEFAULT INPUTS (run from project root)
  cross_subject_no_warp/outputs/05_single_reference_reconstruction/01_execution_status.csv
  cross_subject_no_warp/outputs/04_single_reference_parameters/01_single_reference_selected_parameters.csv
  Optional: Script 05 manifest / channel audits (checksums retained if present).

COMMANDS
  python cross_subject_no_warp/scripts/05_finalize_single_reference.py --check-only
  python cross_subject_no_warp/scripts/05_finalize_single_reference.py
  python cross_subject_no_warp/scripts/05_finalize_single_reference.py --input-status PATH --selection PATH --output-root PATH

OUTPUTS (in outputs/05_single_reference_final)
  01_execution_status.csv            Original rows, unchanged except sorted order
  02_method_reference_summary.csv    Method/reference success and failure totals
  03_failed_cases.csv                All failed records, unchanged reasons
  04_channel_selection_audit.csv     PASS channel metadata, counts and fingerprints
  05_validation_report.json          Counts, checks, SHA256 of input files
  06_provenance.csv                   Source file SHA256 for each output dataset
  README.txt                         Limits and provenance

POLICY
  A complete experiment has exactly 12 methods x 61 test runs x 6 other
  references = 4,392 unique attempts. FAIL records are retained, NOT repaired.
  No reference can equal its test subject. Every PASS must have a finite
  estimate, valid channel list excluding test-bad channels, and recorded EEG
  fingerprint. Single-reference parameter selection must match Script 04.
  The destination must not preexist; a temporary directory is published only
  after all checks and outputs succeed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter, defaultdict

METHODS=('MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC','LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME')
SUBJECTS=tuple(f'sub-{i:02d}' for i in range(1,8))
KEY=('method','reference_subject','test_subject','run','condition')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def fail(message):raise ValueError('AUDIT FAILED: '+message)

def parse_json_list(value,key):
    try:v=json.loads(value)
    except Exception as e:fail(f'Invalid {key}: {e}')
    if not isinstance(v,list):fail(f'{key} is not a JSON list')
    return v

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    base=Path('cross_subject_no_warp')
    p.add_argument('--input-status',type=Path,default=base/'outputs/05_single_reference_reconstruction/01_execution_status.csv')
    p.add_argument('--selection',type=Path,default=base/'outputs/04_single_reference_parameters/01_single_reference_selected_parameters.csv')
    p.add_argument('--output-root',type=Path,default=base/'outputs/05_single_reference_final')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    for path in (a.input_status,a.selection):
        if not path.is_file():fail(f'Input not found: {path}')
    records=read_csv(a.input_status); selected=read_csv(a.selection)
    if len(records)!=4392:fail(f'Expected 4392 rows, got {len(records)}')
    if len(selected)!=84:fail(f'Expected 84 selected configurations, got {len(selected)}')
    sel={}
    for r in selected:
        key=r['method'],r['reference_subject']
        if key in sel:fail(f'Duplicate parameter selection {key}')
        if r.get('selection_status')!='SELECTED' or r.get('eligible','').strip().lower() not in ('true','1'):
            fail(f'Ineligible selection: {key}')
        if r.get('training_subjects')!=r['reference_subject']:fail(f'Non-reference training: {key}')
        sel[key]=r
    if set(sel)!={(m,s) for m in METHODS for s in SUBJECTS}:fail('Missing reference-method selections')
    keys=set(); coverage=defaultdict(set); counts=Counter(); channel_audit=[]; failed=[]
    for i,r in enumerate(records,1):
        key=tuple(r.get(k,'') for k in KEY)
        if key in keys:fail(f'Duplicate result: {key}')
        keys.add(key)
        m,ref,test,run,condition=key
        if m not in METHODS or ref not in SUBJECTS or test not in SUBJECTS or ref==test:fail(f'Invalid reference/test job {key}')
        if condition!='single_reference_no_warp':fail(f'Unexpected condition {key}')
        if r.get('model_subject')!=ref:fail(f'Wrong model subject {key}')
        if r.get('parameter_selection')!='single_reference':fail(f'Wrong parameter strategy {key}')
        chosen=sel[(m,ref)]
        if r.get('configuration_key')!=chosen['configuration_key']:fail(f'Configuration key changed {key}')
        if r.get('montage_template_subject') not in ('',ref):fail(f'Wrong montage template {key}')
        if r.get('status') not in ('PASS','FAIL'):fail(f'Invalid status {key}')
        counts[(m,ref,r['status'])]+=1
        coverage[(m,ref)].add((test,run))
        if r['status']=='FAIL':
            if not r.get('failure_reason'):fail(f'FAIL missing reason {key}')
            failed.append(r)
            continue
        if r.get('failure_reason'):fail(f'PASS with failure reason {key}')
        if r.get('montage_template_subject')!=ref:fail(f'PASS missing reference montage {key}')
        for fld in ('estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm'):
            try:val=float(r[fld])
            except Exception:fail(f'PASS missing coordinate {fld} {key}')
            if not math.isfinite(val):fail(f'Nonfinite coordinate {key}')
        channels=parse_json_list(r.get('actual_channels_json',''),f'actual channels {key}')
        bad=parse_json_list(r.get('original_bad_channels_json',''),f'bad channels {key}')
        nominal=parse_json_list(r.get('nominal_channels_json',''),f'nominal channels {key}')
        replacements=parse_json_list(r.get('replacement_details_json',''),f'replacements {key}')
        if not channels or len(set(channels))!=len(channels):fail(f'Empty or duplicate channels {key}')
        if set(channels)&set(bad):fail(f'Bad channel included {key}')
        if int(float(r.get('channels',0)))!=len(channels):fail(f'Channel count mismatch {key}')
        for fld in ('eeg_data_sha256','epoch_times_sha256'):
            x=r.get(fld,'');
            if len(x)!=64 or any(c not in '0123456789abcdef' for c in x):fail(f'Missing/malformed {fld} {key}')
        if int(float(r.get('n_epochs',0)))<1:fail(f'No epochs {key}')
        channel_audit.append(dict(method=m,reference_subject=ref,test_subject=test,run=run,
            condition=condition,status='PASS',montage_template_subject=ref,
            n_selected_channels=len(channels),n_bad_test_channels=len(bad),
            n_nominal_channels=len(nominal),n_replacements=len(replacements),
            actual_channels_json=r['actual_channels_json'],nominal_channels_json=r['nominal_channels_json'],
            replacement_details_json=r['replacement_details_json'],original_bad_channels_json=r['original_bad_channels_json'],
            eeg_data_sha256=r['eeg_data_sha256'],epoch_times_sha256=r['epoch_times_sha256']))
    # Derive exact 61 distinct test runs from records, independently of reference duplication.
    runpairs={(r['test_subject'],r['run']) for r in records}
    if len(runpairs)!=61 or {test for test,_ in runpairs}!=set(SUBJECTS):fail(f'Expected 61 test runs across 7 subjects, got {len(runpairs)}')
    expected={(m,ref,test,run,'single_reference_no_warp') for m in METHODS for test,run in runpairs for ref in SUBJECTS if ref!=test}
    if keys!=expected:fail(f'Incorrect schedule: missing {len(expected-keys)}, extras {len(keys-expected)}')
    if len(channel_audit)!=4378 or len(failed)!=14:fail(f'Unexpected PASS/FAIL: {len(channel_audit)}/{len(failed)}; inspect before finalizing')
    summary=[]
    for m in METHODS:
        for ref in SUBJECTS:
            pa=counts[(m,ref,'PASS')];fa=counts[(m,ref,'FAIL')]
            expected_n=sum(1 for t,_ in runpairs if t!=ref)
            if pa+fa!=expected_n:fail(f'Coverage mismatch {m}/{ref}')
            summary.append(dict(method=m,reference_subject=ref,expected=expected_n,pass_count=pa,fail_count=fa))
    report={'status':'RECORD_AUDIT_PASSED','expected_jobs':4392,'observed_jobs':len(records),
            'pass_count':len(channel_audit),'fail_count':len(failed),'distinct_test_runs':len(runpairs),
            'parameter_configurations':len(sel),'source_status_path':str(a.input_status.resolve()),
            'source_selection_path':str(a.selection.resolve()),'source_status_sha256':sha(a.input_status),
            'source_selection_sha256':sha(a.selection),
            'limitations':['Record audit only; not a numerical solver rerun',
                           'No cross-subject localization error is computed',
                           'No anatomical correspondence is established between unregistered head models']}
    print('RECORD AUDIT PASSED | 4392 jobs | 4378 PASS | 14 FAIL | 84 selections | 61 test runs')
    if a.check_only:
        print('Check-only: no files written');return
    if a.output_root.exists():fail(f'Output directory already exists; refusing overwrite: {a.output_root}')
    a.output_root.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.05_final_tmp_',dir=a.output_root.parent))
    try:
        write_csv(stage/'01_execution_status.csv',sorted(records,key=lambda r:tuple(r[k] for k in KEY)),list(records[0]))
        write_csv(stage/'02_method_reference_summary.csv',summary,list(summary[0]))
        write_csv(stage/'03_failed_cases.csv',failed,list(records[0]))
        write_csv(stage/'04_channel_selection_audit.csv',channel_audit,list(channel_audit[0]))
        (stage/'05_validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
        provenance=[{'artifact':path.name,'sha256':sha(path)} for path in sorted(stage.glob('*'))]
        provenance.extend([{'artifact':'INPUT '+str(a.input_status),'sha256':sha(a.input_status)},
                           {'artifact':'INPUT '+str(a.selection),'sha256':sha(a.selection)}])
        write_csv(stage/'06_provenance.csv',provenance,['artifact','sha256'])
        (stage/'README.txt').write_text('Final Script 05 RECORD audit: 4392 attempts, 4378 PASS, 14 FAIL.\n'
            'FAIL records are retained. No parameters were changed. No localization errors calculated.\n'
            'Input source checksums and output hashes appear in provenance and validation report.\n'
            'Use this dataset as the canonical single-reference reconstruction status for Script 06.\n'
            'Source reconstruction success does not establish ground-truth coordinate comparability.\n')
        os.rename(stage,a.output_root)
    except Exception:
        shutil.rmtree(stage,ignore_errors=True);raise
    print('Final written:',a.output_root)

if __name__=='__main__':main()
