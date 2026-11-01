#!/usr/bin/env python3
"""Script 03F — Validate and finalize reference-based LOSO no-warp results.

GOAL
    Create ONE analysis-ready Script 03 folder. Preserve originals untouched.
    Merge original PASS rows with patch retries ONLY for original FAIL keys.
    Keep Gamma-MAP's confirmed no-left-hemisphere failure as FAIL.
    Never turn failures into PASS or silently delete cases.

INPUTS (from project root)
    cross_subject_no_warp/outputs/03_reference_based/
        01_execution_status.csv, 08_paired_input_audit.csv,
        10_channel_selection_summary.csv (if available)
    cross_subject_no_warp/outputs/03_reference_based_patch/
        01_execution_status.csv, 08_paired_input_audit.csv (if available),
        10_channel_selection_summary.csv (if available)
    Optional old-versus-new MNE control CSV for an exact comparison.

OUTPUTS (ONLY after every mandatory validation succeeds)
    cross_subject_no_warp/outputs/03_reference_based_final/
      01_execution_status.csv          exactly 1,464 uniquely keyed records
      02_method_condition_summary.csv PASS/FAIL counts
      03_merge_provenance.csv          row source and recorded hashes
      04_validation_report.json       counts, checks, input checksums
      05_channel_selection_summary.csv (if both source summaries are complete)
      06_paired_input_audit.csv        recomputed from final rows
      07_known_failure.json           Gamma-MAP original/retry diagnostic
      README.txt

RULES
    * Original PASS rows NEVER overwritten by retry rows.
    * Every original FAIL must be present exactly once in retry CSV.
    * Config, reference/test identity, nominal/actual channel policy and
      nonempty EEG/time hashes must be compatible for all completed pairs.
    * A row with FAIL keeps blank estimated peak columns.
    * Every method/test/run must have both conditions; 12 x 61 x 2 = 1464.
    * There must be exactly one retained Gamma-MAP failure, by default.
    * Refuse to overwrite an existing nonempty final output folder.
    * This is a RECORD-level merge audit, not independent solver reproduction.

COMMANDS
    # Validate, without writing final results:
    python cross_subject_no_warp/scripts/03_finalize_reference_based.py --check-only

    # Finalize after check-only passes:
    python cross_subject_no_warp/scripts/03_finalize_reference_based.py

    # For custom paths: --original-root PATH --patch-root PATH --output-root PATH
    # Optional: --control-status cross_subject_no_warp/outputs/03_patch_control_mne/01_execution_status.csv

DEPENDENCIES
    Python 3.10+, pandas. No EEG source data or inverse solvers needed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

METHODS = ['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC','LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
KEY = ['method','test_subject','run','condition']
PAIR = ['method','test_subject','run']
IDENTITY = ['fold_id','reference_subject','model_subject','configuration_key','configuration_parameters','montage_template_subject']
MATCHED = ['configuration_key','configuration_parameters','reference_subject','actual_channels_json','nominal_channels_json','replacement_details_json','original_bad_channels_json','reference_channel_scope','eeg_data_sha256','epoch_times_sha256','n_epochs','sampling_frequency_hz','highpass_hz','lowpass_hz','baseline_json','covariance_method_selected','hemisphere_constraint_applied']
FROZEN = ['method','test_subject','run','condition','fold_id','reference_subject','model_subject','configuration_key','configuration_parameters','montage_template_subject']
PEAKS = ['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm','peak_vertex','peak_time_s']

def die(message): raise ValueError(message)
def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def keyof(df):return pd.MultiIndex.from_frame(df[KEY])
def read_csv(path):
    if not path.is_file():die(f'Missing required file: {path}')
    return pd.read_csv(path,dtype=str,keep_default_na=False)
def verify_keys(df,label):
    needed=set(KEY+['status']+FROZEN+MATCHED+PEAKS)
    missing=needed-set(df)
    if missing:die(f'{label}: missing columns: {sorted(missing)}')
    if df[KEY].isna().any().any() or (df[KEY]=='').any().any():die(f'{label}: blank key')
    dupe=df.duplicated(KEY,keep=False)
    if dupe.any():die(f'{label}: duplicate jobs: {df.loc[dupe,KEY].head(5).to_dict("records")}')
    if (~df.status.isin(['PASS','FAIL'])).any():die(f'{label}: statuses other than PASS/FAIL')

def pair_audit(final):
    checks=[]
    for k,g in final.groupby(PAIR,sort=False):
        if set(g.condition)!= {'within','single_reference_no_warp'} or len(g)!=2:
            die(f'Missing/duplicate pair conditions: {k}')
        a=g.set_index('condition').loc['within'];b=g.set_index('condition').loc['single_reference_no_warp']
        compared=[c for c in MATCHED if a['status']=='PASS' and b['status']=='PASS' or c in ('configuration_key','configuration_parameters','reference_subject')]
        mismatch=[c for c in compared if a[c]!=b[c]]
        # For covariance estimator, numerical differences may be legitimate;
        # still require matching estimator name (as in original paired policy).
        if mismatch:die(f'Paired input mismatch {k}: {mismatch}')
        if a['status']=='PASS' and b['status']=='PASS' and (not a['eeg_data_sha256'] or not a['epoch_times_sha256']):
            die(f'Missing sample/time hashes for {k}')
        if a['model_subject']!=a['test_subject'] or b['model_subject']!=b['reference_subject']:
            die(f'Incorrect model mapping: {k}')
        checks.append(dict(zip(PAIR,k),identical_inputs=(a['status']=='PASS' and b['status']=='PASS'),fields_checked=';'.join(compared)))
    return pd.DataFrame(checks)

def merged_channels(original_root,patch_root,merged):
    paths=[original_root/'10_channel_selection_summary.csv',patch_root/'10_channel_selection_summary.csv']
    if not all(p.exists() for p in paths):return None
    summaries=[read_csv(p) for p in paths]
    if not all(set(KEY).issubset(s.columns) for s in summaries):die('Channel summary missing key columns')
    for i,summary in enumerate(summaries):
        if summary.duplicated(KEY).any():die(f'Duplicate channel summary keys in source {i}')
    keys=set(map(tuple,merged[KEY].itertuples(index=False,name=None)))
    # Prefer original channel report on original PASS; patch on original FAIL.
    wanted={tuple(row[c] for c in KEY):row['record_source'] for _,row in merged.iterrows()}
    # Rebuild from matching source individually; no hidden positional assumptions.
    parts=[]
    for source,sm in zip(('original','patch'),summaries):
        ks=[tuple(z) for z in sm[KEY].itertuples(index=False,name=None)]
        mask=[wanted.get(k)==source for k in ks]
        parts.append(sm.loc[mask].copy())
    out=pd.concat(parts,ignore_index=True)
    if len(out)!=len(merged) or out.duplicated(KEY).any():
        die(f'Channel summary incomplete: {len(out)} of {len(merged)}')
    joined=merged[KEY+['status','channels','actual_channels_json','original_bad_channels_json']].merge(out,on=KEY,validate='one_to_one',suffixes=('_record','_summary'))
    if 'status_summary' in joined and (joined.status_record!=joined.status_summary).any():
        die('Channel summary status mismatch')
    for r in joined.to_dict('records'):
        if r.get('status_record')!='PASS':continue
        if r.get('selected_include_bad','').lower() in ('true','1') or r.get('channel_names_unique','').lower() in ('false','0'):
            die(f'Channel validation failed: {[r[c] for c in KEY]}')
    return out.sort_values(KEY).reset_index(drop=True)

def channel_audit_from_records(final):
    """Build a complete self-contained channel audit from final status fields."""
    rows=[]
    for _,r in final.iterrows():
        def parse(col):
            if not r[col]:return None
            try:return json.loads(r[col])
            except (ValueError,TypeError):die(f'Unparseable {col} for {tuple(r[k] for k in KEY)}')
        actual=parse('actual_channels_json');nominal=parse('nominal_channels_json')
        bad=parse('original_bad_channels_json');replacement=parse('replacement_details_json')
        if r.status=='PASS':
            if not isinstance(actual,list) or not isinstance(bad,list):
                die(f'Missing actual/bad channels for successful job {tuple(r[k] for k in KEY)}')
            if len(actual)!=len(set(actual)) or any(c in bad for c in actual):
                die(f'Duplicate/bad channel selected: {tuple(r[k] for k in KEY)}')
            if int(float(r.channels))!=len(actual):
                die(f'Channel count mismatch: {tuple(r[k] for k in KEY)}')
        rows.append({**{k:r[k] for k in KEY}, 'status':r.status,
                     'record_source':r.record_source,
                     'reference_subject':r.reference_subject,
                     'model_subject':r.model_subject,
                     'montage_template_subject':r.montage_template_subject,
                     'n_nominal':len(nominal) if isinstance(nominal,list) else '',
                     'n_selected':len(actual) if isinstance(actual,list) else '',
                     'n_original_bad':len(bad) if isinstance(bad,list) else '',
                     'n_replacements':len(replacement) if isinstance(replacement,list) else '',
                     'channel_names_unique':len(actual)==len(set(actual)) if isinstance(actual,list) else '',
                     'selected_include_bad':bool(set(actual)&set(bad)) if isinstance(actual,list) and isinstance(bad,list) else '',
                     'actual_channels_json':r.actual_channels_json,
                     'nominal_channels_json':r.nominal_channels_json,
                     'original_bad_channels_json':r.original_bad_channels_json,
                     'replacement_details_json':r.replacement_details_json,
                     'eeg_data_sha256':r.eeg_data_sha256})
    return pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    base=Path('cross_subject_no_warp/outputs')
    ap.add_argument('--original-root',type=Path,default=base/'03_reference_based')
    ap.add_argument('--patch-root',type=Path,default=base/'03_reference_based_patch')
    ap.add_argument('--output-root',type=Path,default=base/'03_reference_based_final')
    ap.add_argument('--control-status',type=Path,default=None)
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args()
    oldpath=a.original_root/'01_execution_status.csv';newpath=a.patch_root/'01_execution_status.csv'
    old=read_csv(oldpath);new=read_csv(newpath)
    verify_keys(old,'original');verify_keys(new,'patch')
    if set(old.method)!=set(METHODS):die('Original methods not exactly the expected 12')
    if len(old)!=1464:die(f'Original must have 1,464 rows, found {len(old)}')
    if len(old[old.status=='PASS'])!=1361 or len(old[old.status=='FAIL'])!=103:
        die('Unexpected original PASS/FAIL counts; inspect manually')
    failed=old[old.status=='FAIL'];failedidx=keyof(failed)
    if set(keyof(new))!=set(failedidx):die('Patch keys differ from original FAIL keys')
    pre=failed.set_index(KEY);post=new.set_index(KEY)
    for col in FROZEN:
        if col in KEY:continue
        bad=(pre[col]!='') & (pre[col]!=post.loc[pre.index,col])
        if bad.any():die(f'Changed frozen input "{col}" for {list(pre.index[bad][:5])}')
    # For an originally failed row, provenance of selected channels may have been
    # blank because the solver failed before recording results. Do NOT compare
    # those values against the new successful reconstruction.
    kept=old[old.status=='PASS'].copy();kept['record_source']='original'
    repaired=new.copy();repaired['record_source']='patch'
    final=pd.concat([kept,repaired],ignore_index=True)
    verify_keys(final,'merged')
    if len(final)!=1464 or set(keyof(final))!=set(keyof(old)):die('Merged schedule mismatch')
    if (final.status=='PASS').sum()!=1463 or (final.status=='FAIL').sum()!=1:
        die('Unexpected merged PASS/FAIL count')
    fail=final[final.status=='FAIL']
    expected=('Gamma-MAP','sub-05','run-08','single_reference_no_warp')
    if tuple(fail.iloc[0][KEY])!=expected:die('Unexpected remaining failure identity')
    if 'no vertices' not in fail.iloc[0].failure_reason:die('Gamma-MAP failure reason differs')
    for c in PEAKS:
        if fail.iloc[0][c] not in ('','nan','NaN'):die(f'Failed case has populated {c}')
    paired=pair_audit(final)
    if len(paired)!=732:die('Incorrect paired case count')
    channels=channel_audit_from_records(final)
    # Optional numerical regression test of old PASS cases against re-run controls.
    if a.control_status:
        control=read_csv(a.control_status);verify_keys(control,'control')
        earlier=old.set_index(KEY)
        for _,r in control.iterrows():
            k=tuple(r[c] for c in KEY)
            if k not in earlier.index or earlier.loc[k,'status']!='PASS':die(f'Control not an original PASS: {k}')
            original=earlier.loc[k]
            ignored={'runtime_seconds'}
            for col in (set(old.columns)&set(control.columns))-ignored-set(KEY):
                if original[col] and r[col]:
                    try:
                        if abs(float(original[col])-float(r[col]))<1e-9:continue
                    except ValueError:pass
                if original[col]!=r[col]:die(f'Control changed {col}: {k}')
    provenance=final[KEY+['status','record_source','configuration_key','eeg_data_sha256','epoch_times_sha256']].copy()
    provenance['source_csv_sha256']=provenance.record_source.map({'original':filehash(oldpath),'patch':filehash(newpath)})
    summary=final.groupby(['method','condition','status']).size().rename('count').reset_index()
    report={'status':'RECORD_AUDIT_PASSED','n_expected':1464,'pass':1463,'fail':1,
            'original_pass_reused':1361,'patched_pass':102,'patched_fail':1,
            'pairs':len(paired),'channel_summary_complete':True,
            'optional_control_checked':bool(a.control_status),
            'caution':'Record checks do NOT prove old solver results reproducible for every method, validity of unregistered anatomical localization errors, or convergence of every PASS.',
            'source_sha256':{str(oldpath):filehash(oldpath),str(newpath):filehash(newpath)},
            'known_failure':dict(zip(KEY,expected))}
    print('RECORD AUDIT PASSED | 1464 rows | 1463 PASS | 1 FAIL | 732 paired cases')
    print('Channel audit reconstructed from all records:',len(channels))
    if a.check_only:
        print('Check-only: no outputs written');return
    if a.output_root.exists() and any(a.output_root.iterdir()):
        die(f'Final output not empty: {a.output_root}. Never overwrite; inspect first.')
    a.output_root.mkdir(parents=True,exist_ok=True)
    final=final.sort_values(KEY).reset_index(drop=True)
    final.to_csv(a.output_root/'01_execution_status.csv',index=False)
    summary.to_csv(a.output_root/'02_method_condition_summary.csv',index=False)
    provenance.to_csv(a.output_root/'03_merge_provenance.csv',index=False)
    (a.output_root/'04_validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    channels.to_csv(a.output_root/'05_channel_selection_summary.csv',index=False)
    paired.to_csv(a.output_root/'06_paired_input_audit.csv',index=False)
    (a.output_root/'07_known_failure.json').write_text(json.dumps({'job':dict(zip(KEY,expected)),'failure_reason':fail.iloc[0].failure_reason,
        'diagnostic':'Earlier diagnostic log: vertices [0,1]; requested lh; convergence warning. Log should be archived separately.'},indent=2)+'\n')
    (a.output_root/'README.txt').write_text('Script 03 final RECORD-level audit passed. Use this folder for subsequent result processing. Original and patch folders are immutable archives. Gamma-MAP sub-05 run-08 no-warp remains FAIL. No anatomical error computed. See 04_validation_report.json.\n')
    print('Final written:',a.output_root)

if __name__=='__main__':main()
