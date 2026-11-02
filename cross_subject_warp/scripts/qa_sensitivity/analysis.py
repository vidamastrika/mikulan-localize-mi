"""New QA015, saved-peak sensitivity, and matched cross-policy comparison stages."""
from __future__ import annotations
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import core as c


def run_qa(args):
    dataset=c.protect(args)
    qa,mapping,ev,base,ledger,primary,parent,exe=c.context(args,full=True)
    c.require_manifest(args.attribution_root,'manifest.json','ATTRIBUTION_CONTROLS_PASS',ledger)
    precision_path=args.attribution_root/'05_precision_and_vector_convention.json'
    ledger.track(precision_path)
    field_basis=np.asarray(json.loads(precision_path.read_text())['component_to_ras'],float)
    rows=[]
    for reference in qa.SUBJECTS:
        for test in qa.SUBJECTS:
            if reference==test:continue
            print('Revalidating unchanged MRI gates:',reference,'->',test,flush=True)
            rows.extend(c.validate_pair(mapping,exe,qa,args.resource_root,dataset,reference,test,field_basis))
    table=pd.DataFrame(rows)
    old={tuple(r[k] for k in c.PAIR):r for r in primary.to_dict('records')}
    parity=[]
    for row in rows:
        before=old[tuple(row[k] for k in c.PAIR)]
        row['primary_qa_status']=before['status'];row['primary_reason']=before['reason']
        row['qa_policy']=c.SENSITIVITY.name;row['pair_tolerance_mm']=c.SENSITIVITY.pair_tolerance_mm
        row['source_tolerance_mm']=c.SENSITIVITY.source_tolerance_mm
        row['qa_sensitivity_status']='PASS_QA015_ONLY' if row[f'{c.SENSITIVITY.name}_pass'] else 'FAIL_QA015'
        parity.append(dict(**{k:row[k] for k in c.PAIR},original_primary_pass=before['status']=='PASS_NUMERICAL_MAPPING_QA',
                           reproduced_primary_pass=row[f'{c.PRIMARY.name}_pass'],
                           matches=(before['status']=='PASS_NUMERICAL_MAPPING_QA')==row[f'{c.PRIMARY.name}_pass']))
    table=pd.DataFrame(rows);counts=[]
    for policy in [c.PRIMARY,c.SENSITIVITY]:
        for mode in ['affine','syn','ALL']:
            part=table if mode=='ALL' else table[table.registration_mode.eq(mode)]
            passed=int(part[f'{policy.name}_pass'].sum())
            counts.append(dict(qa_policy=policy.name,pair_tolerance_mm=policy.pair_tolerance_mm,
                               source_tolerance_mm=policy.source_tolerance_mm,registration_mode=mode,
                               checks=len(part),passed=passed,failed=len(part)-passed))
    complete=all(r['matches'] for r in parity) and len(rows)==84
    c.publish(args,ledger,dict(status='SENSITIVITY_QA_COMPLETE' if complete else 'SENSITIVITY_QA_BLOCKED_PRIMARY_PARITY',
        primary_parity_pass=complete,checks=84,new_policy_passed=int(table[f'{c.SENSITIVITY.name}_pass'].sum()),
        automatic_primary_reclassification=False,posthoc_exploratory=True,source_point_tolerance_mm=.1,
        runtime_validation=dict(primary_parity_passed=sum(r['matches'] for r in parity),primary_parity_failed=sum(not r['matches'] for r in parity),
            qa015_passed=int(table[f'{c.SENSITIVITY.name}_pass'].sum()),qa015_failed=int((~table[f'{c.SENSITIVITY.name}_pass']).sum()))),
        {'01_pair_mode_qa.csv':table,'02_threshold_counts.csv':pd.DataFrame(counts),'03_primary_parity.csv':pd.DataFrame(parity)})
    return 0 if complete else 2


def labelled(rows,policy):
    rows=rows.copy();rows['qa_policy']=policy.name;rows['peak_scenario']=rows.scenario
    rows['pair_tolerance_mm']=policy.pair_tolerance_mm;rows['source_tolerance_mm']=policy.source_tolerance_mm
    return rows


def same_common(primary,sensitivity):
    keys=[*c.KEY,'registration_mode'];merged=primary.merge(sensitivity,on=keys,suffixes=('_primary','_sensitivity'),validate='one_to_one')
    if len(merged)!=len(primary) or len(primary)!=len(sensitivity):raise ValueError('Policy row schedules differ')
    for field in ['status','configuration_key','configuration_parameters','failure_reason']:
        if not merged[f'{field}_primary'].fillna('').astype(str).eq(merged[f'{field}_sensitivity'].fillna('').astype(str)).all():
            raise ValueError(f'Native reconstruction changed across QA policies: {field}')
    for field in ['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm']:
        a=pd.to_numeric(merged[f'{field}_primary'],errors='coerce');b=pd.to_numeric(merged[f'{field}_sensitivity'],errors='coerce')
        if not a.isna().equals(b.isna()) or not np.allclose(a.dropna(),b.dropna(),atol=1e-12,rtol=0):raise ValueError(f'Native peak changed: {field}')
    primary_valid=c.yes(merged.localization_error_valid_primary);sensitivity_valid=c.yes(merged.localization_error_valid_sensitivity)
    if (primary_valid&~sensitivity_valid).any():raise ValueError('Pair-only sensitivity lost previously valid primary observations')
    common=primary_valid&sensitivity_valid
    for field in ['localization_error_mm','mapped_test_ras_x_mm','mapped_test_ras_y_mm','mapped_test_ras_z_mm']:
        a=pd.to_numeric(merged.loc[common,f'{field}_primary'],errors='coerce')
        b=pd.to_numeric(merged.loc[common,f'{field}_sensitivity'],errors='coerce')
        # Within mapped fields are blank in both policies.
        valid=a.notna()|b.notna()
        if not np.isfinite(a[valid]).all() or not np.isfinite(b[valid]).all() or not np.allclose(a[valid],b[valid],atol=1e-9,rtol=0):
            raise ValueError(f'Common-cohort numerical regression: {field}')
    within=merged.condition.eq('within')
    if not (c.yes(merged.loc[within,'localization_error_valid_primary'])==c.yes(merged.loc[within,'localization_error_valid_sensitivity'])).all():raise ValueError('Within coverage changed')
    return merged,common


def evaluate(args):
    dataset=c.protect(args)
    qa,mapping,ev,base,ledger,old_pairs,parent,exe=c.context(args,full=True)
    c.require_manifest(args.sensitivity_qa,'manifest.json','SENSITIVITY_QA_COMPLETE',ledger)
    table=c.read(args.sensitivity_qa/'01_pair_mode_qa.csv')
    ledger.track(args.sensitivity_qa/'01_pair_mode_qa.csv')
    if table.duplicated(c.PAIR).any() or set(map(tuple,table[c.PAIR].values))!=set(map(tuple,old_pairs[c.PAIR].values)):raise ValueError('QA015 coverage differs')
    if set(table.qa_policy)!={c.SENSITIVITY.name} or not pd.to_numeric(table.pair_tolerance_mm).eq(.15).all() or not pd.to_numeric(table.source_tolerance_mm).eq(.1).all():raise ValueError('Policy thresholds differ')
    # Explicitly labelled policy adapter for the unchanged mapping kernel. The
    # original primary status is a separate immutable field, never overwritten.
    effective={}
    for r in table.to_dict('records'):
        key=tuple(r[k] for k in c.PAIR)
        if r['qa_sensitivity_status'] not in ['PASS_QA015_ONLY','FAIL_QA015']:raise ValueError('Unknown QA015 status')
        r['status']='PASS_NUMERICAL_MAPPING_QA' if r['qa_sensitivity_status']=='PASS_QA015_ONLY' else 'FAIL'
        r['reason']=r.get('reason','') or 'QA015 unchanged-gate or roundtrip failure'
        effective[key]=r
    primary_manifest=args.primary_evaluation/'07_evaluation_manifest.json'
    old_state=json.loads(primary_manifest.read_text());ledger.track(primary_manifest)
    if old_state.get('status')!='EVALUATION_COMPLETE_WITH_RETAINED_FAILURES' or old_state.get('mapping_execution_failures')!=0:raise ValueError('Primary evaluation incomplete')
    for entry in json.loads((args.primary_evaluation/'11_source_provenance.json').read_text()):ledger.track(entry['path'],entry['sha256'])
    matrices=json.loads((args.bridge_root/'02_native_physical_bridges.json').read_text())
    ledger.track(args.bridge_root/'02_native_physical_bridges.json')
    gt_table=c.frame(args.primary_evaluation/'02_coordinate_validation.csv');ledger.track(args.primary_evaluation/'02_coordinate_validation.csv')
    truth={}
    for r in gt_table.to_dict('records'):
        truth[(r['test_subject'],r['run'])]=dict(head_mm=np.array([r[f'gt_test_head_{a}_mm'] for a in 'xyz']),
            ras_mm=np.array([r[f'gt_test_ras_{a}_mm'] for a in 'xyz']),stimulation_pair=r['stimulation_pair'],hemisphere=r['hemisphere'])
    if len(truth)!=61:raise ValueError('Ground-truth schedule differs')
    data={}
    for label,(_,_,folder,_,_) in base.SPECS.items():
        path=c.EXP/'outputs'/folder/'01_execution_status.csv';ledger.track(path)
        # Native checksum is already tied to the primary evaluation provenance.
        data[label]=base.read(path)
    images={}
    def load_image(path):
        path=Path(path).resolve()
        if path not in images:images[path]=nib.load(path)
        return images[path]
    # qa.MAX_ROUNDTRIP_MM remains .1. Only the explicit pair eligibility map
    # differs. No global or primary helper constant is altered.
    if qa.MAX_ROUNDTRIP_MM!=c.SENSITIVITY.source_tolerance_mm:raise ValueError('Source-point validation policy changed')
    rows,raw=ev.evaluate_records(data,base.SPECS,truth,matrices,effective,qa,mapping,exe,args.resource_root,load_image)
    rows=labelled(rows,c.SENSITIVITY)
    old=[]
    for file in ['03_primary_localization.csv','04_sensitivity_localization.csv']:
        path=args.primary_evaluation/file;ledger.track(path);old.append(c.frame(path))
    original=pd.concat(old,ignore_index=True)
    merged,common=same_common(original,rows)
    if len(rows)!=21960 or rows.duplicated([*c.KEY,'registration_mode']).any():raise ValueError('Evaluation schedule differs')
    primary_map={tuple(r[k] for k in c.PAIR):r['status'] for r in old_pairs.to_dict('records')}
    rows['primary_qa_status']=[primary_map.get(tuple(r[k] for k in c.PAIR),'NOT_APPLICABLE') for r in rows.to_dict('records')]
    summary,matches=ev.summaries_and_matches(rows,base.KEY)
    for t in [summary,matches,raw]:
        t['qa_policy']=c.SENSITIVITY.name;t['peak_scenario']=t.scenario
    failed=rows[~rows.localization_error_valid]
    execution=int(rows.evaluation_status.eq('MAPPING_EXECUTION_FAIL').sum())
    tables={'01_input_validation.csv':pd.DataFrame([dict(check='native_results_and_common_policy_parity',passed=True,common_valid_rows=int(common.sum()))]),
        '02_coordinate_validation.csv':gt_table,'03_primary_localization.csv':rows[rows.scenario.eq('primary')],
        '04_sensitivity_localization.csv':rows[rows.scenario.eq('sensitivity')],
        '05_method_summary.csv':summary,'06_failure_analysis.csv':failed,
        '08_raw_head_diagnostic.csv':raw,'09_registration_readiness.csv':table,'12_matched_affine_syn.csv':matches,
        '13_source_point_mapping_audit.csv':rows[[*c.KEY,'registration_mode','qa_policy','peak_scenario','mapping_status','source_point_roundtrip_mm','evaluation_failure_reason']]}
    manifest=dict(status='SENSITIVITY_EVALUATION_COMPLETE' if not execution else 'SENSITIVITY_EVALUATION_EXECUTION_ERRORS',
        qa_policy=c.SENSITIVITY.name,source_point_tolerance_mm=.1,pair_tolerance_mm=.15,
        rows=len(rows),mapping_execution_failures=execution,common_valid_rows=int(common.sum()),
        runtime_validation=dict(schedule_rows_checked=len(rows),common_cohort_parity_pass=True,
            mapping_execution_failures=execution,point_validation_exclusions=int(rows.evaluation_status.isin(['POINT_DOMAIN_FAIL','POINT_ROUNDTRIP_FAIL']).sum())),
        posthoc_exploratory=True,localization_errors_calculated=True,
        interpretation='Eligibility sensitivity only; unchanged transforms, peaks and source threshold. Not improved registration.')
    # Retain the familiar numbered evaluation manifest in addition to the new
    # stage manifest, which hashes all outputs and is consumed by comparison.
    c.publish(args,ledger,manifest,tables,{'07_evaluation_manifest.json':manifest})
    return 2 if execution else 0


def matched(rows,ev,base,policy):
    summary,m=ev.summaries_and_matches(rows,base.KEY)
    for t in [summary,m]:t['qa_policy']=policy.name;t['peak_scenario']=t.scenario
    return summary,m


def paired_stats(matches,cohort):
    records=[]
    for cols in [['qa_policy','strategy','scenario','method'],['qa_policy','strategy','scenario']]:
        for values,g in matches.groupby(cols,sort=True):
            record=dict(zip(cols,values));record.setdefault('method','ALL_DESCRIPTIVE')
            d=pd.to_numeric(g.syn_minus_affine_mm)
            record.update(cohort=cohort,matched_n=len(g),affine_median_mm=pd.to_numeric(g.localization_error_mm_affine).median(),
                syn_median_mm=pd.to_numeric(g.localization_error_mm_syn).median(),median_paired_delta_mm=d.median(),
                mean_paired_delta_mm=d.mean(),q25_paired_delta_mm=d.quantile(.25),q75_paired_delta_mm=d.quantile(.75),
                subject_pairs=len(g[['reference_subject','test_subject']].drop_duplicates()))
            records.append(record)
    return pd.DataFrame(records,columns=['qa_policy','strategy','scenario','method','cohort','matched_n',
        'affine_median_mm','syn_median_mm','median_paired_delta_mm','mean_paired_delta_mm',
        'q25_paired_delta_mm','q75_paired_delta_mm','subject_pairs'])


def compare(args):
    c.protect(args)
    # Comparison is real historical-output validation, not a tool/field run.
    qa,mapping,ev,base=c.modules();ledger=c.Ledger()
    c.require_manifest(args.sensitivity_evaluation,'manifest.json','SENSITIVITY_EVALUATION_COMPLETE',ledger)
    tables=[]
    for root,policy in [(args.primary_evaluation,c.PRIMARY),(args.sensitivity_evaluation,c.SENSITIVITY)]:
        parts=[]
        for file in ['03_primary_localization.csv','04_sensitivity_localization.csv']:
            path=root/file;ledger.track(path);parts.append(c.frame(path))
        tables.append(labelled(pd.concat(parts,ignore_index=True),policy))
    primary,sensitivity=tables;merged,common=same_common(primary,sensitivity)
    summ0,m0=matched(primary,ev,base,c.PRIMARY);summ1,m1=matched(sensitivity,ev,base,c.SENSITIVITY)
    match_keys=c.KEY
    common_matches=m0.merge(m1[match_keys],on=match_keys,validate='one_to_one')
    new=m1.merge(m0[match_keys],on=match_keys,how='left',indicator=True,validate='one_to_one')
    new=new[new['_merge'].eq('left_only')].drop(columns='_merge')
    common_keys=common_matches[match_keys]
    common1=m1.merge(common_keys,on=match_keys,validate='one_to_one')
    common_stats=pd.concat([paired_stats(common_matches,'common_both_policies'),paired_stats(common1,'common_both_policies')],ignore_index=True)
    all_matches=pd.concat([m0,m1],ignore_index=True)
    counts=pd.concat([summ0,summ1],ignore_index=True)
    # Verify existing within baseline against actual completed no-warp output.
    no_warp=c.ROOT/'cross_subject_no_warp/outputs/06_evaluate_localization'
    regress=[]
    for scenario,file in [('primary','03_primary_localization.csv'),('sensitivity','04_sensitivity_localization.csv')]:
        path=no_warp/file;ledger.track(path)
        old=c.frame(path);old=old[old.condition.eq('within')]
        new_within=primary[primary.condition.eq('within')&primary.scenario.eq(scenario)]
        keys=[k for k in c.KEY if k!='scenario']
        join=new_within.merge(old,on=keys,suffixes=('_warp','_no_warp'),validate='one_to_one')
        if len(join)!=732 or not np.allclose(pd.to_numeric(join.localization_error_mm_warp),pd.to_numeric(join.localization_error_mm_no_warp),atol=1e-9,rtol=0):raise ValueError('No-warp within regression failed')
        regress.append(dict(check='historical_no_warp_within',scenario=scenario,rows=len(join),passed=True,
            max_difference_mm=float(np.max(np.abs(pd.to_numeric(join.localization_error_mm_warp)-pd.to_numeric(join.localization_error_mm_no_warp))))))
    for path in sorted((c.EXP/'scripts/qa_sensitivity').glob('*.py')):ledger.track(path)
    c.publish(args,ledger,dict(status='THRESHOLD_COMPARISON_COMPLETE',primary_unchanged=True,common_numerical_parity_pass=True,
        common_valid_rows=int(common.sum()),newly_eligible_matched_observations=len(new),posthoc_exploratory=True,
        inference='Descriptive paired differences; no unplanned significance tests or independent-sample assumptions.',
        runtime_validation=dict(common_cohort_parity_pass=True,historical_no_warp_checks_passed=len(regress),historical_no_warp_checks_failed=0)),
        {'01_coverage_ledger.csv':counts,'02_within_policy_matched_statistics.csv':paired_stats(all_matches,'within_policy'),
         '03_common_cohort_statistics.csv':common_stats,'04_newly_eligible_observations.csv':new,
         '05_newly_eligible_statistics.csv':paired_stats(new,'newly_eligible'),
         '06_historical_regression_checks.csv':pd.DataFrame(regress),'07_all_matched_observations.csv':all_matches})
    return 0
