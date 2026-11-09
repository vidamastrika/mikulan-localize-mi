#!/usr/bin/env python3
# OBJECTIVE / PRIMARY POLICY
# Compare template/fsaverage and original cross-subject no-warp using their
# existing unregistered HEAD-coordinate distances. NO MRI alignment. This is a
# coordinate diagnostic, not anatomically validated localization accuracy.
# INPUTS (read only)
# Script 04 job diagnostics/manifest; no-warp Script 06 raw diagnostic CSV,
# primary evaluated ledger and manifest; recorded primary-comparison policy;
# Script 26 palette/style definitions (AST literals only, no script execution),
# plus its complete_all_pass historical 12-method selected baseline (732 rows).
# Baseline is contextual: within-subject anatomical error with in-sample global
# parameter selection, unlike cross-model unregistered distances. No mixed-metric
# baseline delta or percentage improvement is calculated. Repeated references
# collapse within each run before adding one baseline value.
# Script 05 MRI registrations, MRI images, native transforms, sensor positions,
# EEG arrays and sensitivity outcomes are never used as comparison inputs.
# DESIGN / VALIDATION
# 7 subjects, 61 runs, 12 methods; 732 LOSO + 4,392 Single Reference = 5,124.
# Match full strategy/method/reference/test/run identities. Require identical
# frozen configuration keys and parameters. Check paired PASS EEG metadata and
# identical audited test HEAD ground truth. Recompute both stored diagnostics
# from saved HEAD coordinates to check units, metric and ledger consistency.
# Use ONLY no-warp scenario=primary, condition=single_reference_no_warp; exclude
# within-subject and mixed-policy sensitivity rows. Both primary failure statuses
# remain in denominators; never impute failed distances. Paired deltas use only
# BOTH_PASS and mean template distance MINUS no-warp distance (mm).
# Aggregate repeated reference evaluations within each run and expose both pooled
# and equal-participant summaries. No statistical independence or p-value claims.
# Configurations/preprocessing match; geometry and selected channel identities
# may differ by design. Both anatomy and sensor geometry change from no-warp.
# OUTPUTS (fresh variant output root only)
# tables/01_paired_jobs.csv; 02_method_summary.csv; 03_participant_summary.csv;
# 04_reference_summary.csv; 05_equal_participant_summary.csv;
# 06_unique_run_summary.csv; 07_input_inventory.csv; 08_nonpaired_jobs.csv;
# 09_configuration_preprocessing_audit.csv; tables 10-15 include run balancing
# and historical baseline context. Nine 300-dpi PNG comparison figures;
# comparison_manifest.json, REPORT.md, method_styles.csv and script snapshot.
# PARAMETERS / DEPENDENCIES
# --diagnostic-root, --no-warp-root, --baseline-root, --output-root, --dpi 300,
# --no-figures. Default baseline is Script 26 complete_all_pass.
# Python >=3.10, NumPy/pandas; Matplotlib for figures. No MNE/ANTs or SciPy.
# All inputs/outputs hashed; no overwrite, fitting, parameter selection or retry.
# Exact expected current pairs: 5,109 BOTH_PASS, 10 TEMPLATE_ONLY_PASS, 5 BOTH_FAIL.
# Counts are validated from input statuses, not forced to these historical values.
# COMMAND (repository root; user executes manually):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/06_compare_no_warp.py
# Tables only (use a fresh output root):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/06_compare_no_warp.py \
#   --no-figures --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/06_compare_no_warp_tables
# Exit 0 = diagnostic comparison complete, NOT anatomical validation.
"""Paired primary no-registration comparison: template/fsaverage versus no-warp."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

VARIANT=Path(__file__).resolve().parents[1]
ROOT=VARIANT.parents[1]
KEY=['strategy','method','reference_subject','test_subject','run']
STRATEGIES=['LOSO','Single-reference']
METHODS=['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','MxNE','RAP-MUSIC','LCMV',
         'irMxNE','Champagne','Gamma-MAP','SESAME']
PREPROCESS=['n_epochs','sampling_frequency_hz','highpass_hz','lowpass_hz','baseline_json',
            'epoch_times_sha256','reference_channel_scope','original_bad_channels_json']


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def equivalent(a,b,field):
    if field.endswith('_json'):return json.loads(a)==json.loads(b)
    if field in ['n_epochs','sampling_frequency_hz','highpass_hz','lowpass_hz']:
        return np.isclose(float(a),float(b),rtol=0,atol=1e-10)
    return a==b


def styles_from_script26(path):
    """Read only literal palette/style data; never import or run Script 26."""
    tree=ast.parse(path.read_text())
    assignments={target.id:n.value for n in tree.body if isinstance(n,ast.Assign)
                 for target in n.targets if isinstance(target,ast.Name)}
    colors=ast.literal_eval(assignments['COLORS'])
    order=sum((ast.literal_eval(assignments[n]) for n in ['ORIGINAL','OTHER','NEW']),[])
    expression=assignments['PARTICIPANT_STYLES']
    require(isinstance(expression,ast.Call) and isinstance(expression.func,ast.Name)
            and expression.func.id=='dict' and len(expression.args)==1,'Script 26 style schema changed')
    zipped=expression.args[0]
    require(isinstance(zipped,ast.Call) and isinstance(zipped.func,ast.Name)
            and zipped.func.id=='zip' and len(zipped.args)==2,'Script 26 zip schema changed')
    styles=dict(zip(order,ast.literal_eval(zipped.args[1])))
    require(set(order)==set(METHODS) and set(colors)==set(METHODS) and set(styles)==set(METHODS),
            'Incomplete Script 26 method styling')
    return colors,styles


def summary(data,groups):
    records=[]
    for values,g in data.groupby(groups,dropna=False,sort=True):
        if not isinstance(values,tuple):values=(values,)
        paired=g.loc[g.pair_status.eq('BOTH_PASS')]
        delta=paired.paired_delta_mm
        old=g.no_warp_distance_mm.dropna();new=g.template_distance_mm.dropna()
        records.append(dict(zip(groups,values))|dict(
            n_planned=len(g),n_unique_runs=len(g[['test_subject','run']].drop_duplicates()),
            no_warp_pass=int(g.no_warp_status.eq('PASS').sum()),template_pass=int(g.template_status.eq('PASS').sum()),
            no_warp_fail=int(g.no_warp_status.eq('FAIL').sum()),template_fail=int(g.template_status.eq('FAIL').sum()),
            no_warp_pass_rate=float(g.no_warp_status.eq('PASS').mean()),template_pass_rate=float(g.template_status.eq('PASS').mean()),
            n_paired=len(paired),n_template_only_pass=int(g.pair_status.eq('TEMPLATE_ONLY_PASS').sum()),
            n_no_warp_only_pass=int(g.pair_status.eq('NO_WARP_ONLY_PASS').sum()),n_both_fail=int(g.pair_status.eq('BOTH_FAIL').sum()),
            no_warp_all_pass_median_mm=old.median(),template_all_pass_median_mm=new.median(),
            no_warp_paired_median_mm=paired.no_warp_distance_mm.median(),
            template_paired_median_mm=paired.template_distance_mm.median(),
            paired_median_delta_mm=delta.median(),paired_mean_delta_mm=delta.mean(),
            paired_delta_q25_mm=delta.quantile(.25),paired_delta_q75_mm=delta.quantile(.75),
            n_template_smaller_diagnostic=int((delta < -1e-9).sum()),
            n_template_larger_diagnostic=int((delta > 1e-9).sum()),
            n_tied_diagnostic=int((delta.abs()<=1e-9).sum())))
    return pd.DataFrame(records)


def figures(data,method,participant,out,colors,styles,dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    out.mkdir()
    written=[]
    for strategy in STRATEGIES:
        selected=data.loc[data.strategy.eq(strategy)&data.pair_status.eq('BOTH_PASS')]
        fig,ax=plt.subplots(figsize=(18,8))
        for offset,column,alpha in [(-.19,'no_warp_distance_mm',.35),(.19,'template_distance_mm',.85)]:
            arrays=[selected.loc[selected.method.eq(m),column].to_numpy(float) for m in METHODS]
            boxes=ax.boxplot(arrays,positions=np.arange(12)+offset,widths=.33,patch_artist=True,
                             showfliers=True,manage_ticks=False,
                             flierprops=dict(marker='o',markersize=2,alpha=.3),
                             medianprops=dict(color='#222222',linewidth=1.1))
            for m,box in zip(METHODS,boxes['boxes']):box.set(facecolor=colors[m],alpha=alpha)
        ax.set_xticks(np.arange(12),METHODS,rotation=35,ha='right')
        ax.set_ylabel('Unregistered HEAD distance (mm)')
        ax.set_title(f'{strategy}: identical successful job pairs\nCoordinate diagnostics only; no MRI alignment')
        ax.legend(handles=[Patch(facecolor='#555555',alpha=.35,label='No-warp'),
                           Patch(facecolor='#555555',alpha=.85,label='Template/fsaverage')],frameon=False)
        ax.grid(axis='y',alpha=.2);fig.tight_layout()
        name=f'0{1 if strategy=="LOSO" else 2}_{strategy.lower()}_paired_distributions.png'
        fig.savefig(out/name,dpi=dpi,bbox_inches='tight');plt.close(fig);written.append(name)
    def differences(table,filename,title):
        fig,axes=plt.subplots(1,2,figsize=(17,8),sharey=True)
        for ax,strategy in zip(axes,STRATEGIES):
            g=table.loc[table.strategy.eq(strategy)].set_index('method').reindex(METHODS)
            values=g.paired_median_delta_mm.to_numpy(float)
            ax.barh(np.arange(12),values,color=[colors[m] for m in METHODS],alpha=.82)
            ax.set_yticks(np.arange(12),METHODS);ax.axvline(0,color='#333333',lw=1)
            ax.set_title(strategy);ax.set_xlabel('Median paired difference: template − no-warp (mm)')
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
            for i,(value,count) in enumerate(zip(values,g.n_paired)):
                if np.isfinite(value):ax.text(value,i,f' {value:+.2f} (n={int(count)}) ',va='center',
                                              ha='left' if value>=0 else 'right',fontsize=8)
            ax.margins(x=.28)
        axes[0].invert_yaxis()
        fig.suptitle(title+'\nNegative = smaller coordinate separation; anatomical accuracy unvalidated')
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/filename,dpi=dpi,bbox_inches='tight');plt.close(fig)
        written.append(filename)
    differences(method,'03_paired_median_differences.png','All seven participants: paired coordinate differences')
    fig,axes=plt.subplots(1,2,figsize=(18,8),sharey=True)
    for ax,strategy in zip(axes,STRATEGIES):
        pivot=participant.loc[participant.strategy.eq(strategy)].pivot(index='test_subject',columns='method',values='paired_median_delta_mm')
        for m in METHODS:
            marker,line=styles[m]
            ax.plot(np.arange(len(pivot)),pivot[m],color=colors[m],alpha=.82,marker=marker,linestyle=line,label=m,lw=1.3)
        ax.axhline(0,color='#555555',lw=1);ax.set_xticks(np.arange(len(pivot)),pivot.index,rotation=25)
        ax.set_title(strategy);ax.set_xlabel('Participant');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Participant median paired HEAD-distance difference (mm)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=6,loc='lower center',frameon=False)
    fig.suptitle('Paired coordinate differences by participant (includes sub-03)\nUnregistered diagnostics; no MRI alignment')
    fig.tight_layout(rect=(0,.12,1,.92));name='04_participant_paired_differences.png'
    fig.savefig(out/name,dpi=dpi,bbox_inches='tight');plt.close(fig);written.append(name)
    fig,axes=plt.subplots(1,2,figsize=(18,7),sharey=True)
    for ax,strategy in zip(axes,STRATEGIES):
        g=method.loc[method.strategy.eq(strategy)].set_index('method').reindex(METHODS)
        for offset,column,alpha in [(-.18,'no_warp_fail',.35),(.18,'template_fail',.85)]:
            bars=ax.bar(np.arange(12)+offset,g[column],width=.34,color=[colors[m] for m in METHODS],alpha=alpha)
            ax.bar_label(bars,labels=[str(int(x)) if x else '' for x in g[column]],padding=3)
        ax.set_xticks(np.arange(12),METHODS,rotation=40,ha='right');ax.set_title(strategy);ax.set_xlabel('Method')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Failed jobs (all planned jobs retained)')
    fig.legend(handles=[Patch(facecolor='#555555',alpha=.35,label='No-warp'),Patch(facecolor='#555555',alpha=.85,label='Template/fsaverage')],
               loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.9));name='05_failure_counts.png'
    fig.savefig(out/name,dpi=dpi,bbox_inches='tight');plt.close(fig);written.append(name)
    differences(participant.loc[participant.test_subject.eq('sub-03')],
                '06_sub03_paired_differences.png','Sub-03: paired coordinate differences (included in primary cohort)')
    return written


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-root',type=Path,default=VARIANT/'outputs/04_evaluate_diagnostics')
    p.add_argument('--no-warp-root',type=Path,default=ROOT/'cross_subject_no_warp/outputs/06_evaluate_localization')
    p.add_argument('--baseline-root',type=Path,default=ROOT/'outputs/26_twelve_methods_comparison/complete_all_pass')
    p.add_argument('--output-root',type=Path,default=VARIANT/'outputs/06_compare_no_warp')
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--no-figures',action='store_true')
    args=p.parse_args(argv)
    require(args.dpi>0,'DPI must be positive')
    newroot=args.diagnostic_root.expanduser().resolve();oldroot=args.no_warp_root.expanduser().resolve()
    out=args.output_root.expanduser().resolve()
    require((VARIANT/'outputs').resolve() in out.parents and not out.exists(),'Use a fresh directory inside this variant/outputs')
    hashes={};inventory=[]
    def verify(path,role,expected=None):
        path=Path(path).resolve();value=sha(path)
        require(expected is None or value==expected,f'Input SHA mismatch: {path}')
        hashes[str(path)]=value;inventory.append(dict(role=role,path=str(path),sha256=value,size_bytes=path.stat().st_size))
        require(path.parent not in out.parents and out!=path.parent,'Output overlaps inputs')
        return path
    manifest_path=verify(newroot/'12_evaluation_manifest.json','template_diagnostic_manifest')
    manifest=json.loads(manifest_path.read_text())
    require(manifest['status']=='DIAGNOSTIC_EVALUATION_COMPLETE_ANATOMICAL_ERROR_BLOCKED','Completed Script 04 required')
    newpath=verify(newroot/'01_run_level_diagnostics.csv','template_primary_diagnostics',manifest['output_sha256']['01_run_level_diagnostics.csv'])
    oldpath=verify(oldroot/'08_raw_head_diagnostic.csv','no_warp_raw_diagnostics')
    primary_path=verify(oldroot/'03_primary_localization.csv','no_warp_primary_coordinates_configurations')
    verify(oldroot/'07_evaluation_manifest.json','no_warp_evaluation_manifest')
    policy=json.loads(verify(VARIANT/'assumptions/primary_comparison_no_mri_alignment.json','user_primary_policy').read_text())
    require(policy['MRI_registration_used_in_primary_metric'] is False and policy['no_warp_scenario']=='primary','Primary policy differs')
    verify(newpath,'policy_pinned_template_input',policy['input_sha256'][policy['template_input']])
    verify(oldpath,'policy_pinned_no_warp_input',policy['input_sha256'][policy['no_warp_input']])
    style_path=verify(ROOT/'scripts/26_compare_twelve_methods.py','shared_plot_palette_source')
    verify(Path(__file__),'comparison_script')
    baseline_root=args.baseline_root.expanduser().resolve()
    baseline_path=verify(baseline_root/'tables/13_selected_configuration_results_long.csv','historical_selected_baseline')
    baseline_definitions=verify(baseline_root/'tables/01_selected_configuration_definitions.csv','historical_baseline_parameters')
    helper_path=verify(Path(__file__).with_name('comparison_baseline_context.py'),'baseline_context_helper')
    import comparison_baseline_context as baseline_context
    colors,styles=styles_from_script26(style_path)
    new=pd.read_csv(newpath,keep_default_na=False)
    raw=pd.read_csv(oldpath,keep_default_na=False)
    raw=raw.loc[raw.scenario.eq('primary')].copy()
    primary=pd.read_csv(primary_path,keep_default_na=False)
    old=primary.loc[primary.scenario.eq('primary')&primary.condition.eq('single_reference_no_warp')].copy()
    for name,frame in [('template',new),('no_warp_raw',raw),('no_warp_primary',old)]:
        require(len(frame)==5124 and not frame.duplicated(KEY).any(),f'{name}: expected 5,124 unique jobs')
        require(set(frame.status)<={'PASS','FAIL'},f'{name}: incomplete/unknown status')
        require(frame.groupby('strategy').size().to_dict()=={'LOSO':732,'Single-reference':4392},f'{name}: strategy counts differ')
        require(set(frame.method)==set(METHODS) and frame.test_subject.nunique()==7
                and len(frame[['test_subject','run']].drop_duplicates())==61,f'{name}: method/subject/run coverage differs')
    new=new.set_index(KEY).sort_index();old=old.set_index(KEY).sort_index();raw=raw.set_index(KEY).sort_index()
    require(new.index.equals(old.index) and new.index.equals(raw.index),'Planned comparison keys differ')
    require(old.status.eq(raw.status).all(),'No-warp raw/primary statuses differ')
    require(new.localization_error_valid.astype(str).str.lower().eq('false').all()
            and raw.valid_localization_error.astype(str).str.lower().eq('false').all(),
            'Primary inputs must retain diagnostic-only validity flags')
    require(new.coordinate_frame.loc[new.status.eq('PASS')].eq('template_head').all(),'Unexpected template coordinate frame')
    require(old.coordinate_frame.loc[old.status.eq('PASS')].eq('model_head').all(),'Unexpected no-warp coordinate frame')
    require(new.mapping_status.eq('ASSUMED_UNVERIFIED').all(),'Template mapping policy changed')
    for frame in [new,old]:
        require(frame.loc[frame.status.eq('PASS'),'coordinate_units'].eq('mm').all(),'Coordinate units must be mm')
    old_distance=pd.to_numeric(raw.raw_head_separation_mm,errors='coerce')
    new_distance=pd.to_numeric(new.diagnostic_distance_mm,errors='coerce')
    checks=[]
    for index,n in new.iterrows():
        o=old.loc[index]
        require(n.configuration_key==o.configuration_key and json.loads(n.configuration_parameters)==json.loads(o.configuration_parameters),
                f'Frozen configuration mismatch: {index}')
        for a in 'xyz':require(abs(float(n[f'gt_head_{a}_mm'])-float(o[f'gt_head_{a}_mm']))<=1e-8,'Ground-truth references differ')
        both=n.status==o.status=='PASS'
        if both:
            require(n.stimulation_pair==o.stimulation_pair and n.hemisphere==o.hemisphere,
                    f'Paired stimulation identity differs: {index}')
        metadata_ok=all(equivalent(n[field],o[field],field) for field in PREPROCESS) if both else None
        require(metadata_ok is not False,f'Paired preprocessing mismatch: {index}')
        for r,distance,frame in [(n,new_distance.loc[index],'template_head'),(o,old_distance.loc[index],'reference_model_head')]:
            if r.status=='PASS':
                predicted=np.array([float(r[f'estimated_head_{a}_mm']) for a in 'xyz'])
                gt=np.array([float(r[f'gt_head_{a}_mm']) for a in 'xyz'])
                require(np.isfinite(predicted).all() and np.isfinite(distance) and distance>=0
                        and abs(np.linalg.norm(predicted-gt)-distance)<=1e-6,f'Inconsistent {frame} diagnostic: {index}')
            else:require(pd.isna(distance),f'Failed job has a distance: {index}')
        checks.append(dict(zip(KEY,index))|dict(configuration_match=True,ground_truth_match=True,
            paired_PASS_metadata_checked=both,preprocessing_match=metadata_ok,
            actual_channels_equal=(json.loads(n.actual_channels_json)==json.loads(o.actual_channels_json)) if both else None,
            channel_identity_difference_is_expected=True))
    pair=new.reset_index()[KEY+['configuration_key','configuration_parameters']].copy()
    for prefix,frame,distance in [('no_warp',old,old_distance),('template',new,new_distance)]:
        pair[f'{prefix}_status']=frame.status.to_numpy()
        pair[f'{prefix}_distance_mm']=distance.to_numpy(float)
        pair[f'{prefix}_failure_reason']=frame.failure_reason.to_numpy()
    oldpass=pair.no_warp_status.eq('PASS');newpass=pair.template_status.eq('PASS')
    pair['pair_status']=np.select([oldpass&newpass,~oldpass&newpass,oldpass&~newpass],
                                  ['BOTH_PASS','TEMPLATE_ONLY_PASS','NO_WARP_ONLY_PASS'],default='BOTH_FAIL')
    pair['paired_delta_mm']=np.where(oldpass&newpass,pair.template_distance_mm-pair.no_warp_distance_mm,np.nan)
    pair['metric']='unregistered_HEAD_coordinate_separation_mm'
    pair['valid_anatomical_error']=False
    method=summary(pair,['strategy','method'])
    participant=summary(pair,['strategy','method','test_subject'])
    reference=summary(pair.loc[pair.strategy.eq('Single-reference')],['strategy','method','reference_subject'])
    runs=summary(pair,['strategy','method','test_subject','run'])
    # Give each EEG run equal weight after collapsing repeated reference pairs,
    # then expose equal-participant summaries as a separate descriptive view.
    subject_balanced=runs.groupby(['strategy','method','test_subject'],as_index=False).agg(
        n_runs=('run','count'),n_runs_with_pairs=('paired_median_delta_mm','count'),
        participant_median_of_run_paired_medians_mm=('paired_median_delta_mm','median'))
    equal=subject_balanced.groupby(['strategy','method'],as_index=False).agg(
        n_subjects=('test_subject','nunique'),n_subjects_with_pairs=('participant_median_of_run_paired_medians_mm','count'),
        mean_participant_balanced_delta_mm=('participant_median_of_run_paired_medians_mm','mean'),
        median_participant_balanced_delta_mm=('participant_median_of_run_paired_medians_mm','median'))
    baseline,baseline_configs,baseline_runs,baseline_methods,baseline_participants=baseline_context.prepare(
        baseline_path,baseline_definitions,runs,METHODS)
    for path,h in hashes.items():require(sha(path)==h,f'Input changed during comparison: {path}')
    out.mkdir(parents=True,exist_ok=False);tables=out/'tables';tables.mkdir()
    outputs={
        '01_paired_jobs.csv':pair,'02_method_summary.csv':method,'03_participant_summary.csv':participant,
        '04_reference_summary.csv':reference,'05_equal_participant_summary.csv':equal,
        '06_unique_run_summary.csv':runs,'07_input_inventory.csv':pd.DataFrame(inventory),
        '08_nonpaired_jobs.csv':pair.loc[~pair.pair_status.eq('BOTH_PASS')],
        '09_configuration_preprocessing_audit.csv':pd.DataFrame(checks),
        '10_participant_balanced_run_summary.csv':subject_balanced,
        '11_historical_baseline_unique_runs.csv':baseline,
        '12_historical_baseline_configuration_definitions.csv':baseline_configs,
        '13_baseline_context_unique_runs.csv':baseline_runs,
        '14_baseline_context_method_summary.csv':baseline_methods,
        '15_baseline_context_participant_summary.csv':baseline_participants}
    for name,table in outputs.items():table.to_csv(tables/name,index=False)
    pd.DataFrame([dict(method=m,color=colors[m],marker=styles[m][0],linestyle=styles[m][1]) for m in METHODS]).to_csv(out/'method_styles.csv',index=False)
    shutil.copy2(Path(__file__),out/'script_snapshot.py.txt')
    shutil.copy2(helper_path,out/'baseline_context_helper_snapshot.py.txt')
    written=[] if args.no_figures else figures(pair,method,participant,out/'figures',colors,styles,args.dpi)
    if not args.no_figures:
        written+=baseline_context.figures(baseline_runs,baseline_methods,baseline_participants,
                                         out/'figures',colors,styles,METHODS,args.dpi)
    counts={str(k):int(v) for k,v in pair.pair_status.value_counts().items()}
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PRIMARY_NO_REGISTRATION_DIAGNOSTIC_COMPARISON_COMPLETE',
        planned_jobs=len(pair),pair_status_counts=counts,input_sha256=hashes,
        output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        figures=written,dpi=args.dpi,no_warp_scenario='primary',
        difference_definition='template diagnostic minus no-warp diagnostic, mm; BOTH_PASS only',
        MRI_alignment_used=False,Script05_inputs_used=False,anatomical_error_valid=False,
        inverse_reconstruction_executed=False,frozen_configurations_match=True,
        paired_preprocessing_metadata_matches=True,
        historical_baseline_rows=len(baseline),baseline_role='context_only_different_metric_and_selection',
        baseline_pseudoreplication_avoided=True,
        tie_tolerance_mm=1e-9,
        limitations=['Both metrics subtract model/template HEAD from individual HEAD without anatomical alignment.',
            'Smaller diagnostic coordinate separation is not proof of better anatomical accuracy.',
            'Single Reference repeats each run under six reference parameter owners; jobs are not independent participants.',
            'Paired delta median differs from subtracting two condition medians; both condition medians are shown separately.',
            'Paired analyses omit nonpaired distances but preserve every failed job in completion tables and denominators.',
            'Nominal and actual channel identities may differ with geometry even though configuration settings and preprocessing match.',
            'Both anatomy and sensor geometry differ; this is not an anatomy-only comparison.',
            'No mixed-peak-policy no-warp sensitivity results or MRI-based supplementary results enter this primary report.',
            'Historical baseline uses within-subject anatomical errors and global in-sample selected parameters; it is contextual, not a matched anatomical-accuracy benchmark.'])
    lines=['# Primary no-registration diagnostic comparison','',
        '**Primary no-warp/template values are unregistered HEAD-coordinate distances, not validated anatomical localization errors.** '
        'Historical baseline within-subject errors appear separately as context.','',
        f'Planned jobs: {len(pair):,}. Paired PASS: {counts.get("BOTH_PASS",0):,}; template-only PASS: {counts.get("TEMPLATE_ONLY_PASS",0)}; '
        f'no-warp-only PASS: {counts.get("NO_WARP_ONLY_PASS",0)}; both FAIL: {counts.get("BOTH_FAIL",0)}.','',
        'All 7 participants, including sub-03, and all 61 EEG runs are included. Only original no-warp primary cross-subject rows enter this comparison. '
        'No-warp/template frozen configurations match. Checked EEG preprocessing metadata agrees on paired successes. '
        'No MRI mapping, inverse fits, parameter changes or sensitivity substitution.','',
        'Difference = template distance minus no-warp distance on identical successful jobs. Negative means smaller **coordinate separation**, '
        'not established anatomical improvement. Median paired difference is not the difference between condition medians.','',
        '| Strategy | Method | Paired jobs | No-warp paired median (mm) | Template paired median (mm) | Median paired delta (mm) | Failures no-warp / template |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for strategy in STRATEGIES:
        for m in METHODS:
            r=method.loc[method.strategy.eq(strategy)&method.method.eq(m)].iloc[0]
            lines.append(f'| {strategy} | {m} | {int(r.n_paired)} | {r.no_warp_paired_median_mm:.2f} | {r.template_paired_median_mm:.2f} | '
                         f'{r.paired_median_delta_mm:+.2f} | {int(r.no_warp_fail)} / {int(r.template_fail)} |')
    lines += ['', '## Output guide','',
        'tables/01: full paired ledger; 02: method summaries; 03: participant summaries; 04: reference-owner summaries; '
        '05: equal-participant summaries after reference/run balancing; 06: unique-run summaries; 07: input hashes; '
        '08: all nonpaired/failing jobs; 09: configuration/preprocessing checks; 10: participant run-balanced medians.','',
        'Figures: 01/02 paired distributions by strategy; 03 median paired differences; 04 participant profiles; '
        '05 failure counts; 06 explicit sub-03 comparison. Method colors/markers follow Script 26.','',
        'Single Reference uses six parameter owners per test run; these are repeated evaluations, not six independent recordings. '
        'Use the run, participant and reference tables alongside pooled summaries. No p-values or independence claims are made.','',
        'MRI-based Script 05 outputs remain supplementary and were not read. Full limitations and input/output hashes are in comparison_manifest.json.']
    lines+=baseline_context.report_lines(baseline_methods,METHODS)
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    report['output_sha256']['REPORT.md']=sha(out/'REPORT.md')
    (out/'comparison_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{report["status"]}: {out}')
    print(f'Paired job counts: {counts}')


if __name__=='__main__':
    main()
