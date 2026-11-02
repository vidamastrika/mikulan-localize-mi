#!/usr/bin/env python3
# SCRIPT 06 HELPER — Diagnose prior QA failures without changing acceptance rules.
# From project root, with the project's .venv Python and ANTs runtime:
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_diagnose_registration_roundtrips.py
#
# For an alternate prior report and a NEW diagnostic output directory:
# python cross_subject_warp/scripts/06_diagnose_registration_roundtrips.py \
#   --qa-root cross_subject_warp/outputs/06_registration_qa \
#   --output-root cross_subject_warp/outputs/06_roundtrip_diagnostics_review
#
# Help: python cross_subject_warp/scripts/06_diagnose_registration_roundtrips.py --help
# Exit 0 means diagnostics written, NOT registration QA passed.
# Only prior failed mode/pairs are sampled. No fitting, changed thresholds,
# localization errors, EEG loading, QA-status replacement or input modification.
"""Measure both MRI-only roundtrip directions for failed registration QA cases.

Repeat the original 200-point foreground samples with seed 2901 and the same
0.1 mm threshold/75% coverage. Report percentiles, counts above tolerance and
per-point coordinates/validity. This is diagnostic evidence, not a replacement
QA run or permission to use a failed pair in the primary analysis.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'
KEY=['reference_subject','test_subject','registration_mode']


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj


def measure(point_map,exe,source,destination,forward_terms,inverse_terms,qa):
    points=qa.foreground_points(source)
    mapped=np.asarray(point_map(exe,points,forward_terms),float)
    if mapped.shape!=points.shape or not np.isfinite(mapped).all():raise ValueError('Invalid forward mapped coordinates')
    vox=nib.affines.apply_affine(np.linalg.inv(destination.affine),mapped)
    valid=np.all((vox>=1)&(vox<np.asarray(destination.shape[:3])-2),axis=1)
    back=np.full_like(points,np.nan);errors=np.full(len(points),np.nan)
    if valid.any():
        returned=np.asarray(point_map(exe,mapped[valid],inverse_terms),float)
        if returned.shape!=points[valid].shape or not np.isfinite(returned).all():raise ValueError('Invalid returned points')
        back[valid]=returned;errors[valid]=np.linalg.norm(returned-points[valid],axis=1)
    finite=errors[valid]
    coverage=float(valid.mean())
    exceeds=valid&(errors>qa.MAX_ROUNDTRIP_MM)
    summary=dict(tested=len(points),valid=int(valid.sum()),coverage=coverage,
        coverage_pass=coverage>=qa.MIN_COVERAGE,over_threshold=int(exceeds.sum()),threshold_mm=qa.MAX_ROUNDTRIP_MM,
        median_mm=float(np.median(finite)) if len(finite) else None,
        p95_mm=float(np.percentile(finite,95)) if len(finite) else None,
        p99_mm=float(np.percentile(finite,99)) if len(finite) else None,
        maximum_mm=float(finite.max()) if len(finite) else None,
        roundtrip_pass=bool(len(finite) and finite.max()<=qa.MAX_ROUNDTRIP_MM))
    source_vox=nib.affines.apply_affine(np.linalg.inv(source.affine),points)
    edge=np.min(np.column_stack([source_vox,np.asarray(source.shape[:3])-1-source_vox]),axis=1)
    rows=[]
    for i in range(len(points)):
        row=dict(sample_index=i,destination_interior=bool(valid[i]),roundtrip_mm=float(errors[i]) if valid[i] else None,
                 over_threshold=bool(exceeds[i]),source_fov_edge_distance_voxels=float(edge[i]))
        for prefix,values in [('source_ras',points),('mapped_ras',mapped),('returned_ras',back)]:
            for axis,j in zip('xyz',range(3)):row[f'{prefix}_{axis}_mm']=float(values[i,j]) if np.isfinite(values[i,j]) else None
        rows.append(row)
    return summary,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qa-root',type=Path,default=EXP/'outputs/06_registration_qa')
    p.add_argument('--resource-root',type=Path,default=EXP/'source/01_registration_resources/outputs')
    p.add_argument('--output-root',type=Path,default=EXP/'outputs/06_roundtrip_diagnostics')
    args=p.parse_args()
    for name,value in vars(args).items():setattr(args,name,value.expanduser().resolve())
    if args.output_root.exists() or args.output_root.is_symlink():raise FileExistsError('Output exists; use a NEW --output-root')
    protected=[args.qa_root,args.resource_root,EXP/'source',EXP/'scripts',ROOT/'cross_subject',ROOT/'cross_subject_no_warp']
    protected += [EXP/'outputs'/name for name in ['03_reference_based_final','05_single_reference_final',
        '03_gamma_wholebrain_sensitivity','05_wholebrain_hemisphere_sensitivity','03_05_native_completion_audit',
        '06_evaluate_localization','06_coordinate_bridge']]
    for path in protected:
        if args.output_root==path or path in args.output_root.parents or args.output_root in path.parents:
            raise ValueError(f'Output overlaps protected input: {path}')
    qa_path=EXP/'scripts/06_validate_registration_resources.py'
    qa=module(qa_path,'roundtrip_qa_checks')
    manifest_path=args.qa_root/'05_registration_qa_manifest.json'
    table_path=args.qa_root/'01_pair_mode_qa.csv'
    provenance_path=args.qa_root/'04_source_provenance.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('mode_pair_checks_expected')!=84 or manifest.get('pairs_expected')!=42:raise ValueError('Unexpected parent QA design')
    if manifest.get('samples')!=qa.SAMPLES or manifest.get('foreground_seed')!=2901 or manifest.get('max_roundtrip_mm')!=qa.MAX_ROUNDTRIP_MM or manifest.get('min_coverage')!=qa.MIN_COVERAGE:
        raise ValueError('Parent QA sampling/thresholds differ from the checked implementation')
    parent={str(Path(r['path']).resolve()):r for r in json.loads(provenance_path.read_text())}
    tracked={}
    def track(path,parent_required=False):
        path=Path(path).resolve();actual=qa.sha(path)
        if parent_required and (str(path) not in parent or parent[str(path)]['sha256']!=actual):
            raise ValueError(f'Input differs from parent QA: {path}')
        tracked[str(path)]=dict(path=str(path),sha256=actual,size_bytes=path.stat().st_size)
    track(qa_path,True)
    table=pd.read_csv(table_path,dtype=str,keep_default_na=False)
    expected={(r,t,m) for r in qa.SUBJECTS for t in qa.SUBJECTS if r!=t for m in ['affine','syn']}
    if table.duplicated(KEY).any() or set(map(tuple,table[KEY].values))!=expected:raise ValueError('Parent pair/mode coverage differs')
    if not table.status.isin(['PASS_NUMERICAL_MAPPING_QA','FAIL']).all():raise ValueError('Unknown parent status')
    if int(table.status.eq('PASS_NUMERICAL_MAPPING_QA').sum())!=manifest['mode_pair_checks_passed']:
        raise ValueError('Parent CSV/manifest counts disagree')
    failed=table.loc[table.status.eq('FAIL')]
    if failed.empty:raise ValueError('No prior failed cases to diagnose')
    mapping,source_path=qa.historical_mapping();track(source_path,True)
    exe=qa.locate_ants(Path(manifest['ants_executable']));track(exe,True)
    if qa.sha(exe)!=manifest['ants_executable_sha256']:raise ValueError('Point-transform executable changed')
    # Validate every required input against the prior QA BEFORE any ANTs call.
    for row in failed.to_dict('records'):
        for name in ['fixed_mri','moving_mri']:track(row[name],True)
        files=qa.files_for_pair(args.resource_root,row['reference_subject'],row['test_subject'])
        needed=['aff','aff_image','aff_log'] if row['registration_mode']=='affine' else ['syn_aff','warp','invwarp','syn_image','syn_inverse_image','syn_log','syn_quality']
        for name in needed:track(files[name],True)
        for image in [Path(row['fixed_mri']),Path(row['moving_mri'])]:
            if image.parent.name!='anat' or image.parents[2].name!='epochs' or image.parents[3].name!='derivatives':
                raise ValueError('Unexpected historical dataset MRI path layout')
            dataset=image.parents[4]
            if args.output_root==dataset or dataset in args.output_root.parents or args.output_root in dataset.parents:
                raise ValueError('Output overlaps original dataset')
    summaries=[];samples=[]
    for row in failed.to_dict('records'):
        context={k:row[k] for k in KEY};context['original_qa_reason']=row['reason']
        files=qa.files_for_pair(args.resource_root,row['reference_subject'],row['test_subject']);terms=mapping.terms(files)
        prefix='aff' if row['registration_mode']=='affine' else 'syn'
        fixed=nib.load(row['fixed_mri']);moving=nib.load(row['moving_mri'])
        print('Diagnosing',row['reference_subject'],'->',row['test_subject'],row['registration_mode'],flush=True)
        for direction,source,destination,forward_terms,inverse_terms in [
            ('reference_to_test_to_reference',moving,fixed,terms[f'{prefix}_r2t'],terms[f'{prefix}_t2r']),
            ('test_to_reference_to_test',fixed,moving,terms[f'{prefix}_t2r'],terms[f'{prefix}_r2t'])]:
            try:
                summary,points=measure(mapping.point_map,str(exe),source,destination,forward_terms,inverse_terms,qa)
                summaries.append(dict(**context,direction=direction,diagnostic_status='MEASURED',**summary))
                samples.extend(dict(**context,direction=direction,**point) for point in points)
                print(f'  {direction}: {summary["over_threshold"]}/{summary["valid"]} above 0.1 mm; max={summary["maximum_mm"]}',flush=True)
            except Exception as exc:
                summaries.append(dict(**context,direction=direction,diagnostic_status='ERROR',diagnostic_error=f'{type(exc).__name__}: {exc}'))
    for path in [manifest_path,table_path,provenance_path,Path(__file__)]:track(path)
    for path,item in tracked.items():
        if qa.sha(path)!=item['sha256']:raise ValueError(f'Input changed during diagnostics: {path}')
    result=dict(status='DIAGNOSTICS_ONLY_PARENT_QA_UNCHANGED',failed_mode_pairs=len(failed),directions_requested=len(summaries),directions_measured=sum(r['diagnostic_status']=='MEASURED' for r in summaries),
        diagnostic_errors=sum(r['diagnostic_status']=='ERROR' for r in summaries),threshold_mm=qa.MAX_ROUNDTRIP_MM,
        samples=qa.SAMPLES,seed=2901,qa_status_changed=False,anatomical_evaluation_ready=False,
        inverse_reconstruction_executed=False,registration_fitting_executed=False,localization_errors_calculated=False,
        parent_qa=str(args.qa_root),next_step='Review direction/coverage/tail statistics. Preserve the original failed-pair flags; do not relax primary thresholds after inspecting outcomes.')
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.roundtrip_diagnostics_',dir=args.output_root.parent) as temp:
        stage=Path(temp)/'results';stage.mkdir()
        pd.DataFrame(summaries).to_csv(stage/'01_direction_summary.csv',index=False)
        pd.DataFrame(samples).to_csv(stage/'02_point_diagnostics.csv',index=False)
        (stage/'03_source_provenance.json').write_text(json.dumps(list(tracked.values()),indent=2)+'\n')
        (stage/'04_diagnostic_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        (stage/'README.txt').write_text(__doc__+'\nNo previously failed QA case was reclassified.\n')
        if args.output_root.exists():raise FileExistsError(args.output_root)
        stage.rename(args.output_root)
    print('Diagnostics:',args.output_root,'; original QA flags remain unchanged.')
    return 2 if result['diagnostic_errors'] else 0


if __name__=='__main__':raise SystemExit(main())
