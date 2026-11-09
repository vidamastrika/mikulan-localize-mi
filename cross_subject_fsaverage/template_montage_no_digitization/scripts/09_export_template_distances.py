#!/usr/bin/env python3
"""Export clear distance tables for the completed template/fsaverage experiment.

Uses saved Script 04 coordinates only. Checks no individual geometry was used in
reconstruction, recomputes unregistered template-HEAD to test-HEAD distances on
PASS jobs, and verifies exact agreement with the existing primary diagnostics.
No MRI mapping, test electrode positions, forward models, EEG or inverse fits
are loaded/computed. Failed jobs retain missing distances. Physical channel
mapping remains ASSUMED_UNVERIFIED; distances are not validated anatomical errors.

Writes new 00_template_fsaverage_distances.csv, 00_template_fsaverage_method_summary.csv
and provenance JSON inside the existing comparison tables/ folder; adds hashes
to its manifest. No original analysis table, figure or reconstruction changes.
Run from repository root with .venv/bin/python and this script path.
Requires NumPy/pandas. Existing export files are refused.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

VARIANT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    source=VARIANT/'outputs/04_evaluate_diagnostics'
    destination=VARIANT/'outputs/06_compare_no_warp'
    audit=json.loads((source/'12_evaluation_manifest.json').read_text())
    forward_path=VARIANT/'outputs/02_template_forward_assumed_v1/01_forward_manifest.json'
    forward=json.loads(forward_path.read_text())
    require(forward['policy']['individual_geometry_used'] is False,'Unexpected individual geometry in forward policy')
    require(audit['registration_applied'] is False and audit['anatomical_localization_error_valid'] is False,
            'Expected original no-registration diagnostics')
    inputs=[source/'12_evaluation_manifest.json',forward_path,Path(__file__)]
    for filename in ['01_run_level_diagnostics.csv','03_method_summary.csv']:
        path=source/filename
        require(sha(path)==audit['output_sha256'][filename],f'Original diagnostic output changed: {path}')
        inputs.append(path)
    require(sha(forward_path)==audit['input_sha256'][str(forward_path)],'Forward policy differs from audited run')
    data=pd.read_csv(source/'01_run_level_diagnostics.csv',keep_default_na=False)
    require(len(data)==5124 and set(data.status)<={'PASS','FAIL'},'Expected completed 5,124-job ledger')
    passed=data.status.eq('PASS')
    require(data.loc[passed,'coordinate_frame'].eq('template_head').all(),'Unexpected estimate frame')
    estimate=data.loc[passed,[f'estimated_head_{a}_mm' for a in 'xyz']].to_numpy(float)
    truth=data.loc[passed,[f'gt_head_{a}_mm' for a in 'xyz']].to_numpy(float)
    computed=np.linalg.norm(estimate-truth,axis=1)
    require(np.isfinite(computed).all() and np.allclose(computed,data.loc[passed,'diagnostic_distance_mm'].to_numpy(float),
            rtol=0,atol=1e-8),'Recomputed distances disagree with Script 04')
    columns=['strategy','method','test_subject','run','reference_subject','status','failure_reason','mapping_status']
    result=data[columns].copy()
    result['diagnostic_distance_mm']=np.nan
    result.loc[passed,'diagnostic_distance_mm']=computed
    result['reconstruction_anatomy']='fsaverage'
    result['reconstruction_sensor_geometry']='GSN-HydroCel-256_template'
    result['test_subject_electrode_positions_used']=False
    result['MRI_alignment_used_for_distance']=False
    result['valid_anatomical_localization_error']=False
    summary=pd.read_csv(source/'03_method_summary.csv')
    require(len(summary)==24 and set(summary.method)==set(data.method),'Expected 12 methods for each transfer strategy')
    require(int(summary.n_distances.sum())==int(passed.sum()),'Summary count differs')
    stats_path=destination/'tables/00_template_fsaverage_method_summary.csv'
    values_path=destination/'tables/00_template_fsaverage_distances.csv'
    note_path=destination/'tables/00_template_fsaverage_distance_export.json'
    require(not any(p.exists() for p in [stats_path,values_path,note_path]),'Distance export exists; originals are preserved')
    hashes={str(p):sha(p) for p in inputs}
    result.to_csv(values_path,index=False);summary.to_csv(stats_path,index=False)
    note=dict(experiment='fsaverage with fixed HydroCel template, no individual electrode placement',
        total_jobs=len(result),n_distances=int(passed.sum()),n_failures=int((~passed).sum()),
        definition='Euclidean norm of saved template HEAD estimate minus audited individual HEAD ground truth, mm; unregistered diagnostic',
        individual_electrode_positions_used=False,MRI_alignment_used=False,
        montage_mapping='ASSUMED_UNVERIFIED',valid_anatomical_localization_error=False,
        input_sha256=hashes,output_sha256={str(p.relative_to(destination)):sha(p) for p in [stats_path,values_path]})
    note_path.write_text(json.dumps(note,indent=2,allow_nan=False)+'\n')
    manifest_path=destination/'comparison_manifest.json';manifest=json.loads(manifest_path.read_text())
    for p in [stats_path,values_path,note_path]:manifest['output_sha256'][str(p.relative_to(destination))]=sha(p)
    manifest['template_distance_export']=dict(values=str(values_path.relative_to(destination)),
        method_summary=str(stats_path.relative_to(destination)),provenance=str(note_path.relative_to(destination)))
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Verified and exported {int(passed.sum())} distances; {int((~passed).sum())} failures retain missing distances.')
    print(values_path);print(stats_path)


if __name__=='__main__':
    main()
