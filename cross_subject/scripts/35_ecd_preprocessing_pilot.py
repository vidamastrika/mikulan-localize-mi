#!/usr/bin/env python3
"""Read-only two-case Continuous-ECD reproduction pilot.

Outputs ONLY to cross_subject/outputs/35_ecd_reproduction_test/pilot/.
Does not change main Script 35 results. Run at repository root.
"""
import importlib.util
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT=Path.cwd().resolve()
SCRIPT35=PROJECT/'cross_subject/scripts/35_ecd_preprocessing_test.py'
SCRIPT34=PROJECT/'cross_subject/scripts/34_cross_subject_minimum_norm.py'
SCRIPT09=PROJECT/'scripts/09_full_parameter_grid.py'
OUT=PROJECT/'cross_subject/outputs/35_ecd_reproduction_test/pilot_preprocessing'
CASES=(('sub-02','run-03'),('sub-02','run-05'))
GRID=PROJECT/'outputs/15_full_additional_method_grid/continuous_ecd/grid_results.csv'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError(f'Cannot import {path}')
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

def main():
    for path in [SCRIPT35,SCRIPT34,SCRIPT09,GRID]:
        if not path.is_file():raise FileNotFoundError(path)
    s35=load(SCRIPT35,'ecd_audit_s35')
    s34=load(SCRIPT34,'ecd_audit_s34')
    s09=load(SCRIPT09,'ecd_audit_s09')
    a=s35.parser()
    a.quiet=True
    a.output_root=OUT
    cache={}
    forward=s34.get_forward(a.dataset,'sub-01',cache)
    names,xyz,nominal=s09.fixed_spatial_subsets(forward)
    template_info=(names,xyz)
    old=pd.read_csv(GRID,low_memory=False)
    old=old[(old['method']=='Continuous-ECD') & (old['montage'].astype(str)=='128') &
            np.isclose(pd.to_numeric(old['ecd_min_dist_mm'],errors='coerce'),15.0) &
            (old['status']=='PASS')]
    if old[['subject','run']].duplicated().any():raise ValueError('Duplicate baseline rows')
    records=[]
    for subject,run in CASES:
        original=old[(old['subject']==subject)&(old['run']==run)]
        if len(original)!=1:raise ValueError(f'Expected one original row for {subject} {run}, got {len(original)}')
        print(f'Computing {subject} {run} ...',flush=True)
        new=s35.fit(a,('Continuous-ECD',subject,subject,run),
                    dict(s35.FROZEN['Continuous-ECD']),s09,s34,cache,nominal,template_info)
        oldrow=original.iloc[0]
        if new['status']!='PASS':raise RuntimeError(f'{subject} {run} failed: {new["failure_reason"]}')
        row=dict(test_subject=subject,run=run,status=new['status'],
                 old_peak_time_ms=float(oldrow['selected_time_ms']),
                 new_peak_time_ms=float(new['peak_time_s'])*1000,
                 old_distance_mm=float(oldrow['localization_distance_mm']),
                 new_head_x_mm=float(new['reference_head_x_mm']),
                 new_head_y_mm=float(new['reference_head_y_mm']),
                 new_head_z_mm=float(new['reference_head_z_mm']),
                 runtime_seconds=float(new['runtime_seconds']),
                 bem_solution_path=new.get('bem_solution_path',''),
                 solver_notes=new.get('solver_notes',''))
        row['peak_time_match']=bool(np.isclose(row['old_peak_time_ms'],row['new_peak_time_ms'],atol=.001))
        records.append((row,new))
        print('  peak time old/new:',row['old_peak_time_ms'],row['new_peak_time_ms'],
              ' | match:',row['peak_time_match'],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([r for r,_ in records]).to_csv(OUT/'01_two_case_comparison.csv',index=False)
    pd.DataFrame([r for _,r in records]).to_csv(OUT/'02_two_case_reconstructions.csv',index=False)
    print('Saved to:',OUT)
    print('NOTE: head coordinates cannot be directly compared against Script 15 MRI/surface coordinates.')
    print('Reconstruction completed. Localization-distance validation requires HEAD-to-surface transform.')

if __name__=='__main__':main()
