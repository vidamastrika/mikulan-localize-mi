#!/usr/bin/env python3
"""READ ONLY: compare Script 15 and Script 35 within-subject Continuous-ECD.

Uses already-completed output files. Writes only under a separate audit folder.
No reconstruction, no modification of Script 35's main results.
"""
from pathlib import Path
import argparse
import hashlib
import pandas as pd
import numpy as np


def checksum(p):
    p=Path(str(p))
    if not p.is_file(): return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--grid',type=Path,default=Path('outputs/15_full_additional_method_grid/continuous_ecd/grid_results.csv'))
p.add_argument('--main-root',type=Path,default=Path('cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne'))
p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/35_ecd_reproduction_test/audit'))
a=p.parse_args()
old=pd.read_csv(a.grid,low_memory=False)
new=pd.read_csv(a.main_root/'01_execution_status.csv',low_memory=False)
ev=pd.read_csv(a.main_root/'02_localization_errors.csv',low_memory=False)
old=old[(old.method=='Continuous-ECD')&(old.montage.astype(str)=='128')&np.isclose(pd.to_numeric(old.ecd_min_dist_mm,errors='coerce'),15)].copy()
new=new[(new.method=='Continuous-ECD')&(new.is_within.astype(str).str.lower()=='true')].copy()
within=ev[(ev.method=='Continuous-ECD')].drop_duplicates(['test_subject','run'])[['test_subject','run','within_subject_error_mm']]
new=new.merge(within,on=['test_subject','run'],validate='one_to_one')
joined=old.merge(new,left_on=['subject','run'],right_on=['test_subject','run'],validate='one_to_one',suffixes=('_15','_35'))
assert len(joined)==61,f'Expected 61 matched rows, got {len(joined)}'
joined['error_delta_mm']=pd.to_numeric(joined.within_subject_error_mm)-pd.to_numeric(joined.localization_distance_mm)
joined['abs_delta_mm']=joined.error_delta_mm.abs()
joined['time_delta_ms']=pd.to_numeric(joined.peak_time_s)*1000-pd.to_numeric(joined.selected_time_ms)
joined['peak_time_matches']=joined.time_delta_ms.abs()<=0.001
joined['bem_script15_exists']=joined.bem_file.map(lambda x:Path(str(x)).is_file())
joined['bem_script35_exists']=joined.bem_solution_path.map(lambda x:Path(str(x)).is_file())
joined['bem_script15_sha256']=joined.bem_file.map(checksum)
joined['bem_script35_sha256']=joined.bem_solution_path.map(checksum)
joined['bem_same_content']=np.where(joined.bem_script15_sha256.notna()&joined.bem_script35_sha256.notna(),joined.bem_script15_sha256==joined.bem_script35_sha256,None)
joined['error_matches']=joined.abs_delta_mm<=.05
cols=['subject','run','status_15','status_35','localization_distance_mm','within_subject_error_mm','error_delta_mm','abs_delta_mm','selected_time_ms','peak_time_s','time_delta_ms','peak_time_matches','bem_file','bem_solution_path','bem_script15_exists','bem_script35_exists','bem_same_content','bem_script15_sha256','bem_script35_sha256']
a.output_root.mkdir(parents=True,exist_ok=True)
joined[cols].sort_values(['subject','run']).to_csv(a.output_root/'01_ecd_grid_vs_cross_audit.csv',index=False)
miss=joined[~joined.error_matches].sort_values('abs_delta_mm',ascending=False)
miss[cols].to_csv(a.output_root/'02_ecd_mismatched_runs.csv',index=False)
print('Matched:',len(joined),' | Error matches <=0.05 mm:',int(joined.error_matches.sum()),' | Differences:',len(miss))
print('Peak-time matches:',int(joined.peak_time_matches.sum()),'/61')
print('BEM files found (Script 15 / Script 35):',int(joined.bem_script15_exists.sum()),'/',int(joined.bem_script35_exists.sum()))
valid=joined.bem_same_content.notna()
print('BEM content identical where both exist:',int((joined.bem_same_content==True).sum()),'/',int(valid.sum()))
print('\n16 largest discrepancies (mm):')
print(miss[['subject','run','localization_distance_mm','within_subject_error_mm','error_delta_mm','selected_time_ms','peak_time_s','bem_same_content']].head(16).to_string(index=False))
print('\nSaved read-only audit:',a.output_root)
print('Important: do not compare HEAD and MRI coordinates directly.')
