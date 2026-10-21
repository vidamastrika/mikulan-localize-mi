#!/usr/bin/env python3
"""Verify released Localize-MI surface-to-T1w electrode coordinates.

Read-only and no fitted transform is used as an output. Compare the published
head↔surface transform to the named electrode coordinates, preserving metres.
This does NOT validate MRI-to-MRI registration or anatomical point accuracy.

Usage:
 python cross_subject/scripts/13_verify_surface_to_t1w_transform.py \
   --target sub-02 --donor sub-01
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def xyz_tables(root, subject):
    folder = root/'derivatives'/'epochs'/subject/'ieeg'
    native = folder/f'{subject}_task-seegstim_space-surface_electrodes.tsv'
    dest = folder/f'{subject}_task-seegstim_space-T1w_electrodes.tsv'
    a = pd.read_csv(native, sep='\t')
    b = pd.read_csv(dest, sep='\t')
    if not {'name','x','y','z'}.issubset(a) or not {'name','x','y','z'}.issubset(b):
        raise ValueError('Electrode tables missing expected columns')
    joined = a[['name','x','y','z']].merge(b[['name','x','y','z']],on='name',suffixes=('_surface','_t1w'),validate='one_to_one')
    if len(joined)<4: raise ValueError(f'{subject}: insufficient shared electrodes')
    x = joined[['x_surface','y_surface','z_surface']].to_numpy(float)
    y = joined[['x_t1w','y_t1w','z_t1w']].to_numpy(float)
    if np.max(np.abs(x))>=1 or np.max(np.abs(y))>=1:
        raise ValueError('Expected electrode coordinates to be in metres')
    return joined['name'].tolist(),x,y,native,dest


def apply(x,mat):
    return (np.c_[x,np.ones(len(x))]@mat.T)[:,:3]


def summary(errors):
    return {'median_mm':float(np.median(errors)), 'p95_mm':float(np.percentile(errors,95)),
            'max_mm':float(np.max(errors)), 'mean_mm':float(np.mean(errors))}


def audit(root, sub):
    names,x,y,native,dest=xyz_tables(root,sub)
    xf=root/'derivatives'/'sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
    with h5py.File(xf,'r') as h:
        h2s=np.asarray(h['trans'],dtype=float)
    if h2s.shape != (4,4): raise ValueError('head-to-surface transform not 4x4')
    s2h=np.linalg.inv(h2s)
    candidates={'identity':np.eye(4),'inverse_head_to_surface':s2h,'head_to_surface':h2s}
    outcomes={}
    for label,mat in candidates.items():
        pred=apply(x,mat)
        errs=np.linalg.norm(pred-y,axis=1)*1000
        outcomes[label]=summary(errs)
    vectors=(y-x)*1000
    report={'subject':sub,'contacts':len(names),'surface_electrodes':str(native),
            't1w_electrodes':str(dest),'head_to_surface_file':str(xf),
            'head_to_surface_matrix':h2s.tolist(),'inverse_head_to_surface_matrix':s2h.tolist(),
            'measured_offset_median_mm':np.median(vectors,axis=0).tolist(),
            'measured_offset_spread_max_mm':float(np.max(np.linalg.norm(vectors-np.median(vectors,axis=0),axis=1))),
            'candidates':outcomes,
            'interpretation':'Match to T1w electrode TSV only; T1w TSV metadata must separately be checked against NIfTI physical frame.'}
    rows=[{'subject':sub,'candidate':label,'contacts':len(names),**stats} for label,stats in outcomes.items()]
    return report,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',default='sub-02')
    p.add_argument('--donor',default='sub-01')
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--output-dir',type=Path,default=Path('cross_subject/outputs/13_surface_to_t1w_verification'))
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    results={};records=[]
    for sub in dict.fromkeys([a.target,a.donor]):
        record,rows=audit(a.dataset,sub)
        results[sub]=record;records+=rows
        print(f'{sub}: {record["contacts"]} shared electrode contacts')
        for label,stats in record['candidates'].items():
            print(f'  {label}: median={stats["median_mm"]:.4f} mm, max={stats["max_mm"]:.4f} mm')
        print('  observed T1w-minus-surface shift (mm):',np.round(record['measured_offset_median_mm'],4))
    output=a.output_dir/f'{a.donor}_to_{a.target}_surface_to_t1w_verification'
    output.with_suffix('.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY','subjects':results},indent=2))
    pd.DataFrame(records).to_csv(output.with_suffix('.csv'),index=False)
    print('Saved:',output.with_suffix('.json'))


if __name__=='__main__': main()
