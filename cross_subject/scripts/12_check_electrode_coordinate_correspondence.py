#!/usr/bin/env python3
"""Inspect supplied Localize-MI native-surface vs T1w electrode coordinates.

This is a read-only diagnostic, not a registration or validated transform.
Run from the repository root:
  python cross_subject/scripts/12_check_electrode_coordinate_correspondence.py \
    --target sub-02 --donor sub-01

Produces a JSON report and CSV comparison under cross_subject/outputs/12_electrode_coordinate_audit/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='sub-02')
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--dataset', type=Path, default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/12_electrode_coordinate_audit'))
    return p.parse_args()


def mm_from_tsv(path):
    df = pd.read_csv(path, sep='\t')
    required = {'name', 'x', 'y', 'z'}
    if not required.issubset(df):
        raise ValueError(f'{path}: missing columns {sorted(required - set(df.columns))}')
    if df['name'].duplicated().any():
        raise ValueError(f'{path}: duplicate electrode names')
    return df[['name','x','y','z']].copy()


def identify_units(x, label):
    """Require explicit plausible mm or m magnitudes; never silently fit scale."""
    vmax = float(np.nanmax(np.abs(x)))
    if vmax < 1.0:
        return x * 1000.0, 'metres (converted to mm)'
    if 1.0 < vmax < 300.0:
        return x, 'millimetres'
    raise ValueError(f'{label}: ambiguous coordinate magnitudes (max |xyz|={vmax:.4g}); verify units manually')


def apply_mm(points, transform):
    return np.column_stack([points, np.ones(len(points))]) @ transform.T[:, :3]


def analyze_subject(root, subject):
    ieeg = root/'derivatives'/'epochs'/subject/'ieeg'
    native_file = ieeg/f'{subject}_task-seegstim_space-surface_electrodes.tsv'
    if not native_file.exists():
        raise FileNotFoundError(native_file)
    other_files = sorted(ieeg.glob(f'{subject}_task-seegstim_space-*electrodes.tsv'))
    other_files = [f for f in other_files if f != native_file and 'MNI' not in f.name]
    native = mm_from_tsv(native_file)
    left = root/'derivatives'/'sourcemodelling'/subject/'anat'/f'{subject}_hemi-L_pial.surf.gii'
    image = nib.load(str(left))
    coord_system = image.darrays[0].coordsys
    if coord_system is None:
        raise ValueError(f'{left}: no GIFTI coordinate transform recorded')
    metadata_transform = np.asarray(coord_system.xform, float)
    if metadata_transform.shape != (4, 4):
        raise ValueError('Unexpected GIFTI transform shape')
    candidates = {
        'identity': np.eye(4),
        'gifti_forward': metadata_transform,
        'gifti_inverse': np.linalg.inv(metadata_transform),
    }
    results=[]
    info={'subject':subject,'source_file':str(native_file),'other_electrode_files':[str(f) for f in other_files],
          'gifti_source_file':str(left),
          'gifti_dataspace':int(coord_system.dataspace),'gifti_xformspace':int(coord_system.xformspace),
          'gifti_transform_mm':metadata_transform.tolist(),'comparisons':[]}
    if not other_files:
        info['warning']='No matching subject-native T1w/mod-T1w electrode files; cannot independently validate mapping.'
    for dest in other_files:
        comparison={'target_file':str(dest)}
        other = mm_from_tsv(dest)
        matched = native.merge(other, on='name', suffixes=('_surface','_target'), validate='one_to_one')
        comparison['matched_contacts']=len(matched)
        if len(matched)<4:
            comparison['warning']='Too few named paired contacts to compare'
            info['comparisons'].append(comparison)
            continue
        x,xunits=identify_units(matched[['x_surface','y_surface','z_surface']].to_numpy(float),str(native_file))
        y,yunits=identify_units(matched[['x_target','y_target','z_target']].to_numpy(float),str(dest))
        comparison.update(source_unit=xunits, target_unit=yunits)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            comparison['warning']='Nonfinite paired-contact positions; excluded'
            info['comparisons'].append(comparison)
            continue
        # The files may define LPS or RAS. Sign flip is recorded as an explicit
        # diagnostic candidate, not assumed correct or automatically applied.
        ras_to_lps=np.diag([-1.,-1.,1.])
        for cname,mat in candidates.items():
            mapped=apply_mm(x,mat)
            for convention,projected in [('same_axes',mapped),('RAS_LPS_flip',mapped@ras_to_lps)]:
                norms=np.linalg.norm(projected-y,axis=1)
                label=cname+'__'+convention
                comparison[label]={
                    'median_mm':float(np.median(norms)),
                    'mean_mm':float(np.mean(norms)),
                    'p95_mm':float(np.percentile(norms,95)),
                    'max_mm':float(np.max(norms)),
                }
                results.append({'subject':subject,'destination':dest.name,'candidate':label,
                                'n_contacts':len(norms),'median_error_mm':float(np.median(norms)),
                                'p95_error_mm':float(np.percentile(norms,95)),
                                'max_error_mm':float(np.max(norms))})
        info['comparisons'].append(comparison)
    return info,results


def main():
    a=args_parser()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    info={'purpose':'diagnostic only, NOT a validated MRI-registration chain',
          'dataset':str(a.dataset),'subjects':{}}
    records=[]
    for sub in (a.target,a.donor):
        data,summary=analyze_subject(a.dataset,sub)
        info['subjects'][sub]=data
        records.extend(summary)
    report=a.output_dir/f'{a.donor}_to_{a.target}_electrode_correspondence.json'
    report.write_text(json.dumps(info,indent=2),encoding='utf8')
    csv=a.output_dir/f'{a.donor}_to_{a.target}_electrode_correspondence.csv'
    pd.DataFrame(records).to_csv(csv,index=False)
    print(f'Report: {report}\nCSV: {csv}')
    if records:
        result=pd.DataFrame(records)
        for (sub,dest),table in result.groupby(['subject','destination']):
            print(f'\n{sub} → {dest} | {int(table.iloc[0].n_contacts)} matched contacts')
            print(table.sort_values('median_error_mm')[['candidate','median_error_mm','p95_error_mm']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nDIAGNOSTIC ONLY. Compare file descriptions, coordinate conventions, and anatomical validity before accepting any transform.')


if __name__=='__main__':
    main()
