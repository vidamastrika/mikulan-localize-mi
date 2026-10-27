#!/usr/bin/env python3
"""Read-only LCMV reproduction distance audit: Script 15 vs Script 35.

Reads ONLY the within-subject LCMV results in a separate reproduction folder.
Uses each test subject's released head-to-surface H5 matrix, then compares
source positions and ground-truth Euclidean localization errors in surface mm.
Never writes into or changes the main Script 35 output directory.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI')
    p.add_argument('--grid', type=Path, default=Path('outputs/15_full_additional_method_grid/lcmv/grid_results.csv'))
    p.add_argument('--repro-root', type=Path, default=Path('cross_subject/outputs/35_lcmv_reproduction_test'))
    p.add_argument('--tolerance-mm', type=float, default=0.05)
    a = p.parse_args()
    execution = a.repro_root / '01_execution_status.csv'
    for path in [a.grid, execution]:
        if not path.is_file(): p.error(f'Missing file: {path}')
    old = pd.read_csv(a.grid, low_memory=False)
    new = pd.read_csv(execution, low_memory=False)
    old = old.loc[
        old['method'].eq('LCMV') & old['montage'].astype(str).eq('32') &
        np.isclose(pd.to_numeric(old['lcmv_reg'], errors='coerce'), 0) &
        np.isclose(pd.to_numeric(old['lcmv_window_half_ms'], errors='coerce'), 10) &
        old['lcmv_data_covariance_method'].eq('diagonal_fixed') &
        old['lcmv_weight_norm'].eq('unit-noise-gain-invariant') &
        old['lcmv_pick_ori'].eq('max-power') &
        old['lcmv_reduce_rank'].astype(str).str.lower().eq('true')
    ].copy()
    new = new.loc[
        new['method'].eq('LCMV') & new['is_within'].astype(str).str.lower().eq('true')
    ].copy()
    for desc, df, cols in [
        ('Script 15', old, ['subject', 'run']),
        ('Script 35 reproduction', new, ['test_subject', 'run'])
    ]:
        if len(df) != 61 or df.duplicated(cols).any():
            raise ValueError(f'{desc}: expected exactly 61 unique within-subject runs; got {len(df)}')
        if not df['status'].eq('PASS').all():
            raise ValueError(f'{desc}: one or more runs did not PASS')
    if not new['test_subject'].eq(new['reference_subject']).all():
        raise ValueError('A supposedly within-subject record uses different reference and test subjects')
    for field, value in [('montage', '32'), ('lcmv_reg', 0), ('lcmv_window_half_ms', 10),
                         ('lcmv_data_covariance_method','diagonal_fixed'),
                         ('lcmv_weight_norm','unit-noise-gain-invariant')]:
        values = new[field]
        if isinstance(value, str):
            ok = values.astype(str).eq(value).all()
        else:
            ok = np.isclose(pd.to_numeric(values,errors='coerce'),value).all()
        if not ok: raise ValueError(f'Reproduction parameters do not match: {field}')
    joined = old.merge(new, left_on=['subject','run'], right_on=['test_subject','run'],
                       suffixes=('_15','_35'), validate='one_to_one')
    if len(joined) != 61: raise ValueError(f'Only {len(joined)} runs matched')
    results = []
    transform_cache = {}
    for r in joined.to_dict('records'):
        subj, run = r['subject'], r['run']
        if subj not in transform_cache:
            path = a.dataset/'derivatives'/'sourcemodelling'/subj/'xfm'/f'{subj}_from-head_to-surface.h5'
            if not path.is_file(): raise FileNotFoundError(path)
            with h5py.File(path,'r') as f: matrix = np.asarray(f['trans'],float)
            if matrix.shape!=(4,4) or not np.allclose(matrix[3], [0,0,0,1]):
                raise ValueError(f'Invalid H5 affine: {path}')
            transform_cache[subj] = matrix
        # Both HEAD and surface coordinates are expressed here in millimetres.
        head = np.array([float(r[f'reference_head_{axis}_mm']) for axis in 'xyz'])
        surface = (transform_cache[subj] @ np.r_[head/1000.,1.])[:3]*1000.
        old_surface = np.array([float(r[f'selected_{axis}_mm']) for axis in 'xyz'])
        truth = np.array([float(r[f'ground_truth_{axis}_mm']) for axis in 'xyz'])
        old_dist = float(r['localization_distance_mm'])
        new_dist = float(np.linalg.norm(surface-truth))
        old_recalc = float(np.linalg.norm(old_surface-truth))
        time_old = float(r['selected_time_ms'])
        time_new = float(r['peak_time_s'])*1000.
        results.append(dict(subject=subj, run=run,
            peak_vertex_script15=int(r['peak_vertex_15']),peak_vertex_script35=int(r['peak_vertex_35']),
            peak_time_script15_ms=time_old,peak_time_script35_ms=time_new,
            original_error_mm=old_dist,reproduced_error_mm=new_dist,
            error_difference_mm=new_dist-old_dist,
            coordinate_difference_mm=float(np.linalg.norm(surface-old_surface)),
            original_error_recalc_difference_mm=old_recalc-old_dist,
            vertex_match=bool(int(r['peak_vertex_15'])==int(r['peak_vertex_35'])),
            time_match=bool(np.isclose(time_old,time_new,atol=.001,rtol=0)),
        ))
    df = pd.DataFrame(results).sort_values(['subject','run'])
    output = a.repro_root/'audit'
    output.mkdir(parents=True,exist_ok=True)
    outpath=output/'03_lcmv_within_distance_validation.csv'
    df.to_csv(outpath,index=False)
    print('Matched runs:', len(df))
    print('Peak vertices equal:',df.vertex_match.sum(),'/61')
    print('Peak times equal:',df.time_match.sum(),'/61')
    print('Original median localization error:',f'{df.original_error_mm.median():.6f}','mm')
    print('Reproduction median localization error:',f'{df.reproduced_error_mm.median():.6f}','mm')
    print('Maximum per-run distance difference:',f'{df.error_difference_mm.abs().max():.6f}','mm')
    print('Maximum source-coordinate difference:',f'{df.coordinate_difference_mm.max():.6f}','mm')
    print('Maximum original-distance recalculation difference:',f'{df.original_error_recalc_difference_mm.abs().max():.6f}','mm')
    good = (df.vertex_match.all() and df.time_match.all() and
            df.error_difference_mm.abs().le(a.tolerance_mm).all() and
            df.coordinate_difference_mm.le(a.tolerance_mm).all() and
            df.original_error_recalc_difference_mm.abs().le(a.tolerance_mm).all())
    print('RESULT:', 'PASS — 61/61 distances reproduced' if good else 'MISMATCH — inspect CSV')
    print('Details:',outpath)
    if not good: raise SystemExit(1)

if __name__=='__main__': main()
