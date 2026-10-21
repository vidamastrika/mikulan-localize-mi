#!/usr/bin/env python3
"""Localize-MI cross-subject Script 02: read-only geometry/metadata validation.

Requires mne, numpy, pandas, h5py. Does NOT construct cross-subject transforms.
Run from repository root; use --dataset-root to override dataset location.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import h5py
import mne
from mne.transforms import _frame_to_str

DEFAULT_ROOT = Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI'

def safe_json(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return str(x)

def frame(value):
    try: return _frame_to_str.get(int(value), str(value))
    except Exception: return str(value)

def geometry_summary(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0: return {'count': int(len(p)), 'valid': False}
    finite = np.isfinite(p).all(axis=1)
    q = p[finite]
    if len(q) == 0: return {'count': len(p), 'finite_count': 0, 'valid': False}
    return {'count':len(p),'finite_count':len(q),'valid':bool(len(q)==len(p)),
            'min_m':q.min(axis=0).tolist(),'max_m':q.max(axis=0).tolist(),
            'center_m':q.mean(axis=0).tolist(),
            'extent_mm':((q.max(axis=0)-q.min(axis=0))*1000).tolist()}

def inspect_h5(path):
    ans={'path':str(path),'exists':path.is_file()}
    if not path.is_file():return ans
    entries=[]
    try:
        with h5py.File(path,'r') as h:
            def visit(name, obj):
                entry={'name':name,'kind':'dataset' if isinstance(obj,h5py.Dataset) else 'group'}
                if isinstance(obj,h5py.Dataset):
                    entry['shape']=list(obj.shape); entry['dtype']=str(obj.dtype)
                    if obj.size<=16:
                        try:entry['value_preview']=safe_json(obj[()])
                        except Exception:pass
                if obj.attrs:
                    entry['attrs']={str(k):safe_json(v) for k,v in obj.attrs.items()}
                entries.append(entry)
            h.visititems(visit)
        ans['entries']=entries
    except Exception as exc:ans['error']=f'{type(exc).__name__}: {exc}'
    return ans

def inspect_forward(path):
    ans={'path':str(path),'exists':path.is_file()}
    if not path.is_file():return ans
    try:
        fwd=mne.read_forward_solution(path, verbose='ERROR')
        ans.update(nsource=int(fwd['nsource']),nchan=int(fwd['nchan']),
                   source_ori=int(fwd['source_ori']),coord_frame=frame(fwd['coord_frame']),
                   source_spaces=[])
        trans=fwd.get('mri_head_t')
        if trans is not None:
            ans['mri_head_t']={'from':frame(trans['from']),'to':frame(trans['to']),
                               'matrix':np.asarray(trans['trans']).tolist(),
                               'det_rotation':float(np.linalg.det(trans['trans'][:3,:3]))}
        for src in fwd['src']:
            entry={'type':str(src.get('type')),'coord_frame':frame(src.get('coord_frame')),
                   'nuse':int(src.get('nuse',0)), 'np':int(src.get('np',0)),
                   'id':int(src.get('id',-1))}
            rr=src.get('rr')
            if rr is not None:entry['geometry']=geometry_summary(rr)
            ans['source_spaces'].append(entry)
        info=fwd['info']
        ans['channel_names']=list(info['ch_names'])
        electrodes=[ch['loc'][:3] for ch in info['chs'] if ch['kind']==mne.io.constants.FIFF.FIFFV_EEG_CH]
        ans['eeg_sensor_geometry_head_m']=geometry_summary(electrodes) if electrodes else None
        ans['eeg_sensor_coordinate_frames']=sorted({frame(ch['coord_frame']) for ch in info['chs'] if ch['kind']==mne.io.constants.FIFF.FIFFV_EEG_CH})
    except Exception as exc:
        ans['error']=f'{type(exc).__name__}: {exc}'
        ans['traceback']=traceback.format_exc(limit=2)
    return ans

def inspect_coords(subdir, folder, kind):
    files=sorted((subdir/folder).glob('*coordsystem.json'))
    out=[]
    for path in files:
        try: out.append({'path':str(path),'metadata':json.loads(path.read_text())})
        except Exception as exc:out.append({'path':str(path),'error':str(exc)})
    return out

def inspect_electrodes(subdir, folder):
    out=[]
    for path in sorted((subdir/folder).glob('*electrodes.tsv')):
        entry={'path':str(path)}
        try:
            df=pd.read_csv(path,sep='\t')
            entry['rows']=len(df);entry['columns']=list(df.columns)
            if {'x','y','z'}.issubset(df.columns):
                pts=df[['x','y','z']].apply(pd.to_numeric,errors='coerce').to_numpy()
                good=pts[np.isfinite(pts).all(axis=1)]
                entry['finite_coordinates']=len(good)
                if len(good):
                    entry['coordinate_min_native']=good.min(axis=0).tolist()
                    entry['coordinate_max_native']=good.max(axis=0).tolist()
                    entry['coordinate_median_radius_native']=float(np.median(np.linalg.norm(good,axis=1)))
            entry['first_names']=df['name'].head(5).astype(str).tolist() if 'name' in df.columns else []
        except Exception as exc:entry['error']=str(exc)
        out.append(entry)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/02_validation'))
    ap.add_argument('--subjects',nargs='+',default=[f'sub-{i:02d}' for i in range(1,8)])
    args=ap.parse_args()
    root=args.dataset_root.expanduser().resolve()
    if not root.is_dir():ap.error(f'Dataset directory not found: {root}')
    output=args.output_root.expanduser();output.mkdir(parents=True,exist_ok=True)
    details={'dataset_root':str(root),'versions':{'mne':mne.__version__,'numpy':np.__version__},'subjects':{}}
    summary=[]
    for subject in args.subjects:
        ep=root/'derivatives'/'epochs'/subject
        sm=root/'derivatives'/'sourcemodelling'/subject
        result={'forward':inspect_forward(sm/'fwd'/f'{subject}_fwd.fif'),
                'h5_transform':inspect_h5(sm/'xfm'/f'{subject}_from-head_to-surface.h5'),
                'eeg_coordinate_systems':inspect_coords(ep,'eeg','EEG'),
                'ieeg_coordinate_systems':inspect_coords(ep,'ieeg','iEEG'),
                'eeg_electrodes':inspect_electrodes(ep,'eeg'),
                'ieeg_electrodes':inspect_electrodes(ep,'ieeg'),
                'surfaces':{name:(sm/'anat'/f'{subject}_{name}.surf.gii').is_file()
                            for name in ['inner_skull','outer_skull','outer_skin','hemi-L_pial','hemi-R_pial']},
                'n_runs':len(list((ep/'eeg').glob('*_epochs.npy')))}
        details['subjects'][subject]=result
        f=result['forward']
        summary.append({'subject':subject,'runs':result['n_runs'],
                        'forward_readable':bool(f.get('nsource')),
                        'forward_n_sources':f.get('nsource'),
                        'forward_n_channels':f.get('nchan'),
                        'forward_coord_frame':f.get('coord_frame'),
                        'forward_has_mri_head_transform':'mri_head_t' in f,
                        'source_spaces':len(f.get('source_spaces',[])),
                        'h5_readable':'entries' in result['h5_transform'],
                        'eeg_coord_metadata_files':len(result['eeg_coordinate_systems']),
                        'ieeg_coord_metadata_files':len(result['ieeg_coordinate_systems']),
                        'all_five_surfaces':all(result['surfaces'].values()),
                        'error':f.get('error','')})
        print(f"{subject}: runs={result['n_runs']} forward={'OK' if f.get('nsource') else 'FAIL'} "
              f"sources={f.get('nsource','?')} H5={'OK' if 'entries' in result['h5_transform'] else 'FAIL'}")
    (output/'geometry_details.json').write_text(json.dumps(details,indent=2,default=safe_json))
    pd.DataFrame(summary).to_csv(output/'subject_validation_summary.csv',index=False)
    report=['SCRIPT 02 — LOCALIZE-MI CROSS-SUBJECT GEOMETRY VALIDATION',
            f'Dataset: {root}','',
            'NOTE: File readability is not proof that different subjects can be safely registered.',
            'No cross-subject forward model was constructed. No location error was calculated.','',
            'SUBJECT SUMMARIES']
    for row in summary:report.append(str(row))
    report += ['', 'REQUIRED BEFORE SCRIPT 03:',
               '1. Confirm EEG coordinate metadata units and frame (in geometry_details.json).',
               '2. Interpret head-to-surface H5 transforms and MRI-to-head transform directions.',
               '3. Choose and validate a subject-to-subject anatomical registration method.',
               '4. Ensure predicted and true stimulation positions share a physical coordinate frame.',
               '5. Validate within-subject reproduction before attempting any mismatched model.']
    (output/'validation_report.txt').write_text('\n'.join(report)+'\n')
    print(f'\nSaved outputs to: {output.resolve()}')

if __name__=='__main__':main()
