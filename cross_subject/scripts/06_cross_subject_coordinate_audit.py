#!/usr/bin/env python3
"""Script 06: audit coordinate mapping for the SESAME cross-subject pilot.

This is a READ-ONLY feasibility audit, NOT a registration algorithm. It reads
only the donor anatomy/transform metadata and the Script 05 result. The target
MRI/forward/digitized EEG positions are never used. Target SEEG ground truth
is not accessed at this stage. No localization error is calculated unless a
later, separately validated mapping is provided in a separate evaluation step.

The donor's head-to-surface transform is NOT a surface-to-MNI transform.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_DATASET = Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI'


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--target', default='sub-02')
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--run', default='01', help='01 or run-01')
    p.add_argument('--pilot-json', type=Path, default=None)
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/06_coordinate_audit'))
    return p.parse_args()


def safe_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        return {'_read_error': f'{type(exc).__name__}: {exc}'}


def collect_transform_inventory(folder):
    """Only donor metadata files; don't scan any target directories."""
    allowed = {'.json','.h5','.txt','.tsv','.mat','.npy','.npz','.lta','.xfm','.tfm','.hdf5','.hdf','.nii','.gz'}
    patterns = re.compile(r'mni|template|warp|affine|transform|trans|xfm|surface|t1w|native|fsaverage|space', re.I)
    records=[]
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        if not (patterns.search(path.name) or 'xfm' in path.parts):
            continue
        record={'relative_path': str(path.relative_to(folder)), 'bytes':path.stat().st_size,
                'suffix':path.suffix.lower(),'potential_role':'UNKNOWN — name alone does not verify transform direction'}
        if path.suffix.lower()=='.json' and path.stat().st_size < 2_000_000:
            content=safe_json(path)
            if isinstance(content, dict):
                record['metadata_keys']=', '.join(str(k) for k in list(content)[:50])
                interesting={k:v for k,v in content.items() if re.search(r'space|coordinate|reference|transform|template|mni|warp|from|to|units',str(k),re.I)}
                record['coordinate_metadata_excerpt']=json.dumps(interesting,default=str)[:1200]
        records.append(record)
    return records


def inspect_head_surface_affine(path, point):
    """Inspect a donor-head→donor-surface transform; do NOT imply MNI mapping."""
    result={'path':str(path),'exists':path.is_file(),'applied':False,
            'warning':'head-to-surface remains in DONOR NATIVE space; not MNI'}
    if not path.is_file():
        return result
    try:
        import h5py
        with h5py.File(path,'r') as handle:
            if 'trans' not in handle:
                result['error']='HDF5 has no trans dataset'; return result
            matrix=np.asarray(handle['trans'][()],dtype=float)
        result['shape']=list(matrix.shape)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1],atol=1e-6):
            result['error']='Not a valid finite homogeneous 4x4 transform'; return result
        result['matrix']=matrix.tolist()
        result['donor_surface_coordinate_m']=(matrix @ np.r_[point,1])[:3].tolist()
        result['applied']=True
    except ImportError:
        result['error']='h5py not installed'
    except (OSError,ValueError,KeyError) as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    return result


def main():
    a=args_parser()
    run=a.run if a.run.startswith('run-') else f'run-{a.run}'
    root=a.dataset.expanduser().resolve()
    donor_dir=root/'derivatives'/'sourcemodelling'/a.donor
    if a.target==a.donor:
        raise SystemExit('Target and donor must differ')
    if not donor_dir.is_dir():
        raise SystemExit(f'Donor model directory not found: {donor_dir}')
    result_file=a.pilot_json or (Path('cross_subject/outputs/05_sesame_pilot')/f'{a.target}_with_{a.donor}_{run}'/'sesame_pilot_result.json')
    result_file=result_file.expanduser().resolve()
    if not result_file.is_file():
        raise SystemExit(f'Pilot result not found: {result_file} (use --pilot-json)')
    pilot=safe_json(result_file)
    if pilot.get('status')!='PASS' or pilot.get('target')!=a.target or pilot.get('donor')!=a.donor or pilot.get('run')!=run:
        raise SystemExit('Pilot JSON must report PASS and match target/donor/run')
    xyz=np.asarray(pilot.get('peak_coordinate_donor_src_rr_m'),dtype=float)
    if xyz.shape!=(3,) or not np.isfinite(xyz).all():
        raise SystemExit('Invalid donor source coordinate in pilot JSON')
    out=(a.output_root/f'{a.target}_with_{a.donor}_{run}').resolve()
    out.mkdir(parents=True,exist_ok=True)
    transform_path=donor_dir/'xfm'/f'{a.donor}_from-head_to-surface.h5'
    transform=inspect_head_surface_affine(transform_path,xyz)
    inventory=collect_transform_inventory(donor_dir)
    # Inspect only donor-owned files. The existence of MNI strings cannot
    # establish that an actual donor-native→MNI transform is available.
    hint_files=[x['relative_path'] for x in inventory if re.search('mni|template|warp|fsaverage|affine',x['relative_path'],re.I)]
    warnings=[]
    if float(pilot.get('peak_time_s',0)) < -0.002001 or float(pilot.get('peak_time_s',0)) > 0.002001:
        warnings.append('Pilot peak_time_s lies outside the -2 to +2 ms target interval; do not interpret as physical latency before checking SESAME time-axis conventions.')
    if not transform['applied']:
        warnings.append('Could not verify the donor head-to-surface transform; donor-native output remains untransformed.')
    if not hint_files:
        warnings.append('No donor file names indicate an MNI/template transform. An actual donor-native→MNI mapping must be sourced or derived and validated separately.')
    else:
        warnings.append('Candidate MNI/template files require independent checking of origin, destination, units, handedness and subject specificity.')
    report={
        'script':'06_cross_subject_coordinate_audit',
        'target':a.target,'donor':a.donor,'run':run,'method':pilot.get('method'),
        'reconstruction_source':str(result_file),
        'donor_peak_hemisphere':pilot.get('peak_hemisphere'),
        'donor_peak_vertex':pilot.get('peak_vertex'),
        'donor_forward_source_rr_coordinate_m':xyz.tolist(),
        'donor_native_head_to_surface_transform':transform,
        'donor_anatomical_metadata_inventory_count':len(inventory),
        'mni_template_candidate_filenames':hint_files,
        'warnings':warnings,
        'target_mri_accessed':False,
        'target_eeg_electrodes_accessed':False,
        'target_ground_truth_accessed':False,
        'donor_to_mni_mapping_verified':False,
        'localization_error_mm':None,
        'evaluation_status':'NOT READY: donor→MNI mapping unverified',
        'note':'This script never maps native donor coordinates to target-native coordinates and never calculates a misleading distance.'
    }
    json_file=out/'coordinate_audit.json'
    json_file.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    csv_file=out/'donor_transform_inventory.csv'
    with csv_file.open('w',newline='',encoding='utf-8') as f:
        fields=['relative_path','bytes','suffix','potential_role','metadata_keys','coordinate_metadata_excerpt']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(inventory)
    print('Cross-subject coordinate audit completed')
    print('Donor surface transform applied:',transform['applied'])
    print('Donor-native→MNI mapping verified: NO')
    print('Localization error: NOT CALCULATED')
    for warning in warnings: print('WARNING:',warning)
    print('Audit:',json_file)
    print('Inventory:',csv_file)

if __name__=='__main__':
    main()
