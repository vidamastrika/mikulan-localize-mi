"""Audit frames, scalp extents and rigid ICP basins; never replace accepted fits."""
import argparse
import json
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from mne.bem import _extract_volume_info

from common import ROOT, EXP, digest
from execution import module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment-root', type=Path, default=EXP/'outputs/04x_exploratory_alignment_retry1')
    parser.add_argument('--output-root', type=Path, default=EXP/'outputs/04y_scalp_fit_diagnosis')
    args = parser.parse_args()
    cfg = json.loads((EXP/'config/experiment.json').read_text())
    manifest = json.loads((args.alignment_root/'05_manifest.json').read_text())
    for filename, sha in manifest['input_sha256'].items():
        if digest(Path(filename)) != sha:
            raise ValueError(f'Alignment provenance changed: {filename}')
    args.output_root.mkdir(parents=True, exist_ok=False)
    helper = module(ROOT/'cross_subject_fsaverage/scripts/02_coregister_eeg.py', 'diagnostic_icp')
    nominal = pd.read_csv(args.alignment_root/'01_alignment.csv')
    source = EXP/'work/bem_candidates/local_inner_1mm/validation_subjects/average_seven'
    anatomy = {'average_seven': source, 'fsaverage': Path.home()/'mne_data/MNE-fsaverage-data/fsaverage'}
    native_rows, extents, basin_rows, headers = [], [], [], []
    hashes = {str(p): digest(p) for p in [Path(__file__).resolve(), Path(helper.__file__),
               args.alignment_root/'05_manifest.json', args.alignment_root/'01_alignment.csv']}
    prepared = {}
    for arm, subject in anatomy.items():
        path = subject/'bem/outer_skin.surf'
        rr, tris, metadata = mne.read_surface(path, read_metadata=True)
        scalp = rr/1000
        t1 = subject/'mri/T1.mgz'
        expected = _extract_volume_info(str(t1))
        mismatch = [key for key in ['volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras']
                    if not np.allclose(metadata[key], expected[key], rtol=0, atol=1e-5)]
        image = nib.load(t1)
        bridge = image.affine@np.linalg.inv(image.header.get_vox2ras_tkr())
        surface_roundtrip = nib.affines.apply_affine(np.linalg.inv(bridge), nib.affines.apply_affine(bridge, rr))
        headers.append(dict(arm=arm, metadata_mismatch_keys=mismatch,
                            surface_scanner_ras_roundtrip_max_mm=float(np.linalg.norm(rr-surface_roundtrip,axis=1).max()),
                            voxel_sizes_mm=list(map(float,image.header.get_zooms()[:3])),
                            note='Affine handedness is an image-axis convention, not evidence of a reflected rigid fit'))
        fidfile = args.alignment_root/arm/f'{arm}-estimated-fiducials.fif'
        fids, _ = mne.io.read_fiducials(fidfile)
        points = {int(d['ident']): np.array(d['r'],dtype=float) for d in fids}
        initial = np.linalg.inv(mne.transforms.get_ras_to_neuromag_trans(points[2],points[1],points[3]))
        prepared[arm] = scalp, initial
        extents.append(dict(model=arm, x_extent_mm=np.ptp(rr[:,0]), y_extent_mm=np.ptp(rr[:,1]), z_extent_mm=np.ptp(rr[:,2])))
        for path in [path,t1,fidfile]:hashes[str(path.resolve())]=digest(path)
    for subject in cfg['participants']:
        root = ROOT/cfg['dataset']/'derivatives/sourcemodelling'/subject
        forward_path = root/'fwd'/f'{subject}_fwd.fif'
        surface_path = root/'anat'/f'{subject}_outer_skin.surf.gii'
        forward = mne.read_forward_solution(forward_path, verbose=False)
        sensors = np.array([ch['loc'][:3] for ch in forward['info']['chs']])
        native_trans = forward['mri_head_t']
        if int(native_trans['from']) != 5 or int(native_trans['to']) != 4:
            raise ValueError('Expected MRI-to-HEAD native transform')
        image = nib.load(surface_path)
        points = next(a for a in image.darrays if a.intent==1008)
        rr = points.data
        cras = np.array([float(points.meta[f'VolGeomC_{axis}']) for axis in ['R','A','S']])
        native_inverse = np.linalg.inv(native_trans['trans'])
        distances = cKDTree(rr/1000).query(helper.apply(native_inverse,sensors))[0]*1000
        native_rows.append(dict(test_subject=subject,
            rotation_identity_max_error=float(np.abs(native_trans['trans'][:3,:3]-np.eye(3)).max()),
            translation_minus_gifti_cras_max_mm=float(np.abs(native_trans['trans'][:3,3]*1000-cras).max()),
            native_median_mm=float(np.median(distances)),native_p95_mm=float(np.percentile(distances,95)),
            gifti_coordinate_system=str(points.coordsys),
            conclusion='Native HEAD convention not independently established by these numerical checks'))
        extents.append(dict(model=subject, x_extent_mm=np.ptp(rr[:,0]), y_extent_mm=np.ptp(rr[:,1]), z_extent_mm=np.ptp(rr[:,2])))
        for path in [forward_path,surface_path]:hashes[str(path.resolve())]=digest(path)
        for arm,(scalp,initial) in prepared.items():
            record = nominal[(nominal.arm==arm)&(nominal.test_subject==subject)].iloc[0]
            nominal_matrix = mne.read_trans(record['transform'])['trans']
            starts = [('estimated_landmarks',initial),('identity_axes',np.eye(4)),
                      ('native_inverse_diagnostic',native_inverse),('continue_previous_fit',nominal_matrix)]
            # Fixed diagnostic starts, not a new production selection rule.
            center = helper.apply(initial,sensors).mean(0)
            for axis in range(3):
                for sign in [-1,1]:
                    translation=np.eye(4);translation[axis,3]=sign*.01
                    starts.append((f'translate_{"xyz"[axis]}_{sign*10:+d}mm',translation@initial))
                    vector=np.zeros(3);vector[axis]=np.deg2rad(sign*15)
                    rotation=Rotation.from_rotvec(vector).as_matrix()
                    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=center-rotation@center
                    starts.append((f'rotate_{"xyz"[axis]}_{sign*15:+d}deg',transform@initial))
            for label,start in starts:
                mat,d,it=helper.icp(sensors,scalp,start,40,.15)
                if not np.allclose(mat[:3,:3].T@mat[:3,:3],np.eye(3),atol=1e-7,rtol=0) or abs(np.linalg.det(mat[:3,:3])-1)>1e-7:
                    raise ValueError('Diagnostic transform is not proper rigid')
                median,p95=np.percentile(d*1000,[50,95])
                shift=helper.apply(mat,sensors)-helper.apply(nominal_matrix,sensors)
                basin_rows.append(dict(arm=arm,test_subject=subject,start=label,median_mm=median,p95_mm=p95,
                    numerical_pass=bool(median<=10 and p95<=20),iterations=it,
                    sensor_rms_shift_from_nominal_mm=float(np.sqrt(np.mean(np.sum(shift**2,axis=1)))*1000),
                    initial_transform=start.tolist(),final_transform=mat.tolist()))
            subset=basin_rows[-len(starts):]
            print(arm,subject,'diagnostic pass starts',sum(r['numerical_pass'] for r in subset),'/',len(starts),
                  'p95 range',round(min(r['p95_mm'] for r in subset),2),round(max(r['p95_mm'] for r in subset),2),flush=True)
    pd.DataFrame(native_rows).to_csv(args.output_root/'01_native_frame_audit.csv',index=False)
    pd.DataFrame(extents).to_csv(args.output_root/'02_scalp_extents.csv',index=False)
    (args.output_root/'03_header_audit.json').write_text(json.dumps(headers,indent=2)+'\n')
    (args.output_root/'04_basin_transforms.json').write_text(json.dumps(basin_rows,indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in row.items() if k not in ['initial_transform','final_transform']} for row in basin_rows]).to_csv(args.output_root/'05_basin_metrics.csv',index=False)
    report=dict(status='DIAGNOSTIC_COMPLETE',anatomy_review='PENDING',localization_error_valid=False,
        formal_transforms_replaced=False,scaling=False,input_sha256=hashes,
        caveats=['Scalp extents depend on surface coverage and orientation.',
                 'Native inverse is a diagnostic start, not native-to-template MRI registration.',
                 'Native HEAD tags do not establish a conventional fiducial-defined frame.',
                 'Starting positions diagnose ICP basins; no automatic best-fit selection is adopted.'])
    (args.output_root/'06_manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
