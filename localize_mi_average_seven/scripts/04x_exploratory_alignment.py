"""Estimated-landmark alignment and initialization sensitivity; no anatomical approval."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from mne.io.constants import FIFF

from common import ROOT, EXP, digest
from execution import validate_mesh, module


def initialization(points):
    # Points ordered LPA, NAS, RPA; output maps HEAD metres to MRI metres.
    return np.linalg.inv(mne.transforms.get_ras_to_neuromag_trans(points[1], points[0], points[2]))


def rigid_checks(matrix):
    rotation = matrix[:3, :3]
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous transform')
    if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(rotation)-1)>1e-8:
        raise ValueError('Transform must be rigid without reflection or scaling')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=EXP/'outputs/04x_exploratory_alignment')
    parser.add_argument('--fsaverage-root', type=Path, default=Path.home()/'mne_data/MNE-fsaverage-data/fsaverage')
    args = parser.parse_args()
    cfg = json.loads((EXP/'config/experiment.json').read_text())
    cp = cfg['coregistration']
    helper = module(ROOT/'cross_subject_fsaverage/scripts/02_coregister_eeg.py', 'exploratory_icp')
    repaired = EXP/'work/bem_candidates/local_inner_1mm'
    candidate_validation = json.loads((repaired/'validation.json').read_text())
    if candidate_validation.get('bem_solution') != 'PASS':
        raise ValueError('Repaired candidate must pass numerical BEM checks')
    arms = {'average_seven': repaired/'validation_subjects/average_seven',
            'fsaverage': args.fsaverage_root.expanduser().resolve()}
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    hashes = {str(p): digest(p) for p in [Path(__file__).resolve(), Path(helper.__file__),
              EXP/'config/experiment.json', repaired/'repair.json', repaired/'validation.json']}
    data = {}
    for subject in cfg['participants']:
        path = ROOT/cfg['dataset']/'derivatives/sourcemodelling'/subject/'fwd'/f'{subject}_fwd.fif'
        hashes[str(path)] = digest(path)
        fwd = mne.read_forward_solution(path, verbose=False)
        channels = fwd['info']['chs']
        points = np.array([ch['loc'][:3] for ch in channels])
        if not np.isfinite(points).all() or any(int(ch['coord_frame'])!=int(FIFF.FIFFV_COORD_HEAD) for ch in channels):
            raise ValueError('Expected finite native HEAD sensor coordinates in metres')
        if len(set(fwd['info']['ch_names'])) != len(points):
            raise ValueError('Duplicate channels')
        data[subject] = (fwd['info']['ch_names'], points)
    rows, sensitivity, sensor_rows, plot_data = [], [], [], {}
    for arm, anatomy in arms.items():
        output = args.output_root/arm
        output.mkdir()
        scalp_file = anatomy/'bem/outer_skin.surf'
        hashes[str(scalp_file)] = digest(scalp_file)
        scalp, faces = mne.read_surface(scalp_file)
        validate_mesh(scalp/1000, faces)
        scalp = scalp/1000
        if not .05 < np.percentile(np.linalg.norm(scalp, axis=1), 95) < .25:
            raise ValueError('Unexpected scalp units')
        # Identical estimation policy in both arms, rather than user-clicked landmarks.
        fids = mne.coreg.get_mni_fiducials(anatomy.name, subjects_dir=anatomy.parent)
        by_id = {int(d['ident']): np.array(d['r'], dtype=np.float64) for d in fids}
        if any(int(d['coord_frame']) != int(FIFF.FIFFV_COORD_MRI) for d in fids):
            raise ValueError('Estimated fiducials must be MRI surface-RAS metres')
        fidpoints = np.array([by_id[i] for i in [1, 2, 3]])
        fidfile = output/f'{arm}-estimated-fiducials.fif'
        mne.io.write_fiducials(fidfile, fids, coord_frame='mri', overwrite=False)
        hashes[str(fidfile.resolve())] = digest(fidfile)
        if arm == 'average_seven':
            for name in ['mri/T1.mgz', 'mri/transforms/talairach.xfm']:
                hashes[str(anatomy/name)] = digest(anatomy/name)
        perturbations = [('nominal', fidpoints.copy())]
        for landmark in range(3):
            for axis in range(3):
                for sign in [-1, 1]:
                    points = fidpoints.copy()
                    points[landmark, axis] += sign*.002
                    perturbations.append((f'{["LPA","NAS","RPA"][landmark]}_{"xyz"[axis]}_{sign*2:+d}mm', points))
        tree = cKDTree(scalp)
        for subject, (names, sensors) in data.items():
            fits = []
            for label, points in perturbations:
                start = initialization(points)
                rigid_checks(start)
                matrix, distances, iterations = helper.icp(sensors, scalp, start, cp['iterations'], cp['trim_fraction'])
                rigid_checks(matrix)
                mapped = helper.apply(matrix, sensors)
                med, p95 = np.percentile(distances*1000, [50, 95])
                numeric_pass = bool(med <= cp['median_limit_mm'] and p95 <= cp['p95_limit_mm'])
                if label == 'nominal':
                    nominal, nominal_mapped = matrix, mapped
                    path = output/f'{subject}-trans.fif'
                    mne.write_trans(path, mne.transforms.Transform('head', 'mri', matrix), overwrite=False)
                    hashes[str(path.resolve())] = digest(path)
                    plot_data[arm, subject] = (scalp, mapped, distances, fidpoints)
                    nearest = tree.query(mapped)[1]
                    for name, before, after, closest, dist in zip(names, sensors, mapped, scalp[nearest], distances):
                        sensor_rows.append(dict(arm=arm, test_subject=subject, channel=name,
                            head_x_m=before[0], head_y_m=before[1], head_z_m=before[2],
                            mri_x_m=after[0], mri_y_m=after[1], mri_z_m=after[2],
                            nearest_scalp_x_m=closest[0], nearest_scalp_y_m=closest[1], nearest_scalp_z_m=closest[2],
                            distance_mm=dist*1000))
                    rows.append(dict(arm=arm, test_subject=subject, median_mm=med, p95_mm=p95,
                        numerical_pass=numeric_pass, anatomy_review='PENDING', fiducial_review='ESTIMATED_UNREVIEWED',
                        localization_error_valid=False, transform=str(path.resolve())))
                shift = np.linalg.norm(mapped-nominal_mapped, axis=1)*1000
                rotation = np.degrees(np.arccos(np.clip((np.trace(matrix[:3,:3]@nominal[:3,:3].T)-1)/2, -1, 1)))
                item = dict(arm=arm, test_subject=subject, perturbation=label, median_mm=med, p95_mm=p95,
                    numerical_pass=numeric_pass, iterations=iterations,
                    sensor_rms_shift_mm=float(np.sqrt(np.mean(shift**2))), sensor_max_shift_mm=float(shift.max()),
                    rotation_change_deg=float(rotation), initial_head_to_mri=start.tolist(), final_head_to_mri=matrix.tolist())
                fits.append(item)
                sensitivity.append(item)
            print(arm, subject, f'nominal median={fits[0]["median_mm"]:.2f}, p95={fits[0]["p95_mm"]:.2f} mm; '
                  f'worst perturbation RMS shift={max(f["sensor_rms_shift_mm"] for f in fits):.2f} mm', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root/'01_alignment.csv', index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ['initial_head_to_mri','final_head_to_mri']} for row in sensitivity]).to_csv(args.output_root/'02_sensitivity.csv', index=False)
    (args.output_root/'03_all_transforms.json').write_text(json.dumps(sensitivity, indent=2)+'\n')
    pd.DataFrame(sensor_rows).to_csv(args.output_root/'04_sensors.csv', index=False)
    for subject in cfg['participants']:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        for row, arm in enumerate(arms):
            scalp, mapped, distances, fids = plot_data[arm, subject]
            for col, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)]):
                ax = axes[row, col]
                ax.scatter(scalp[:, x]*1000, scalp[:, y]*1000, s=.4, c='lightgray', rasterized=True)
                scatter = ax.scatter(mapped[:, x]*1000, mapped[:, y]*1000, s=10, c=distances*1000, vmin=0, vmax=25, cmap='viridis')
                ax.scatter(fids[:, x]*1000, fids[:, y]*1000, marker='x', c='red', s=45)
                for label, point in zip(['LPA', 'NAS', 'RPA'], fids):
                    ax.annotate(label, (point[x]*1000, point[y]*1000), fontsize=8)
                ax.set(xlabel=f'MRI {"xyz"[x]} (mm)', ylabel=f'MRI {"xyz"[y]} (mm)', title=arm)
                ax.set_aspect('equal')
        fig.suptitle(f'{subject}: estimated-landmark rigid alignment — UNREVIEWED\n'
                     'Gray: scalp projections; colored: EEG sensors; red: estimated landmarks')
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Nearest scalp vertex distance (mm)', shrink=.6)
        fig.savefig(args.output_root/f'overlay_{subject}.png', dpi=130)
        plt.close(fig)
    manifest = dict(status='EXPLORATORY_ALIGNMENT_COMPLETE', anatomy_review='PENDING',
        localization_error_valid=False, configuration=cfg, input_sha256=hashes,
        mne_version=mne.__version__, numpy_version=np.__version__,
        fiducial_policy='MNE get_mni_fiducials in both arms; average template uses inverse Talairach affine',
        protocol_override='Estimated rather than manually reviewed MRI fiducials; formal experiment config unchanged',
        icp=cp, sensitivity='18 one-landmark/one-axis +/-2 mm initialization perturbations; not confidence intervals',
        interpretation='Scalp proximity and initialization stability do not validate anatomical correspondence.',
        no_inverse_or_forward_jobs_run=True, formal_review_gate_unchanged=True)
    (args.output_root/'05_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
