#!/usr/bin/env python3
"""Localize-MI cross-subject Script 03: READ-ONLY geometry feasibility pilot.

Compare the target EEG sensor locations EMBEDDED IN THE FORWARD FIF file
with target (control) and donor scalp surfaces. EEG electrode TSVs are not used. This is NOT a registration, not a forward-model calculation,
and not a localization validation. The script deliberately keeps uncertain
surface coordinate conventions visible rather than silently fitting anatomy.

Outputs: geometry_metrics.csv, registration_pilot.json, geometry_pilot.png,
and report.txt. Only writes under cross_subject/outputs/03_registration_pilot.

Dependencies: mne, numpy, pandas, nibabel, scipy, matplotlib.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_DATA = Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI'
DEFAULT_OUT = Path(__file__).resolve().parents[1] / 'outputs' / '03_registration_pilot'


def load_forward_sensor_locations(root: Path, sub: str):
    import mne
    from mne.io.constants import FIFF
    path = root / 'derivatives' / 'sourcemodelling' / sub / 'fwd' / f'{sub}_fwd.fif'
    if not path.is_file():
        raise FileNotFoundError(path)
    fwd = mne.read_forward_solution(str(path), verbose='ERROR')
    info = fwd['info']
    chans = [(ch['ch_name'], np.asarray(ch['loc'][:3], dtype=float), ch['coord_frame'])
             for ch in info['chs'] if ch['kind'] == FIFF.FIFFV_EEG_CH]
    if not chans or not all(cf == FIFF.FIFFV_COORD_HEAD for _, _, cf in chans):
        raise ValueError(f'{sub}: expected valid EEG sensors in MNE HEAD coordinates')
    sensors = np.stack([x[1] for x in chans])
    if not np.isfinite(sensors).all() or np.any(np.linalg.norm(sensors, axis=1) < 0.02):
        raise ValueError(f'{sub}: sensor positions seem invalid')
    t = fwd.get('mri_head_t')
    if t is None or t['from'] != FIFF.FIFFV_COORD_MRI or t['to'] != FIFF.FIFFV_COORD_HEAD:
        raise ValueError(f'{sub}: MRI -> head transformation is unavailable/incorrect')
    matrix = np.asarray(t['trans'], dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{sub}: malformed MRI -> head matrix')
    return [x[0] for x in chans], sensors, matrix


def read_outer_skin(root: Path, sub: str):
    import nibabel as nib
    from nibabel.nifti1 import intent_codes
    path = root / 'derivatives' / 'sourcemodelling' / sub / 'anat' / f'{sub}_outer_skin.surf.gii'
    if not path.is_file():
        raise FileNotFoundError(path)
    g = nib.load(str(path))
    verts = [np.asarray(a.data, dtype=float) for a in g.darrays
             if int(a.intent) == int(intent_codes['NIFTI_INTENT_POINTSET'])]
    if len(verts) != 1 or verts[0].ndim != 2 or verts[0].shape[1] != 3:
        raise ValueError(f'{sub}: could not uniquely identify POINTSET vertices in {path}')
    v = verts[0]
    if not np.isfinite(v).all():
        raise ValueError(f'{sub}: nonfinite surface coordinates')
    # Some GIFTI surfaces may be encoded in millimetres. Preserve this as explicit metadata.
    scale = 1e-3 if float(np.percentile(np.linalg.norm(v, axis=1), 50)) > 2 else 1.0
    return v * scale, {'path': str(path), 'vertices': len(v),
                       'detected_unit': 'mm' if scale == 1e-3 else 'm',
                       'metadata': dict(g.meta),
                       'coord_systems': [
                           {'dataspace': int(a.coordsys.dataspace), 'xformspace': int(a.coordsys.xformspace),
                            'xform': np.asarray(a.coordsys.xform).tolist()}
                           for a in g.darrays if a.coordsys is not None]}


def apply_4x4(points: np.ndarray, mat: np.ndarray):
    return points @ mat[:3, :3].T + mat[:3, 3]


def metrics(electrodes: np.ndarray, scalp: np.ndarray):
    dist, _ = cKDTree(scalp).query(electrodes)
    return {'median_distance_mm': round(float(np.median(dist) * 1000), 3),
            'mean_distance_mm': round(float(np.mean(dist) * 1000), 3),
            'p95_distance_mm': round(float(np.percentile(dist, 95) * 1000), 3),
            'max_distance_mm': round(float(np.max(dist) * 1000), 3),
            'count_over_15mm': int(np.sum(dist > .015)),
            'count_over_25mm': int(np.sum(dist > .025)),
            'n_electrodes': len(electrodes)}


def plot_comparison(target_elec, surfaces, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    for i, (label, scalp, note) in enumerate(surfaces):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        # Downsample displayed vertices only; metrics always use the full surface.
        skip = max(1, len(scalp) // 1800)
        s = scalp[::skip] * 1000
        p = target_elec * 1000
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=1, alpha=.18, color='steelblue', rasterized=True)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=8, alpha=.85, color='darkorange', rasterized=True)
        both = np.concatenate([s, p], axis=0)
        center = (both.max(axis=0) + both.min(axis=0)) / 2
        half = float(np.max(both.max(axis=0) - both.min(axis=0)) / 2 * 1.12)
        ax.set_xlim(center[0]-half, center[0]+half)
        ax.set_ylim(center[1]-half, center[1]+half)
        ax.set_zlim(center[2]-half, center[2]+half)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.view_init(elev=22, azim=-65)
        ax.set_title(label + '\n' + note, fontsize=10)
    fig.suptitle('FIF EEG sensors vs scalp models (no registration performed)')
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--eeg-subject', default='sub-02')
    ap.add_argument('--donor-subject', default='sub-01')
    args = ap.parse_args()
    if args.eeg_subject == args.donor_subject:
        ap.error('Select different EEG and donor subjects')
    args.output_root.mkdir(parents=True, exist_ok=True)

    target_names, sensors, target_mat = load_forward_sensor_locations(args.dataset_root, args.eeg_subject)
    donor_names, donor_sensors, donor_mat = load_forward_sensor_locations(args.dataset_root, args.donor_subject)
    control_skin, control_meta = read_outer_skin(args.dataset_root, args.eeg_subject)
    donor_skin, donor_meta = read_outer_skin(args.dataset_root, args.donor_subject)

    rows = []
    surfaces = {}
    for tag, skin, mat in [('control', control_skin, target_mat), ('donor', donor_skin, donor_mat)]:
        # Two explicit, competing interpretations because coordinate convention of GIFTI
        # is not guaranteed by filename alone. We do NOT optimize or select a model for use.
        for interpretation, positions in [('GIFTI_as_head', skin),
                                          ('GIFTI_as_MRI_then_mri_head_t', apply_4x4(skin, mat))]:
            surfaces[(tag, interpretation)] = positions
            rows.append({'model': tag, 'subject': args.eeg_subject if tag == 'control' else args.donor_subject,
                         'interpretation': interpretation, **metrics(sensors, positions)})

    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / 'geometry_metrics.csv', index=False)
    # Plot a consistent hypothesis across both subjects. Plot is for inspection only.
    visual_interpretation = 'GIFTI_as_MRI_then_mri_head_t'
    plot_comparison(sensors,
                    [(f'Control: {args.eeg_subject}', surfaces[('control', visual_interpretation)],
                      visual_interpretation),
                     (f'Donor: {args.donor_subject}', surfaces[('donor', visual_interpretation)],
                      visual_interpretation)],
                    args.output_root / 'geometry_pilot.png')

    # All electrode positions come from the original MNE forward FIF metadata.
    # This diagnostic difference does not establish anatomical registration.
    sensor_pairs_mm = None
    if target_names == donor_names:
        sensor_pairs_mm = np.linalg.norm(sensors - donor_sensors, axis=1) * 1000
    result = {
        'eeg_subject': args.eeg_subject, 'donor_subject': args.donor_subject,
        'n_eeg_sensors_forward': len(sensors), 'n_donor_sensors_forward': len(donor_sensors),
        'channel_names_identical': target_names == donor_names,
        'sensor_source': 'existing forward FIF info/channels (HEAD coordinates)',
        'eeg_electrode_tsv_used': False,
        'paired_forward_sensor_distance_mm': {
            'median': float(np.median(sensor_pairs_mm)),
            'mean': float(np.mean(sensor_pairs_mm)),
            'max': float(np.max(sensor_pairs_mm))
        } if sensor_pairs_mm is not None else None,
        'target_mri_to_head': target_mat.tolist(), 'donor_mri_to_head': donor_mat.tolist(),
        'control_surface': control_meta, 'donor_surface': donor_meta,
        'geometry_metrics': rows,
        'limitations': [
            'This is geometry inspection, NOT cross-subject registration.',
            'GIFTI coordinate convention must be verified before using any candidate distances as physical alignment quality.',
            'Nearest-vertex distance is unsigned; it cannot detect whether sensors lie inside the scalp.',
            'No transformation from donor anatomy to target participant has been calculated.',
            'No ground-truth or stimulation coordinates were used.',
            'No forward solutions were recomputed or modified.',
            'EEG TSV files were not used; positions come from target FIF forward metadata.'
        ]}
    (args.output_root/'registration_pilot.json').write_text(json.dumps(result, indent=2, default=str))
    lines = [f'EEG subject: {args.eeg_subject}, donor anatomy: {args.donor_subject}',
             f'Target forward sensors: {len(sensors)}; donor forward sensors: {len(donor_sensors)}',
             f'Channel names identical: {target_names == donor_names}',
             'Sensor source: pre-existing FIF forward solutions; EEG TSV files NOT used.',
             f'Median paired sensor difference (mm): {float(np.median(sensor_pairs_mm)) if sensor_pairs_mm is not None else "N/A"}', '',
             table.to_string(index=False), '',
             'CAUTION: raw surface/EEG coordinates are NOT a cross-subject registration.',
             'Inspect both candidate surface conventions, self-control and donor geometry.',
             'Check anatomical orientation and GIFTI frame convention before producing a new forward.',
             'A high donor mismatch can represent frame error, not genuine anatomical mismatch.']
    report = '\n'.join(lines) + '\n'
    (args.output_root/'report.txt').write_text(report)
    print(report)
    print(f'Wrote outputs to {args.output_root}')


if __name__ == '__main__':
    main()
