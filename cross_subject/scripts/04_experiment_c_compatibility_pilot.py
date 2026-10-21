#!/usr/bin/env python3
"""Experiment C: MRI-free and digitizer-free cross-subject compatibility pilot.

Target (sub-02): EEG epoch array only. Donor (sub-01): complete existing FIF
forward model, including its electrodes, source space and anatomy.

This script deliberately DOES NOT use target forward, MRI, surface, sensor
coordinates, electrode TSV, or ground truth. It does not compute localization
or a localization error. Its purpose is to validate assumptions before inverse
modeling. Epoch .npy arrays have no channel-label metadata: ordering must be
verified from the original dataset preprocessing code before interpretation.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import mne

DEFAULT_ROOT = Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--target', default='sub-02')
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--run', default='01', help='Target EEG run number')
    p.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/04_experiment_c_pilot'))
    p.add_argument('--assume-sequential-channels', action='store_true',
                   help='Explicitly allow provisional e1,e2,...e256 mapping. This does NOT verify ordering.')
    args = p.parse_args()
    root = args.dataset_root.expanduser().resolve()
    run = f'{int(args.run):02d}'
    eeg_file = root / 'derivatives' / 'epochs' / args.target / 'eeg' / f'{args.target}_task-seegstim_run-{run}_epochs.npy'
    fwd_file = root / 'derivatives' / 'sourcemodelling' / args.donor / 'fwd' / f'{args.donor}_fwd.fif'
    for file in (eeg_file, fwd_file):
        if not file.is_file():
            p.error(f'Missing input: {file}')
    arr = np.load(eeg_file, mmap_mode='r', allow_pickle=False)
    fwd = mne.read_forward_solution(fwd_file, verbose='ERROR')
    donor_names = list(fwd['info']['ch_names'])
    checks = []
    def check(name, passed, detail, severity='ERROR'):
        checks.append({'check': name, 'passed': bool(passed), 'severity': severity, 'detail': str(detail)})
    check('Epoch array is 3D', arr.ndim == 3, f'actual shape={arr.shape}')
    n_chan = int(arr.shape[1]) if arr.ndim == 3 else -1
    check('EEG channel count matches donor forward', n_chan == len(donor_names),
          f'target npy channels={n_chan}; donor FIF channels={len(donor_names)}')
    expected = [f'e{i}' for i in range(1, n_chan+1)]
    check('Donor channel labels are sequential e1..eN', donor_names == expected,
          f'first={donor_names[:5]}, last={donor_names[-5:]}')
    check('Forward lead field is finite', np.isfinite(fwd['sol']['data']).all(),
          f'leadfield shape={fwd["sol"]["data"].shape}')
    check('Donor forward channel dimension consistent', fwd['sol']['data'].shape[0] == n_chan,
          f'rows={fwd["sol"]["data"].shape[0]}, channels={n_chan}')
    check('Target/donor different', args.target != args.donor, f'{args.target} vs {args.donor}')
    check('Explicit acceptance of unverified NPY channel order', args.assume_sequential_channels,
          'NPY format has no channel-name metadata; verify e1..e256 ordering from preprocessing source code.',
          severity='BLOCKER')
    chunks = []
    n_epoch = arr.shape[0] if arr.ndim == 3 else 0
    for k in range(n_epoch):
        epoch = np.asarray(arr[k])
        chunks.append({'epoch': k, 'finite': bool(np.isfinite(epoch).all()),
                       'min': float(np.nanmin(epoch)), 'max': float(np.nanmax(epoch)),
                       'max_abs': float(np.nanmax(np.abs(epoch)))})
    check('All epoch signals finite', all(x['finite'] for x in chunks), f'{sum(not x["finite"] for x in chunks)} bad epochs')
    check('Nonempty run', n_epoch > 0, f'epochs={n_epoch}')
    bads = sorted(set(fwd['info'].get('bads', [])))
    summary = {
        'experiment': 'C: use target EEG only with donor original forward; no subject-specific MRI or digitization',
        'target': args.target, 'donor': args.donor, 'run': run,
        'inputs': {'target_eeg_epochs_npy': str(eeg_file), 'donor_forward_fif': str(fwd_file)},
        'not_accessed': ['target MRI', 'target forward FIF', 'target EEG electrode coordinates',
                         'target head surfaces', 'electrode TSV', 'target ground truth'],
        'epoch_shape': list(arr.shape), 'epoch_dtype': str(arr.dtype),
        'donor_nsource': int(fwd['nsource']), 'donor_nchan': int(fwd['nchan']),
        'donor_leadfield_shape': list(fwd['sol']['data'].shape),
        'donor_forward_bad_channels': bads,
        'sampling_hz_expected_from_prior_experiments': 8000,
        'channel_mapping': ('UNVERIFIED: ASSUMED e1..e256 from NPY axis 1'
                            if args.assume_sequential_channels else 'NOT ACCEPTED; verify dataset code first'),
        'checks': checks,
        'passed_required_checks': all(c['passed'] for c in checks),
        'notes': [
            'Passing these checks is not proof that channel mapping or EEG referencing is correct.',
            'Original run-specific bad-channel lists and signal units must be verified before inverse modeling.',
            'Do not compare donor-native coordinates directly to target-native ground truth.',
            'Source estimates can be evaluated only after an independent donor-to-target/common-space mapping.',
            'No target subject anatomical information is permitted during reconstruction.'
        ]
    }
    out = args.output_dir.expanduser().resolve() / f'{args.target}_with_{args.donor}_run-{run}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'compatibility_report.json').write_text(json.dumps(summary, indent=2))
    with (out/'checks.csv').open('w', newline='') as fp:
        wr=csv.DictWriter(fp, fieldnames=['check','passed','severity','detail']);wr.writeheader();wr.writerows(checks)
    with (out/'signal_summary.csv').open('w', newline='') as fp:
        wr=csv.DictWriter(fp, fieldnames=['epoch','finite','min','max','max_abs']);wr.writeheader();wr.writerows(chunks)
    print(f'Experiment C pilot: {args.target} EEG + {args.donor} donor forward')
    print(f'EEG epochs: {arr.shape} | forward matrix: {fwd["sol"]["data"].shape}')
    for c in checks:
        print(f'{"PASS" if c["passed"] else "FAIL"} {c["check"]}: {c["detail"]}')
    print(f'Report: {out}/compatibility_report.json')
    print('NEXT: verify channel ordering, referencing, units, and bad channels before running inverse methods.')
    if not summary['passed_required_checks']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
