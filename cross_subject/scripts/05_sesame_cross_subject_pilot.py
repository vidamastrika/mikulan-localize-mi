#!/usr/bin/env python3
"""Cross-subject SESAME pilot: target EEG + donor forward without target anatomy.

Reconstructs in donor source space ONLY. No localization error is computed,
because donor source coordinates and target stimulation truth require a validated
common-space mapping. Uses target channels.tsv, NOT EEG electrodes.tsv.

Baseline operationalization follows earlier Script 23:
 fs=0.15; noise_std = fs * max(abs(target Evoked data))
 n_parts=100, max_n_dips=1, lam=0.25, hyper_q=True, seed=0,
 noise_cov=None. SESAME does NOT use prestimulus covariance in this setup.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI')
    p.add_argument('--target', default='sub-02')
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--run', default='01', help='01 or run-01')
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/05_sesame_pilot'))
    p.add_argument('--fs', type=float, default=0.15)
    p.add_argument('--n-parts', type=int, default=100)
    p.add_argument('--max-n-dips', type=int, default=1)
    p.add_argument('--lam', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def load_inputs(a):
    run = a.run if a.run.startswith('run-') else f'run-{a.run}'
    root = a.dataset.expanduser().resolve()
    eeg_dir = root / 'derivatives' / 'epochs' / a.target / 'eeg'
    stem = f'{a.target}_task-seegstim_{run}'
    ep_path = eeg_dir / f'{stem}_epochs.npy'
    ch_path = eeg_dir / f'{stem}_channels.tsv'
    events_path = eeg_dir / f'{stem}_epochs.tsv'
    fwd_path = root / 'derivatives' / 'sourcemodelling' / a.donor / 'fwd' / f'{a.donor}_fwd.fif'
    for path in (ep_path, ch_path, events_path, fwd_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    arr = np.load(ep_path)
    channels = pd.read_csv(ch_path, sep='\t')
    events = pd.read_csv(events_path, sep='\t')
    if arr.ndim != 3 or arr.shape[1] != len(channels) or arr.shape[0] != len(events):
        raise ValueError(f'Inconsistent epochs/channels/events: {arr.shape}, {len(channels)}, {len(events)}')
    if not np.isfinite(arr).all():
        raise ValueError('EEG contains nonfinite samples')
    for col in ('name', 'status', 'sampling_frequency', 'units'):
        if col not in channels:
            raise ValueError(f'Missing metadata column {col}')
    names = channels['name'].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate EEG channel names')
    sfreqs = channels['sampling_frequency'].astype(float).unique()
    if len(sfreqs) != 1:
        raise ValueError(f'Inconsistent sampling frequency: {sfreqs}')
    units = sorted(set(channels['units'].dropna().astype(str).str.strip().str.lower()))
    if not units or not set(units).issubset({'v', 'volt', 'volts'}):
        raise ValueError(f'EEG unit metadata not confirmed as volts: {units}; inspect before conversion')
    statuses = channels['status'].astype(str).str.strip().str.lower()
    if not statuses.isin({'good', 'bad'}).all():
        raise ValueError('Unexpected channel status: ' + str(sorted(set(statuses))))
    good_names = channels.loc[statuses.eq('good'), 'name'].astype(str).tolist()
    bad_names = channels.loc[statuses.eq('bad'), 'name'].astype(str).tolist()
    if len(good_names) < 20:
        raise ValueError('Too few good EEG channels')
    if 'zero_time' not in events:
        raise ValueError('Missing events zero_time needed to reconstruct epoch time axis')
    zero = events['zero_time'].astype(float).unique()
    if len(zero) != 1:
        raise ValueError('Inconsistent zero_time')
    forward = mne.read_forward_solution(fwd_path, verbose=False)
    fwd_names = list(forward['info']['ch_names'])
    if not set(good_names).issubset(set(fwd_names)):
        raise ValueError('Target good channels absent from donor forward: ' + str(sorted(set(good_names)-set(fwd_names))))
    # Important: the NPY file does not contain its own channel labels.
    # Like the original Localize-MI loader, we rely on the matching channels.tsv row order.
    info = mne.create_info(names, float(sfreqs[0]), ch_types='eeg', verbose=False)
    with info._unlock():
        info['bads'] = list(bad_names)
    ep = mne.EpochsArray(arr, info, tmin=-float(zero[0]), baseline=None, verbose=False)
    # First remove target-bad channels. Donor's historical bads do not govern this target EEG.
    ep.pick(good_names)
    # Assign donor EEG sensor positions, not target digitization or target MRI.
    pos = {x['ch_name']:np.asarray(x['loc'][:3], float) for x in forward['info']['chs'] if x['ch_name'] in good_names}
    montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head')
    ep.set_montage(montage, on_missing='raise', verbose=False)
    donor = copy.deepcopy(forward)
    # Prevent donor historical bad-channel flags from affecting target channel inclusion.
    with donor['info']._unlock():
        donor['info']['bads'] = []
    donor = mne.pick_channels_forward(donor, include=good_names, exclude=[], ordered=True, copy=True, verbose=False)
    assert ep.ch_names == donor['info']['ch_names']
    return ep, donor, {'epochs_path':str(ep_path), 'channels_path':str(ch_path),'donor_forward_path':str(fwd_path), 'run':run, 'units':units,'n_epochs':len(ep),'n_good_channels':len(good_names),'n_bad_channels':len(bad_names),'good_names':good_names,'bad_names':bad_names,'sampling_hz':float(sfreqs[0]), 'channel_order_assumption':'NPY channel axis follows target channels.tsv rows (same as original loader)'}


def solve_sesame(evoked, fwd, a):
    if not hasattr(np, 'in1d'):
        np.in1d = np.isin  # older sesameeg compatibility with NumPy >= 2
    try:
        from sesameeg.mne import prepare_sesame
        import sesameeg.sesame as sesame_module
    except ImportError as exc:
        raise ImportError('SESAMEEG unavailable; check sesameeg and pyvista installations') from exc
    peak = float(np.max(np.abs(evoked.data)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f'Invalid evoked peak {peak}')
    noise_std = float(a.fs) * peak
    np.random.seed(a.seed)
    ses = prepare_sesame(fwd, evoked, n_parts=a.n_parts, noise_std=noise_std,
                         top_min=float(evoked.times[0]), top_max=float(evoked.times[-1]),
                         dip_mom_std=None, hyper_q=True, noise_cov=None, subsample=None,
                         max_n_dips=a.max_n_dips, lam=float(a.lam), verbose=not a.quiet)
    # Retain backward-compatibility shim from the user's previous Script 23.
    original_particle = sesame_module.Particle
    def compatible_particle(n_verts_value, lam_value, dip_mom_std=None, prior_locs=None, fixed_ori=None, hyper_q=None):
        if prior_locs is None: prior_locs = ses.prior_locs
        if fixed_ori is None: fixed_ori = ses.fixed_ori
        if hyper_q is None: hyper_q = ses.hyper_q
        return original_particle(n_verts_value, lam_value, dip_mom_std=dip_mom_std,
                                 prior_locs=prior_locs, fixed_ori=fixed_ori, hyper_q=hyper_q)
    sesame_module.Particle = compatible_particle
    try:
        ses.apply_sesame()
    finally:
        sesame_module.Particle = original_particle
    if not ses.est_n_dips or int(ses.est_n_dips[-1]) < 1:
        raise RuntimeError('SESAME selected a zero-source model')
    stc = ses.compute_stc()
    return ses, stc, noise_std, peak


def main():
    a = arguments()
    if a.target == a.donor:
        raise ValueError('Cross-subject pilot requires target != donor')
    if a.fs <= 0 or a.n_parts < 2 or a.max_n_dips < 1 or a.lam <= 0:
        raise ValueError('Invalid SESAME parameter')
    out = a.output_root / f'{a.target}_with_{a.donor}_{a.run.replace("run-", "run-") if a.run.startswith("run-") else "run-"+a.run}'
    out.mkdir(parents=True, exist_ok=True)
    report={'experiment':'C: target EEG + donor original forward (no target digitization/MRI)',
            'target':a.target, 'donor':a.donor, 'method':'SESAME',
            'parameters':{'fs':a.fs,'n_parts':a.n_parts,'max_n_dips':a.max_n_dips,'lam':a.lam,'hyper_q':True,'seed':a.seed},
            'ground_truth_used':False, 'target_forward_used':False,'target_electrode_tsv_used':False}
    begin=time.perf_counter()
    try:
        ep, fwd, metadata=load_inputs(a)
        report.update(metadata)
        ep.set_eeg_reference(ref_channels='average',projection=True,verbose=False)
        ep.apply_proj(verbose=False)
        evoked=ep.average().crop(-0.002,0.002,include_tmax=True)
        ses, stc, noise_std,peak=solve_sesame(evoked, fwd,a)
        if stc.data.size == 0 or not np.isfinite(stc.data).all():
            raise RuntimeError('SESAME STC is empty or nonfinite')
        idx=np.unravel_index(np.argmax(np.abs(stc.data)),stc.data.shape)
        row=int(idx[0]); t=int(idx[1]); lh_len=len(stc.vertices[0])
        hemi='lh' if row < lh_len else 'rh'
        vert=int(stc.vertices[0][row] if hemi=='lh' else stc.vertices[1][row-lh_len])
        src_idx=0 if hemi=='lh' else 1
        xyz=fwd['src'][src_idx]['rr'][vert].tolist()
        report.update(status='PASS',noise_std_v=noise_std,evoked_peak_abs_v=peak,
                      n_source_estimate_rows=int(stc.data.shape[0]),
                      est_n_dips_final=int(ses.est_n_dips[-1]),
                      peak_hemisphere=hemi,peak_vertex=vert,peak_time_s=float(stc.times[t]),
                      peak_coordinate_donor_src_rr_m=xyz,
                      peak_coordinate_frame='donor forward source-space rr; do not compare directly with target ground truth',
                      source_localization_error_mm=None)
        print(f'PASS: {a.target} EEG, {a.donor} donor model, SESAME')
        print(f'Good channels: {metadata["n_good_channels"]}; bad excluded: {metadata["n_bad_channels"]}')
        print(f'Source peak: {hemi} vertex={vert}, t={float(stc.times[t])*1e3:.3f} ms')
        print('Localization distance: NOT calculated (unregistered donor/target spaces)')
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        print('FAILED:',report['error'],file=sys.stderr)
    finally:
        report['runtime_seconds']=round(time.perf_counter()-begin,3)
        output=out/'sesame_pilot_result.json'
        output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('Report:',output)
    if report['status'] != 'PASS':
        raise SystemExit(1)

if __name__=='__main__':
    main()
