"""Forward-model spatial ambiguity illustration; no inverse reconstruction."""
import hashlib
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def render(args, load_surfaces, load_transform):
    if args.illposed_recorded:
        return render_recorded(args, load_surfaces, load_transform)
    dataset = args.dataset.expanduser().resolve()
    root = args.output_root.expanduser().resolve() / f'{args.subject}_{args.run}'
    output = root / 'figures' / '06_inverse_spatial_ambiguity.png'
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite')
    minimum = args.illposed_min_separation_mm
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('Minimum source separation must be positive and finite')
    forward_path = dataset / 'derivatives/sourcemodelling' / args.subject / 'fwd' / f'{args.subject}_fwd.fif'
    channels_path = dataset / 'derivatives/epochs' / args.subject / 'eeg' / f'{args.subject}_task-seegstim_{args.run}_channels.tsv'
    forward = mne.read_forward_solution(forward_path, verbose=False)
    if any(src['type'] != 'surf' for src in forward['src']):
        raise ValueError('This illustration requires a cortical surface forward model')
    forward = mne.convert_forward_solution(forward, surf_ori=True, force_fixed=True,
                                          use_cps=True, copy=True, verbose=False)
    channels = pd.read_csv(channels_path, sep='\t')
    good = set(channels.loc[channels.status.str.lower().ne('bad'), 'name'])
    picks = [i for i, name in enumerate(forward['sol']['row_names']) if name in good]
    if len(picks) < 4:
        raise ValueError('At least four usable EEG electrodes are needed')
    gain = forward['sol']['data'][picks].astype(float)
    gain -= gain.mean(axis=0, keepdims=True)  # EEG average reference.
    norms = np.linalg.norm(gain, axis=0)
    valid = np.flatnonzero(np.isfinite(gain).all(axis=0) & (norms > 0))
    if len(valid) < 2:
        raise ValueError('Too few finite nonzero lead-field columns')
    rr = forward['source_rr']
    unit = gain[:, valid] / norms[valid]
    best = (-1., None, None)
    print(f'Searching {len(valid)} cortical sources, separation >= {minimum:g} mm...', flush=True)
    # Exhaustive deterministic search, blocking to avoid an N-source squared allocation.
    for first in range(0, len(valid), 128):
        stop = min(first + 128, len(valid))
        correlation = np.abs(unit[:, first:stop].T @ unit)
        distance = np.linalg.norm(rr[valid[first:stop], None] - rr[valid][None], axis=2)
        eligible = (distance >= minimum / 1000.) & (np.arange(first, stop)[:, None] < np.arange(len(valid))[None])
        correlation[~eligible] = -1
        flat = np.argmax(correlation)
        a, b = np.unravel_index(flat, correlation.shape)
        score = float(correlation[a, b])
        if score > best[0]:
            best = (score, int(valid[first + a]), int(valid[b]))
    _, a, b = best
    if a is None:
        raise ValueError('No source pair meets the requested separation')
    # A has 1 microvolt sensor RMS; B's signed amplitude is a least-squares fit.
    amplitude_a = 1e-6 / np.sqrt(np.mean(gain[:, a] ** 2))
    amplitude_b = amplitude_a * np.dot(gain[:, b], gain[:, a]) / np.dot(gain[:, b], gain[:, b])
    signal_a = gain[:, a] * amplitude_a * 1e6
    signal_b = gain[:, b] * amplitude_b * 1e6
    correlation = float(np.corrcoef(signal_a, signal_b)[0, 1])
    relative_error = float(np.linalg.norm(signal_a - signal_b) / np.linalg.norm(signal_a))
    separation = float(np.linalg.norm(rr[a] - rr[b]) * 1000)
    names = [forward['sol']['row_names'][i] for i in picks]
    by_name = {c['ch_name']: c['loc'][:3] for c in forward['info']['chs']}
    info = mne.create_info(names, 1., 'eeg')
    info.set_montage(mne.channels.make_dig_montage(ch_pos={n: by_name[n] for n in names}, coord_frame='head'))
    transform = load_transform(dataset, args.subject)
    points = mne.transforms.apply_trans(transform, rr[[a, b]]) * 1000
    surfaces = load_surfaces(dataset, args.subject)
    colors = ['#3666aa', '#e47832']
    fig = plt.figure(figsize=(22, 7.5))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1.1], wspace=.35)
    ax = fig.add_subplot(grid[0], projection='3d')
    for hemisphere in ['lh', 'rh']:
        vertices, triangles = surfaces[hemisphere]
        vertices = vertices * 1000
        ax.plot_trisurf(*vertices.T, triangles=triangles, color='#b8bec6', alpha=.12,
                        linewidth=0, shade=False, rasterized=True)
    for i, label in enumerate(['A', 'B']):
        ax.scatter(*points[i], color=colors[i], s=110, depthshade=False)
        ax.text(*points[i], f'  {label}', fontsize=17, color=colors[i], weight='bold')
    ax.plot(*points.T, color='#444444', linestyle='--', linewidth=1.5)
    ax.view_init(elev=80, azim=90)
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    ax.set_axis_off()
    ax.set_title(f'Different source locations\n{separation:.1f} mm apart', fontsize=17, pad=16)
    limit = max(np.abs(signal_a).max(), np.abs(signal_b).max())
    for i, (label, values) in enumerate([('A', signal_a), ('B', signal_b)], start=1):
        ax = fig.add_subplot(grid[i])
        im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap='RdBu_r',
                                    vlim=(-limit, limit), contours=6, sensors=True,
                                    sphere=(0., 0., 0., .095))
        ax.set_title(f'Source {label}: predicted EEG', fontsize=17, pad=20)
    colorbar_ax = fig.add_axes([.35, .23, .30, .025])
    bar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
    bar.set_label('Average-referenced voltage (µV); same scale', fontsize=13)
    bar.ax.tick_params(labelsize=12)
    ax = fig.add_subplot(grid[3])
    ax.scatter(signal_a, signal_b, s=24, color='#4b657a', alpha=.65)
    ax.plot([-limit, limit], [-limit, limit], '--', color='#888888', linewidth=1)
    ax.set(xlabel='Source A: electrode voltage (µV)', ylabel='Source B: electrode voltage (µV)')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.set_title(f'Similar across {len(picks)} electrodes\nr = {correlation:.3f} | relative difference = {relative_error:.1%}', fontsize=15, pad=20)
    ax.grid(alpha=.2)
    fig.suptitle('Different cortical sources can produce similar scalp EEG', fontsize=23, weight='bold', y=.98)
    fig.text(.5, .10, f'{args.subject} {args.run} head model and usable electrodes · Simulated EEG, not recorded responses',
             ha='center', fontsize=14)
    fig.text(.5, .045, f'Best matching pair among {len(valid):,} cortical sites ≥ {minimum:g} mm apart; cortical-normal dipoles with fitted relative amplitude.\n'
             'Illustrates spatial ambiguity under this forward model; the sources are hypothetical and the match is approximate.',
             ha='center', fontsize=12)
    fig.subplots_adjust(top=.79, bottom=.33, left=.02, right=.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    (root / 'csv').mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'channel': names, 'source_a_uv': signal_a, 'source_b_uv': signal_b,
                  'difference_uv': signal_a - signal_b}).to_csv(root / 'csv/06_ambiguity_sensor_voltages.csv', index=False)
    vertices = np.concatenate([s['vertno'] for s in forward['src']])
    pd.DataFrame({'source': ['A', 'B'], 'forward_source_index': [a, b],
                  'hemisphere': ['lh' if i < forward['src'][0]['nuse'] else 'rh' for i in [a, b]],
                  'vertex': vertices[[a, b]], 'surface_x_mm': points[:, 0],
                  'surface_y_mm': points[:, 1], 'surface_z_mm': points[:, 2],
                  'signed_amplitude_nAm': np.array([amplitude_a, amplitude_b]) * 1e9
                  }).to_csv(root / 'csv/06_ambiguity_sources.csv', index=False)
    metadata = dict(subject=args.subject, run=args.run, source_count=len(valid), electrode_count=len(picks),
                    minimum_separation_mm=minimum, separation_mm=separation, correlation=correlation,
                    relative_l2_difference=relative_error, reference='average of usable electrodes',
                    source_orientation='fixed cortical normal, MNE use_cps=True',
                    selection='Exhaustive maximum absolute lead-field correlation subject to minimum Euclidean separation',
                    amplitude='A scaled to 1 µV sensor RMS; signed B amplitude fitted by least squares',
                    interpretation='Hypothetical single-dipole forward simulations; approximate similarity, not exact nonuniqueness or an empirical fit',
                    inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [forward_path, channels_path]})
    (root / 'csv/06_ambiguity_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{separation:.1f} mm apart; r={correlation:.6f}; relative difference={relative_error:.2%}\n{output}')


def render_recorded(args, load_surfaces, load_transform):
    """Fit two separated cortical-normal dipoles to one measured EEG topography."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from localize_mi.io import load_run

    dataset = args.dataset.expanduser().resolve()
    root = args.output_root.expanduser().resolve() / f'{args.subject}_{args.run}'
    output = root / 'figures/06_inverse_spatial_ambiguity_recorded.png'
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite')
    minimum = args.illposed_min_separation_mm
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('Minimum source separation must be positive and finite')
    loaded = load_run(dataset, subject=args.subject, run=args.run)
    evoked = loaded.epochs.copy().set_eeg_reference('average', projection=False, verbose=False).average()
    evoked.pick('eeg', exclude='bads')
    times = evoked.times
    window = np.flatnonzero((times >= -.002 - 1e-10) & (times <= .002 + 1e-10))
    if not len(window):
        raise ValueError('Recording does not cover the -2 to +2 ms target window')
    time_index = window[np.argmax(np.std(evoked.data[:, window], axis=0))]
    observed = evoked.data[:, time_index]
    forward = mne.pick_channels_forward(loaded.forward, include=evoked.ch_names,
                                       ordered=True, copy=True, verbose=False)
    if any(src['type'] != 'surf' for src in forward['src']):
        raise ValueError('This illustration requires a cortical surface forward model')
    forward = mne.convert_forward_solution(forward, surf_ori=True, force_fixed=True,
                                          use_cps=True, copy=True, verbose=False)
    gain = forward['sol']['data'].astype(float)
    gain -= gain.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(gain, axis=0)
    valid = np.isfinite(gain).all(axis=0) & (norms > 0)
    if np.linalg.norm(observed) == 0:
        raise ValueError('Measured target topography is zero')
    scores = np.full(gain.shape[1], -np.inf)
    scores[valid] = np.abs(gain[:, valid].T @ observed) / (norms[valid] * np.linalg.norm(observed))
    a = int(np.argmax(scores))
    distances = np.linalg.norm(forward['source_rr'] - forward['source_rr'][a], axis=1) * 1000
    alternatives = scores.copy()
    alternatives[distances < minimum] = -np.inf
    b = int(np.argmax(alternatives))
    if not np.isfinite(alternatives[b]):
        raise ValueError('No valid alternative meets the minimum separation')
    indices = [a, b]
    amplitude = np.array([np.dot(gain[:, i], observed) / np.dot(gain[:, i], gain[:, i]) for i in indices])
    predicted = gain[:, indices] * amplitude[None, :]
    residual = observed[:, None] - predicted
    relative_error = np.linalg.norm(residual, axis=0) / np.linalg.norm(observed)
    correlations = [float(np.corrcoef(observed, predicted[:, i])[0, 1]) for i in range(2)]
    explained = 1 - relative_error ** 2
    time_ms = float(times[time_index] * 1000)
    points = mne.transforms.apply_trans(load_transform(dataset, args.subject), forward['source_rr'][indices]) * 1000
    surfaces = load_surfaces(dataset, args.subject)
    uv = observed * 1e6
    predicted_uv = predicted * 1e6
    colors = ['#3666aa', '#e47832']
    fig = plt.figure(figsize=(23, 12))
    grid = fig.add_gridspec(2, 4, height_ratios=[.8, 1.2], hspace=.55, wspace=.4)
    ax = fig.add_subplot(grid[0, :3])
    shown = (times >= -.005 - 1e-10) & (times <= .005 + 1e-10)
    ax.plot(times[shown] * 1000, evoked.data[:, shown].T * 1e6,
            color='#58778b', alpha=.18, linewidth=.65)
    ax.axvspan(-2, 2, color='#ededed', zorder=-1)
    ax.axvline(time_ms, color='#b64c45', linewidth=2, label=f'Selected snapshot: {time_ms:g} ms')
    ax.set_title(f'Recorded trial-average EEG · {len(loaded.epochs)} trials · {len(evoked.ch_names)} usable electrodes', fontsize=17)
    ax.set_xlabel('Time relative to stimulation (ms)', fontsize=14)
    ax.set_ylabel('Voltage (µV)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc='best', frameon=False)
    ax.grid(alpha=.2)
    ax = fig.add_subplot(grid[0, 3])
    for i, label in enumerate(['A', 'B']):
        ax.scatter(uv, predicted_uv[:, i], color=colors[i], s=12, alpha=.5, label=f'Fit {label}')
    bound = max(np.abs(uv).max(), np.abs(predicted_uv).max())
    ax.plot([-bound, bound], [-bound, bound], '--', color='#888888', linewidth=1)
    ax.set_title('Each point is one electrode', fontsize=15)
    ax.set_xlabel('Recorded (µV)', fontsize=13)
    ax.set_ylabel('Predicted (µV)', fontsize=13)
    ax.legend(fontsize=12, frameon=False)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.2)
    ax = fig.add_subplot(grid[1, 0], projection='3d')
    for hemisphere in ['lh', 'rh']:
        vertices, triangles = surfaces[hemisphere]
        ax.plot_trisurf(*(vertices * 1000).T, triangles=triangles, color='#b8bec6',
                        alpha=.12, linewidth=0, shade=False, rasterized=True)
    for i, label in enumerate(['A', 'B']):
        ax.scatter(*points[i], color=colors[i], s=110, depthshade=False)
        ax.text(*points[i], f' {label}', fontsize=17, color=colors[i], weight='bold')
    ax.plot(*points.T, color='#444444', linestyle='--', linewidth=1.5)
    ax.view_init(elev=80, azim=90)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()
    ax.set_title(f'Two candidate source locations\n{distances[b]:.1f} mm apart', fontsize=16, pad=18)
    limit = max(np.abs(uv).max(), np.abs(predicted_uv).max())
    titles = [f'Measured EEG at {time_ms:g} ms',
              f'Fit A: best cortical dipole\nr = {correlations[0]:.3f} · error = {relative_error[0]:.1%}',
              f'Fit B: separated alternative\nr = {correlations[1]:.3f} · error = {relative_error[1]:.1%}']
    for col, (values, title) in enumerate(zip([uv, predicted_uv[:, 0], predicted_uv[:, 1]], titles), start=1):
        ax = fig.add_subplot(grid[1, col])
        im, _ = mne.viz.plot_topomap(values, evoked.info, axes=ax, show=False,
                                    cmap='RdBu_r', vlim=(-limit, limit), contours=6,
                                    sphere=(0., 0., 0., .095))
        ax.set_title(title, fontsize=15, pad=20)
    colorbar_ax = fig.add_axes([.37, .145, .40, .018])
    bar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
    bar.set_label('Average-referenced voltage (µV); shared scale', fontsize=13)
    bar.ax.tick_params(labelsize=11)
    fig.suptitle(f'Two source explanations of the same recorded EEG · {args.subject} {args.run}',
                 fontsize=23, weight='bold', y=.98)
    fig.text(.5, .065, 'Trial-average stimulation EEG; snapshot chosen by maximum scalp standard deviation in the −2 to +2 ms target window.',
             ha='center', fontsize=12)
    fig.text(.5, .035, 'Independent, unweighted single-dipole fits with fixed cortical-normal orientations; B is the best fit ≥ '
             f'{minimum:g} mm from A. Error = residual L2 norm / measured L2 norm.\n'
             'Both predictions are fitted to the same recording; the alternative fit is weaker. This is a descriptive illustration, not evidence of two active sources.',
             ha='center', fontsize=11)
    fig.subplots_adjust(left=.055, right=.98, bottom=.22, top=.89)
    output.parent.mkdir(parents=True, exist_ok=True)
    (root / 'csv').mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'channel': evoked.ch_names, 'recorded_uv': uv,
                  'predicted_a_uv': predicted_uv[:, 0], 'predicted_b_uv': predicted_uv[:, 1],
                  'residual_a_uv': residual[:, 0] * 1e6, 'residual_b_uv': residual[:, 1] * 1e6
                  }).to_csv(root / 'csv/06_recorded_ambiguity_sensor_voltages.csv', index=False)
    vertices = np.concatenate([src['vertno'] for src in forward['src']])
    pd.DataFrame({'source': ['A', 'B'], 'forward_source_index': indices,
                  'hemisphere': ['lh' if i < forward['src'][0]['nuse'] else 'rh' for i in indices],
                  'vertex': vertices[indices], 'surface_x_mm': points[:, 0],
                  'surface_y_mm': points[:, 1], 'surface_z_mm': points[:, 2],
                  'signed_amplitude_nAm': amplitude * 1e9,
                  'correlation_with_recording': correlations,
                  'relative_l2_error': relative_error, 'explained_sensor_energy': explained
                  }).to_csv(root / 'csv/06_recorded_ambiguity_sources.csv', index=False)
    pd.DataFrame({'time_ms': times[shown] * 1000, **{
        name: evoked.data[i, shown] * 1e6 for i, name in enumerate(evoked.ch_names)
    }}).to_csv(root / 'csv/06_recorded_ambiguity_waveforms.csv', index=False)
    eeg = dataset / 'derivatives/epochs' / args.subject / 'eeg'
    basename = f'{args.subject}_task-seegstim_{args.run}'
    inputs = [eeg / f'{basename}_{suffix}' for suffix in ['epochs.npy', 'epochs.tsv', 'epochs.json', 'channels.tsv']]
    inputs += [dataset / 'derivatives/sourcemodelling' / args.subject / 'fwd' / f'{args.subject}_fwd.fif']
    metadata = dict(subject=args.subject, run=args.run, trial_count=len(loaded.epochs),
                    electrode_count=len(evoked.ch_names), time_ms=time_ms, target_window_ms=[-2, 2],
                    time_selection='Maximum scalp standard deviation of trial-average EEG within target window',
                    reference='Average of usable electrodes; released baseline correction retained',
                    source_orientation='Fixed cortical normal; MNE use_cps=True', source_count=gain.shape[1],
                    source_separation_mm=float(distances[b]), minimum_separation_mm=minimum,
                    correlation_with_recording=correlations, relative_l2_error=relative_error.tolist(),
                    explained_sensor_energy=explained.tolist(),
                    selection='A: best exhaustive single-dipole unweighted least-squares fit. B: best such fit at least minimum separation from A.',
                    interpretation='Descriptive fits to real trial-average stimulation EEG; no claim of equal fits, neural origin, or two simultaneously active sources',
                    input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (root / 'csv/06_recorded_ambiguity_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Recorded EEG at {time_ms:g} ms; A r={correlations[0]:.4f}, B r={correlations[1]:.4f}; '
          f'{distances[b]:.1f} mm apart\n{output}')
