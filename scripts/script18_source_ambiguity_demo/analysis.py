#!/usr/bin/env python3
"""Script 18 companion: separated cortical dipoles with similar forward EEG.

No computation occurs on import. Outputs are created only by run_demo()/main().
This is a forward simulation, independent of the earlier recorded-data fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ORANGE = '#E87524'
BLUE = '#2879B9'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(dataset, subject, run):
    """Reuse the released forward solution, run channel mask and native anatomy."""
    model = dataset / 'derivatives/sourcemodelling' / subject
    forward_path = model / 'fwd' / f'{subject}_fwd.fif'
    channels_path = dataset / 'derivatives/epochs' / subject / 'eeg' / f'{subject}_task-seegstim_{run}_channels.tsv'
    transform_path = model / 'xfm' / f'{subject}_from-head_to-surface.h5'
    surfaces = [model / 'anat' / f'{subject}_hemi-{h}_pial.surf.gii' for h in ['L', 'R']]
    inputs = [forward_path, channels_path, transform_path, *surfaces]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing demonstration inputs:\n' + '\n'.join(missing))
    forward = mne.read_forward_solution(forward_path, verbose=False)
    if forward['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
        raise ValueError('Expected the validated released forward in head coordinates')
    if len(forward['src']) != 2 or any(s['type'] != 'surf' for s in forward['src']):
        raise ValueError('Expected two cortical surface source spaces')
    channels = pd.read_csv(channels_path, sep='\t')
    if channels.name.duplicated().any():
        raise ValueError('Duplicate channel names')
    if channels.name.tolist() != forward['sol']['row_names']:
        raise ValueError('Channel table and forward order differ; baseline expects identical order')
    good_names = channels.loc[channels.type.str.upper().eq('EEG') &
                             channels.status.str.lower().ne('bad'), 'name'].tolist()
    forward_bads = set(forward['info'].get('bads', []))
    good_names = [name for name in good_names if name not in forward_bads]
    if len(good_names) < 4:
        raise ValueError('At least four usable EEG electrodes required')
    selected = mne.pick_channels_forward(forward, include=good_names, ordered=True,
                                         copy=True, verbose=False)
    selected = mne.convert_forward_solution(selected, surf_ori=True, force_fixed=True,
                                             use_cps=True, copy=True, verbose=False)
    rr = np.asarray(selected['source_rr'], float)
    gain = np.asarray(selected['sol']['data'], float).copy()
    if gain.shape != (len(good_names), len(rr)):
        raise ValueError('Fixed-orientation gain shape is inconsistent with source positions')
    # R @ G, R = I - 11^T/N: same projection as the baseline average reference.
    gain -= gain.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(gain, axis=0)
    finite = np.isfinite(gain).all(axis=0) & np.isfinite(rr).all(axis=1) & np.isfinite(norms)
    scale = norms[finite].max() if finite.any() else 0.
    valid = np.flatnonzero(finite & (norms > max(scale * 1e-12, np.finfo(float).tiny)))
    if len(valid) < 2:
        raise ValueError('Too few valid, nondegenerate average-referenced lead-field columns')
    by_name = {c['ch_name']: c for c in selected['info']['chs']}
    if any(by_name[n]['coord_frame'] != FIFF.FIFFV_COORD_HEAD for n in good_names):
        raise ValueError('Electrodes must use the same head coordinate frame')
    positions = np.array([by_name[n]['loc'][:3] for n in good_names])
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError('Invalid electrode coordinates')
    info = mne.create_info(good_names, 1., 'eeg')
    info.set_montage(mne.channels.make_dig_montage(
        ch_pos=dict(zip(good_names, positions)), coord_frame='head'))
    with h5py.File(transform_path, 'r') as stream:
        transform = np.asarray(stream['trans'][()], float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)):
        raise ValueError('Head-to-surface transform must be finite and rigid')
    geometry = []
    for path in surfaces:
        image = nib.load(path)
        vertices = np.asarray(image.darrays[0].data, float)
        faces = np.asarray(image.darrays[1].data, int)
        if not np.isfinite(vertices).all():
            raise ValueError(f'Invalid surface coordinates: {path}')
        # Same released-surface unit handling as Script 18's load_surface().
        if np.max(np.abs(vertices)) > 1:
            vertices = vertices / 1000.
        geometry.append((vertices, faces))
    return dict(forward=selected, gain=gain, norms=norms, valid=valid, rr=rr,
                info=info, names=good_names, positions=positions, transform=transform,
                geometry=geometry, inputs=inputs)


def search_pairs(model, max_sources, seed, thresholds, top_k):
    """Uniform seeded source subsample, all unordered pairs, deterministic ranking."""
    if max_sources < 2 or top_k < 1 or seed < 0:
        raise ValueError('max_sources >= 2, top_k >= 1 and seed >= 0 required')
    valid = model['valid']
    candidates = (np.sort(np.random.default_rng(seed).choice(valid, max_sources, replace=False))
                  if len(valid) > max_sources else valid.copy())
    unit = model['gain'][:, candidates] / model['norms'][candidates]
    rr = model['rr'][candidates]
    best = {t: [] for t in thresholds}
    eligible_counts = {t: 0 for t in thresholds}
    # Blocks bound memory even when --max-sources requests the full source space.
    for first in range(0, len(candidates), 128):
        stop = min(first + 128, len(candidates))
        corr = np.clip(np.abs(unit[:, first:stop].T @ unit), 0., 1.)
        distance = np.linalg.norm(rr[first:stop, None] - rr[None], axis=2) * 1000
        upper = np.arange(first, stop)[:, None] < np.arange(len(candidates))[None]
        for threshold in thresholds:
            eligible = np.flatnonzero(upper & (distance >= threshold))
            eligible_counts[threshold] += len(eligible)
            if not len(eligible):
                continue
            # Stable ordering handles ties reproducibly (source indices ascending).
            order = np.argsort(-corr.ravel()[eligible], kind='stable')[:top_k]
            for flat in eligible[order]:
                local_a, local_b = np.unravel_index(flat, corr.shape)
                best[threshold].append((float(corr[local_a, local_b]),
                                        int(candidates[first + local_a]), int(candidates[local_b])))
            best[threshold] = sorted(best[threshold], key=lambda p: (-p[0], p[1], p[2]))[:top_k]
    return candidates, best, eligible_counts


def describe_pair(model, a, b):
    """A = 1 µV RMS; B signed least-squares amplitude; no channelwise fitting."""
    gain = model['gain']
    current_a = 1e-6 / np.sqrt(np.mean(gain[:, a] ** 2))
    current_b = current_a * np.dot(gain[:, b], gain[:, a]) / np.dot(gain[:, b], gain[:, b])
    raw_r = float(np.dot(gain[:, a], gain[:, b]) / (model['norms'][a] * model['norms'][b]))
    signal_a = gain[:, a] * current_a * 1e6
    signal_b = gain[:, b] * current_b * 1e6
    residual = float(np.linalg.norm(signal_a - signal_b) / np.linalg.norm(signal_a))
    row = dict(source_a_index=a, source_b_index=b,
               distance_mm=float(np.linalg.norm(model['rr'][a] - model['rr'][b]) * 1000),
               signed_pearson_r=raw_r, absolute_pearson_r=abs(raw_r),
               relative_residual=residual, polarity_reversed=bool(current_b < 0),
               source_a_current_nAm=current_a * 1e9, source_b_current_nAm=current_b * 1e9,
               source_a_rms_uv=float(np.sqrt(np.mean(signal_a ** 2))),
               source_b_aligned_rms_uv=float(np.sqrt(np.mean(signal_b ** 2))),
               coordinate_frame='head', coordinate_units='mm')
    src = model['forward']['src']
    vertices = np.concatenate([s['vertno'] for s in src])
    for label, index in [('a', a), ('b', b)]:
        row[f'source_{label}_hemisphere'] = 'lh' if index < src[0]['nuse'] else 'rh'
        row[f'source_{label}_vertex'] = int(vertices[index])
        for axis, value in zip('xyz', model['rr'][index] * 1000):
            row[f'source_{label}_{axis}_mm'] = float(value)
        normal = model['forward']['source_nn'][index]
        for axis, value in zip('xyz', normal):
            row[f'source_{label}_normal_{axis}_head'] = float(value)
    return row, signal_a, signal_b


def make_figure(model, row, signal_a, signal_b, subject, run, dpi, path):
    """One 16:9 white figure, three panels; B contains two matched topographies."""
    with plt.rc_context({'font.size': 16, 'axes.facecolor': 'white', 'figure.facecolor': 'white'}):
        fig = plt.figure(figsize=(20, 11.25))
        grid = fig.add_gridspec(1, 3, width_ratios=[1., 1.8, 1.65], wspace=.32,
                               left=.025, right=.98, top=.77, bottom=.30)
        ax = fig.add_subplot(grid[0], projection='3d')
        for vertices, faces in model['geometry']:
            ax.plot_trisurf(*(vertices * 1000).T, triangles=faces, color='#bcc3cb',
                            alpha=.10, linewidth=0, shade=False, rasterized=True)
        indices = [row['source_a_index'], row['source_b_index']]
        points = mne.transforms.apply_trans(model['transform'], model['rr'][indices]) * 1000
        for i, (label, color) in enumerate([('A', ORANGE), ('B', BLUE)]):
            ax.scatter(*points[i], color=color, s=140, depthshade=False)
            ax.text(*points[i], f' {label}', color=color, fontsize=23, weight='bold')
        ax.plot(*points.T, '--', color='#555555', linewidth=1.5)
        ax.view_init(elev=80, azim=90)
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.set_axis_off()
        ax.set_title(f'A   Different brain sources\n{row["distance_mm"]:.1f} mm apart', fontsize=20, pad=22)
        inner = grid[1].subgridspec(1, 2, wspace=.14)
        limit = max(np.max(np.abs(signal_a)), np.max(np.abs(signal_b)))
        for i, (values, label, color) in enumerate([(signal_a, 'A', ORANGE), (signal_b, 'B', BLUE)]):
            ax = fig.add_subplot(inner[i])
            image, _ = mne.viz.plot_topomap(values, model['info'], axes=ax, show=False,
                                            cmap='RdBu_r', vlim=(-limit, limit), contours=5,
                                            sphere=(0., 0., 0., .095))
            ax.set_title(f'Source {label}', fontsize=20, color=color, pad=18)
        fig.text(.465, .81, 'B   Predicted scalp EEG', fontsize=20, ha='center', weight='bold')
        cax = fig.add_axes([.32, .25, .29, .022])
        cb = fig.colorbar(image, cax=cax, orientation='horizontal')
        cb.set_label('Potential (µV); shared colour scale', fontsize=16)
        cb.ax.tick_params(labelsize=13)
        ax = fig.add_subplot(grid[2])
        x = np.arange(len(signal_a))
        ax.plot(x, signal_a, color=ORANGE, linewidth=2.3, label='Source A')
        ax.plot(x, signal_b, color=BLUE, linestyle='--', linewidth=2., label='Source B (aligned)')
        ticks = np.linspace(0, len(x) - 1, 5).astype(int)
        ax.set_xticks(ticks, [model['names'][i] for i in ticks], fontsize=13)
        ax.set_xlabel('EEG channels (baseline order)', fontsize=16)
        ax.set_ylabel('Predicted potential (µV)', fontsize=16)
        ax.set_title(f'C   Scalp potential comparison\n|r| = {row["absolute_pearson_r"]:.3f} · residual = {row["relative_residual"]:.1%}',
                     fontsize=19, pad=22)
        ax.legend(frameon=False, fontsize=15, loc='best')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Different brain sources can produce similar scalp EEG patterns',
                     fontsize=27, weight='bold', y=.95)
        fig.text(.5, .89, f'{subject} · {len(signal_a)} usable electrodes from {run} · subject-specific forward model',
                 ha='center', fontsize=19)
        polarity = 'reversed' if row['polarity_reversed'] else 'unchanged'
        fig.text(.5, .15, f'Fixed cortical-normal dipoles · average reference · A normalized to 1 µV RMS\n'
                 f'B: one optimal signed amplitude fitted to A; polarity {polarity}',
                 ha='center', fontsize=17, linespacing=1.6)
        fig.text(.5, .055, 'Forward simulations, not recorded responses or an inverse-algorithm comparison.\n'
                 'Similarity after amplitude alignment illustrates spatial ambiguity; it does not establish exact non-uniqueness.',
                 ha='center', fontsize=15, linespacing=1.5)
        fig.savefig(path, dpi=dpi, facecolor='white')
        plt.close(fig)


def run_demo(dataset, subject, run, output_dir, *, dpi=400, max_sources=2048, seed=2026,
             thresholds=(20., 40., 60.), top_k=10):
    """Execute only on explicit request; never overwrite an existing output folder."""
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose a new output directory')
    thresholds = sorted(set(float(t) for t in thresholds))
    if not thresholds or not all(np.isfinite(t) and t > 0 for t in thresholds) or dpi <= 0:
        raise ValueError('Positive finite separation thresholds and positive dpi required')
    model = load_model(dataset, subject, run)
    sampled, best, counts = search_pairs(model, max_sources, seed, thresholds, top_k)
    rows = []
    for threshold, pairs in best.items():
        for rank, (_, a, b) in enumerate(pairs, start=1):
            row, _, _ = describe_pair(model, a, b)
            row.update(minimum_separation_mm=threshold, rank_within_threshold=rank,
                       eligible_pair_count=counts[threshold])
            rows.append(row)
    if not rows:
        raise ValueError('No source pair satisfies any requested threshold; no results written')
    # Predeclared presentation rule: 40 mm if available; otherwise nearest available threshold.
    available = [t for t in thresholds if best[t]]
    chosen_threshold = min(available, key=lambda t: (abs(t - 40), t))
    chosen = next(r for r in rows if r['minimum_separation_mm'] == chosen_threshold and r['rank_within_threshold'] == 1)
    _, signal_a, signal_b = describe_pair(model, chosen['source_a_index'], chosen['source_b_index'])
    table = pd.DataFrame(rows)
    table['shown_in_figure'] = (table.minimum_separation_mm.eq(chosen_threshold) & table.rank_within_threshold.eq(1))
    metadata = dict(subject=subject, run=run, seed=seed, max_sources=max_sources,
                    valid_source_count=len(model['valid']), total_source_count=len(model['rr']),
                    excluded_source_count=len(model['rr']) - len(model['valid']),
                    sampled_source_indices=sampled.tolist(), channel_names=model['names'],
                    source_coordinates='head frame, metres internally; CSV uses millimetres',
                    electrode_coordinates='head frame from forward Info, not released electrode TSV',
                    electrode_head_coordinates_m=model['positions'].tolist(),
                    source_orientation='MNE fixed cortical normal, surf_ori=True, force_fixed=True, use_cps=True',
                    nondegenerate_cutoff='norm > max(finite maximum norm * 1e-12, float tiny)',
                    sampling='NumPy default_rng(seed), uniform without replacement, sorted indices; all valid sites when below limit',
                    ranking='Absolute Pearson correlation descending, then source A and B forward indices ascending',
                    alignment='A positive current scaled to 1 µV RMS; B signed least-squares fit to A',
                    selection='Best at 40 mm if available, otherwise threshold nearest 40 mm (ties choose smaller)',
                    thresholds_mm=thresholds, eligible_pair_counts=counts, top_k=top_k,
                    figure_pair=chosen, numpy_version=np.__version__, mne_version=mne.__version__,
                    input_sha256={str(p): sha256(p) for p in model['inputs']},
                    script_sha256=sha256(Path(__file__)))
    # Exclusive creation also protects against another process creating the same directory.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'figures').mkdir()
    (output / 'numerical').mkdir()
    table.to_csv(output / 'numerical/source_ambiguity_pairs.csv', index=False)
    pd.DataFrame({'channel': model['names'], 'source_a_uv': signal_a, 'source_b_aligned_uv': signal_b,
                  'difference_uv': signal_a - signal_b}).to_csv(output / 'numerical/source_ambiguity_channel_patterns.csv', index=False)
    (output / 'numerical/source_ambiguity_manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    make_figure(model, chosen, signal_a, signal_b, subject, run, dpi,
                output / 'figures/source_ambiguity_demo.png')
    summary = [f'# Forward-model spatial ambiguity: {subject} / {run}', '',
               '## Modelling and selection', '',
               'Reused the released subject-specific forward solution, its two cortical source spaces, '
               'its participant-specific head-coordinate electrode positions, the run EEG good-channel mask '
               'and released head-to-surface transform/pial surfaces. No BEM, forward model or inverse solution was recomputed.', '',
               f'{len(model["names"])} EEG channels; average reference applied to G by subtracting each column mean. '
               'Fixed cortical normals use MNE surf_ori=True, force_fixed=True, use_cps=True. '
               'No whitening, noise simulation, recorded-response fitting or waveform synthesis.', '',
               f'{len(sampled)} uniformly sampled valid sources of {len(model["valid"])}; '
               f'NumPy default_rng seed {seed}. Finite source coordinates and finite nondegenerate referenced columns required. '
               'Every unordered sampled pair was evaluated; ranking is absolute Pearson r, then ascending source indices. '
               f'Top {top_k} pairs per threshold are saved. Sampling indices and input SHA-256 hashes are in the manifest.', '',
               '## Results', '', '| Minimum distance | Eligible pairs | Best distance | Absolute r | Relative residual | Polarity reversed |',
               '|---|---:|---:|---:|---:|---|']
    for threshold in thresholds:
        selected = [r for r in rows if r['minimum_separation_mm'] == threshold and r['rank_within_threshold'] == 1]
        if selected:
            r = selected[0]
            summary.append(f'| {threshold:g} mm | {counts[threshold]} | {r["distance_mm"]:.2f} mm | '
                           f'{r["absolute_pearson_r"]:.6f} | {r["relative_residual"]:.2%} | {r["polarity_reversed"]} |')
        else:
            summary.append(f'| {threshold:g} mm | 0 | No eligible pair | — | — | — |')
    summary += ['', 'Figure pair uses the predeclared 40 mm threshold when available, otherwise the available threshold nearest 40 mm. '
                'No selection by visual appearance or assumed similarity cutoff was used.', '',
                'A is scaled to 1 µV sensor RMS. B uses one signed least-squares amplitude; its polarity reversal is explicitly recorded. '
                'Relative residual = ||y_A − alpha y_B||₂ / ||y_A||₂. Absolute Pearson correlation describes spatial shape, '
                'not equal potentials for equal physical source current. Amplitudes in nAm and aligned RMS values are saved. '
                'With centered maps and unconstrained scalar fitting, residual = sqrt(1 − r²), up to numerical precision.', '',
                '## Interpretation and limitations', '',
                'This reports the best observed match, even if its similarity is modest. Similar forward patterns at separated sites '
                'illustrate spatial ambiguity under this model; they do not establish exact non-uniqueness, indistinguishability '
                'at a measured noise level, simultaneous active sources, or failure of an inverse algorithm. '
                'Subsampling may miss stronger pairs; cortical-normal orientation, channel selection, head conductivity and '
                'electrode coverage influence the result. No measured EEG was used. Source distances are Euclidean, not cortical geodesic distances.', '',
                'The forward file embeds the computed lead field and source spaces. Standalone BEM solutions are not needed. '
                'Repository modelling code specifies a three-shell BEM (0.3, 0.006, 0.3 S/m) and oct6 sources; '
                'those historical parameters were not independently re-estimated or verified from a released standalone BEM.', '']
    (output / 'source_ambiguity_summary.md').write_text('\n'.join(summary))
    print(f'Created {output}\nBest displayed pair: {chosen["distance_mm"]:.2f} mm; '
          f'|r|={chosen["absolute_pearson_r"]:.6f}; residual={chosen["relative_residual"]:.2%}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT / 'data/Localize-MI')
    parser.add_argument('--subject', default='sub-01')
    parser.add_argument('--run', default='run-01')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-sources', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--separations-mm', type=float, nargs='+', default=[20., 40., 60.])
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--dpi', type=int, default=400)
    args = parser.parse_args()
    output = args.output_dir or PROJECT / 'outputs/18_presentation_figures/source_ambiguity_demo' / f'{args.subject}_{args.run}'
    run_demo(args.dataset, args.subject, args.run, output, dpi=args.dpi,
             max_sources=args.max_sources, seed=args.seed, thresholds=args.separations_mm, top_k=args.top_k)


if __name__ == '__main__':
    main()
