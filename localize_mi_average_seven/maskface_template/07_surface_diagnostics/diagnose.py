#!/usr/bin/env python3
"""Prepare watershed variants and review skull boundaries in their own MRI frames."""
import argparse
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
import json
from pathlib import Path
import shlex
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from input_utils import MRI, apply, native_bridge
from stage2_common import OLD, OLD_FS, ROOT, VAR, Run, add_fs_arguments, fs_command, fs_runtime
from stage2_qa import contains, fs_bridge, geometry_row, intersections, overlay_figure, read_surface, topology

BASE = VAR / 'outputs'
OUTPUT = BASE / '07_surface_diagnostics'
CONSTRUCTION = BASE / '03_template/run_05'
FREESURFER = BASE / '05_freesurfer/run_07'
QA = BASE / '06_surface_QA/run_10'
LAYERS = ('brain', 'inner_skull', 'outer_skull', 'outer_skin')
COLORS = dict(zip(LAYERS, ('lime', 'red', 'yellow', 'cyan')))
SETTINGS = {'baseline_replay': [], 't1': ['-T1'], 'atlas': ['-atlas']}
REGIONS = ('cranial_vault', 'facial', 'inferior_head', 'cranial_transition')


def verify_upstream(run):
    run.verify(CONSTRUCTION, stage='03_CONSTRUCT')
    fs = run.verify(FREESURFER, stage='05_FREESURFER_AND_WATERSHED')
    qa = run.verify(QA, stage='06_SURFACE_AND_ANATOMICAL_QA')
    for root in (CONSTRUCTION, FREESURFER):
        if str(root / 'manifest.json') not in qa['input_sha256']:
            raise ValueError(f'Surface QA did not record {root}')
    run.use(Path(overlay_figure.__code__.co_filename))
    run.use(Path(topology.__code__.co_filename))
    if fs.get('watershed_status') != 'COMPLETE_RETURN_ZERO':
        raise ValueError('Watershed baseline has no successful completion marker')
    return fs['subject']


def candidate(args, run, subject):
    image, license_path = fs_runtime(args, run)
    t1 = run.use(FREESURFER / 'subjects' / subject / 'mri/T1.mgz')
    command = fs_command(
        args.runtime, run.out,
        ['mri_watershed', '-useSRAS', *SETTINGS[args.setting], '-surf',
         str(run.out / subject), str(t1), str(run.out / 'watershed.mgz')],
        args.docker_command, image, license_path,
    )
    run.doc.update(setting=args.setting, watershed_command=command, t1=str(t1),
                   surface_frame='FreeSurfer surface RAS (tkregister)')
    (run.out / 'command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + shlex.join(command) + '\n')
    run.save()
    print(shlex.join(command), flush=True)
    with (run.out / 'watershed.log').open('w') as log:
        subprocess.run(command, cwd=run.out, stdout=log, stderr=subprocess.STDOUT, check=True)
    for layer in LAYERS:
        read_surface(run.out / f'{subject}_{layer}_surface')
    if not (run.out / 'watershed.mgz').is_file():
        raise FileNotFoundError('Expected watershed.mgz')
    if 'use surfaceRAS to save surface vertex positions' not in (run.out / 'watershed.log').read_text():
        raise ValueError('Watershed log did not confirm the requested surface RAS convention')


def surfaces_for(run, paths, t1, native=False):
    meshes = {}
    image = nib.load(t1)
    bridge = None if native else fs_bridge(t1)
    for layer, path in paths.items():
        vertices, faces, metadata, _ = read_surface(run.use(path))
        if native:
            bridge, scanner, shape = native_bridge(metadata)
            if tuple(shape) != image.shape or not np.allclose(scanner, image.affine, atol=1e-3):
                raise ValueError(f'Native volume geometry mismatch: {path}')
        meshes[layer] = (apply(bridge, vertices), faces)
    return meshes


def screen(model, meshes):
    rows = []
    topology = {layer: geometry_row(model, layer, *mesh) for layer, mesh in meshes.items()}
    for layer, row in topology.items():
        rows.append(dict(row, check='topology'))
    for inner, outer in combinations(meshes, 2):
        v, f = meshes[inner]
        w, g = meshes[outer]
        rows.append(dict(model=model, check='pair_intersection', inner=inner, outer=outer,
                         **screen_result(intersections, v, f, w, g)))
        if not topology[outer]['sign_topology_eligible']:
            result = {'status': 'WITHHELD_OUTER_TOPOLOGY_INELIGIBLE'}
        else:
            result = screen_result(vertex_containment, v, w, g)
        rows.append(dict(model=model, check='nesting', inner=inner, outer=outer, **result))
    for layer, mesh in meshes.items():
        rows.append(dict(model=model, layer=layer, check='self_intersection',
                         **screen_result(intersections, *mesh)))
    return rows


def vertex_containment(v, w, g):
    inside = contains(v, w, g, eligible=True)
    return dict(vertices_tested=len(inside), inside_fraction=float(inside.mean()))


def screen_result(function, *args):
    try:
        return dict(status='SCREEN_COMPLETED', **function(*args))
    except Exception as exc:
        return dict(status='UNRESOLVED', reason=f'{type(exc).__name__}: {exc}')


def regions(points, anchor):
    x, y, z = (points - anchor).T
    region = np.full(len(points), 'cranial_transition', dtype=object)
    region[z >= 15] = 'cranial_vault'
    region[(z < 15) & (y >= 25)] = 'facial'
    region[(z < -25) & (y < 25)] = 'inferior_head'
    sector = np.where(x < -20, 'left', np.where(x > 20, 'right', 'midline'))
    return region, sector


def gap_table(model, meshes, anchor):
    tables = []
    for inner, outer in [('inner_skull', 'outer_skull'), ('outer_skull', 'outer_skin')]:
        for source, target in [(inner, outer), (outer, inner)]:
            points = meshes[source][0]
            targets = meshes[target][0]
            gap, index = cKDTree(targets).query(points)
            region, sector = regions(points, anchor)
            target_region, _ = regions(targets[index], anchor)
            rows = dict(model=model, source=source, target=target, vertex=np.arange(len(points)),
                        target_vertex=index, region=region, sector=sector, target_region=target_region,
                        gap_mm=gap, top_5pct=gap >= np.quantile(gap, .95))
            for i, axis in enumerate('xyz'):
                rows[f'{axis}_mm'] = points[:, i]
                rows[f'target_{axis}_mm'] = targets[index, i]
            tables.append(pd.DataFrame(rows))
    return pd.concat(tables, ignore_index=True)


def intensity_cues(model, meshes, anchor, first, second, comparison):
    rows = []
    for layer, (points, _) in meshes.items():
        a, av = first.sample(points)
        b, bv = second.sample(points)
        ok = av & bv & np.isfinite(a) & np.isfinite(b)
        a, b = a / first.vmax, b / second.vmax
        region, _ = regions(points, anchor)
        for label in REGIONS:
            selected = ok & (region == label)
            count = int(selected.sum())
            rows.append(dict(model=model, comparison=comparison, layer=layer, region=label,
                             vertices=int((region == label).sum()), paired_finite=count,
                             first_median=np.median(a[selected]) if count else np.nan,
                             second_median=np.median(b[selected]) if count else np.nan,
                             median_change=np.median((b-a)[selected]) if count else np.nan,
                             fraction_second_signal_first_low=np.mean(
                                 (b[selected] > .05) & (a[selected] <= .05)) if count else np.nan))
    return rows


def assess(args, run, subject):
    config = json.loads(run.use(VAR / 'config/variant.json').read_text())
    dataset = ROOT / config['dataset']
    baseline_t1 = run.use(FREESURFER / 'subjects' / subject / 'mri/T1.mgz')
    models = [('maskface_baseline', baseline_t1, {
        layer: FREESURFER / 'watershed_surfaces' / f'{subject}_{layer}_surface' for layer in LAYERS
    }, False)]
    failures = []
    for root in args.candidate_root:
        root = root.resolve()
        manifest_path = run.use(root / 'manifest.json')
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] != 'COMPLETE':
            failures.append(dict(candidate=str(root), status=manifest['status'],
                                 reason=manifest.get('error', 'No completion marker')))
            continue
        manifest = run.verify(root, stage='07_WATERSHED_CANDIDATE')
        if manifest['t1'] != str(baseline_t1):
            raise ValueError(f'Candidate used another T1: {root}')
        models.append((f"{manifest['setting']}_{root.name}", baseline_t1, {
            layer: root / f'{subject}_{layer}_surface' for layer in LAYERS
        }, False))
    for model, root in [('pydeface_average_seven', OLD_FS), ('fsaverage', args.fsaverage)]:
        models.append((model, root / 'mri/T1.mgz',
                       {layer: root / 'bem' / f'{layer}.surf' for layer in LAYERS[1:]}, False))
    for participant in config['participants']:
        models.append((participant, dataset / config['original_T1w_pattern'].format(subject=participant),
                       {layer: dataset / 'derivatives/sourcemodelling' / participant / 'anat'
                        / f'{participant}_{layer}.surf.gii' for layer in LAYERS[1:]}, True))

    old_group = MRI(run.use(OLD / 'group_template0.nii.gz'), np.eye(4))
    new_group = MRI(run.use(CONSTRUCTION / 'group_template0.nii.gz'), np.eye(4))
    if old_group.image.shape != new_group.image.shape or not np.allclose(
            old_group.image.affine, new_group.image.affine, atol=1e-4):
        raise ValueError('Group intensity cue requires the same scanner grid')
    screens, gaps, cues, anchors = [], [], [], {}
    figures = run.out / 'figures'
    figures.mkdir()
    for model, t1, paths, native in models:
        print(f'Reviewing {model}', flush=True)
        t1 = run.use(t1)
        meshes = surfaces_for(run, paths, t1, native)
        anchor = anchors['maskface_baseline'] if t1 == baseline_t1 and anchors else np.median(
            meshes['inner_skull'][0], axis=0)
        anchors[model] = anchor
        table = gap_table(model, meshes, anchor)
        gaps.append(table)
        screens.extend(screen(model, meshes))
        mri = MRI(t1, np.eye(4))
        entries = {model: [(layer, *mesh, COLORS[layer]) for layer, mesh in meshes.items()]}
        if t1 == baseline_t1:
            cues.extend(intensity_cues(model, meshes, anchor, old_group, new_group,
                                       'pydeface_vs_maskface_group_template'))
        if native:
            maskface = MRI(run.use(dataset / config['maskface_pattern'].format(subject=model)), np.eye(4))
            if mri.image.shape != maskface.image.shape or not np.allclose(
                    mri.image.affine, maskface.image.affine, atol=1e-4):
                raise ValueError(f'Native intensity pairing grid mismatch: {model}')
            cues.extend(intensity_cues(model, meshes, anchor, mri, maskface, 'same_participant_release'))
        worst = table[(table.source == 'outer_skull') & (table.target == 'inner_skull')]
        hotspot = worst.loc[worst.gap_mm.idxmax(), ['x_mm', 'y_mm', 'z_mm']].to_numpy(float)
        for label, focus in [('vault', anchor + [0, 0, 35]), ('facial', anchor + [0, 45, -15]),
                             ('inferior', anchor + [0, 0, -35]), ('outer_skull_hotspot', hotspot)]:
            overlay_figure([(model, mri)], entries, figures / f'{model}_{label}.png',
                           f'{model} | {label} | boundaries on own MRI in scanner RAS', 200, focus)
        if model == 'maskface_baseline':
            for label, focus in [('facial', anchor + [0, 45, -15]), ('hotspot', hotspot)]:
                labels = ('Pydeface group template', 'Maskface group template')
                overlay_figure(list(zip(labels, (old_group, new_group))),
                               {name: entries[model] for name in labels},
                               figures / f'group_signal_{label}.png',
                               'Same scanner grid | maskface contours on both volumes | intensity cues',
                               200, focus)

    all_gaps = pd.concat(gaps, ignore_index=True)
    by_region = []
    for keys, group in all_gaps.groupby(['model', 'source', 'target', 'region'], sort=False):
        by_region.append(dict(zip(('model', 'source', 'target', 'region'), keys),
                              n=len(group), median_mm=group.gap_mm.median(),
                              p95_mm=group.gap_mm.quantile(.95), max_mm=group.gap_mm.max(),
                              global_tail_count=int(group.top_5pct.sum())))
    all_gaps.to_csv(run.out / '01_vertex_gaps.csv.gz', index=False)
    pd.DataFrame(by_region).to_csv(run.out / '02_regional_candidate_comparison.csv', index=False)
    all_gaps[all_gaps.top_5pct].groupby(
        ['model', 'source', 'target', 'region', 'sector', 'target_region']
    ).size().rename('tail_vertices').to_csv(run.out / '03_gap_tail_locations.csv')
    (run.out / '04_geometry_screens.json').write_text(json.dumps(screens, indent=2, default=json_scalar))
    pd.DataFrame(cues).to_csv(run.out / '05_maskface_intensity_cues.csv', index=False)
    pd.DataFrame(failures, columns=['candidate', 'status', 'reason']).to_csv(
        run.out / '06_candidate_failures.csv', index=False)
    run.doc.update(region_anchors_scanner_ras_mm={k: v.tolist() for k, v in anchors.items()},
                   assessed_models=list(anchors), anatomical_review='PENDING', candidate_selected=False)
    write_report(run.out, all_gaps, screens, failures)


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_report(out, gaps, screens, failures):
    lines = ['# Watershed surface diagnostics', '',
             'The surfaces from the baseline and each finished setting were screened in their own MRI geometry. '
             'Visual anatomical review is pending. No candidate was selected.', '',
             'The watershed log reports surface RAS. The overlays use the T1 header to convert the '
             'vertices to scanner RAS, correcting the frame assumption in the Stage 2 report.', '',
             '## Largest baseline skull gaps', '']
    data = gaps[(gaps.model == 'maskface_baseline') & (gaps.source == 'outer_skull') &
                (gaps.target == 'inner_skull')]
    for region, group in data.groupby('region'):
        lines.append(f'- {region}: median {group.gap_mm.median():.1f} mm, '
                     f'p95 {group.gap_mm.quantile(.95):.1f} mm, '
                     f'{group.top_5pct.sum()} vertices in the model-wide upper tail.')
    flagged = sum(row.get('status', '').startswith(('UNRESOLVED', 'WITHHELD')) or
                  row.get('intersection_pairs', 0) > 0 or row.get('inside_fraction', 1) < 1 or
                  not row.get('sign_topology_eligible', True) for row in screens)
    lines += ['', '## Reading the outputs', '',
              f'{flagged} geometry screen rows had a flag or unresolved result. '
              f'{len(failures)} supplied candidate runs lacked a completion marker.', '',
              'Inspect the three orthogonal planes for each focused overlay, including the outer-skull hotspot. '
              'Trace the calvarium, orbital and facial outlines, and inferior boundary. '
              'Compare the baseline and settings in the same MRI. Use the independent reference heads '
              'to contextualize the regional distributions.', '',
              'Nearest-vertex gaps are mesh and sampling dependent; they do not measure true skull thickness. '
              'Outer-to-inner and inner-to-outer distances answer different questions. Review tail counts '
              'with the regional denominators and retain left/right and target-region information.', '',
              'Paired scanner-grid intensity changes are cues for maskface-associated signal, with independent '
              'MRI intensity normalization. They do not identify synthetic voxels. The group contrast also reflects '
              'independent registrations, averaging and preprocessing. Use the native pairs as supporting evidence, '
              'and assess whether visible boundary excursions coincide with the signal changes.', '',
              'A small gap or numerically closed mesh cannot establish tissue correspondence. '
              'Record anatomical findings and unresolved regions before any BEM decision.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    generate = commands.add_parser('candidate')
    generate.add_argument('--setting', choices=SETTINGS, required=True)
    add_fs_arguments(generate)
    review = commands.add_parser('review')
    review.add_argument('--fsaverage', type=Path,
                        default=Path.home() / 'mne_data/MNE-fsaverage-data/fsaverage')
    review.add_argument('--candidate-root', action='append', type=Path, default=[])
    for command in (generate, review):
        command.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    if OUTPUT.resolve() not in args.output_root.resolve().parents:
        parser.error(f'Choose a fresh output directory under {OUTPUT}')
    stage = '07_WATERSHED_CANDIDATE' if args.action == 'candidate' else '07_SURFACE_DIAGNOSTIC'
    with Run(args.output_root, stage) as run:
        subject = verify_upstream(run)
        packages = {}
        for package in ('nibabel', 'numpy', 'scipy', 'pandas', 'matplotlib', 'vtk'):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        run.doc.update(python=sys.version, package_versions=packages)
        if args.action == 'candidate':
            candidate(args, run, subject)
        else:
            assess(args, run, subject)


if __name__ == '__main__':
    main()
