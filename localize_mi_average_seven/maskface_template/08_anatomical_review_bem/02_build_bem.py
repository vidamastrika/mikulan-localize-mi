"""Build and validate BEM and cortical sources."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import shutil

import numpy as np

from review_common import approve, candidate, new_run, verified_diagnostics, write_json
from diagnose import contains, geometry_row, intersections

CONDUCTIVITY = (0.3, 0.006, 0.3)
ICO = 4


def validate_geometry(models, layers):
    if len(models) != 3 or {int(s['id']) for s in models} != set(layers.values()):
        raise ValueError('Expected three distinct BEM compartments')
    by_id = {int(s['id']): s for s in models}
    meshes = {layer: (by_id[ident]['rr'] * 1000, by_id[ident]['tris'])
              for layer, ident in layers.items()}
    report = []
    for layer, (v, f) in meshes.items():
        if v.shape != (2562, 3) or f.shape != (5120, 3) or not np.isfinite(v).all():
            raise ValueError(f'Invalid ico=4 size or coordinates for {layer}')
        if not .05 < np.percentile(np.linalg.norm(v, axis=1), 95) / 1000 < .25:
            raise ValueError(f'Unexpected MRI coordinate scale for {layer}')
        metrics = geometry_row('maskface_baseline_ico4', layer, v, f)
        contacts = intersections(v, f)
        passed = (metrics['sign_topology_eligible'] and metrics['euler'] == 2
                  and metrics['signed_volume_mm3'] > 0 and contacts['intersection_pairs'] == 0)
        report.append(dict(layer=layer, passed=bool(passed), topology=metrics, self_contacts=contacts))
    for inner, outer in combinations(meshes, 2):
        v, f = meshes[inner]
        w, g = meshes[outer]
        contacts = intersections(v, f, w, g)
        eligible = next(r['topology']['sign_topology_eligible'] for r in report if r.get('layer') == outer)
        inside = contains(v, w, g, eligible)
        outside = None if inside is None else int((~inside).sum())
        report.append(dict(inner=inner, outer=outer, outside_vertices=outside,
                           contacts=contacts, passed=outside == 0 and contacts['intersection_pairs'] == 0))
    return meshes, report


def validate_solution(solution, expected, layers, mri_frame):
    matrix = solution['solution']
    n = sum(len(s['rr']) for s in expected)
    if (matrix.shape != (n, n) or not np.isfinite(matrix).all()
            or np.any(np.linalg.norm(matrix, axis=1) == 0)):
        raise ValueError('Invalid BEM solution matrix dimensions, values or zero row')
    actual = {int(s['id']): s for s in solution['surfs']}
    if set(actual) != set(layers.values()) or len(solution['surfs']) != 3:
        raise ValueError('BEM solution has wrong compartments')
    for source, conductivity in zip(('inner_skull', 'outer_skull', 'outer_skin'), CONDUCTIVITY):
        a = actual[layers[source]]
        e = next(s for s in expected if s['id'] == a['id'])
        if (int(a['coord_frame']) != mri_frame or not np.isclose(a['sigma'], conductivity)
                or not np.allclose(a['rr'], e['rr'], rtol=0, atol=1e-8)
                or not np.array_equal(a['tris'], e['tris'])):
            raise ValueError(f'BEM solution surface or conductivity mismatch: {source}')
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('review-root', 'output-root'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    review_mode = parser.add_mutually_exclusive_group(required=True)
    review_mode.add_argument('--approval', type=Path)
    review_mode.add_argument('--exploratory', action='store_true',
                             help='Build with anatomical review pending')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required to build BEM and cortical sources')

    with new_run(args.output_root, '08_BEM_AND_SOURCES') as run:
        run.use(Path(__file__))
        run.doc['exploratory'] = args.exploratory
        review = args.review_root.resolve()
        if args.exploratory:
            subject = verified_diagnostics(run)
            run.verify(review, stage='08_ANATOMICAL_REVIEW')
            pending = json.loads(run.use(review / 'APPROVAL_PENDING.json').read_text())
            for path, checksum in pending['input_sha256'].items():
                run.use(path, checksum)
        else:
            subject = approve(run, review, args.approval)
            run.doc['anatomy_review'] = 'APPROVED'
        import mne
        from mne.io.constants import FIFF

        t1, paths = candidate(subject)
        fs_dir = t1.parent.parent
        for name in ('lh.white', 'rh.white', 'lh.sphere', 'rh.sphere'):
            run.use(fs_dir / 'surf' / name)
        subjects = run.out / 'subjects'
        target = subjects / subject
        (target / 'bem').mkdir(parents=True)
        (target / 'mri').symlink_to(fs_dir / 'mri', target_is_directory=True)
        (target / 'surf').symlink_to(fs_dir / 'surf', target_is_directory=True)
        run.use(fs_dir / 'mri/transforms/talairach.xfm')
        # Preserve the exact watershed vertex coordinates in surface RAS mm.
        for layer, path in paths.items():
            shutil.copyfile(path, target / 'bem' / f'{layer}.surf')

        layers = {'inner_skull': int(FIFF.FIFFV_BEM_SURF_ID_BRAIN),
                  'outer_skull': int(FIFF.FIFFV_BEM_SURF_ID_SKULL),
                  'outer_skin': int(FIFF.FIFFV_BEM_SURF_ID_HEAD)}
        frame = int(FIFF.FIFFV_COORD_MRI)
        run.doc.update(subject=subject, conductivity_s_per_m=CONDUCTIVITY, ico=ICO,
                       source_spacing='ico5', mne_version=mne.__version__)
        models = mne.make_bem_model(subject, ico=ICO, conductivity=CONDUCTIVITY, subjects_dir=subjects)
        meshes, geometry = validate_geometry(models, layers)
        write_json(run.out / '01_downsampled_geometry.json', geometry)
        if not all(r['passed'] for r in geometry):
            raise ValueError('Downsampled BEM geometry failed. See 01_downsampled_geometry.json.')

        sources = mne.setup_source_space(subject, spacing='ico5', subjects_dir=subjects, add_dist=False)
        inner, faces = meshes['inner_skull']
        containment = []
        for hemisphere, src in zip(('lh', 'rh'), sources, strict=True):
            vertices = src['rr'][src['vertno']]
            if int(src['coord_frame']) != frame or not len(vertices) or not np.isfinite(vertices).all():
                raise ValueError(f'Invalid source-space frame or active vertices: {hemisphere}')
            outside = src['vertno'][~contains(vertices * 1000, inner, faces, True)]
            containment.append(dict(hemisphere=hemisphere, active_vertices=len(vertices),
                                    outside_count=len(outside), outside_vertex_ids=outside.tolist()))
        write_json(run.out / '02_source_containment.json', containment)
        if any(row['outside_count'] for row in containment):
            raise ValueError('Active cortical sources extend outside the downsampled inner skull')

        solution = mne.make_bem_solution(models)
        values = validate_solution(solution, models, layers, frame)
        bem, src = run.out / 'maskface-bem-sol.fif', run.out / 'maskface-ico5-src.fif'
        mne.write_bem_solution(bem, solution, overwrite=False)
        readback = validate_solution(mne.read_bem_solution(bem), models, layers, frame)
        if not np.allclose(readback, values, rtol=1e-6, atol=1e-7):
            raise ValueError('BEM FIF matrix round trip failed')
        mne.write_source_spaces(src, sources, overwrite=False)
        for a, b in zip(mne.read_source_spaces(src), sources, strict=True):
            if (int(a['coord_frame']) != frame or not np.array_equal(a['vertno'], b['vertno'])
                    or not np.allclose(a['rr'], b['rr'], atol=1e-8)):
                raise ValueError('Source-space FIF round trip failed')
        write_json(run.out / '03_solution_validation.json',
                   dict(status='PASS', matrix_shape=list(values.shape), fif_roundtrip=True,
                        conductivity_s_per_m=CONDUCTIVITY, compartments=layers, source_containment='PASS'))
        run.doc.update(bem=str(bem), src=str(src),
                       scalp=str(target / 'bem/outer_skin.surf'), subjects_dir=str(subjects))


if __name__ == '__main__':
    main()
