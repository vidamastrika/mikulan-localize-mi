#!/usr/bin/env python3
"""Compare uncorrected maskface watershed surfaces with MRI and baseline."""
import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from input_utils import MRI, image_header
from stage2_common import (EXP, OLD, OLD_FS, VAR, Run, write_json)
from stage2_qa import fs_bridge, geometry_row, overlay_figure, read_surface
from stage2_qa import contains, intersections, distance_stats


LAYERS = ('outer_skin', 'outer_skull', 'inner_skull', 'brain')
COLORS = {'outer_skin': 'cyan', 'outer_skull': 'yellow',
          'inner_skull': 'red', 'brain': 'lime'}


def mesh_row_gap(inner, outer, label):
    from scipy.spatial import cKDTree
    iv, _ = inner
    ov, _ = outer
    distances = cKDTree(ov).query(iv, k=1)[0]
    return dict(pair=label, **distance_stats(distances),
                interpretation='nearest-vertex gap proxy; not true thickness and not a nesting proof')


def mgh_header(path):
    image = nib.load(path)
    try:
        return image_header(image)
    except AttributeError:
        affine = np.asarray(image.affine, float)
        corners = np.array([[x, y, z] for x in [-.5, image.shape[0]-.5]
                            for y in [-.5, image.shape[1]-.5]
                            for z in [-.5, image.shape[2]-.5]])
        ras = corners @ affine[:3, :3].T + affine[:3, 3]
        return dict(shape=list(image.shape[:3]), voxel_spacing_mm=list(image.header.get_zooms()[:3]),
                    axis_codes=list(nib.aff2axcodes(affine)), units='mm',
                    affine_ras_mm=affine.tolist(), scanner_ras_min_mm=ras.min(0).tolist(),
                    scanner_ras_max_mm=ras.max(0).tolist(), qform_sform_agree='not_available_for_MGH')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freesurfer-root', type=Path,
                        default=VAR/'outputs/05_freesurfer/run_04')
    parser.add_argument('--construction-root', type=Path,
                        default=VAR/'outputs/03_template/run_04')
    parser.add_argument('--output-root', type=Path,
                        default=VAR/'outputs/06_surface_QA/run_04')
    parser.add_argument('--dpi', type=int, default=240)
    args = parser.parse_args()
    if args.dpi < 200:
        parser.error('Use at least 200 DPI for review figures')

    with Run(args.output_root, '06_SURFACE_AND_ANATOMICAL_QA') as run:
        fs_manifest = run.verify(args.freesurfer_root, stage='05_FREESURFER_AND_WATERSHED')
        run.verify(args.construction_root, stage='03_CONSTRUCT')
        if fs_manifest.get('reconstruction_status') != 'COMPLETE_RETURN_ZERO':
            raise ValueError('FreeSurfer completion marker is absent')
        if fs_manifest.get('watershed_status') != 'COMPLETE_RETURN_ZERO':
            raise ValueError('Watershed completion marker is absent')
        (run.out/'figures').mkdir()

        subject = fs_manifest['subject']
        surf_dir = args.freesurfer_root/'watershed_surfaces'
        old_t1_path = run.use(OLD_FS/'mri/T1.mgz')
        old_bridge = fs_bridge(old_t1_path)
        surfaces = {}
        old_surfaces = {}
        surface_rows = []
        for layer in LAYERS:
            path = run.use(surf_dir/f'{subject}_{layer}_surface')
            v, f, meta, messages = read_surface(path)
            surfaces[layer] = (v, f, meta)
            surface_rows.append(dict(geometry_row('maskface_watershed', layer, v, f),
                                     path=str(path), warnings=' | '.join(messages),
                                     coordinate_assumption='mri_watershed -useSRAS output interpreted in scanner RAS; verify against T1.mgz before anatomical claims'))
            old_path = run.use(OLD_FS/'bem'/f'{layer}.surf')
            old_v, old_f, old_meta, old_messages = read_surface(old_path)
            from input_utils import apply
            old_v_scanner = apply(old_bridge, old_v)
            old_surfaces[layer] = (old_v_scanner, old_f, old_meta)
            surface_rows.append(dict(geometry_row('original_pydeface', layer, old_v_scanner, old_f),
                                     path=str(old_path), warnings=' | '.join(old_messages),
                                     coordinate_assumption='original FreeSurfer surface RAS mapped through its own T1.mgz tkregister-to-scanner-RAS bridge'))
        pd.DataFrame(surface_rows).to_csv(run.out/'01_surface_geometry.csv', index=False)

        # Use scanner-RAS physical coordinates for the -useSRAS surfaces.
        fs_t1_path = run.use(args.freesurfer_root/'subjects'/subject/'mri/T1.mgz')
        fs_t1 = MRI(fs_t1_path, np.eye(4))
        maskface_label = 'maskface FreeSurfer T1.mgz'
        mesh_entries = {maskface_label: [
            (layer, surfaces[layer][0], surfaces[layer][1], COLORS[layer]) for layer in LAYERS]}
        focuses = {'facial': (0, 55, -15), 'left_lateral': (-55, 0, 25),
                   'cranial': (0, 0, 60), 'inferior': (0, 0, -25)}
        for region, focus in focuses.items():
            overlay_figure([(maskface_label, fs_t1)], mesh_entries,
                           run.out/'figures'/f'maskface_{region}.png',
                           'Uncorrected watershed boundaries over the FreeSurfer T1; scanner-RAS assumption requires visual confirmation',
                           args.dpi, focus)

        # Original surfaces are in FreeSurfer surface RAS; bridge them only to
        # their own original T1 scanner-RAS, never to participant native MRI.
        old_entries = []
        old_meshes = {}
        for layer in LAYERS:
            v, f, _ = old_surfaces[layer]
            old_entries.append((f'original {layer}', v, f, COLORS[layer]))
        old_mri = MRI(old_t1_path, np.eye(4))
        old_meshes['original Pydeface FreeSurfer T1'] = old_entries
        for region, focus in focuses.items():
            overlay_figure([('original Pydeface FreeSurfer T1', old_mri)], old_meshes,
                           run.out/'figures'/f'original_{region}.png',
                           'Original template FreeSurfer surfaces over original T1; independent from maskface reconstruction',
                           args.dpi, focus)

        # Pre-FreeSurfer template MRI and header-level comparison. These common
        # voxel-grid cues do not demonstrate biological correspondence.
        new_template_path = run.use(args.construction_root/'group_template0.nii.gz')
        original_template_path = run.use(OLD/'group_template0.nii.gz')
        old_grid_bridge = fs_bridge(old_t1_path)
        original_template = MRI(original_template_path, old_grid_bridge)
        maskface_template = MRI(new_template_path, old_grid_bridge)
        pd.DataFrame([
            dict(image='original_pydeface_template', **image_header(nib.load(original_template_path))),
            dict(image='new_maskface_template', **image_header(nib.load(new_template_path))),
            dict(image='maskface_FreeSurfer_T1', **mgh_header(fs_t1_path))]).to_csv(
                run.out/'02_MRI_headers.csv', index=False)
        paired_rows = []
        for label, a, b in [('template', original_template, maskface_template)]:
            paired_rows.append(dict(comparison=label,
                                    same_shape=a.data.shape == b.data.shape,
                                    same_affine=bool(np.allclose(a.image.affine, b.image.affine, atol=1e-4)),
                                    note='same-grid signal comparison only; no new intertemplate registration'))
        pd.DataFrame(paired_rows).to_csv(run.out/'03_template_grid_comparison.csv', index=False)

        # Boundary geometry screens. Numerical contact checks are recorded only
        # when VTK exists; no missing dependency is converted to a pass.
        topology_rows = []
        for layer in LAYERS:
            v, f, _ = surfaces[layer]
            row = geometry_row('maskface_watershed', layer, v, f)
            try:
                row.update(self_intersection_screen=intersections(v, f),
                           intersection_status='SCREEN_COMPLETED_NOT_CERTIFICATION')
            except Exception as exc:
                row.update(self_intersection_screen=None,
                           intersection_status=f'FAILED_OR_UNAVAILABLE:{type(exc).__name__}:{exc}')
            topology_rows.append(row)
        pd.DataFrame(topology_rows).to_json(run.out/'04_topology_intersection_screens.json',
                                           orient='records', indent=2)

        gap_rows = []
        pairs = [('brain', 'inner_skull'), ('inner_skull', 'outer_skull'),
                 ('outer_skull', 'outer_skin')]
        for inner_name, outer_name in pairs:
            inner = surfaces[inner_name][:2]
            outer = surfaces[outer_name][:2]
            gap_rows.append(mesh_row_gap(inner, outer, f'{inner_name}_to_{outer_name}'))
            v, _ = surfaces[inner_name][:2]
            row = dict(pair=f'{inner_name}_inside_{outer_name}', tested_surface=inner_name)
            eligible = bool(geometry_row('maskface_watershed', outer_name, outer[0], outer[1])['sign_topology_eligible'])
            if not eligible:
                row.update(vertices_tested=len(v), vertices_inside_fraction=None,
                           status='WITHHELD_OUTER_TOPOLOGY_INELIGIBLE')
            else:
                try:
                    inside = contains(v, *outer, eligible=True)
                    row.update(vertices_tested=len(inside), vertices_inside_fraction=float(inside.mean()),
                               status='VERTEX_CONTAINMENT_SCREEN_ONLY')
                except Exception as exc:
                    row.update(vertices_tested=len(v), vertices_inside_fraction=None,
                               status=f'FAILED_OR_UNAVAILABLE:{type(exc).__name__}:{exc}')
            gap_rows.append(row)
        pd.DataFrame(gap_rows).to_csv(run.out/'05_layer_nesting_and_gap_screens.csv', index=False)

        # Intensity-boundary profiles at mesh vertices compare each candidate
        # with its own T1, not with unregistered native anatomy.
        from scipy.ndimage import map_coordinates
        profile_rows = []
        inv_affine = np.linalg.inv(fs_t1.image.affine)
        for layer, (vertices, _, _) in surfaces.items():
            sample_idx = np.linspace(0, len(vertices)-1, min(1200, len(vertices))).astype(int)
            points = vertices[sample_idx]
            vox = (inv_affine[:3, :3] @ points.T + inv_affine[:3, 3:4]).T
            values = map_coordinates(fs_t1.data, vox.T, order=1, mode='constant', cval=np.nan,
                                     prefilter=False)
            profile_rows.append(dict(layer=layer, sample_vertices=len(values),
                                     samples_outside_image=int((~np.isfinite(values)).sum()),
                                     median_MRI_intensity=float(np.nanmedian(values)),
                                     p10_MRI_intensity=float(np.nanpercentile(values, 10)),
                                     p90_MRI_intensity=float(np.nanpercentile(values, 90)),
                                     caveat='boundary intensity summary, not tissue classification'))
        pd.DataFrame(profile_rows).to_csv(run.out/'06_surface_MRI_boundary_cues.csv', index=False)

        # Preserve warning/failure lines and provide a review-only report.
        warning_rows = []
        for log in [args.freesurfer_root/'recon-all.log', args.freesurfer_root/'watershed.log']:
            for number, line in enumerate(log.read_text(errors='replace').splitlines(), 1):
                if any(term in line.lower() for term in ('warn', 'error', 'fail', 'cannot', 'nan', 'inf')):
                    warning_rows.append(dict(log=log.name, line=number, text=line.strip()))
        pd.DataFrame(warning_rows, columns=['log', 'line', 'text']).to_csv(run.out/'07_warnings.csv', index=False)
        run.doc.update(surface_count=len(surfaces), surface_method='unmodified watershed outputs',
                       anatomical_review='REQUIRED', BEM_solution=False, source_space=False,
                       EEG_alignment=False, forward=False, inverse=False,
                       frame_note='maskface watershed interpreted scanner-RAS from -useSRAS; original FreeSurfer meshes bridged to their own scanner-RAS; native participant references are not overlaid onto group space')
        (run.out/'REPORT_stage2.md').write_text(
            '# Stage 2 anatomical QA\n\n'
            'This report contains geometry screens and MRI overlays for review. `-useSRAS` output coordinates are interpreted as scanner RAS; confirm that the contours follow the matching FreeSurfer T1 in every view before interpreting comparisons. Original and new participant/template anatomies are not claimed to be voxelwise registered.\n\n'
            'Review facial, left-lateral, cranial and inferior overlays; surface topology/intersection screens; conditional nesting results; nearest-vertex layer gaps; boundary MRI intensities; and both processing logs. Contact tests and vertex containment are numerical screens, not anatomical certification. A completed recon-all is not evidence of valid tissue boundaries. Missing or unavailable checks remain unresolved, never a pass.\n\n'
            'No surface was moved, repaired, adopted as a BEM, or used in forward/inverse calculations. No source space, EEG alignment or BEM solution was created. Stage 2 stops at anatomical review.\n')


if __name__ == '__main__':
    main()
