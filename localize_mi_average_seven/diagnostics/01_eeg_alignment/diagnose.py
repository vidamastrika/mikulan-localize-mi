#!/usr/bin/env python3
"""Read-only released-data audit. No fitting, scaling, anatomy writes or inverse."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mne
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'localize_mi_average_seven'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def apply(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def rigid(matrix):
    matrix = np.asarray(matrix, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError('Expected proper rigid homogeneous transform')
    return matrix


def triangle_distance(points, vertices, faces):
    """Exact unsigned point-to-triangle minimum; all faces, bounded memory.

    Project onto triangle plane when inside, otherwise onto its three segments.
    Degenerate faces contribute their edges. Returned closest points are in mm.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    n2 = np.einsum('ij,ij->i', normal, normal)
    valid = n2 > 1e-20
    closest, distances = [], []
    for point in points:
        projection = point - normal * (np.einsum('ij,ij->i', point-a, normal)
                                        / np.where(valid, n2, 1))[:, None]
        ap = projection-a
        d00 = np.einsum('ij,ij->i', ab, ab)
        d01 = np.einsum('ij,ij->i', ab, ac)
        d11 = np.einsum('ij,ij->i', ac, ac)
        d20 = np.einsum('ij,ij->i', ap, ab)
        d21 = np.einsum('ij,ij->i', ap, ac)
        denom = np.where(valid, n2, 1)
        u = (d11*d20-d01*d21)/denom
        v = (d00*d21-d01*d20)/denom
        inside = valid & (u >= -1e-12) & (v >= -1e-12) & (u+v <= 1+1e-12)
        best_d = np.where(inside, np.sum((projection-point)**2, axis=1), np.inf)
        best_p = projection.copy()
        for start, end in [(a, b), (b, c), (c, a)]:
            edge = end-start
            e2 = np.einsum('ij,ij->i', edge, edge)
            t = np.clip(np.einsum('ij,ij->i', point-start, edge)/np.where(e2 > 0, e2, 1), 0, 1)
            q = start+t[:, None]*edge
            d = np.sum((q-point)**2, axis=1)
            mask = d < best_d
            best_p[mask], best_d[mask] = q[mask], d[mask]
        index = np.argmin(best_d)
        closest.append(best_p[index]); distances.append(np.sqrt(best_d[index]))
    return np.asarray(distances), np.asarray(closest)


def mesh(path):
    if path.name.endswith('.gii'):
        image = nib.load(path)
        p = next(d for d in image.darrays if d.intent == 1008)
        f = next(d for d in image.darrays if d.intent == 1009)
        vertices, faces = np.asarray(p.data, float), np.asarray(f.data, int)
        metadata = dict(p.meta)
        metadata['coordsys'] = str(p.coordsys)
    else:
        vertices, faces, meta = mne.read_surface(path, read_metadata=True)
        metadata = {k: np.asarray(v).tolist() for k, v in meta.items()}
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f'Invalid surface: {path}')
    # Declared pipeline contract: FreeSurfer surface coordinates in mm.
    # Do not silently infer or repair units from magnitude.
    if not 40 < np.max(np.ptp(vertices, axis=0)) < 500:
        raise ValueError(f'Surface inconsistent with expected mm contract: {path}')
    return vertices, faces, metadata


def stats(d):
    return dict(n=len(d), median_mm=float(np.median(d)), p95_mm=float(np.percentile(d, 95)),
                mean_mm=float(np.mean(d)), max_mm=float(np.max(d)))


def regions(points, vertices):
    """Fixed native surface-RAS geometric bins, not cortical/anatomical labels."""
    center = np.median(vertices, axis=0)
    labels = []
    for p in points-center:
        if p[2] > np.percentile(vertices[:, 2]-center[2], 70):
            labels.append('superior_proxy')
        elif abs(p[0]) > abs(p[1]):
            labels.append('right_proxy' if p[0] > 0 else 'left_proxy')
        else:
            labels.append('anterior_proxy' if p[1] > 0 else 'posterior_proxy')
    return np.array(labels)


def figure(subject, panels, out):
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    for col, (model, vertices, faces, points, nearest, distance, names) in enumerate(panels):
        for row, (elev, azim, view) in enumerate([(25, -65, 'oblique'), (90, -90, 'superior'), (0, 0, 'right')]):
            ax = fig.add_subplot(3, 3, row*3+col+1, projection='3d')
            # Rendering alone is subsampled; both distance metrics use full meshes.
            poly = Poly3DCollection(vertices[faces[::max(1, len(faces)//7000)]], alpha=.09, linewidths=0)
            poly.set_facecolor('gray'); ax.add_collection3d(poly)
            ax.scatter(*points.T, c=distance, cmap='viridis', norm=Normalize(0, 40), s=12, depthshade=False)
            for i in np.argsort(distance)[-10:]:
                ax.plot(*np.vstack([points[i], nearest[i]]).T, color='crimson', linewidth=.7)
                ax.text(*points[i], names[i], fontsize=5)
            center = (vertices.max(0)+vertices.min(0))/2
            for setter, value in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
                setter(value-140, value+140)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
            ax.set(xlabel='surface RAS X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
            ax.set_title(f'{model}: {view}\ntriangle median {np.median(distance):.1f}, p95 {np.percentile(distance,95):.1f} mm', fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 40), cmap='viridis'), ax=fig.axes,
                 shrink=.5, label='Unsigned point-to-triangle distance (mm); saturated above 40')
    fig.suptitle(f'{subject}: saved transforms only; top ten residuals labelled; convention review pending')
    fig.savefig(out, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/Localize-MI')
    parser.add_argument('--alignment-root', type=Path, default=EXP/'outputs/04x_exploratory_alignment_retry1')
    parser.add_argument('--average-seven', type=Path, default=EXP/'work/bem_candidates/local_inner_1mm/validation_subjects/average_seven')
    parser.add_argument('--fsaverage', type=Path, default=Path.home()/'mne_data/MNE-fsaverage-data/fsaverage')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent/'run_01')
    args = parser.parse_args()
    cfgpath = EXP/'config/experiment.json'
    cfg = json.loads(cfgpath.read_text())
    limits = cfg['coregistration']
    if limits['median_limit_mm'] != 10 or limits['p95_limit_mm'] != 20 or limits['scaling']:
        raise ValueError('Frozen 10/20 mm thresholds or scaling policy changed')
    if args.output_root.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_root}; use a fresh numbered run')
    consumed = {}
    def use(path):
        path = Path(path).resolve()
        consumed[str(path)] = sha(path)
        return path
    use(Path(__file__)); use(cfgpath)
    manifest_path = use(args.alignment_root/'05_manifest.json')
    prior = json.loads(manifest_path.read_text())
    for filename, expected in prior['input_sha256'].items():
        if sha(filename) != expected:
            raise ValueError(f'Existing alignment input changed: {filename}')
    nominal = pd.read_csv(use(args.alignment_root/'01_alignment.csv'))
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root/'figures').mkdir()
    # RUNNING survives an interruption; COMPLETE is written only after all files.
    manifest = dict(status='RUNNING', created_utc=datetime.now(timezone.utc).isoformat(),
                    argv=sys.argv, thresholds_mm=dict(median=10, p95=20), scaling=False,
                    fitting=False, inverse=False, anatomical_review='PENDING',
                    coordinate_convention='UNRESOLVED', localization_error_valid=False)
    def write_manifest():
        manifest['input_sha256'] = consumed
        (args.output_root/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    write_manifest()
    audits, sensors_out, summaries, residuals, shapes, landmarks, transforms, inventory = [], [], [], [], [], [], [], []
    models, mesh_meta, fids = {}, {}, {}
    for arm, directory in [('average_seven', args.average_seven), ('fsaverage', args.fsaverage)]:
        models[arm] = mesh(use(directory/'bem/outer_skin.surf'))[:2]
        mesh_meta[arm] = mesh(directory/'bem/outer_skin.surf')[2]
        fidpath = use(args.alignment_root/arm/f'{arm}-estimated-fiducials.fif')
        dig, frame = mne.io.read_fiducials(fidpath)
        if int(frame) != 5:
            raise ValueError(f'{fidpath}: fiducials must be MRI frame')
        fids[arm] = {int(d['ident']): np.asarray(d['r'])*1000 for d in dig}
    sidecar_signatures, tsv_signatures, forward_geometries = {}, {}, {}
    for subject in cfg['participants']:
        print(f'Auditing {subject}', flush=True)
        native = args.dataset/'derivatives/sourcemodelling'/subject
        eeg = args.dataset/'derivatives/epochs'/subject/'eeg'
        fwpath = use(native/'fwd'/f'{subject}_fwd.fif')
        fw = mne.read_forward_solution(fwpath, verbose=False)
        channels = fw['info']['chs']; names = [c['ch_name'] for c in channels]
        if len(names) != 256 or len(set(names)) != 256:
            raise ValueError(f'{subject}: expected 256 unique forward labels')
        points = np.array([c['loc'][:3] for c in channels], float)
        frames = sorted({int(c['coord_frame']) for c in channels})
        if frames != [4] or not np.isfinite(points).all():
            raise ValueError(f'{subject}: sensors must be finite with HEAD tags')
        trans = fw['mri_head_t']
        if (int(trans['from']), int(trans['to'])) != (5, 4):
            raise ValueError('Native transform must be MRI to HEAD')
        head_to_surface = np.linalg.inv(rigid(trans['trans']))
        native_points = apply(head_to_surface, points)*1000
        transforms.append(dict(subject=subject, model='native', source='inverse forward mri_head_t',
                               matrix_m=head_to_surface.tolist(), verified_direction='MRI to HEAD tag inverted'))
        skin = use(native/'anat'/f'{subject}_outer_skin.surf.gii')
        vv, ff, meta = mesh(skin); models[subject] = vv, ff; mesh_meta[subject] = meta
        cras = np.array([float(meta[f'VolGeomC_{axis}']) for axis in 'RAS'])
        h5path = native/'xfm'/f'{subject}_from-head_to-surface.h5'
        h5error = None
        if h5path.is_file():
            with h5py.File(use(h5path), 'r') as f:
                hh = rigid(f['trans'][()])
                h5error = float(np.max(np.abs(hh-head_to_surface)))
                transforms.append(dict(subject=subject, model='native', source=str(h5path), matrix_m=hh.tolist(),
                                       forward_inverse_max_abs=h5error))
        tsvpath = use(eeg/f'{subject}_task-seegstim_electrodes.tsv')
        tsv = pd.read_csv(tsvpath, sep='\t')
        if tsv.name.duplicated().any() or set(tsv.name) != set(names):
            raise ValueError(f'{subject}: TSV labels missing, extra or duplicated')
        released = tsv.set_index('name').loc[names, ['x','y','z']].to_numpy(float)
        sidepath = use(eeg/f'{subject}_task-seegstim_coordsystem.json')
        side = json.loads(sidepath.read_text())
        if side.get('EEGCoordinateUnits') != 'm' or side.get('AnatomicalLandmarkCoordinateUnits') != 'm':
            raise ValueError('Unexpected sidecar units; no silent conversion permitted')
        coords = side['AnatomicalLandmarkCoordinates']
        fid = np.array([coords[k] for k in ['LPA', 'NAS', 'RPA']], float)*1000
        sidecar_signatures[subject] = json.dumps(coords, sort_keys=True)
        tsv_signatures[subject] = released
        forward_geometries[subject] = dict(zip(names, points))
        for ident, label in [(1,'LPA'),(2,'NAS'),(3,'RPA')]:
            for hypothesis, fp in [('sidecar_as_surface_RAS', fid[ident-1]),
                                   ('sidecar_as_forward_HEAD', apply(head_to_surface, fid[ident-1:ident]/1000)[0]*1000)]:
                landmarks.append(dict(subject=subject, model='native', landmark=label, hypothesis=hypothesis,
                                      x_mm=fp[0], y_mm=fp[1], z_mm=fp[2],
                                      nearest_vertex_mm=float(cKDTree(vv).query(fp)[0]), status='UNVERIFIED_CONVENTION'))
        # Channels in every released run are independently inspected. The NPY
        # signal arrays and epoch JSONs do not supply independent digitization.
        chanfiles = sorted(eeg.glob('*_channels.tsv'))
        mismatches = []
        for cp in chanfiles:
            ct = pd.read_csv(use(cp), sep='\t')
            cn = ct.loc[ct.type.str.upper().eq('EEG'), 'name'].tolist()
            if cn != names:
                mismatches.append(cp.name)
        digitization = []
        for d in fw['info'].get('dig') or []:
            digitization.append(dict(kind=int(d['kind']), ident=int(d['ident']), frame=int(d['coord_frame']), r_m=np.asarray(d['r']).tolist()))
        fid_comparisons = []
        for d in digitization:
            if d['kind'] == 1 and d['ident'] in [1,2,3]:
                fid_comparisons.append(dict(ident=d['ident'], frame=d['frame'],
                    raw_sidecar_difference_mm=float(np.linalg.norm(np.array(d['r_m'])*1000-fid[d['ident']-1])),
                    interpretation='raw numeric comparison only; frame conventions unresolved'))
        audits.append(dict(subject=subject, n_forward=len(names), forward_frames=json.dumps(frames),
            eeg_units='m (FIF contract)', eeg_sidecar_frame=side.get('EEGCoordinateSystem'),
            sidecar_landmark_frame=side.get('AnatomicalLandmarkCoordinateSystem'),
            intended_for_exists=(args.dataset/side.get('IntendedFor','').lstrip('/')).is_file(),
            tsv_order_matches_forward=tsv.name.tolist()==names,
            released_run_channel_files=len(chanfiles), channel_order_mismatch_files=json.dumps(mismatches),
            tsv_forward_rms_mm=float(np.sqrt(np.mean(np.sum((points-released)**2, axis=1)))*1000),
            rotation_identity_error=float(np.max(np.abs(trans['trans'][:3,:3]-np.eye(3)))),
            forward_translation_minus_cras_max_mm=float(np.max(np.abs(trans['trans'][:3,3]*1000-cras))),
            h5_inverse_forward_max_abs=h5error, forward_dig_json=json.dumps(digitization),
            forward_dig_sidecar_comparison_json=json.dumps(fid_comparisons),
            independent_sensor_verification='NOT_ESTABLISHED: exported TSV not original digitizer; channel metadata only'))
        for i, name in enumerate(names):
            sensors_out.append(dict(subject=subject, channel=name, frame_tag=4,
                x_m=points[i,0], y_m=points[i,1], z_m=points[i,2],
                tsv_delta_mm=float(np.linalg.norm(points[i]-released[i])*1000),
                reference_loc_json=json.dumps(channels[i]['loc'][3:6].tolist())))
        native_regions = regions(native_points, vv)
        panels = []
        for model in ['native', 'fsaverage', 'average_seven']:
            vertices, faces = models[subject if model == 'native' else model]
            if model == 'native':
                mapped = native_points
            else:
                record = nominal[(nominal.arm==model)&(nominal.test_subject==subject)]
                if len(record) != 1:
                    raise ValueError('Expected one saved nominal transform per arm/subject')
                tp = use(Path(record.iloc[0]['transform']))
                tr = mne.read_trans(tp)
                if (int(tr['from']), int(tr['to'])) != (4, 5):
                    raise ValueError('Saved template transform must be HEAD to MRI')
                matrix = rigid(tr['trans']); mapped = apply(matrix, points)*1000
                transforms.append(dict(subject=subject, model=model, source=str(tp), matrix_m=matrix.tolist()))
            vertex_dist, indices = cKDTree(vertices).query(mapped)
            tri_dist, nearest = triangle_distance(mapped, vertices, faces)
            if np.any(tri_dist > vertex_dist+1e-6):
                raise ValueError('Triangle distance exceeds vertex upper bound')
            s = stats(vertex_dist)
            numerical = s['median_mm'] <= 10 and s['p95_mm'] <= 20
            if model != 'native':
                old = record.iloc[0]
                if not np.allclose([s['median_mm'], s['p95_mm']], [old.median_mm, old.p95_mm], atol=1e-4, rtol=0):
                    raise ValueError('Saved nominal nearest-vertex metrics did not reproduce')
            summaries.append(dict(subject=subject, model=model, metric='nearest_vertex', numerical_pass=numerical, **s))
            summaries.append(dict(subject=subject, model=model, metric='point_to_triangle', numerical_pass='NOT_A_GATE', **stats(tri_dist)))
            for i, name in enumerate(names):
                residuals.append(dict(subject=subject, model=model, channel=name, native_region_proxy=native_regions[i],
                    vertex_distance_mm=vertex_dist[i], triangle_distance_mm=tri_dist[i],
                    over_20mm=bool(vertex_dist[i]>20), x_mm=mapped[i,0], y_mm=mapped[i,1], z_mm=mapped[i,2],
                    residual_x_mm=mapped[i,0]-nearest[i,0], residual_y_mm=mapped[i,1]-nearest[i,1], residual_z_mm=mapped[i,2]-nearest[i,2]))
            panels.append((model, vertices, faces, mapped, nearest, tri_dist, names))
        figure(subject, panels, args.output_root/'figures'/f'{subject}_scalp_residuals.png')
        write_manifest()
    # Full mesh extents and robust coordinate spans are descriptive, coverage dependent.
    # Horizontal sections compare contours without scaling or new MRI registration.
    profiles = []
    for model, (vertices, faces) in models.items():
        lo, hi = np.percentile(vertices, [2,98], axis=0)
        shapes.append(dict(model=model, vertices=len(vertices), triangles=len(faces),
            **{f'{axis}_extent_mm':float(np.ptp(vertices[:,i])) for i,axis in enumerate('xyz')},
            **{f'{axis}_p02_p98_span_mm':float(hi[i]-lo[i]) for i,axis in enumerate('xyz')}))
        for z in [0, 20, 40, 60, 80]:
            band = vertices[np.abs(vertices[:,2]-z) <= 2.5]
            if len(band) > 10:
                profiles.append(dict(model=model, surface_ras_z_mm=z, n_vertices=len(band),
                                     width_mm=float(np.ptp(band[:,0])), depth_mm=float(np.ptp(band[:,1]))))
        if model in fids:
            for ident, point in fids[model].items():
                landmarks.append(dict(subject='template', model=model, landmark={1:'LPA',2:'NAS',3:'RPA'}[ident],
                                      hypothesis='estimated_MRI_fiducial', x_mm=point[0],y_mm=point[1],z_mm=point[2],
                                      nearest_vertex_mm=float(cKDTree(vertices).query(point)[0]),status='ESTIMATED_UNREVIEWED'))
    regional = []
    rd = pd.DataFrame(residuals)
    for (subject, model, region), group in rd.groupby(['subject','model','native_region_proxy']):
        all_group = rd[(rd.subject==subject)&(rd.model==model)]
        regional.append(dict(subject=subject, model=model, region=region,
            n=len(group), over_20mm=int(group.over_20mm.sum()), fraction_over_20mm=float(group.over_20mm.mean()),
            share_of_total_squared_triangle_residual=float(np.sum(group.triangle_distance_mm**2)/max(np.sum(all_group.triangle_distance_mm**2),1e-30)),
            **{k:v for k,v in stats(group.triangle_distance_mm.to_numpy()).items() if k!='n'}))
    pairs = []
    subjects = cfg['participants']
    for i, a in enumerate(subjects):
        for b in subjects[i+1:]:
            nn = sorted(forward_geometries[a])
            delta = np.array([forward_geometries[a][n]-forward_geometries[b][n] for n in nn])*1000
            pairs.append(dict(subject_a=a, subject_b=b, identical_landmarks=sidecar_signatures[a]==sidecar_signatures[b],
                identical_tsv=np.array_equal(tsv_signatures[a],tsv_signatures[b]),
                forward_same_label_rms_mm=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                interpretation='raw declared-frame comparison; does not establish intersubject anatomical correspondence'))
    # Inventory explicitly limited to dataset tree plus template/alignment inputs.
    for p in sorted(args.dataset.rglob('*')):
        if not p.is_file():
            continue
        n = p.name.lower()
        if ('xfm' in p.parts or any(k in n for k in ['coordsystem','electrodes','fiduc','-trans.','warp','affine','polhemus','egi_dig'])
            or p.suffix.lower() in ['.hpts','.elp','.lta','.tfm']):
            use(p)
            inventory.append(dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=consumed[str(p.resolve())],
                                  interpretation='file name/metadata inventory; convention and direction require verification'))
    landmark_dims = []
    for subject in subjects:
        coords = json.loads(sidecar_signatures[subject])
        for a,b in [('LPA','RPA'),('LPA','NAS'),('RPA','NAS')]:
            landmark_dims.append(dict(model=subject, pair=f'{a}-{b}', distance_mm=float(np.linalg.norm(np.array(coords[a])-coords[b])*1000), provenance='repeated released sidecar; anatomical convention unresolved'))
    for model, ffids in fids.items():
        for a,b in [(1,3),(1,2),(3,2)]:
            landmark_dims.append(dict(model=model, pair={1:'LPA',2:'NAS',3:'RPA'}[a]+'-'+{1:'LPA',2:'NAS',3:'RPA'}[b], distance_mm=float(np.linalg.norm(ffids[a]-ffids[b])), provenance='estimated unreviewed MRI landmarks'))
    shape_comparisons = []
    shape_lookup = {r['model']:r for r in shapes}
    for subject in subjects:
        for arm in ['fsaverage','average_seven']:
            shape_comparisons.append(dict(subject=subject, template=arm,
                **{f'{axis}_extent_template_minus_native_mm':shape_lookup[arm][f'{axis}_extent_mm']-shape_lookup[subject][f'{axis}_extent_mm'] for axis in 'xyz'},
                interpretation='unregistered surface axes; coverage and orientation confounded'))
    tables = {'01_coordinate_audit':audits,'02_recovered_sensors':sensors_out,'03_model_distances':summaries,
              '04_electrode_residuals':residuals,'05_regional_residuals':regional,'06_scalp_dimensions':shapes,
              '07_shape_sections':profiles,'08_landmark_audit':landmarks,'09_cross_participant_identity':pairs,
              '10_available_coordinate_files':inventory, '11_landmark_dimensions':landmark_dims,
              '12_template_native_dimensions':shape_comparisons}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(args.output_root/f'{name}.csv', index=False)
    (args.output_root/'transforms.json').write_text(json.dumps(transforms, indent=2)+'\n')
    (args.output_root/'surface_metadata.json').write_text(json.dumps(mesh_meta, indent=2, default=str)+'\n')
    # Cohort residual heatmaps retain channel labels and all seven participants.
    fig, axes = plt.subplots(3, 1, figsize=(18, 9), constrained_layout=True)
    for ax, model in zip(axes, ['native','fsaverage','average_seven']):
        table = rd[rd.model==model].pivot(index='subject', columns='channel', values='triangle_distance_mm')
        cols = sorted(table.columns, key=lambda x: int(''.join(filter(str.isdigit,x)) or 0))
        table = table.reindex(index=subjects, columns=cols)
        im = ax.imshow(table, aspect='auto', vmin=0, vmax=40, cmap='viridis')
        ax.set_title(model); ax.set_yticks(range(7), subjects)
        ax.set_xticks(range(0,len(cols),8), cols[::8], rotation=90, fontsize=6)
    fig.colorbar(im, ax=axes, label='Unsigned triangle distance (mm)')
    fig.savefig(args.output_root/'figures/cohort_channel_residuals.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
    for model, group in pd.DataFrame(profiles).groupby('model'):
        for ax, key in zip(axes,['width_mm','depth_mm']):
            ax.plot(group.surface_ras_z_mm, group[key], 'o-', label=model)
            ax.set(xlabel='surface RAS Z (mm)',ylabel=key)
    axes[1].legend(fontsize=7); fig.suptitle('Unregistered horizontal mesh bands; orientation and coverage confounded')
    fig.savefig(args.output_root/'figures/scalp_sections.png',dpi=160); plt.close(fig)
    distance_table = pd.DataFrame(summaries)
    counts = distance_table[(distance_table.metric=='nearest_vertex') & (distance_table.model!='native')].groupby('model').numerical_pass.sum().to_dict()
    lines = ['# EEG alignment diagnostic report', '',
        'Status: released-data diagnostics complete; anatomical and coordinate-convention review pending.', '',
        '## Verified numerical findings', '',
        f'Saved nominal numerical passes reproduced: {counts}. Thresholds remain median ≤10 mm and p95 ≤20 mm, nearest vertex, all 256 sensors. Visual review remains required.', '',
        f'All-participant sidecar landmark identity: {len(set(sidecar_signatures.values())) == 1}. See pairwise electrode/landmark comparisons in 09_cross_participant_identity.csv.', '',
        'Original forward sensor coordinates are recovered without refitting in 02_recovered_sensors.csv. HDF5 versus inverse-forward checks and GIFTI volume-center checks are in 01_coordinate_audit.csv. These test consistency of released artifacts, not independent digitizer provenance.', '',
        '## Model and regional comparisons', '',
        distance_table.to_markdown(index=False) if __import__('importlib').util.find_spec('tabulate') else distance_table.to_string(index=False), '',
        'Regional concentration is reported continuously as fraction >20 mm and share of squared triangle residual, with region counts. A high share in a small region supports localized mismatch; broad elevations support global mismatch. No new pass/fail or concentration threshold is introduced.', '',
        pd.DataFrame(regional).to_string(index=False), '',
        '## Coordinate provenance and unresolved conventions', '',
        'FIF HEAD tags and metre units are verified file contracts. Their anatomical meaning is unresolved. GIFTI positions are treated as FreeSurfer surface RAS millimetres under the existing pipeline contract; magnitude checks do not independently prove units or orientation.', '',
        'The exporter fx_bids.make_coordsys_eeg_json copies epoch dig fiducials and labels them T1w; make_electrodes_eeg_tsv copies epoch channel loc. This explains what the code writes, but does not prove which input files produced the release or why all participants repeat the same values. Shared montage/export input is a hypothesis, not a verified cause.', '',
        'fx_preprocess.make_dig_files negates the first two Polhemus coordinates and writes HPTS/FCsv. Original digitizer records and the exact MNE montage import history are needed to verify units, handedness, and anatomical correspondence independently.', '',
        'Released channel TSVs independently check label sets/order only. NPY epoch arrays contain signals, not independent sensor coordinates. Matching HDF5 and forward transforms can share provenance. No independent anatomical validation is claimed.', '',
        'Sidecar landmarks are evaluated under two explicit hypotheses (surface RAS versus forward HEAD). Neither interpretation is selected. Template landmarks are estimated and unreviewed. No verified native anatomical fiducials are available; direct anatomical landmark correspondence remains unresolved.', '',
        'Full extents, robust spans and horizontal sections compare all seven scalps with both templates. Sections use original surface RAS axes, without new MRI registration; orientation, surface coverage, facial masking and neck extent confound size/shape attribution.', '',
        'Regions are native surface-RAS geometric proxies (superior, left, right, anterior, posterior). Each channel retains the same native bin across models. These are not atlas labels or independently validated anatomical regions.', '',
        'Nearest-vertex distances reproduce the original gate. Exact unsigned triangle distances supplement them and do not redefine acceptance. Neither metric distinguishes inside from outside or measures digitization error. Plots use shared mm axes and 0–40 mm color scales; rendering alone is subsampled.', '',
        '## Scope and preservation', '',
        'No anatomy, saved transform, existing result, alignment threshold or cohort membership was changed. No scaling, ICP, forward calculation or inverse localization was run. The BEM repair remains pending anatomical review and is not validated by scalp proximity.', '',
        'Inventory scope is the accessible dataset tree, selected templates and saved alignment provenance; absence here is not proof of absence elsewhere. See manifest.json for content hashes, software versions and output hashes.']
    (args.output_root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for path, value in consumed.items():
        if sha(path) != value:
            raise ValueError(f'Input changed during run: {path}')
    manifest.update(status='COMPLETE', nominal_numerical_pass_counts={k:int(v) for k,v in counts.items()},
        software=dict(python=platform.python_version(),mne=mne.__version__,numpy=np.__version__,nibabel=nib.__version__,pandas=pd.__version__),
        output_sha256={str(p.relative_to(args.output_root)):sha(p) for p in args.output_root.rglob('*') if p.is_file() and p.name!='manifest.json'})
    write_manifest()
    print(args.output_root)


if __name__ == '__main__':
    main()
