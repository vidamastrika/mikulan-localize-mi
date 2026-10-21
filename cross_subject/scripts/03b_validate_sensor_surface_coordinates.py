#!/usr/bin/env python3
"""Read-only Localize-MI sensor/TSV/scalp coordinate diagnostics.

No registration, fitting, or forward-model computation is performed.
This script tests coordinate hypotheses; none is automatically considered valid.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_DATA = Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI'
DEFAULT_OUT = Path(__file__).resolve().parents[1] / 'outputs' / '03b_coordinate_validation'


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def get_fwd(root, subject):
    import mne
    from mne.io.constants import FIFF
    path = root / 'derivatives/sourcemodelling' / subject / 'fwd' / f'{subject}_fwd.fif'
    fwd = mne.read_forward_solution(str(path), verbose='ERROR')
    chs = [c for c in fwd['info']['chs'] if c['kind'] == FIFF.FIFFV_EEG_CH]
    if not chs or any(c['coord_frame'] != FIFF.FIFFV_COORD_HEAD for c in chs):
        raise ValueError(f'{subject}: EEG sensors must be in HEAD coordinates')
    names = [c['ch_name'] for c in chs]
    xyz = np.array([c['loc'][:3] for c in chs], dtype=float)
    t = fwd['mri_head_t']
    if t['from'] != FIFF.FIFFV_COORD_MRI or t['to'] != FIFF.FIFFV_COORD_HEAD:
        raise ValueError(f'{subject}: expected MRI->HEAD transform')
    return dict(zip(names, xyz)), np.asarray(t['trans'], dtype=float)


def get_tsv(root, subject):
    file = root / 'derivatives/epochs' / subject / 'eeg' / f'{subject}_task-seegstim_electrodes.tsv'
    framefile = file.with_name(f'{subject}_task-seegstim_coordsystem.json')
    metadata = json.loads(framefile.read_text())
    units = metadata.get('EEGCoordinateUnits')
    if units not in ('m', 'mm'):
        raise ValueError(f'{subject}: unknown electrode unit {units!r}')
    scale = 1 if units == 'm' else 1e-3
    table = pd.read_csv(file, sep='\t').set_index('name')
    coords = table[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce') * scale
    return dict(zip(coords.index, coords.to_numpy())), metadata


def get_scalp(root, subject):
    import nibabel as nib
    from nibabel.nifti1 import intent_codes
    file = root / 'derivatives/sourcemodelling' / subject / 'anat' / f'{subject}_outer_skin.surf.gii'
    gii = nib.load(str(file))
    point = [a for a in gii.darrays if int(a.intent) == int(intent_codes['NIFTI_INTENT_POINTSET'])]
    tri = [a for a in gii.darrays if int(a.intent) == int(intent_codes['NIFTI_INTENT_TRIANGLE'])]
    if len(point) != 1 or len(tri) != 1:
        raise ValueError(f'{subject}: expected one POINTSET and one TRIANGLE array')
    v = np.asarray(point[0].data, dtype=float)
    t = np.asarray(tri[0].data, dtype=np.int64)
    scale = 1e-3 if np.median(np.linalg.norm(v, axis=1)) > 2 else 1
    v *= scale
    if not (np.isfinite(v).all() and t.min() >= 0 and t.max() < len(v)):
        raise ValueError('Invalid scalp mesh geometry')
    return v, t, 'mm' if scale == 1e-3 else 'm'


def distances(a, b):
    delta = (a-b)*1000
    norms = np.linalg.norm(delta, axis=1)
    return dict(n=len(norms), median_mm=float(np.median(norms)),
                mean_mm=float(np.mean(norms)), p95_mm=float(np.percentile(norms,95)),
                max_mm=float(np.max(norms)), max_abs_axis_mm=float(np.abs(delta).max()))


def scalp_distances(electrodes, vertices):
    d = cKDTree(vertices).query(electrodes)[0]*1000
    return dict(n=len(d), median_mm=float(np.median(d)), mean_mm=float(np.mean(d)),
                p95_mm=float(np.percentile(d,95)), max_mm=float(d.max()),
                over_15mm=int((d > 15).sum()))


def plot_mesh_comparison(panels, file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    for idx, (label, verts, triangles, electrodes) in enumerate(panels):
        ax = fig.add_subplot(1, 2, idx+1, projection='3d')
        # Display capped mesh triangle count for speed; distance calculations always full vertices.
        stride = max(1, len(triangles)//4500)
        faces = verts[triangles[::stride]]*1000
        mesh = Poly3DCollection(faces, alpha=.13, linewidths=0)
        mesh.set_facecolor('steelblue')
        ax.add_collection3d(mesh)
        dist = cKDTree(verts).query(electrodes)[0]*1000
        sc = ax.scatter(*(electrodes*1000).T, c=dist, s=13, cmap='viridis', vmin=0,
                        vmax=max(25, np.percentile(dist,95)), depthshade=False)
        allpts = np.concatenate([verts, electrodes])*1000
        midpoint = (allpts.max(axis=0)+allpts.min(axis=0))/2
        half = np.max(allpts.max(axis=0)-allpts.min(axis=0))/2*1.08
        ax.set_xlim(midpoint[0]-half,midpoint[0]+half)
        ax.set_ylim(midpoint[1]-half,midpoint[1]+half)
        ax.set_zlim(midpoint[2]-half,midpoint[2]+half)
        ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)')
        ax.set_title(label, fontsize=10)
        ax.view_init(elev=20,azim=-62)
        fig.colorbar(sc, ax=ax, shrink=.6, pad=.07, label='Unsigned nearest-vertex distance (mm)')
    fig.suptitle('Sensor/scalp diagnostic — hypothesis: GIFTI MRI -> forward HEAD (not registered)')
    fig.savefig(file,dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--eeg-subject',default='sub-02')
    parser.add_argument('--donor-subject',default='sub-01')
    args=parser.parse_args()
    if args.eeg_subject == args.donor_subject: parser.error('Choose different participants')
    args.output_root.mkdir(parents=True,exist_ok=True)
    root=args.dataset_root
    target_fwd,target_mat=get_fwd(root,args.eeg_subject)
    donor_fwd,donor_mat=get_fwd(root,args.donor_subject)
    target_tsv,target_meta=get_tsv(root,args.eeg_subject)
    donor_tsv,donor_meta=get_tsv(root,args.donor_subject)
    comparison=[]
    for sub,forward,tsv,mri_head,metadata in [(args.eeg_subject,target_fwd,target_tsv,target_mat,target_meta),
                                               (args.donor_subject,donor_fwd,donor_tsv,donor_mat,donor_meta)]:
        common=sorted(forward.keys() & tsv.keys())
        if not common: raise ValueError(f'{sub}: no overlapping channel names')
        f=np.stack([forward[n] for n in common]); p=np.stack([tsv[n] for n in common])
        for hypothesis,v in [('TSV_as_HEAD',p),('TSV_as_MRI_to_HEAD',transform(p,mri_head)),
                             ('TSV_as_HEAD_to_MRI',transform(p,np.linalg.inv(mri_head)))]:
            comparison.append(dict(subject=sub,comparison='TSV_to_forward_HEAD',hypothesis=hypothesis,
                                   eeg_tsv_frame=metadata.get('EEGCoordinateSystem'),**distances(f,v)))
    common=sorted(target_fwd.keys() & donor_fwd.keys())
    comparison.append(dict(subject=f'{args.eeg_subject}_vs_{args.donor_subject}',comparison='forward_to_forward_HEAD',
                           hypothesis='same_named_sensors', **distances(np.stack([target_fwd[n] for n in common]),
                                                                        np.stack([donor_fwd[n] for n in common]))))
    common=sorted(target_tsv.keys() & donor_tsv.keys())
    comparison.append(dict(subject=f'{args.eeg_subject}_vs_{args.donor_subject}',comparison='TSV_to_TSV',
                           hypothesis='same_named_sensors', **distances(np.stack([target_tsv[n] for n in common]),
                                                                        np.stack([donor_tsv[n] for n in common]))))
    pd.DataFrame(comparison).to_csv(args.output_root/'sensor_coordinate_comparisons.csv',index=False)
    skins=[]; surfrows=[]
    electrodes=np.stack([target_fwd[n] for n in sorted(target_fwd)])
    for label,sub,mat in [('control',args.eeg_subject,target_mat),('donor',args.donor_subject,donor_mat)]:
        skin,tri,unit=get_scalp(root,sub)
        for h,vertices in [('GIFTI_as_HEAD',skin),('GIFTI_as_MRI_to_HEAD',transform(skin,mat))]:
            surfrows.append(dict(model=label,subject=sub,hypothesis=h,unit_detected=unit,
                                 **scalp_distances(electrodes,vertices)))
        skins.append((f'{label}: {sub}',transform(skin,mat),tri,electrodes))
    pd.DataFrame(surfrows).to_csv(args.output_root/'scalp_distance_comparisons.csv',index=False)
    plot_mesh_comparison(skins,args.output_root/'scalp_mesh_comparison.png')
    result=dict(eeg_subject=args.eeg_subject,donor_subject=args.donor_subject,
                comparisons=comparison,scalp_comparisons=surfrows,
                cautions=['Nearest-vertex distance is not signed and does not detect sensors inside the scalp.',
                          'GIFTI surface coordinate convention is not independently validated.',
                          'No subject-to-subject registration has been computed.',
                          'These results do not establish a valid cross-subject forward model.'])
    (args.output_root/'diagnostic.json').write_text(json.dumps(result,indent=2))
    print('Wrote diagnostics to',args.output_root)
    print(pd.DataFrame(comparison).to_string(index=False))
    print(pd.DataFrame(surfrows).to_string(index=False))

if __name__=='__main__':
    main()
