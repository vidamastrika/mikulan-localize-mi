#!/usr/bin/env python3
"""Script 26: exploratory SESAME source visualization in MNI152NLin2009aSym.

Run from project root after Scripts 20, 22, and 25. This script DOES NOT
register images, fit source models, or recompute localization errors. It maps
all four target-native points through the SAME target->MNI transformation.

The displayed brain boundary is a mesh of the TemplateFlow brain mask, NOT a
pial/cortical surface. The legend reports separately the target-native 3D
localization errors and the 3D distances after nonlinear MNI normalization.
MNI distances are secondary, registration-dependent quantities.

Dependencies in .venv: numpy, nibabel, pandas, matplotlib, scikit-image.
ANTs executable: antsApplyTransformsToPoints in micromamba environment.

Example:
  micromamba run -n localizemi-registration .venv/bin/python \\
    cross_subject/scripts/26_visualize_sesame_on_mni.py \\
    --target sub-02 --donor sub-01 --run run-01
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
import pandas as pd

TEMPLATE = 'MNI152NLin2009aSym'
COLORS = {'Ground truth':'#22c55e', 'Within-subject SESAME':'#2697ec',
          'Cross-subject affine':'#dc37b7', 'Cross-subject SyN':'#f59e0b'}
ORDER = list(COLORS)
FLIP = np.array([-1., -1., 1.])


def antspath(name: str, env: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    p = Path.home()/'micromamba'/'envs'/env/'bin'/name
    if p.is_file():
        return str(p)
    raise FileNotFoundError(f'{name} not found; run via micromamba run -n {env} or update --micromamba-env')


def points_via_ants(exe: str, ras: np.ndarray, transforms: list[str]) -> np.ndarray:
    ras = np.asarray(ras, float)
    if ras.ndim != 2 or ras.shape[1] != 3 or not np.isfinite(ras).all():
        raise ValueError('Expected finite Nx3 RAS points in millimetres')
    lps = ras * FLIP
    with tempfile.TemporaryDirectory(prefix='localizemi_26_') as directory:
        input_csv = Path(directory)/'points.csv'; output_csv = Path(directory)/'mapped.csv'
        pd.DataFrame(dict(x=lps[:,0], y=lps[:,1], z=lps[:,2], t=np.zeros(len(lps)))).to_csv(
            input_csv, index=False, float_format='%.12f')
        command = [exe, '-d','3','-i',str(input_csv),'-o',str(output_csv)]
        for item in transforms:
            command += ['-t', item]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0 or not output_csv.is_file():
            raise RuntimeError('ANTs point transform failed:\n'+proc.stdout+'\n'+proc.stderr)
        result = pd.read_csv(output_csv)[['x','y','z']].to_numpy(float) * FLIP
    if result.shape != ras.shape or not np.isfinite(result).all():
        raise RuntimeError('Invalid mapped point output')
    return result


def plot_mni(points, mask_image, out, native_errors, mni_distances, step_size):
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError('Install scikit-image into .venv: .venv/bin/python -m pip install scikit-image') from exc
    brain_mask = np.asanyarray(mask_image.dataobj) > 0.5
    if brain_mask.sum() < 1000:
        raise ValueError('Invalid empty template brain mask')
    # Pad to avoid open boundary if mask contacts the image edge.
    padded = np.pad(brain_mask.astype(np.uint8), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=.5, step_size=step_size)
    vertices -= 1
    world_vertices = nib.affines.apply_affine(mask_image.affine, vertices)
    # Matplotlib CPU rendering becomes expensive with hundreds of thousands of triangles.
    if len(faces) > 45000:
        faces = faces[::int(np.ceil(len(faces)/45000))]
    tris = world_vertices[faces]
    fig = plt.figure(figsize=(15, 7.5), facecolor='white')
    panels = [(25, -70, 'Oblique lateral'), (70, -90, 'Superior / oblique')]
    coords = np.stack([points[name] for name in ORDER])
    mins = np.minimum(world_vertices.min(axis=0), coords.min(axis=0))-8
    maxs = np.maximum(world_vertices.max(axis=0), coords.max(axis=0))+8
    centre = (mins+maxs)/2
    half_width = np.max(maxs-mins)/2
    for idx, (elev, azim, label) in enumerate(panels, 1):
        ax = fig.add_subplot(1,2,idx, projection='3d')
        mesh = Poly3DCollection(tris, alpha=.095, facecolor='#87909e', edgecolor='none', zsort='average')
        ax.add_collection3d(mesh)
        for name in ORDER:
            x,y,z=points[name]
            ax.scatter([x],[y],[z], marker='o', s=135, c=COLORS[name],
                       edgecolors='black', linewidths=.8, depthshade=False)
        ax.set_xlim(centre[0]-half_width,centre[0]+half_width)
        ax.set_ylim(centre[1]-half_width,centre[1]+half_width)
        ax.set_zlim(centre[2]-half_width,centre[2]+half_width)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('R–L (RAS X, mm)'); ax.set_ylabel('A–P (RAS Y, mm)'); ax.set_zlabel('S–I (RAS Z, mm)')
        ax.grid(False)
        ax.set_axis_off()
    handles=[]
    for name in ORDER:
        distances = ('reference' if name == 'Ground truth' else
                     f'native {native_errors[name]:.2f} mm | MNI {mni_distances[name]:.2f} mm')
        handles.append(Line2D([0],[0],marker='o',linestyle='None',markersize=10,
                              markerfacecolor=COLORS[name], markeredgecolor='black',
                              label=f'{name} — {distances}'))
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(.5,.055))
    fig.suptitle('SESAME locations mapped to MNI152NLin2009aSym — exploratory', fontsize=16, y=.97)
    fig.text(.5,.03,'Native = primary localization error; MNI = distance after nonlinear registration (not directly comparable).\n'
             'Template brain-mask boundary (not pial cortex); T1w RAS assumption and normalization uncertainty apply.',
             ha='center',va='center',fontsize=9, color='#555555')
    fig.subplots_adjust(left=.01,right=.99,bottom=.17,top=.91,wspace=.01)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return {'mesh_vertices':len(world_vertices),'mesh_faces_plotted':len(faces),
            'surface_description':'TemplateFlow whole-brain binary mask isosurface (not pial cortex)'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',default='sub-02')
    parser.add_argument('--donor',default='sub-01')
    parser.add_argument('--run',default='run-01')
    parser.add_argument('--micromamba-env',default='localizemi-registration')
    parser.add_argument('--step-size',type=int,default=3,help='Marching cubes stride (>=1)')
    parser.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/26_mni_sesame_visualization'))
    args=parser.parse_args()
    if args.step_size<1:
        parser.error('--step-size must be >= 1')
    tag=f'{args.target}_with_{args.donor}_{args.run}'
    audit22_path=Path('cross_subject/outputs/22_three_brain_sesame')/tag/'visualization_audit.json'
    audit25_path=Path('cross_subject/outputs/25_native_to_mni_registration')/args.target/'native_to_mni_registration_audit.json'
    for path in (audit22_path,audit25_path):
        if not path.is_file():
            raise FileNotFoundError(f'Input audit not found: {path}. Complete Scripts 22 and 25.')
    audit22=json.loads(audit22_path.read_text())
    audit25=json.loads(audit25_path.read_text())
    if audit22.get('target_subject')!=args.target or audit22.get('donor_subject')!=args.donor or audit22.get('run')!=args.run:
        raise RuntimeError('Script 22 audit subject/run mismatch')
    if audit25.get('subject')!=args.target or audit25.get('template_id')!=TEMPLATE:
        raise RuntimeError('Script 25 audit subject/template mismatch')
    if audit25.get('status')!='EXPLORATORY_MNI_REGISTRATION_CONTACT_VALIDATION':
        raise RuntimeError('Script 25 is not validated to expected audit stage')
    native_points=np.array([audit22['target_native_points_mm'][k] for k in ORDER],float)
    errors={k:float(v) for k,v in audit22['target_native_error_mm'].items()}
    truth=native_points[0]
    for name,point in zip(ORDER[1:],native_points[1:]):
        distance=float(np.linalg.norm(point-truth))
        if not np.isclose(distance, errors[name],rtol=0,atol=.02):
            raise RuntimeError(f'Native error mismatch {name}: recomputed {distance:.5f}, expected {errors[name]:.5f}')
    # Dataset uses source 'surface' coordinates (millimetres). The original
    # HDF5 'head -> surface' matrix is verified in earlier scripts to relate
    # native surface to T1w electrodes via its INVERSE.
    import h5py
    dataset=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI'
    h5=dataset/'derivatives'/'sourcemodelling'/args.target/'xfm'/f'{args.target}_from-head_to-surface.h5'
    if not h5.is_file():
        raise FileNotFoundError(h5)
    with h5py.File(h5,'r') as handle:
        head_to_surface=np.asarray(handle['trans'],float)
    if head_to_surface.shape!=(4,4) or not np.allclose(head_to_surface[3],[0,0,0,1]):
        raise RuntimeError('Invalid head-to-surface matrix')
    surface_to_t1w=np.linalg.inv(head_to_surface)
    # HDF5 transform uses metres; converting both sides into metres explicitly.
    native_h=np.column_stack([native_points/1000.,np.ones(len(native_points))])
    t1w_mm=(native_h@surface_to_t1w.T)[:,:3]*1000.
    affine=Path(audit25['transform_file_affine'])
    inverse_warp=Path(audit25['transform_file_inverse_warp'])
    forward_warp=Path(audit25['transform_file_forward_warp'])
    for f in (affine,inverse_warp,forward_warp):
        if not f.is_file() or f.stat().st_size==0:
            raise FileNotFoundError(f)
    exe=antspath('antsApplyTransformsToPoints',args.micromamba_env)
    to_mni=[f'[{affine},1]',str(inverse_warp)]
    to_native=[str(forward_warp),str(affine)]
    points_mni=points_via_ants(exe,t1w_mm,to_mni)
    mni_distances={name:float(np.linalg.norm(points_mni[i]-points_mni[0]))
                   for i,name in enumerate(ORDER) if i>0}
    returned=points_via_ants(exe,points_mni,to_native)
    roundtrip=np.linalg.norm(returned-t1w_mm,axis=1)
    if float(roundtrip.max())>.5:
        raise RuntimeError(f'MNI point round-trip too large: {roundtrip.max():.3f} mm')
    mask_path=Path(audit25['template_mask'])
    mask=nib.load(str(mask_path))
    out=args.output_root/tag; out.mkdir(parents=True,exist_ok=True)
    figure=out/'sesame_four_sources_mni_template.png'
    plotted={k:points_mni[i] for i,k in enumerate(ORDER)}
    mesh_info=plot_mni(plotted,mask,figure,errors,mni_distances,args.step_size)
    csv=out/'sesame_sources_native_and_mni.csv'
    pd.DataFrame([{'point':name,'target_native_x_mm':native_points[i,0],
                   'target_native_y_mm':native_points[i,1],'target_native_z_mm':native_points[i,2],
                   'target_t1w_x_mm':t1w_mm[i,0],'target_t1w_y_mm':t1w_mm[i,1],
                   'target_t1w_z_mm':t1w_mm[i,2],
                   'mni_x_mm':points_mni[i,0],'mni_y_mm':points_mni[i,1],
                   'mni_z_mm':points_mni[i,2],
                   'target_native_error_mm':0. if i==0 else errors[name],
                   'mni_distance_from_ground_truth_mm':0. if i==0 else mni_distances[name],
                   'mni_roundtrip_mm':roundtrip[i]} for i,name in enumerate(ORDER)]).to_csv(csv,index=False)
    report={'status':'EXPLORATORY_MNI_VISUALIZATION',
            'target':args.target,'donor':args.donor,'run':args.run,
            'template_id':TEMPLATE,'template_mask':str(mask_path),
            'target_surface_to_t1w_matrix':surface_to_t1w.tolist(),
            'point_sequence_target_to_mni':to_mni,'point_sequence_mni_to_target':to_native,
            'max_roundtrip_mm':float(roundtrip.max()),'points_mni_ras_mm':{k:v.tolist() for k,v in plotted.items()},
            'target_native_error_mm':errors,
            'mni_distance_from_ground_truth_mm':mni_distances,
            'mni_distance_definition':'Euclidean 3D distance between transformed source and ground truth in MNI152NLin2009aSym RAS mm',
            'mesh':mesh_info,
            'script25_mni_contact_discrepancy_median_mm':audit25['contact_discrepancy_to_author_mni_mm']['median'],
            'outputs':{'figure':str(figure),'csv':str(csv)},
            'limitations':['Provisional assumption: Localize-MI native T1w coordinates equal physical NIfTI RAS',
                           'MNI normalization differs from author contact positions (independent validation)',
                           'Mesh is whole-brain mask boundary, NOT left/right pial surfaces',
                           'MNI-space distances are secondary and depend on nonlinear template normalization; not direct native-space localization errors',
                           'Point roundtrip verifies implementation, not anatomical accuracy',
                           'No localization ground truth used to fit transforms']}
    json_path=out/'mni_visualization_audit.json'
    json_path.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved MNI figure:',figure)
    print('Saved point CSV:',csv)
    print('Saved audit:',json_path)
    print('Point roundtrip max: %.4f mm'%roundtrip.max())
    print('3D distance comparison (native mm | MNI mm):')
    for name in ORDER[1:]:
        print(f'  {name}: {errors[name]:.2f} | {mni_distances[name]:.2f}')
    print('CAVEAT: Template brain MASK mesh (not pial cortex); MNI distances depend on nonlinear normalization.')

if __name__=='__main__':
    main()
