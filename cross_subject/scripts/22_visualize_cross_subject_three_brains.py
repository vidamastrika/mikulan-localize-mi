#!/usr/bin/env python3
"""3-D cortical visualization of the audited Localize-MI SESAME pilot.

Uses Script 20's *provisional* RAS/SyN coordinate chain. The target and donor
figures are generated without rerunning SESAME/registration. An MNI view is
created ONLY if independently validated MNI coordinates and matching MNI pial
surfaces are explicitly supplied; no landmark-based approximation is made.

Run from the repository root, with ANTs available on PATH (or under the
micromamba environment described in --help).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd

FLIP = np.diag([-1., -1., 1.])
COLORS = {
    'Ground truth': '#20c65a',
    'Within-subject SESAME': '#239dff',
    'Cross-subject affine': '#e640bd',
    'Cross-subject SyN': '#ff9b1a',
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, default=Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    ap.add_argument('--donor', default='sub-01')
    ap.add_argument('--target', default='sub-02')
    ap.add_argument('--run', default='run-01')
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--micromamba-env', default='localizemi-registration')
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--dpi', type=int, default=210)
    ap.add_argument('--surface-step', type=int, default=1, help='Plot every Nth face (1 = complete original mesh)')
    ap.add_argument('--mni-points-json', type=Path, help='EXTERNALLY VERIFIED mapped coordinates in matching template surface frame')
    ap.add_argument('--mni-left-gii', type=Path)
    ap.add_argument('--mni-right-gii', type=Path)
    return ap.parse_args()


def get_exe(name, env):
    p = shutil.which(name)
    if p: return p
    candidate = Path.home() / 'micromamba/envs' / env / 'bin' / name
    if candidate.exists(): return str(candidate)
    raise RuntimeError(f'{name} not found; use micromamba run -n {env} .venv/bin/python ...')


def ants_points(exe, ras_points, transforms):
    points = np.asarray(ras_points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all(): raise ValueError('Non-finite point')
    lps = points @ FLIP.T
    with tempfile.TemporaryDirectory(prefix='localizemi_fig22_') as td:
        inp, out = Path(td)/'input.csv', Path(td)/'output.csv'
        pd.DataFrame({'x': lps[:,0], 'y': lps[:,1], 'z': lps[:,2], 't': np.zeros(len(lps))}).to_csv(inp, index=False, float_format='%.12f')
        command = [exe, '-d','3','-i',str(inp),'-o',str(out)]
        for tr in transforms: command += ['-t',str(tr)]
        r = subprocess.run(command, capture_output=True, text=True)
        if r.returncode or not out.exists(): raise RuntimeError(f'ANTs point transform failed: {r.stderr}\n{r.stdout}')
        results = pd.read_csv(out)[['x','y','z']].to_numpy(float)
        if results.shape != points.shape or not np.isfinite(results).all(): raise RuntimeError('Bad ANTs mapped points')
        return results @ FLIP.T


def h5(dataset, sub):
    f = dataset/'derivatives/sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
    with h5py.File(f) as handle: m = np.asarray(handle['trans'],float)
    if m.shape != (4,4): raise ValueError(f'Unexpected H5 transform: {f}')
    return m


def transform(point_mm, matrix_in_m):
    return nib.affines.apply_affine(matrix_in_m,np.asarray(point_mm)/1000.)*1000.


def surf(path):
    f = nib.load(str(path))
    pts = np.asarray(f.darrays[0].data,dtype=float)
    if np.max(np.abs(pts)) < 1: pts *= 1000.
    tri = np.asarray(f.darrays[1].data,dtype=int)
    return pts,tri


def native_surfaces(dataset,sub):
    base=dataset/'derivatives/sourcemodelling'/sub/'anat'
    return [(surf(base/f'{sub}_hemi-L_pial.surf.gii'),'#dedee3'),
            (surf(base/f'{sub}_hemi-R_pial.surf.gii'),'#dedee3')], surf(base/f'{sub}_outer_skin.surf.gii')


def render(path, title, meshes, scalp, points, descriptions, dpi, face_step, disclaimer):
    fig=plt.figure(figsize=(17,9.4))
    ax=fig.add_axes([0.025,0.04,0.73,0.88],projection='3d')
    limits=[]
    for (xyz,tri),color in meshes:
        t=tri[::face_step]
        ax.plot_trisurf(xyz[:,0],xyz[:,1],xyz[:,2],triangles=t,color=color,alpha=.095,linewidth=0,shade=True)
        limits.append(xyz)
    if scalp is not None:
        xyz,tri=scalp
        ax.plot_trisurf(xyz[:,0],xyz[:,1],xyz[:,2],triangles=tri[::max(1,face_step)],color='#d9c8b4',alpha=.014,linewidth=0,shade=True)
    for name,coords in points.items():
        ax.scatter(*coords,s=125,color=COLORS[name],edgecolors='black',linewidths=.7,depthshade=False,zorder=20)
    truth=points.get('Ground truth')
    if truth is not None:
        for name,xyz in points.items():
            if name != 'Ground truth':
                ax.plot([truth[0],xyz[0]],[truth[1],xyz[1]],[truth[2],xyz[2]],color=COLORS[name],alpha=.85,linewidth=2)
    bounds=np.vstack(limits)
    centre=(bounds.min(0)+bounds.max(0))/2
    rad=max((bounds.max(0)-bounds.min(0)).max()/2,1.)
    for setter,cen in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),centre): setter(cen-rad,cen+rad)
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=18,azim=-70)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(title,fontsize=14,pad=15)
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS[n],markeredgecolor='black',markersize=10,label=descriptions.get(n,n)) for n in points]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.76,.54),frameon=False,fontsize=10)
    fig.text(.76,.19,disclaimer,fontsize=9,ha='left',va='top',wrap=True)
    fig.savefig(path,dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def main():
    a=parse_args()
    if a.surface_step < 1: raise ValueError('--surface-step must be >= 1')
    repo=a.repo.resolve(); data=a.dataset.expanduser().resolve()
    out=a.output_dir or repo/'cross_subject/outputs/22_three_brain_sesame'/f'{a.target}_with_{a.donor}_{a.run}'
    out.mkdir(parents=True,exist_ok=True)
    auditfile=repo/'cross_subject/outputs/20_syn_point_validation'/f'{a.target}_with_{a.donor}_{a.run}'/'syn_paired_sesame_audit.json'
    if not auditfile.exists(): raise FileNotFoundError(f'Missing Script 20 audit: {auditfile}')
    audit=json.loads(auditfile.read_text())
    if audit.get('metrics',{}).get('status') != 'PROVISIONAL_SYN_AFFINE_COMPARISON_NOT_PUBLICATION_VALIDATED':
        raise ValueError('Unexpected or unvalidated Script 20 audit status')
    coords=audit['source_coordinates_mm']; metrics=audit['metrics']
    tgt={
        'Ground truth':np.asarray(coords['ground_truth_target_surface'],float),
        'Within-subject SESAME':np.asarray(coords['within_subject_target_surface'],float),
        'Cross-subject affine':np.asarray(coords['target_surface_via_affine'],float),
        'Cross-subject SyN':np.asarray(coords['target_surface_via_syn'],float),
    }
    errors={'Within-subject SESAME':metrics['within_subject_error_mm'],
            'Cross-subject affine':metrics['cross_subject_affine_error_mm'],
            'Cross-subject SyN':metrics['cross_subject_syn_error_mm']}
    for name,err in errors.items():
        actual=np.linalg.norm(tgt[name]-tgt['Ground truth'])
        if abs(actual-err)>.03: raise ValueError(f'Distance mismatch {name}: {actual} vs {err}')
    targetmesh, targetscalp=native_surfaces(data,a.target)
    description={'Ground truth':'Ground truth (stimulation midpoint)',
                 **{k:f'{k}: {v:.2f} mm in target native space' for k,v in errors.items()}}
    render(out/'01_target_native_three_methods.png',f'Target native cortex — {a.target} {a.run}',targetmesh,targetscalp,tgt,description,a.dpi,a.surface_step,'Provisional RAS + registration assumptions. Distances are 3D target-native mm; not inferred from image projections.')
    # Transform target reference points into donor native coordinates (for
    # visualization ONLY). Target MRI -> donor MRI follows ANTs image-resampling
    # terms [forward warp, affine], validated by Script 20.
    exe=get_exe('antsApplyTransformsToPoints',a.micromamba_env)
    terms=audit['point_mapping_validation']['target_to_donor_terms_as_passed_to_antsApplyTransformsToPoints']
    terms=[str(repo/t) if not Path(t).is_absolute() else t for t in terms]
    for term in terms:
        if not Path(term).exists(): raise FileNotFoundError(f'SyN transform missing: {term}')
    t_h5=h5(data,a.target); d_h5=h5(data,a.donor)
    reference_names=['Ground truth','Within-subject SESAME']
    t1w=np.vstack([transform(tgt[n],np.linalg.inv(t_h5)) for n in reference_names])
    mapped=ants_points(exe,t1w,terms)
    donorpts={n:transform(p,d_h5) for n,p in zip(reference_names,mapped)}
    donorpts['Cross-subject SyN']=np.asarray(coords['donor_surface'],float)
    # The donor source is the same raw SESAME estimate for both affine and SyN;
    # it should never be plotted as two distinct donor-native points.
    dmesh,dscalp=native_surfaces(data,a.donor)
    render(out/'02_donor_native_three_points.png',f'Donor native cortex — {a.donor}',dmesh,dscalp,donorpts,
           {'Ground truth':'Target ground truth mapped to donor (SyN)',
            'Within-subject SESAME':'Target within-subject estimate mapped to donor (SyN)',
            'Cross-subject SyN':'Cross-subject SESAME — donor native (same source for affine/SyN)'},
           a.dpi,a.surface_step,'Donor view is illustrative only. Donor-space separations are not the target-native localization errors. Mapped reference points depend on SyN anatomical correspondence.')
    mni_status='NOT GENERATED: independently verified MNI source transforms / template surfaces not provided.'
    mniflags=(a.mni_points_json,a.mni_left_gii,a.mni_right_gii)
    if any(v is not None for v in mniflags):
        if not all(v is not None for v in mniflags): raise ValueError('MNI mode requires --mni-points-json, --mni-left-gii and --mni-right-gii together')
        mni=json.loads(a.mni_points_json.read_text())
        # Must be a validated external mapping for EVERY relevant point.
        if mni.get('validation_status') != 'independently_verified':
            raise ValueError('MNI mapping must declare validation_status=independently_verified')
        if mni.get('coordinate_frame') != 'template_surface_RAS_mm':
            raise ValueError('MNI coordinates must explicitly use template_surface_RAS_mm')
        pts={name:np.asarray(mni['points'][name],float) for name in tgt}
        if any(p.shape!=(3,) or not np.isfinite(p).all() for p in pts.values()): raise ValueError('Invalid MNI points')
        meshes=[(surf(a.mni_left_gii),'#dedee3'),(surf(a.mni_right_gii),'#dedee3')]
        render(out/'03_mni_verified.png','Template cortex — independently verified mapping',meshes,None,pts,
               description,a.dpi,a.surface_step,'Template points are illustrative; target-native errors shown in legend are not MNI-space distances. Review separate template-registration validation.')
        mni_status='GENERATED from user-provided, explicitly verified template coordinates and pial meshes.'
    report={'status':'PROVISIONAL_TARGET_AND_DONOR_VISUALIZATION','source_audit':str(auditfile),
            'target_subject':a.target,'donor_subject':a.donor,'run':a.run,
            'target_native_points_mm':{k:v.tolist() for k,v in tgt.items()},
            'donor_native_points_mm':{k:v.tolist() for k,v in donorpts.items()},
            'target_native_error_mm':errors,
            'mni_status':mni_status,
            'limitations':['Dataset T1w coordinates provisionally assumed NIfTI RAS',
                           'SyN point directions inherited from Script 20 validation',
                           'No template transform is invented or fitted using ground truth',
                           'Donor-side distances are visualization-only']}
    (out/'visualization_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved target and donor three-dimensional anatomical figures to',out)
    print('MNI:',mni_status)
    for name,err in errors.items(): print(f'{name}: {err:.2f} mm (target native; provisional)')

if __name__=='__main__': main()
