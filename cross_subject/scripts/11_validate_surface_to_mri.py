#!/usr/bin/env python3
"""Diagnostic Localize-MI GIFTI-surface ↔ T1w-MRI overlay (NO registration).

Run from repository root with the original research .venv:
 python cross_subject/scripts/11_validate_surface_to_mri.py --target sub-02 --donor sub-01

Generates candidate-coordinate overlays per subject, not validated transforms.
The three candidates are identity, GIFTI POINTSET metadata matrix, inverse matrix.
Explicit coordinate conventions: GIFTI native values in mm; NIfTI world RAS+ mm.
The presence of a GIFTI coordsys matrix does not imply it maps to NIfTI space.
Check the generated images before choosing ANY transformation. No localization
errors are calculated; nothing is fed into the inverse solver.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


def native_mesh(path):
    im = nib.load(str(path))
    vertices = np.asarray(im.darrays[0].data, dtype=float)
    if not np.isfinite(vertices).all():
        raise ValueError(f'Invalid non-finite mesh vertices: {path}')
    # Localize-MI meshes are stored in mm. Explicitly reject unexpected units.
    if not (20 < np.percentile(np.linalg.norm(vertices, axis=1), 90) < 250):
        raise ValueError(f'Unexpected mesh vertex coordinate size (expected mm): {path}')
    coordsys = im.darrays[0].coordsys
    matrix = np.asarray(coordsys.xform, dtype=float) if coordsys is not None else None
    info = {'path':str(path), 'vertex_count':int(len(vertices)),
            'coordinate_system_dataspace':str(getattr(coordsys,'dataspace',None)),
            'coordinate_system_xformspace':str(getattr(coordsys,'xformspace',None)),
            'coordinate_system_matrix':matrix.tolist() if matrix is not None else None}
    return vertices, matrix, info


def affine_points(points, matrix):
    return np.asarray(points) @ matrix[:3,:3].T + matrix[:3,3]


def points_ijk(points_mm, im):
    return nib.affines.apply_affine(np.linalg.inv(im.affine), points_mm)


def summarize_candidate(im, data, vertices, n=12000):
    pts = vertices[::max(1,len(vertices)//n)]
    ijk = points_ijk(pts, im)
    inside = np.all((ijk>=0)&(ijk < np.array(im.shape[:3])-1),axis=1)
    finite = np.isfinite(ijk).all(1)
    inside &= finite
    out={'samples':int(len(pts)), 'inside_mri_fov_fraction':float(inside.mean()),
         'vertices_bounds_mm':[pts.min(0).tolist(),pts.max(0).tolist()]}
    if inside.any():
        # MRI intensity at surface vertices is only a rough alignment diagnostic.
        intensity=map_coordinates(data, ijk[inside].T, order=1, mode='constant', cval=0.0)
        out['sampled_mri_intensity_median']=float(np.median(intensity))
        out['sampled_mri_positive_fraction']=float(np.mean(intensity>0))
    return out


def project_views(img, vertices, source_points, out, title):
    # Reorient to canonical RAS axes so slice panels always have consistent views.
    can = nib.as_closest_canonical(img)
    data=np.asarray(can.dataobj, dtype=np.float32)
    if data.ndim != 3: raise ValueError('Expected a 3-D MRI')
    mat=np.linalg.inv(can.affine)
    pts=nib.affines.apply_affine(mat, vertices)
    values=np.asarray(list(source_points.values()),float) if source_points else np.empty((0,3))
    pred=nib.affines.apply_affine(mat,values) if len(values) else np.empty((0,3))
    colors={'Real source':'#159447','Within-subject SESAME':'#2365b3',
            'Cross-subject SESAME':'#7932b5'}
    fig, axes = plt.subplots(1,3,figsize=(16,5),facecolor='white')
    axes_map=[(0,1,2,'Sagittal (Y–Z)'), (1,0,2,'Coronal (X–Z)'), (2,0,1,'Axial (X–Y)')]
    for ax,(sliceaxis,xaxis,yaxis,name) in zip(axes,axes_map):
        # Choose slice passing through source, when available, otherwise cortical centroid.
        z=float(pred[0,sliceaxis]) if len(pred) else float(np.median(pts[:,sliceaxis]))
        index=int(np.clip(round(z),0,data.shape[sliceaxis]-1))
        if sliceaxis==0: sl=data[index,:,:].T
        elif sliceaxis==1: sl=data[:,index,:].T
        else: sl=data[:,:,index].T
        vmax=float(np.nanpercentile(sl[sl>0],99.5)) if np.any(sl>0) else 1
        ax.imshow(sl,origin='lower',cmap='gray',vmin=0,vmax=vmax,interpolation='nearest')
        sel=np.abs(pts[:,sliceaxis]-index)<2.0
        nsel=int(sel.sum())
        if nsel>12000:
            cand=np.flatnonzero(sel)[::int(np.ceil(nsel/12000))]
            coords=pts[cand]
        else: coords=pts[sel]
        ax.scatter(coords[:,xaxis],coords[:,yaxis],s=1.2,c='#05e1da',alpha=.55,rasterized=True)
        for i,(label,_) in enumerate(source_points.items()):
            q=pred[i]
            # Always plot projected point; tell reader whether it is on this slice.
            off=abs(q[sliceaxis]-index)
            ax.scatter(q[xaxis],q[yaxis],s=90,marker='x',c=colors.get(label,'#ffce55'),
                       linewidths=2.5,alpha=1 if off<=5 else .35,label=f'{label} ({off:.0f} vox from slice)')
        ax.set_xlim(0,sl.shape[1]);ax.set_ylim(0,sl.shape[0]);ax.set_aspect('equal')
        ax.set_title(f'{name} | index {index}',fontsize=11)
        ax.set_xlabel('voxel (canonical RAS)');ax.set_ylabel('voxel (canonical RAS)')
        if source_points: ax.legend(fontsize=7,loc='lower left',framealpha=.9)
    fig.suptitle(title+'\nCyan = pial surface vertices near slice; markers = projected source points',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(out,dpi=150,bbox_inches='tight');plt.close(fig)


def read_source_points(summary, subject, target, donor):
    if not summary or not summary.is_file(): return {}
    s=json.loads(summary.read_text())
    if s.get('target')!=target or s.get('donor')!=donor:
        raise ValueError('Comparison summary subject identifiers do not match --target and --donor')
    points={}
    if subject==target:
        for name, keys in [('Real source',['real_native_m']),
                           ('Within-subject SESAME',['within_subject_native_m','previous_native_m'])]:
            for k in keys:
                if k in s:
                    points[name]=np.asarray(s[k],float)*1000
                    break
    if subject==donor:
        for k in ['cross_subject_donor_native_m','new_donor_native_m']:
            if k in s:
                points['Cross-subject SESAME']=np.asarray(s[k],float)*1000
                break
    return points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',default='sub-02');p.add_argument('--donor',default='sub-01')
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--comparison-summary',type=Path,default=Path('cross_subject/outputs/07_sesame_comparison/sub-02_with_sub-01_run-01/comparison_summary.json'))
    p.add_argument('--output-dir',type=Path,default=Path('cross_subject/outputs/11_surface_mri_validation'))
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'target':args.target,'donor':args.donor,'purpose':'DIAGNOSTIC ONLY: no validated registration and no distance calculated','subjects':{}}
    for subject in (args.target,args.donor):
        path=args.dataset/'derivatives'/'epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
        if not path.exists(): raise FileNotFoundError(path)
        im=nib.load(str(path)); data=np.asarray(im.dataobj,dtype=np.float32)
        anat=args.dataset/'derivatives'/'sourcemodelling'/subject/'anat'
        meshes=[]; matrices=[]; gifti=[]
        for hemi in ('L','R'):
            points, mat, info=native_mesh(anat/f'{subject}_hemi-{hemi}_pial.surf.gii')
            meshes.append(points);matrices.append(mat);gifti.append(info)
        verts=np.vstack(meshes)
        mat=matrices[0]
        if mat is not None and matrices[1] is not None and not np.allclose(mat,matrices[1]):
            raise ValueError('Left and right GIFTI coordinate metadata differ; handle separately')
        # These candidates are exploratory. GIFTI metadata can refer to a different
        # target space; it is not automatically the MRI RAS transform.
        candidates={'identity_raw_GIFTI_vertices':np.eye(4)}
        if mat is not None and np.isfinite(mat).all() and np.linalg.matrix_rank(mat)==4:
            candidates['forward_GIFTI_metadata_matrix']=mat
            candidates['inverse_GIFTI_metadata_matrix']=np.linalg.inv(mat)
        subject_out={'mri':str(path),'mri_affine':np.asarray(im.affine,float).tolist(),'gifti':gifti,'candidates':{}}
        source_points=read_source_points(args.comparison_summary,subject,args.target,args.donor)
        for label,transform in candidates.items():
            vertices_mm=affine_points(verts,transform)
            transformed_sources={k:affine_points(np.asarray(v)[None,:],transform)[0] for k,v in source_points.items()}
            diagnostic=summarize_candidate(im,data,vertices_mm)
            diagnostic['matrix_applied_to_native_mm']=transform.tolist()
            output=args.output_dir/f'{subject}_{label}_mri_overlay.png'
            # Decimation does not affect underlying point coordinates or QA report.
            sample=vertices_mm[::max(1,len(vertices_mm)//50000)]
            project_views(im,sample,transformed_sources,output,
                          f'{subject} | {label} | candidate only — NOT VALIDATED')
            diagnostic['figure']=str(output)
            subject_out['candidates'][label]=diagnostic
            print(f'{subject} {label}: inside MRI FOV={diagnostic["inside_mri_fov_fraction"]:.1%} -> {output}',flush=True)
        report['subjects'][subject]=subject_out
    report_path=args.output_dir/f'{args.donor}_to_{args.target}_surface_mri_candidates.json'
    report_path.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('QA report:',report_path)
    print('IMPORTANT: candidate overlays are not evidence of validated anatomical correspondence.')

if __name__=='__main__': main()
