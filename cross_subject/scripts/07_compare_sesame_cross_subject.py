#!/usr/bin/env python3
"""07_compare_sesame_cross_subject.py — matched-case SESAME comparison.

EXAMPLES (run from repository root; activate .venv first):

One existing cross-subject pilot (target sub-02, donor sub-01):
 python cross_subject/scripts/07_compare_sesame_cross_subject.py \
   --case sub-02 run-01 --donor sub-01 \
   --previous-results outputs/23_full_next_method_grid/sesame/grid_results.csv

Representative cases from the earlier paper (needs a cross-subject Script-05
pilot for EACH target/run; sub-01 with donor sub-01 is not cross-subject):
 python cross_subject/scripts/07_compare_sesame_cross_subject.py \
   --case sub-01 run-01 \
   --case sub-05 run-06 \
   --case sub-07 run-05 \
   --case sub-07 run-07 \
   --donor sub-02 \
   --previous-results outputs/23_full_next_method_grid/sesame/grid_results.csv

Exploratory 3-point MNI figure using RIGID correspondence of native-surface and MNI
SEEG electrode coordinates. IMPORTANT: this rigid fit is NOT a validated
MRI-registration solution; reported distances are exploratory, not a final
localization-accuracy result. Use only for diagnostic visualization:
 python cross_subject/scripts/07_compare_sesame_cross_subject.py \
   --case sub-02 run-01 --donor sub-01 \
   --previous-results outputs/23_full_next_method_grid/sesame/grid_results.csv \
   --dataset "$HOME/Documents/DATASETS SOURCE LOCALIZATION/Localize-MI" \
   --mapping exploratory-landmark-rigid \
   --mni-template /ACTUAL/PATH/MNI152NLin2009aSym_T1w.nii.gz

Required for native figures: existing Script-05 cross-subject JSON and
previous same-parameter SESAME Script-23 result. For MNI diagnostic: matching
surface/MNI coordinate electrode tables of both subjects. The MNI template
MUST represent the exact MNI152NLin2009aSym space; pass its real NIfTI path.

Do not use Subject-02 MRI/forward for the new cross-subject reconstruction.
Target ground-truth coordinates are read here AFTER reconstruction only.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', nargs=2, metavar=('SUBJECT','RUN'), action='append', help='Repeat for multiple target runs')
    p.add_argument('--target', help='Backward-compatible single target')
    p.add_argument('--run', help='Backward-compatible single run')
    p.add_argument('--donor', required=True, help='Single donor subject applied to each case')
    p.add_argument('--previous-results', type=Path, default=Path('outputs/23_full_next_method_grid/sesame/grid_results.csv'))
    p.add_argument('--run-geometry', type=Path, default=Path('outputs/anatomical_inspection/run_anatomical_geometry.csv'))
    p.add_argument('--dataset', type=Path, default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--cross-root', type=Path, default=Path('cross_subject/outputs/05_sesame_pilot'))
    p.add_argument('--cross-json', type=Path, help='Single-case Script-05 JSON override')
    p.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/07_sesame_comparison'))
    p.add_argument('--mapping', choices=['none','exploratory-landmark-rigid','exploratory-landmark-affine'], default='none',
                   help='Never equate exploratory landmark affine with validated brain registration')
    p.add_argument('--mni-template', type=Path, help='Actual matching MNI152NLin2009aSym template NIfTI')
    p.add_argument('--min-landmarks', type=int, default=6,
                   help='Minimum paired landmarks for exploratory rigid fit (default 6; 3-D rank and holdout checks still required)')
    p.add_argument('--max-heldout-error-mm', type=float, default=15.,
                   help='Refuse exploratory mapping if cross-validated median error exceeds threshold')
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def norm_run(run):
    return 'run-'+str(run).removeprefix('run-').zfill(2)


def dist_mm(a,b): return float(np.linalg.norm(np.asarray(a)-np.asarray(b))*1000)


def xyz(row,columns,scale=1.): return np.array([float(row[c]) for c in columns],float)*scale


def serial(obj):
    if isinstance(obj,(np.integer,np.floating)): return obj.item()
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,Path): return str(obj)
    raise TypeError(type(obj).__name__)


def savejson(path,value):
    path.write_text(json.dumps(value,indent=2,default=serial),encoding='utf8')


def previous_case(df,target,run,params):
    need=['subject','run','method','status','selected_x_mm','selected_y_mm','selected_z_mm',
          'ground_truth_x_mm','ground_truth_y_mm','ground_truth_z_mm','localization_distance_mm']
    missing=[c for c in need if c not in df.columns]
    if missing: raise ValueError(f'Previous SESAME CSV missing: {missing}')
    s=df.loc[(df.subject.astype(str)==target)&(df.run.astype(str)==run)&
             (df.method.astype(str).str.lower()=='sesame')&
             (df.status.astype(str).str.upper()=='PASS')].copy()
    expected={'sesame_fs':params['fs'],'sesame_n_parts':params['n_parts'],
              'sesame_max_n_dips':params['max_n_dips'],'sesame_lam':params['lam'],
              'sesame_random_seed':params['seed']}
    for name,value in expected.items():
        if name not in s: raise ValueError(f'Cannot match previous parameter {name}: column absent')
        s=s.loc[np.isclose(pd.to_numeric(s[name],errors='coerce'),float(value),equal_nan=False)]
    if 'montage' in s: s=s.loc[s.montage.astype(str).str.lower().isin(['all_good','all-good','all good'])]
    if 'sesame_hyper_q' in s:
        expected_bool=bool(params['hyper_q'])
        s=s.loc[s.sesame_hyper_q.astype(str).str.lower().isin(['true','1','yes'])==expected_bool]
    if len(s)!=1:
        raise ValueError(f'Expected exactly ONE matched previous SESAME result for {target} {run}; got {len(s)}. '
                         'Do not select a minimum-error run for a paired comparison.')
    return s.iloc[0]


def fit_rigid(x, y):
    """Least-squares rigid fit (metres, row-vectors), no reflection or scale."""
    xm, ym = np.mean(x, axis=0), np.mean(y, axis=0)
    u, singular, vt = np.linalg.svd((x - xm).T @ (y - ym))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    transform = np.vstack([rotation, ym - xm @ rotation])
    if not np.all(np.isfinite(transform)) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('Invalid rigid rotation fit')
    return transform


def electrode_pairs(root,sub,min_landmarks,max_heldout):
    folder=root/'derivatives'/'epochs'/sub/'ieeg'
    a=folder/f'{sub}_task-seegstim_space-surface_electrodes.tsv'
    b=folder/f'{sub}_task-seegstim_space-MNI152NLin2009aSym_electrodes.tsv'
    if not a.exists() or not b.exists(): raise FileNotFoundError(f'Missing corresponding surface/MNI electrodes: {a}, {b}')
    an=pd.read_csv(a,sep='\t'); bn=pd.read_csv(b,sep='\t')
    for table in (an,bn):
        if not {'name','x','y','z'}.issubset(table.columns): raise ValueError('Electrode TSV lacks name/x/y/z')
    join=an[['name','x','y','z']].merge(bn[['name','x','y','z']],on='name',suffixes=('_surface','_mni'))
    join=join.replace([np.inf,-np.inf],np.nan).dropna()
    x=join[['x_surface','y_surface','z_surface']].to_numpy(float)
    y=join[['x_mni','y_mni','z_mni']].to_numpy(float)
    if len(x)<min_landmarks: raise ValueError(f'Only {len(x)} matched landmarks; need at least {min_landmarks}. An exploratory affine cannot be fitted reliably.')
    # Files must use metres, as the older Script 07 used. Do NOT silently rescale unknown units.
    if np.max(np.abs(x))>1 or np.max(np.abs(y))>1: raise ValueError('Unexpected electrode units; expected metres')
    X=np.column_stack([x,np.ones(len(x))])
    if np.linalg.matrix_rank(X) < 4:
        raise ValueError('Electrode landmarks lack full 3-D coverage')
    fit = fit_rigid(x, y)
    pred = X @ fit
    residual = np.linalg.norm(pred-y,axis=1)*1000
    loo=[]
    for i in range(len(x)):
        xi=np.delete(x,i,axis=0); yi=np.delete(y,i,axis=0)
        if np.linalg.matrix_rank(np.column_stack([xi,np.ones(len(xi))])) < 4:
            raise ValueError('Degenerate held-out rigid mapping')
        fitted=fit_rigid(xi,yi)
        loo.append(float(np.linalg.norm(np.r_[x[i],1]@fitted-y[i])*1000))
    loo=np.asarray(loo)
    quality={'subject':sub,'model':'EXPLORATORY rigid rotation + translation from SEEG landmarks',
             'landmark_count':len(x),'fit_median_mm':float(np.median(residual)),
             'sparse_landmark_warning': ('Only 6 landmarks; cortical extrapolation may be unreliable' if len(x)<=6 else None),
             'heldout_median_mm':float(np.median(loo)),
             'heldout_p95_mm':float(np.percentile(loo,95)),
             'heldout_max_mm':float(np.max(loo)),
             'max_allowed_heldout_median_mm':max_heldout,
             'validated_whole_brain':False,
             'caution':'This rigid fit preserves brain shape but is NOT a validated MRI-to-MNI registration; localization distances remain exploratory.',
             'native_path':str(a),'mni_path':str(b)}
    if np.median(loo)>max_heldout:
        raise ValueError(f'{sub}: exploratory rigid fit heldout median {np.median(loo):.2f} mm exceeds {max_heldout:.2f} mm')
    return fit,quality



def ground_truth_mni_from_electrodes(dataset, target, prevrow, native_real):
    """Recover MNI stimulation midpoint using the exact native-ground-truth pair.

    The previous SESAME CSV gives ground_truth_*_mm in native surface space,
    but not the MNI coordinates. Match that known native midpoint to one unique
    pair in the subject's native-surface electrode file, then use precisely the
    same two electrode names in the MNI electrode file. This is *evaluation*
    only, and never enters the reconstruction.
    """
    folder = dataset / 'derivatives' / 'epochs' / target / 'ieeg'
    native_path = folder / f'{target}_task-seegstim_space-surface_electrodes.tsv'
    mni_path = folder / f'{target}_task-seegstim_space-MNI152NLin2009aSym_electrodes.tsv'
    for path in (native_path, mni_path):
        if not path.is_file():
            raise FileNotFoundError(f'Ground-truth electrode file not found: {path}')
    native = pd.read_csv(native_path, sep='\t')
    mni = pd.read_csv(mni_path, sep='\t')
    required = {'name', 'x', 'y', 'z'}
    for tab, path in ((native, native_path), (mni, mni_path)):
        if not required.issubset(tab.columns):
            raise ValueError(f'{path} missing {sorted(required - set(tab.columns))}')
        if tab['name'].duplicated().any():
            raise ValueError(f'Duplicate electrode names in {path}')
    names = native['name'].astype(str).tolist()
    coords = native[['x', 'y', 'z']].to_numpy(float)
    if len(coords) < 2:
        raise ValueError('Fewer than two native stimulation contacts')
    # The native-space known point provides a strong, independent matching
    # check; do not guess from ambiguous punctuation in stimulation_pair.
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = float(np.linalg.norm((coords[i] + coords[j]) / 2 - native_real) * 1000)
            if d <= 0.05:  # previous Script 07 uses 0.05 mm agreement
                pairs.append((names[i], names[j], d))
    if len(pairs) != 1:
        label = str(prevrow.get('stimulation_pair', 'unavailable'))
        raise ValueError(
            f'Cannot unambiguously recover stimulation pair {label!r} '
            f'from native midpoint; {len(pairs)} candidate pairs within 0.05 mm. '
            'Supply an existing run_anatomical_geometry.csv instead.'
        )
    first, second, mismatch_mm = pairs[0]
    indexed = mni.set_index('name')
    if first not in indexed.index or second not in indexed.index:
        raise ValueError(f'Matched contacts {first}, {second} missing from {mni_path}')
    mni_xyz = indexed.loc[[first, second], ['x','y','z']].astype(float).mean().to_numpy()
    if not np.isfinite(mni_xyz).all() or np.max(np.abs(mni_xyz)) > 1:
        raise ValueError('Unexpected MNI coordinate units or non-finite values')
    return mni_xyz, {'source':'electrode_midpoint_fallback',
                     'contacts':[first,second], 'native_midpoint_agreement_mm':mismatch_mm,
                     'native_electrodes':str(native_path),'mni_electrodes':str(mni_path)}

def apply_affine(coord,fit): return np.r_[coord,1.]@fit




def load_gifti_mesh(dataset, subject):
    """Load combined left/right pial mesh for a subject in native surface space."""
    anat = dataset/'derivatives'/'sourcemodelling'/subject/'anat'
    left = anat/f'{subject}_hemi-L_pial.surf.gii'
    right = anat/f'{subject}_hemi-R_pial.surf.gii'
    for path in (left, right):
        if not path.is_file():
            raise FileNotFoundError(f'Brain surface not found: {path}')
    def read_one(path):
        g = nib.load(str(path))
        verts = np.asarray(g.darrays[0].data, float)
        faces = np.asarray(g.darrays[1].data, int)
        if np.max(np.abs(verts)) > 1:  # mm -> m
            verts = verts / 1000.0
        return verts, faces
    lv, lf = read_one(left)
    rv, rf = read_one(right)
    faces = np.vstack([lf, rf + len(lv)])
    verts = np.vstack([lv, rv])
    return verts, faces


def maybe_transform_vertices(vertices, fit):
    if fit is None:
        return vertices
    X = np.column_stack([vertices, np.ones(len(vertices))])
    return X @ fit


def plot_brain_mesh(ax, vertices, faces, sample_step=6, alpha=0.085, color='gray'):
    if vertices is None or faces is None:
        return
    faces = faces[::max(1, sample_step)]
    ax.plot_trisurf(vertices[:,0], vertices[:,1], faces, vertices[:,2],
                    color=color, alpha=alpha, linewidth=0.05, edgecolor='none', shade=False)


def balanced_limits(points, margin_ratio=0.08, min_half=0.035):
    centre = points.mean(0)
    half = max(np.ptp(points, axis=0).max() / 2, min_half) * (1 + margin_ratio)
    return centre, half

def plot_native(output,real,prev,cross,sub,donor,run,which,dpi,brain_vertices=None,brain_faces=None):
    fig=plt.figure(figsize=(9,7.5)); ax=fig.add_subplot(111,projection='3d')
    if brain_vertices is not None and brain_faces is not None:
        plot_brain_mesh(ax, brain_vertices, brain_faces, sample_step=6, alpha=0.08, color='grey')
    if which=='target':
        points=[(real,'Real source','#168a42'),(prev,'Within-subject SESAME','#2464b2')]
        ax.plot([real[0],prev[0]],[real[1],prev[1]],[real[2],prev[2]],ls='--',c='#2464b2',
                label=f'Within-subject error (native): {dist_mm(real,prev):.2f} mm')
        title=f'Target {sub} native surface | {run}'
    else:
        points=[(cross,'Cross-subject SESAME','#7432b0')]
        title=f'Donor {donor} native surface | target {sub} | {run}'
    for pt,name,col in points:
        ax.scatter(*pt,s=110,c=col,label=name,depthshade=False)
        ax.text(*pt,' '+name,fontsize=9,color=col)
    coordinate_parts = [np.asarray(p[0], dtype=float).reshape(1, 3) for p in points]
    if brain_vertices is not None and len(brain_vertices):
        step = max(1, len(brain_vertices) // 2500)
        coordinate_parts.append(np.asarray(brain_vertices[::step], dtype=float))
    coords = np.vstack(coordinate_parts)
    center,half = balanced_limits(coords if len(coords)>1 else np.vstack([coords, coords+0.01]), min_half=0.025)
    ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
           zlim=(center[2]-half,center[2]+half),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    if which=='target':
        title += f'\nOfficial within-subject distance: {dist_mm(real,prev):.2f} mm'
    else:
        title += '\nCoordinate '+', '.join(f'{v*1000:.2f}' for v in cross)+' mm'
    ax.set_title(title);ax.legend(fontsize=9,loc='upper left');ax.view_init(elev=25,azim=115)
    fig.tight_layout();fig.savefig(output,dpi=dpi,bbox_inches='tight');plt.close(fig)


def template_cloud(template):
    import nibabel as nib
    im=nib.load(str(template)); d=im.get_fdata(dtype=np.float32)
    if not np.isfinite(d).any() or np.nanmax(d)<=0: raise ValueError('Template image is empty')
    # Generate outline of bright tissue; this is a visualization, not a diagnostic segmentation.
    mask=np.isfinite(d)&(d>np.nanpercentile(d[d>0],45))
    surf=mask & ~binary_erosion(mask)
    vox=np.argwhere(surf)
    if len(vox)>15000: vox=vox[::math.ceil(len(vox)/15000)]
    world=nib.affines.apply_affine(im.affine,vox)
    if np.max(np.abs(world))>10: world/=1000
    return world


def plot_mni(output,real,prev,cross,cloud,sub,donor,run,dpi,
             brain_vertices=None,brain_faces=None,official_native_within_mm=None):
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(12.5,9.5))
    ax=fig.add_subplot(111,projection='3d')
    if brain_vertices is not None and brain_faces is not None:
        plot_brain_mesh(ax, brain_vertices, brain_faces, sample_step=5, alpha=0.055, color='grey')
    elif cloud is not None:
        ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],color='grey',s=.6,alpha=.05)

    points=[(real,'Real source','#168a42'),
            (prev,'Within-subject SESAME','#2464b2'),
            (cross,'Cross-subject SESAME','#7432b0')]
    for pt,name,col in points:
        ax.scatter(*pt,s=145,c=col,depthshade=False,zorder=10)
    # One label on the left, one on the right, one above — avoid stacking text.
    ax.text(real[0]-0.025,real[1],real[2]+0.007,'Real source',color='#168a42',fontsize=10,ha='right')
    ax.text(prev[0]+0.012,prev[1],prev[2]-0.008,'Within-subject SESAME',color='#2464b2',fontsize=10,ha='left')
    ax.text(cross[0]+0.012,cross[1],cross[2]+0.014,'Cross-subject SESAME',color='#7432b0',fontsize=10,ha='left')

    cross_err=dist_mm(real,cross)
    shift=dist_mm(prev,cross)
    # Spatial lines in exploratory MNI coordinates. Original/native within-subject
    # error is reported in legend, not incorrectly identified with the mapped line.
    for a,b,col,style in ((real,prev,'#2464b2','--'),
                          (real,cross,'#7432b0','--'),
                          (prev,cross,'#c44a30',':')):
        ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],c=col,ls=style,lw=2)

    allparts=[np.vstack([real,prev,cross])]
    if brain_vertices is not None:
        allparts.append(brain_vertices[::max(1,len(brain_vertices)//3000)])
    elif cloud is not None:
        allparts.append(cloud[::max(1,len(cloud)//2000)])
    allpoints=np.vstack(allparts)
    centre,half=balanced_limits(allpoints,min_half=.09)
    if half > .25:
        raise ValueError(f'MNI brain backdrop bounding radius {half*1000:.1f} mm is implausibly large; check native mesh coordinate units and landmark mapping')
    ax.set(xlim=(centre[0]-half,centre[0]+half),
           ylim=(centre[1]-half,centre[1]+half),
           zlim=(centre[2]-half,centre[2]+half),
           xlabel='MNI X (mm)',ylabel='MNI Y (mm)',zlabel='MNI Z (mm)')
    # Coordinate values are metres internally. Relabel axes in mm for readability.
    from matplotlib.ticker import FuncFormatter
    fmter=FuncFormatter(lambda v,pos: f'{v*1000:.0f}')
    ax.xaxis.set_major_formatter(fmter)
    ax.yaxis.set_major_formatter(fmter)
    ax.zaxis.set_major_formatter(fmter)
    native_error=float(official_native_within_mm) if official_native_within_mm is not None else float('nan')
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markersize=9,label=n)
             for _,n,c in points]
    handles += [
        Line2D([0],[0],color='#2464b2',ls='--',lw=2,
               label=f'Within-subject error (ORIGINAL native): {native_error:.2f} mm'),
        Line2D([0],[0],color='#7432b0',ls='--',lw=2,
               label=f'Real → cross-subject (EXPLORATORY MNI): {cross_err:.2f} mm'),
        Line2D([0],[0],color='#c44a30',ls=':',lw=2,
               label=f'Within → cross shift (EXPLORATORY MNI): {shift:.2f} mm'),
    ]
    ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.5,-.20),
              fontsize=9,framealpha=.95)
    ax.set_title('EXPLORATORY rigid-landmark MNI comparison — NOT VALIDATED MRI REGISTRATION\n'
                 f'Target {sub} | Donor {donor} | {run}\n'
                 'Original native within-subject error retained; mapped cross-subject error is exploratory',fontsize=11,pad=15)
    ax.view_init(elev=22,azim=118)
    fig.subplots_adjust(top=.86,bottom=.23,left=.02,right=.98)
    fig.savefig(output,dpi=dpi,bbox_inches='tight')
    plt.close(fig)

def main():
    args=arguments()
    if args.mapping=='exploratory-landmark-affine':
        print('NOTICE: deprecated --mapping exploratory-landmark-affine now uses exploratory RIGID fitting to avoid cortical mesh distortion. Use --mapping exploratory-landmark-rigid instead.')
    cases=[(a,norm_run(b)) for a,b in (args.case or [])]
    if args.target or args.run:
        if not args.target or not args.run: raise ValueError('Both --target and --run needed')
        cases.append((args.target,norm_run(args.run)))
    if not cases: raise ValueError('Provide --case SUBJECT RUN (repeatable), or --target and --run')
    cases=list(dict.fromkeys(cases))
    previous=pd.read_csv(args.previous_results)
    geometry=pd.read_csv(args.run_geometry) if args.run_geometry.exists() else None
    mapping_cache={}; results=[]
    cloud=None
    if args.mni_template is not None:
        if not args.mni_template.is_file(): raise FileNotFoundError(f'MNI template file not found: {args.mni_template}')
        cloud=template_cloud(args.mni_template)
    mesh_cache={} 
    for target,run in cases:
        out=args.output_dir/f'{target}_with_{args.donor}_{run}';out.mkdir(parents=True,exist_ok=True)
        record={'target':target,'donor':args.donor,'run':run,'status':'INCOMPLETE','figures':{}}
        try:
            if target==args.donor: raise ValueError('Target equals donor, so this is not a cross-subject case. Select another donor.')
            cross_path=args.cross_json or args.cross_root/f'{target}_with_{args.donor}_{run}'/'sesame_pilot_result.json'
            if not cross_path.exists(): raise FileNotFoundError(f'Run Script 05 first for this target/donor/run: {cross_path}')
            cross=json.loads(cross_path.read_text());
            if str(cross.get('status')).upper()!='PASS': raise ValueError('Cross-subject SESAME pilot is not PASS')
            params=cross['parameters'];prevrow=previous_case(previous,target,run,params)
            prev=xyz(prevrow,['selected_x_mm','selected_y_mm','selected_z_mm'],.001)
            real=xyz(prevrow,['ground_truth_x_mm','ground_truth_y_mm','ground_truth_z_mm'],.001)
            cross_native=np.asarray(cross['peak_coordinate_donor_src_rr_m'],float)
            # Head→surface transform from the validated Script-06 output, or apply directly from H5.
            import h5py
            xfm=args.dataset/'derivatives'/'sourcemodelling'/args.donor/'xfm'/f'{args.donor}_from-head_to-surface.h5'
            with h5py.File(xfm,'r') as f: mat=np.asarray(f['trans'][()],float)
            if mat.shape!=(4,4): raise ValueError(f'Invalid donor head→surface matrix {xfm}')
            donor_native=(mat@np.r_[cross_native,1.])[:3]
            reported=dist_mm(real,prev)
            if target not in mesh_cache: mesh_cache[target]=load_gifti_mesh(args.dataset,target)
            if args.donor not in mesh_cache: mesh_cache[args.donor]=load_gifti_mesh(args.dataset,args.donor)
            target_mesh=mesh_cache[target]
            donor_mesh=mesh_cache[args.donor]
            if not np.isclose(reported,float(prevrow['localization_distance_mm']),atol=.2):
                raise ValueError('Previous SESAME native distance does not match CSV; coordinate frame mismatch?')
            record.update(status='NATIVE_FIGURES',within_subject_error_native_mm=reported,
                          within_subject_native_m=prev,real_native_m=real,cross_subject_donor_native_m=donor_native,
                          previous_csv=str(args.previous_results),cross_json=str(cross_path),
                          donor_head_to_surface_transform=str(xfm))
            f1=out/'01_target_native_previous_vs_real.png';f2=out/'02_donor_native_new_cross_subject.png'
            plot_native(f1,real,prev,donor_native,target,args.donor,run,'target',args.dpi,*target_mesh)
            plot_native(f2,real,prev,donor_native,target,args.donor,run,'donor',args.dpi,*donor_mesh)
            record['figures'].update(target_native=str(f1),donor_native=str(f2))
            if args.mapping in ('exploratory-landmark-affine','exploratory-landmark-rigid'):
                for sub in (target,args.donor):
                    if sub not in mapping_cache:
                        mapping_cache[sub]=electrode_pairs(args.dataset,sub,args.min_landmarks,args.max_heldout_error_mm)
                target_fit,target_quality=mapping_cache[target];donor_fit,donor_quality=mapping_cache[args.donor]
                if geometry is not None:
                    rows=geometry.loc[(geometry.subject.astype(str)==target)&(geometry.run.astype(str)==run)]
                    if len(rows)!=1:
                        raise ValueError(f'Expected one MNI ground truth row in {args.run_geometry}; found {len(rows)}')
                    gmni=xyz(rows.iloc[0],['mni_x_m','mni_y_m','mni_z_m'])
                    truth_source={'source':'run_anatomical_geometry.csv', 'path':str(args.run_geometry)}
                else:
                    gmni,truth_source=ground_truth_mni_from_electrodes(args.dataset,target,prevrow,real)
                record['mni_ground_truth_provenance']=truth_source
                pmni=apply_affine(prev,target_fit)
                cmni=apply_affine(donor_native,donor_fit)
                mni_distance={'real_to_within_subject_exploratory_mm':dist_mm(gmni,pmni),
                              'real_to_cross_subject_exploratory_mm':dist_mm(gmni,cmni),
                              'within_subject_to_cross_subject_shift_mm':dist_mm(pmni,cmni),
                              'official_within_subject_native_mm': reported}
                mni_mesh_vertices = maybe_transform_vertices(target_mesh[0], target_fit)
                f3=out/'03_mni_three_point_comparison_EXPLORATORY.png'
                plot_mni(f3,gmni,pmni,cmni,cloud,target,args.donor,run,args.dpi,mni_mesh_vertices,target_mesh[1],reported)
                record.update(status='EXPLORATORY_RIGID_MNI_FIGURE',mni_distances_exploratory_mm=mni_distance,
                              native_to_mni_qa={target:target_quality,args.donor:donor_quality},
                              real_mni_m=gmni,within_subject_mni_exploratory_m=pmni,cross_subject_mni_exploratory_m=cmni)
                record['figures']['mni_exploratory']=str(f3)
                record['warning']='MNI results are exploratory estimates from intracranial contact rigid fit, NOT validated MRI-registration localization errors. Keep the official within-subject native error separate from the exploratory mapped distances.'
            else:
                record['warning']='MNI three-point figure not generated: --mapping none; native distances cannot be mixed.'
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'
            print(f'[{target} {run}] {record["error"]}')
        savejson(out/'comparison_summary.json',record)
        results.append(record)
        print(f'[{target} {run}] {record["status"]} | {out}')
    pd.DataFrame([{'target':r['target'],'donor':r['donor'],'run':r['run'],
                   'status':r['status'],'within_subject_error_native_mm':r.get('within_subject_error_native_mm'),
                   'real_to_cross_subject_exploratory_mm':(r.get('mni_distances_exploratory_mm') or {}).get('real_to_cross_subject_exploratory_mm'),
                   'within_to_cross_subject_shift_mm':(r.get('mni_distances_exploratory_mm') or {}).get('within_subject_to_cross_subject_shift_mm'),
                   'error':r.get('error','')} for r in results]).to_csv(args.output_dir/'07_all_cases_summary.csv',index=False)
    print('DONE: ',args.output_dir/'07_all_cases_summary.csv')

if __name__=='__main__': main()
