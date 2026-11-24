#!/usr/bin/env python3
"""Read-only construction/scalp/MRI diagnostics; run manually in a fresh folder."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import warnings as python_warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mne
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.io import loadmat
from geometry import apply, topology, triangle_distance, signed_distance, plane_segments, region

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'localize_mi_average_seven'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def json_write(path, obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),allow_nan=False)+'\n')


def native_bridge(meta):
    """Reconstruct FS volume geometry; scannerRAS @ inverse(tkregisterRAS)."""
    size=np.array([int(meta['VolGeom'+k]) for k in ['Width','Height','Depth']])
    zoom=np.array([float(meta['VolGeom'+k+'size']) for k in 'XYZ'])
    directions=np.array([[float(meta[f'VolGeom{a}_{r}']) for a in 'XYZ'] for r in 'RAS'])
    center=np.array([float(meta[f'VolGeomC_{r}']) for r in 'RAS'])
    scanner=np.eye(4);scanner[:3,:3]=directions@np.diag(zoom)
    scanner[:3,3]=center-scanner[:3,:3]@(size/2)
    tkr=np.array([[-zoom[0],0,0,zoom[0]*size[0]/2],
                  [0,0,zoom[2],-zoom[2]*size[2]/2],
                  [0,-zoom[1],0,zoom[1]*size[1]/2],[0,0,0,1.]])
    return scanner@np.linalg.inv(tkr),scanner,size


def load_mesh(path):
    if path.suffix=='.gii':
        g=nib.load(path);point=next(x for x in g.darrays if x.intent==1008)
        faces=next(x for x in g.darrays if x.intent==1009)
        v,f=np.array(point.data,float),np.array(faces.data,int);meta=dict(point.meta)
        meta['coordsys']=str(point.coordsys)
    elif path.suffix=='.fif':
        # Diagnostic ingestion only: retain defects for our explicit topology audit.
        # This does not accept the mesh for BEM, signing or anatomical use.
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter('always')
            surfaces=mne.read_bem_surfaces(path,on_defects='warn',verbose=False)
        heads=[s for s in surfaces if int(s['id'])==4]
        if len(heads)!=1 or int(heads[0]['coord_frame'])!=5:
            raise ValueError(f'Expected single MRI-frame head surface: {path}')
        v,f=heads[0]['rr']*1000,heads[0]['tris'];meta={'contract':'FIF MRI surface; rr metres converted to mm',
            'reader_warnings':[str(item.message) for item in caught]}
    else:
        v,f,meta=mne.read_surface(path,read_metadata=True)
    if not np.isfinite(v).all() or np.max(np.ptp(v,axis=0))<40 or np.max(np.ptp(v,axis=0))>500:
        raise ValueError(f'Mesh inconsistent with mm contract: {path}')
    if f.min()<0 or f.max()>=len(v):
        raise ValueError(f'Invalid triangle indices: {path}')
    return np.asarray(v,float),np.asarray(f,int),meta


def check_surface_volume_metadata(meta, image, path):
    """Check all available FreeSurfer volume geometry, not only its center."""
    zoom=np.array(image.header.get_zooms()[:3],float)
    expected={'volume':np.array(image.shape[:3]),'voxelsize':zoom,
        'xras':image.affine[:3,0]/zoom[0], 'yras':image.affine[:3,1]/zoom[1],
        'zras':image.affine[:3,2]/zoom[2],
        'cras':apply(image.affine,np.array(image.shape[:3],float)[None,:]/2)[0]}
    checked=[]
    for key,value in expected.items():
        if key in meta:
            if not np.allclose(meta[key],value,atol=1e-3,rtol=0):
                raise ValueError(f'Surface volume geometry mismatch ({key}): {path}')
            checked.append(key)
    return checked


class MRI:
    def __init__(self, path, bridge):
        self.path=path;self.image=nib.load(path)
        self.data=np.asarray(self.image.dataobj,dtype=np.float32)
        if self.data.ndim!=3:raise ValueError(f'Expected scalar MRI: {path}')
        self.bridge=np.asarray(bridge,float)
        self.to_voxel=np.linalg.inv(self.image.affine)@self.bridge
        positive=self.data[np.isfinite(self.data)&(self.data>0)]
        self.vmax=float(np.percentile(positive,99.5)) if len(positive) else 1
    def sample(self, points):
        points=np.asarray(points);vox=apply(self.to_voxel,points.reshape(-1,3))
        valid=((vox>=0)&(vox<=np.array(self.data.shape)-1)).all(1)
        values=map_coordinates(self.data,vox.T,order=1,mode='constant',cval=np.nan,prefilter=False)
        return values.reshape(points.shape[:-1]),valid.reshape(points.shape[:-1])
    def slice(self,axis,position,limits,resolution=1.):
        remaining=[i for i in range(3) if i!=axis]
        x=np.arange(limits[0],limits[1]+resolution/2,resolution)
        y=np.arange(limits[2],limits[3]+resolution/2,resolution)
        xx,yy=np.meshgrid(x,y);p=np.zeros((*xx.shape,3))
        p[:,:,axis]=position;p[:,:,remaining[0]]=xx;p[:,:,remaining[1]]=yy
        values,valid=self.sample(p)
        return values,[x[0],x[-1],y[0],y[-1]]


def draw_slice(ax, mri, meshes, axis, position, limits, title, points=None):
    values,extent=mri.slice(axis,position,limits)
    ax.imshow(values,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=mri.vmax,interpolation='nearest')
    for label,v,f,color in meshes:
        seg=plane_segments(v,f,axis,position)
        if len(seg):ax.add_collection(LineCollection(seg,colors=color,linewidths=1.1,label=label))
    if points:
        remaining=[i for i in range(3) if i!=axis]
        for label,p,q,signed in points:
            # Project full vector onto plane through electrode. Its nearest point
            # may be off-plane: annotate offset explicitly, never imply coplanarity.
            ax.scatter(*q[remaining],s=38,facecolors='none',edgecolors='lime',linewidths=1.4,zorder=4)
            ax.scatter(*p[remaining],s=40,color='magenta',zorder=5)
            ax.annotate('',xy=p[remaining],xytext=q[remaining],arrowprops=dict(arrowstyle='->',color='magenta',lw=1.5))
            distance=np.linalg.norm(p-q)
            sign_text=f'{signed:+.1f} mm' if np.isfinite(signed) else 'sign unresolved'
            ax.text(.02,.02,f'{label}: d={distance:.1f} mm; {sign_text}\nnearest point off-plane {q[axis]-position:+.1f} mm',
                    transform=ax.transAxes,color='white',fontsize=10,bbox=dict(facecolor='black',alpha=.8,pad=4))
    remaining=[i for i in range(3) if i!=axis]
    labels=['X: left → right','Y: posterior → anterior','Z: inferior → superior']
    ax.set(xlim=limits[:2],ylim=limits[2:],xlabel=labels[remaining[0]]+' (mm)',ylabel=labels[remaining[1]]+' (mm)',title=title)
    ax.set_aspect('equal')


def overview(name,mri,meshes,v,out,dpi):
    lo,hi=np.percentile(v,[1,99],axis=0);center=(lo+hi)/2
    fig,axes=plt.subplots(3,3,figsize=(18,17),layout='constrained')
    for row,(axis,label) in enumerate([(2,'Axial'),(1,'Coronal'),(0,'Sagittal')]):
        positions=([center[2]-45,center[2],center[2]+45] if axis==2 else
                   [center[1],hi[1]-40,hi[1]-15] if axis==1 else [lo[0]+15,lo[0]+40,center[0]])
        rem=[i for i in range(3) if i!=axis]
        limits=[lo[rem[0]]-25,hi[rem[0]]+25,lo[rem[1]]-25,hi[rem[1]]+25]
        for ax,position in zip(axes[row],positions):
            draw_slice(ax,mri,meshes,axis,float(position),limits,f'{label}: {"XYZ"[axis]}={position:.1f} mm')
    axes[0,0].legend(loc='upper left',fontsize=9)
    fig.suptitle(f'{name}: scalp contours on MRI; surface RAS mm\nOriginal image sampled trilinearly for display only; no MRI registration or anatomy writes',fontsize=15)
    fig.savefig(out,dpi=dpi);plt.close(fig)


def target_figure(subject,model,channel,p,q,sd,mri,meshes,out,dpi):
    fig,axes=plt.subplots(1,3,figsize=(20,7),layout='constrained')
    for ax,axis,label in zip(axes,[2,1,0],['Axial','Coronal','Sagittal']):
        rem=[i for i in range(3) if i!=axis]
        center=(p+q)/2
        half=max(45,float(np.max(np.abs(p-q))/2+20))
        limits=[center[rem[0]]-half,center[rem[0]]+half,center[rem[1]]-half,center[rem[1]]+half]
        draw_slice(ax,mri,meshes,axis,p[axis],limits,f'{label}: {"XYZ"[axis]}={p[axis]:.1f} mm',[(channel,p,q,sd)])
    axes[0].legend(fontsize=9,loc='upper right')
    fig.suptitle(f'{subject} / {model} / {channel}: electrode (magenta), closest triangle point (green)\nVectors are plane projections; saved nominal fit only; signed result conditional on mesh checks',fontsize=14)
    fig.savefig(out,dpi=dpi);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=EXP/'diagnostics/01_eeg_alignment/run_01')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/Localize-MI')
    parser.add_argument('--construction',type=Path,default=EXP/'outputs/02_template_retry1')
    parser.add_argument('--average-seven',type=Path,default=EXP/'work/freesurfer/average_seven')
    parser.add_argument('--fsaverage',type=Path,default=Path.home()/'mne_data/MNE-fsaverage-data/fsaverage')
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'run_01')
    parser.add_argument('--dpi',type=int,default=220)
    args=parser.parse_args()
    if args.dpi<180:parser.error('--dpi must be at least 180 for readable MRI figures')
    out=args.output_root.resolve()
    if out.exists():raise FileExistsError(f'Refusing existing output: {out}; select a fresh numbered run')
    hashes={}
    def use(path):
        p=Path(path).resolve()
        if str(p) not in hashes:hashes[str(p)]=sha(p)
        return p
    for p in [Path(__file__),Path(__file__).with_name('geometry.py')]:use(p)
    old=json.loads(use(args.previous/'manifest.json').read_text())
    if old['status']!='COMPLETE':raise ValueError('Previous diagnostics must be COMPLETE')
    for p,value in old['input_sha256'].items():
        if sha(p)!=value:raise ValueError(f'Previous diagnostic input changed: {p}')
        hashes[str(Path(p).resolve())]=value
    for p,value in old['output_sha256'].items():
        if sha(args.previous/p)!=value:raise ValueError(f'Previous output changed: {p}')
        use(args.previous/p)
    cfg=json.loads(use(EXP/'config/experiment.json').read_text())
    if cfg['coregistration']['median_limit_mm']!=10 or cfg['coregistration']['p95_limit_mm']!=20 or cfg['coregistration']['scaling']:
        raise ValueError('Frozen acceptance policy changed')
    construction=json.loads(use(args.construction/'04_execution_manifest.json').read_text())
    if construction['status']!='COMPLETE':raise ValueError('Construction not complete')
    for p,value in construction['input_sha256'].items():
        if sha(use(p))!=value:raise ValueError(f'Construction input changed: {p}')
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    manifest=dict(status='RUNNING',created_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        scaling=False,refitting=False,forward=False,inverse=False,anatomy_modified=False,
        alignment_thresholds_mm=dict(median=10,p95=20),anatomy_review='PENDING',coordinate_provenance='UNRESOLVED')
    def save_manifest():
        manifest['input_sha256']=hashes;json_write(out/'manifest.json',manifest)
    save_manifest()
    geometry_rows=[];mri_rows=[];coverage_rows=[];profile_rows=[];alt_rows=[];sign_rows=[];landmarks=[];sections=[]
    transform_rows=[];bridges=[];slice_rows=[];patch_rows=[];warnings=[];mris={};models={};surfaces={};topos={}
    segmentation_evidence=[]
    for p in sorted((args.average_seven/'scripts').glob('recon-all*')):
        if p.is_file() and p.suffix in ['.log','.cmd','.done']:
            use(p)
            for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                if re.search(r'mri_convert|conform|crop|resampl|interpol|/orig/|recon-all -|recon-all.*finished',line,re.I):
                    segmentation_evidence.append(dict(path=str(p.resolve()),line=i,text=line.strip()))
    print('Audit construction provenance, image geometry and transforms',flush=True)
    command_path=use(args.construction/'01_build_template.sh')
    log_path=use(args.construction/'03_construction.log')
    log=log_path.read_text(errors='replace')
    patterns=r'Interpolation type:|Sharpening method|physical space|bigimage|AverageImages|antsApplyTransforms|Registration using|Iteration:|gradient step|[Cc]rop|[Pp]add|[Rr]esampl'
    excerpts=[dict(line=i,text=line.strip()) for i,line in enumerate(log.splitlines(),1) if re.search(patterns,line)]
    pd.DataFrame(excerpts).to_csv(out/'01_construction_log_evidence.csv',index=False)
    json_write(out/'construction_provenance.json',dict(command=command_path.read_text(),excerpts='01_construction_log_evidence.csv',
        construction_manifest=construction,interpretation='Logged operations describe the run. Missing log keywords do not prove an operation absent; original ANTs script/version may be unavailable.'))
    # Audit scalar image grids (not warp images) at every retained construction stage.
    images=[('input_'+s,args.dataset/s/'anat'/f'{s}_T1w.nii') for s in cfg['participants']]
    images += [(p.name,p) for p in sorted(args.construction.rglob('*.nii*')) if not any(k in p.name.lower() for k in ['warp.nii','inversewarp.nii'])]
    images += [('average_seven_'+str(p.relative_to(args.average_seven/'mri')),p) for p in [args.average_seven/'mri/orig/001.mgz',args.average_seven/'mri/orig.mgz',args.average_seven/'mri/T1.mgz'] if p.is_file()]
    images += [('fsaverage_T1',args.fsaverage/'mri/T1.mgz')]
    final=nib.load(use(args.construction/'group_template0.nii.gz'))
    def audit_image(label,path):
        img=nib.load(use(path));shape=np.array(img.shape[:3]);zooms=np.array(img.header.get_zooms()[:3])
        corners=apply(img.affine,np.array(list(itertools.product(*[[-.5,n-.5] for n in shape]))))
        is_nifti=hasattr(img,'get_qform')
        q,qc=img.get_qform(coded=True) if is_nifti else (None,0)
        s,sc=img.get_sform(coded=True) if is_nifti else (None,0)
        row=dict(image=label,path=str(path.resolve()),shape=json.dumps(shape.tolist()),voxel_mm=json.dumps(zooms.tolist()),
            axis_codes=''.join(nib.aff2axcodes(img.affine)),affine_ras_mm=json.dumps(img.affine.tolist()),
            units=img.header.get_xyzt_units()[0] if is_nifti else 'mm (MGH contract)',
            fov_edge_mm=json.dumps((shape*zooms).tolist()),scanner_ras_min_mm=json.dumps(corners.min(0).tolist()),
            scanner_ras_max_mm=json.dumps(corners.max(0).tolist()),qform_code=int(qc),sform_code=int(sc),
            qform_sform_agree=None if q is None or s is None else bool(np.allclose(q,s,atol=1e-3)),
            same_grid_as_final=bool(tuple(img.shape)==tuple(final.shape) and np.allclose(img.affine,final.affine,atol=1e-4)))
        # Intensity support bounds at multiple relative thresholds are sensitive
        # to background/defacing and not a biological head segmentation.
        data=np.asarray(img.dataobj,dtype=np.float32)
        if data.ndim==3:
            positive=data[np.isfinite(data)&(data>0)]
            p99=float(np.percentile(positive,99)) if len(positive) else 0
            for fraction in [.01,.05,.10]:
                support=np.isfinite(data)&(data>p99*fraction)
                bounds=[]
                for axis in range(3):
                    active=np.flatnonzero(support.any(axis=tuple(x for x in range(3) if x!=axis)))
                    bounds.append([int(active.min()),int(active.max())] if len(active) else None)
                    for side,index in [('low',0),('high',data.shape[axis]-1)]:
                        coverage_rows.append(dict(kind='MRI_FOV_support',model=label,region=f'voxel_axis{axis}_{side}',
                            threshold_fraction_p99=fraction,n=int(np.take(support,index,axis=axis).sum()),
                            denominator=int(np.prod([data.shape[x] for x in range(3) if x!=axis]))))
                row[f'support_bbox_fraction_{fraction}']=json.dumps(bounds)
        mri_rows.append(row)
    for label,path in images:audit_image(label,path)
    # Transform inventory: ANTs affine parameters remain LPS; no transforms applied.
    for path in sorted(args.construction.glob('*GenericAffine.mat')):
        data=loadmat(use(path));keys=[k for k in data if not k.startswith('__')]
        params=next((np.ravel(data[k]) for k in keys if 'AffineTransform' in k),None)
        fixed=np.ravel(data.get('fixed',np.array([])))
        transform_rows.append(dict(path=str(path.resolve()),type='ANTs affine LPS',keys=json.dumps(keys),
            parameters=json.dumps(params.tolist()) if params is not None else '',fixed_parameters=json.dumps(fixed.tolist()),
            determinant=float(np.linalg.det(params[:9].reshape(3,3))) if params is not None and len(params)>=12 else None,
            interpretation='parameters retained; center/fixed parameters are essential; filename alone does not establish point mapping direction'))
    for path in sorted(args.construction.glob('*Warp.nii.gz')):
        img=nib.load(use(path));arr=np.asarray(img.dataobj,dtype=np.float32).squeeze()
        if arr.ndim!=4 or arr.shape[-1]!=3:
            warnings.append(f'Unrecognized warp layout: {path}');continue
        # Coarse physical Jacobian of displacement field alone (not composite).
        u=arr[::4,::4,::4].astype(float)*np.array([-1,-1,1])
        jac=np.stack([np.stack(np.gradient(u[...,component],axis=(0,1,2)),axis=-1) for component in range(3)],axis=-2)
        jac=jac@np.linalg.inv(img.affine[:3,:3]*4)+np.eye(3)
        det=np.linalg.det(jac[1:-1,1:-1,1:-1]);norm=np.linalg.norm(u,axis=-1)
        transform_rows.append(dict(path=str(path.resolve()),type='ANTs displacement LPS vectors; NIfTI affine RAS mm',
            shape=json.dumps(img.shape),sample_stride=4,displacement_p50_mm=float(np.median(norm)),displacement_p99_mm=float(np.percentile(norm,99)),
            sampled_jacobian_min=float(det.min()),sampled_jacobian_p01=float(np.percentile(det,1)),sampled_jacobian_nonpositive=int((det<=0).sum()),
            interpretation='coarse field-only Jacobian; not composite or full-resolution fold certification; no warp applied'))
    # Load primary scalp and MRI geometry. FS bridge comes from the actual volume.
    for model,folder in [('average_seven',args.average_seven),('fsaverage',args.fsaverage)]:
        v,f,meta=load_mesh(use(folder/'bem/outer_skin.surf'));surfaces[model]=(v,f)
        image=nib.load(use(folder/'mri/T1.mgz'))
        checked=check_surface_volume_metadata(meta,image,folder/'bem/outer_skin.surf')
        bridge=image.affine@np.linalg.inv(image.header.get_vox2ras_tkr())
        bridges.append(dict(model=model,source='actual T1.mgz affine @ inverse(vox2ras_tkr)',surface_to_scanner_ras_mm=bridge.tolist(),status='HEADER_DERIVED',surface_volume_metadata_checked=checked))
        mris[model]=MRI(folder/'mri/T1.mgz',bridge)
        models[model]=[('primary scalp',v,f,'cyan')]
        # Metadata check against actual FS volume geometry.
        if 'cras' in meta:
            header_center=apply(image.affine,np.array(image.shape[:3],float)[None,:]/2)[0]
            bridges[-1]['surface_header_cras_max_difference_mm']=float(np.max(np.abs(np.asarray(meta['cras'])-header_center)))
    for subject in cfg['participants']:
        v,f,meta=load_mesh(use(args.dataset/'derivatives/sourcemodelling'/subject/'anat'/f'{subject}_outer_skin.surf.gii'))
        bridge,scanner,size=native_bridge(meta)
        path=use(args.dataset/subject/'anat'/f'{subject}_T1w.nii');image=nib.load(path)
        match=tuple(size)==tuple(image.shape) and np.allclose(scanner,image.affine,atol=1e-3)
        if not match:
            raise ValueError(f'{subject}: native GIFTI volume geometry does not match released T1w; cannot silently assume bridge')
        bridges.append(dict(model=subject,source='GIFTI VolGeom scanner affine @ inverse(tkr)',surface_to_scanner_ras_mm=bridge.tolist(),
                            volume_affine_max_difference_mm=float(np.max(np.abs(scanner-image.affine))),status='GIFTI_AND_MRI_HEADERS_MATCH; anatomical provenance unverified'))
        surfaces[subject]=(v,f);mris[subject]=MRI(path,bridge);models[subject]=[('native scalp',v,f,'cyan')]
    # Alternative representations only. Compare numerically and overlay; never adopt.
    candidate_paths=[]
    for folder in [args.average_seven/'bem/watershed',EXP/'work/bem_candidates/atlas_mode',EXP/'work/bem_candidates/t1_mode',EXP/'work/bem_candidates/local_inner_1mm']:
        p=folder/'average_seven_outer_skin_surface'
        if p.is_file():candidate_paths.append(('average_seven',p))
    for model,folder in [('average_seven',args.average_seven),('fsaverage',args.fsaverage)]:
        for p in sorted((folder/'bem').glob('*head*.fif')):candidate_paths.append((model,p))
    alternatives=[]
    for number,(model,path) in enumerate(candidate_paths):
        print('Reading alternative scalp:',path,flush=True)
        v,f,meta=load_mesh(use(path));name=f'{model}_alternative_{number+1:02d}'
        for message in meta.get('reader_warnings',[]):
            warnings.append(f'{path}: {message}')
        check_surface_volume_metadata(meta,mris[model].image,path)
        alternatives.append((model,name,path,v,f));surfaces[name]=(v,f)
        # Alternative FS meshes must be in MRI frame. .surf metadata cras must
        # agree with model T1 before overlaying in the same surface frame.
        if 'cras' in meta:
            center=apply(mris[model].image.affine,np.array(mris[model].data.shape)[None,:]/2)[0]
            if not np.allclose(meta['cras'],center,atol=1e-3):
                raise ValueError(f'Alternative surface volume center mismatch: {path}')
        base=surfaces[model][0]
        d1=cKDTree(base).query(v)[0];d2=cKDTree(v).query(base)[0]
        alt_rows.append(dict(model=model,alternative=name,path=str(path.resolve()),vertices=len(v),faces=len(f),
            reader_warnings_json=json.dumps(meta.get('reader_warnings',[])),
            same_vertex_and_face_arrays=bool(v.shape==base.shape and f.shape==surfaces[model][1].shape and np.array_equal(v,base) and np.array_equal(f,surfaces[model][1])),
            alternative_to_primary_vertex_p95_mm=float(np.percentile(d1,95)),primary_to_alternative_vertex_p95_mm=float(np.percentile(d2,95)),
            interpretation='sampling-dependent bidirectional vertex distances; alternative anatomy unreviewed'))
    # Model geometry, actual mesh plane sections and area-weighted region coverage.
    for model,(v,f) in surfaces.items():
        print('Scalp geometry:',model,flush=True)
        topo=topology(v,f);topos[model]=topo
        lo,hi=v.min(0),v.max(0);center=np.median(v,axis=0)
        geometry_rows.append(dict(model=model,**topo,**{f'{a}_min_mm':float(lo[i]) for i,a in enumerate('xyz')},
            **{f'{a}_max_mm':float(hi[i]) for i,a in enumerate('xyz')},
            width_mm=float(hi[0]-lo[0]),ap_length_mm=float(hi[1]-lo[1]),height_mm=float(hi[2]-lo[2]),
            coverage_convention='area sectors about own median surface vertex; no intersubject registration'))
        tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        centroids=tri.mean(1);sectors=region(centroids,center)
        for sector in ['anterior','left','right','posterior']:
            mask=sectors==sector
            for bottom,top in [(-1000,-40),(-40,0),(0,40),(40,80),(80,1000)]:
                keep=mask&(centroids[:,2]>=bottom)&(centroids[:,2]<top)
                if keep.any():
                    normals=np.cross(tri[keep,1]-tri[keep,0],tri[keep,2]-tri[keep,0])
                    norm=np.linalg.norm(normals,axis=1)
                    normals=normals/np.maximum(norm[:,None],1e-30)
                    weights=area[keep]/max(float(area[keep].sum()),1e-30)
                    patch_rows.append(dict(model=model,region=sector,z_low_mm=bottom,z_high_mm=top,
                        area_mm2=float(area[keep].sum()),normal_coherence=float(np.linalg.norm(np.sum(normals*weights[:,None],axis=0))),
                        **{f'normal_near_axis_{a}_area_fraction':float(weights[np.abs(normals[:,i])>.98].sum()) for i,a in enumerate('xyz')},
                        **{f'{a}_span_mm':float(np.ptp(tri[keep,:,i])) for i,a in enumerate('xyz')},
                        interpretation='high normal coherence/axis alignment may indicate planar patches; not an anatomical defect diagnosis'))
                coverage_rows.append(dict(kind='surface_area',model=model,region=sector,z_low_mm=bottom,z_high_mm=top,
                    n=int(keep.sum()),area_mm2=float(area[keep].sum()),fraction_total_area=float(area[keep].sum()/area.sum())))
        for axis,positions in [(2,range(-80,101,10)),(1,range(-80,101,10)),(0,range(-80,81,10))]:
            rem=[i for i in range(3) if i!=axis]
            for position in positions:
                seg=plane_segments(v,f,axis,position)
                if len(seg):
                    points=seg.reshape(-1,2)
                    sections.append(dict(model=model,axis='XYZ'[axis],position_mm=position,segments=len(seg),
                        axis_u='XYZ'[rem[0]],axis_v='XYZ'[rem[1]],u_min_mm=float(points[:,0].min()),u_max_mm=float(points[:,0].max()),
                        v_min_mm=float(points[:,1].min()),v_max_mm=float(points[:,1].max()),
                        contour_length_mm=float(np.linalg.norm(seg[:,1]-seg[:,0],axis=1).sum())))
        if model not in mris:continue
        mri=mris[model];vox=apply(mri.to_voxel,v)
        for axis in range(3):
            for side,dist in [('low',vox[:,axis]),('high',mri.data.shape[axis]-1-vox[:,axis])]:
                coverage_rows.append(dict(kind='surface_FOV_proximity',model=model,region=f'voxel_axis{axis}_{side}',
                    n=int((dist<2).sum()),denominator=len(v),minimum_margin_voxel=float(dist.min())))
        # Radial profiles sample 1024 deterministic vertices. Strongest outward
        # intensity drop is a review cue, not a scalp segmentation algorithm.
        ids=np.unique(np.linspace(0,len(v)-1,min(1024,len(v)),dtype=int));p=v[ids]
        direction=p-center;direction/=np.linalg.norm(direction,axis=1)[:,None]
        offsets=np.arange(-15,15.1,.5);samples=p[:,None,:]+direction[:,None,:]*offsets[None,:,None]
        values,valid=mri.sample(samples);deriv=np.diff(values,axis=1)/.5
        derivative_valid=valid[:,:-1]&valid[:,1:]&np.isfinite(deriv)
        score=np.where(derivative_valid,-deriv,-np.inf);best=np.argmax(score,axis=1)
        for j,vertex in enumerate(ids):
            profile_rows.append(dict(model=model,vertex=int(vertex),region=region(p[j:j+1],center)[0],
                x_mm=p[j,0],y_mm=p[j,1],z_mm=p[j,2],all_samples_in_fov=bool(valid[j].all()),
                strongest_outward_drop_offset_mm=float((offsets[best[j]]+offsets[best[j]+1])/2) if derivative_valid[j].any() else None,
                outward_drop_per_mm=float(score[j,best[j]]) if derivative_valid[j].any() else None,
                interpretation='radial intensity edge cue only; skull, hair, masking and bias may generate competing edges'))
        overview(model,mri,models[model],v,out/'figures'/f'{model}_MRI_scalp_overview.png',args.dpi)
    # Construction-versus-FS inspection: same scalp mapped through scanner RAS.
    bridge=mris['average_seven'].bridge
    construction_mri=MRI(use(args.construction/'group_template0.nii.gz'),bridge)
    overview('average_seven_original_template',construction_mri,models['average_seven'],surfaces['average_seven'][0],
             out/'figures/average_seven_original_template_scalp.png',args.dpi)
    imported_path=args.average_seven/'mri/orig/001.mgz'
    if imported_path.is_file():
        imported=MRI(use(imported_path),bridge)
        overview('average_seven_FS_import',imported,models['average_seven'],surfaces['average_seven'][0],out/'figures/average_seven_FS_import_scalp.png',args.dpi)
    # Existing representation overlays rendered individually to keep labels readable.
    for model,name,path,v,f in alternatives:
        overview(name,mris[model],[*models[model],(path.parent.name+'/'+path.name,v,f,'orange')],surfaces[model][0],
                 out/'figures'/f'{name}_overlay.png',args.dpi)
    residuals=pd.read_csv(use(args.previous/'04_electrode_residuals.csv'))
    path_profiles=[]
    for (subject,model),group in residuals.groupby(['subject','model'],sort=False):
        print('Signed residuals and MRI:',subject,model,flush=True)
        key=subject if model=='native' else model
        v,f=surfaces[key];mri=mris[key]
        points=group[['x_mm','y_mm','z_mm']].to_numpy(float)
        distances,nearest=triangle_distance(points,v,f)
        if not np.allclose(distances,group.triangle_distance_mm,atol=1e-4,rtol=0):
            raise ValueError(f'{subject}/{model}: primary scalp differs from prior diagnostic; no silent substitution')
        sd,states,w=signed_distance(points,v,f,distances,topos[key])
        vox=apply(mri.to_voxel,points)
        for j,(_,r) in enumerate(group.iterrows()):
            sign_rows.append(dict(subject=subject,model=model,channel=r.channel,region_proxy=r.native_region_proxy,
                distance_mm=float(distances[j]),signed_distance_mm=float(sd[j]) if np.isfinite(sd[j]) else None,
                classification=states[j],winding_number=float(w[j]),
                electrode_in_mri_fov=bool(((vox[j]>=0)&(vox[j]<=np.array(mri.data.shape)-1)).all()),
                electrode_x_mm=points[j,0],electrode_y_mm=points[j,1],electrode_z_mm=points[j,2],
                nearest_x_mm=nearest[j,0],nearest_y_mm=nearest[j,1],nearest_z_mm=nearest[j,2]))
        problem=group.channel.str.fullmatch(r'e(23[8-9]|24[0-9]|25[0-2])')
        selected=[]
        for mask in [problem.to_numpy(),~problem.to_numpy()]:
            candidates=np.flatnonzero(mask)
            if len(candidates):selected.append(candidates[np.argmax(distances[candidates])])
        for j in selected:
            channel=group.iloc[j].channel
            target_figure(subject,model,channel,points[j],nearest[j],sd[j],mri,models[key],
                out/'figures'/f'{subject}_{model}_{channel}_MRI_residual.png',args.dpi)
            if model=='average_seven' and j==selected[0]:
                target_figure(subject,model+'_original_template',channel,points[j],nearest[j],sd[j],construction_mri,models[key],
                    out/'figures'/f'{subject}_{model}_{channel}_original_template_MRI.png',args.dpi)
            slice_rows.append(dict(subject=subject,model=model,channel=channel,selection='largest problematic-block residual or largest outside-block residual',
                                   file=f'figures/{subject}_{model}_{channel}_MRI_residual.png'))
        # Full 3D intensity paths for all problematic channels and top residuals.
        selected_path=np.unique(np.r_[np.flatnonzero(problem.to_numpy()),np.argsort(distances)[-5:]])
        for j in selected_path:
            unit=(points[j]-nearest[j])/max(distances[j],1e-12)
            steps=np.linspace(-10,distances[j]+10,101);locations=nearest[j]+steps[:,None]*unit
            values,valid=mri.sample(locations)
            for t,value,inside in zip(steps,values,valid):
                path_profiles.append(dict(subject=subject,model=model,channel=group.iloc[j].channel,
                    mm_from_nearest_toward_electrode=float(t),electrode_distance_mm=float(distances[j]),
                    intensity=float(value) if np.isfinite(value) else None,in_fov=bool(inside),
                    convention='q to electrode direction; not necessarily outward'))
        # Alternative exact distances/signs: problematic block plus top five only.
        for parent,name,path,av,af in alternatives:
            if parent!=model:continue
            dd,qq=triangle_distance(points[selected_path],av,af)
            ss,st,ww=signed_distance(points[selected_path],av,af,dd,topos[name])
            for k,j in enumerate(selected_path):
                alt_rows.append(dict(model=model,alternative=name,subject=subject,channel=group.iloc[j].channel,
                    distance_mm=float(dd[k]),primary_distance_mm=float(distances[j]),distance_change_mm=float(dd[k]-distances[j]),
                    signed_distance_mm=float(ss[k]) if np.isfinite(ss[k]) else None,classification=st[k],
                    interpretation='targeted channels only; alternative is not adopted or accepted'))
    # Fiducials: template MRI estimates, and two explicit native sidecar hypotheses.
    previous_landmarks=pd.read_csv(use(args.previous/'08_landmark_audit.csv'))
    for _,r in previous_landmarks.iterrows():
        key=r.model if r.subject=='template' else r.subject
        if key not in mris:continue
        p=r[['x_mm','y_mm','z_mm']].to_numpy(float)
        val,valid=mris[key].sample(p[None,:]);v,f=surfaces[key]
        d,_=triangle_distance(p[None,:],v,f)
        landmarks.append(dict(subject=r.subject,model=key,landmark=r.landmark,hypothesis=r.hypothesis,
            x_mm=p[0],y_mm=p[1],z_mm=p[2],nearest_triangle_mm=float(d[0]),in_fov=bool(valid[0]),
            mri_intensity=float(val[0]) if np.isfinite(val[0]) else None,status=r.status))
    for key in mris:
        rows=[r for r in landmarks if r['model']==key]
        if not rows:continue
        hypotheses=list(dict.fromkeys(r['hypothesis'] for r in rows))
        fig,axes=plt.subplots(len(hypotheses),3,figsize=(19,6*len(hypotheses)),squeeze=False,layout='constrained')
        for row,hypothesis in enumerate(hypotheses):
            for col,label in enumerate(['LPA','NAS','RPA']):
                record=next((r for r in rows if r['hypothesis']==hypothesis and r['landmark']==label),None)
                if record is None:axes[row,col].set_visible(False);continue
                point=np.array([record['x_mm'],record['y_mm'],record['z_mm']])
                limits=[point[0]-45,point[0]+45,point[1]-45,point[1]+45]
                draw_slice(axes[row,col],mris[key],models[key],2,point[2],limits,f'{label}: Z={point[2]:.1f} mm\n{hypothesis}')
                axes[row,col].scatter(point[0],point[1],c='magenta',marker='+',s=100,linewidths=2)
        fig.suptitle(f'{key}: axial MRI through each fiducial under explicitly labelled hypotheses\nNative sidecars are repeated/unverified; template landmarks estimated/unreviewed',fontsize=14)
        fig.savefig(out/'figures'/f'{key}_fiducial_hypotheses.png',dpi=args.dpi);plt.close(fig)
    # Final warped average comparison is descriptive: ANTs applies a subsequent
    # shape update. It is not expected to equal the mean of retained warped inputs.
    warped=sorted(args.construction.glob('group_input*-WarpedToTemplate.nii.gz'))
    average_check=dict(n_warped=len(warped),interpretation='retained warped inputs may precede final shape update; mismatch is not proof of averaging failure')
    if len(warped)==7:
        mean=np.zeros(final.shape,dtype=np.float64);same=True
        for p in warped:
            img=nib.load(use(p))
            if img.shape!=final.shape or not np.allclose(img.affine,final.affine,atol=1e-4):same=False;break
            arr=np.asarray(img.dataobj,dtype=np.float32);positive=arr[arr>0]
            mean+=arr/max(float(np.mean(positive)),1e-12)/7
        average_check['same_grids']=same
        if same:
            template=np.asarray(final.dataobj,dtype=np.float32);support=(mean>0)&(template>0)
            tv=template[support];mv=mean[support];tv=tv/max(float(tv.mean()),1e-12)
            average_check.update(normalized_mean_correlation=float(np.corrcoef(tv,mv)[0,1]),normalized_rmse=float(np.sqrt(np.mean((tv-mv)**2))))
            # View final scalar template versus mean on a common scanner grid.
            mean_mri=MRI(args.construction/'group_template0.nii.gz',bridge);mean_mri.data=mean.astype(np.float32)
            mean_mri.vmax=float(np.percentile(mean[mean>0],99.5))
            overview('normalized retained warped-input mean',mean_mri,models['average_seven'],surfaces['average_seven'][0],out/'figures/retained_warped_mean_scalp.png',args.dpi)
    if len(warped)==7:
        target_rows=residuals[(residuals.subject=='sub-03')&(residuals.model=='average_seven')]
        target=target_rows.loc[target_rows.triangle_distance_mm.idxmax(),['x_mm','y_mm','z_mm']].to_numpy(float)
        stage_slices=[]
        for label,path in [('final template',args.construction/'group_template0.nii.gz')]+[(p.name.split('modality0-')[-1].split('_T1w')[0],p) for p in warped]:
            stage=MRI(use(path),bridge)
            panels=[]
            for axis in [2,1,0]:
                rem=[i for i in range(3) if i!=axis]
                limits=[target[rem[0]]-65,target[rem[0]]+65,target[rem[1]]-65,target[rem[1]]+65]
                values,extent=stage.slice(axis,target[axis],limits)
                panels.append((values,extent,limits))
            stage_slices.append((label,stage.vmax,panels))
            del stage
        for plane,(axis,label) in enumerate([(2,'axial'),(1,'coronal'),(0,'sagittal')]):
            fig,axes=plt.subplots(2,4,figsize=(21,11),layout='constrained')
            rem=[i for i in range(3) if i!=axis]
            for ax,(name,vmax,panels) in zip(axes.flat,stage_slices):
                values,extent,limits=panels[plane]
                ax.imshow(values,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=vmax)
                v,f=surfaces['average_seven'];seg=plane_segments(v,f,axis,target[axis])
                if len(seg):ax.add_collection(LineCollection(seg,colors='cyan',linewidths=1.1))
                ax.scatter(*target[rem],c='magenta',s=45)
                ax.set(xlim=limits[:2],ylim=limits[2:],title=name,xlabel='XYZ'[rem[0]]+' mm',ylabel='XYZ'[rem[1]]+' mm')
                ax.set_aspect('equal')
            fig.suptitle(f'{label}, {"XYZ"[axis]}={target[axis]:.1f} mm: largest sub-03 average-seven residual location\nFinal template versus retained registered inputs; each image independently windowed; warped inputs may precede final shape update',fontsize=14)
            fig.savefig(out/'figures'/f'construction_inputs_{label}_problem_region.png',dpi=args.dpi);plt.close(fig)
    json_write(out/'averaging_check.json',average_check)
    json_write(out/'coordinate_bridges.json',bridges)
    tables={'02_mri_geometry':mri_rows,'03_registration_transforms':transform_rows,'04_scalp_geometry_topology':geometry_rows,
        '05_regional_coverage':coverage_rows,'06_surface_sections':sections,'07_mri_radial_edge_cues':profile_rows,
        '08_electrode_signed_distances':sign_rows,'09_alternative_scalps':alt_rows,'10_fiducial_MRI_audit':landmarks,
        '11_residual_MRI_paths':path_profiles,'12_targeted_figure_index':slice_rows,'15_regional_planarity_cues':patch_rows,
        '16_FreeSurfer_processing_evidence':segmentation_evidence}
    for filename,rows in tables.items():pd.DataFrame(rows).to_csv(out/f'{filename}.csv',index=False)
    signs=pd.DataFrame(sign_rows)
    sign_summary=signs.groupby(['subject','model','classification']).size().rename('n').reset_index()
    sign_summary.to_csv(out/'13_inside_outside_counts.csv',index=False)
    cues=pd.DataFrame(profile_rows)
    cue_summary=cues.groupby(['model','region']).agg(n=('vertex','size'),median_edge_offset_mm=('strongest_outward_drop_offset_mm','median'),
        fraction_profiles_in_fov=('all_samples_in_fov','mean')).reset_index()
    cue_summary.to_csv(out/'14_radial_edge_summary.csv',index=False)
    report=['# Scalp geometry and MRI investigation','',
        'Status: diagnostic measurements complete; anatomical assessment pending. No previous output or anatomy was modified.','',
        '## Confirmed file and numerical checks','',
        f'Prior run COMPLETE with verified input/output hashes. Primary exact triangle distances reproduced for all 21 subject/model combinations. {len(images)} scalar MRI records and {len(transform_rows)} transform records audited.','',
        'Thresholds remain nearest-vertex median ≤10 mm and p95 ≤20 mm, with visual review. No fit, scaling, forward, inverse or new MRI registration was performed.','',
        'Construction shell and exact log excerpts are retained. FreeSurfer recon-all resampling/conformation evidence is in 16_FreeSurfer_processing_evidence.csv. Input hashes are verified against the completed construction manifest. Scalar MRI shapes, voxel sizes, qform/sform, FOV, intensity-support bounds and boundary support are exported. Log lines identify interpolation and repeated template shape updates where recorded.','',
        '## Inside/outside relative to estimated scalp','',sign_summary.to_string(index=False),'',
        'Positive signed distance means outside; negative means inside. Labels are CONDITIONAL: closed single-component consistently oriented nondegenerate topology, absolute winding near 0 or 1, agreement of three oblique ray parities, and distance >0.1 mm. Other points remain UNRESOLVED. Full self-intersection absence is not certified. These signs describe the mesh, not anatomy or digitization correctness.','',
        '## Surface geometry','',pd.DataFrame(geometry_rows).to_string(index=False),'',
        'Dimensions are in unregistered surface RAS mm. Regional areas use angular sectors about each mesh median and fixed Z bands; medians, surface coverage, orientation, defacing and neck extent confound comparisons. Section extrema are exact triangle-plane intersections. No scaling or surface alignment is used.','',
        '## MRI boundary evidence and interpretation','',
        '15_regional_planarity_cues.csv adds area-weighted normal coherence, normal axis alignment and patch extents to identify potentially flattened patches for review. High coherence can also occur in normal anatomy.','',cue_summary.to_string(index=False),'',
        'Radial edge offsets are strongest outward intensity drops within ±15 mm at 1024 deterministic surface vertices. They are review cues, not an independently validated scalp segmentation. Competing skull, tissue, hair, masking and bias-field boundaries can dominate. FOV support at 1%, 5%, 10% of positive-intensity p99 is also descriptive. A closed surface can still have an anatomically flattened or missing region.','',
        'Inspect average_seven_original_template_scalp.png, average_seven_FS_import_scalp.png and average_seven_MRI_scalp_overview.png together. A boundary already absent in original inputs/registered images suggests source coverage or construction effects; a visible MRI boundary outside the scalp across stages suggests segmentation mismatch. Expert slice review is required to distinguish these. Electrode distance alone cannot establish an anatomically invalid template.','',
        'Warp-field Jacobians are coarse field-only samples, not full-resolution or composite deformation validation. Retained warped images may precede the final template shape update; their normalized mean need not reproduce the final template. An absence of crop keywords does not prove cropping never occurred upstream.','',
        '## Coordinate conventions and figure reading','',
        'All panels use physical surface RAS millimetres: X right, Y anterior, Z superior. Native bridge reconstructed from GIFTI VolGeom scanner affine and tkregister geometry must match the actual released T1w affine/shape. Template bridges use each actual T1.mgz scanner affine and vox2ras_tkr. No unconditional cras-only approximation is made. See coordinate_bridges.json. These are header consistency checks, not independent electrode provenance.','',
        'ANTs affine/warp vector conventions are LPS; NIfTI affines are exposed by nibabel in RAS. Warp displacement statistics explicitly convert LPS vectors to RAS before physical Jacobian sampling. No ANTs transform is applied to electrodes.','',
        'Original template and imported MRI are sampled via scanner RAS using the FreeSurfer scalp bridge, permitting construction-versus-segmentation review. Trilinear interpolation is for diagnostic sampling/rendering only.','',
        'MRI residual figures center on the largest e238–e252 residual and the largest residual outside that block, for every participant and model. The plane passes through the electrode; the closest point and residual arrow are projections and nearest-point off-plane offsets are printed. Full 3D residual coordinates and MRI intensity paths are exported. Do not infer that both vector endpoints lie on the displayed slice.','',
        'Native fiducials retain two unverified sidecar interpretations from run_01; template fiducials are estimated/unreviewed. Missing retained native dig fiducials and repeated release sidecars remain unresolved. Native surfaces provide a reference, not independent sensor validation.','',
        'Alternative representations are compared without replacement. Representation agreement is not independent anatomical validation. Alternative exact sensor distances/signs are computed for e238–e252 plus each configuration’s five largest primary residuals, rather than treated as new alignment fits.','',
        '## Causal assessment','',
        'Confirmed: file/grid/topology/containment measurements, prior-distance reproduction and saved-transform reuse. Hypotheses needing review: template construction bias, source defacing/cropping, scalp segmentation collapse, and participant electrode convention errors. This script does not automatically select a cause from intensity edges or sensor distances.','',
        'Review the original/native MRI panels and construction stages before attributing failure. Any final causal finding should cite specific slice coordinates and evidence from both MRI and geometry. No BEM or alignment approval is granted.','',
        'Warnings: '+json.dumps(warnings)]
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    for path,value in hashes.items():
        if sha(path)!=value:raise ValueError(f'Input changed during run: {path}')
    manifest.update(status='COMPLETE',software=dict(python=platform.python_version(),mne=mne.__version__,nibabel=nib.__version__,numpy=np.__version__,pandas=pd.__version__),
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'},
        geometry_sign_self_intersections_certified=False,warnings=warnings)
    save_manifest();print(out,flush=True)


if __name__=='__main__':main()
