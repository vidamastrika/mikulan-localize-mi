#!/usr/bin/env python3
"""Read-only coverage, residual sensitivity and BEM-boundary investigation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from coverage_math import frame, region_proxy, patch_points, support, summary, mirrored_pairs

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'localize_mi_average_seven'
PREVIOUS=EXP/'diagnostics/02_scalp_geometry'
# Reuse the tested MRI/header/mesh utilities, without invoking their main().
sys.path.insert(0,str(PREVIOUS))
spec=importlib.util.spec_from_file_location('coverage_previous',PREVIOUS/'diagnose.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
from geometry import apply, plane_segments, topology, triangle_distance


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')


def probes(mri,points,center):
    samples=patch_points(points,center)
    values,valid=mri.sample(samples)
    positive=mri.data[np.isfinite(mri.data)&(mri.data>0)]
    p99=float(np.percentile(positive,99)) if len(positive) else 0.
    labels,fractions=support(values,valid,p99)
    return labels,fractions,values,valid


def plane_panel(ax,mri,meshes,origin,basis,limits,title,p=None,q=None):
    """Exact arbitrary-plane mesh intersections; physical-mm MRI samples."""
    u=np.arange(limits[0],limits[1]+.25,.5);v=np.arange(limits[2],limits[3]+.25,.5)
    uu,vv=np.meshgrid(u,v)
    locations=origin+uu[:,:,None]*basis[0]+vv[:,:,None]*basis[1]
    image,valid=mri.sample(locations)
    ax.imshow(image,origin='lower',extent=[u[0],u[-1],v[0],v[-1]],cmap='gray',vmin=0,vmax=mri.vmax)
    for name,vertices,faces,color in meshes:
        local=(vertices-origin)@basis.T
        segments=plane_segments(local,faces,2,0.)
        if len(segments):
            ax.add_collection(LineCollection(segments,colors=color,linewidths=1.05,
                linestyles='dashed' if 'repair' in name else 'solid',label=name))
    for label,point,color in [('electrode',p,'magenta'),('closest scalp',q,'lime')]:
        if point is not None:
            loc=(point-origin)@basis.T
            if abs(loc[2])<1e-5:
                ax.scatter(loc[0],loc[1],c=color,s=45,edgecolors='black',linewidths=.4,zorder=5)
            else:
                ax.text(.02,.02,f'{label} {loc[2]:+.1f} mm off-plane; not drawn',transform=ax.transAxes,
                    fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.7))
    if p is not None and q is not None:
        pp=(p-origin)@basis.T;qq=(q-origin)@basis.T
        if max(abs(pp[2]),abs(qq[2]))<1e-5:
            ax.annotate('',xy=pp[:2],xytext=qq[:2],arrowprops=dict(color='magenta',arrowstyle='->',lw=1.4))
    ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='plane U (mm)',ylabel='plane V (mm)',title=title)
    ax.set_aspect('equal')
    finite=image[np.isfinite(image)]
    return dict(in_fov_fraction=float(valid.mean()),positive_intensity_fraction=float(np.mean(finite>mri.vmax*.01)) if len(finite) else 0.)


def residual_figure(subject,model,label,channel,p,q,mri,meshes,path,dpi):
    d=float(np.linalg.norm(p-q));basis=frame(p-q)
    limits=[-80,max(35,d+15),-70,70]
    fig,axes=plt.subplots(1,3,figsize=(21,8),layout='constrained')
    definitions=[(q,basis,'Residual + superior transverse plane'),
                 (q,np.array([basis[0],basis[2],-basis[1]]),'Residual + orthogonal transverse plane'),
                 (q,np.eye(3),'Axial plane through nearest scalp point')]
    records=[]
    for ax,(origin,b,title) in zip(axes,definitions):
        records.append(plane_panel(ax,mri,meshes,origin,b,limits,title,p,q))
        p_offset=float(((p-origin)@b.T)[2]);q_offset=float(((q-origin)@b.T)[2])
        full_residual=title.startswith('Residual')
        if full_residual and max(abs(p_offset),abs(q_offset))>1e-6:
            raise ValueError('Residual plane does not contain both endpoints')
        records[-1].update(origin_surface_ras_mm=origin.tolist(),basis_rows=b.tolist(),title=title,
            electrode_off_plane_mm=p_offset,scalp_off_plane_mm=q_offset,contains_full_residual=full_residual)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle(f'{subject} / {model} / {label} / {channel}: unsigned residual {d:.1f} mm\nFirst two planes contain BOTH endpoints; scalp-centered context retained. RAS physical coordinates; saved transform only.',fontsize=14)
    fig.savefig(path,dpi=dpi);plt.close(fig)
    return records


def stage_figure(entries,meshes,p,q,path,dpi,title):
    basis=frame(p-q);d=float(np.linalg.norm(p-q));limits=[-85,max(35,d+15),-70,70]
    rows=int(np.ceil(len(entries)/3));fig,axes=plt.subplots(rows,3,figsize=(20,6*rows),squeeze=False,layout='constrained')
    evidence=[]
    for ax,(label,mri) in zip(axes.flat,entries):
        evidence.append(dict(stage=label,**plane_panel(ax,mri,meshes,q,basis,limits,label,p,q)))
    for ax in list(axes.flat)[len(entries):]:ax.set_visible(False)
    axes[0,0].legend(fontsize=8,loc='upper left')
    fig.suptitle(title+'\nPlane contains full residual and extends 85 mm inward from nearest scalp; independently windowed MRIs',fontsize=14)
    fig.savefig(path,dpi=dpi);plt.close(fig)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment',type=Path,default=EXP/'diagnostics/01_eeg_alignment/run_01')
    parser.add_argument('--geometry',type=Path,default=PREVIOUS/'run_02')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/Localize-MI')
    parser.add_argument('--construction',type=Path,default=EXP/'outputs/02_template_retry1')
    parser.add_argument('--average-seven',type=Path,default=EXP/'work/freesurfer/average_seven')
    parser.add_argument('--fsaverage',type=Path,default=Path.home()/'mne_data/MNE-fsaverage-data/fsaverage')
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'run_01')
    parser.add_argument('--dpi',type=int,default=220)
    args=parser.parse_args();out=args.output_root.resolve()
    if out.exists():raise FileExistsError(f'Refusing existing output {out}; use a fresh numbered run')
    if args.dpi<180:parser.error('Use at least 180 DPI')
    hashes={}
    def use(path):
        path=Path(path).resolve()
        if str(path) not in hashes:hashes[str(path)]=sha(path)
        return path
    for p in [Path(__file__),Path(__file__).with_name('coverage_math.py'),PREVIOUS/'diagnose.py',PREVIOUS/'geometry.py',PREVIOUS/'REVIEW_run_02.md']:
        use(p)
    # Read review and verify both completed runs before consuming their tables.
    review=(PREVIOUS/'REVIEW_run_02.md').read_text()
    for folder in [args.alignment,args.geometry]:
        manifest=json.loads(use(folder/'manifest.json').read_text())
        if manifest['status']!='COMPLETE':raise ValueError(f'Previous run not COMPLETE: {folder}')
        for path,value in manifest['input_sha256'].items():
            if sha(path)!=value:raise ValueError(f'Previous input changed: {path}')
            use(path)
        for path,value in manifest['output_sha256'].items():
            if sha(folder/path)!=value:raise ValueError(f'Previous output changed: {folder/path}')
            use(folder/path)
    cfg=json.loads(use(EXP/'config/experiment.json').read_text());subjects=cfg['participants']
    if cfg['coregistration']['scaling'] or cfg['coregistration']['median_limit_mm']!=10 or cfg['coregistration']['p95_limit_mm']!=20:
        raise ValueError('Original numerical policy changed')
    construction_manifest=json.loads(use(args.construction/'04_execution_manifest.json').read_text())
    if construction_manifest['status']!='COMPLETE':raise ValueError('Construction incomplete')
    for path,value in construction_manifest['input_sha256'].items():
        if sha(use(path))!=value:raise ValueError('Template input changed')
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    validation=dict(status='RUNNING',created_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        primary_montage_changed=False,thresholds_changed=False,scaling=False,refitting=False,surface_modification=False,
        facial_reconstruction=False,forward=False,inverse=False,anatomical_review='PENDING',coverage_labels='PROXIES_NOT_ANATOMICAL_TRUTH')
    def save():validation['input_sha256']=hashes;write_json(out/'validation_manifest.json',validation)
    save()
    evidence=[];channel_rows=[];stage_rows=[];figure_rows=[];layers_rows=[];contributions=[];sensitivity=[];left_rows=[];pair_rows=[]
    def text_evidence(path,pattern,role):
        text=use(path).read_text(errors='replace')
        for number,line in enumerate(text.splitlines(),1):
            if re.search(pattern,line,re.I):evidence.append(dict(role=role,path=str(path.resolve()),line=number,text=line.strip()))
    text_evidence(args.dataset/'README.md',r'original MRI|anonym','release documentation')
    text_evidence(ROOT/'_08_bids_conversion.py',r'T1_pydefaced|T1w.nii|T1_mf|maskface','export code')
    text_evidence(ROOT/'fx_bids.py',r'Pydeface|Face Masking|ImageProcessingSoftware','sidecar generation code')
    text_evidence(args.construction/'01_build_template.sh',r'T1w.nii','construction command')
    sidecars=[]
    print('Load source MRIs, scalp geometry, and documented anonymization provenance',flush=True)
    mris={};meshes={};centers={};maskface={};sources={}
    for model,folder in [('average_seven',args.average_seven),('fsaverage',args.fsaverage)]:
        v,f,_=helper.load_mesh(use(folder/'bem/outer_skin.surf'));meshes[model]=(v,f);centers[model]=np.median(v,axis=0)
        img=nib.load(use(folder/'mri/T1.mgz'));bridge=img.affine@np.linalg.inv(img.header.get_vox2ras_tkr())
        mris[model]=helper.MRI(folder/'mri/T1.mgz',bridge)
    for subject in subjects:
        sidepath=args.dataset/subject/'anat'/f'{subject}_T1w.json'
        side=json.loads(use(sidepath).read_text());sidecars.append(dict(subject=subject,metadata=side))
        evidence.append(dict(role='MRI processing sidecar',path=str(sidepath.resolve()),line=0,text=json.dumps(side)))
        v,f,meta=helper.load_mesh(use(args.dataset/'derivatives/sourcemodelling'/subject/'anat'/f'{subject}_outer_skin.surf.gii'))
        bridge,scanner,size=helper.native_bridge(meta);path=use(args.dataset/subject/'anat'/f'{subject}_T1w.nii')
        image=nib.load(path)
        if tuple(size)!=tuple(image.shape) or not np.allclose(scanner,image.affine,atol=1e-3):raise ValueError('Native MRI/scalp geometry mismatch')
        mris[subject]=helper.MRI(path,bridge);meshes[subject]=(v,f);centers[subject]=np.median(v,axis=0)
        alternative=args.dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
        if alternative.is_file():
            maskface[subject]=helper.MRI(use(alternative),bridge)
            sp=alternative.with_suffix('.json')
            if sp.is_file():text_evidence(sp,r'.','existing alternative anonymized MRI metadata')
    residuals=pd.read_csv(use(args.alignment/'04_electrode_residuals.csv'))
    signed=pd.read_csv(use(args.geometry/'08_electrode_signed_distances.csv'))
    native=signed[signed.model=='native'].copy()
    coverage={};region_by_subject={};native_q={}
    for subject in subjects:
        rows=native[native.subject==subject].set_index('channel').sort_index()
        q=rows[['nearest_x_mm','nearest_y_mm','nearest_z_mm']].to_numpy(float)
        labels,fractions,values,valid=probes(mris[subject],q,centers[subject])
        # Geometric anatomy uses native closest scalp, not the electrode offset.
        bins=region_proxy(q);coverage[subject]=dict(zip(rows.index,labels));region_by_subject[subject]=dict(zip(rows.index,bins));native_q[subject]=rows
        alt_labels=alt_fractions=None
        if subject in maskface:alt_labels,alt_fractions,_,_=probes(maskface[subject],q,centers[subject])
        for i,channel in enumerate(rows.index):
            channel_rows.append(dict(subject=subject,channel=channel,region_proxy=bins[i],native_mri_coverage=labels[i],
                support_01=fractions[i,0],support_05=fractions[i,1],support_10=fractions[i,2],all_probes_in_fov=bool(valid[i].all()),
                q_x_mm=q[i,0],q_y_mm=q[i,1],q_z_mm=q[i,2],
                alternative_maskface_coverage=alt_labels[i] if alt_labels is not None else 'UNAVAILABLE',
                convention='native surface RAS mm; inward MRI patch at closest native scalp; proxy, not verified physical channel location'))
        # Fixed anatomical-cutoff sensitivity, independent of residual outcomes.
        for y,z in itertools.product([20.,40.,60.],[0.,15.,30.]):
            alternative_bins=region_proxy(q,y,z)
            for label in np.unique(alternative_bins):
                sensitivity.append(dict(subject=subject,model='coverage_definition',subset=f'y{y:g}_z{z:g}_{label}',
                    n=int((alternative_bins==label).sum()),policy='geometric proxy cutoff sensitivity; not an alignment gate'))
        # Existing pydeface vs maskface images, no new image reconstruction.
        if subject in maskface:
            idx=np.flatnonzero(bins=='facial_proxy')
            focus=np.median(q[idx],axis=0) if len(idx) else np.array([0.,60.,-25.])
            fig,axes=plt.subplots(2,3,figsize=(18,11),layout='constrained')
            rows_img=[('released Pydeface',mris[subject]),('existing maskface alternative',maskface[subject])]
            for axrow,(name,mri) in zip(axes,rows_img):
                for ax,axis,title in zip(axrow,[2,1,0],['axial','coronal','sagittal']):
                    rem=[i for i in range(3) if i!=axis];limits=[focus[rem[0]]-80,focus[rem[0]]+80,focus[rem[1]]-80,focus[rem[1]]+80]
                    helper.draw_slice(ax,mri,[('original native scalp',*meshes[subject],'cyan')],axis,focus[axis],limits,f'{name}: {title} at {"XYZ"[axis]}={focus[axis]:.1f} mm')
            fig.suptitle(subject+': existing anonymized MRI representations against original native scalp; no substitution',fontsize=14)
            fig.savefig(out/'figures'/f'{subject}_anonymization_comparison.png',dpi=args.dpi);plt.close(fig)
            maskface.pop(subject)
        save()
    # Same channel identifiers across models; consensus subset across ALL seven.
    names=sorted(coverage[subjects[0]])
    if any(set(coverage[s])!=set(names) for s in subjects):raise ValueError('Channel labels differ across participants')
    common_supported={c for c in names if all(coverage[s][c]=='MRI_SUPPORTED_PROXY' for s in subjects)}
    common_cranial={c for c in common_supported if all(region_by_subject[s][c]=='cranial_proxy' for s in subjects)}
    write_json(out/'fixed_subset_definitions.json',dict(common_supported_channels=sorted(common_supported),
        common_supported_cranial_channels=sorted(common_cranial),selection='native MRI patches and native scalp geometry only; no residual selection; lexical labels do not establish physical numbering',
        coverage_cutoffs=dict(intensity_fractions_positive_p99=[.01,.05,.10],supported_fraction=.8,low_support_fraction=.2),
        geometric_proxies=dict(facial='z<15, y>=40 and y>=0.65*abs(x)',inferior_lateral='z<15 and abs(x)>=40 except facial',cranial='z>=15',transition='all others')))
    all_channel_groups=[]
    for (subject,model),group in residuals.groupby(['subject','model'],sort=False):
        print('Regions and residual-aligned MRI:',subject,model,flush=True)
        group=group.copy();group['region_proxy']=group.channel.map(region_by_subject[subject]);group['coverage_proxy']=group.channel.map(coverage[subject])
        all_channel_groups.append(group)
        full=summary(group.vertex_distance_mm)
        original=pd.read_csv(args.alignment/'03_model_distances.csv')
        record=original[(original.subject==subject)&(original.model==model)&(original.metric=='nearest_vertex')].iloc[0]
        if not np.allclose([full['median_mm'],full['p95_mm']],[record.median_mm,record.p95_mm],atol=1e-6):raise ValueError('Full montage statistics changed')
        masks={'full_montage_primary_unchanged':np.ones(len(group),bool),
            'native_MRI_supported_proxy':group.coverage_proxy.eq('MRI_SUPPORTED_PROXY').to_numpy(),
            'native_supported_cranial_proxy':(group.coverage_proxy.eq('MRI_SUPPORTED_PROXY')&group.region_proxy.eq('cranial_proxy')).to_numpy(),
            'all7_common_supported_channels':group.channel.isin(common_supported).to_numpy(),
            'all7_common_supported_cranial_channels':group.channel.isin(common_cranial).to_numpy()}
        for subset,mask in masks.items():
            st=summary(group.loc[mask,'vertex_distance_mm'])
            sensitivity.append(dict(subject=subject,model=model,subset=subset,**st,
                meets_original_limits_subset_only=(st['median_mm']<=10 and st['p95_mm']<=20) if st['n'] else None,
                policy='UNCHANGED_FULL_MONTAGE' if subset=='full_montage_primary_unchanged' else 'DIAGNOSTIC_ONLY_SAME_SAVED_TRANSFORM_NO_NEW_ACCEPTANCE',
                selected_channels_json=json.dumps(group.loc[mask,'channel'].tolist())))
        for field in ['region_proxy','coverage_proxy']:
            for label,part in group.groupby(field):
                contributions.append(dict(subject=subject,model=model,grouping=field,region=label,**summary(part.vertex_distance_mm),
                    above20_vertex=int((part.vertex_distance_mm>20).sum()),above20_triangle=int((part.triangle_distance_mm>20).sum()),
                    squared_triangle_residual_share=float(np.sum(part.triangle_distance_mm**2)/max(np.sum(group.triangle_distance_mm**2),1e-30)),
                    interpretation='two separate partitions; do not sum across grouping fields'))
        # Independently examine supported left/right cranial regions.
        qtable=native_q[subject]
        index=group.channel.tolist();q=qtable.loc[index,['nearest_x_mm','nearest_y_mm','nearest_z_mm']].to_numpy(float)
        for side,mask in [('left',(q[:,0]<-40)),('right',(q[:,0]>40))]:
            mask &= group.region_proxy.eq('cranial_proxy').to_numpy()&group.coverage_proxy.eq('MRI_SUPPORTED_PROXY').to_numpy()
            st=summary(group.loc[mask,'vertex_distance_mm'])
            left_rows.append(dict(subject=subject,model=model,side=side,**st,
                residual_x_mean_mm=float(group.loc[mask,'residual_x_mm'].mean()) if mask.any() else None,
                residual_y_mean_mm=float(group.loc[mask,'residual_y_mm'].mean()) if mask.any() else None,
                residual_z_mean_mm=float(group.loc[mask,'residual_z_mm'].mean()) if mask.any() else None,
                channels_json=json.dumps(group.loc[mask,'channel'].tolist())))
        pairs=mirrored_pairs(q,group.region_proxy.to_numpy(),group.coverage_proxy.to_numpy())
        for i,j,d in pairs:pair_rows.append(dict(subject=subject,model=model,left_channel=index[i],right_channel=index[j],mirror_q_distance_mm=d,
            left_vertex_mm=float(group.iloc[i].vertex_distance_mm),right_vertex_mm=float(group.iloc[j].vertex_distance_mm),
            difference_mm=float(group.iloc[i].vertex_distance_mm-group.iloc[j].vertex_distance_mm),
            interpretation='geometric nearest mirror proxy, <=15 mm; not physical cap pairing; right channels may repeat'))
        selections={
            'facial_or_low_support':(group.region_proxy.eq('facial_proxy')|group.coverage_proxy.eq('LOW_MRI_SUPPORT_PROXY')),
            'supported_left_cranial':((q[:,0]<-40)&group.region_proxy.eq('cranial_proxy')&group.coverage_proxy.eq('MRI_SUPPORTED_PROXY')),
            'inferior_lateral':group.region_proxy.eq('inferior_lateral_proxy')}
        key=subject if model=='native' else model
        for label,mask in selections.items():
            if not np.any(mask):continue
            r=group.loc[mask].sort_values('triangle_distance_mm',ascending=False).iloc[0]
            p=r[['x_mm','y_mm','z_mm']].to_numpy(float)
            qpoint=p-r[['residual_x_mm','residual_y_mm','residual_z_mm']].to_numpy(float)
            if np.linalg.norm(p-qpoint)<1e-8:continue
            records=residual_figure(subject,model,label,r.channel,p,qpoint,mris[key], [('scalp',*meshes[key],'cyan')],
                out/'figures'/f'{subject}_{model}_{label}_{r.channel}.png',args.dpi)
            for plane in records:figure_rows.append(dict(subject=subject,model=model,region=label,channel=r.channel,
                file=f'figures/{subject}_{model}_{label}_{r.channel}.png',**plane))
            if model=='average_seven' and label in ['facial_or_low_support','supported_left_cranial']:
                if label not in sources or np.linalg.norm(p-qpoint)>np.linalg.norm(sources[label][2]-sources[label][3]):
                    sources[label]=(subject,r.channel,p,qpoint)
        save()
    enriched=pd.concat(all_channel_groups,ignore_index=True)
    # Fixed anatomical box samples document stage support without pretending
    # that subject-native and template RAS coordinates are registered.
    stage_paths=[('initial',args.construction/'intermediateTemplates/initial_group_template0.nii.gz')]
    stage_paths += [(p.stem,p) for p in sorted((args.construction/'intermediateTemplates').glob('SyN_iteration*_group_template0.nii.gz'))]
    stage_paths += [('final',args.construction/'group_template0.nii.gz')]
    for label,(subject,channel,p,qpoint) in sources.items():
        entries=[]
        for stage,path in stage_paths:
            if path.is_file():entries.append((stage,helper.MRI(use(path),mris['average_seven'].bridge)))
        info=stage_figure(entries,[('final scalp',*meshes['average_seven'],'cyan')],p,qpoint,
            out/'figures'/f'construction_progression_{label}.png',args.dpi,
            f'{label}: construction stages, common final surface-RAS plane, {subject}/{channel}')
        for item in info:stage_rows.append(dict(comparison=label,**item,interpretation='same physical grid; earlier stages not anatomically registered to final by this diagnostic'))
        del entries
        warped=sorted(args.construction.glob('group_input*-WarpedToTemplate.nii.gz'))
        entries=[('final template',helper.MRI(use(args.construction/'group_template0.nii.gz'),mris['average_seven'].bridge))]
        entries += [(path.name.split('modality0-')[-1].split('_T1w')[0],helper.MRI(use(path),mris['average_seven'].bridge)) for path in warped]
        info=stage_figure(entries,[('final scalp',*meshes['average_seven'],'cyan')],p,qpoint,
            out/'figures'/f'registered_inputs_{label}.png',args.dpi,
            f'{label}: retained registered inputs vs final; inputs may precede final shape update')
        for item in info:stage_rows.append(dict(comparison=label+'_registered',**item))
        del entries
    # Layer inspection only; do not build or solve a BEM. Layer low MRI signal
    # is ambiguous (e.g. skull bone is naturally dark), never a validity test.
    layer_manifest=[];repair_rows=[]
    for model in subjects+['average_seven','fsaverage']:
        print('Inspect BEM-relevant layer coverage:',model,flush=True)
        layer_meshes=[('scalp',*meshes[model],'cyan')]
        available=[]
        for name,color in [('outer_skull','yellow'),('inner_skull','red'),('brain','lime')]:
            if model in subjects:path=args.dataset/'derivatives/sourcemodelling'/model/'anat'/f'{model}_{name}.surf.gii'
            else:path=(args.average_seven if model=='average_seven' else args.fsaverage)/'bem'/f'{name}.surf'
            if not path.is_file():continue
            v,f,meta=helper.load_mesh(use(path));layer_meshes.append((name,v,f,color));available.append((name,path,v,f))
            helper.check_surface_volume_metadata(meta,mris[model].image,path)
            if model in subjects:
                _,layer_scanner,layer_size=helper.native_bridge(meta)
                if tuple(layer_size)!=tuple(mris[model].data.shape) or not np.allclose(layer_scanner,mris[model].image.affine,atol=1e-3):
                    raise ValueError(f'Native layer MRI geometry mismatch: {path}')
            ids=np.unique(np.linspace(0,len(v)-1,min(512,len(v)),dtype=int));vertices=v[ids]
            labs,fracs,_,valid=probes(mris[model],vertices,centers[model]);bins=region_proxy(vertices)
            for label in np.unique(bins):
                mask=bins==label
                layers_rows.append(dict(model=model,layer=name,region_proxy=label,n=int(mask.sum()),
                    MRI_supported_probes=int(np.sum(labs[mask]=='MRI_SUPPORTED_PROXY')),
                    low_MRI_support_probes=int(np.sum(labs[mask]=='LOW_MRI_SUPPORT_PROXY')),
                    uncertain_probes=int(np.sum(~np.isin(labs[mask],['MRI_SUPPORTED_PROXY','LOW_MRI_SUPPORT_PROXY']))),
                    x_min_mm=float(vertices[mask,0].min()),x_max_mm=float(vertices[mask,0].max()),
                    y_min_mm=float(vertices[mask,1].min()),y_max_mm=float(vertices[mask,1].max()),
                    z_min_mm=float(vertices[mask,2].min()),z_max_mm=float(vertices[mask,2].max()),
                    interpretation='MRI probe support only, not tissue/boundary validity; dark skull and CSF confound'))
            layer_manifest.append(dict(model=model,layer=name,path=str(path.resolve()),**topology(v,f)))
        if model=='average_seven':
            repaired=EXP/'work/bem_candidates/local_inner_1mm/validation_subjects/average_seven/bem/inner_skull.surf'
            if repaired.is_file():
                v,f,meta=helper.load_mesh(use(repaired));layer_meshes.append(('inner skull repair candidate',v,f,'magenta'))
                original_inner=next(item for item in available if item[0]=='inner_skull')
                original_v,original_f=original_inner[2],original_inner[3]
                if v.shape==original_v.shape and np.array_equal(f,original_f):
                    change=np.linalg.norm(v-original_v,axis=1);bins=region_proxy(original_v)
                    for label in np.unique(bins):
                        mask=bins==label
                        repair_rows.append(dict(region_proxy=label,vertices=int(mask.sum()),changed_vertices=int((change[mask]>1e-4).sum()),
                            max_change_mm=float(change[mask].max()),mean_change_mm=float(change[mask].mean()),
                            interpretation='existing repair candidate versus original; no adoption; geometric region proxies only'))
                else:
                    repair_rows.append(dict(region_proxy='ALL',interpretation='original/candidate mesh indexing differs; paired displacement withheld'))
            for name in ['validation.json','repair.json','intersection_cortex_checks.json','source_containment.json']:
                path=EXP/'work/bem_candidates/local_inner_1mm'/name
                if path.is_file():use(path)
        # Axial, coronal and sagittal planes through facial cut region and
        # a separate preserved cranial control, using native scalp q proxies.
        fig,axes=plt.subplots(2,3,figsize=(19,12),layout='constrained')
        for row,focus in enumerate([np.array([0.,55.,-15.]),np.array([-45.,0.,45.])]):
            for ax,axis,label in zip(axes[row],[2,1,0],['axial','coronal','sagittal']):
                rem=[i for i in range(3) if i!=axis];limits=[focus[rem[0]]-100,focus[rem[0]]+100,focus[rem[1]]-100,focus[rem[1]]+100]
                helper.draw_slice(ax,mris[model],layer_meshes,axis,focus[axis],limits,f'{label}: {"XYZ"[axis]}={focus[axis]:g} mm')
        axes[0,0].legend(fontsize=8,loc='upper left')
        fig.suptitle(model+': scalp / outer skull / inner skull / brain-boundary coverage\nTop: facial/inferior region; bottom: cranial control. No numerical BEM certification.',fontsize=14)
        fig.savefig(out/'figures'/f'{model}_BEM_boundary_coverage.png',dpi=args.dpi);plt.close(fig)
    # Feasibility only: no standard montage loader (which can auto-normalize),
    # no placement, projection, fit, scaling or numbering inference.
    montage_root=ROOT/'cross_subject_fsaverage/template_montage_no_digitization'
    assumption_path=montage_root/'assumptions/numbered_hydrocel_v1/decision.json'
    assumption=json.loads(use(assumption_path).read_text()) if assumption_path.is_file() else {}
    mne_package=Path(helper.mne.__file__).resolve().parent
    resource=use(mne_package/'channels/data/montages/GSN-HydroCel-256.sfp')
    raw=np.loadtxt(resource,dtype=str);labels=raw[:,0].tolist()
    loader=use(mne_package/'channels/_standard_montage_utils.py')
    for path in [montage_root/'source/01_montage/README.md',montage_root/'source/01_montage/documentary_review_2026-11-06/followup_01/README.md']:
        if path.is_file():use(path)
    montage=dict(status='FEASIBLE_IN_PRINCIPLE_SEPARATE_ASSUMED_EXPERIMENT_NOT_PLACED',resource=str(resource),resource_sha256=sha(resource),
        channels=len([x for x in labels if re.fullmatch(r'E\d+',x)]),fiducials=[x for x in labels if x.startswith('Fid')],
        physical_numbering_verified=assumption.get('physical_channel_correspondence_verified',False),existing_assumption=assumption,
        scaling_applied=False,placement_applied=False,
        proposed_equivalent_procedure=['Pin one exact HydroCel resource and one declared physical-unit convention for both arms.',
            'Use the same unscaled sensor coordinates; rigid fiducial-frame placement only, identically defined using reviewed landmarks in each anatomy.',
            'Report off-scalp distances and missing-support sensors in both arms; do not project sensors onto a cut-away facial boundary.',
            'Keep template montage as a separate modeling assumption, not recovered participant digitization; validate eN to EN or retain ASSUMED_UNVERIFIED.',
            'If the resource has no verified physical scale, leave placement blocked pending that convention; do not silently use MNE 95 mm normalization.'],
        limitation='Prior assumed experiment explicitly used 95 mm normalization. This diagnostic neither repeats nor adopts that normalization. Anatomical coverage and landmark ambiguity prevent declaring equivalent anatomical validity.')
    write_json(out/'montage_feasibility.json',montage)
    origin=dict(documented_pydeface_subjects=[r['subject'] for r in sidecars if 'pydeface' in json.dumps(r['metadata']).lower()],
        release_README_original_surface_statement='BEM, pial and inflated surfaces created with original MRIs; released MRIs anonymized',
        conclusion='Facial removal is documented in source images before template construction. Averaging is not the sole origin. Additional registration/averaging effects remain possible.',
        upstream_unanonymized_images_used=False,exact_additional_template_coverage_loss_quantified=False)
    write_json(out/'anatomy_origin_assessment.json',origin)
    tables={'01_anonymization_evidence':evidence,'02_channel_anatomy_coverage':channel_rows,
        '03_regional_residual_contributions':contributions,'04_subset_sensitivity':sensitivity,
        '05_preserved_left_right_comparison':left_rows,'06_mirrored_native_proxy_pairs':pair_rows,
        '07_construction_slice_support':stage_rows,'08_BEM_layer_coverage_cues':layers_rows,
        '09_residual_plane_validation':figure_rows,'13_existing_inner_skull_repair_comparison':repair_rows}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    enriched.to_csv(out/'10_all_electrodes_with_coverage.csv',index=False)
    write_json(out/'BEM_layer_inventory.json',layer_manifest)
    pd.DataFrame(channel_rows).groupby(['channel','region_proxy','native_mri_coverage']).size().rename('participants').reset_index().to_csv(out/'11_channel_coverage_consensus.csv',index=False)
    # Keep original full montage metrics verbatim alongside sensitivity results.
    original_path=use(args.alignment/'03_model_distances.csv')
    (out/'12_original_full_montage_results.csv').write_bytes(original_path.read_bytes())
    st=pd.DataFrame(sensitivity);lr=pd.DataFrame(left_rows);ctr=pd.DataFrame(contributions)
    report=['# Anatomical coverage investigation','',
        'Status: diagnostics complete; anatomy, channel identity and coordinate-convention review remain pending. No reconstruction, scaling, surface edits, refitting, or forward/inverse computation was performed.','',
        '## Origin of missing anatomy','',origin['conclusion'],'',
        f"MRI sidecars documenting Pydeface: {origin['documented_pydeface_subjects']}. The release exporter copies T1_pydefaced.nii to the T1w inputs listed in the completed template construction. README documents original-MRI surfaces separately. See 01_anonymization_evidence.csv and verified construction input hashes.",'',
        'This establishes upstream anonymization, not the exact amount of additional loss introduced by registration or segmentation. No unredacted image is reconstructed or inferred as ground truth. Existing maskface MRI is compared only as an alternative released representation, not adopted.','',
        '## Channel and anatomical coverage proxies','',
        pd.DataFrame(channel_rows).groupby(['subject','region_proxy','native_mri_coverage']).size().rename('n').to_string(),'',
        'Region proxies use the closest point on original native scalp, in native surface RAS mm. Facial: Z<15, Y>=40, Y>=0.65|X|. Inferior lateral: Z<15 and |X|>=40 except facial. Cranial: Z>=15. Remaining locations are transition/uncertain. These rules are explicit geometric approximations, not validated cap-anatomy labels. The e238–e252 block is not automatically called facial.','',
        'Coverage uses 27 MRI probes per native closest point: depths 4/8/12 mm inward radially and tangent offsets -4/0/4 mm. Conservative agreement across 1/5/10% of positive-intensity p99 defines MRI_SUPPORTED_PROXY (>=80% probes), LOW_MRI_SUPPORT_PROXY (<=20%), otherwise uncertain; nonfinite/out-of-FOV probes remain uncertain. Probe thresholds are diagnostic coverage definitions, not alignment thresholds. Deep tissue support does not prove that superficial scalp is intact.','',
        'Labels use native MRI/scalp geometry, never template residual size. They are fixed across the three models for each participant. Native electrode/scalp correspondence remains uncertain, so no channel is declared independently anatomically verified.','',
        '## Residual contribution and sensitivity','',ctr.to_string(index=False),'',
        'Nearest-vertex full-montage median <=10 mm and p95 <=20 mm remain the original criteria. Original metrics are reproduced and retained verbatim in 12_original_full_montage_results.csv. Subset limits are descriptive comparisons only, not new alignment passes. No ICP is repeated and no electrode is removed from the primary experiment.','',
        f'Common MRI-supported subset: {len(common_supported)} channels. Common MRI-supported cranial subset: {len(common_cranial)} channels. Both require consensus across all seven participants and use identical labels in both template arms. No residual-based cutoff, fitted transform or physical channel mapping is introduced.','',
        st[st.model!='coverage_definition'].drop(columns=['selected_channels_json'],errors='ignore').to_string(index=False),'',
        '## Remaining left-lateral mismatch','',lr.to_string(index=False),'',
        'Left/right tests restrict to MRI-supported cranial proxies at X<-40 versus X>40 mm and report mean residual vector components as well as distances. Mirror-pair differences use native scalp q mirrored about X=0, nearest right q within 15 mm; these are geometric pairs, not validated physical net pairs, and may reuse right electrodes. Persistence in these regions supports a problem beyond facial coverage, without identifying whether it is head shape, cap geometry or coordinate conventions.','',
        '## MRI planes and BEM boundaries','',
        'The first two target panels are orthogonal oblique planes containing the full 3D residual; the third is axial through the nearest scalp point and explicitly omits off-plane electrodes. Stored plane origins/bases permit reproduction. Construction progression and registered-input panels extend inward from nearest scalp, avoiding the prior blank electrode-centered planes. Earlier stages/retained warped inputs may precede final shape updates and are not assumed anatomically identical. Blank-region fractions are exported so remaining limitations are visible.','',
        'Existing original-versus-repaired inner-skull displacements are reported in 13_existing_inner_skull_repair_comparison.csv without adopting the repair. Brain surface is a segmentation boundary proxy, not a certified cortical boundary.','',
        'All nine native/template heads receive scalp/outer-skull/inner-skull/available brain-boundary overlays through facial and cranial-control planes. Average-seven also shows its existing repaired inner skull candidate without adoption. Layer MRI-support fractions are review cues only: naturally dark skull/CSF can cause low support. These checks cannot certify intact skull compartments, conductivity boundaries, nesting or BEM numerical/anatomical validity. Existing repair review remains pending.','',
        '## Common HydroCel montage feasibility','',
        'A common GSN-HydroCel-256 montage is technically possible as a separate assumed model if one resource, one justified physical scale convention, and the same rigid fiducial-placement procedure are used in both arms. No montage is placed here. The installed SFP and loader are hashed. The existing numbered_hydrocel_v1 decision is ASSUMED_UNVERIFIED, not proof of eN-to-EN physical correspondence. Its predefined 95 mm normalization is not applied by this investigation.','',
        'Projection onto the defaced scalp would systematically relocate facial sensors into the cut-away boundary. Equivalent code does not guarantee equivalent anatomical placement; missing-support sensors and unreviewed landmarks must remain explicit. Do not infer participant digitization from a template montage.','',
        '## Recommendation','',
        'Retain the current average-seven template and full-montage outcomes as the preserved baseline; do not declare the template globally anatomically invalid. Investigate alternative anatomical construction only with existing appropriately governed data or anatomically justified resources, treating it as a separate experiment. No reconstruction of removed faces is justified by these diagnostics.','',
        'A separate reduced-montage sensitivity experiment is reasonable to plan if coverage-supported cranial subsets have sufficient cross-participant membership. Use the exported common subsets as explicit provisional definitions, review anatomical labels independently, freeze the same channel set/procedure in both arms and report all seven participants. Current statistics are fixed-transform sensitivity only; they do not predict outcomes after refitting or re-estimating inverse models.','',
        'Prioritize remaining supported-left residuals and layered MRI review before accepting a template/BEM or moving to forward modeling. If left mismatch persists on the supported subset, facial removal is a contributor rather than a complete explanation. No automated selection of a favorable subset or new primary acceptance rule is adopted.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    for path,value in hashes.items():
        if sha(path)!=value:raise ValueError(f'Input changed during run: {path}')
    # Exact full-residual oblique-plane construction is analytically verified.
    coplanar=True
    for row in figure_rows:
        basis=np.array(row['basis_rows']);coplanar &= bool(np.allclose(basis@basis.T,np.eye(3),atol=1e-8))
    if not coplanar:raise ValueError('Nonorthonormal MRI plane')
    validation.update(status='COMPLETE',full_montage_reproduced=True,oblique_bases_orthonormal=True,
        n_low_support_target_planes=sum(r['positive_intensity_fraction']<.005 for r in figure_rows),
        n_common_MRI_supported=len(common_supported),n_common_MRI_supported_cranial=len(common_cranial),
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,nibabel=nib.__version__,mne=helper.mne.__version__),
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='validation_manifest.json'})
    save();print(out,flush=True)


if __name__=='__main__':main()
