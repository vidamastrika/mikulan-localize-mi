#!/usr/bin/env python3
"""Stage 1 only. Read-only paired-MRI feasibility; never starts construction."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import pandas as pd
from input_utils import MRI,apply,native_bridge,image_header,compare_samples,support_states,region_labels

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'localize_mi_average_seven'
VAR=EXP/'maskface_template'
GEO=EXP/'diagnostics/02_scalp_geometry'
sys.path.insert(0,str(GEO))
from geometry import plane_segments


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')


def surface(path):
    image=nib.load(path);p=next(x for x in image.darrays if x.intent==1008);f=next(x for x in image.darrays if x.intent==1009)
    v,faces=np.asarray(p.data,float),np.asarray(f.data,int)
    if not np.isfinite(v).all() or faces.min()<0 or faces.max()>=len(v):raise ValueError(f'Invalid surface geometry: {path}')
    bridge,scanner,size=native_bridge(dict(p.meta))
    return v,faces,bridge,scanner,size


def paired_sample_table(py,mask,bridge,py_scale,mask_scale,stride=2):
    """Deterministic paired voxel samples; no resampling or intensity fitting."""
    to_surface=np.linalg.inv(bridge)@py.image.affine
    chunks={label:[[],[]] for label in ['facial_proxy','inferior_lateral_proxy','cranial_proxy','other_or_transition']}
    nx,ny,nz=py.data.shape
    xx,yy=np.meshgrid(np.arange(0,nx,stride),np.arange(0,ny,stride),indexing='ij')
    for z in range(0,nz,stride):
        voxel=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,z)])
        points=apply(to_surface,voxel);labels=region_labels(points)
        a=py.data[::stride,::stride,z].ravel();b=mask.data[::stride,::stride,z].ravel()
        for label in chunks:
            selected=labels==label;chunks[label][0].append(a[selected]);chunks[label][1].append(b[selected])
    rows=[];scale=max(py_scale,mask_scale,1e-12)
    for label,parts in chunks.items():
        a,b=(np.concatenate(part) for part in parts)
        finite=np.isfinite(a)&np.isfinite(b)
        for fraction in [.01,.05,.10]:
            states=support_states(a,b,py_scale,mask_scale,fraction)
            common=(states=='COMMON_SIGNAL')&finite
            metrics=compare_samples(a,b,common,common,scale)
            for state in ['COMMON_SIGNAL','MASKFACE_ONLY_SIGNAL_UNVERIFIED','PYDEFACE_ONLY_SIGNAL','BACKGROUND_OR_LOW_SIGNAL']:
                selected=(states==state)&finite
                rows.append(dict(region_proxy=label,intensity_fraction_p99=fraction,state=state,
                    n_samples=int(selected.sum()),n_region_samples=len(a),sample_stride_voxels=stride,
                    common_signal_exact_equal_fraction=metrics['exact_equal_fraction'],
                    common_signal_normalized_absdiff_p95=metrics['normalized_absdiff_p95'],common_signal_pearson=metrics['pearson'],
                    interpretation='sampled voxel support, not tissue identity; common-signal metrics repeated for each state'))
    return rows


def compare_figure(subject,py,mask,meshes,focus,path,dpi,description):
    fig,axes=plt.subplots(3,3,figsize=(20,17),layout='constrained')
    scale=max(py.vmax,mask.vmax,1e-12)
    for row,(axis,name) in enumerate([(2,'axial'),(1,'coronal'),(0,'sagittal')]):
        rem=[i for i in range(3) if i!=axis]
        limits=[focus[rem[0]]-100,focus[rem[0]]+100,focus[rem[1]]-100,focus[rem[1]]+100]
        a,extent=py.slice(axis,focus[axis],limits,.75);b,_=mask.slice(axis,focus[axis],limits,.75)
        for col,(values,title) in enumerate([(a,'Pydeface'),(b,'maskface'),((b-a)/scale,'maskface − Pydeface / shared p99.5')]):
            ax=axes[row,col]
            if col<2:ax.imshow(values,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=scale)
            else:
                im=ax.imshow(values,origin='lower',extent=extent,cmap='coolwarm',vmin=-.5,vmax=.5)
                fig.colorbar(im,ax=ax,shrink=.7,label='normalized intensity difference (display only)')
            for label,v,f,color in meshes:
                seg=plane_segments(v,f,axis,focus[axis])
                if len(seg):ax.add_collection(LineCollection(seg,colors=color,linewidths=.9,label=label))
            ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='XYZ'[rem[0]]+' (mm)',ylabel='XYZ'[rem[1]]+' (mm)',
                title=f'{title}\n{name}: {"XYZ"[axis]}={focus[axis]:g} mm')
            ax.set_aspect('equal')
    axes[0,0].legend(fontsize=8,loc='upper left')
    fig.suptitle(f'{subject}: {description}\nPhysical native surface RAS mm; original surfaces are references, not maskface segmentations. No MRI registration.',fontsize=14)
    fig.savefig(path,dpi=dpi);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/Localize-MI')
    parser.add_argument('--previous',type=Path,default=EXP/'diagnostics/03_anatomical_coverage/run_01')
    parser.add_argument('--output-root',type=Path,default=VAR/'outputs/01_input_feasibility/run_01')
    parser.add_argument('--dpi',type=int,default=220)
    args=parser.parse_args();out=args.output_root.resolve()
    if (VAR/'outputs').resolve() not in out.parents:raise ValueError('Outputs must be inside maskface_template/outputs')
    if out.exists():raise FileExistsError(f'Refusing existing output: {out}; use a fresh numbered run')
    if args.dpi<180:parser.error('Use at least 180 DPI')
    hashes={}
    def use(path):
        path=Path(path).resolve()
        if str(path) not in hashes:hashes[str(path)]=sha(path)
        return path
    for path in [Path(__file__),Path(__file__).with_name('input_utils.py'),GEO/'geometry.py',VAR/'config/variant.json',VAR/'docs/METHOD_EVIDENCE.md']:
        use(path)
    config=json.loads((VAR/'config/variant.json').read_text())
    original=use(ROOT/config['original_configuration'])
    if sha(original)!=config['original_configuration_sha256']:raise ValueError('Original experiment configuration changed')
    original_cfg=json.loads(original.read_text())
    if config['stage2_authorized'] or config['stage']!='01_INPUT_FEASIBILITY_ONLY':raise ValueError('Expected Stage 1-only variant configuration')
    if original_cfg['coregistration']['scaling'] or original_cfg['coregistration']['median_limit_mm']!=10 or original_cfg['coregistration']['p95_limit_mm']!=20:
        raise ValueError('Original alignment thresholds/scaling policy changed')
    prior_manifest=json.loads(use(args.previous/'validation_manifest.json').read_text())
    if prior_manifest['status']!='COMPLETE':raise ValueError('Coverage diagnostic not complete')
    for path,value in prior_manifest['input_sha256'].items():
        if sha(path)!=value:raise ValueError(f'Prior input changed: {path}')
        use(path)
    for path,value in prior_manifest['output_sha256'].items():
        if sha(args.previous/path)!=value:raise ValueError(f'Prior output changed: {path}')
        use(args.previous/path)
    review=args.previous.parent/'REVIEW_run_01.md'
    if review.is_file():use(review).read_text()
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    manifest=dict(status='RUNNING',created_utc=datetime.now(timezone.utc).isoformat(),stage=1,argv=sys.argv,
        stage2_authorized=False,construction_executed=False,FreeSurfer_executed=False,segmentation_executed=False,
        scaling=False,face_reconstruction=False,source_or_surface_modification=False,alignment_executed=False,forward=False,inverse=False,
        feasibility_review='PENDING',original_full_montage_thresholds_mm=dict(median=10,p95=20))
    def save():manifest['input_sha256']=hashes;write_json(out/'manifest.json',manifest)
    save()
    headers=[];pairs=[];regional=[];profiles=[];core_rows=[];evidence=[];inventory=[];triage=[]
    for path,pattern,role in [(ROOT/'_08_bids_conversion.py',r'T1_mf|T1_pydefaced|maskface|T1w.nii','MRI export code'),
        (ROOT/'fx_bids.py',r'Milchenko|Face Masking|Pydeface|ImageProcessingSoftware','sidecar generation'),
        (ROOT/'_99_make_anon_bems.sh',r'T1_mf|mri_watershed|mask face','historical watershed invocation; not proof of released outputs'),
        (args.dataset/'README.md',r'original MRI|anonym','release documentation')]:
        for number,line in enumerate(use(path).read_text(errors='replace').splitlines(),1):
            if re.search(pattern,line,re.I):evidence.append(dict(role=role,path=str(path.resolve()),line=number,text=line.strip()))
    for subject in config['participants']:
        print('Stage 1 input feasibility:',subject,flush=True)
        py_path=args.dataset/config['original_T1w_pattern'].format(subject=subject)
        mask_path=args.dataset/config['maskface_pattern'].format(subject=subject)
        if not py_path.is_file() or not mask_path.is_file():
            triage.append(dict(subject=subject,technical_inputs_valid=False,reason='Missing paired MRI',same_grid=False));continue
        metadata={}
        for name,path in [('Pydeface',py_path),('maskface',mask_path)]:
            p=use(path);img=nib.load(p);head=image_header(img)
            meta_path=path.with_suffix('.json')
            side=json.loads(use(meta_path).read_text()) if meta_path.is_file() else {}
            metadata[name]=side
            inventory.append(dict(subject=subject,version=name,path=str(p),sha256=sha(p),bytes=p.stat().st_size,
                processing_metadata_json=json.dumps(side),sidecar_present=meta_path.is_file()))
            for key,value in side.items():evidence.append(dict(role=f'{subject} {name} sidecar',path=str(meta_path.resolve()),line=0,text=json.dumps({key:value})))
            headers.append(dict(subject=subject,version=name,path=str(p),**head))
        py_image=nib.load(py_path);mask_image=nib.load(mask_path)
        technical=all(r['technical_header_valid'] for r in headers if r['subject']==subject)
        if not technical:
            triage.append(dict(subject=subject,technical_inputs_valid=False,reason='Invalid/ambiguous MRI header or units',same_grid=False));continue
        skin_path=use(args.dataset/'derivatives/sourcemodelling'/subject/'anat'/f'{subject}_outer_skin.surf.gii')
        v,f,bridge,scanner,size=surface(skin_path)
        if tuple(size)!=tuple(py_image.shape) or not np.allclose(scanner,py_image.affine,atol=1e-3):
            triage.append(dict(subject=subject,technical_inputs_valid=False,reason='Native GIFTI/Pydeface geometry mismatch',same_grid=False));continue
        py=MRI(py_path,bridge);mask=MRI(mask_path,bridge)
        same_grid=bool(py.data.shape==mask.data.shape and np.allclose(py_image.affine,mask_image.affine,atol=1e-4,rtol=0))
        finite=bool(np.isfinite(py.data).all() and np.isfinite(mask.data).all())
        positive_py=py.data[np.isfinite(py.data)&(py.data>0)];positive_mask=mask.data[np.isfinite(mask.data)&(mask.data>0)]
        scales=[float(np.percentile(x,99)) if len(x) else 0. for x in [positive_py,positive_mask]]
        pairs.append(dict(subject=subject,same_voxel_grid=same_grid,
            affine_max_difference_mm=float(np.max(np.abs(py_image.affine-mask_image.affine))),
            mask_voxel_to_py_voxel_json=json.dumps((np.linalg.inv(py_image.affine)@mask_image.affine).tolist()),
            Pydeface_nonfinite_voxels=int((~np.isfinite(py.data)).sum()),maskface_nonfinite_voxels=int((~np.isfinite(mask.data)).sum()),
            Pydeface_positive_p99=scales[0],maskface_positive_p99=scales[1],
            maskface_provenance_mentions_face_masking='mask' in json.dumps(metadata['maskface']).lower(),
            surface_to_scanner_ras_mm_json=json.dumps(bridge.tolist()),
            interpretation='same grid is geometric consistency, not proof of identical acquisition/unmodified anatomy; physical sampling uses actual headers'))
        technical=technical and finite and min(scales)>0
        triage.append(dict(subject=subject,technical_inputs_valid=bool(technical),same_grid=same_grid,
            reason='Readable finite mm MRI pair; visual/tissue-modification review required' if technical else 'Nonfinite or empty image',
            FreeSurfer_suitability='NOT_ESTABLISHED_INPUT_TRIAGE_ONLY',scalp_suitability='MODIFIED_BOUNDARY_REQUIRES_REVIEW'))
        if same_grid:
            for row in paired_sample_table(py,mask,bridge,*scales):regional.append(dict(subject=subject,**row))
        # Layer geometry locates sampling only. No new surface segmentation.
        meshes=[('original native scalp',v,f,'cyan')];center=np.median(v,axis=0)
        for layer,color in [('outer_skin','cyan'),('outer_skull','yellow'),('inner_skull','red'),('brain','lime')]:
            path=use(args.dataset/'derivatives/sourcemodelling'/subject/'anat'/f'{subject}_{layer}.surf.gii')
            lv,lf,lb,ls,sz=surface(path)
            if tuple(sz)!=tuple(py.data.shape) or not np.allclose(ls,py_image.affine,atol=1e-3):raise ValueError(f'{subject}/{layer}: native volume geometry mismatch')
            if layer!='outer_skin':meshes.append((layer,lv,lf,color))
            ids=np.unique(np.linspace(0,len(lv)-1,256,dtype=int));points=lv[ids]
            direction=points-center;direction/=np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-12)
            offsets=np.arange(-20.,20.1,1.);samples=points[:,None,:]+offsets[None,:,None]*direction[:,None,:]
            av,valid_a=py.sample(samples);bv,valid_b=mask.sample(samples);labels=region_labels(points)
            for j,index in enumerate(ids):
                for k,offset in enumerate(offsets):
                    profiles.append(dict(subject=subject,layer=layer,vertex=int(index),region_proxy=labels[j],offset_mm=float(offset),
                        Pydeface_intensity=float(av[j,k]) if np.isfinite(av[j,k]) else None,
                        maskface_intensity=float(bv[j,k]) if np.isfinite(bv[j,k]) else None,
                        both_in_fov=bool(valid_a[j,k]&valid_b[j,k]),
                        reference='original native surface radial profile; not independently validated layer/normal'))
            if layer=='brain':
                core_samples=points[:,None,:]-np.array([10.,20.])[None,:,None]*direction[:,None,:]
                a,va=py.sample(core_samples);b,vb=mask.sample(core_samples)
                row=compare_samples(a,b,va,vb,max(scales))
                core_rows.append(dict(subject=subject,**row,all_samples_in_fov=bool((va&vb).all()),
                    n_nonpositive_Pydeface=int((a<=0).sum()),n_nonpositive_maskface=int((b<=0).sum()),
                    interpretation='radially inward original brain-boundary proxy; not a brain mask or unmodified reference'))
        compare_figure(subject,py,mask,meshes,np.array([0.,60.,-20.]),out/'figures'/f'{subject}_facial_input_comparison.png',args.dpi,'facial/inferior input comparison')
        compare_figure(subject,py,mask,meshes,np.array([-45.,0.,45.]),out/'figures'/f'{subject}_cranial_input_comparison.png',args.dpi,'cranial/left-lateral control comparison')
        # Plot mean regional radial curves on the exact same reference surfaces.
        rows=pd.DataFrame([r for r in profiles if r['subject']==subject])
        fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
        for ax,layer in zip(axes.flat,['outer_skin','outer_skull','inner_skull','brain']):
            part=rows[rows.layer==layer]
            for region,color in [('facial_proxy','tab:orange'),('cranial_proxy','tab:blue'),('inferior_lateral_proxy','tab:green')]:
                group=part[(part.region_proxy==region)&part.both_in_fov].groupby('offset_mm')[['Pydeface_intensity','maskface_intensity']].mean()
                if len(group):
                    ax.plot(group.index,group.Pydeface_intensity,color=color,ls='--',label=region+' Pydeface')
                    ax.plot(group.index,group.maskface_intensity,color=color,label=region+' maskface')
            ax.axvline(0,color='gray',lw=.7);ax.set(title=layer,xlabel='radial offset from original surface (mm; negative inward)',ylabel='raw MRI intensity')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(subject+': regional image profiles; tissue identity and radial-normal convention are proxies',fontsize=13)
        fig.savefig(out/'figures'/f'{subject}_layer_intensity_profiles.png',dpi=args.dpi);plt.close(fig)
        del py,mask,positive_py,positive_mask
        save()
    # Aggregate profile agreement without treating skull darkness as tissue loss.
    pd_profiles=pd.DataFrame(profiles);layer_summary=[]
    if len(pd_profiles):
        for (subject,layer,region),group in pd_profiles.groupby(['subject','layer','region_proxy']):
            scale=max(r['Pydeface_positive_p99'] for r in pairs if r['subject']==subject)
            for band,subset in [('inward_20_to_5mm',group[group.offset_mm.between(-20,-5)]),('near_surface_4mm',group[group.offset_mm.between(-4,4)]),('outward_5_to_20mm',group[group.offset_mm.between(5,20)])]:
                a=subset.Pydeface_intensity.to_numpy(float);b=subset.maskface_intensity.to_numpy(float);valid=subset.both_in_fov.to_numpy(bool)
                layer_summary.append(dict(subject=subject,layer=layer,region_proxy=region,band=band,
                    **compare_samples(a,b,valid,valid,scale),interpretation='raw signal consistency only; not FreeSurfer or BEM anatomical validation'))
    dependencies=dict(ANTs=shutil.which('antsMultivariateTemplateConstruction2.sh'),FreeSurfer=shutil.which('recon-all'),
        note='Availability on PATH only. No commands executed; original experiment also used containerized FreeSurfer.')
    write_json(out/'processing_feasibility.json',dict(status='REVIEW_REQUIRED',dependency_inventory=dependencies,
        FreeSurfer='Readable suitable T1 geometry is necessary but does not establish reconstruction success; no recon-all executed.',
        scalp='Modified facial/ear surface may smooth or displace the apparent skin boundary; preserved intensity does not prove an anatomically correct scalp.',
        BEM='No skull segmentation, nesting/intersection checks, skull thickness or source containment is newly validated in Stage 1.',
        provenance_limit='Release method citation identifies a family of boundary-layer modifications; exact variant/version/parameters/altered-voxel mask/brain-protection settings unavailable.',
        additional_signal='MASKFACE_ONLY_SIGNAL_UNVERIFIED is neither certified original anatomy nor proven wholly synthetic tissue; original unmodified reference unavailable.',
        no_automatic_stage2=True))
    tables={'01_input_inventory':inventory,'02_pair_geometry':pairs,'03_regional_signal_support':regional,
        '04_layer_radial_samples':profiles,'05_layer_signal_agreement':layer_summary,'06_brain_inward_proxy':core_rows,
        '07_provenance_evidence':evidence,'08_subject_feasibility_triage':triage}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    write_json(out/'MRI_headers.json',headers)
    all_technical=len(triage)==7 and all(r['technical_inputs_valid'] for r in triage)
    all_same_grid=len(pairs)==7 and all(r['same_voxel_grid'] for r in pairs)
    disposition='REVIEW_REQUIRED_STAGE2_NOT_AUTHORIZED' if all_technical else 'TECHNICAL_INPUT_ISSUES_STAGE2_NOT_AUTHORIZED'
    write_json(out/'stage_gate.json',dict(disposition=disposition,all_seven_technical_inputs_valid=all_technical,
        all_seven_pair_grids_match=all_same_grid,stage2_authorized=False,
        review_requirements=['Inspect all paired cranial and facial slices and brain inward-proxy discrepancies.',
            'Distinguish increased coverage from modified/smoothed surface shape; review skull and brain controls.',
            'Review method/version uncertainty and whether modified anatomy is acceptable for a separate experimental template.',
            'Explicitly review Stage 1 results before preparing or executing Stage 2.'],
        future_stage2='Reuse original registration settings where justified, but only in new variant directories after review; no Stage 2 launcher emitted.'))
    report=['# Maskface template variant — Stage 1 input feasibility','',f'Status: {disposition}. Stage 2 has NOT been authorized or executed.','',
        '## Scope and provenance','',
        'This variant preserves the original average-seven experiment and all diagnostics. All seven existing maskface/Pydeface input pairs are located by explicit paths, hashed and audited. The release exporter copies T1_mf.nii to the maskface derivative and T1_pydefaced.nii to the original experiment T1w input. Sidecars identify their processing methods; exact masking execution logs and unmodified reference images are unavailable.','',
        'The cited masking method deliberately modifies a thin surface layer; it does not establish untouched facial anatomy. Its paper evaluates skull stripping and gain-field correction, not FreeSurfer/BEM validity for these data. See [Milchenko and Marcus](https://pmc.ncbi.nlm.nih.gov/articles/PMC3538950/), Methods and Evaluation, and the hashed docs/METHOD_EVIDENCE.md. Exact algorithm variant and parameters for this release remain unknown.','',
        '## Image geometry and relationship','',pd.DataFrame(triage).to_string(index=False),'',pd.DataFrame(pairs).to_string(index=False),'',
        'MRI_headers.json records actual qform/sform, NIfTI RAS affines, voxel dimensions/spacing, FOV and units. Native GIFTI volume geometry must match the Pydeface scanner affine before anatomical overlays are permitted. Maskface uses its own actual scanner affine; a same-grid comparison does not fit or assume a new registration. Different grids withhold direct voxel comparisons and require review. Display/profile sampling is trilinear in memory only.','',
        '## Anatomical signal preserved and modified','',
        '03_regional_signal_support.csv samples paired voxels at stride 2 and classifies common signal, maskface-only signal, Pydeface-only signal and low/background signal at 1%, 5%, 10% of each image positive p99. Common-signal exact equality, p95 differences and correlations exclude zero/background domination. These thresholds define exploratory signal support, not tissue classes, alignment acceptance or anatomical validity.','',
        'Additional maskface-only support has no unmodified paired reference. It may include usable retained deep structures as well as smoothed/modified boundary signal; its original-versus-filtered identity cannot be determined voxelwise from these two images. Common unchanged voxels are direct shared-data evidence, not certification of the entire image. No unmodified anatomy is reconstructed.','',
        '## Brain, skull and scalp controls','',pd.DataFrame(core_rows).to_string(index=False),'',
        'Radial profiles are sampled around the original native scalp, outer-skull, inner-skull and brain segmentation surfaces. Separate facial, inferior-lateral and cranial geometric regions and inward/near/outward bands expose where the two image versions agree or diverge. Brain inward probes at 10 and 20 mm are control proxies, not a certified eroded brain mask. Skull bone and CSF can be dark; low intensity is not by itself an anatomical defect. Original surfaces are references, not segmentations of the maskface MRI.','',
        '## Suitability for FreeSurfer and scalp segmentation','',
        'Finite, geometrically consistent T1 inputs support technical plausibility only. Stage 1 does not test recon-all completion, bias correction, cortical reconstruction, watershed behavior or BEM quality. Inspect paired cranial controls and inward brain signal before treating the inputs as suitable for an exploratory reconstruction. Modified facial profiles may yield a smoother, displaced or otherwise synthetic apparent scalp despite greater image support. Historical repository code invoked watershed on T1_mf.nii, but that is not evidence that these particular inputs generated valid surfaces.','',
        '## Review decision and next stages','',
        f'All seven technical input checks pass: {all_technical}. All seven grids match: {all_same_grid}. These findings do not open the stage gate.','',
        'Recommendation: retain this as a separate candidate variant. Decide after slice/provenance review whether maskface anatomy is acceptable for exploratory template construction; do not approve it from coverage gain alone. If brain/cranial differences or synthetic-boundary effects are material, investigate those before construction.','',
        'After review only, Stage 2 would reuse the original construction settings where appropriate in a distinct average_seven_maskface subject/work tree, followed by independent anatomical, skull/nesting/intersection/thickness and source-containment assessment. Stage 3 would reuse identical electrode inputs, rigid procedure, no scaling, and original median<=10 mm/p95<=20 mm gates; retain all seven subjects including sub-03 and compare all original arms. These later stages are documented intent, not executable actions emitted by Stage 1.','',
        'No template construction, FreeSurfer, new surface segmentation, scaling, face reconstruction, alignment, forward or inverse calculation was run. Original configurations and results remain untouched.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    for path,value in hashes.items():
        if sha(path)!=value:raise ValueError(f'Input changed during run: {path}')
    manifest.update(status='COMPLETE',disposition=disposition,technical_input_checks_pass=all_technical,
        all_pair_grids_match=all_same_grid,stage2_authorized=False,
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,nibabel=nib.__version__,matplotlib=matplotlib.__version__),
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    save();print(out,flush=True)


if __name__=='__main__':main()
