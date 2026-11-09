#!/usr/bin/env python3
# Read-only compatibility audit; user executes manually. NEVER imports/runs
# Script 01 or later experiment stages. No forward, BEM construction, inverse,
# EEG loading, individual MRI/digitization, ICP, warping, scaling search, sensor
# correction/projection or outcome-based montage selection. Writes NEW audit
# reports only; input hashes are checked again before publication.
# Inputs: EVIDENCE.json beside this script; installed pinned MNE 1.12.1 montage
# resources; existing assumed-v1 Script 01 manifest's fsaverage-only snapshots.
# Compare GSN-HydroCel-256, EGI_256, fsaverage_1005 availability and standard_1005
# (version-dependent 10-05 control, NEVER a substitute EEG channel map).
# Native template fiducials define HEAD; predefined fsaverage HEAD->MRI handles
# the second frame step. This is coordinate placement, not individualized fit.
# Existing 10 mm median/20 mm p95 nearest-vertex numerical gates remain intact;
# candidate QA does not establish physical Localize-MI channel correspondence.
# Point-to-triangle distances and winding classify raw candidate positions, with
# 0.001 mm surface and 1e-5 winding tolerances (diagnostics, not new fit gates).
# Templates are visualized even when mapping is blocked, with template names
# only. No executable dataset montage/Info/forward is produced from unverified
# eN->EN proposals. Same-name layout comparisons are NOT physical mapping proof.
# Dependencies: Python>=3.10, NumPy, pandas, SciPy, MNE 1.12.1, Matplotlib.
# Expected framework retained: 7 subjects/61 runs/12 methods; 732+4392=5124 jobs.
# Command from repository root (fresh output required):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/audits/compatibility_2026-11-09/audit_montage_compatibility.py
# Exit 3 = scientific BLOCKED with complete reports (expected); 2 invalid inputs;
# no automatic scientific PASS is possible from labels/geometry alone.
"""Audit template candidates and geometry without modifying any experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

VARIANT=Path(__file__).resolve().parents[2]
KINDS=['GSN-HydroCel-256','EGI_256','fsaverage_1005','standard_1005']


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def triangle_metrics(points,vertices,faces):
    """Finite triangle distances and closed-mesh winding; no point modification."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    undirected,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    balance=np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    adjacency=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices)))
    components=int(connected_components(adjacency,directed=False,return_labels=False))
    reliable=bool(np.all(counts==2) and np.all(balance==0) and components==1)
    topology=dict(vertices=len(vertices),triangles=len(faces),boundary_edges=int((counts==1).sum()),
                  nonmanifold_edges=int((counts>2).sum()),orientation_errors=int((balance!=0).sum()),
                  connected_components=components,closed_oriented_connected=reliable,self_intersections_checked=False)
    a,b,c=[vertices[faces[:,i]] for i in range(3)];ab=b-a;ac=c-a
    normal=np.cross(ab,ac);normal_sq=np.einsum('ij,ij->i',normal,normal)
    d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
    denom=d00*d11-d01*d01
    require(np.all(normal_sq>0) and np.all(denom>0),'Degenerate scalp triangles')
    distances=[];winding=[]
    for point in points:
        delta=point-a;plane=np.einsum('ij,ij->i',delta,normal)
        d20=np.einsum('ij,ij->i',delta,ab);d21=np.einsum('ij,ij->i',delta,ac)
        u=(d11*d20-d01*d21)/denom;v=(d00*d21-d01*d20)/denom
        squared=np.where((u>=0)&(v>=0)&(u+v<=1),plane**2/normal_sq,np.inf)
        for start,end in [(a,b),(b,c),(c,a)]:
            edge=end-start
            t=np.clip(np.einsum('ij,ij->i',point-start,edge)/np.einsum('ij,ij->i',edge,edge),0,1)
            residual=point-(start+t[:,None]*edge)
            squared=np.minimum(squared,np.einsum('ij,ij->i',residual,residual))
        distances.append(float(np.sqrt(squared.min())))
        va,vb,vc=a-point,b-point,c-point
        na,nb,nc=[np.linalg.norm(x,axis=1) for x in [va,vb,vc]]
        numerator=np.einsum('ij,ij->i',va,np.cross(vb,vc))
        denominator=na*nb*nc+np.einsum('ij,ij->i',va,vb)*nc+np.einsum('ij,ij->i',vb,vc)*na+np.einsum('ij,ij->i',vc,va)*nb
        winding.append(float(np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi)))
    return np.array(distances),np.array(winding),topology


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=VARIANT/'outputs/00_montage_compatibility_audit_2026-11-09')
    args=p.parse_args(argv);out=args.output_root.expanduser().resolve()
    require((VARIANT/'outputs').resolve() in out.parents and not out.exists(),'Use a fresh output directory inside this variant/outputs')
    require(importlib.metadata.version('mne')=='1.12.1','Audit the frozen MNE 1.12.1 environment; do not upgrade the experiment to obtain fsaverage_1005')
    inputs={}
    def verify(path,expected=None):
        path=Path(path).resolve();value=sha(path)
        require(expected is None or expected==value,f'Input changed: {path}')
        require(path.parent not in out.parents and out!=path.parent,'Output overlaps an input directory')
        inputs[str(path)]=value;return path
    evidence=json.loads(verify(Path(__file__).with_name('EVIDENCE.json')).read_text())
    require(evidence['physical_channel_mapping_verified'] is False,'New mapping evidence needs documentary review; do not flip this flag to force PASS')
    verify(Path(__file__))
    preparation=VARIANT/'outputs/01_prepare_experiment_assumed_v1'
    manifest=json.loads(verify(preparation/'01_experiment_manifest.json').read_text())
    resources={}
    for role in ['fsaverage_scalp','fsaverage_head_mri','fsaverage_fiducials']:
        matches=[i for i in manifest['source_inventory'] if i['role']==role]
        require(len(matches)==1,f'Ambiguous fsaverage snapshot: {role}')
        resources[role]=verify(matches[0]['copy_path'],matches[0]['sha256'])
    import mne
    from mne.io.constants import FIFF
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    package=Path(mne.__file__).parent
    for relative in ['channels/montage.py','channels/_standard_montage_utils.py','transforms.py']:
        verify(package/relative)
    trans=mne.read_trans(resources['fsaverage_head_mri'],verbose=False)
    if (int(trans['from']),int(trans['to']))==(int(FIFF.FIFFV_COORD_MRI),int(FIFF.FIFFV_COORD_HEAD)):
        trans=mne.transforms.invert_transform(trans)
    require((int(trans['from']),int(trans['to']))==(int(FIFF.FIFFV_COORD_HEAD),int(FIFF.FIFFV_COORD_MRI)),'Unexpected predefined transform frames')
    matrix=np.asarray(trans['trans'])
    require(np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5,rtol=0)
            and abs(np.linalg.det(matrix[:3,:3])-1)<1e-5,'Nonrigid fsaverage transform')
    vertices,faces=mne.read_surface(resources['fsaverage_scalp'],verbose=False);vertices=np.asarray(vertices,float)/1000
    tree=cKDTree(vertices)
    fiducials,frame=mne.io.read_fiducials(resources['fsaverage_fiducials'],verbose=False)
    require(int(frame)==int(FIFF.FIFFV_COORD_MRI),'Fsaverage fiducials must be MRI coordinates')
    target_fids={int(f['ident']):np.asarray(f['r']) for f in fiducials}
    available=mne.channels.get_builtin_montages();catalog=[];geometries=[];summaries=[];fids=[];transforms={};layouts={}
    for kind in KINDS:
        entry=dict(montage=kind,available=kind in available,physical_dataset_mapping_verified=False)
        if kind not in available:
            entry['note']='Absent in pinned MNE; fsaverage_1005 was added in 1.13; no replacement or upgrade performed'
            catalog.append(entry);continue
        paths=list((package/'channels/data/montages').glob(kind+'.*'))
        require(len(paths)==1,f'Ambiguous resource: {kind}')
        resource=verify(paths[0])
        if kind=='GSN-HydroCel-256':require(sha(resource)==evidence['pinned_hydrocel_sha256'],'HydroCel resource differs from reviewed definition')
        montage=mne.channels.make_standard_montage(kind,head_size='auto');pos=montage.get_positions()
        names=montage.ch_names;native=np.array([pos['ch_pos'][n] for n in names])
        require(np.isfinite(native).all() and len(set(names))==len(names),'Invalid template coordinate table')
        if pos['coord_frame']=='unknown':
            native_head=mne.channels.compute_native_head_t(montage,verbose=False)
            composite=matrix@native_head['trans'];mri=mne.transforms.apply_trans(composite,native)
            policy='template fiducial native->HEAD then predefined fsaverage HEAD->MRI; no fitting'
        elif pos['coord_frame']=='mri':
            composite=np.eye(4);mri=native.copy()
            policy='MRI-labelled resource shown as supplied; 10-05 control only, no EEG mapping or additional fit'
        else:raise ValueError(f'Unsupported template frame: {pos["coord_frame"]}')
        entry.update(n_electrodes=len(names),label_set_e1_e256=len(names)==256 and {n.casefold() for n in names}=={f'e{i}' for i in range(1,257)},
            coordinate_frame=pos['coord_frame'],coordinate_units='MNE output metres',
            scaling='fixed auto 95mm normalization' if pos['coord_frame']=='unknown' else 'MRI resource units handled by installed parser; no auto spherical normalization',
            raw_physical_units='not established solely by this resource; normalization is a model convention',
            fiducial_source='E31 and weighted E67/E94, E219/E190 electrode estimates' if kind=='EGI_256' else 'resource fiducials',
            resource_path=str(resource),resource_sha256=sha(resource),geometry_policy=policy)
        catalog.append(entry);transforms[kind]=dict(native_to_mri=composite.tolist(),native_frame=pos['coord_frame'],policy=policy)
        dvertex,_=tree.query(mri);dtriangle,winding,topology=triangle_metrics(mri,vertices,faces)
        relation=np.full(len(names),'AMBIGUOUS',dtype=object)
        if topology['closed_oriented_connected']:
            relation[np.abs(winding)<1e-5]='OUTSIDE';relation[np.abs(np.abs(winding)-1)<1e-5]='INSIDE'
            relation[dtriangle<=1e-6]='ON_SURFACE'
        for name,point,dv,dt,win,rel in zip(names,mri,dvertex,dtriangle,winding,relation):
            geometries.append(dict(montage=kind,template_channel=name,fsaverage_mri_x_m=point[0],fsaverage_mri_y_m=point[1],fsaverage_mri_z_m=point[2],
                nearest_vertex_distance_mm=dv*1000,nearest_triangle_distance_mm=dt*1000,winding_number=win,scalp_relation=rel,
                physical_dataset_mapping_verified=False,coordinates_corrected=False))
        summaries.append(dict(montage=kind,n_electrodes=len(names),vertex_median_mm=float(np.median(dvertex)*1000),
            vertex_p95_mm=float(np.percentile(dvertex,95)*1000),triangle_median_mm=float(np.median(dtriangle)*1000),
            triangle_p95_mm=float(np.percentile(dtriangle,95)*1000),triangle_maximum_mm=float(dtriangle.max()*1000),
            n_inside=int((relation=='INSIDE').sum()),n_outside=int((relation=='OUTSIDE').sum()),
            original_vertex_gates_pass=bool(np.median(dvertex)*1000<=10 and np.percentile(dvertex,95)*1000<=20),
            anatomical_validation='NOT_ESTABLISHED',mesh=topology))
        for label,identifier in [('nasion',FIFF.FIFFV_POINT_NASION),('lpa',FIFF.FIFFV_POINT_LPA),('rpa',FIFF.FIFFV_POINT_RPA)]:
            if pos[label] is not None:
                point=mne.transforms.apply_trans(composite,pos[label])
                fids.append(dict(montage=kind,fiducial=label,residual_mm=float(np.linalg.norm(point-target_fids[int(identifier)])*1000),
                                 interpretation='convention diagnostic, not a fit or anatomical PASS'))
        layouts[kind]=(names,mri)
    mapping=[dict(dataset_channel=f'e{i}',proposed_template_channel=f'E{i}',
        mapping_status='BLOCKED_UNVERIFIED_PHYSICAL_CORRESPONDENCE',physical_mapping_verified=False,
        justification='label-number agreement is a hypothesis; custom acquisition/export permutation unverified') for i in range(1,257)]
    differences=[]
    if all(k in layouts for k in ['GSN-HydroCel-256','EGI_256']):
        a=dict(zip(*layouts['GSN-HydroCel-256']));b=dict(zip(*layouts['EGI_256']))
        for label in sorted(set(a)&set(b)):
            differences.append(dict(shared_template_label=label,coordinate_difference_mm=float(np.linalg.norm(a[label]-b[label])*1000),
                                    interpretation='same-string layout comparison ONLY; not physical sensor correspondence'))
    fig=plt.figure(figsize=(6*len(layouts),10))
    for j,(kind,(names,points)) in enumerate(layouts.items()):
        for row,(elev,azim) in enumerate([(15,90),(15,0)]):
            ax=fig.add_subplot(2,len(layouts),row*len(layouts)+j+1,projection='3d')
            ax.plot_trisurf(*(vertices*1000).T,triangles=faces,color='#BFC9D4',alpha=.12,lw=0,rasterized=True)
            ax.scatter(*(points*1000).T,c='#0077B6',s=7,depthshade=False)
            ax.set_title(f'{kind}: template-only QA\nNo dataset mapping validated')
            ax.set(xlabel='MRI X (mm)',ylabel='MRI Y (mm)',zlabel='MRI Z (mm)')
            ax.set_box_aspect(np.ptp(vertices,axis=0));ax.view_init(elev,azim)
    fig.suptitle('Candidate templates versus fsaverage scalp: fixed coordinates, no fitting/projection',fontsize=14)
    fig.subplots_adjust(top=.9,wspace=.05)
    for path,value in inputs.items():require(sha(path)==value,f'Input changed while auditing: {path}')
    out.mkdir(parents=True,exist_ok=False)
    fig.savefig(out/'07_candidate_templates_on_fsaverage.png',dpi=200,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(catalog).to_csv(out/'01_candidate_montage_comparison.csv',index=False)
    pd.DataFrame(mapping).to_csv(out/'02_channel_mapping_audit.csv',index=False)
    pd.DataFrame(geometries).to_csv(out/'03_template_only_scalp_geometry.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='mesh'} for r in summaries]).to_csv(out/'04_scalp_distance_summary.csv',index=False)
    pd.DataFrame(fids).to_csv(out/'05_fiducial_convention_qa.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'06_same_template_name_layout_differences.csv',index=False)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='BLOCKED',forward_modeling_authorized=False,
        independent_physical_channel_mapping_verified=False,exact_acquisition_net_revision_verified=False,
        blockers=evidence['blockers'],evidence=evidence,input_sha256=inputs,
        mne_version=mne.__version__,coordinate_units='metres internally; distance summaries in mm',
        head_to_fsaverage_mri=matrix.tolist(),candidate_transforms=transforms,geometry_summaries=summaries,
        no_subject_specific_registration=True,subject_digitization_used=False,individual_MRI_used=False,
        localization_outcomes_used=False,sensor_coordinate_correction=False,
        original_gate_median_mm=10,original_gate_p95_mm=20,
        original_schedules_and_parameters_changed=False,planned_evaluations=5124,
        historical_exploratory_results_status='ASSUMED_UNVERIFIED; preserved, not retroactively validated',
        next_step='Review documentary channel mapping and template anatomy QA; do not start forward modeling from this audit')
    (out/'08_compatibility_readiness.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text('# Read-only montage compatibility audit\n\n'
        'Scientific status BLOCKED: physical Localize-MI eN-to-template EN correspondence is not established.\n'
        '01: candidate availability/frames/units/resource hashes; 02: unverified channel proposals;\n'
        '03/04: template-only scalp distances and classifications; 05: fiducial convention residuals;\n'
        '06: same-template-name coordinate differences (not physical mapping evidence);\n'
        '07: candidate/scalp visual QA; 08: evidence, coordinate transforms, hashes and readiness.\n\n'
        'Geometry is inspected independently of dataset correspondence. No sensor is corrected, fitted or projected to pass.\n'
        'All outputs are new; Script 01, original results and existing exploratory assumptions remain unchanged.\n')
    print(f'BLOCKED: {out}');print('Missing independent acquisition/export sensor correspondence; numerical geometry cannot supply it.')
    return 3


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(2)
