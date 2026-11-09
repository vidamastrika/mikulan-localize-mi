#!/usr/bin/env python3
# OBJECTIVE / EVALUATION-ONLY ASSUMPTIONS
# Map independently known stimulation midpoints from each native MRI to fsaverage
# for later evaluation. Individual MRI is evaluation-only privileged information;
# it never changes the completed template montage, forward, parameters or estimates.
# No localization outcomes select/optimize a transform. No localization errors are
# calculated here. Numerical round-trip success is not anatomical validation.
# INPUTS
# Script 04 reference coordinates/manifest; audited native physical bridge JSON,
# audit CSV, source-provenance inventory and bridge manifest; native maskface T1,
# native H5 coordinate bridge/contact hashes; fsaverage T1.mgz; installed ANTs.
# OUTPUTS (fresh directory inside this variant/outputs only)
# 01_input_inventory.csv, 02_registration_status.csv (all seven subjects, including
# NOT_REQUESTED), 03_mapped_ground_truth.csv (requested runs only),
# 04_mapping_manifest.json, source_snapshots/, fixed fsaverage NIfTI (register),
# registrations/SUBJECT/: registration.log, command.json, ANTs affine/warp/inverse
# warp/warped/inverse-warped files, command logs for point checks, MRI checkerboard
# and mapped-point QA PNGs. README. No existing experiment files are overwritten.
# DESIGN / PARAMETERS
# Full framework: 61 runs, 7 subjects, 12 methods; 732 LOSO + 4,392 Single Reference
# = 5,124 evaluations. Default is sub-01 ONLY; --all-subjects explicitly selects 7.
# --phase audit (default): paths/hashes/headers/bridges/point domains only. No ANTs
# fitting, point executable, resampling or figure rendering. --phase register:
# fixed=fsaverage, moving=native; original SyNQuick -t s, double precision; wrapper
# default convergence. Reproducibility seed=1, threads=1 (both explicit options).
# All shell-wrapper input/output filenames are whitespace-free temporary aliases.
# Never trust wrapper return code alone: require complete nonempty output files,
# readable consistent images and no fatal exception text in the registration log.
# POINT CONVENTIONS / VALIDATION
# RAS mm externally, LPS mm for ANTs CSV (negate X and Y). Moving->fixed points:
# -t [affine,1] -t inverseWarp; fixed->moving points: -t Warp -t affine.
# Verify composed versus staged point operations (0.001 mm numerical tolerance),
# domains on actual field lattices, native/fixed MRI interior (original 1-voxel
# margin), and original round-trip threshold <=0.1 mm. Native HEAD->scanner bridge
# is derived from audited Norig @ inv(Torig) @ H5_mm, not fitted to electrodes.
# fsaverage scanner->surface RAS = vox2ras_tkr @ inv(scanner affine), in mm.
# Ground-truth and MRI QA follows fitting; it never tunes the fit. MRI similarity
# and figures are descriptive; no new anatomical acceptance threshold is invented.
# Status PASS_NUMERICAL_REVIEW_REQUIRED still has anatomical_mapping_approved=False.
# DEPENDENCIES
# Python >=3.10; NumPy, pandas, nibabel, h5py; register adds Matplotlib and
# local antsRegistrationSyNQuick.sh, antsRegistration, PrintHeader and
# antsApplyTransformsToPoints.
# No MNE import, EEG processing, forward/inverse calculation or dataset download.
# COMMANDS (repository root; USER RUNS MANUALLY; fresh folders)
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/05_map_ground_truth.py \
#   --phase audit --subject sub-01 \
#   --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/05_ground_truth_mapping_audit_sub01
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/05_map_ground_truth.py \
#   --phase register --subject sub-01 \
#   --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/05_ground_truth_mapping_sub01
# AFTER pilot review, a fresh all-subject run can use --all-subjects and a new root.
# Exit 0 audit/numerical completion (anatomical review still required); 1 fit/QA
# failure. No resume/overwrite; failures keep logs and status. No next stage runs.
# PRIMARY REFERENCE (point direction/order and RAS/LPS):
# https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians
"""Prepare or fit evaluation-only native-MRI to fsaverage ground-truth mapping."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import traceback

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

VARIANT=Path(__file__).resolve().parents[1]
ROOT=VARIANT.parents[1]
SUBJECTS=[f'sub-{i:02d}' for i in range(1,8)]
ROUNDTRIP_MM=.1
FLIP=np.diag([-1.,-1.,1.])
MAPPED_COLUMNS=['test_subject','run','gt_stimulation_pair','gt_hemisphere',
    'native_scanner_ras_x_mm','native_scanner_ras_y_mm','native_scanner_ras_z_mm',
    'fsaverage_scanner_ras_x_mm','fsaverage_scanner_ras_y_mm','fsaverage_scanner_ras_z_mm',
    'fsaverage_surface_ras_x_mm','fsaverage_surface_ras_y_mm','fsaverage_surface_ras_z_mm',
    'roundtrip_mm','staged_composition_error_mm','inside_inverse_field','inside_forward_field',
    'inside_fixed_mri','point_status','anatomical_mapping_approved']


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def apply(matrix,points):
    matrix=np.asarray(matrix,float);points=np.asarray(points,float)
    return points@matrix[:3,:3].T+matrix[:3,3]


def valid_matrix(matrix):
    m=np.asarray(matrix,float)
    return m.shape==(4,4) and np.isfinite(m).all() and np.allclose(m[3],[0,0,0,1],atol=1e-8,rtol=0) and abs(np.linalg.det(m[:3,:3]))>1e-10


def inside(image,points):
    vox=apply(np.linalg.inv(image.affine),points)
    return np.isfinite(vox).all(axis=1)&(vox>=1).all(axis=1)&(vox<np.asarray(image.shape[:3])-2).all(axis=1)


def point_map(exe,points,terms,work,tag,commands,environment):
    points=np.asarray(points,float)
    require(points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all(),'Invalid RAS points')
    lps=points@FLIP.T
    incoming=work/f'{tag}_input.csv';outgoing=work/f'{tag}_output.csv'
    pd.DataFrame(dict(x=lps[:,0],y=lps[:,1],z=lps[:,2],t=np.zeros(len(lps)))).to_csv(incoming,index=False,float_format='%.12f')
    argv=[str(exe),'-d','3','-i',str(incoming),'-o',str(outgoing)]
    for term in terms:argv+=['-t',term]
    result=subprocess.run(argv,cwd=work,env=environment,text=True,capture_output=True)
    commands.append(dict(tag=tag,argv=argv,returncode=result.returncode))
    (work/'output'/f'{tag}.log').write_text(result.stdout+'\nSTDERR\n'+result.stderr)
    require(result.returncode==0 and outgoing.is_file(),f'ANTs point mapping failed: {tag}; see point log')
    output=pd.read_csv(outgoing)[['x','y','z']].to_numpy(float)
    require(output.shape==points.shape and np.isfinite(output).all(),f'Invalid point output: {tag}')
    # Retain raw LPS CSVs alongside explicit coordinate provenance for review.
    shutil.copy2(incoming,work/'output'/incoming.name)
    shutil.copy2(outgoing,work/'output'/outgoing.name)
    return output@FLIP.T


def image_qa(fixed,warped,points,subset,directory):
    """Diagnostic MRI-only checkerboard and post-fit mapped-stimulation slices."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=nib.as_closest_canonical(fixed);w=nib.as_closest_canonical(warped)
    require(f.shape==w.shape and np.allclose(f.affine,w.affine,rtol=0,atol=1e-4),'QA images occupy different physical grids')
    raw_f=np.asarray(f.dataobj,dtype=np.float32);raw_w=np.asarray(w.dataobj,dtype=np.float32)
    require(np.isfinite(raw_f).all() and np.isfinite(raw_w).all(),'Nonfinite registered image values')
    def normalized(data):
        positive=data[data>0]
        require(len(positive)>100,'Empty/invalid MRI intensities')
        low,high=np.percentile(positive,[1,99])
        require(high>low,'Constant MRI intensities')
        return np.clip((data-low)/(high-low),0,1)
    a,b=normalized(raw_f),normalized(raw_w)
    sample_a=a[::4,::4,::4].ravel();sample_b=b[::4,::4,::4].ravel()
    mask=(sample_a>.1)&(sample_b>.1)
    correlation=float(np.corrcoef(sample_a[mask],sample_b[mask])[0,1]) if mask.sum()>100 else None
    if correlation is not None and not np.isfinite(correlation):correlation=None
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for axis,name in enumerate(['Sagittal','Coronal','Axial']):
        for j,fraction in enumerate([.3,.5,.7]):
            index=int(round((a.shape[axis]-1)*fraction))
            fa=np.take(a,index,axis=axis).T;wa=np.take(b,index,axis=axis).T
            yy,xx=np.indices(fa.shape);tile=((xx//12+yy//12)%2)==0
            axes[axis,j].imshow(np.where(tile,fa,wa),cmap='gray',vmin=0,vmax=1,origin='lower')
            axes[axis,j].set_title(f'{name}, {fraction:.0%} of volume');axes[axis,j].axis('off')
    fig.suptitle('Evaluation-only registration QA: fsaverage / warped-native checkerboard\nNumerical completion is not anatomical approval',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(directory/'qa_mri_checkerboard.png',dpi=200);plt.close(fig)
    vox=apply(np.linalg.inv(f.affine),points)
    fig,axes=plt.subplots(len(points),3,figsize=(12,max(4,2.8*len(points))),squeeze=False)
    for i,(point,row) in enumerate(zip(vox,subset.itertuples())):
        for axis,name in enumerate(['Sagittal','Coronal','Axial']):
            index=int(np.clip(np.rint(point[axis]),0,a.shape[axis]-1))
            image=np.take(a,index,axis=axis).T
            axes[i,axis].imshow(image,cmap='gray',vmin=0,vmax=1,origin='lower')
            warped_slice=np.take(b,index,axis=axis).T
            if warped_slice.max()>.3:
                axes[i,axis].contour(warped_slice,levels=[.3,.6],colors=['#F58518','#72B7B2'],linewidths=.55,alpha=.6)
            others=[d for d in range(3) if d!=axis]
            axes[i,axis].scatter(point[others[0]],point[others[1]],marker='*',s=65,c='#FFD92F',edgecolors='black',linewidths=.5)
            axes[i,axis].set_title(f'{row.run} {row.gt_stimulation_pair}: {name}',fontsize=9)
            axes[i,axis].axis('off')
    fig.suptitle('Mapped stimulation points (post-fit diagnostic references only)\nGray: fsaverage; contours: warped native MRI; stars: mapped midpoints',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(directory/'qa_mapped_stimulation.png',dpi=200);plt.close(fig)
    return dict(sampled_normalized_intensity_correlation=correlation,
                correlation_is_acceptance_gate=False,visual_review_required=True,
                self_intersections_or_deformation_folding_checked=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['audit','register'],default='audit')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--subject',choices=SUBJECTS,action='append')
    group.add_argument('--all-subjects',action='store_true')
    parser.add_argument('--diagnostic-root',type=Path,default=VARIANT/'outputs/04_evaluate_diagnostics')
    parser.add_argument('--bridge-root',type=Path,default=ROOT/'cross_subject_warp/outputs/06_coordinate_bridge')
    parser.add_argument('--fsaverage-root',type=Path,default=Path.home()/'mne_data/MNE-fsaverage-data/fsaverage')
    parser.add_argument('--ants-bin',type=Path,default=Path.home()/'micromamba/envs/localizemi-registration/bin')
    parser.add_argument('--output-root',type=Path,default=VARIANT/'outputs/05_ground_truth_mapping_audit_sub01')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--random-seed',type=int,default=1)
    args=parser.parse_args(argv)
    require(args.threads>0 and args.random_seed>0,'Threads and seed must be positive')
    subjects=SUBJECTS if args.all_subjects else list(dict.fromkeys(args.subject or ['sub-01']))
    out=args.output_root.expanduser().resolve();diagnostics=args.diagnostic_root.expanduser().resolve()
    bridge_root=args.bridge_root.expanduser().resolve();fsroot=args.fsaverage_root.expanduser().resolve()
    antsbin=args.ants_bin.expanduser().resolve()
    require((VARIANT/'outputs').resolve() in out.parents and not out.exists(),'Use a fresh directory inside this variant/outputs')
    require(VARIANT in diagnostics.parents,'Diagnostic reference must belong to this variant')
    hashes={};inventory=[]
    def verify(path,role,expected=None):
        path=Path(path).expanduser().resolve();value=sha(path)
        require(expected is None or value==expected,f'Input SHA differs: {path}')
        if str(path) not in hashes:
            hashes[str(path)]=value;inventory.append(dict(role=role,path=str(path),sha256=value,size_bytes=path.stat().st_size))
        return path
    evaluation_path=verify(diagnostics/'12_evaluation_manifest.json','diagnostic_manifest')
    evaluation=json.loads(evaluation_path.read_text())
    require(evaluation['status']=='DIAGNOSTIC_EVALUATION_COMPLETE_ANATOMICAL_ERROR_BLOCKED','Completed Script 04 required')
    ground_path=verify(diagnostics/'02_ground_truth_diagnostic_reference.csv','evaluation_only_coordinates',
                        evaluation['output_sha256']['02_ground_truth_diagnostic_reference.csv'])
    ground=pd.read_csv(ground_path,keep_default_na=False)
    require(len(ground)==61 and not ground.duplicated(['test_subject','run']).any()
            and set(ground.test_subject)==set(SUBJECTS) and ground.coordinate_audit.eq('PASS_NUMERICAL').all(),
            'Expected 61 audited stimulation midpoints')
    # Recover the dataset identity without importing/loading reconstruction code.
    frozen_paths=[Path(p) for p in evaluation['input_sha256'] if Path(p).name=='06_frozen_run_manifest.json']
    require(len(frozen_paths)==1,'Ambiguous reconstruction provenance')
    frozen_path=verify(frozen_paths[0],'reconstruction_identity_only',evaluation['input_sha256'][str(frozen_paths[0])])
    dataset=Path(json.loads(frozen_path.read_text())['dataset']).resolve()
    bridges_path=verify(bridge_root/'02_native_physical_bridges.json','native_bridge')
    bridges=json.loads(bridges_path.read_text())
    bridge_audit=pd.read_csv(verify(bridge_root/'01_subject_bridge_audit.csv','native_bridge_audit'))
    bridge_manifest=json.loads(verify(bridge_root/'05_bridge_manifest.json','native_bridge_manifest').read_text())
    require(bridge_manifest['status']=='ALL_NATIVE_BRIDGES_PASS' and set(bridges)==set(SUBJECTS)
            and len(bridge_audit)==7 and set(bridge_audit.subject)==set(SUBJECTS)
            and bridge_audit.subject.is_unique and bridge_audit.status.eq('PASS_HEADER_GEOMETRY_AND_RELEASED_CONTACTS').all(),
            'Independent native bridge audit is incomplete')
    provenance=json.loads(verify(bridge_root/'04_source_provenance.json','bridge_source_inventory').read_text())
    for item in provenance:
        path=Path(item['path'])
        if any(subject in path.parts or subject in path.name for subject in subjects):
            verify(path,'native_bridge_source_evaluation_only',item['sha256'])
    fs_path=verify(fsroot/'mri/T1.mgz','fixed_fsaverage_MRI')
    fixed=nib.load(fs_path)
    require(fixed.ndim==3 and isinstance(fixed.header,nib.freesurfer.mghformat.MGHHeader),'Expected fsaverage 3D MGH MRI')
    scanner_to_surface=fixed.header.get_vox2ras_tkr()@np.linalg.inv(fixed.affine)
    require(valid_matrix(scanner_to_surface),'Invalid fsaverage scanner/surface bridge')
    native_paths={};sources={}
    for subject in subjects:
        native_paths[subject]=verify(dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii','moving_native_MRI_evaluation_only')
        image=nib.load(native_paths[subject])
        require(image.ndim==3 and image.header.get_xyzt_units()[0]=='mm','Native MRI must be 3D millimetres')
        bridge=bridges[subject]
        for field in ['norig','torig','head_to_surface_mm','head_to_scanner_ras_mm','scanner_ras_to_head_mm']:
            require(valid_matrix(bridge[field]),f'Invalid native bridge: {subject} {field}')
        corners=np.array(np.meshgrid(*[(0,n-1) for n in image.shape],indexing='ij')).reshape(3,-1).T
        header_error=np.linalg.norm(apply(image.affine,corners)-apply(bridge['norig'],corners),axis=1).max()
        require(header_error<=.1,f'{subject}: MRI header differs from audited Norig')
        subset=ground.loc[ground.test_subject.eq(subject)]
        h5_path=dataset/'derivatives/sourcemodelling'/subject/'xfm'/f'{subject}_from-head_to-surface.h5'
        for record in subset.itertuples():
            verify(h5_path,'evaluation_only_H5_bridge',record.transform_sha256)
            verify(dataset/'derivatives/epochs'/subject/'ieeg'/f'{subject}_task-seegstim_space-surface_electrodes.tsv',
                   'evaluation_only_contact_provenance',record.contact_sha256)
        with h5py.File(h5_path,'r') as handle:h5=np.asarray(handle['trans'][()],float)
        require(valid_matrix(h5),'Invalid native HEAD-to-surface H5')
        h5_mm=h5.copy();h5_mm[:3,3]*=1000
        derived=np.asarray(bridge['norig'])@np.linalg.inv(bridge['torig'])@h5_mm
        require(np.allclose(derived,bridge['head_to_scanner_ras_mm'],rtol=0,atol=1e-5),f'{subject}: native bridge derivation differs')
        require(np.allclose(np.asarray(bridge['scanner_ras_to_head_mm'])@derived,np.eye(4),rtol=0,atol=1e-5),'Invalid inverse native bridge')
        head=subset[[f'gt_head_{a}_mm' for a in 'xyz']].to_numpy(float)
        surface=subset[[f'gt_surface_{a}_m' for a in 'xyz']].to_numpy(float)*1000
        require(np.isfinite(head).all() and np.allclose(apply(h5_mm,head),surface,rtol=0,atol=.1),'Ground-truth HEAD/surface mismatch')
        sources[subject]=apply(derived,head)
        require(inside(image,sources[subject]).all(),f'{subject}: native midpoint outside MRI interior')
    executables={name:verify(antsbin/name,'ANTs_executable') for name in
                 ['antsRegistrationSyNQuick.sh','antsRegistration','antsApplyTransformsToPoints','PrintHeader']}
    require(all(os.access(p,os.X_OK) for p in executables.values()),'ANTs files must be executable')
    ants_packages=[]
    for path in sorted((antsbin.parent/'conda-meta').glob('ants-*.json')):
        record=json.loads(verify(path,'ANTs_package_build_metadata').read_text())
        ants_packages.append({key:record.get(key) for key in ['name','version','build','channel']})
    verify(Path(__file__),'mapping_script')
    for path in hashes:
        require(Path(path).parent not in out.parents and Path(path).parent!=out,'Output overlaps protected input directory')
    rows=[dict(subject=s,status='READY_AUDIT_ONLY' if s in subjects else 'NOT_REQUESTED',reason='',
               runtime_seconds=0.,mapped_runs=0,anatomical_mapping_approved=False) for s in SUBJECTS]
    mapped_rows=[];outputs={};commands_by_subject={}
    out.mkdir(parents=True,exist_ok=False)
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for path in [ground_path,bridges_path,evaluation_path,Path(__file__),
                 bridge_root/'01_subject_bridge_audit.csv',bridge_root/'04_source_provenance.json',
                 bridge_root/'05_bridge_manifest.json']:shutil.copy2(path,snapshots/path.name)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','nibabel','h5py']}
    def checkpoint():
        pd.DataFrame(inventory).to_csv(out/'01_input_inventory.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'02_registration_status.csv',index=False)
        pd.DataFrame(mapped_rows,columns=MAPPED_COLUMNS).to_csv(out/'03_mapped_ground_truth.csv',index=False)
        report=dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=args.phase,
            requested_subjects=subjects,planned_subjects=7,planned_runs=61,planned_inverse_evaluations=5124,
            status='AUDIT_COMPLETE_NO_REGISTRATION' if args.phase=='audit' else 'REGISTRATION_RESULTS_REVIEW_REQUIRED',
            input_sha256=hashes,generated_output_sha256=outputs,software_versions=versions,
            ants_package_builds=ants_packages,
            registration_protocol=dict(wrapper='antsRegistrationSyNQuick.sh',type='s',precision='double',
                seed=args.random_seed,threads=args.threads,convergence='unchanged installed wrapper defaults'),
            fsaverage_scanner_to_surface_ras_mm=scanner_to_surface.tolist(),
            point_direction='native moving->fsaverage fixed: inverse affine, inverse warp; reverse: forward warp, affine',
            point_coordinates='RAS mm externally, LPS mm in ANTs CSV; negate X/Y',
            max_roundtrip_mm=ROUNDTRIP_MM,field_domain_margin_voxels=1,
            anatomical_mapping_approved=False,anatomical_errors_calculated=False,
            reconstruction_changed=False,sensor_positions_used_for_fitting=False,
            individual_MRI_used_for_evaluation_only=True,registration_fitting_executed=args.phase=='register',
            inverse_reconstruction_executed=False,
            references=['https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians'],
            limitations=['Numerical inverse consistency does not establish anatomical alignment.',
                'Maskface native MRI versus intact fsaverage can register imperfectly; inspect whole-brain and stimulation-region QA.',
                'MRI QA statistics are descriptive, not new acceptance thresholds.',
                'Deformation folding is not assessed by this script.',
                'No automatic mapping approval or anatomical localization errors; retain Script 04 diagnostic status until review.',
                'The exploratory electrode-number assumption and sensor/scalp mismatch remain reconstruction limitations.'])
        write_json(out/'04_mapping_manifest.json',report)
    checkpoint()
    (out/'README.md').write_text('# Evaluation-only ground-truth mapping\n\n'
        '01: input inventory; 02: seven-subject status (NOT_REQUESTED retained); 03: mapped requested runs; 04: manifest.\n'
        'registrations/SUBJECT/: logs, commands, transforms, warped images, raw LPS point checks, MRI checkerboard and mapped-stimulation figures.\n'
        'source_snapshots/: exact script, reference and bridge snapshots.\n\n'
        'Fixed fsaverage, moving native MRI. Registration uses whitespace-free aliases. Existing data/reconstruction are untouched.\n'
        'Original 0.1 mm round-trip gate retained. PASS_NUMERICAL_REVIEW_REQUIRED is not anatomical approval.\n'
        'Inspect checkerboards across the brain and stimulation-region slices before accepting any mapping.\n'
        'The known midpoint is mapped after MRI fitting; it never tunes a transform or montage. No inverse fits or anatomical errors run.\n')
    if args.phase=='audit':
        print(f'AUDIT_COMPLETE_NO_REGISTRATION: {out} | requested: {subjects}')
        return 0
    fixed_file=out/'fsaverage_T1_scanner_ras.nii.gz'
    fixed_nifti=nib.Nifti1Image(np.asarray(fixed.dataobj,dtype=np.float32),fixed.affine)
    fixed_nifti.header.set_xyzt_units('mm');fixed_nifti.set_qform(fixed.affine,code=1);fixed_nifti.set_sform(fixed.affine,code=1)
    nib.save(fixed_nifti,fixed_file);outputs[str(fixed_file.relative_to(out))]=sha(fixed_file)
    environment=os.environ.copy();environment['PATH']=str(antsbin)+os.pathsep+environment.get('PATH','')
    environment['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']=str(args.threads)
    environment['ANTS_RANDOM_SEED']=str(args.random_seed)
    for subject in subjects:
        row=next(r for r in rows if r['subject']==subject);started=time.perf_counter()
        directory=out/'registrations'/subject;directory.mkdir(parents=True)
        row.update(status='RUNNING_REGISTRATION');checkpoint()
        commands=[];commands_by_subject[subject]=commands
        try:
            with tempfile.TemporaryDirectory(prefix='template_gt_',dir='/tmp') as temp:
                work=Path(temp);require(not re.search(r'\s',str(work)),'Temporary path must be whitespace-free')
                (work/'fixed.nii.gz').symlink_to(fixed_file)
                (work/'moving.nii').symlink_to(native_paths[subject])
                (work/'output').symlink_to(directory,target_is_directory=True)
                prefix=f'output/{subject}_to_fsaverage_'
                argv=[str(executables['antsRegistrationSyNQuick.sh']),'-d','3','-f','fixed.nii.gz',
                      '-m','moving.nii','-o',prefix,'-t','s','-p','d','-n',str(args.threads),'-e',str(args.random_seed)]
                print(f'{subject}: fitting evaluation-only MRI registration; follow {directory/"registration.log"}',flush=True)
                with (directory/'registration.log').open('w') as log:
                    result=subprocess.run(argv,cwd=work,env=environment,text=True,stdout=log,stderr=subprocess.STDOUT)
                commands.append(dict(tag='registration',argv=argv,returncode=result.returncode))
                paths={name:directory/f'{subject}_to_fsaverage_{suffix}' for name,suffix in
                       [('affine','0GenericAffine.mat'),('warp','1Warp.nii.gz'),('inverse','1InverseWarp.nii.gz'),
                        ('warped','Warped.nii.gz'),('inverse_warped','InverseWarped.nii.gz')]}
                missing=[p.name for p in paths.values() if not p.is_file() or p.stat().st_size==0]
                log=(directory/'registration.log').read_text(errors='replace')
                fatal=bool(re.search(r'ITK ERROR|Exception Object caught|Segmentation fault|does not exist|command not found',log,re.I))
                require(result.returncode==0 and not missing and not fatal,
                        f'Registration failed: exit={result.returncode}, missing={missing}, fatal_log={fatal}')
                row['status']='RUNNING_POINT_QA';checkpoint()
                for name in ['affine','warp','inverse']:(work/paths[name].name).symlink_to(paths[name])
                affine=paths['affine'].name;warp=paths['warp'].name;inverse=paths['inverse'].name
                subset=ground.loc[ground.test_subject.eq(subject)].copy();source=sources[subject]
                exe=executables['antsApplyTransformsToPoints']
                pre=point_map(exe,source,[f'[{affine},1]'],work,'inverse_affine_only',commands,environment)
                mapped=point_map(exe,source,[f'[{affine},1]',inverse],work,'native_to_fsaverage',commands,environment)
                staged=point_map(exe,pre,[inverse],work,'inverse_field_only',commands,environment)
                returned=point_map(exe,mapped,[warp,affine],work,'fsaverage_to_native',commands,environment)
                errors=np.linalg.norm(returned-source,axis=1);composition=np.linalg.norm(staged-mapped,axis=1)
                field_inv=nib.load(paths['inverse']);field_forward=nib.load(paths['warp'])
                for field in [field_inv,field_forward]:
                    require(len(field.shape) in (4,5) and all(n>2 for n in field.shape[:3]),'Invalid displacement-field dimensions')
                    require(field.shape[-1]==3 and field.header.get_xyzt_units()[0]=='mm','Invalid vector-field components/units')
                inverse_ok=inside(field_inv,pre);forward_ok=inside(field_forward,mapped);fixed_ok=inside(fixed,mapped)
                valid=np.isfinite(errors)&(errors<=ROUNDTRIP_MM)&(composition<=.001)&inverse_ok&forward_ok&fixed_ok
                surface=apply(scanner_to_surface,mapped)
                require(np.allclose(apply(np.linalg.inv(scanner_to_surface),surface),mapped,rtol=0,atol=1e-5),'fsaverage surface/scanner bridge fails round-trip')
                for record,src,target,surf,error,comp,a,b,c,ok in zip(subset.to_dict('records'),source,mapped,surface,errors,composition,inverse_ok,forward_ok,fixed_ok,valid):
                    item=dict(test_subject=subject,run=record['run'],gt_stimulation_pair=record['gt_stimulation_pair'],gt_hemisphere=record['gt_hemisphere'],
                        roundtrip_mm=float(error),staged_composition_error_mm=float(comp),inside_inverse_field=bool(a),
                        inside_forward_field=bool(b),inside_fixed_mri=bool(c),
                        point_status='PASS_NUMERICAL_REVIEW_REQUIRED' if ok else 'FAIL_POINT_QA',anatomical_mapping_approved=False)
                    for coordinates,label in [(src,'native_scanner_ras'),(target,'fsaverage_scanner_ras'),(surf,'fsaverage_surface_ras')]:
                        for i,axis in enumerate('xyz'):item[f'{label}_{axis}_mm']=float(coordinates[i])
                    mapped_rows.append(item)
                row.update(mapped_runs=len(subset),max_roundtrip_mm=float(errors.max()),
                           max_staged_composition_error_mm=float(composition.max()))
                require(valid.all(),f'{int((~valid).sum())} points fail round-trip/domain/composition QA')
                warped=nib.load(paths['warped'])
                require(warped.shape==fixed.shape and np.allclose(warped.affine,fixed.affine,rtol=0,atol=1e-4),
                        'Warped MRI grid differs from fsaverage fixed MRI')
                image_info=image_qa(fixed_nifti,warped,mapped,subset,directory)
                write_json(directory/'image_qa.json',image_info)
                row.update(status='PASS_NUMERICAL_REVIEW_REQUIRED',reason='Inspect MRI and mapped-point QA; no automatic anatomical approval')
        except Exception as exc:
            row.update(status='FAIL_REGISTRATION_OR_QA',reason=f'{type(exc).__name__}: {exc}')
            (directory/'failure_traceback.txt').write_text(traceback.format_exc())
        finally:
            row['runtime_seconds']=time.perf_counter()-started
            write_json(directory/'command.json',dict(commands=commands,
                aliases=dict(fixed=str(fixed_file),moving=str(native_paths[subject]),output=str(directory)),
                temporary_paths='Ephemeral whitespace-free /tmp/template_gt_*; aliases were not part of original input filenames',
                threads=args.threads,random_seed=args.random_seed))
            for path in directory.iterdir():
                if path.is_file():outputs[str(path.relative_to(out))]=sha(path)
            checkpoint()
            print(f'{subject}: {row["status"]} | {row["reason"]}',flush=True)
    for path,expected in hashes.items():require(sha(path)==expected,f'Input changed during fitting: {path}')
    checkpoint()
    print(f'Wrote evaluation-only mapping/QA: {out}; anatomical approval remains False.')
    return 1 if any(r['status']=='FAIL_REGISTRATION_OR_QA' for r in rows) else 0


if __name__=='__main__':
    raise SystemExit(main())
