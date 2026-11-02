#!/usr/bin/env python3
# SCRIPT 06 HELPER — Derive/check native surface-to-scanner geometry; no fitting.
# Run from the project root with .venv active:
# python cross_subject_warp/scripts/06_validate_coordinate_bridge.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# Repeat into a NEW directory (previous reports are preserved):
# python cross_subject_warp/scripts/06_validate_coordinate_bridge.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --output-root cross_subject_warp/outputs/06_coordinate_bridge_review
#
# Help: python cross_subject_warp/scripts/06_validate_coordinate_bridge.py --help
# Exit 0: all seven native bridges pass the declared geometry/contact checks.
# Exit 2: inspect failed subjects; bridges are not published for failed subjects.
# Neither result clears MRI-registration point/grid QA or enables evaluation.
# No EEG arrays, fitting, ANTs calls, point registration or localization errors.
"""Check native FreeSurfer surface RAS -> registration MRI scanner RAS bridges.

Use archived VolGeom metadata from BOTH native pial GIFTIs to reconstruct Norig
(voxel -> scanner RAS) and Torig (voxel -> FreeSurfer surface RAS). The bridge is
Norig @ inv(Torig), per FreeSurfer's documented coordinate convention. Derive it
from volume geometry, never from electrode fitting or localization outcomes.

Compare volume geometry to the actual maskface MRI header over its voxel-grid
corners, and validate released surface/T1w contact coordinates under that bridge.
Connect recorded native head peaks through H5 and the forward MRI/head transform.
A GIFTI unknown->Talairach xform is recorded but never applied to native points.

Passing means native geometry/metadata/contact consistency, not cortical MRI
registration accuracy. Registration matrices/fields are not used or modified.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'
SUBJECTS=[f'sub-{i:02d}' for i in range(1,8)]
TOLERANCE_MM=0.1
REFERENCES=[
 'https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems',
 'https://nipy.org/nibabel/reference/nibabel.freesurfer.html']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def apply(matrix,points):return np.asarray(points,float)@matrix[:3,:3].T+matrix[:3,3]


def rigid(matrix):
    matrix=np.asarray(matrix,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Invalid homogeneous transform')
    if not np.allclose(matrix[3],[0,0,0,1],atol=1e-8,rtol=0):raise ValueError('Invalid homogeneous bottom row')
    r=matrix[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(r)-1)>1e-5:
        raise ValueError('Expected proper rigid physical-coordinate transform')
    return matrix


def h5_metres_to_mm(matrix):
    matrix=rigid(matrix)
    scale=np.diag([1000.,1000.,1000.,1.])
    return scale@matrix@np.linalg.inv(scale)


def volume_geometry(meta):
    """FreeSurfer centre is dims/2, not the centre of the voxel-index extrema."""
    dims=np.array([float(meta[f'VolGeom{x}']) for x in ['Width','Height','Depth']])
    zooms=np.array([float(meta[f'VolGeom{x}size']) for x in 'XYZ'])
    direction=np.column_stack([[float(meta[f'VolGeom{x}_{axis}']) for axis in 'RAS'] for x in 'XYZ'])
    centre=np.array([float(meta[f'VolGeomC_{axis}']) for axis in 'RAS'])
    if not np.isfinite(np.r_[dims,zooms,centre,direction.ravel()]).all():raise ValueError('Nonfinite volume geometry')
    if (dims<=0).any() or not np.array_equal(dims,np.rint(dims)) or (zooms<=0).any():raise ValueError('Invalid dimensions/voxel sizes')
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5,rtol=0):raise ValueError('Invalid voxel direction cosines')
    norig=np.eye(4);norig[:3,:3]=direction*zooms
    norig[:3,3]=centre-norig[:3,:3]@(dims/2.)
    half=dims*zooms/2.
    torig=np.array([[-zooms[0],0,0,half[0]],[0,0,zooms[2],-half[2]],
                    [0,-zooms[1],0,half[1]],[0,0,0,1]],float)
    return dict(dimensions=dims.astype(int).tolist(),voxel_sizes_mm=zooms.tolist(),
                centre_scanner_ras_mm=centre.tolist(),norig=norig,torig=torig,
                surface_to_scanner=rigid(norig@np.linalg.inv(torig)))


def grid_discrepancy_mm(left,right,dimensions):
    corners=np.array(list(itertools.product(*[(0.,float(n)-1) for n in dimensions])))
    return float(np.linalg.norm(apply(left,corners)-apply(right,corners),axis=1).max())


def contacts_in_mm(surface,t1w,bridge):
    fields=['name','x','y','z']
    if not set(fields).issubset(surface) or not set(fields).issubset(t1w):raise ValueError('Contact table lacks name/x/y/z')
    if surface.name.duplicated().any() or t1w.name.duplicated().any():raise ValueError('Duplicate contact names')
    if set(surface.name)!=set(t1w.name) or len(surface)<4:raise ValueError('Contact identities differ or fewer than four contacts')
    joined=surface[fields].merge(t1w[fields],on='name',suffixes=('_surface','_t1w'),validate='one_to_one')
    source=joined[[f'{a}_surface' for a in 'xyz']].to_numpy(float)*1000
    target=joined[[f'{a}_t1w' for a in 'xyz']].to_numpy(float)*1000
    if not np.isfinite(np.r_[source.ravel(),target.ravel()]).all():raise ValueError('Nonfinite contacts')
    if np.max(np.abs(np.r_[source.ravel(),target.ravel()]))>1000:raise ValueError('Contact coordinate scale inconsistent with declared metres')
    predicted=apply(bridge,source)
    errors=np.linalg.norm(predicted-target,axis=1)
    return joined.name.tolist(),errors


def inspect_subject(dataset,subject,preparation_row):
    import h5py
    import nibabel as nib
    import mne
    from mne.io.constants import FIFF
    paths=[];row=dict(subject=subject,status='FAIL',reason='');matrices=None;contact_rows=[]
    try:
        model=dataset/'derivatives/sourcemodelling'/subject
        mri=dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
        h5=model/'xfm'/f'{subject}_from-head_to-surface.h5'
        forward=model/'fwd'/f'{subject}_fwd.fif'
        paths.extend([mri,h5,forward])
        for field,hashfield in [('forward_path','forward_sha256'),('head_to_surface_path','h5_sha256'),('contacts_path','contacts_sha256')]:
            p=Path(preparation_row[field])
            if sha(p)!=preparation_row[hashfield]:raise ValueError(f'Audited native input changed: {p}')
        geometry=[];gifti_info=[]
        for hemi in ['L','R']:
            path=model/'anat'/f'{subject}_hemi-{hemi}_pial.surf.gii';paths.append(path)
            image=nib.load(str(path));points=[a for a in image.darrays if int(a.intent)==1008]
            if len(points)!=1:raise ValueError('Expected one native GIFTI pointset')
            pointset=points[0]
            if np.asarray(pointset.data).ndim!=2 or np.asarray(pointset.data).shape[1]!=3:raise ValueError('Invalid pointset dimensions')
            if not np.isfinite(pointset.data).all():raise ValueError('Nonfinite surface points')
            meta=dict(pointset.meta)
            if str(meta.get('GeometricType','')).lower()!='anatomical' or str(meta.get('AnatomicalStructureSecondary','')).lower()!='pial':
                raise ValueError('GIFTI is not labelled native anatomical pial')
            if str(meta.get('AnatomicalStructurePrimary',''))!=('CortexLeft' if hemi=='L' else 'CortexRight'):
                raise ValueError('GIFTI hemisphere identity differs')
            if not str(meta.get('Name','')).endswith('lh.pial' if hemi=='L' else 'rh.pial'):
                raise ValueError('Native FreeSurfer pial origin is not recorded')
            if int(pointset.coordsys.dataspace)!=0 or int(pointset.coordsys.xformspace) not in [0,3]:
                raise ValueError('Unexpected GIFTI pointset coordinate-system annotation')
            if int(pointset.coordsys.xformspace)==0 and not np.allclose(pointset.coordsys.xform,np.eye(4),atol=1e-8,rtol=0):
                raise ValueError('Unspecified nonidentity GIFTI coordinate transform')
            geometry.append(volume_geometry(meta))
            gifti_info.append(dict(hemisphere=hemi,path=str(path),pointset_metadata=meta,
                dataspace=int(pointset.coordsys.dataspace),xformspace=int(pointset.coordsys.xformspace),
                coordinate_system_xform=pointset.coordsys.xform.tolist(),native_pointset_xform_applied=False))
        g=geometry[0]
        if g['dimensions']!=geometry[1]['dimensions']:raise ValueError('Hemisphere volume dimensions differ')
        hemi_difference=grid_discrepancy_mm(g['norig'],geometry[1]['norig'],g['dimensions'])
        if hemi_difference>TOLERANCE_MM:raise ValueError('Hemisphere volume geometry differs')
        if not np.allclose(g['torig'],geometry[1]['torig'],atol=1e-6,rtol=0):raise ValueError('Hemisphere surface RAS geometry differs')
        image=nib.load(str(mri))
        if image.ndim!=3 or tuple(image.shape)!=tuple(g['dimensions']):raise ValueError('Registration MRI grid differs from native surface volume')
        if image.header.get_xyzt_units()[0]!='mm':raise ValueError('Registration MRI units are not millimetres')
        header_error=grid_discrepancy_mm(g['norig'],image.affine,g['dimensions'])
        if header_error>TOLERANCE_MM:raise ValueError(f'GIFTI volume vs registration MRI header differs by {header_error:g} mm')
        qform,qcode=image.get_qform(coded=True);sform,scode=image.get_sform(coded=True)
        if not int(qcode) or not int(scode):raise ValueError('MRI qform/sform must both be explicitly coded')
        if int(qcode)!=1 or int(scode)!=1:raise ValueError('MRI qform/sform are not labelled scanner anatomical')
        qs_error=grid_discrepancy_mm(qform,sform,g['dimensions'])
        if qs_error>TOLERANCE_MM:raise ValueError('MRI qform/sform disagree')
        with h5py.File(h5,'r') as file:head_to_surface_m=rigid(np.asarray(file['trans'],float))
        fwd=mne.read_forward_solution(str(forward),verbose=False);transform=fwd['mri_head_t']
        frames=(int(transform['from']),int(transform['to']))
        if frames==(int(FIFF.FIFFV_COORD_MRI),int(FIFF.FIFFV_COORD_HEAD)):
            surface_to_head_m=rigid(np.asarray(transform['trans'],float))
        elif frames==(int(FIFF.FIFFV_COORD_HEAD),int(FIFF.FIFFV_COORD_MRI)):
            surface_to_head_m=rigid(np.linalg.inv(transform['trans']))
        else:raise ValueError('Unexpected forward MRI/head frame codes')
        if not np.allclose(np.linalg.inv(head_to_surface_m),surface_to_head_m,atol=1e-5,rtol=0):
            raise ValueError('H5 and forward MRI/head transform differ')
        head_to_surface_mm=h5_metres_to_mm(head_to_surface_m)
        surface_to_scanner=g['surface_to_scanner']
        head_to_scanner=rigid(surface_to_scanner@head_to_surface_mm)
        ieeg=dataset/'derivatives/epochs'/subject/'ieeg'
        tables={}
        for space in ['surface','T1w']:
            coords=ieeg/f'{subject}_task-seegstim_space-{space}_coordsystem.json'
            table=ieeg/f'{subject}_task-seegstim_space-{space}_electrodes.tsv'
            paths.extend([coords,table]);metadata=json.loads(coords.read_text())
            if metadata.get('iEEGCoordinateUnits')!='m':raise ValueError('Released electrode units are not metres')
            tables[space]=pd.read_csv(table,sep='\t')
        names,errors=contacts_in_mm(tables['surface'],tables['T1w'],surface_to_scanner)
        if float(errors.max())>TOLERANCE_MM:raise ValueError(f'Derived physical bridge vs named T1w contacts differs by {errors.max():g} mm')
        contact_rows=[dict(subject=subject,contact=name,residual_mm=float(error)) for name,error in zip(names,errors)]
        probe=np.array(list(itertools.product([-150.,150.],repeat=3)))
        legacy_error=float(np.linalg.norm(apply(head_to_scanner,probe)-probe,axis=1).max())
        row.update(status='PASS_HEADER_GEOMETRY_AND_RELEASED_CONTACTS',contacts=len(names),
            left_right_header_max_error_mm=hemi_difference,gifti_mri_header_max_error_mm=header_error,
            qform_sform_max_error_mm=qs_error,contact_max_error_mm=float(errors.max()),
            legacy_head_as_scanner_identity_max_error_mm=legacy_error,
            legacy_identity_supported=legacy_error<=TOLERANCE_MM,
            reason='Native bridge derived independently from archived surface volume geometry and checked against MRI and released contacts')
        matrices=dict(subject=subject,input_frame='recorded_model_head',input_units='mm',output_frame='registration_mri_scanner_ras',output_units='mm',
            derivation='Norig @ inv(Torig); head_to_scanner = surface_to_scanner @ H5_mm',
            surface_to_scanner_ras_mm=surface_to_scanner.tolist(),scanner_ras_to_surface_mm=np.linalg.inv(surface_to_scanner).tolist(),
            head_to_surface_mm=head_to_surface_mm.tolist(),head_to_scanner_ras_mm=head_to_scanner.tolist(),
            scanner_ras_to_head_mm=np.linalg.inv(head_to_scanner).tolist(),norig=g['norig'].tolist(),torig=g['torig'].tolist(),
            gifti_evidence=gifti_info,forward_mri_head_frame_codes=list(frames),native_bridge_status=row['status'],
            registration_point_mapping_validated=False)
    except Exception as exc:row['reason']=f'{type(exc).__name__}: {exc}'
    return row,matrices,contact_rows,paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment_review')
    p.add_argument('--output-root',type=Path,default=EXP/'outputs/06_coordinate_bridge')
    args=p.parse_args()
    for key,value in vars(args).items():setattr(args,key,value.expanduser().resolve())
    if args.output_root.exists() or args.output_root.is_symlink():raise FileExistsError('Output exists; choose a NEW --output-root')
    native_folders=[EXP/'outputs'/name for name in ['03_reference_based_final','05_single_reference_final',
        '03_gamma_wholebrain_sensitivity','05_wholebrain_hemisphere_sensitivity','03_05_native_completion_audit','06_evaluate_localization']]
    for protected in [args.preparation_root,EXP/'source',EXP/'scripts',ROOT/'cross_subject',ROOT/'cross_subject_no_warp',*native_folders]:
        if args.output_root==protected or protected in args.output_root.parents or args.output_root in protected.parents:
            raise ValueError(f'Output overlaps protected input tree: {protected}')
    source=args.preparation_root/'10_source_provenance.json';source_doc=json.loads(source.read_text())
    dataset=Path(source_doc['locations']['dataset']).resolve()
    if args.output_root==dataset or dataset in args.output_root.parents or args.output_root in dataset.parents:
        raise ValueError('Output overlaps dataset')
    audit_path=args.preparation_root/'05_coordinate_audit.csv';audit=pd.read_csv(audit_path,dtype=str,keep_default_na=False)
    if len(audit)!=7 or set(audit.subject)!=set(SUBJECTS) or audit.subject.duplicated().any() or not audit.status.eq('PASS_NUMERICAL').all():
        raise ValueError('Preparation numerical audit is incomplete')
    indexed=audit.set_index('subject').to_dict('index')
    rows=[];bridges={};contacts=[];paths=[source,audit_path,Path(__file__)]
    readme=dataset/'README.md'
    if readme.is_file():paths.append(readme)
    initial_hashes={}
    for path in paths:
        if path.is_file():initial_hashes[path]=sha(path)
    for subject in SUBJECTS:
        model=dataset/'derivatives/sourcemodelling'/subject
        ieeg=dataset/'derivatives/epochs'/subject/'ieeg'
        subject_paths=[dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii',
            model/'xfm'/f'{subject}_from-head_to-surface.h5',model/'fwd'/f'{subject}_fwd.fif']
        subject_paths += [model/'anat'/f'{subject}_hemi-{h}_pial.surf.gii' for h in ['L','R']]
        subject_paths += [ieeg/f'{subject}_task-seegstim_space-{space}_{suffix}' for space in ['surface','T1w']
                          for suffix in ['coordsystem.json','electrodes.tsv']]
        for path in subject_paths:
            if path.is_file():initial_hashes[path]=sha(path)
        print(f'Checking archived geometry: {subject}',flush=True)
        row,bridge,contact_rows,used=inspect_subject(dataset,subject,indexed[subject])
        rows.append(row);contacts.extend(contact_rows);paths.extend(used)
        if bridge is not None:bridges[subject]=bridge
        print(f'  {row["status"]}: {row["reason"]}',flush=True)
    for path,before in initial_hashes.items():
        if sha(path)!=before:raise ValueError(f'Input changed during bridge inspection: {path}')
    provenance=[dict(path=str(path),sha256=sha(path),size_bytes=path.stat().st_size)
                for path in sorted(set(paths)) if path.is_file()]
    passed=len(bridges)==7
    manifest=dict(status='ALL_NATIVE_BRIDGES_PASS' if passed else 'NATIVE_BRIDGE_CHECK_INCOMPLETE',subjects_passed=len(bridges),
        subjects_expected=7,tolerance_mm=TOLERANCE_MM,derivation='Norig @ inv(Torig)',references=REFERENCES,
        native_pointset_talairach_xform_applied=False,electrodes_used_to_fit_transform=False,
        inverse_reconstruction_executed=False,registration_fitting_executed=False,registration_point_mapping_validated=False,
        anatomical_evaluation_ready=False,
        remaining=['Current registration transform checksums and fitting-time MRI provenance.',
                   'Affine/SyN field grids, point direction/order, RAS/LPS conversions and MRI-only mapping QA.',
                   'Mapped evaluation implementation after these native bridges and registration QA pass.'])
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.native_bridge_',dir=args.output_root.parent) as temp:
        stage=Path(temp)/'results';stage.mkdir()
        pd.DataFrame(rows).to_csv(stage/'01_subject_bridge_audit.csv',index=False)
        (stage/'02_native_physical_bridges.json').write_text(json.dumps(bridges,indent=2)+'\n')
        pd.DataFrame(contacts,columns=['subject','contact','residual_mm']).to_csv(stage/'03_contact_consistency.csv',index=False)
        (stage/'04_source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        (stage/'05_bridge_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (stage/'README.txt').write_text(__doc__+'\nRegistration point/grid QA remains separate.\n')
        if args.output_root.exists():raise FileExistsError(args.output_root)
        stage.rename(args.output_root)
    print('Reports:',args.output_root)
    print('Native bridge subjects passed:',len(bridges),'/ 7; anatomical evaluation remains blocked pending registration QA.')
    return 0 if passed else 2


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='coordinate-bridge-config-') as config:
        os.environ['_MNE_FAKE_HOME_DIR']=config
        os.environ['MPLCONFIGDIR']=config
        raise SystemExit(main())
