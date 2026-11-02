#!/usr/bin/env python3
# SCRIPT 06 HELPER — Validate existing registration resources; NEVER fit them.
# From project root with .venv active:
# python cross_subject_warp/scripts/06_validate_registration_resources.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# If ANTs needs its micromamba environment, use the SAME project Python:
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_validate_registration_resources.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# Repeat into a NEW output directory (originals and earlier QA are preserved):
# python cross_subject_warp/scripts/06_validate_registration_resources.py \
#   --output-root cross_subject_warp/outputs/06_registration_qa_review
#
# Help: python cross_subject_warp/scripts/06_validate_registration_resources.py --help
# Exit 0: all 84 mode/pair numerical checks passed. Exit 2: inspect failed rows.
# Hashes the imported ~18 GB at start/end, checks grids/finite fields, and calls
# antsApplyTransformsToPoints only. No EEG/contacts/peaks for QA sample selection,
# registration fitting, inverse fitting, or localization-error calculation.
"""MRI-only validation of existing affine/SyN point mappings for all 42 pairs.

Requires all seven native bridges to pass. Verify copied resource checksums,
current bridge MRI identities, saved fixed/moving identifiers and full SyN chains.
Reuse the pinned historical point-transform terms and MRI intensity consistency
check. Sample MRI foreground with fixed seed 2901; never tune using source errors.

Fixed=test; moving=reference. RAS mm is flipped to ITK LPS for ANTs and back.
Reference->test: inverse affine; SyN inverse affine then inverse field.
Test->reference: affine; SyN forward field then its own affine.
Both fields in this collapsed historical SyN pipeline are on the fixed grid.

A numerical pass validates implementation consistency, not cortical registration
accuracy. Fitting-time MRI byte hashes/ANTs versions are not assumed from file
presence; current hashes, log identities and saved-image reproduction are
recorded separately from unverified historical provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'
SUBJECTS=[f'sub-{i:02d}' for i in range(1,8)]
SAMPLES=200
MAX_RMSE=.015
MAX_ROUNDTRIP_MM=.1
GRID_TOLERANCE_MM=.02
MIN_COVERAGE=.75
REFERENCE_URL='https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images%2C-pointsets-and-Jacobians'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def historical_mapping():
    path=EXP/'source/01_registration_resources/scripts/32_all_reference_localization_evaluation.py'
    pinned=json.loads((EXP/'source/provenance_manifest.json').read_text())
    entry=next(r for r in pinned['files'] if r['path']=='source/01_registration_resources/scripts/32_all_reference_localization_evaluation.py')
    if sha(path)!=entry['sha256']:raise ValueError('Pinned point/image QA implementation changed')
    spec=importlib.util.spec_from_file_location('registration_mapping32',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,path


def files_for_pair(root,reference,test):
    pair=f'{reference}_to_{test}';aff=root/'09_anatomical_registration'/pair;syn=root/'19_syn_registration'/pair
    affine_log=aff/'ants_registration.log'
    if not affine_log.is_file():affine_log=aff/'registration.log'
    return dict(aff=aff/f'{pair}_0GenericAffine.mat',aff_image=aff/f'{reference}_in_{test}.nii.gz',aff_log=affine_log,
                syn_aff=syn/f'{pair}_syn_0GenericAffine.mat',warp=syn/f'{pair}_syn_1Warp.nii.gz',
                invwarp=syn/f'{pair}_syn_1InverseWarp.nii.gz',syn_image=syn/'donor_syn_in_target.nii.gz',
                syn_inverse_image=syn/'target_syn_in_donor.nii.gz',syn_log=syn/'ants_registration.log',
                syn_quality=syn/'syn_registration_quality.json')


def grid_error_mm(image,reference):
    if tuple(image.shape[:3])!=tuple(reference.shape[:3]):raise ValueError('Grid dimensions differ')
    corners=np.array([[x,y,z] for x in [0,reference.shape[0]-1] for y in [0,reference.shape[1]-1] for z in [0,reference.shape[2]-1]])
    a=nib.affines.apply_affine(image.affine,corners);b=nib.affines.apply_affine(reference.affine,corners)
    error=float(np.linalg.norm(a-b,axis=1).max())
    if not np.isfinite(error) or error>GRID_TOLERANCE_MM:raise ValueError(f'Grid physical coordinates differ by {error:g} mm')
    if image.header.get_xyzt_units()[0]!='mm':raise ValueError('Expected image/field spatial units in mm')
    return error


def check_field(path,fixed):
    image=nib.load(str(path))
    if tuple(image.shape[3:]) not in [(3,),(1,3)]:raise ValueError('Expected 3D three-component displacement field')
    if int(image.header['intent_code']) not in [1006,1007]:raise ValueError('Expected displacement/vector intent')
    grid_error_mm(image,fixed)
    values=np.asarray(image.dataobj,dtype=np.float32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite displacement field')
    return dict(path=str(path),shape=list(image.shape),max_abs_component_mm=float(np.abs(values).max()),
                grid_domain='fixed_test',finite=True)


def foreground_points(image,n=SAMPLES):
    values=np.asarray(image.dataobj,dtype=np.float32)
    if values.ndim!=3 or not np.isfinite(values).all():raise ValueError('MRI is not a finite 3D image')
    positive=values[values>0]
    if not len(positive):raise ValueError('No MRI foreground')
    candidates=np.flatnonzero(values>.1*np.percentile(positive,99))
    if len(candidates)<n:raise ValueError('Insufficient MRI foreground samples')
    indices=candidates[np.random.default_rng(2901).choice(len(candidates),size=n,replace=False)]
    vox=np.column_stack(np.unravel_index(indices,values.shape))
    return nib.affines.apply_affine(image.affine,vox)


def roundtrip_qa(point_map,exe,source,destination,forward_terms,inverse_terms):
    points=foreground_points(source)
    mapped=point_map(exe,points,forward_terms)
    vox=nib.affines.apply_affine(np.linalg.inv(destination.affine),mapped)
    valid=np.isfinite(mapped).all(axis=1)&np.all((vox>=1)&(vox<np.asarray(destination.shape[:3])-2),axis=1)
    count=int(valid.sum())
    if count<MIN_COVERAGE*len(points):raise ValueError(f'Point-map destination coverage too low: {count}/{len(points)}')
    back=point_map(exe,mapped[valid],inverse_terms)
    errors=np.linalg.norm(back-points[valid],axis=1)
    maximum=float(errors.max())
    if not np.isfinite(errors).all() or maximum>MAX_ROUNDTRIP_MM:
        raise ValueError(f'Point roundtrip max {maximum:g} mm exceeds {MAX_ROUNDTRIP_MM}')
    return dict(tested=len(points),valid=count,max_roundtrip_mm=maximum,median_roundtrip_mm=float(np.median(errors)))


def verify_log_identity(text,fixed,moving):
    fixed=str(fixed);moving=str(moving)
    # Historical logs either include the command or ANTs' explicit initialisation.
    explicit=f'fixed image: {fixed}' in text and f'moving image: {moving}' in text
    command=f'[{fixed},{moving},1]' in text or f'MI[{fixed},{moving},' in text
    if not (explicit or command):raise ValueError('Log does not establish the expected fixed/test and moving/reference identities')


def validate_pair(mapping,exe,root,dataset,reference,test):
    files=files_for_pair(root,reference,test)
    fixed_path=dataset/'derivatives/epochs'/test/'anat'/f'{test}_mod-T1w_maskface.nii'
    moving_path=dataset/'derivatives/epochs'/reference/'anat'/f'{reference}_mod-T1w_maskface.nii'
    fixed=nib.load(str(fixed_path));moving=nib.load(str(moving_path))
    if fixed.ndim!=3 or moving.ndim!=3:raise ValueError('Expected 3D native MRI inputs')
    terms=mapping.terms(files)
    results=[];details=[]
    for mode in ['affine','syn']:
        row=dict(reference_subject=reference,test_subject=test,registration_mode=mode,status='FAIL',reason='',
            fixed_mri=str(fixed_path),moving_mri=str(moving_path),fitting_time_mri_checksums='NOT_VERIFIED_HISTORICALLY',
            original_ants_version='NOT_VERIFIED_HISTORICALLY')
        try:
            if mode=='affine':
                verify_log_identity(files['aff_log'].read_text(),fixed_path,moving_path)
                warped=files['aff_image'];to_reference=terms['aff_t2r'];to_test=terms['aff_r2t']
            else:
                verify_log_identity(files['syn_log'].read_text(),fixed_path,moving_path)
                quality=json.loads(files['syn_quality'].read_text())
                if quality.get('donor')!=reference or quality.get('target')!=test:
                    raise ValueError('SyN quality subject identities differ')
                if Path(quality['fixed_mri']).resolve()!=fixed_path.resolve() or Path(quality['moving_mri']).resolve()!=moving_path.resolve():
                    raise ValueError('SyN fixed/moving MRI identifiers differ')
                details.extend([check_field(files['warp'],fixed),check_field(files['invwarp'],fixed)])
                inverse_image=nib.load(str(files['syn_inverse_image']))
                grid_error_mm(inverse_image,moving)
                warped=files['syn_image'];to_reference=terms['syn_t2r'];to_test=terms['syn_r2t']
            grid_error=grid_error_mm(nib.load(str(warped)),fixed)
            qa=mapping.image_qa(exe,fixed_path,moving_path,warped,to_reference,SAMPLES)
            if not np.isfinite(qa['rmse']) or qa['rmse']>MAX_RMSE:
                raise ValueError(f'MRI reproduction RMSE {qa["rmse"]:g} exceeds {MAX_RMSE}')
            inverse_rmse=None
            if mode=='syn':
                inverse_qa=mapping.image_qa(exe,moving_path,fixed_path,files['syn_inverse_image'],to_test,SAMPLES)
                inverse_rmse=inverse_qa['rmse']
                if not np.isfinite(inverse_rmse) or inverse_rmse>MAX_RMSE:
                    raise ValueError(f'Inverse MRI reproduction RMSE {inverse_rmse:g} exceeds {MAX_RMSE}')
            outbound=roundtrip_qa(mapping.point_map,exe,moving,fixed,to_test,to_reference)
            inbound=roundtrip_qa(mapping.point_map,exe,fixed,moving,to_reference,to_test)
            row.update(status='PASS_NUMERICAL_MAPPING_QA',grid_max_error_mm=grid_error,image_reproduction_rmse=qa['rmse'],inverse_image_reproduction_rmse=inverse_rmse,
                image_valid_samples=qa['n_valid'],reference_roundtrip_max_mm=outbound['max_roundtrip_mm'],
                test_roundtrip_max_mm=inbound['max_roundtrip_mm'],reference_coverage=outbound['valid']/outbound['tested'],
                test_coverage=inbound['valid']/inbound['tested'],
                reference_to_test_terms=json.dumps(to_test),test_to_reference_terms=json.dumps(to_reference),
                reason='Current MRI/log identities, grids, saved-image consistency and MRI-only roundtrips passed')
        except Exception as exc:row['reason']=f'{type(exc).__name__}: {exc}'
        results.append(row)
    return results,details


def locate_ants(requested):
    if requested is not None:path=requested.expanduser().resolve()
    else:
        found=shutil.which('antsApplyTransformsToPoints')
        path=Path(found) if found else Path.home()/'micromamba/envs/localizemi-registration/bin/antsApplyTransformsToPoints'
    if not path.is_file() or not path.name.startswith('antsApplyTransformsToPoints'):
        raise FileNotFoundError('Need antsApplyTransformsToPoints; use micromamba command in the opening comments or --ants-bin PATH')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment_review')
    p.add_argument('--bridge-root',type=Path,default=EXP/'outputs/06_coordinate_bridge')
    p.add_argument('--resource-root',type=Path,default=EXP/'source/01_registration_resources/outputs')
    p.add_argument('--output-root',type=Path,default=EXP/'outputs/06_registration_qa')
    p.add_argument('--ants-bin',type=Path)
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    if args.output_root.exists() or args.output_root.is_symlink():raise FileExistsError('Output exists; choose a NEW --output-root')
    protected=[args.preparation_root,args.bridge_root,args.resource_root,EXP/'source',EXP/'scripts',
               ROOT/'cross_subject',ROOT/'cross_subject_no_warp']
    protected += [EXP/'outputs'/name for name in ['03_reference_based_final','05_single_reference_final',
        '03_gamma_wholebrain_sensitivity','05_wholebrain_hemisphere_sensitivity','03_05_native_completion_audit','06_evaluate_localization']]
    for path in protected:
        if args.output_root==path or path in args.output_root.parents or args.output_root in path.parents:
            raise ValueError(f'Output overlaps protected input: {path}')
    mapping,source_path=historical_mapping()
    exe=locate_ants(args.ants_bin)
    bridge_manifest=args.bridge_root/'05_bridge_manifest.json'
    bridge=json.loads(bridge_manifest.read_text())
    if bridge.get('status')!='ALL_NATIVE_BRIDGES_PASS' or bridge.get('subjects_passed')!=7:
        raise ValueError('All seven native bridges must pass first')
    matrices=json.loads((args.bridge_root/'02_native_physical_bridges.json').read_text())
    if set(matrices)!=set(SUBJECTS):raise ValueError('Native bridge subject coverage differs')
    source_manifest=args.preparation_root/'10_source_provenance.json'
    doc=json.loads(source_manifest.read_text());dataset=Path(doc['locations']['dataset']).resolve()
    if args.output_root==dataset or dataset in args.output_root.parents or args.output_root in dataset.parents:raise ValueError('Output overlaps dataset')
    # Every imported registration/QA file must still match the recorded copy.
    checks=[];provenance={}
    def record(path,expected=None):
        path=Path(path).resolve()
        actual=sha(path)
        provenance[str(path)]=dict(path=str(path),sha256=actual,size_bytes=path.stat().st_size)
        if expected is not None and actual!=expected:raise ValueError(f'Checksum changed: {path}')
        return actual
    for index,item in enumerate(doc['imported_files'],1):
        copied=Path(item['copied_path']).resolve()
        if args.resource_root not in copied.parents:raise ValueError('Imported resources do not match --resource-root; explicitly reimport moved resources first')
        if index%50==1:print(f'Verifying imported checksums: {index}/{len(doc["imported_files"])}',flush=True)
        actual=record(copied,item['sha256'])
        checks.append(dict(path=str(copied),original_path=item['original_path'],sha256=actual,status='PASS_IMPORTED_CHECKSUM'))
    for item in json.loads((args.bridge_root/'04_source_provenance.json').read_text()):record(item['path'],item['sha256'])
    for subject in SUBJECTS:
        path=dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
        if str(path.resolve()) not in provenance:raise ValueError('Native bridge MRI provenance missing')
    # Confirm every file this helper will use was part of the verified import.
    for reference in SUBJECTS:
        for test in SUBJECTS:
            if reference==test:continue
            for path in files_for_pair(args.resource_root,reference,test).values():
                if str(path.resolve()) not in provenance:raise ValueError(f'Registration file lacks verified import provenance: {path}')
    version=subprocess.run([str(exe),'--version'],capture_output=True,text=True,timeout=30)
    ants_version=(version.stdout+'\n'+version.stderr).strip()[:2000]
    if version.returncode:ants_version='VERSION_QUERY_UNAVAILABLE: '+ants_version
    rows=[];fields=[]
    for reference in SUBJECTS:
        for test in SUBJECTS:
            if reference==test:continue
            print(f'Validating MRI-only point mapping: {reference} -> {test}',flush=True)
            pair_rows,details=validate_pair(mapping,str(exe),args.resource_root,dataset,reference,test)
            rows.extend(pair_rows)
            fields.extend(dict(reference_subject=reference,test_subject=test,**detail) for detail in details)
            for row in pair_rows:print(f'  {row["registration_mode"]}: {row["status"]}: {row["reason"]}',flush=True)
    for path in [source_manifest,bridge_manifest,args.bridge_root/'02_native_physical_bridges.json',
                 args.bridge_root/'04_source_provenance.json',source_path,exe,Path(__file__)]:record(path)
    # A resource modified during QA must not receive a passing published report.
    for path,item in list(provenance.items()):
        if sha(path)!=item['sha256']:raise ValueError(f'Input changed during QA: {path}')
    passed=sum(r['status']=='PASS_NUMERICAL_MAPPING_QA' for r in rows)
    complete=passed==84 and len(rows)==84
    manifest=dict(status='ALL_NUMERICAL_MAPPING_QA_PASS' if complete else 'REGISTRATION_QA_INCOMPLETE',
        pairs_expected=42,mode_pair_checks_expected=84,mode_pair_checks_passed=passed,
        point_mapping_numerically_validated=complete,ready_to_implement_mapped_evaluation=complete,
        anatomical_accuracy_established=False,localization_errors_calculated=False,
        inverse_reconstruction_executed=False,registration_fitting_executed=False,
        ants_executable=str(exe),ants_executable_sha256=sha(exe),ants_version=ants_version,
        samples=SAMPLES,foreground_seed=2901,max_roundtrip_mm=MAX_ROUNDTRIP_MM,
        min_coverage=MIN_COVERAGE,max_image_rmse=MAX_RMSE,grid_tolerance_mm=GRID_TOLERANCE_MM,
        registration_modes=['affine','syn'],reference=REFERENCE_URL,
        limitations=['Saved foreground/image consistency is not proof of cortical anatomical alignment.',
            'Fitting-time MRI byte identity and ANTs version have not been established by this helper; historical log/quality files remain available by checksum.',
            'Current MRI identity is tied to native bridge hashes and saved fixed/moving identifiers; saved-image reproduction is checked.',
            'These numerical checks use MRI foreground, not EEG source peaks or stimulation contacts.',
            'Mapped evaluation must retain per-point roundtrip/domain checks and numerical-QA/provenance labels.'])
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.registration_qa_',dir=args.output_root.parent) as temp:
        stage=Path(temp)/'results';stage.mkdir()
        pd.DataFrame(rows).to_csv(stage/'01_pair_mode_qa.csv',index=False)
        pd.DataFrame(checks).to_csv(stage/'02_import_checksum_audit.csv',index=False)
        (stage/'03_field_grid_audit.json').write_text(json.dumps(fields,indent=2)+'\n')
        (stage/'04_source_provenance.json').write_text(json.dumps(list(provenance.values()),indent=2)+'\n')
        (stage/'05_registration_qa_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (stage/'README.txt').write_text(__doc__+'\nNo localization errors or new fitted transforms were created.\n')
        if args.output_root.exists():raise FileExistsError(args.output_root)
        stage.rename(args.output_root)
    print(f'Numerical mapping QA: {passed} / 84 mode-pair checks passed. Reports: {args.output_root}')
    return 0 if complete else 2


if __name__=='__main__':raise SystemExit(main())
