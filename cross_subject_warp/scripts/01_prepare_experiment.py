#!/usr/bin/env python3
"""Prepare plans and audit local resources only; never fit registrations or inverses.

Existing destinations are refused, including partial imports. Run from any directory.
Exit 2 means preparation completed with unresolved readiness gates.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'cross_subject_warp'


def load_reference():
    path = EXP / 'source/00_no_warp_workflow/scripts/01_prepare_experiment.py'
    spec = importlib.util.spec_from_file_location('historical_preparation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path.home()/'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    p.add_argument('--cross-subject-outputs', type=Path, default=ROOT/'cross_subject/outputs')
    p.add_argument('--no-warp-outputs', type=Path, default=ROOT/'cross_subject_no_warp/outputs')
    p.add_argument('--output-root', type=Path, default=EXP/'outputs/01_prepare_experiment')
    p.add_argument('--resource-root', type=Path, default=EXP/'source/01_registration_resources/outputs')
    p.add_argument('--verify-existing-imports', action='store_true',
                   help='Read and checksum existing copies instead of writing them; requires every file to match')
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.expanduser().resolve())
    h = load_reference()
    import pandas as pd
    schedule_path = a.cross_subject_outputs/'30_all_reference_sesame_batch/experiment_schedule_all_references.csv'
    schedule = h.read_schedule(schedule_path)
    folds, members, cases, jobs = h.make_plan(schedule, 'sub-01', 'sub-02')
    counts = schedule[['test_subject','run']].drop_duplicates().groupby('test_subject').size().tolist()
    if counts != [11,6,7,9,14,7,7]:
        raise ValueError(f'Unexpected run counts: {counts}')
    # Fits are shared by affine/SyN: mapping modes belong to evaluation, not inverse jobs.
    jobs['source_condition'] = jobs.condition
    jobs['condition'] = jobs.condition.replace({'single_reference_no_warp':'cross_reference'})
    names = ['09_anatomical_registration','19_syn_registration','31_all_reference_registration']
    names += [x.name for x in sorted(a.cross_subject_outputs.iterdir())
              if x.is_dir() and x.name[:2] in {'10','11','12','13','14','15','16','18','20','21'}]
    imports = [a.cross_subject_outputs/n for n in names if (a.cross_subject_outputs/n).is_dir()]
    # Preflight every destination before the first write. No overwrite/resume switch.
    if a.output_root.exists() or a.output_root.is_symlink():
        raise FileExistsError(a.output_root)
    for src in imports:
        dst = a.resource_root/src.name
        if dst.is_symlink() or (dst.exists() and not a.verify_existing_imports):
            raise FileExistsError(dst)
        if a.verify_existing_imports and not dst.is_dir():
            raise FileNotFoundError(dst)
        if any(x.is_symlink() for x in src.rglob('*')):
            raise ValueError(f'Symlinks require explicit inspection: {src}')
    size = sum(x.stat().st_size for s in imports for x in s.rglob('*') if x.is_file())
    if not a.verify_existing_imports and shutil.disk_usage(EXP).free < size + 100_000_000:
        raise OSError(f'Insufficient disk space for {size} bytes')
    a.output_root.mkdir(parents=True, exist_ok=False)
    provenance = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                      locations={k:str(v) for k,v in vars(a).items()}, imported_files=[],
                      source_files=[], frozen_source_checks=[], missing=[])
    def record(path, role):
        if not path.is_file():
            provenance['missing'].append(dict(path=str(path), role=role))
            return
        provenance['source_files'].append(dict(path=str(path), role=role,
            size_bytes=path.stat().st_size, sha256=h.sha256(path)))
    record(schedule_path, 'authoritative schedule')
    original = json.loads((EXP/'source/provenance_manifest.json').read_text())
    for item in original['files']:
        path = EXP/item['path']
        actual = h.sha256(path) if path.is_file() else None
        provenance['frozen_source_checks'].append(dict(path=str(path), expected=item['sha256'],
            actual=actual, matches=actual == item['sha256']))
    for src in imports:
        print(f'{"Verifying" if a.verify_existing_imports else "Importing"} {src.name} ...', flush=True)
        dst = a.resource_root/src.name
        if not a.verify_existing_imports:
            dst.mkdir(parents=True, exist_ok=False)
        else:
            expected = {str(x.relative_to(src)) for x in src.rglob('*')}
            observed = {str(x.relative_to(dst)) for x in dst.rglob('*')}
            if expected != observed or any(x.is_symlink() for x in dst.rglob('*')):
                raise ValueError(f'Existing import tree differs: {dst}')
        for path in sorted(src.rglob('*')):
            target = dst/path.relative_to(src)
            if path.is_dir():
                if not a.verify_existing_imports:
                    target.mkdir(exist_ok=False)
            elif path.is_file():
                before = h.sha256(path)
                if not a.verify_existing_imports:
                    with path.open('rb') as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output, length=1024*1024)
                after = h.sha256(target)
                if before != after or before != h.sha256(path):
                    raise ValueError(f'Copy checksum/source stability failure: {path}')
                provenance['imported_files'].append(dict(original_path=str(path), copied_path=str(target),
                    sha256=before, copied_sha256=after, size_bytes=target.stat().st_size))
    audits=[]; transforms={}
    # Isolate MNE's writable configuration without touching the user's configuration.
    with tempfile.TemporaryDirectory(prefix='warp-mne-') as config_home:
        os.environ['_MNE_FAKE_HOME_DIR'] = config_home
        for subject in h.SUBJECTS:
            print(f'Auditing {subject}', flush=True)
            row, matrix = h.audit_subject(a.dataset, subject)
            audits.append(row)
            if matrix is not None:
                transforms[subject] = matrix
            model=a.dataset/'derivatives/sourcemodelling'/subject
            for path in [model/'fwd'/f'{subject}_fwd.fif', model/'xfm'/f'{subject}_from-head_to-surface.h5',
                         a.dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii',
                         ROOT/'outputs/12_additional_methods_all_runs/bem'/f'{subject}-bem-sol.fif']:
                record(path, 'model/MRI/BEM')
            epoch_dir=a.dataset/'derivatives/epochs'/subject
            for run in sorted(schedule.loc[schedule.test_subject.eq(subject), 'run'].unique()):
                for suffix in ['epochs.npy','epochs.json','epochs.tsv','channels.tsv']:
                    expected=epoch_dir/'eeg'/f'{subject}_task-seegstim_{run}_{suffix}'
                    if not expected.is_file():
                        provenance['missing'].append(dict(path=str(expected), role='required EEG run input'))
            contact=epoch_dir/'ieeg'/f'{subject}_task-seegstim_space-surface_electrodes.tsv'
            if not contact.is_file():
                provenance['missing'].append(dict(path=str(contact),role='stimulation contacts'))
            for path in sorted(epoch_dir.rglob('*')):
                if path.is_file() and path.suffix in {'.json','.tsv','.npy'}:
                    record(path, 'EEG/contact metadata or array')
    pairs=[]; qa=[]
    for test in h.SUBJECTS:
        for ref in h.SUBJECTS:
            if ref == test:
                continue
            pair=f'{ref}_to_{test}'
            for mode, folder in [('affine','09_anatomical_registration'),('syn','19_syn_registration')]:
                base=a.resource_root/folder/pair
                expected=([f'{pair}_0GenericAffine.mat', f'{ref}_in_{test}.nii.gz'] if mode=='affine' else
                    [f'{pair}_syn_0GenericAffine.mat',f'{pair}_syn_1Warp.nii.gz',f'{pair}_syn_1InverseWarp.nii.gz',
                     'donor_syn_in_target.nii.gz','target_syn_in_donor.nii.gz','syn_registration_quality.json','ants_registration.log'])
                missing=[n for n in expected if not (base/n).is_file() or (base/n).stat().st_size == 0]
                pairs.append(dict(reference_subject=ref,test_subject=test,registration_mode=mode,
                    resource_directory=str(base),files_complete=not missing,missing_files=';'.join(missing),
                    readiness='UNVALIDATED',fixed_subject=test,moving_subject=ref,
                    mri_input_hash_match='UNVERIFIED',physical_frame_bridge='UNRESOLVED'))
                for path in sorted(base.glob('*quality*.json')):
                    qa.append(dict(reference_subject=ref,test_subject=test,registration_mode=mode,
                        report_path=str(path),sha256=h.sha256(path),assessment='SAVED_EVIDENCE_ONLY'))
    canonical=[]
    for folder in ['03_reference_based_final','05_single_reference_final',
                   '03_gamma_wholebrain_sensitivity','05_wholebrain_hemisphere_sensitivity']:
        directory=a.no_warp_outputs/folder
        status=directory/({'03_gamma_wholebrain_sensitivity':'01_execution_status_wholebrain_variant.csv',
                           '05_wholebrain_hemisphere_sensitivity':'04_execution_status_variant.csv'}.get(folder,'01_execution_status.csv'))
        record(status, 'canonical/sensitivity execution records')
        entry=dict(folder=str(directory), exists=directory.is_dir(),reuse_status='PENDING_FINGERPRINT_CONFIG_SOFTWARE_VALIDATION')
        if status.is_file():
            table=pd.read_csv(status)
            entry.update(rows=len(table),columns=list(table.columns))
            if 'status' in table:
                entry['statuses']=table.status.value_counts().to_dict()
        canonical.append(entry)
        if directory.is_dir():
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path != status:
                    record(path,'canonical/sensitivity supporting evidence')
    provenance['canonical_reconstruction_inventory']=canonical
    for name in ['09_anatomical_registration','19_syn_registration','31_all_reference_registration']:
        if not (a.cross_subject_outputs/name).is_dir():
            provenance['missing'].append(dict(path=str(a.cross_subject_outputs/name),role='registration resource folder'))
    for filename, table in [('01_loso_folds.csv',folds),('02_fold_membership.csv',members),
        ('03_single_reference_cases.csv',cases),('04_reconstruction_plan.csv',jobs),
        ('05_coordinate_audit.csv',pd.DataFrame(audits)),('08_registration_pair_inventory.csv',pd.DataFrame(pairs)),
        ('09_registration_qa.csv',pd.DataFrame(qa,columns=['reference_subject','test_subject','registration_mode','report_path','sha256','assessment']))]:
        table.to_csv(a.output_root/filename,index=False)
    schedule.to_csv(a.output_root/'11_all_reference_cases.csv',index=False)
    unresolved=['Head/surface to NIfTI physical RAS bridge requires authoritative geometry for all seven subjects.',
        'Contact units and anatomical fiducials require metadata confirmation; numerical rigidity is insufficient.',
        'Historical MRI hashes, ANTs versions/commands, field grids and point directions require validation.',
        'Saved foreground QA is not proof of cortical alignment.',
        'Canonical inverse reuse requires configuration, software, data/channel fingerprints and peak policy verification.',
        'Retain historical hemisphere restriction and separate whole-brain sensitivities.']
    manifest=dict(experiment='LOSO/single-reference post-reconstruction affine/SyN mapping',
        preparation_script_sha256=h.sha256(__file__), n_subjects=7,n_runs=61,n_methods=12,n_folds=7,
        n_all_reference_cases=366,n_registration_pairs=42,n_loso_fits=1464,n_single_reference_fits=4392,
        registration_modes=['affine','syn'],registration_fitting_executed=False,inverse_reconstruction_executed=False,
        coordinate_audit_passed=all(r['status']=='PASS_NUMERICAL' for r in audits),
        registration_file_complete_by_mode={mode:sum(r['files_complete'] for r in pairs if r['registration_mode']==mode) for mode in ['affine','syn']},
        readiness='BLOCKED_UNRESOLVED_COORDINATES_AND_RESOURCE_VALIDATION',unresolved_assumptions=unresolved,
        missing_resources=provenance['missing'], frozen_sources_match=all(r['matches'] for r in provenance['frozen_source_checks']))
    for filename, content in [('06_coordinate_transforms.json',transforms),('07_experiment_manifest.json',manifest),('10_source_provenance.json',provenance)]:
        (a.output_root/filename).write_text(json.dumps(content,indent=2)+'\n')
    (a.output_root/'README.txt').write_text('Preparation/audit only. No registration fitting or inverse reconstruction.\n'
        'Affine and SyN share native reconstructed peaks. Fixed MRI=test; moving MRI=reference.\n'
        'Point mapping must retain historical RAS/LPS conversion and tested inverse transform order.\n'
        'Files complete does not mean anatomically validated. Original resources remain in place.\n'+'\n'.join(unresolved)+'\n')
    print(json.dumps(manifest,indent=2))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
