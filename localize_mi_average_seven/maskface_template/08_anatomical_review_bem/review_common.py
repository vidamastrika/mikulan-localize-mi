"""Paths and approval contract shared by the review and modeling stages."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '07_surface_diagnostics'))
from diagnose import FREESURFER, OUTPUT, ROOT, VAR, LAYERS, verify_upstream
from diagnose import fs_bridge
from stage2_common import Run, sha, write_json

DIAGNOSTICS = OUTPUT / 'run_01'
OUT = VAR / 'outputs/08_anatomical_review_bem'
FSAVERAGE = Path.home() / 'mne_data/MNE-fsaverage-data/fsaverage'


def new_run(path, stage):
    if OUT.resolve() not in path.resolve().parents:
        raise ValueError(f'Choose a fresh directory under {OUT}')
    run = Run(path, stage)
    run.use(Path(__file__))
    return run


def candidate(subject):
    t1 = FREESURFER / 'subjects' / subject / 'mri/T1.mgz'
    surfaces = {layer: FREESURFER / 'watershed_surfaces' / f'{subject}_{layer}_surface'
                for layer in LAYERS}
    return t1, surfaces


def verified_diagnostics(run):
    subject = verify_upstream(run)
    run.verify(DIAGNOSTICS, stage='07_SURFACE_DIAGNOSTIC')
    return subject


def approve(run, review, approval):
    subject = verified_diagnostics(run)
    run.verify(review, stage='08_ANATOMICAL_REVIEW')
    record = json.loads(run.use(approval).read_text())
    expected = json.loads(run.use(review / 'APPROVAL_PENDING.json').read_text())
    if record.get('review_manifest_sha256') != sha(review / 'manifest.json'):
        raise ValueError('Approval must identify the exact completed review package')
    for key in ('candidate', 'input_sha256'):
        if record.get(key) != expected[key]:
            raise ValueError(f'Approval does not match reviewed {key}')
    if record.get('decision') != 'APPROVED_FOR_BEM_CONSTRUCTION':
        raise ValueError('Anatomical approval is required')
    for key in ('reviewer', 'review_date', 'anatomical_rationale'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f'Approval requires {key}')
    for key in ('cranial_vault', 'face_orbits', 'inferior_skull_base', 'frame_and_layer_correspondence'):
        if record.get('regional_decisions', {}).get(key) != 'ACCEPTABLE_FOR_EXPLORATORY_BEM':
            raise ValueError(f'Anatomical review required for {key}')
    for path, checksum in expected['input_sha256'].items():
        run.use(path, checksum)
    run.doc['anatomical_approval'] = record
    return subject
