#!/usr/bin/env python3
# OPTIONAL REVIEW-ONLY MODE (same Script 01; no later stage implemented)
# --review-preparation-root reads an existing preparation's coordinate CSV,
# transform provenance, frozen fsaverage scalp and readiness/manifest files.
# It checks mesh closure/orientation/connectivity, calculates exact unsigned
# point-to-triangle distances and winding-number interior classifications,
# then produces two 300-dpi PNGs, a per-electrode CSV, JSON report and README.
# It never refreezes selections, relocates electrodes, or changes preparation QA.
# Numerical classification tolerances (0.001 mm at surface; winding 1e-5) are
# diagnostic tolerances, not replacement acceptance gates. Self-intersections
# of the mesh are not tested. Additional dependency for this mode: Matplotlib.
# Local review command (fresh output folder):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
#   --review-preparation-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_assumed_v1 \
#   --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_geometry_review_assumed_v1
# OPTIONAL ACTUAL-BEM INTERPOLATION AUDIT (same Script 01)
# --audit-bem-preparation-root reads the existing preparation and its hash-pinned
# three-layer fsaverage BEM solution. It inspects the actual BEM scalp, which can
# differ from outer_skin.surf, and reproduces only MNE's geometric interpolation
# query for one monopolar EEG location per channel. This derives audit-only
# effective locations; stored template coordinates and original QA stay intact.
# Outputs: two 300-dpi PNGs, per-channel coordinates/weights/displacements CSV,
# JSON provenance/mesh/displacement report, README in a fresh output directory.
# Requires MNE 1.12.1 (private API reviewed), Matplotlib, and existing dependencies.
# Effective positions must agree with independent selected-triangle PLANE
# projections to 0.001 mm. MNE reproject=True may extrapolate beyond triangle
# edges; record those weights and actual surface residuals rather than assuming
# a clamped nearest-surface projection. This is not an anatomical acceptance gate.
# No BEM construction, forward matrix, inverse, EEG or individual geometry input.
# Local command (repository root; user executes manually):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
#   --audit-bem-preparation-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_assumed_v1 \
#   --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_bem_sensor_audit_assumed_v1
# RESEARCH OBJECTIVE
# Prepare Localize-MI localization with fsaverage anatomy and a documented template
# EEG cap, assuming only test EEG and channel identities are available. Individual
# MRI, digitization, individual rigid/ICP fitting, affine/SyN registration and
# stimulation outcomes must never determine template selection or geometry.
#
# SCOPE / ASSUMPTIONS
# Script 01 ONLY: schedule/configuration freezing and template QA. It does not read
# EEG samples, participant forwards, participant anatomy/transforms, contacts or
# localization results. It does not preprocess, compute forwards or run inverses.
# Channel labels alone are insufficient evidence of physical electrode identity.
# USER-AUTHORIZED EXPLORATORY POLICY (separate opt-in; original audit preserved)
# --mapping-policy documented-assumption --assumption-record PATH explicitly
# allows eN->EN for the pinned GSN-HydroCel-256 resource as ASSUMED_UNVERIFIED.
# This permits template geometry inspection, not a claim of physical mapping
# validation. Verified mode remains the default. Geometry failures still BLOCK.
# Use fresh --source-root/--output-root; see assumptions/numbered_hydrocel_v1/.
# Historical parameter selections transfer unchanged; their scores are provenance,
# not input to montage selection. Geometry QA is performed only after independent
# documentary channel correspondence and template conventions have been validated.
#
# INPUTS (read-only; defaults relative to repository root)
# cross_subject_fsaverage/outputs/01_prepare_experiment/source_configurations/:
#   test_runs.csv (61), loso_folds.csv (7), single_schedule.csv (366),
#   loso_selection.csv (84), single_selection.csv (84), selection manifests.
# cross_subject_fsaverage/outputs/05_reconstruct_sub03_exploratory/:
#   05_resolved_configurations.json (168 resolved parameter records; no estimates).
# Frozen no-warp solver helpers and preprocessing/channel-policy source code.
# MNE installed standard-montage resources; local fsaverage scalp/fiducials/trans.
# Optional --mapping-evidence: independently reviewed evidence JSON + full mapping
# CSV and documentary files (schema in source/01_montage/README.md).
#
# OUTPUTS (all NEW files within this variant; originals remain read-only)
# source/01_montage/: raw candidate resources and parser snapshots; evidence copies;
#   validated template coordinates/subsets ONLY if documentary validation succeeds.
# source/02_frozen_configurations/: byte-exact configuration/code snapshots + hashes.
# outputs/01_prepare_experiment/: 01_experiment_manifest.json,
#   02_planned_reconstruction_schedule.csv, 03_frozen_configuration_inventory.csv,
#   04_template_montage_availability.csv, 05_channel_mapping_validation.csv,
#   08_readiness_report.json, README.md; ONLY with validated mapping:
#   06_sensor_scalp_geometry_qa.csv, 07_coordinate_transform_provenance.json,
#   09_fiducial_alignment_qa.csv. Geometry failures are reported, not repaired.
#
# PARAMETERS / VALIDATION
# --montage-kind defaults to GSN-HydroCel-256 (candidate, never assumed valid).
# --fsaverage-root defaults to ~/mne_data/MNE-fsaverage-data/fsaverage.
# --frozen-design-root and --resolved-configurations override frozen design inputs.
# --mapping-evidence is optional; absence yields BLOCKED, with reports still written.
# --output-root defaults to this variant's outputs/01_prepare_experiment; it must
# be empty/new. --source-root defaults to this variant's source. Both must remain
# strictly inside this variant. Existing files are never overwritten. No downloads.
# Seven subjects, 61 runs, 12 methods, 732 LOSO + 4,392 Single Reference = 5,124
# evaluations; these are NOT 5,124 distinct forward computations.
# Exact keys, reference identities, training exclusions, configuration keys and
# recorded parameters/windows are validated; missing settings are never inferred.
# Explicit one-to-one e1..e256 correspondence and documentary review are required.
# No fsaverage_1005 nearest-neighbour relabeling; no outcome-based choice.
# Fixed MNE head_size='auto' uses the predefined 95-mm normalization for spherical
# caps. This is NOT a unit conversion or a fit; no scale search or coordinate
# correction runs. The optional BEM audit derives solver interpolation positions
# separately without changing input coordinates or granting a geometry PASS.
# Its scientific appropriateness and fiducial interpretation require evidence.
# Template-native->HEAD uses only template fiducials. HEAD->fsaverage MRI uses
# the supplied fsaverage predefined transform, never a test-subject transform.
# Nearest scalp-VERTEX distances are upper bounds on triangle-surface distances.
# Existing fsaverage numerical gates remain median <=10 mm, p95 <=20 mm;
# no thresholds are relaxed and numerical PASS still requires scientific review.
#
# DEPENDENCIES
# Python >=3.10, NumPy, pandas, SciPy, MNE; local frozen files and fsaverage assets.
# Geometry inspection reads only template/fsaverage assets and uses a KD tree;
# no BEM/forward construction, registration optimization or inverse fit is called.
#
# COMPLETE TERMINAL EXAMPLES (repository root; run manually after review)
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py
# User-authorized assumption audit (no forward or inverse operations):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
#   --mapping-policy documented-assumption \
#   --assumption-record cross_subject_fsaverage/template_montage_no_digitization/assumptions/numbered_hydrocel_v1/decision.json \
#   --source-root cross_subject_fsaverage/template_montage_no_digitization/source_assumed_v1 \
#   --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_assumed_v1
# With independently reviewed evidence (see schema):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
#   --mapping-evidence /absolute/path/to/evidence.json \
#   --source-root cross_subject_fsaverage/template_montage_no_digitization/source_reviewed \
#   --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_reviewed
# Exit codes: 0 preparation numerically ready (review required); 3 scientific/input
# BLOCKED with reports; 2 invalid CLI/protected/existing output path. Never delete
# or bypass a frozen manifest to retry. Use new source/output roots on later runs.
"""Prepare a separate template-montage experiment; default to blocked without evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

VARIANT = Path(__file__).resolve().parents[1]
ROOT = VARIANT.parents[1]
FS = ROOT / 'cross_subject_fsaverage'
NW = ROOT / 'cross_subject_no_warp'
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 8)]
METHODS = ['MNE', 'dSPM', 'sLORETA', 'eLORETA', 'Continuous-ECD', 'RAP-MUSIC',
           'LCMV', 'MxNE', 'irMxNE', 'Gamma-MAP', 'Champagne', 'SESAME']
CHANNELS = [f'e{i}' for i in range(1, 257)]
KEY = ['strategy', 'method', 'reference_subject', 'test_subject', 'run']
WINDOWS = dict(covariance_tmin_s=-.250, covariance_tmax_s=-.050,
               target_tmin_s=-.002, target_tmax_s=.002)
SNAPSHOT_NAMES = ['test_runs.csv', 'loso_folds.csv', 'single_schedule.csv',
                  'loso_selection.csv', 'single_selection.csv',
                  'loso_selection_manifest.json', 'single_selection_manifest.json']
DOC_ROLES = {'dataset_net_and_numbering', 'template_numbering',
             'fiducial_coordinate_convention', 'template_unit_scaling'}
REQUIRED_SETTINGS = {
    **{m:['loose','depth','snr','lambda2'] for m in METHODS[:4]},
    'Continuous-ECD':['ecd_min_dist_mm'], 'RAP-MUSIC':['rap_music_n_dipoles'],
    'LCMV':['lcmv_reg','lcmv_window_half_ms','lcmv_pick_ori','lcmv_weight_norm'],
    'MxNE':['mxne_alpha','mxne_loose','mxne_depth'],
    'irMxNE':['irmxne_alpha','irmxne_loose','irmxne_depth','irmxne_iterations','irmxne_random_state'],
    'Gamma-MAP':['gamma_alpha','gamma_loose','gamma_depth','gamma_xyz_same_gamma','gamma_update_mode','gamma_maxit','gamma_tol'],
    'Champagne':['champagne_alpha','champagne_prune','champagne_pruning_thresh','champagne_convergence_criterion','champagne_max_iter'],
    'SESAME':['sesame_fs','sesame_n_parts','sesame_max_n_dips','sesame_lam','sesame_hyper_q','sesame_random_seed'],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def snapshot(path, destination, inventory, role):
    path = Path(path).resolve()
    expected = sha(path)
    require(not destination.exists(), f'Refusing to replace snapshot: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    require(sha(destination) == expected and sha(path) == expected, f'Input changed during snapshot: {path}')
    inventory.append(dict(role=role, original_path=str(path), copy_path=str(destination),
                          sha256=expected, size_bytes=path.stat().st_size))
    return destination


def exact(frame, columns, expected, label):
    require(set(columns) <= set(frame), f'{label}: missing columns')
    require(not frame.duplicated(columns).any(), f'{label}: duplicate identities')
    actual = set(map(tuple, frame[columns].itertuples(index=False, name=None)))
    require(actual == expected and len(frame) == len(expected), f'{label}: identities differ from design')


def freeze_design(args, source, inventory):
    frozen = source / '02_frozen_configurations'
    for name in SNAPSHOT_NAMES:
        snapshot(args.frozen_design_root / name, frozen / name, inventory, 'frozen_design')
    resolved_path = snapshot(args.resolved_configurations, frozen / 'resolved_configurations.json',
                             inventory, 'fully_resolved_inverse_settings')
    snapshot(args.frozen_design_root.parent/'02_fsaverage_reconstruction_plan.csv',
             frozen/'original_fsaverage_job_schedule.csv',inventory,'original_job_identities')
    # Snapshot policy definitions, not participant geometry or old outcomes.
    for path in [NW/'scripts/03_run_localization.py', ROOT/'src/localize_mi/io.py',
                 NW/'source/03_solver_code/scripts/09_full_parameter_grid.py',
                 NW/'source/03_solver_code/src/localize_mi/io.py',
                 NW/'source/03_solver_code/src/localize_mi/inverse.py',
                 NW/'source/03_solver_code/scripts/19_next_method_pilot.py']:
        snapshot(path, frozen/'policy_code'/path.relative_to(ROOT), inventory, 'policy_reference_only_do_not_execute_loader')
    read = lambda name: pd.read_csv(frozen / name, dtype=str, keep_default_na=False)
    runs, folds, single_schedule = [read(n) for n in ['test_runs.csv','loso_folds.csv','single_schedule.csv']]
    run_keys = set(zip(runs.test_subject, runs.run))
    require(len(runs) == len(run_keys) == 61 and set(runs.test_subject) == set(SUBJECTS), 'Expected seven subjects and 61 runs')
    require(runs.groupby('test_subject').size().to_dict()==dict(zip(SUBJECTS,[11,6,7,9,14,7,7])), 'Per-subject run counts differ from Localize-MI frozen design')
    require(not runs.duplicated(['test_subject','run']).any(), 'Duplicate test runs')
    exact(folds, ['test_subject'], {(s,) for s in SUBJECTS}, 'LOSO folds')
    for row in folds.itertuples():
        require(set(row.training_subjects.split(';')) == set(SUBJECTS)-{row.test_subject}, 'LOSO training leakage/missing subject')
        require(row.reference_subject in set(SUBJECTS)-{row.test_subject}, 'Invalid LOSO reference')
        require(runs.loc[runs.test_subject.eq(row.test_subject),'reference_subject'].eq(row.reference_subject).all(), 'LOSO run/fold reference mismatch')
    expected_single = {(ref,test,run) for test,run in run_keys for ref in SUBJECTS if ref != test}
    exact(single_schedule, ['reference_subject','test_subject','run'], expected_single, 'Single Reference schedule')
    resolved = json.loads(resolved_path.read_text())
    expected_resolved = set()
    selections = {}
    for strategy, name, owner in [('LOSO','loso_selection.csv','test_subject'),
                                   ('Single-reference','single_selection.csv','reference_subject')]:
        selected = read(name)
        exact(selected, ['method',owner], {(m,s) for m in METHODS for s in SUBJECTS}, strategy)
        require(selected.selection_status.eq('SELECTED').all(), f'{strategy}: unselected configuration')
        for row in selected.to_dict('records'):
            subject = row[owner]
            selection_json = json.loads(row['configuration_parameters'])
            if strategy == 'LOSO':
                require(set(row['training_subjects'].split(';')) == set(SUBJECTS)-{subject}, 'Selection training set mismatch')
                require(row['reference_subject'] == folds.set_index('test_subject').loc[subject,'reference_subject'], 'Selection reference differs from fold')
            else:
                require(row['training_subjects']==subject, 'Single Reference selection must use only its reference subject')
            key = f'{strategy}|{row["method"]}|{subject}'
            expected_resolved.add(key)
            cfg = resolved[key]
            require(cfg['configuration_key'] == row['configuration_key'], f'{key}: changed configuration key')
            parameters = cfg['parameters']
            require(set(REQUIRED_SETTINGS[row['method']]) <= set(parameters), f'{key}: incomplete resolved method settings')
            require(set(selection_json) <= set(parameters), f'{key}: missing selected settings')
            for field, value in selection_json.items():
                actual = parameters[field]
                # Numeric strings and resolved numbers may differ in serialization only.
                if isinstance(value, bool) or isinstance(actual, bool):
                    equal = str(value).lower() == str(actual).lower()
                else:
                    try:
                        equal = np.isclose(float(value), float(actual), rtol=1e-12, atol=1e-12)
                    except (ValueError, TypeError):
                        equal = str(value).lower() == str(actual).lower()
                require(equal, f'{key}: resolved setting differs: {field}')
            for field,value in WINDOWS.items():
                require(field in parameters and np.isclose(float(parameters[field]),value,rtol=0,atol=1e-12), f'{key}: missing/changed time window')
            require(str(parameters.get('montage')) in {'32','64','128','all_good'}, f'{key}: unsupported montage size')
            selections[(strategy,row['method'],subject)] = cfg
    require(set(resolved) == expected_resolved, 'Resolved parameter inventory must have exactly 168 entries')
    schedule = pd.concat([
        runs[['reference_subject','test_subject','run']].assign(strategy='LOSO'),
        single_schedule[['reference_subject','test_subject','run']].assign(strategy='Single-reference')],ignore_index=True)
    schedule = schedule.merge(pd.DataFrame({'method':METHODS}),how='cross')
    require(len(schedule)==5124 and not schedule.duplicated(KEY).any(), 'Expected 5,124 unique jobs')
    require(schedule.groupby('strategy').size().to_dict()=={'LOSO':732,'Single-reference':4392}, 'Strategy counts changed')
    original=read('original_fsaverage_job_schedule.csv')
    exact(schedule,KEY,set(map(tuple,original[KEY].itertuples(index=False,name=None))),'Original full job schedule')
    records = []
    for row in schedule.to_dict('records'):
        owner = row['test_subject'] if row['strategy']=='LOSO' else row['reference_subject']
        cfg = selections[(row['strategy'],row['method'],owner)]
        records.append(row | dict(model_subject='fsaverage', sensor_geometry='template_only',
            mapping_policy=args.mapping_policy,
            mapping_status='ASSUMED_UNVERIFIED' if args.mapping_policy=='documented-assumption' else 'DOCUMENTARY_VALIDATION_PENDING',
            parameter_owner=owner, configuration_key=cfg['configuration_key'],
            configuration_parameters=json.dumps(cfg['parameters'],sort_keys=True),
            montage_size=str(cfg['parameters']['montage']),
            coordinate_policy='no_individual_MRI_or_digitization',status='PLANNED_NOT_EXECUTED'))
    return pd.DataFrame(records)


def montage_catalog(source, inventory):
    import mne
    resources = Path(mne.__file__).parent / 'channels/data/montages'
    available = mne.channels.get_builtin_montages()
    candidates = sorted({name for name in available if
        name.startswith(('GSN-','EGI_')) or name in {'biosemi256','standard_1005'}})
    rows, montages, paths = [], {}, {}
    for name in candidates:
        matches = list(resources.glob(name+'.*'))
        row = dict(montage=name, available=True, n_channels=None,
                   exact_casefold_e1_e256=False, correspondence='NOT_VALIDATED_BY_LABELS',
                   eligible_model_family=name in {'GSN-HydroCel-256','EGI_256'},
                   loader_head_size='auto (predefined 0.095 m for spherical caps)',
                   native_coordinate_units='MNE output metres; raw units/normalization require documentary review',
                   exception='')
        try:
            montage = mne.channels.make_standard_montage(name,head_size='auto')
            pos = montage.get_positions()
            folded = [c.casefold() for c in montage.ch_names]
            row.update(n_channels=len(montage.ch_names), coordinate_frame=pos['coord_frame'],
                       exact_casefold_e1_e256=len(folded)==256 and set(folded)==set(CHANNELS),
                       duplicate_casefold_labels=len(folded)-len(set(folded)),
                       fiducials_present=all(pos[k] is not None for k in ['nasion','lpa','rpa']),
                       fiducial_origin='electrode-derived estimates in MNE' if name=='EGI_256' else
                           'resource fiducials; anatomical landmark interpretation requires review')
            montages[name] = montage
            for path in matches:
                paths[name] = path
                snapshot(path,source/'01_montage/candidate_resources'/path.name,inventory,'unvalidated_MNE_template_resource')
                row.update(resource_path=str(path),resource_sha256=sha(path))
        except (ValueError, OSError, RuntimeError) as exc:
            row['exception'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
    # Freeze the parser/normalization definitions as well as resource bytes.
    for relative in ['channels/_standard_montage_utils.py','channels/montage.py']:
        path = Path(mne.__file__).parent/relative
        snapshot(path,source/'01_montage/MNE_code'/relative,inventory,'MNE_montage_parser')
    return pd.DataFrame(rows), montages, paths


def validate_mapping(args, montage, resource, source, inventory):
    mapping = pd.DataFrame({'dataset_channel':CHANNELS, 'template_channel':'',
                            'validation_status':'BLOCKED_NO_DOCUMENTARY_EVIDENCE', 'evidence_id':''})
    # Lexical suggestions are availability diagnostics, not accepted mappings.
    mapping['unvalidated_label_candidate'] = [
        ';'.join(name for name in montage.ch_names if name.casefold()==c) for c in CHANNELS]
    mapping['validation_detail']='Physical correspondence has not been established independently.'
    reasons = []
    if args.mapping_policy=='documented-assumption':
        try:
            require(args.assumption_record is not None,'Explicit --assumption-record is required')
            record=json.loads(args.assumption_record.read_text())
            require(args.montage_kind==record['montage_kind']=='GSN-HydroCel-256', 'Assumption mode is restricted to the documented HydroCel candidate')
            pinned='63d3de763f3bd36bb9096989a828068edcb60fea81607ddb96022d3bd2126e1c'
            require(sha(resource)==record['template_resource_sha256']==pinned,'Assumption record/template differs from the reviewed resource')
            require(record['status']=='ASSUMED_UNVERIFIED','Assumption must remain labeled unverified')
            require(record['physical_channel_correspondence_verified'] is False,'Do not present an assumption as verified correspondence')
            require(record['accepted_for_exploratory_geometry_only'] is True,'Exploratory geometry assumption was not accepted')
            require(record['independent_of_individual_geometry_and_localization_outcomes'] is True,'Assumption must not use individual geometry or localization outcomes')
            require(set(record['subjects_covered'])==set(SUBJECTS),'Assumption must explicitly cover all seven subjects')
            require(record['mapping_convention']=='eN -> EN for N=1..256','Unsupported numbering assumption')
            require(record['template_head_size']=='MNE head_size=auto, predefined 0.095 m spherical normalization','Normalization assumption differs from the fixed loader')
            require(record.get('authorization','').strip() and record.get('rationale','').strip(), 'Assumption needs recorded authorization and rationale')
            require(record.get('fiducial_convention','').strip() and record.get('supporting_sources'), 'Assumption needs explicit fiducial convention and supporting sources')
            require(len(montage.ch_names)==256 and set(montage.ch_names)=={f'E{i}' for i in range(1,257)},'Template names do not support the assumed bijection')
            mapping=pd.DataFrame({'dataset_channel':CHANNELS,
                'template_channel':[f'E{i}' for i in range(1,257)],
                'validation_status':'ASSUMED_UNVERIFIED','evidence_id':'numbered_hydrocel_v1_assumption',
                'validation_detail':'User-authorized identity-number assumption; exact acquisition/export correspondence is unverified.'})
            snapshot(args.assumption_record,source/'01_montage/assumptions/decision.json',inventory,'explicit_unverified_model_assumption')
        except (ValueError,KeyError,TypeError,OSError) as exc:
            reasons.append(f'Assumption record rejected: {type(exc).__name__}: {exc}')
            mapping['validation_status']='BLOCKED_INVALID_ASSUMPTION'
            mapping['validation_detail']=reasons[-1]
        return mapping,reasons
    if args.mapping_evidence is None:
        return mapping, ['Independent Localize-MI net model/numbering and fiducial/unit evidence has not been supplied.']
    try:
        evidence_path = args.mapping_evidence.resolve()
        evidence = json.loads(evidence_path.read_text())
        require(evidence['montage_kind']==args.montage_kind, 'Evidence refers to a different template')
        require(evidence['template_resource_sha256']==sha(resource), 'Evidence template resource hash differs')
        require(set(evidence['subjects_covered'])==set(SUBJECTS), 'Acquisition/net evidence must cover all seven subjects')
        for field in ['reviewer','reviewed_utc','dataset_net_model','numbering_correspondence_explanation',
                      'fiducial_interpretation','unit_and_scaling_interpretation']:
            require(isinstance(evidence.get(field),str) and evidence[field].strip(), f'Missing evidence field: {field}')
        require(evidence['independent_of_subject_geometry_and_localization_outcomes'] is True, 'Evidence must be independent of outcomes/subject geometry')
        require(evidence['template_head_size_auto_095m_accepted'] is True, 'Fixed MNE scale convention has not been accepted')
        require(evidence['fiducial_convention_verified'] is True, 'Meaning of template fiducials remains unverified')
        docs = evidence['documents']
        require(DOC_ROLES <= {doc['role'] for doc in docs}, 'Missing required documentary roles')
        require(len({doc['id'] for doc in docs})==len(docs), 'Duplicate evidence IDs')
        for doc in docs:
            require(doc.get('citation','').strip() and doc.get('relevant_excerpt_or_section','').strip(), 'Document needs citation and relevant section')
            path = (evidence_path.parent/doc['path']).resolve()
            require(sha(path)==doc['sha256'], f'Document hash mismatch: {doc["id"]}')
        path = (evidence_path.parent/evidence['mapping_csv']).resolve()
        require(sha(path)==evidence['mapping_sha256'],'Mapping CSV hash differs from evidence')
        supplied = pd.read_csv(path,dtype=str,keep_default_na=False)
        require({'dataset_channel','template_channel','evidence_id'} <= set(supplied), 'Mapping columns missing')
        mapping=supplied[['dataset_channel','template_channel','evidence_id']].copy()
        missing=set(CHANNELS)-set(mapping.dataset_channel)
        if missing:
            mapping=pd.concat([mapping,pd.DataFrame({'dataset_channel':sorted(missing),
                'template_channel':'','evidence_id':''})],ignore_index=True)
        mapping['duplicate_dataset_label']=mapping.dataset_channel.duplicated(keep=False)
        mapping['duplicate_template_label']=mapping.template_channel.ne('') & mapping.template_channel.duplicated(keep=False)
        mapping['unknown_dataset_label']=~mapping.dataset_channel.isin(CHANNELS)
        mapping['missing_or_unknown_template_label']=~mapping.template_channel.isin(montage.ch_names)
        mapping['validation_status']='BLOCKED_INVALID_EVIDENCE'
        exact(supplied,['dataset_channel'],{(c,) for c in CHANNELS},'Localize-MI channels')
        require(not supplied.template_channel.duplicated().any(),'Duplicate physical template mapping')
        require(len(montage.ch_names)==256 and len(set(montage.ch_names))==256,'Template must contain exactly 256 distinct EEG sensors')
        require(len({c.casefold() for c in montage.ch_names})==256,'Ambiguous case-insensitive template labels')
        require(set(supplied.template_channel)==set(montage.ch_names),'Mapping must cover exact template channel names (case sensitive)')
        require(set(supplied.evidence_id)<={doc['id'] for doc in docs},'Unknown channel evidence ID')
        require(not supplied.evidence_id.eq('').any(),'Every correspondence needs documentary evidence')
        mapping = supplied[['dataset_channel','template_channel','evidence_id']].copy()
        mapping['validation_status']='VALIDATED_DOCUMENTARY_REVIEW'
        snapshot(evidence_path,source/'01_montage/evidence/evidence.json',inventory,'independent_mapping_review')
        snapshot(path,source/'01_montage/evidence/channel_mapping.csv',inventory,'explicit_channel_correspondence')
        for index,doc in enumerate(docs):
            path=(evidence_path.parent/doc['path']).resolve()
            snapshot(path,source/'01_montage/evidence/documents'/f'{index:02d}_{path.name}',inventory,'document_'+doc['role'])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        reasons.append(f'Mapping evidence rejected: {type(exc).__name__}: {exc}')
        mapping['validation_status']='BLOCKED_INVALID_EVIDENCE'
        mapping['validation_detail']=reasons[-1]
    return mapping,reasons


def template_geometry(args, montage, mapping, source, inventory, out):
    import mne
    from mne.io.constants import FIFF
    from scipy.spatial import cKDTree
    before=montage.get_positions()
    require(before['coord_frame']=='unknown','Only reviewed native-template unknown-frame resources are supported')
    require(all(before[k] is not None for k in ['nasion','lpa','rpa']), 'Missing template fiducials; no invented landmarks allowed')
    xyz=np.array([before['ch_pos'][name] for name in mapping.template_channel])
    require(np.isfinite(xyz).all() and len(np.unique(xyz,axis=0))==256,'Nonfinite or duplicated template positions')
    require(np.max(np.linalg.norm(xyz,axis=1))<1,'Implausible template metre scale')
    native_head=mne.channels.compute_native_head_t(montage,on_missing='raise')
    require(int(native_head['to'])==int(FIFF.FIFFV_COORD_HEAD),'Template transform does not end in HEAD')
    fsroot=args.fsaverage_root.expanduser().resolve()
    paths={name:fsroot/'bem'/fname for name,fname in {
        'head_mri':'fsaverage-trans.fif','fiducials':'fsaverage-fiducials.fif','scalp':'outer_skin.surf'}.items()}
    for name,path in paths.items():
        # Files are fsaverage-only, never derived from a subject's fitting output.
        snapshot(path,source/'01_montage/fsaverage_geometry'/path.name,inventory,'fsaverage_'+name)
    head_mri=mne.read_trans(paths['head_mri'],verbose=False)
    if (int(head_mri['from']),int(head_mri['to']))==(int(FIFF.FIFFV_COORD_MRI),int(FIFF.FIFFV_COORD_HEAD)):
        head_mri=mne.transforms.invert_transform(head_mri)
    require((int(head_mri['from']),int(head_mri['to']))==(int(FIFF.FIFFV_COORD_HEAD),int(FIFF.FIFFV_COORD_MRI)), 'Predefined fsaverage transform has wrong direction')
    for transform in [native_head,head_mri]:
        matrix=np.asarray(transform['trans'])
        require(np.isfinite(matrix).all() and np.allclose(matrix[3],[0,0,0,1],atol=1e-5,rtol=0), 'Invalid homogeneous transform')
        rotation=matrix[:3,:3]
        require(np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5,rtol=0) and abs(np.linalg.det(rotation)-1)<1e-5,'Template transform is not rigid')
    head=mne.transforms.apply_trans(native_head,xyz)
    mri=mne.transforms.apply_trans(head_mri,head)
    vertices,triangles=mne.read_surface(paths['scalp'],verbose=False)
    # FreeSurfer .surf stores surface RAS in mm; MNE read_surface does not convert.
    require(np.isfinite(vertices).all() and len(vertices)>3,'Invalid scalp surface')
    vertices_m=vertices/1000.
    distance,index=cKDTree(vertices_m).query(mri)
    require(np.isfinite(distance).all(), 'Nonfinite scalp distances')
    # Report local normal offsets, not a claimed robust inside/outside classifier.
    surface=mne.surface.complete_surface_info(dict(rr=vertices_m,tris=triangles),copy=True,verbose=False)
    normal_offset=((mri-vertices_m[index])*surface['nn'][index]).sum(axis=1)*1000
    qa=mapping.copy()
    for axis,i in zip('xyz',range(3)):
        qa[f'template_native_{axis}_m']=xyz[:,i]
        qa[f'template_head_{axis}_m']=head[:,i]
        qa[f'fsaverage_surface_ras_{axis}_m']=mri[:,i]
    qa['nearest_scalp_vertex']=index
    qa['nearest_scalp_vertex_distance_mm']=distance*1000
    qa['nearest_vertex_normal_offset_mm']=normal_offset
    qa['normal_offset_interpretation']='local normal diagnostic only; not a reliable signed surface distance'
    qa.to_csv(out/'06_sensor_scalp_geometry_qa.csv',index=False)
    qa.to_csv(source/'01_montage/template_coordinates.csv',index=False)
    fsfid,frame=mne.io.read_fiducials(paths['fiducials'],verbose=False)
    require(int(frame)==int(FIFF.FIFFV_COORD_MRI),'Fsaverage fiducials must be MRI/surface RAS')
    fidmap={int(point['ident']):np.asarray(point['r']) for point in fsfid}
    fidrows=[]
    for name,ident in [('nasion',FIFF.FIFFV_POINT_NASION),('lpa',FIFF.FIFFV_POINT_LPA),('rpa',FIFF.FIFFV_POINT_RPA)]:
        transformed=mne.transforms.apply_trans(head_mri,mne.transforms.apply_trans(native_head,before[name]))
        require(int(ident) in fidmap,'Fsaverage fiducial missing')
        fidrows.append(dict(fiducial=name,template_fsaverage_mri_xyz_m=json.dumps(transformed.tolist()),
            fsaverage_mri_xyz_m=json.dumps(fidmap[int(ident)].tolist()),
            residual_mm=float(np.linalg.norm(transformed-fidmap[int(ident)])*1000)))
    pd.DataFrame(fidrows).to_csv(out/'09_fiducial_alignment_qa.csv',index=False)
    med,p95,maximum=np.percentile(distance*1000,[50,95,100]).tolist()
    numerical=med<=10 and p95<=20
    provenance=dict(source_frame='MNE template native unknown frame',source_units='metres after fixed MNE auto normalization',
        mapping_policy=args.mapping_policy,
        mapping_status='ASSUMED_UNVERIFIED' if args.mapping_policy=='documented-assumption' else 'VALIDATED_DOCUMENTARY_REVIEW',
        native_to_head=native_head['trans'].tolist(),head_to_fsaverage_mri=head_mri['trans'].tolist(),
        native_to_fsaverage_mri=(head_mri['trans']@native_head['trans']).tolist(),
        scalp_file_units='FreeSurfer surface RAS mm; divided by 1000 for comparisons',
        template_fiducial_origin='MNE electrode-derived estimates' if args.montage_kind=='EGI_256' else 'resource FidNz/FidT9/FidT10, documentary interpretation required',
        transform_policy='template fiducial coordinate definition + predefined fsaverage transform; no subject registration',
        individual_subject_geometry_used=False,optimized_parameters=[],projection_or_scaling_search=False,
        mne_template_scaling='head_size=auto; fixed 0.095m spherical convention; not a physical unit conversion',
        distance_metric='nearest scalp vertex (upper bound on triangle-surface distance)',
        median_scalp_distance_mm=med,p95_scalp_distance_mm=p95,max_scalp_distance_mm=maximum,
        gates=dict(median_mm=10.,p95_mm=20.,rigidity_tolerance=1e-5),
        fiducial_residuals_mm={r['fiducial']:r['residual_mm'] for r in fidrows},
        fiducial_acceptance='reported for independent review; no fitted landmark residual minimization',
        local_normal_offset_min_mm=float(normal_offset.min()),
        local_normal_offset_max_mm=float(normal_offset.max()),
        negative_local_normal_offsets=int((normal_offset<0).sum()),
        numerical_status='PASS_NUMERICAL_REVIEW_REQUIRED' if numerical else 'BLOCKED_GEOMETRY',
        surface_topology_signed_distance_not_verified=True)
    write_json(out/'07_coordinate_transform_provenance.json',provenance)
    # Same farthest-point policy as frozen Script09, now on template HEAD only.
    # Geometry source necessarily changes; original reference-specific lists are not reused.
    names=list(mapping.dataset_channel)
    ranks=sorted(range(256),key=lambda i:names[i])
    start=min(ranks,key=lambda i:(-head[i,2],names[i]))
    picked=[start]; minimum_sq=np.sum((head-head[start])**2,axis=1);minimum_sq[start]=-1.
    while len(picked)<128:
        candidate=min(ranks,key=lambda i:(-minimum_sq[i],names[i]))
        picked.append(candidate)
        minimum_sq=np.minimum(minimum_sq,np.sum((head-head[candidate])**2,axis=1));minimum_sq[picked]=-1.
    write_json(source/'01_montage/template_nominal_subsets.json',dict(
        mapping_policy=args.mapping_policy,
        mapping_status='ASSUMED_UNVERIFIED' if args.mapping_policy=='documented-assumption' else 'VALIDATED_DOCUMENTARY_REVIEW',
        policy='original greatest-Z anchor, lexicographic ties, nested farthest-point picks; template HEAD geometry',
        subsets={str(n):[names[i] for i in picked[:n]] for n in [32,64,128]},
        all_good='all EEG channels not marked bad; no waveform loading in Script01',
        bad_replacement='reserve healthy nominal channels; nearest unused good channel in template geometry; lexicographic ties',
        reference_policy_change='reference identities retained for parameter transfer; one template geometry for all identities'))
    return provenance


def triangle_surface_metrics(points, vertices, faces):
    """Exact unsigned point/triangle distances plus generalized winding numbers.

    The solid-angle formula also underlies MNE's surface inclusion check. We use
    its absolute winding number so a globally reversed but consistent surface
    orientation does not invert classifications. Sensor coordinates are read only.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    require(faces.ndim==2 and faces.shape[1]==3,'Expected triangular scalp mesh')
    require(faces.min()>=0 and faces.max()<len(vertices),'Scalp triangle indices out of range')
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    undirected,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    balance=np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    adjacency=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices)))
    components=int(connected_components(adjacency,directed=False,return_labels=False))
    a,b,c=(vertices[faces[:,i]] for i in range(3))
    ab,ac=b-a,c-a
    normal=np.cross(ab,ac); area_sq=np.einsum('ij,ij->i',normal,normal)
    require(np.all(area_sq>0),'Degenerate scalp triangles prevent distance calculation')
    topology=dict(vertices=len(vertices),triangles=len(faces),unique_edges=len(undirected),
        boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),
        inconsistent_oriented_edges=int((balance!=0).sum()),connected_components=components,
        euler_characteristic=int(len(vertices)-len(undirected)+len(faces)),
        self_intersections_checked=False)
    reliable=bool(np.all(counts==2) and np.all(balance==0) and components==1)
    topology['closed_oriented_connected_mesh']=reliable
    d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
    denominator=d00*d11-d01*d01
    require(np.all(denominator>0),'Ill-conditioned scalp triangles')
    distances=[];nearest=[];winding=[]
    for point in points:
        delta=point-a
        plane_dot=np.einsum('ij,ij->i',delta,normal)
        # Barycentric coordinates of the orthogonal plane projection. This
        # projection is only a distance calculation, never a sensor relocation.
        d20=np.einsum('ij,ij->i',delta,ab);d21=np.einsum('ij,ij->i',delta,ac)
        u=(d11*d20-d01*d21)/denominator;v=(d00*d21-d01*d20)/denominator
        in_triangle=(u>=0)&(v>=0)&(u+v<=1)
        squared=np.where(in_triangle,plane_dot**2/area_sq,np.inf)
        for start,end in [(a,b),(b,c),(c,a)]:
            edge=end-start
            fraction=np.clip(np.einsum('ij,ij->i',point-start,edge)/np.einsum('ij,ij->i',edge,edge),0,1)
            residual=point-(start+fraction[:,None]*edge)
            squared=np.minimum(squared,np.einsum('ij,ij->i',residual,residual))
        index=int(squared.argmin());nearest.append(index);distances.append(float(np.sqrt(squared[index])))
        va,vb,vc=a-point,b-point,c-point
        na,nb,nc=(np.linalg.norm(vv,axis=1) for vv in [va,vb,vc])
        triple=np.einsum('ij,ij->i',va,np.cross(vb,vc))
        divisor=na*nb*nc+np.einsum('ij,ij->i',va,vb)*nc+np.einsum('ij,ij->i',vb,vc)*na+np.einsum('ij,ij->i',vc,va)*nb
        winding.append(float(np.sum(2*np.arctan2(triple,divisor))/(4*np.pi)))
    return np.array(distances),np.array(nearest),np.array(winding),topology


def review_existing_geometry(preparation, output):
    """Review an existing preparation into fresh files; no preparation rerun."""
    import mne
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    preparation=preparation.expanduser().resolve();output=output.expanduser().resolve()
    require(VARIANT in preparation.parents and VARIANT in output.parents,'Review paths must stay inside this variant')
    require(not output.exists(),'Geometry review requires a fresh --output-root')
    require(not (preparation in output.parents or output in preparation.parents),'Review must be separate from preparation outputs')
    files=[preparation/name for name in ['01_experiment_manifest.json','06_sensor_scalp_geometry_qa.csv',
        '07_coordinate_transform_provenance.json','08_readiness_report.json','09_fiducial_alignment_qa.csv']]
    input_hashes={str(path):sha(path) for path in files}
    manifest=json.loads(files[0].read_text());provenance=json.loads(files[2].read_text())
    readiness=json.loads(files[3].read_text())
    require(readiness['geometry_evaluated'] is True,'Preparation has no completed template geometry')
    for item in manifest['source_inventory']:
        require(sha(Path(item['copy_path']))==item['sha256'],f'Snapshot changed: {item["copy_path"]}')
    matches=[item for item in manifest['source_inventory'] if item['role']=='fsaverage_scalp']
    require(len(matches)==1,'Expected exactly one frozen fsaverage scalp asset')
    scalp=Path(matches[0]['copy_path'])
    input_hashes[str(scalp)]=sha(scalp)
    table=pd.read_csv(files[1])
    require(len(table)==256 and set(table.dataset_channel)==set(CHANNELS),'Expected 256 mapped EEG sensors')
    require(not table.dataset_channel.duplicated().any() and not table.template_channel.duplicated().any(),'Ambiguous sensor identities')
    points=table[[f'fsaverage_surface_ras_{axis}_m' for axis in 'xyz']].to_numpy(float)
    original_points=points.copy()
    native=table[[f'template_native_{axis}_m' for axis in 'xyz']].to_numpy(float)
    require(np.isfinite(points).all(),'Nonfinite template sensor positions')
    matrix=np.asarray(provenance['native_to_fsaverage_mri'],float)
    require(np.allclose(native@matrix[:3,:3].T+matrix[:3,3],points,atol=1e-10,rtol=0),'Stored coordinates/transform disagree')
    vertices_mm,faces=mne.read_surface(scalp,verbose=False)
    vertices=vertices_mm/1000.
    require(np.isfinite(vertices).all(),'Nonfinite scalp vertices')
    distance,nearest,winding,topology=triangle_surface_metrics(points,vertices,faces)
    require(np.isfinite(distance).all() and np.isfinite(winding).all(),'Nonfinite triangle distance or winding number')
    # 0.001 mm is a numerical surface-classification tolerance, not a change to
    # preparation acceptance gates. Non-integer windings stay ambiguous.
    on_surface=distance<=1e-6
    relation=np.full(len(points),'AMBIGUOUS',dtype=object)
    if topology['closed_oriented_connected_mesh']:
        relation[np.abs(winding)<1e-5]='OUTSIDE'
        relation[np.abs(np.abs(winding)-1)<1e-5]='INSIDE'
    relation[on_surface]='ON_SURFACE'
    signed=np.where(relation=='INSIDE',-distance,np.where(relation=='OUTSIDE',distance,np.where(relation=='ON_SURFACE',0,np.nan)))*1000
    vertex_distance=table.nearest_scalp_vertex_distance_mm.to_numpy(float)
    require(np.all(distance*1000<=vertex_distance+1e-6),'Triangle distance exceeds nearest-vertex bound')
    table['nearest_scalp_triangle']=nearest
    table['scalp_triangle_distance_mm']=distance*1000
    table['generalized_winding_number']=winding
    table['surface_relation']=relation
    table['signed_scalp_triangle_distance_mm']=signed
    table['signed_distance_convention']='negative inside, positive outside; NaN ambiguous; original electrodes unchanged'
    counts={name:int((relation==name).sum()) for name in ['INSIDE','OUTSIDE','ON_SURFACE','AMBIGUOUS']}
    concerns=[]
    if not topology['closed_oriented_connected_mesh']:
        concerns.append('Scalp mesh is not closed, consistently oriented and connected; interior classification is unreliable.')
    if counts['AMBIGUOUS']:
        concerns.append('Some electrodes have ambiguous interior classification.')
    if counts['INSIDE']:
        concerns.append('Template electrodes lie inside the closed scalp mesh; review signed depths before forward modeling.')
    if counts['OUTSIDE']:
        concerns.append('Some electrodes lie outside the scalp; review separation from the modeled boundary.')
    reasons=concerns or ['No placement concern was detected numerically; scientific review is still required.']
    mapping_status=sorted(table.validation_status.unique().tolist())
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),review_script_sha256=sha(Path(__file__)),
        status='PLACEMENT_CONCERN_REVIEW_REQUIRED' if concerns else 'NUMERICAL_PLACEMENT_REVIEW_REQUIRED',
        mapping_policy=manifest.get('mapping_policy',readiness.get('mapping_policy','verified')),mapping_status=mapping_status,
        physical_channel_mapping_verified=readiness['independent_channel_mapping_validated'],
        concerns=reasons,counts=counts,mesh=topology,
        classification_tolerances=dict(on_surface_mm=.001,winding_number=1e-5),
        unsigned_triangle_distance_mm=dict(median=float(np.median(distance)*1000),
            p95=float(np.percentile(distance,95)*1000),maximum=float(distance.max()*1000)),
        maximum_inside_depth_mm=float(distance[relation=='INSIDE'].max()*1000) if counts['INSIDE'] else None,
        maximum_outside_gap_mm=float(distance[relation=='OUTSIDE'].max()*1000) if counts['OUTSIDE'] else None,
        previous_nearest_vertex_QA={k:provenance[k] for k in ['median_scalp_distance_mm','p95_scalp_distance_mm','max_scalp_distance_mm','gates']},
        fiducial_residuals_mm=provenance['fiducial_residuals_mm'],input_sha256=input_hashes,
        software_versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','mne','matplotlib']},
        plotting_units='fsaverage surface RAS millimetres',electrode_coordinates_modified=False,
        sensors_projected_to_scalp=False,individual_geometry_used=False,forward_modeling_executed=False,
        forward_modeling_authorized=False,preparation_status_modified=False,
        limitations=['Inside/outside assumes the supplied closed mesh has no self-intersections; self-intersections were not tested.',
            'This is an independent placement diagnostic, not a new or relaxed preparation threshold.',
            'Template anatomical assumptions and channel correspondence uncertainty remain.'])
    # Plot geometry in mm. Colors use the 3-D classification, not view overlap.
    colors={'INSIDE':'#D1495B','OUTSIDE':'#0077B6','ON_SURFACE':'#239B56','AMBIGUOUS':'#D99A00'}
    point_colors=[colors[r] for r in relation]
    positions=points*1000
    fig=plt.figure(figsize=(15,13))
    for i,(title,elevation,azimuth) in enumerate([('Anterior',0,90),('Left',0,180),('Superior',90,90),('Oblique',25,45)],1):
        ax=fig.add_subplot(2,2,i,projection='3d')
        ax.plot_trisurf(vertices_mm[:,0],vertices_mm[:,1],vertices_mm[:,2],triangles=faces,
                        color='#BFC9D4',alpha=.17,linewidth=0,rasterized=True)
        ax.scatter(*positions.T,c=point_colors,s=15,edgecolor='#202020',linewidth=.2,depthshade=False)
        ax.set_box_aspect(np.ptp(vertices_mm,axis=0));ax.view_init(elev=elevation,azim=azimuth)
        ax.set(xlabel='MRI RAS X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=title)
    legend=[Line2D([],[],marker='o',linestyle='',color=colors[name],label=f'{name}: {counts[name]}') for name in counts]
    fig.legend(handles=legend,loc='lower center',ncol=4,bbox_to_anchor=(.5,.025))
    fig.suptitle('Fixed HydroCel template and fsaverage scalp\nInside/outside from closed-mesh winding numbers; original coordinates retained',fontsize=15)
    fig.subplots_adjust(left=.02,right=.98,bottom=.10,top=.91,wspace=.05,hspace=.12)
    output.mkdir(parents=True,exist_ok=False)
    fig.savefig(output/'01_scalp_overlay_views.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    finite_signed=signed[np.isfinite(signed)]
    extent=max(25.,float(np.ceil(np.max(np.abs(finite_signed))/5)*5)) if len(finite_signed) else 25.
    for name in ['INSIDE','OUTSIDE','ON_SURFACE']:
        values=signed[relation==name]
        if len(values):axes[0].hist(values,bins=np.linspace(-extent,extent,41),alpha=.65,label=name,color=colors[name])
    axes[0].axvline(0,color='black',lw=1)
    axes[0].set(xlabel='Signed triangle distance (mm; negative inside)',ylabel='Electrodes',title='Placement relative to the scalp boundary')
    axes[0].legend()
    worst=np.argsort(distance)[-12:]
    axes[1].barh(np.arange(len(worst)),distance[worst]*1000,color=[point_colors[i] for i in worst])
    axes[1].set_yticks(np.arange(len(worst)),table.dataset_channel.iloc[worst])
    axes[1].set(xlabel='Unsigned triangle distance (mm)',title='Twelve largest boundary separations')
    fig.suptitle('Template-only geometry review: assumed channel mapping, no sensor fitting',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(output/'02_signed_scalp_distance.png',dpi=300,bbox_inches='tight');plt.close(fig)
    require(np.array_equal(points,original_points),'Review unexpectedly modified electrode coordinates')
    for path,expected in input_hashes.items():
        require(sha(Path(path))==expected,f'Review input changed: {path}')
    table.to_csv(output/'03_sensor_triangle_distance.csv',index=False)
    write_json(output/'04_geometry_review.json',report)
    (output/'README.md').write_text('# Template-only scalp placement review\n\n'
        +f'Status: {report["status"]}. Classification counts: {counts}.\n\n'
        +'01_scalp_overlay_views.png shows four views with the fixed electrodes and transparent scalp.\n'
        +'02_signed_scalp_distance.png shows signed distances and largest separations.\n'
        +'03_sensor_triangle_distance.csv preserves original coordinates and adds triangle distances/winding classes.\n'
        +'04_geometry_review.json records topology, numerical tolerances, hashes and interpretation limits.\n\n'
        +'Negative signed distance means inside, positive outside; ambiguous classifications retain missing signed distances.\n'
        +'Scalp topology was inspected, but self-intersections were not tested.\n'
        +'Original preparation thresholds and readiness files are unchanged. No sensors were relocated, projected or optimized.\n'
        +'No participant anatomy/digitization or ground-truth locations were read. No forward model was run or authorized.\n')
    print(f'{report["status"]}: {output}')
    print(f'Inside/outside classification: {counts}')
    return 0


def audit_bem_interpolation(preparation, output):
    """Audit standard MNE BEM electrode interpolation without a forward matrix.

    Simulates only the nearest-surface interpolation query for one unit-weight
    EEG location per channel. It does not call _bem_specify_els, construct a
    conductor model, calculate a leadfield, or alter input sensor locations.
    """
    import mne
    from mne.io.constants import FIFF
    from mne.surface import _project_onto_surface
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    require(importlib.metadata.version('mne')=='1.12.1',
            'Private interpolation API was reviewed for MNE 1.12.1 only; review source before using another version')
    preparation=preparation.expanduser().resolve();output=output.expanduser().resolve()
    require(VARIANT in preparation.parents and VARIANT in output.parents,'Audit paths must stay inside this variant')
    require(not output.exists(),'BEM interpolation audit requires a fresh --output-root')
    require(not (preparation in output.parents or output in preparation.parents),'Audit must be separate from preparation outputs')
    paths=[preparation/n for n in ['01_experiment_manifest.json','06_sensor_scalp_geometry_qa.csv',
        '07_coordinate_transform_provenance.json','08_readiness_report.json']]
    input_hashes={str(p):sha(p) for p in paths}
    manifest=json.loads(paths[0].read_text());transform=json.loads(paths[2].read_text())
    readiness=json.loads(paths[3].read_text())
    require(readiness['geometry_evaluated'] is True,'Preparation has no completed template geometry')
    for item in manifest['source_inventory']:
        require(sha(Path(item['copy_path']))==item['sha256'],f'Frozen snapshot changed: {item["copy_path"]}')
    resource=[r for r in manifest['fsaverage_resources'] if r['role']=='bem_solution']
    require(len(resource)==1 and resource[0]['present'],'Preparation did not identify a local BEM solution')
    bem_path=Path(resource[0]['path']).resolve()
    require(sha(bem_path)==resource[0]['sha256'],'BEM solution differs from preparation manifest')
    input_hashes[str(bem_path)]=resource[0]['sha256']
    package=Path(mne.__file__).parent
    for relative in ['surface.py','forward/_compute_forward.py','forward/_make_forward.py','_fiff/tag.py']:
        p=package/relative;input_hashes[str(p)]=sha(p)
    table=pd.read_csv(paths[1])
    require(len(table)==256 and set(table.dataset_channel)==set(CHANNELS),'Expected 256 template EEG channels')
    require(not table.dataset_channel.duplicated().any() and not table.template_channel.duplicated().any(),'Ambiguous channel identities')
    original=table[[f'fsaverage_surface_ras_{axis}_m' for axis in 'xyz']].to_numpy(float)
    head=table[[f'template_head_{axis}_m' for axis in 'xyz']].to_numpy(float)
    head_to_mri=np.asarray(transform['head_to_fsaverage_mri'],float)
    require(np.isfinite(original).all() and np.isfinite(head).all(),'Invalid template coordinates')
    require(np.allclose(head@head_to_mri[:3,:3].T+head_to_mri[:3,3],original,rtol=0,atol=1e-10),'HEAD/MRI sensor coordinates disagree')
    unchanged=original.copy()
    # Read the existing BEM solution. No make_bem_model/make_bem_solution call.
    bem=mne.read_bem_solution(bem_path,verbose=False)
    require(not bem['is_sphere'] and len(bem['surfs'])==3,'Expected the original three-surface fsaverage EEG BEM')
    surface=bem['surfs'][0]
    require(int(surface['id'])==int(FIFF.FIFFV_BEM_SURF_ID_HEAD),'First BEM surface is not the scalp used by _bem_specify_els')
    require(int(surface['coord_frame'])==int(FIFF.FIFFV_COORD_MRI),'BEM surface must be MRI/surface RAS')
    vertices=np.asarray(surface['rr'],float);faces=np.asarray(surface['tris'],int)
    # Use the SAME function/default method used by installed _bem_specify_els.
    # It returns weights/triangles, not a forward solution or relocated Info.
    weights,triangles=_project_onto_surface(original.copy(),surface)
    require(weights.shape==(256,3) and triangles.shape==(256,),'Unexpected interpolation return shape')
    require(np.isfinite(weights).all() and np.allclose(weights.sum(axis=1),1,atol=1e-7,rtol=0),'Invalid barycentric weights')
    require(np.all((triangles>=0)&(triangles<len(faces))),'Invalid interpolation triangle indices')
    extrapolated=np.any((weights < -1e-7)|(weights > 1+1e-7),axis=1)
    effective=np.einsum('ij,ijk->ik',weights,vertices[faces[triangles]])
    # MNE reproject=True recomputes unconstrained plane barycentric weights
    # after selecting the closest triangle. Negative weights are extrapolation,
    # not automatically an invalid solver result. Never clip/renormalize them.
    selected=vertices[faces[triangles]]
    normals=np.cross(selected[:,1]-selected[:,0],selected[:,2]-selected[:,0])
    norm_sq=np.einsum('ij,ij->i',normals,normals)
    require(np.all(norm_sq>0),'Degenerate selected BEM triangles')
    plane_projection=original-(np.einsum('ij,ij->i',original-selected[:,0],normals)/norm_sq)[:,None]*normals
    plane_error=np.linalg.norm(effective-plane_projection,axis=1)*1000
    require(np.isfinite(effective).all() and plane_error.max()<=.001,'MNE effective locations disagree with independent triangle-plane projections by more than 0.001 mm')
    displacement=effective-original
    shifts=np.linalg.norm(displacement,axis=1)*1000
    true_distance,_,winding,topology=triangle_surface_metrics(original,vertices,faces)
    require(topology['closed_oriented_connected_mesh'],'Actual BEM scalp is not a closed, consistently oriented connected mesh')
    require(np.isfinite(true_distance).all() and np.isfinite(winding).all(),'Invalid BEM distance/inclusion calculation')
    discrepancy=np.abs(shifts-true_distance*1000)
    relation=np.full(256,'AMBIGUOUS',dtype=object)
    relation[np.abs(winding)<1e-5]='OUTSIDE'
    relation[np.abs(np.abs(winding)-1)<1e-5]='INSIDE'
    relation[true_distance<=1e-6]='ON_SURFACE'
    counts={k:int((relation==k).sum()) for k in ['INSIDE','OUTSIDE','ON_SURFACE','AMBIGUOUS']}
    # Plane extrapolation can leave the finite triangle and scalp surface.
    # Preserve this discrepancy as a finding, not a reason to alter coordinates.
    effective_distance,_,_,_=triangle_surface_metrics(effective,vertices,faces)
    require(np.isfinite(effective_distance).all(),'Nonfinite effective-location surface distances')
    audit=table.copy()
    audit['actual_bem_surface_relation']=relation
    audit['actual_bem_winding_number']=winding
    audit['actual_bem_nearest_triangle_distance_mm']=true_distance*1000
    audit['effective_bem_triangle']=triangles
    for i,axis in enumerate('xyz'):
        audit[f'effective_bem_interpolation_{axis}_m']=effective[:,i]
        audit[f'interpolation_displacement_{axis}_mm']=displacement[:,i]*1000
    for i in range(3):
        audit[f'bem_triangle_vertex_{i}']=faces[triangles,i]
        audit[f'bem_interpolation_weight_{i}']=weights[:,i]
    audit['interpolation_displacement_mm']=shifts
    audit['effective_point_surface_residual_mm']=effective_distance*1000
    audit['solver_extrapolates_beyond_selected_triangle']=extrapolated
    audit['independent_triangle_plane_projection_error_mm']=plane_error
    audit['displacement_vs_nearest_surface_distance_difference_mm']=shifts-true_distance*1000
    audit['effective_coordinates_role']='audit-only BEM interpolation query; not replacement input coordinates'
    raw_mm=original*1000;effective_mm=effective*1000;mesh_mm=vertices*1000
    fig=plt.figure(figsize=(16,8))
    for i,(elevation,azimuth,title) in enumerate([(15,65,'Anterior oblique'),(15,165,'Lateral oblique')],1):
        ax=fig.add_subplot(1,2,i,projection='3d')
        ax.plot_trisurf(*mesh_mm.T,triangles=faces,color='#BFC9D4',alpha=.15,linewidth=0,rasterized=True)
        for before,after in zip(raw_mm,effective_mm):
            ax.plot(*np.vstack([before,after]).T,color='#536878',alpha=.45,lw=.5)
        ax.scatter(*raw_mm.T,s=12,c='#D1495B',depthshade=False)
        ax.scatter(*effective_mm.T,s=12,c='#0077B6',depthshade=False)
        ax.set(xlabel='MRI RAS X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=title)
        ax.set_box_aspect(np.ptp(mesh_mm,axis=0));ax.view_init(elevation,azimuth)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(4))
    fig.suptitle('Actual BEM scalp: fixed template versus effective EEG interpolation\nLines show audit displacements; input coordinates are unchanged',fontsize=14)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#D1495B',label='Original template'),
                        Line2D([],[],marker='o',ls='',color='#0077B6',label='Effective interpolation')],
               loc='lower center',ncol=2)
    fig.subplots_adjust(top=.86,bottom=.12,wspace=.05)
    output.mkdir(parents=True,exist_ok=False)
    fig.savefig(output/'01_bem_interpolation_overlay.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    axes[0].hist(shifts,bins=25,color='#4E79A7');axes[0].set(xlabel='Displacement to actual BEM scalp (mm)',ylabel='Electrodes',title='Solver interpolation displacement')
    order=np.argsort(shifts)[-12:]
    axes[1].barh(np.arange(12),shifts[order],color='#D1495B')
    axes[1].set_yticks(np.arange(12),audit.dataset_channel.iloc[order])
    axes[1].set(xlabel='Displacement (mm)',title='Twelve largest interpolation displacements')
    fig.suptitle('Interpolation audit only — no forward model or coordinate adjustment',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(output/'02_bem_interpolation_displacements.png',dpi=300,bbox_inches='tight');plt.close(fig)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='BEM_INTERPOLATION_AUDIT_REVIEW_REQUIRED',
        audit_script_sha256=sha(Path(__file__)),input_sha256=input_hashes,
        mapping_policy=manifest['mapping_policy'],mapping_status=sorted(table.validation_status.unique().tolist()),
        actual_bem_surface=dict(id=int(surface['id']),coordinate_frame='MRI surface RAS',units='metres',mesh=topology),
        original_sensor_relation_to_actual_bem=counts,
        displacement_mm=dict(median=float(np.median(shifts)),p95=float(np.percentile(shifts,95)),maximum=float(shifts.max())),
        maximum_discrepancy_from_independent_triangle_distance_mm=float(discrepancy.max()),
        discrepancy_from_nearest_triangle_is_acceptance_gate=False,
        extrapolated_electrode_count=int(extrapolated.sum()),
        interpolation_weight_minimum=float(weights.min()),interpolation_weight_maximum=float(weights.max()),
        maximum_independent_triangle_plane_projection_error_mm=float(plane_error.max()),
        maximum_effective_point_surface_residual_mm=float(effective_distance.max()*1000),
        effective_points_off_surface_count=int((effective_distance*1000>.001).sum()),
        numerical_comparison_tolerance_mm=.001,
        interpolation_implementation='MNE 1.12.1 _project_onto_surface accurate: selects closest triangle then reproject=True recomputes unconstrained triangle-plane weights; same query used by _bem_specify_els',
        integration_point_assumption='One unit-weight monopolar EEG location per channel; future Info must contain no separate reference-electrode location in loc[3:6]. Average reference is applied separately.',
        solver_interpolation_projection_simulated=True,input_electrode_coordinates_modified=False,
        effective_coordinates_are_audit_only=True,original_QA_thresholds_changed=False,preparation_files_changed=False,
        source_selection_changed=False,individual_geometry_used=False,forward_model_computed=False,
        forward_modeling_authorized=False,
        software_versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','mne','matplotlib']},
        limitations=['This audits the interpolation-location step, not the complete forward calculation.',
            'Original template placement concerns and unverified channel correspondence remain.',
            'BEM topology was inspected but self-intersections were not tested.',
            'Negative/out-of-range weights represent plane extrapolation; effective locations need not lie on the finite triangle or actual scalp surface. No clipping was applied.',
            'Projection/interpolation is documented solver behavior, not a coordinate correction to obtain a geometry PASS.',
            'Future forward construction requires an explicit modeling-policy decision about this interpolation behavior.'])
    require(np.array_equal(original,unchanged),'Audit unexpectedly modified original coordinates')
    for path,expected in input_hashes.items():
        require(sha(Path(path))==expected,f'Audit input changed: {path}')
    audit.to_csv(output/'03_effective_bem_sensor_locations.csv',index=False)
    write_json(output/'04_bem_interpolation_audit.json',report)
    (output/'README.md').write_text('# Actual-BEM sensor interpolation audit\n\n'
        '01_bem_interpolation_overlay.png: original and effective locations on the actual BEM scalp.\n'
        '02_bem_interpolation_displacements.png: displacement distribution and largest channels.\n'
        '03_effective_bem_sensor_locations.csv: original coordinates, effective locations, triangles, barycentric weights and displacements.\n'
        '04_bem_interpolation_audit.json: input/code hashes, mesh checks, statistics and solver assumptions.\n\n'
        'Effective coordinates are derived audit data, not replacement input sensor coordinates.\n'
        'MNE recomputes unconstrained triangle-plane weights; negative weights and off-surface effective points are reported, not clipped.\n'
        'The standard solver interpolates scalp potentials at these locations for monopolar EEG definitions.\n'
        'No BEM construction, forward matrix, inverse solution or individual registration was run.\n'
        'Original placement QA and unverified identity-number assumptions are retained. No automatic forward authorization is given.\n')
    print(f'{report["status"]}: {output}')
    print(f'Interpolation displacements (mm): {report["displacement_mm"]}')
    print(f'Extrapolated electrodes: {report["extrapolated_electrode_count"]}; effective points off scalp: {report["effective_points_off_surface_count"]}')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-bem-preparation-root',type=Path,help='Audit effective sensor interpolation on the actual BEM without a forward calculation')
    parser.add_argument('--review-preparation-root',type=Path,help='Review existing template geometry only; use a fresh --output-root')
    parser.add_argument('--frozen-design-root',type=Path,default=FS/'outputs/01_prepare_experiment/source_configurations')
    parser.add_argument('--resolved-configurations',type=Path,default=FS/'outputs/05_reconstruct_sub03_exploratory/05_resolved_configurations.json')
    parser.add_argument('--montage-kind',choices=['GSN-HydroCel-256','EGI_256'],default='GSN-HydroCel-256')
    parser.add_argument('--mapping-evidence',type=Path)
    parser.add_argument('--mapping-policy',choices=['verified','documented-assumption'],default='verified',
        help='Default requires documentary correspondence; opt-in assumption mode is explicitly unverified')
    parser.add_argument('--assumption-record',type=Path)
    parser.add_argument('--fsaverage-root',type=Path,default=Path.home()/'mne_data/MNE-fsaverage-data/fsaverage')
    parser.add_argument('--source-root',type=Path,default=VARIANT/'source')
    parser.add_argument('--output-root',type=Path,default=VARIANT/'outputs/01_prepare_experiment')
    args=parser.parse_args(argv)
    require(not (args.review_preparation_root is not None and args.audit_bem_preparation_root is not None), 'Choose geometry review or BEM interpolation audit, not both')
    if args.audit_bem_preparation_root is not None:
        return audit_bem_interpolation(args.audit_bem_preparation_root,args.output_root)
    if args.review_preparation_root is not None:
        return review_existing_geometry(args.review_preparation_root,args.output_root)
    args.frozen_design_root=args.frozen_design_root.expanduser().resolve()
    args.resolved_configurations=args.resolved_configurations.expanduser().resolve()
    if args.mapping_evidence is not None:
        args.mapping_evidence=args.mapping_evidence.expanduser().resolve()
    if args.assumption_record is not None:
        args.assumption_record=args.assumption_record.expanduser().resolve()
    require(not (args.mapping_policy=='documented-assumption' and args.mapping_evidence is not None), 'Choose documentary evidence or an explicit assumption, not both')
    require(not (args.mapping_policy=='verified' and args.assumption_record is not None), '--assumption-record requires --mapping-policy documented-assumption')
    source,out=args.source_root.expanduser().resolve(),args.output_root.expanduser().resolve()
    for path in [source,out]:
        require(VARIANT in path.parents,'All writable paths must be strictly inside this variant')
    require(not (source==out or source in out.parents or out in source.parents),'Source/output roots overlap')
    require(not out.exists() or not any(p.name!='.gitkeep' for p in out.iterdir()),'Output directory is not empty; choose a new --output-root')
    # Committed README placeholders are allowed; runtime snapshot reuse is not.
    require(not source.exists() or not any(p.is_file() and p.name not in {'README.md','.gitkeep'} for p in source.rglob('*')), 'Source snapshots already exist; choose a new --source-root')
    source.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    inventory=[];reasons=[];schedule=None;geometry=None;catalog=pd.DataFrame();mapping=pd.DataFrame()
    fsaverage_resources=[]
    for role,name in [('source_space','fsaverage-ico-5-src.fif'),
                      ('bem_solution','fsaverage-5120-5120-5120-bem-sol.fif'),
                      ('scalp','outer_skin.surf'),('fiducials','fsaverage-fiducials.fif'),
                      ('predefined_transform','fsaverage-trans.fif')]:
        path=args.fsaverage_root.expanduser().resolve()/'bem'/name
        present=path.is_file()
        fsaverage_resources.append(dict(role=role,path=str(path),present=present,
            sha256=sha(path) if present else None))
        if not present:
            reasons.append(f'Missing local fsaverage resource (no download attempted): {path}')
    try:
        schedule=freeze_design(args,source,inventory)
        schedule.to_csv(out/'02_planned_reconstruction_schedule.csv',index=False)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        reasons.append(f'Frozen design blocked: {type(exc).__name__}: {exc}')
    try:
        catalog,montages,paths=montage_catalog(source,inventory)
        catalog.to_csv(out/'04_template_montage_availability.csv',index=False)
        require(args.montage_kind in montages and args.montage_kind in paths,'Selected candidate resource unavailable')
        mapping,mapping_reasons=validate_mapping(args,montages[args.montage_kind],paths[args.montage_kind],source,inventory)
        reasons.extend(mapping_reasons)
        mapping.to_csv(out/'05_channel_mapping_validation.csv',index=False)
        if not mapping_reasons:
            geometry=template_geometry(args,montages[args.montage_kind],mapping,source,inventory,out)
            if geometry['numerical_status']!='PASS_NUMERICAL_REVIEW_REQUIRED':
                reasons.append('Fixed template exceeds unchanged scalp-distance numerical QA gates; do not optimize it to pass.')
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        reasons.append(f'Template validation blocked: {type(exc).__name__}: {exc}')
    if mapping.empty:
        pd.DataFrame({'dataset_channel':CHANNELS,'template_channel':'','validation_status':'BLOCKED_TEMPLATE_UNAVAILABLE'}).to_csv(out/'05_channel_mapping_validation.csv',index=False)
    if catalog.empty:
        pd.DataFrame([dict(montage=args.montage_kind,available=False,reason='See readiness report')]).to_csv(out/'04_template_montage_availability.csv',index=False)
    pd.DataFrame(inventory,columns=['role','original_path','copy_path','sha256','size_bytes']).to_csv(out/'03_frozen_configuration_inventory.csv',index=False)
    ready=not reasons
    assumed=args.mapping_policy=='documented-assumption'
    readiness=dict(status=('EXPLORATORY_NUMERICAL_REVIEW_REQUIRED' if assumed else 'PASS_NUMERICAL_REVIEW_REQUIRED') if ready else 'BLOCKED',blockers=reasons,
        mapping_policy=args.mapping_policy,
        assumption_record_accepted=bool(not mapping.empty and mapping.validation_status.eq('ASSUMED_UNVERIFIED').all()),
        uncertainties=['Exact acquisition/export row-to-physical-sensor correspondence is assumed, not verified.',
                       'Fixed template normalization and fiducial placement remain modeling assumptions.'] if assumed else [],
        independent_channel_mapping_validated=bool(not mapping.empty and mapping.validation_status.eq('VALIDATED_DOCUMENTARY_REVIEW').all()),
        geometry_evaluated=geometry is not None,scientific_anatomical_validity_established=False,
        forward_modeling_authorized=False,
        next_step='Review Script01 evidence and geometry before separately implementing Script02; no forward work is executed here.')
    write_json(out/'08_readiness_report.json',readiness)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        dataset='Localize-MI',counts=dict(subjects=7,runs=61,methods=12,loso=732,single_reference=4392,total=5124),
        design_schedule_written=schedule is not None,selected_template_candidate=args.montage_kind,
        mapping_policy=args.mapping_policy,
        mapping_status=sorted(mapping.validation_status.unique().tolist()) if not mapping.empty else ['BLOCKED_TEMPLATE_UNAVAILABLE'],
        assumption_record=json.loads(args.assumption_record.read_text()) if readiness['assumption_record_accepted'] else None,
        source_inventory=inventory,readiness=readiness,
        fsaverage_resources=fsaverage_resources,
        EEG_channel_inventory=dict(expected_channels=CHANNELS,actual_recording_headers_inspected=False,
            later_requirement='Validate EEG header channel identities without loading digitization or anatomy before modeling.'),
        software_versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','mne']},
        reconstruction_constraints=dict(individual_mri=False,individual_digitization=False,test_sensor_coordinates=False,
            individual_icp=False,affine_or_syn=False,ground_truth_for_montage_selection=False),
        preprocessing_policy=dict(windows=WINDOWS,average_reference='all loaded good channels before subset selection',
            noise_covariance='original auto method',filters_baseline_epochs='preserve frozen original loader behavior; no processing in Script01',
            peak_selection='original known-hemisphere cortical peak; max-GOF Continuous-ECD/RAP-MUSIC',
            ground_truth_caveat='known stimulation hemisphere is an explicit extra prior; any later blinded/whole-brain variant must be separately labeled',
            future_loader_adapter='Do not invoke original load_run/Engine.prepare unchanged: they read individual forwards/anatomy. Implement EEG-only adapter in a later reviewed stage.'),
        channel_policy='frozen montage sizes and algorithm; geometry source becomes common template, not reference/test digitization',
        compatibility_limits=['Both anatomy and sensor geometry differ from old no-warp; not an anatomy-only contrast.',
            'No nominal subset is scientifically usable until channel correspondence and template geometry are reviewed.',
            'Transferred parameters are frozen; their optimality under changed geometry is not guaranteed.',
            'Unregistered coordinate separation is diagnostic, not validated anatomical localization error.',
            'Evidence review is documentary human attestation; SHA-256 establishes integrity, not truth of its claims.',
            'No solver compatibility shim is executed or implied; future SESAME capacity handling needs explicit provenance.'],
        expensive_operations_executed=False)
    write_json(out/'01_experiment_manifest.json',manifest)
    (out/'README.md').write_text('# Template montage preparation\n\n'
        'Read 08_readiness_report.json first. BLOCKED is an expected successful audit outcome when evidence is missing.\n\n'
        '01_experiment_manifest.json: assumptions, counts, policies, hashes and limitations.\n'
        '02_planned_reconstruction_schedule.csv: 5,124 identities/configurations, when frozen design validates.\n'
        '03_frozen_configuration_inventory.csv: original/snapshot paths, sizes and SHA-256.\n'
        '04_template_montage_availability.csv: installed candidate counts/frames/fiducials, not a numbering proof.\n'
        '05_channel_mapping_validation.csv: explicit documentary validation or blocked channels.\n'
        '06_sensor_scalp_geometry_qa.csv, 07_coordinate_transform_provenance.json and '
        '09_fiducial_alignment_qa.csv: only attempted after documentary mapping acceptance; inspect for geometry failures.\n'
        'source/ snapshots preserve original files. Geometry is never projected, fitted to a participant, or selected using outcomes.\n'
        'Read mapping_policy and validation_status: assumption mode is ASSUMED_UNVERIFIED, never confirmed physical correspondence.\n'
        'Numerical geometry still requires scientific review; no Script02 is implemented or executed.\n')
    print(f'{readiness["status"]}: {out}')
    for reason in reasons:print(f'BLOCKED: {reason}')
    return 0 if ready else 3


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)
