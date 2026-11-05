#!/usr/bin/env python3
# Script 05 — Reconstruct test EEG with fsaverage anatomy.
# Purpose: Reuse no-warp Engine.prepare/solve and its 12 method implementations;
#   replace only the reconstruction forward/BEM, preserving reference montage policy.
# Inputs: Script 04 forwards, manifest and frozen schedules/configurations; native
#   reference forwards; frozen no-warp solver code/grid snapshots; Localize-MI EEG.
# Outputs: A new directory containing full 5,124-row status/plan ledgers, resolved
#   configurations, failure traces, channel audit, runtime and checksum manifest.
# Parameters: --phase plan (default) or reconstruct; --strategy/--method/
#   --test-subject/--reference-subject/--run filters; --limit 0 (all); --resume;
#   --quiet. Same original preprocessing, known-hemisphere peak and solver settings.
# Dependencies: Scripts 01, 02 and 04; existing no-warp Engine and frozen helpers.
#   No Script 03 MRI registration, coordinate mapping or distance calculation.
# Complete examples (repository root):
#   .venv/bin/python cross_subject_fsaverage/scripts/05_reconstruct.py --phase plan
#   .venv/bin/python cross_subject_fsaverage/scripts/05_reconstruct.py \
#       --phase reconstruct --resume --strategy LOSO --method MNE \
#       --test-subject sub-01 --run run-01 --limit 1 --quiet
#   .venv/bin/python cross_subject_fsaverage/scripts/05_reconstruct.py \
#       --phase reconstruct --resume --quiet
# --resume preserves every PASS/FAIL/BLOCKED record; only PENDING jobs are attempted.
"""Reuse the no-warp inverse engine with test-specific fsaverage forwards."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'cross_subject_fsaverage'
NW = ROOT / 'cross_subject_no_warp'
KEY = ['strategy', 'method', 'reference_subject', 'test_subject', 'run']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['plan', 'reconstruct'], default='plan')
    p.add_argument('--forward-root', type=Path, default=EXP / 'outputs/04_make_forward')
    p.add_argument('--output-root', type=Path, default=EXP / 'outputs/05_reconstruct')
    p.add_argument('--dataset', type=Path, default=Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    p.add_argument('--strategy', choices=['LOSO', 'Single-reference'], action='append')
    p.add_argument('--method', action='append')
    p.add_argument('--test-subject', action='append')
    p.add_argument('--reference-subject', action='append')
    p.add_argument('--run', action='append')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)
    if args.limit < 0:
        p.error('--limit must be nonnegative')
    # Load unchanged no-warp module without importing MNE or running an inverse.
    import importlib.util
    script = NW / 'scripts/03_run_localization.py'
    spec = importlib.util.spec_from_file_location('fsaverage_original_engine', script)
    solver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solver
    spec.loader.exec_module(solver)
    if args.method and not set(args.method) <= set(solver.METHODS):
        p.error('Unknown inverse method')
    forward_root, out, dataset = [x.expanduser().resolve() for x in
        [args.forward_root, args.output_root, args.dataset]]
    if out.exists() and not args.resume:
        raise FileExistsError(f'Output exists; use --resume to preserve records: {out}')
    for protected in [forward_root, NW, EXP / 'scripts', EXP / 'source', dataset]:
        if out == protected or protected in out.parents:
            raise ValueError('Output overlaps protected inputs')
    hashes = {}

    def verify(path, expected=None):
        h = solver.digest(path)
        if expected is not None and h != expected:
            raise ValueError(f'Frozen input changed: {path}')
        hashes[str(path)] = h
        return path

    fm_path = verify(forward_root / '04_forward_manifest.json')
    fm = json.loads(fm_path.read_text())
    if fm['phase'] != 'build':
        raise ValueError('Script 04 build outputs required')
    for path, h in fm['input_sha256'].items():
        verify(Path(path), h)
    status = pd.read_csv(verify(forward_root / '01_subject_forward_status.csv'), keep_default_na=False).set_index('test_subject')
    if len(status) != 7 or not status.index.is_unique or set(status.index) != {f'sub-{i:02d}' for i in range(1, 8)}:
        raise ValueError('Forward subject coverage differs')
    forward_paths = {}
    for subject, row in status.iterrows():
        if subject != 'sub-03' and row.status == 'PASS_FORWARD_NUMERICAL_ONLY':
            forward_paths[subject] = verify(Path(row.forward_path), row.forward_sha256)
    plan_path = verify(forward_root / '02_frozen_reconstruction_plan.csv')
    original_plan_hashes = [h for path, h in fm['input_sha256'].items()
                            if Path(path).name == '02_fsaverage_reconstruction_plan.csv']
    if not original_plan_hashes or any(h != solver.digest(plan_path) for h in original_plan_hashes):
        raise ValueError('Copied schedule differs from Script 04 frozen input')
    plan = pd.read_csv(plan_path, keep_default_na=False)
    if len(plan) != 5124 or plan.duplicated(KEY).any() or not plan.model_subject.eq('fsaverage').all():
        raise ValueError('Invalid reconstruction schedule')
    source = NW / 'source/02_parameter_grids'
    source_manifest = verify(source / 'source_manifest.json')
    doc = json.loads(source_manifest.read_text())
    for item in doc['files']:
        verify(source / item['copy'], item['sha256'])
    code = NW / 'source/03_solver_code'
    code_doc = json.loads(verify(code / 'code_manifest.json').read_text())
    for item in code_doc['files']:
        verify(code / item['path'], item['sha256'])
    configs = {}
    for strategy, filename in [('LOSO', 'loso_selection.csv'), ('Single-reference', 'single_selection.csv')]:
        originals = [h for path, h in fm['input_sha256'].items()
                     if Path(path).name == filename and Path(path).parent.name == 'source_configurations']
        if len(originals) != 1:
            raise ValueError(f'Missing frozen selection hash: {filename}')
        selected = pd.read_csv(verify(forward_root / 'source_configurations' / filename, originals[0]), dtype=str, keep_default_na=False)
        if len(selected) != 84 or not selected.selection_status.eq('SELECTED').all():
            raise ValueError('Expected 84 frozen selections per strategy')
        if strategy == 'Single-reference':
            selected['test_subject'] = selected.reference_subject
        resolved, errors = solver.enrich_configs(selected, source, doc)
        if errors or len(resolved) != 84:
            raise ValueError(f'{strategy} parameter resolution failed: {errors}')
        configs.update({(strategy, *k): v for k, v in resolved.items()})
    for subject, row in status.iterrows():
        native = dataset / 'derivatives/sourcemodelling' / subject / 'fwd' / f'{subject}_fwd.fif'
        verify(native, row.native_forward_sha256)
    bem = verify(Path(fm['forward_parameters']['bem']))
    # Its BEM path was frozen among Script 04's input hashes.
    if str(bem) not in fm['input_sha256']:
        raise ValueError('Fsaverage BEM not in forward provenance')
    verify(script)
    verify(Path(__file__).resolve())
    versions = {}
    for package in ['mne', 'numpy', 'scipy', 'pandas', 'sesameeg', 'invertmeeg']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not_installed'
    fingerprint = dict(input_sha256=hashes, versions=versions, dataset=str(dataset),
        anatomy='fsaverage', montage_template='reference', peak_policy='original known hemisphere',
        registration_fitting=False, anatomical_mapping=False)
    jobs = []
    for record in plan.to_dict('records'):
        owner = record['test_subject'] if record['strategy'] == 'LOSO' else record['reference_subject']
        cfg = configs[(record['strategy'], record['method'], owner)]
        record.update(fold_id=owner, condition='single_reference_fsaverage',
            configuration_key=cfg['configuration_key'], parameter_owner=owner,
            parameter_selection='loso' if record['strategy'] == 'LOSO' else 'single_reference',
            configuration_parameters=json.dumps(cfg['parameters'], sort_keys=True))
        jobs.append(record)
    requested = [j for j in jobs if all(not getattr(args, k) or j[k] in getattr(args, k)
        for k in ['strategy', 'method', 'test_subject', 'reference_subject', 'run'])]
    if not requested:
        raise ValueError('Filters match no jobs')

    class FsaverageEngine(solver.Engine):
        def prepare(self, job, cfg):
            # Super.prepare still obtains nominal/replacement geometry from the
            # native reference forward; only model lookup uses this private key.
            shaped = dict(job, model_subject='fsaverage:' + job['test_subject'])
            return super().prepare(shaped, cfg)

        def forward(self, subject):
            if subject.startswith('fsaverage:'):
                test = subject.split(':', 1)[1]
                if subject not in self.cache:
                    self.cache[subject] = self.mne.read_forward_solution(forward_paths[test], verbose=False)
                return self.cache[subject]
            return super().forward(subject)

    # The existing ECD dispatch calls this module-level helper. Change its
    # anatomy lookup only, retaining the original dipole solver and transform.
    original_find_bem = solver.find_bem
    solver.find_bem = lambda root, subject: bem if subject == 'fsaverage' else original_find_bem(root, subject)
    with solver.lock_output(out):
        frozen = out / '06_frozen_run_manifest.json'
        if frozen.exists():
            if json.loads(frozen.read_text()) != fingerprint:
                raise ValueError('Resume inputs/code/environment differ from frozen run')
        else:
            if args.resume and (out / '01_execution_status.csv').exists():
                raise ValueError('Checkpoint without manifest')
            solver.atomic_json(frozen, fingerprint)
        solver.atomic_csv(out / '02_reconstruction_plan.csv', pd.DataFrame(jobs))
        solver.atomic_json(out / '05_resolved_configurations.json', {'|'.join(k): v for k, v in configs.items()})
        path = out / '01_execution_status.csv'
        if path.exists():
            records = pd.read_csv(path, keep_default_na=False).to_dict('records')
            if len(records) != 5124 or {tuple(r[k] for k in KEY) for r in records} != {tuple(j[k] for k in KEY) for j in jobs}:
                raise ValueError('Checkpoint keys differ from full plan')
        else:
            records = [dict(j, status='PENDING' if j['test_subject'] in forward_paths else 'BLOCKED_COREGISTRATION_QA',
                failure_reason='' if j['test_subject'] in forward_paths else 'Test subject has no eligible forward; sub-03 remains blocked',
                runtime_seconds=0.0) for j in jobs]
        recorded = {tuple(r[k] for k in KEY): r for r in records}

        def checkpoint():
            frame = pd.DataFrame(recorded.values())
            solver.atomic_csv(path, frame)
            solver.atomic_csv(out / '07_execution_summary.csv', frame.groupby(['strategy', 'method', 'status']).size().reset_index(name='count'))
            fields = KEY + ['status', 'configuration_key', 'actual_channels_json', 'nominal_channels_json',
                'replacement_details_json', 'eeg_data_sha256', 'epoch_times_sha256', 'montage_template_subject', 'failure_reason']
            solver.atomic_csv(out / '09_channel_selection_audit.csv', frame.reindex(columns=fields))
        checkpoint()
        if args.phase == 'plan':
            print('Plan valid: 5,124 cases; 588 sub-03 blocked; no EEG loaded or inverse run.')
            return
        engine_args = SimpleNamespace(dataset=dataset, code_source_root=code, quiet=args.quiet, montage_template='reference', bem_root=bem.parent)
        engine = FsaverageEngine(engine_args)
        attempted = 0
        for job in requested:
            key = tuple(job[k] for k in KEY)
            if recorded[key]['status'] != 'PENDING':
                continue
            config = configs[(job['strategy'], job['method'], job['parameter_owner'])]
            row = dict(job, status='FAIL', failure_reason='')
            started = time.perf_counter()
            try:
                row.update(engine.solve(job, config['parameters']))
                row.update(status='PASS', peak_selection_policy='known_stimulation_hemisphere_original',
                    localization_error_valid=False, interpretation='RECONSTRUCTION_ONLY_UNREGISTERED_DISTANCES_DIAGNOSTIC')
            except Exception as exc:
                row['failure_reason'] = f'{type(exc).__name__}: {exc}'
                with (out / '04_failure_tracebacks.log').open('a') as log:
                    log.write(str(key) + '\n' + traceback.format_exc() + '\n')
            row['runtime_seconds'] = time.perf_counter() - started
            row['forward_sha256'] = status.loc[job['test_subject'], 'forward_sha256']
            recorded[key] = row
            checkpoint()
            attempted += 1
            print(f'{key}: {row["status"]} ({row["runtime_seconds"]:.2f}s)', flush=True)
            if args.limit and attempted >= args.limit:
                break
        print(f'Attempted {attempted} new fits. Existing PASS/FAIL/BLOCKED preserved.')


if __name__ == '__main__':
    main()
