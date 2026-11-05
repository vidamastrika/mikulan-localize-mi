#!/usr/bin/env python3
# Script 05b — Separate continuation with SESAME capacity compatibility.
# Purpose: reuse Script 05/no-warp engine; increase only neighbour guard 100->256.
# Inputs: same Script 04 frozen data/forwards, stopped Script 05 status+manifest,
#   and sesame_capacity_compat.py. Radius, probabilities and inverse settings unchanged.
# Outputs: NEW outputs/05_reconstruct_sesame_compat; parent records retained, import
#   audit, runtime/configuration hashes and compatibility adapter provenance.
# Parameters: --phase plan/reconstruct; --import-status (original stopped ledger),
#   --resume, same filters and limit as Script 05. sub-03 remains blocked.
# Dependencies: frozen no-warp Engine and helpers, Script 04, original stopped run.
# Complete commands from repository root:
#   .venv/bin/python cross_subject_fsaverage/scripts/05b_reconstruct_sesame_compat.py --phase plan
#   .venv/bin/python cross_subject_fsaverage/scripts/05b_reconstruct_sesame_compat.py \
#       --phase reconstruct --resume --strategy LOSO --method SESAME \
#       --test-subject sub-01 --run run-01 --limit 1 --quiet
# Do not run the entire continuation before reviewing that pilot.
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
    p.add_argument('--import-status', type=Path, default=EXP / 'outputs/05_reconstruct/01_execution_status.csv')
    p.add_argument('--phase', choices=['plan', 'reconstruct'], default='plan')
    p.add_argument('--forward-root', type=Path, default=EXP / 'outputs/04_make_forward')
    p.add_argument('--output-root', type=Path, default=EXP / 'outputs/05_reconstruct_sesame_compat')
    p.add_argument('--dataset', type=Path, default=Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    p.add_argument('--strategy', choices=['LOSO', 'Single-reference'], action='append')
    p.add_argument('--method', action='append')
    p.add_argument('--test-subject', action='append')
    p.add_argument('--reference-subject', action='append')
    p.add_argument('--run', action='append')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--exploratory-blocked-sub03', action='store_true',
        help='Run sub-03 blocked cases in a separate exploratory output using its failed-QA candidate transform')
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
    exploratory_forward = None
    if args.exploratory_blocked_sub03:
        if out == (EXP / 'outputs/05_reconstruct_sesame_compat').resolve():
            raise ValueError('Exploratory sub-03 runs require a separate --output-root')
        # Build a separately labeled forward from the existing failed-QA candidate.
        # This is a computational sensitivity only; it does not clear the QA failure.
        out.mkdir(parents=True, exist_ok=True)
        exploratory_forward = out / 'exploratory_sub03_fsaverage-fwd.fif'
        coreg = EXP / 'outputs/02_coregister_eeg'
        trans_path = verify(coreg / 'transforms/sub-03_head_to_fsaverage-trans.fif')
        native = dataset / 'derivatives/sourcemodelling/sub-03/fwd/sub-03_fwd.fif'
        verify(native, status.loc['sub-03', 'native_forward_sha256'])
        import mne
        trans = mne.read_trans(trans_path)
        nfwd = mne.read_forward_solution(native, verbose=False)
        names = nfwd['info']['ch_names']
        positions = [ch['loc'][:3] for ch in nfwd['info']['chs']]
        info = mne.create_info(names, sfreq=1.0, ch_types='eeg')
        info.set_montage(mne.channels.make_dig_montage(
            ch_pos=dict(zip(names, positions)), coord_frame='head'))
        src = Path(fm['forward_parameters']['source_space'])
        bem_path = Path(fm['forward_parameters']['bem'])
        candidate = mne.make_forward_solution(info, trans=trans, src=src,
            bem=bem_path, meg=False, eeg=True, mindist=0.0, n_jobs=1)
        mne.write_forward_solution(exploratory_forward, candidate, overwrite=True)
        exploratory_hash = solver.digest(exploratory_forward)
        forward_paths['sub-03'] = verify(exploratory_forward, exploratory_hash)
        status.loc['sub-03', 'forward_sha256'] = exploratory_hash
        status.loc['sub-03', 'forward_path'] = str(exploratory_forward)
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
    adapter_path = verify(EXP / 'scripts/sesame_capacity_compat.py')
    parent_status = verify(args.import_status.expanduser().resolve())
    parent_manifest = verify(parent_status.parent / '06_frozen_run_manifest.json')
    parent_state = json.loads(parent_manifest.read_text())
    # Verify every original run input, including original script/package identity.
    for path, digest in parent_state['input_sha256'].items():
        if args.exploratory_blocked_sub03 and Path(path).resolve() == Path(__file__).resolve():
            continue
        verify(Path(path), digest)
    for path, digest in hashes.copy().items():
        if args.exploratory_blocked_sub03 and Path(path).resolve() == Path(__file__).resolve():
            continue
        if path in parent_state['input_sha256'] and parent_state['input_sha256'][path] != digest:
            raise ValueError('Parent/current reconstruction inputs differ')
    versions = {}
    for package in ['mne', 'numpy', 'scipy', 'pandas', 'sesameeg', 'invertmeeg']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not_installed'
    if versions != parent_state['versions']:
        raise ValueError('Parent/current solver versions differ')
    fingerprint = dict(input_sha256=hashes, versions=versions, dataset=str(dataset),
        anatomy='fsaverage', montage_template='reference', peak_policy='original known hemisphere',
        registration_fitting=False, anatomical_mapping=False,
        exploratory_blocked_sub03=bool(args.exploratory_blocked_sub03),
        exploratory_alignment_note=('Failed-QA candidate transform; results are exploratory and do not clear coregistration QA'
            if args.exploratory_blocked_sub03 else None),
        compatibility=dict(capacity=256, original_capacity=100, radius='unchanged',
            retry_policy='only original SESAME neighbour-capacity failures'), parent_status=str(parent_status))
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
    if args.exploratory_blocked_sub03:
        requested = [j for j in requested if j['test_subject'] == 'sub-03']
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
            records = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict('records')
            if len(records) != 5124 or {tuple(r[k] for k in KEY) for r in records} != {tuple(j[k] for k in KEY) for j in jobs}:
                raise ValueError('Checkpoint keys differ from full plan')
        else:
            records = [dict(j, status='PENDING' if j['test_subject'] in forward_paths else 'BLOCKED_COREGISTRATION_QA',
                failure_reason='' if j['test_subject'] in forward_paths else 'Test subject has no eligible forward; sub-03 remains blocked',
                runtime_seconds=0.0) for j in jobs]
        if not path.exists():
            parent = pd.read_csv(parent_status, dtype=str, keep_default_na=False)
            if (len(parent) != 5124 or parent.duplicated(KEY).any()
                or set(map(tuple, parent[KEY].values)) != {tuple(j[k] for k in KEY) for j in jobs}):
                raise ValueError('Parent checkpoint must contain the exact complete schedule')
            expected = {tuple(j[k] for k in KEY): j for j in jobs}
            records = parent.to_dict('records')
            for r in records:
                j = expected[tuple(r[k] for k in KEY)]
                if (r['configuration_key'] != j['configuration_key']
                    or json.loads(r['configuration_parameters']) != json.loads(j['configuration_parameters'])):
                    raise ValueError('Parent configuration differs')
                if r['status'] not in ['PASS', 'FAIL', 'PENDING', 'BLOCKED_COREGISTRATION_QA']:
                    raise ValueError('Unknown parent status')
                if r['test_subject'] == 'sub-03' and r['status'] != 'BLOCKED_COREGISTRATION_QA':
                    raise ValueError('Parent sub-03 block differs')
                r['imported_original_status'] = r['status']
                r['imported_original_failure_reason'] = r['failure_reason']
                r['imported_original_runtime_seconds'] = r['runtime_seconds']
                if (r['method'] == 'SESAME' and r['status'] == 'FAIL'
                    and 'their maximum number is too big' in r['failure_reason']):
                    r.update(status='PENDING', failure_reason='', runtime_seconds=0.0)
                if args.exploratory_blocked_sub03 and r['test_subject'] == 'sub-03':
                    r.update(status='PENDING', failure_reason='', runtime_seconds=0.0,
                        record_origin='exploratory_failed_qa_candidate')
                r['record_origin'] = 'parent_checkpoint'
            solver.atomic_csv(out / '08_import_audit.csv', pd.DataFrame(records)[KEY + [
                'imported_original_status', 'imported_original_failure_reason',
                'imported_original_runtime_seconds', 'status']])
        recorded = {tuple(r[k] for k in KEY): r for r in records}

        def checkpoint():
            frame = pd.DataFrame(recorded.values())
            solver.atomic_csv(path, frame)
            solver.atomic_csv(out / '07_execution_summary.csv', frame.groupby(['strategy', 'method', 'status']).size().reset_index(name='count'))
            if args.exploratory_blocked_sub03:
                exploratory = frame.loc[frame.test_subject.eq('sub-03')].copy()
                solver.atomic_csv(out / '11_sub03_exploratory_results.csv', exploratory)
                solver.atomic_csv(out / '12_sub03_exploratory_summary.csv',
                    exploratory.groupby(['strategy', 'method', 'status']).size().reset_index(name='count'))
            fields = KEY + ['status', 'configuration_key', 'actual_channels_json', 'nominal_channels_json',
                'replacement_details_json', 'eeg_data_sha256', 'epoch_times_sha256', 'montage_template_subject', 'failure_reason']
            solver.atomic_csv(out / '09_channel_selection_audit.csv', frame.reindex(columns=fields))
        checkpoint()
        if args.phase == 'plan':
            print('Compatibility plan valid; parent PASS/FAIL imported, only SESAME capacity failures scheduled for retry; no inverse run.')
            return
        engine_args = SimpleNamespace(dataset=dataset, code_source_root=code, quiet=args.quiet, montage_template='reference', bem_root=bem.parent)
        engine = FsaverageEngine(engine_args)
        from sesame_capacity_compat import install
        compatibility = install()
        solver.atomic_json(out / '10_sesame_capacity_adapter.json', compatibility)
        attempted = 0
        for job in requested:
            key = tuple(job[k] for k in KEY)
            if recorded[key]['status'] != 'PENDING':
                continue
            config = configs[(job['strategy'], job['method'], job['parameter_owner'])]
            row = dict(job, status='FAIL', failure_reason='', record_origin='compatibility_run')
            for field in ['imported_original_status', 'imported_original_failure_reason', 'imported_original_runtime_seconds']:
                row[field] = recorded[key].get(field, '')
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
