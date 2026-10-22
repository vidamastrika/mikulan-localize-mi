#!/usr/bin/env python3
"""Script 30: full all-reference-subject cross-subject SESAME batch.

FROZEN SESAME CONFIGURATION — SELECTED FROM WITHIN-SUBJECT GRID (ALL-PASS ONLY)
-----------------------------------------------------------------------------
Selection criterion : lowest MEDIAN localization error among configurations
                      with PASS=61 and FAIL=0 on all 61 Localize-MI EEG runs.
Reference CSV       : grid_results(20261021-051319).csv (previous Script 23 grid)
Selected grid ID    : sesame_fs_0p12
EEG montage         : all_good (use only target EEG channels marked "good")
fs                  : 0.12
Noise standard dev. : fs * max(abs(target evoked EEG signal))
n_parts             : 100
max_n_dips          : 1
lam                 : 0.25
hyper_q             : True
random_seed         : 0
noise_covariance    : None (not supplied to SESAME)
Target time window  : -0.002 to +0.002 seconds
Within-subject grid : 61 PASS / 0 FAIL; median distance 14.196784 mm

This configuration is FROZEN before the cross-subject batch. No parameters,
donors, or registrations are selected by minimizing cross-subject error.

Research interpretation:
  * Reconstruction uses TEST EEG and REFERENCE forward/head model ONLY.
  * No test MRI, test forward, or ground-truth coordinates enter fitting.
  * The output is in REFERENCE source coordinates. NO localization error is
    calculated here: target-native mapping is a separate, audited stage.
  * Selection used all seven within-subject participants, so this is a head-
    model transfer experiment, not a strictly nested unseen-subject CV claim.
  * The earlier fs=0.15 pilot is SEPARATE and should not be pooled with fs=0.12.

Usage:
  python cross_subject/scripts/30_all_reference_sesame_batch.py
  python cross_subject/scripts/30_all_reference_sesame_batch.py --refresh-status-only
  python cross_subject/scripts/30_all_reference_sesame_batch.py --execute --resume --limit 10
  python cross_subject/scripts/30_all_reference_sesame_batch.py --execute --resume --quiet

By default, writes a MANIFEST ONLY and does not run SESAME.

Script 30 CSV/metadata output: cross_subject/outputs/30_all_reference_sesame_batch/
Reconstruction cache (unchanged): cross_subject/outputs/27_cross_subject_sesame_batch/pilot_results/
Use --refresh-status-only to reconstruct all 366 status rows and timing without reruns.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FS = 0.12
N_PARTS = 100
MAX_N_DIPS = 1
LAM = 0.25
SEED = 0
HYPER_Q = True
MONTAGE = 'all_good'
EXPECTED_SUBJECTS = 7
EXPECTED_RUNS = 61
CONFIG = {
    'grid_configuration_id': 'sesame_fs_0p12',
    'selection': 'lowest median among all-PASS 61/61 within-subject configurations',
    'fs': FS,
    'montage': MONTAGE,
    'n_parts': N_PARTS,
    'max_n_dips': MAX_N_DIPS,
    'lam': LAM,
    'hyper_q': HYPER_Q,
    'seed': SEED,
    'noise_covariance_supplied': False,
    'target_tmin_s': -0.002,
    'target_tmax_s': 0.002,
    'within_subject_grid_pass': 61,
    'within_subject_grid_fail': 0,
    'within_subject_median_error_mm': 14.19678442163502,
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, default=Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI')
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/30_all_reference_sesame_batch'),
                   help='Script 30 reports and CSVs (never contains the source reconstruction cache)')
    p.add_argument('--results-root', type=Path, default=Path('cross_subject/outputs/27_cross_subject_sesame_batch'),
                   help='Existing Script 27 SESAME cache, containing pilot_results; do not move it')
    p.add_argument('--pilot-script', type=Path, default=Path(__file__).with_name('05_sesame_cross_subject_pilot.py'))
    # Script 30 is restricted to the predeclared six-reference-subject design.
    # In contrast to Script 27, we do not expose a donor-selection switch.
    p.add_argument('--execute', action='store_true', help='Execute jobs; default is safe manifest-only mode')
    p.add_argument('--smoke-one-per-target', action='store_true',
                   help='Execute one representative run per test subject (seven total); not the full grid')
    p.add_argument('--limit', type=int, help='Maximum number of jobs to execute (use 7 for a smoke test)')
    p.add_argument('--resume', action='store_true', help='Reuse only jobs whose result JSON says PASS with frozen parameters')
    p.add_argument('--refresh-status-only', action='store_true',
                   help='Rebuild complete 366-row status CSV from existing result JSONs, without executing SESAME')
    p.add_argument('--allow-incomplete', action='store_true',
                   help='Allow a dataset other than the expected 7 subjects/61 runs; useful for testing only')
    p.add_argument('--quiet', action='store_true', help='Quiet SESAME particle output')
    return p.parse_args()


def discover(dataset):
    pattern = re.compile(r'^(sub-\d+)_task-seegstim_(run-\d+)_epochs\.npy$')
    root = dataset / 'derivatives' / 'epochs'
    if not root.is_dir():
        raise FileNotFoundError(f'No epochs directory: {root}')
    runs = []
    for path in sorted(root.glob('sub-*/eeg/*_task-seegstim_run-*_epochs.npy')):
        m = pattern.fullmatch(path.name)
        if not m or path.parent.parent.name != m.group(1):
            continue
        subject, run = m.groups()
        channels = path.with_name(path.name.replace('_epochs.npy', '_channels.tsv'))
        events = path.with_name(path.name.replace('_epochs.npy', '_epochs.tsv'))
        for needed in (channels, events):
            if not needed.is_file():
                raise FileNotFoundError(f'Missing corresponding EEG metadata: {needed}')
        runs.append({'target': subject, 'run': run, 'epochs_file': str(path)})
    if not runs:
        raise ValueError(f'No valid target EEG runs found in {root}')
    keys = [(r['target'], r['run']) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate target/run combinations')
    return sorted(runs, key=lambda x: (x['target'], int(x['run'].split('-')[1])))


def job_dir(root, row):
    return root / 'pilot_results' / f"{row['target']}_with_{row['donor']}_{row['run']}"


def is_completed(root, row):
    p = job_dir(root, row) / 'sesame_pilot_result.json'
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
        prm = d.get('parameters', {})
        return (d.get('status') == 'PASS' and d.get('target') == row['target']
                and d.get('donor') == row['donor']
                and all(prm.get(k) == v for k, v in [('fs', FS), ('n_parts', N_PARTS),
                       ('max_n_dips', MAX_N_DIPS), ('lam', LAM), ('hyper_q', True), ('seed', SEED)])
                and d.get('target_forward_used') is False
                and d.get('target_electrode_tsv_used') is False)
    except (OSError, ValueError, TypeError):
        return False


def result_runtime_seconds(result_file):
    """Return the pilot's measured SESAME runtime, or blank if unavailable.

    Do not confuse this with zero elapsed time for a resumed/skipped job.
    """
    try:
        result = json.loads(Path(result_file).read_text(encoding='utf-8'))
        value = result.get('runtime_seconds')
        if value is None:
            return ''
        value = float(value)
        if not (0 <= value < float('inf')):
            return ''
        return value
    except (OSError, ValueError, TypeError):
        return ''


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def run():
    a = arguments()
    if a.execute and not a.resume:
        raise ValueError('Script 30 requires --resume for execution: protects the 61 existing results from accidental reruns')
    if a.limit is not None and a.limit < 1:
        raise ValueError('--limit must be positive')
    dataset = a.dataset.expanduser().resolve()
    output = a.output_root.expanduser()
    results_root = a.results_root.expanduser()
    if output.resolve() == results_root.resolve():
        raise ValueError("--output-root and --results-root must be different to preserve Script 27 data")
    runs = discover(dataset)
    subjects = sorted({r['target'] for r in runs})
    if (len(subjects) != EXPECTED_SUBJECTS or len(runs) != EXPECTED_RUNS) and not a.allow_incomplete:
        raise RuntimeError(f'Expected {EXPECTED_SUBJECTS} subjects and {EXPECTED_RUNS} runs; found {len(subjects)} and {len(runs)}. Use --allow-incomplete ONLY if deliberate.')
    if len(subjects) < 2:
        raise ValueError('Cross-subject experiment requires >=2 subjects')
    schedule = []
    for r in runs:
        donors = [s for s in subjects if s != r['target']]
        for donor in donors:
            fwd = dataset / 'derivatives' / 'sourcemodelling' / donor / 'fwd' / f'{donor}_fwd.fif'
            if not fwd.is_file():
                raise FileNotFoundError(f'Missing donor forward model: {fwd}')
            schedule.append({**r, 'donor': donor, 'forward_file': str(fwd),
                             'result_json': str(job_dir(results_root, {**r, 'donor': donor}) / 'sesame_pilot_result.json')})
    output.mkdir(parents=True, exist_ok=True)
    print('Existing reconstruction cache:', results_root / 'pilot_results')
    save_csv(output / 'experiment_schedule_all_references.csv', schedule,
             ['target', 'donor', 'run', 'epochs_file', 'forward_file', 'result_json'])
    (output / 'frozen_sesame_configuration_all_references.json').write_text(json.dumps({
        'config': CONFIG, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'design': 'all-donors', 'subjects': subjects, 'target_run_count': len(runs),
        'job_count': len(schedule), 'ground_truth_used_for_selection': False,
        'note': 'Within-subject grid parameter selection involved these subjects; no cross-subject errors used.'
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Found {len(subjects)} subjects, {len(runs)} EEG runs, {len(schedule)} target/donor/run jobs.')
    print('Frozen: fs=0.12, all_good, n_parts=100, max_n_dips=1, lam=0.25, hyper_q=True, seed=0, noise_cov=None')
    print('Manifest:', output / 'experiment_schedule_all_references.csv')
    print('Jobs per target:', dict(Counter(r['target'] for r in schedule)))
    print('Donor mapping:', {s: sorted({r['donor'] for r in schedule if r['target'] == s}) for s in subjects})
    status_path = output / 'execution_status_all_references.csv'
    status_cols = ['target', 'donor', 'run', 'batch_status', 'return_code', 'result_json', 'runtime_seconds']

    def persist_status(rows):
        # Atomic replacement prevents a half-written CSV after interruption.
        tmp = status_path.with_suffix('.csv.tmp')
        save_csv(tmp, rows, status_cols)
        tmp.replace(status_path)

    if a.refresh_status_only:
        rows = []
        for row in schedule:
            complete = is_completed(results_root, row)
            rows.append({**row, 'batch_status': 'SKIPPED_PASS' if complete else 'PENDING',
                         'return_code': 0 if complete else '',
                         'runtime_seconds': result_runtime_seconds(row['result_json']) if complete else ''})
        persist_status(rows)
        completed = sum(x['batch_status'] == 'SKIPPED_PASS' for x in rows)
        timed = sum(x['runtime_seconds'] != '' for x in rows)
        print(f'Refreshed {status_path}: {completed}/{len(rows)} validated saved PASS results; {timed} measured runtimes.')
        print('No SESAME reconstructions were executed.')
        return
    if not a.execute:
        print('DRY RUN: SESAME was not executed. Use --refresh-status-only to recover cached runtimes or --execute --resume to process all jobs.')
        return
    pilot = a.pilot_script.expanduser().resolve()
    if not pilot.is_file():
        raise FileNotFoundError(f'Pilot implementation not found: {pilot}')
    if len(subjects) == 7 and len(runs) == 61 and len(schedule) != 366:
        raise RuntimeError(f'Expected 366 combinations; received {len(schedule)}')
    jobs = schedule
    if a.smoke_one_per_target:
        first = {}
        for row in jobs:
            first.setdefault(row['target'], row)
        jobs = list(first.values())
    if a.limit:
        jobs = jobs[:a.limit]
    outcomes = []
    persist_status(outcomes)
    for i, row in enumerate(jobs, 1):
        dest = job_dir(results_root, row)
        dest.mkdir(parents=True, exist_ok=True)
        if a.resume and is_completed(results_root, row):
            status = 'SKIPPED_PASS'
            print(f'[{i}/{len(jobs)}] {row["target"]} {row["run"]} with {row["donor"]}: {status}')
            outcomes.append({**row, 'batch_status': status, 'return_code': 0,
                             'runtime_seconds': result_runtime_seconds(row['result_json'])})
            persist_status(outcomes)
            continue
        cmd = [sys.executable, str(pilot), '--dataset', str(dataset),
               '--target', row['target'], '--donor', row['donor'], '--run', row['run'],
               '--output-root', str(results_root / 'pilot_results'), '--fs', str(FS),
               '--n-parts', str(N_PARTS), '--max-n-dips', str(MAX_N_DIPS),
               '--lam', str(LAM), '--seed', str(SEED)]
        if a.quiet:
            cmd.append('--quiet')
        print(f'[{i}/{len(jobs)}] Running {row["target"]} {row["run"]} with {row["donor"]}', flush=True)
        with (dest / 'sesame_execution.log').open('w', encoding='utf-8') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        try:
            result = json.loads((dest / 'sesame_pilot_result.json').read_text(encoding='utf-8'))
            success = rc == 0 and result.get('status') == 'PASS' and is_completed(results_root, row)
        except (OSError, ValueError):
            success = False
        status = 'PASS' if success else 'FAIL'
        print(f'  {status} (exit {rc}); log: {dest / "sesame_execution.log"}', flush=True)
        outcomes.append({**row, 'batch_status': status, 'return_code': rc,
                         'runtime_seconds': result_runtime_seconds(row['result_json']) if success else ''})
        persist_status(outcomes)
    failed = sum(r['batch_status'] == 'FAIL' for r in outcomes)
    print(f'Finished {len(outcomes)} jobs: {failed} failures. Results remain in REFERENCE source space.')
    print('Full-combination status:', status_path)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
