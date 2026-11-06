#!/usr/bin/env python3
"""Evaluate fsaverage reconstructions with the planned unregistered HEAD diagnostic.

Run from any directory with the project's Python environment. Reads existing CSVs
and validates frozen inputs; performs no inverse fits or MRI registration. Writes
only a fresh output directory. Distances are diagnostic, not validated anatomical
localization errors. Sub-03 is reported separately because its scalp alignment
failed QA. PASS denotes solver completion, not anatomical accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'cross_subject_fsaverage'
KEY = ['strategy', 'method', 'reference_subject', 'test_subject', 'run']
XYZ = ['estimated_head_' + axis + '_mm' for axis in 'xyz']
GT_XYZ = ['gt_head_' + axis + '_mm' for axis in 'xyz']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(frame, groups):
    rows = []
    for values, group in frame.groupby(groups, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        distances = group.diagnostic_distance_mm.dropna()
        rows.append(dict(zip(groups, values)) | dict(
            n_total=len(group), n_pass=int(group.status.eq('PASS').sum()),
            n_fail=int(group.status.eq('FAIL').sum()),
            n_pending=int(group.status.eq('PENDING').sum()),
            n_blocked=int(group.status.eq('BLOCKED_COREGISTRATION_QA').sum()),
            n_distances=len(distances),
            pass_rate=float(group.status.eq('PASS').mean()),
            diagnostic_mean_mm=distances.mean(), diagnostic_median_mm=distances.median(),
            diagnostic_q25_mm=distances.quantile(.25), diagnostic_q75_mm=distances.quantile(.75)))
    return pd.DataFrame(rows)


def evaluate(reconstruction_root, ground_truth, output):
    reconstruction_root = reconstruction_root.resolve()
    ground_truth = ground_truth.resolve()
    output = output.resolve()
    require(not output.exists(), f'Output already exists; choose a fresh --output-root: {output}')
    ledger_path = reconstruction_root / '01_execution_status.csv'
    plan_path = reconstruction_root / '02_reconstruction_plan.csv'
    frozen_path = reconstruction_root / '06_frozen_run_manifest.json'
    frozen = json.loads(frozen_path.read_text())
    input_hashes = {str(p): digest(p) for p in [ledger_path, plan_path, frozen_path, ground_truth]}
    for path, expected in frozen['input_sha256'].items():
        require(digest(path) == expected, f'Frozen reconstruction input changed: {path}')
    data = pd.read_csv(ledger_path, keep_default_na=False)
    plan = pd.read_csv(plan_path, keep_default_na=False)
    required = KEY + XYZ + ['status', 'coordinate_frame', 'coordinate_units', 'model_subject',
                           'stimulation_pair', 'hemisphere', 'failure_reason']
    require(set(required) <= set(data), 'Reconstruction ledger is missing required columns')
    require(len(data) == 5124 and not data.duplicated(KEY).any(), 'Expected 5,124 unique reconstruction jobs')
    require(not plan.duplicated(KEY).any(), 'Duplicate keys in reconstruction plan')
    require(set(map(tuple, data[KEY].values)) == set(map(tuple, plan[KEY].values)), 'Ledger differs from frozen plan')
    require(set(data.status) <= {'PASS', 'FAIL', 'PENDING', 'BLOCKED_COREGISTRATION_QA'}, 'Unknown solver status')
    require(data.model_subject.eq('fsaverage').all(), 'Expected fsaverage anatomy throughout')
    gt = pd.read_csv(ground_truth, keep_default_na=False)
    gt_fields = ['test_subject', 'run', 'coordinate_audit', 'gt_stimulation_pair', 'gt_hemisphere'] + GT_XYZ
    require(set(gt_fields) <= set(gt), 'Ground-truth table lacks audited HEAD coordinates')
    require(not gt.duplicated(['test_subject', 'run']).any(), 'Ambiguous ground-truth keys')
    require(len(gt) == 61 and set(map(tuple, gt[['test_subject', 'run']].values)) ==
            set(map(tuple, data[['test_subject', 'run']].values)), 'Ground truth must cover exactly the 61 test runs')
    require(gt.coordinate_audit.eq('PASS_NUMERICAL').all(), 'Ground-truth coordinate audit did not pass')
    gt[GT_XYZ] = gt[GT_XYZ].apply(pd.to_numeric, errors='raise')
    require(np.isfinite(gt[GT_XYZ].to_numpy(float)).all(), 'Nonfinite ground truth')
    # Select explicit fields to avoid reusing any historical evaluation results.
    result = data.merge(gt[gt_fields], on=['test_subject', 'run'], how='left', validate='many_to_one')
    passed = result.status.eq('PASS')
    require(result.loc[passed, 'coordinate_frame'].eq('model_head').all(), 'Unexpected estimate frame')
    require(result.loc[passed, 'coordinate_units'].eq('mm').all(), 'Estimate units must be mm')
    require(result.loc[passed, 'stimulation_pair'].eq(result.loc[passed, 'gt_stimulation_pair']).all(), 'Stimulation pair mismatch')
    require(result.loc[passed, 'hemisphere'].eq(result.loc[passed, 'gt_hemisphere']).all(), 'Stimulation hemisphere mismatch')
    estimates = result.loc[passed, XYZ].apply(pd.to_numeric, errors='raise').to_numpy(float)
    require(np.isfinite(estimates).all(), 'PASS estimate contains nonfinite coordinates')
    result['diagnostic_distance_mm'] = np.nan
    result.loc[passed, 'diagnostic_distance_mm'] = np.linalg.norm(
        estimates - result.loc[passed, GT_XYZ].to_numpy(float), axis=1)
    # Keep diagnostic values out of columns that downstream tools treat as valid errors.
    result['localization_error_mm'] = np.nan
    result['localization_error_valid'] = False
    result['evaluation_status'] = np.where(passed, 'DIAGNOSTIC_ONLY', 'NO_ESTIMATE_' + result.status)
    result['evaluation_cohort'] = np.where(result.test_subject.eq('sub-03'),
        'sub03_failed_coregistration_exploratory', 'other_six_subjects_review_required')
    result['interpretation'] = 'UNREGISTERED_HEAD_DISTANCE_ONLY_ANATOMICAL_ACCURACY_UNVALIDATED'
    method_summary = summarize(result, ['evaluation_cohort', 'strategy', 'method'])
    subject_summary = summarize(result, ['evaluation_cohort', 'strategy', 'method', 'test_subject'])
    # Single-reference runs appear under six references; summarize within each
    # participant before giving every participant equal weight.
    participant_summary = subject_summary.groupby(
        ['evaluation_cohort', 'strategy', 'method'], as_index=False).agg(
            n_subjects=('test_subject', 'nunique'),
            mean_subject_median_diagnostic_mm=('diagnostic_median_mm', 'mean'),
            median_subject_median_diagnostic_mm=('diagnostic_median_mm', 'median'))
    failures = result.loc[~passed, KEY + ['status', 'failure_reason', 'evaluation_cohort']]
    # Detect concurrent reconstruction writes before publishing a coherent evaluation.
    for path, expected in input_hashes.items():
        require(digest(path) == expected, f'Input changed during evaluation: {path}')
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / '01_run_level_diagnostic.csv', index=False)
    gt.to_csv(output / '02_ground_truth_coordinates.csv', index=False)
    method_summary.to_csv(output / '03_method_summary.csv', index=False)
    subject_summary.to_csv(output / '04_subject_summary.csv', index=False)
    participant_summary.to_csv(output / '05_equal_subject_summary.csv', index=False)
    failures.to_csv(output / '06_unsuccessful_cases.csv', index=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(Path(__file__)), input_sha256=input_hashes,
        frozen_inputs_verified=len(frozen['input_sha256']),
        rows=len(result), status_counts=result.status.value_counts().to_dict(),
        diagnostic_distances=int(result.diagnostic_distance_mm.notna().sum()),
        metric='Euclidean distance (mm): estimated model HEAD minus test ground-truth HEAD',
        ground_truth_conversion='Existing audited stimulation midpoint, native surface to native HEAD via test subject H5',
        registration_applied=False, valid_anatomical_localization_error=False,
        limitations=['Unregistered head-coordinate distances are diagnostic only.',
                     'Sub-03 uses a failed-QA candidate alignment and is reported separately.',
                     'Other six subjects retain anatomical review requirements.',
                     'Cortical methods use the original known-stimulation-hemisphere peak policy; ECD/RAP-MUSIC use maximum GOF.',
                     'Single-reference pooled summaries repeat runs across references; use equal-subject summary for participant weighting.',
                     'Failures retain missing distances and remain in pass-rate denominators.'])
    (output / '07_evaluation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.txt').write_text(
        'Diagnostic evaluation only. No MRI registration or new inverse fits.\n'
        'diagnostic_distance_mm contains unregistered HEAD separation.\n'
        'localization_error_mm remains empty; localization_error_valid remains False.\n'
        'Sub-03 failed coregistration QA; its results form a separate exploratory cohort.\n'
        'All statuses remain in the run ledger and pass-rate denominators.\n'
        'See 07_evaluation_manifest.json for inputs, policy, and limitations.\n')
    print(f'Wrote {len(result)} rows; {passed.sum()} diagnostic distances; {(~passed).sum()} without estimates: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction-root', type=Path, default=EXP / 'outputs/05_reconstruct_sub03_exploratory')
    parser.add_argument('--ground-truth', type=Path, default=ROOT / 'cross_subject_no_warp/outputs/06_evaluate_localization/02_coordinate_validation.csv')
    parser.add_argument('--output-root', type=Path, default=EXP / 'outputs/07_evaluate_localization')
    args = parser.parse_args()
    evaluate(args.reconstruction_root, args.ground_truth, args.output_root)


if __name__ == '__main__':
    main()
