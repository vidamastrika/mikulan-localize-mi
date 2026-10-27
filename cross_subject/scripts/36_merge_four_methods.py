#!/usr/bin/env python3
"""Merge new irMxNE/Gamma-MAP with existing Champagne and SESAME evaluations.

Run from the repository root. Inputs are never modified. All statuses and
original CSV values are preserved. SESAME receives an explicit method label.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('cross_subject/outputs')
    defaults = {
        'irmxne': base / '36_final_irmxne_all_pass/02_localization_errors.csv',
        'gamma': base / '36_gamma_final_all_pass/02_localization_errors.csv',
        'champagne': base / '36_cross_subject_irmxne_gammamap_champagne/02_localization_errors.csv',
        'sesame': base / '32_all_reference_localization_evaluation/cross_subject_localization_errors.csv',
    }
    for name, path in defaults.items():
        p.add_argument('--' + name, type=Path, default=path)
    p.add_argument('--output', type=Path, default=base / '36_four_methods_combined/02_localization_errors.csv')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    sources = [('irmxne', 'irMxNE'), ('gamma', 'Gamma-MAP'),
               ('champagne', 'Champagne'), ('sesame', 'SESAME')]
    keys = ['test_subject', 'reference_subject', 'run']
    columns, combined, expected = ['method'], [], None
    for flag, method in sources:
        path = getattr(args, flag)
        if args.output.resolve() == path.resolve():
            p.error('Output cannot replace an input file.')
        if not path.is_file():
            p.error(f'Missing {path}; supply --{flag} PATH to use another location.')
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            required = keys + ['status', 'failure_reason', 'within_subject_error_mm',
                               'cross_affine_error_mm', 'cross_syn_error_mm']
            missing = set(required) - set(fields)
            if missing:
                p.error(f'{path}: missing columns {sorted(missing)}')
            rows = list(reader)
        if 'method' in fields:
            rows = [r for r in rows if r['method'].strip().casefold() == method.casefold()]
        elif method != 'SESAME':
            p.error(f'{path}: missing method column')
        case_keys = [tuple(r[k] for k in keys) for r in rows]
        if len(rows) != 366 or len(set(case_keys)) != 366:
            p.error(f'{method}: expected 366 unique cases, found {len(rows)} rows and {len(set(case_keys))} unique cases')
        if any(not all(k) or k[0] == k[1] for k in case_keys):
            p.error(f'{method}: blank identifiers or within-subject rows found')
        if expected is None:
            expected = set(case_keys)
        elif set(case_keys) != expected:
            p.error(f'{method}: case identifiers differ from the first method')
        for col in fields:
            if col not in columns:
                columns.append(col)
        for row in rows:
            row['method'] = method
            row['merge_source_file'] = str(path)
        combined.extend(rows)
        print(f'{method}: {len(rows)} rows; {dict(Counter(r["status"] for r in rows))}')
    if 'merge_source_file' not in columns:
        columns.append('merge_source_file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open('w' if args.overwrite else 'x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(combined)
    except FileExistsError:
        p.error(f'{args.output} already exists; use --overwrite to replace it.')
    print(f'Saved {len(combined)} rows to {args.output}')


if __name__ == '__main__':
    main()
