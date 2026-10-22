#!/usr/bin/env python3
"""Audit Localize-MI MNI resources without inventing spatial transforms.

This script only reads files. It does not register images, fit landmarks, map
SESAME sources to MNI, or interpret electrode MNI coordinates as a deformation.

Example
-------
python cross_subject/scripts/23_audit_mni_resources.py --donor sub-01 --target sub-02
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--target', default='sub-02')
    p.add_argument('--dataset-root', type=Path,
                   default=Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI')
    p.add_argument('--project-root', type=Path, default=Path.cwd())
    p.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/23_mni_resource_audit'))
    p.add_argument('--max-file-list', type=int, default=120)
    return p.parse_args()


def rel(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def discover_files(dataset_root, subject):
    """Search subject-specific anatomy and xfm paths plus MNI named resources."""
    roots = [dataset_root / 'derivatives' / 'epochs' / subject,
             dataset_root / 'derivatives' / 'sourcemodelling' / subject,
             dataset_root / subject]
    records = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            s = str(p).lower()
            suffix = p.suffix.lower()
            is_related = ('mni' in s or 'template' in s or 'transform' in s or
                          '/xfm/' in s or '/anat/' in s or 'electrodes.tsv' in s or
                          'coordsystem.json' in s)
            if is_related and (suffix in {'.nii', '.gz', '.h5', '.mat', '.txt', '.json', '.tsv', '.fif', '.gii', '.npy'}):
                records.append(p)
    return sorted(set(records), key=str)


def parse_electrodes(tsv):
    if not tsv.exists():
        return None
    with tsv.open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not {'name', 'x', 'y', 'z'}.issubset(reader.fieldnames or []):
            return {'error': 'Missing name/x/y/z columns', 'columns': reader.fieldnames}
        result = {}
        for row in reader:
            name = row['name'].strip()
            try:
                result[name] = [float(row[k]) * 1000.0 for k in ('x', 'y', 'z')]
            except (ValueError, TypeError):
                continue
        return result


def coordinate_summary(dataset_root, subject):
    ieeg = dataset_root / 'derivatives' / 'epochs' / subject / 'ieeg'
    prefix = f'{subject}_task-seegstim_space-'
    spaces = ('surface', 'T1w', 'mod-T1w_maskface', 'MNI152NLin2009aSym')
    found = {}
    coords = {}
    for space in spaces:
        tsv = ieeg / f'{prefix}{space}_electrodes.tsv'
        js = ieeg / f'{prefix}{space}_coordsystem.json'
        value = parse_electrodes(tsv)
        meta = None
        if js.is_file():
            try:
                meta = json.loads(js.read_text())
            except (ValueError, UnicodeError) as exc:
                meta = {'parse_error': str(exc)}
        found[space] = {
            'tsv': rel(tsv, dataset_root), 'tsv_exists': tsv.is_file(),
            'coordsystem_json': rel(js, dataset_root), 'coordsystem_exists': js.is_file(),
            'metadata': meta,
            'contact_count': len(value) if isinstance(value, dict) and 'error' not in value else None,
        }
        if isinstance(value, dict) and 'error' not in value:
            coords[space] = value
        elif value is not None:
            found[space]['parse_issue'] = value
    paired = []
    native = coords.get('T1w', {})
    mni = coords.get('MNI152NLin2009aSym', {})
    for contact in sorted(native.keys() & mni.keys()):
        a, b = np.asarray(native[contact], float), np.asarray(mni[contact], float)
        paired.append({'subject': subject, 'contact': contact,
                       'native_T1w_x_mm': float(a[0]), 'native_T1w_y_mm': float(a[1]), 'native_T1w_z_mm': float(a[2]),
                       'dataset_MNI_x_mm': float(b[0]), 'dataset_MNI_y_mm': float(b[1]), 'dataset_MNI_z_mm': float(b[2]),
                       'raw_difference_norm_mm_NOT_registration_error': float(np.linalg.norm(b - a))})
    return found, paired


def summarize_candidate_image(p):
    record = {'file': str(p), 'exists': True}
    try:
        import nibabel as nib
        img = nib.load(str(p))
        record['image_shape'] = list(img.shape)
        record['affine_RAS_mm'] = np.asarray(img.affine).tolist()
        record['axis_codes'] = list(nib.aff2axcodes(img.affine))
    except Exception as exc:
        record['read_error'] = str(exc)
    return record


def main():
    args = parse_args()
    project = args.project_root.expanduser().resolve()
    dataset = args.dataset_root.expanduser().resolve()
    output = args.output_root.expanduser()
    if not output.is_absolute():
        output = project / output
    output = output.resolve()
    if not dataset.is_dir():
        raise SystemExit(f'Dataset root not found: {dataset}\nUse --dataset-root PATH')
    if args.donor == args.target:
        raise SystemExit('Donor and target must differ.')
    output.mkdir(parents=True, exist_ok=True)

    report = {
        'status': 'RESOURCE_AUDIT_ONLY_NO_MNI_POINT_MAPPING',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_root': str(dataset), 'project_root': str(project),
        'donor': args.donor, 'target': args.target,
        'subjects': {}, 'available_template_like_files': [],
        'conclusions': [
            'MNI electrode TSVs define positions only for named SEEG contacts; they do NOT supply a transform for arbitrary SESAME sources.',
            'Do not fit a generic MNI transform to a handful of SEEG contacts and call it validated anatomical registration.',
            'The described MNI space is ICBM 2009a Nonlinear Symmetric, which is not automatically interchangeable with fsaverage or other MNI variants.',
            'A third-brain view requires a verified per-subject native MRI to the EXACT MNI template transform and template-compatible surface meshes.',
        ]
    }
    rows = []
    for subject in (args.donor, args.target):
        found, contact_pairs = coordinate_summary(dataset, subject)
        files = discover_files(dataset, subject)
        report['subjects'][subject] = {
            'coordinate_spaces': found,
            'matching_T1w_MNI_contact_count': len(contact_pairs),
            'matching_contact_names': [r['contact'] for r in contact_pairs],
            'discovered_relevant_files_total': len(files),
            'discovered_relevant_files': [rel(f, dataset) for f in files[:args.max_file_list]],
            'files_truncated': len(files) > args.max_file_list,
            'candidate_transform_files': [rel(f, dataset) for f in files if
                '/xfm/' in str(f).replace('\\', '/') or
                any(k in f.name.lower() for k in ('warp', 'affine', 'transform', 'syn', 'mni')) and
                f.suffix.lower() in ('.mat', '.h5', '.txt', '.fif', '.nii', '.gz')],
        }
        rows.extend(contact_pairs)

    # Template or normalization image discovery: filename inventory only, no huge MRI scans.
    candidate_dirs = [dataset / 'derivatives', dataset / 'sourcedata', dataset / 'stimuli', project / 'cross_subject']
    seen = set()
    for top in candidate_dirs:
        if not top.exists():
            continue
        for p in top.rglob('*'):
            if not p.is_file():
                continue
            name = p.name.lower()
            if not any(t in name for t in ('mni152', 'icbm', 'template', 'normaliz')):
                continue
            if p.suffix.lower() not in ('.nii', '.gz', '.gii', '.mat', '.h5', '.txt', '.json', '.fif'):
                continue
            if str(p) in seen:
                continue
            seen.add(str(p))
            report['available_template_like_files'].append(rel(p, dataset))
            if len(report['available_template_like_files']) >= 250:
                report['template_list_truncated'] = True
                break
        if report.get('template_list_truncated'):
            break

    report_path = output / 'mni_resources_audit.json'
    csv_path = output / 'native_T1w_to_dataset_MNI_contact_pairs.csv'
    txt_path = output / 'mni_audit_summary.txt'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    header = ('subject', 'contact', 'native_T1w_x_mm', 'native_T1w_y_mm', 'native_T1w_z_mm',
              'dataset_MNI_x_mm', 'dataset_MNI_y_mm', 'dataset_MNI_z_mm',
              'raw_difference_norm_mm_NOT_registration_error')
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        'SCRIPT 23: LOCALIZE-MI MNI RESOURCE AUDIT',
        'Status: inventory only; NO source-to-MNI transform estimated.',
        f'Donor: {args.donor}, Target: {args.target}',
    ]
    for sub in (args.donor, args.target):
        subreport = report['subjects'][sub]
        lines.append(f'\n{sub}:')
        for space, meta in subreport['coordinate_spaces'].items():
            lines.append(f"  {space}: contacts={meta['contact_count']}, TSV={meta['tsv_exists']}, metadata={meta['coordsystem_exists']}")
        lines.append(f"  Matched T1w/MNI contacts: {subreport['matching_T1w_MNI_contact_count']}")
        lines.append(f"  Candidate transformation files: {len(subreport['candidate_transform_files'])}")
        for item in subreport['candidate_transform_files'][:25]:
            lines.append('    ' + item)
    lines.append(f"\nTemplate-like resources found: {len(report['available_template_like_files'])}")
    for item in report['available_template_like_files'][:30]:
        lines.append('  ' + item)
    lines += ['', 'CAUTION: Dataset MNI contact positions are not a reusable transformation for arbitrary source points.',
              'Next: determine if complete native-to-MNI transforms exist or require a new registered, independently validated template normalization.']
    txt_path.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved:\n  {report_path}\n  {csv_path}\n  {txt_path}')


if __name__ == '__main__':
    main()
