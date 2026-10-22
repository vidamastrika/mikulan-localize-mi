#!/usr/bin/env python3
"""Script 24 — locate and verify the EXACT MNI152NLin2009aSym T1w template.

Read-only by default. --fetch permits TemplateFlow to obtain the exact template
(and optionally its brain mask). No registration or source transformation occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

TEMPLATE = 'MNI152NLin2009aSym'
PREFIX = f'tpl-{TEMPLATE}'


def fmt(p):
    return str(p.resolve()) if p else None


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def collect(root):
    if not root or not root.exists():
        return []
    if root.is_file():
        return [root] if PREFIX in root.name else []
    # Search only within a specific template directory where possible.
    if root.name == PREFIX:
        return list(root.glob(f'{PREFIX}*.nii*'))
    direct = root / PREFIX
    if direct.is_dir():
        return list(direct.glob(f'{PREFIX}*.nii*'))
    return list(root.glob(f'{PREFIX}*.nii*'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fetch', action='store_true', help='Allow TemplateFlow to download exact template to its cache')
    ap.add_argument('--template-dir', type=Path, default=None, help='Additional path containing template files')
    ap.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/24_mni_template_verification'))
    args = ap.parse_args()

    roots = [
        args.template_dir,
        Path(os.environ.get('TEMPLATEFLOW_HOME', str(Path.home() / '.cache/templateflow'))),
        Path.home() / '.cache/templateflow',
        Path.home() / 'templateflow',
        Path('cross_subject/templates'),
        Path('cross_subject/data/templates'),
    ]
    unique = {str(p.resolve()): p for p in roots if p is not None}
    candidates = {str(p.resolve()): p for root in unique.values() for p in collect(root)}
    tf_meta = None
    tf_error = None
    if args.fetch:
        try:
            from templateflow import api as tf
            tf_meta = tf.get_metadata(TEMPLATE)
            retrieved = tf.get(TEMPLATE, resolution=1, suffix='T1w', raise_empty=True)
            for p in (retrieved if isinstance(retrieved, (tuple, list)) else [retrieved]):
                if p:
                    p = Path(p)
                    candidates[str(p.resolve())] = p
            try:
                mask = tf.get(TEMPLATE, resolution=1, desc='brain', suffix='mask', raise_empty=False)
                for p in (mask if isinstance(mask, (tuple, list)) else [mask]):
                    if p:
                        p = Path(p)
                        candidates[str(p.resolve())] = p
            except Exception:
                pass
        except Exception as exc:
            tf_error = f'{type(exc).__name__}: {exc}'

    report = {
        'status': 'NO_EXACT_TEMPLATE_FOUND',
        'target_template_id': TEMPLATE,
        'purpose': 'Template identity, geometry, and integrity audit; no registration',
        'fetch_allowed': args.fetch,
        'templateflow_metadata': tf_meta,
        'templateflow_error': tf_error,
        'examined_roots': [fmt(p) for p in unique.values()],
        'files': [],
        'selected_reference_t1w': None,
        'warnings': [
            'A matching filename is not proof this is the exact template version used by the Localize-MI authors.',
            'MNI152NLin2009aSym is different from MNI152NLin2009aAsym, 2009cAsym, and fsaverage.',
            'Native-to-MNI anatomical registration and independent contact comparison must follow before plotting sources.',
        ],
    }
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        print(f'Install nibabel and numpy in your project .venv: {exc}', file=sys.stderr)
        raise SystemExit(2)

    valid_t1w = []
    for p in sorted(candidates.values(), key=str):
        entry = {'file': fmt(p), 'name': p.name, 'exists': p.is_file()}
        if not p.is_file():
            report['files'].append(entry)
            continue
        try:
            entry['size_bytes'] = p.stat().st_size
            entry['is_exact_template_filename'] = p.name.startswith(f'{PREFIX}_')
            entry['is_t1w'] = p.name.endswith('_T1w.nii.gz') or p.name.endswith('_T1w.nii')
            entry['resolution_01'] = ('_res-1_' in p.name or '_res-01_' in p.name)
            img = nib.load(str(p))
            if len(img.shape) != 3:
                raise ValueError(f'Not 3D: shape={img.shape}')
            entry['shape'] = list(img.shape)
            entry['voxel_sizes_mm'] = [float(x) for x in img.header.get_zooms()[:3]]
            entry['affine_RAS_mm'] = np.asarray(img.affine).tolist()
            entry['axis_codes'] = list(nib.aff2axcodes(img.affine))
            entry['qform_code'] = int(img.header['qform_code'])
            entry['sform_code'] = int(img.header['sform_code'])
            entry['sha256'] = checksum(p)
            entry['readable'] = True
            if entry['is_exact_template_filename'] and entry['is_t1w'] and entry['resolution_01']:
                valid_t1w.append(p)
        except Exception as exc:
            entry['readable'] = False
            entry['error'] = str(exc)
        report['files'].append(entry)

    if valid_t1w:
        chosen = sorted(valid_t1w, key=str)[0]
        report['selected_reference_t1w'] = fmt(chosen)
        report['status'] = 'CANDIDATE_TEMPLATE_READY_FOR_REVIEW'
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / 'mni_template_verification.json'
    report_path.write_text(json.dumps(report, indent=2, default=str) + '\n')
    summary = [
        'SCRIPT 24 — EXACT MNI TEMPLATE AUDIT',
        f'Template: {TEMPLATE}',
        f'Status: {report["status"]}',
        f'Candidate files: {len(report["files"])}',
        f'Selected T1w: {report["selected_reference_t1w"] or "NONE"}',
        f'TemplateFlow error: {tf_error or "none"}',
        '',
        'Not yet validated: equivalence to the authors\' original MNI normalization.',
        'No registration or source localization metrics were computed.',
    ]
    (args.output_root / 'mni_template_verification_summary.txt').write_text('\n'.join(summary) + '\n')
    print('\n'.join(summary))
    print('Report:', report_path)
    if not valid_t1w:
        print('\nTo obtain the exact template (if network access is available):')
        print('  .venv/bin/python -m pip install templateflow')
        print('  .venv/bin/python cross_subject/scripts/24_locate_verify_mni_template.py --fetch')


if __name__ == '__main__':
    main()
