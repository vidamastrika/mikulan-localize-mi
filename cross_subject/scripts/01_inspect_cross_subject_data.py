#!/usr/bin/env python3
"""Read-only inventory of Localize-MI files for cross-subject anatomical evaluation.

This is NOT a coregistration script. It identifies files and metadata that must
be checked before transferring a head model between EEG participants.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATASET = Path.home() / 'Documents' / 'DATASETS SOURCE LOCALIZATION' / 'Localize-MI'
KEYWORDS = {
    'epochs': ('epoch',),
    'forward': ('-fwd', '_fwd', 'forward'),
    'transform': ('-trans', '_trans', 'transform'),
    'bem': ('bem', 'inner_skull', 'outer_skull', 'outer_skin'),
    'source_space': ('-src', '_src', 'source_space'),
    'electrodes': ('electrode', 'channel', 'digitiz', 'montage'),
    'ground_truth': ('stim', 'target', 'ground_truth', 'groundtruth'),
    'mri_surface': ('pial', 'inflated', 'white', 'sphere'),
}
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'outputs'}


def subjects_for_file(path: Path) -> set[str]:
    return {part for part in path.parts if part.startswith('sub-') and part[4:].isdigit()}


def category(path: Path) -> str:
    name = path.name.lower()
    for key, tokens in KEYWORDS.items():
        if any(token in name for token in tokens):
            return key
    return 'other'


def basic_metadata(path: Path, *, detailed_fif: bool) -> dict[str, Any]:
    info: dict[str, Any] = {'relative_path': str(path), 'type': category(path), 'extension': path.suffix.lower()}
    try:
        info['bytes'] = path.stat().st_size
        if path.suffix.lower() == '.npy':
            arr = np.load(path, mmap_mode='r', allow_pickle=False)
            info['shape'] = list(arr.shape)
            info['dtype'] = str(arr.dtype)
        elif path.suffix.lower() == '.tsv':
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.reader(handle, delimiter='\t')
                info['columns'] = next(reader, [])
        elif path.suffix.lower() == '.fif' and detailed_fif:
            import mne
            if 'epo' in path.name.lower():
                ep = mne.read_epochs(path, preload=False, verbose='ERROR')
                info.update({'n_channels': len(ep.ch_names), 'sfreq': float(ep.info['sfreq']),
                             'n_eeg': len(mne.pick_types(ep.info, eeg=True, exclude=[])),
                             'n_dig_points': len(ep.info['dig'] or []),
                             'n_bads': len(ep.info['bads'])})
            elif ('raw' in path.name.lower() or 'eeg' in path.name.lower()) and 'fwd' not in path.name.lower():
                raw = mne.io.read_raw_fif(path, preload=False, verbose='ERROR')
                info.update({'n_channels': len(raw.ch_names), 'sfreq': float(raw.info['sfreq']),
                             'n_eeg': len(mne.pick_types(raw.info, eeg=True, exclude=[])),
                             'n_dig_points': len(raw.info['dig'] or []),
                             'n_bads': len(raw.info['bads'])})
    except Exception as exc:
        info['inspection_error'] = f'{type(exc).__name__}: {exc}'
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/01_inspection'))
    ap.add_argument('--detailed-fif', action='store_true', help='Read FIF measurement metadata (slower)')
    args = ap.parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        ap.error(f'Dataset directory not found: {root}. Supply --dataset-root PATH')
    rows = []
    subject_ids: set[str] = set()
    for file in root.rglob('*'):
        if not file.is_file() or any(p in SKIP_DIRS for p in file.relative_to(root).parts):
            continue
        sub_ids = subjects_for_file(file.relative_to(root))
        subject_ids.update(sub_ids)
        item = basic_metadata(file, detailed_fif=args.detailed_fif)
        item['relative_path'] = str(file.relative_to(root))
        item['subjects'] = sorted(sub_ids)
        rows.append(item)
    rows.sort(key=lambda r: r['relative_path'])
    counts = Counter((sub, row['type']) for row in rows for sub in row['subjects'])
    sub_summary = [{'subject': sub, **{k: counts.get((sub, k), 0) for k in KEYWORDS}} for sub in sorted(subject_ids)]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'file_inventory.json').write_text(json.dumps({'dataset_root': str(root), 'files': rows}, indent=2), encoding='utf-8')
    with (output / 'subject_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', *KEYWORDS.keys()])
        writer.writeheader()
        writer.writerows(sub_summary)
    report = [f'Dataset: {root}', f'Files inspected: {len(rows)}', f'Subject folders: {", ".join(sorted(subject_ids)) or "NONE"}', '',
              'Counts below are filename-based and do not prove model compatibility.', '']
    for entry in sub_summary:
        report.append(entry['subject'] + ': ' + ', '.join(f'{k}={entry[k]}' for k in KEYWORDS))
    report.extend(['', 'Before cross-subject source localization:',
                   '1. Identify measured electrode locations and their coordinate frame.',
                   '2. Identify each MRI-to-head transform and verify its landmarks.',
                   '3. Check BEM/source-space units, frame, and surface validity.',
                   '4. Define a registration that maps EEG-head coordinates to donor MRI coordinates.',
                   '5. Define how donor-space estimates map into the recipient ground-truth frame.',
                   '6. Never use a mismatched forward FIF without recomputation.'])
    (output / 'inspection_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))
    print(f'\nSaved: {output / "file_inventory.json"}')
    print(f'Saved: {output / "subject_summary.csv"}')
    print(f'Saved: {output / "inspection_report.txt"}')


if __name__ == '__main__':
    main()
