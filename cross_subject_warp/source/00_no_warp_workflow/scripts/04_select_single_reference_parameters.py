#!/usr/bin/env python3
"""Script 04 — Single-reference parameter selection (Localize-MI; no warp).

PURPOSE
    Select one historical within-subject parameter configuration per
    (reference subject, inverse method), using ONLY runs belonging to that
    reference. The selected parameters will be transferred unchanged to EEG
    recordings from all six other subjects in Script 05.

DATA AND PROVENANCE
    Reads the immutable source/02_parameter_grids snapshot created by Script 02,
    including its source_manifest.json and Script 01's 61-run schedule.
    Imports Script 02's SAME normalization, reliability, configuration repair,
    duplicate-conflict handling, and selection routine. Every snapshot file is
    checksum-verified. Source grids are NOT modified or copied again.
    Crucially, training rows are filtered to the reference subject BEFORE
    any missing-setting repair, scoring, or eligibility evaluation.

SELECTION POLICY (per method and reference)
    Default strict primary-all-pass rule from Script 02: an eligible
    configuration must cover ALL runs of the reference subject, have only
    finite nonnegative PASS errors, no conflicting duplicates, and pass the
    primary parameter-value reliability screen. If no configuration qualifies,
    record the absence and exit nonzero; DO NOT substitute global/LOSO values.
    Score = median localization error across reference subject's runs.
    Ties: mean error, then deterministic configuration key.
    With only one training subject this is equivalent to Script 02's score.

    Each method's grid varies its own settings, including montage; channel
    count is part of the selected configuration. Historical parameter grids
    used the fixed sub-01 montage-template policy. Script 05 instead uses
    reference-derived montage selection. These transferred parameters are NOT
    claimed optimal for the new montage geometries.

EXAMPLE TERMINAL COMMANDS (from project root with .venv active)
    python cross_subject_no_warp/scripts/04_select_single_reference_parameters.py
    python cross_subject_no_warp/scripts/04_select_single_reference_parameters.py \\
        --output-root cross_subject_no_warp/outputs/04_single_reference_parameters
    # Explicitly relax primary-value screening (separate sensitivity analysis):
    python cross_subject_no_warp/scripts/04_select_single_reference_parameters.py \\
        --reliability configuration-all-pass \\
        --output-root cross_subject_no_warp/outputs/04_single_reference_parameters_relaxed

OUTPUTS (only created when output folder is empty)
    01_single_reference_selected_parameters.csv  (84 expected)
    02_reference_selection_coverage.csv           (all 84 selections)
    03_reference_configuration_audit.csv          (all candidates)
    04_reference_primary_value_audit.csv
    05_reference_duplicate_audit.csv
    06_parameter_fields.json
    07_selection_manifest.json (COMPLETE only if all 84 selected)
    08_source_manifest.json (byte-for-byte copy of Script02 manifest)
    09_reference_test_schedule.csv (366 reference-run combinations)
    README.txt

Does not run EEG inverse solvers, compute new localization errors, or change
Script 03's LOSO results. Every test subject is strictly excluded from the
parameter-selection subset for the corresponding reference.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import pandas as pd

METHODS_02 = Path(__file__).with_name('02_select_loso_parameters.py')


def import_shared(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Script 02 is required alongside Script 04: {path}')
    spec = importlib.util.spec_from_file_location('loso_shared_selection', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root',type=Path,default=Path('cross_subject_no_warp/source/02_parameter_grids'))
    p.add_argument('--script02',type=Path,default=METHODS_02)
    p.add_argument('--output-root',type=Path,default=Path('cross_subject_no_warp/outputs/04_single_reference_parameters'))
    p.add_argument('--reliability',choices=['primary-all-pass','configuration-all-pass'],default='primary-all-pass')
    p.add_argument('--error-column',action='append',default=[],help='METHOD=COLUMN if error column ambiguous (same as Script 02)')
    args=p.parse_args()
    mod=import_shared(args.script02)
    root=args.source_root.resolve()
    manifest=root/'source_manifest.json'
    if not manifest.is_file():raise FileNotFoundError(f'Run Script 02 to create the source snapshot: {manifest}')
    doc=json.loads(manifest.read_text())
    for item in doc['files']:
        file=root/item['copy']
        if not file.is_file() or mod.digest(file)!=item['sha256']:
            raise RuntimeError(f'Source snapshot changed or missing: {file}')
    folds,cases=mod.load_plan(root,doc)
    subjects=sorted(set(cases.test_subject))
    if subjects != [f'sub-{i:02d}' for i in range(1,8)]:raise ValueError('Expected subjects sub-01..sub-07')
    out=args.output_root.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}. Use a NEW folder for a new selection.')
    out.mkdir(parents=True,exist_ok=True)
    schema=[];selected=[];coverage=[];audits=[];screens=[];duplicates=[];cache={}
    overrides=mod.kv(args.error_column)
    for method in mod.METHODS:
        print('Selecting',method,flush=True)
        data,error_col=mod.load_method(root,doc,method,cache,overrides.get(method,[None])[0])
        for reference in subjects:
            expected=set(map(tuple,cases.loc[cases.test_subject.eq(reference),['test_subject','run']].values))
            if not expected:raise ValueError(f'No runs for {reference}')
            # select_fold() uses only reference rows, PRIOR to parameter repair.
            winner,audit,screen,dups,fields=mod.select_fold(data,[reference],expected,method,args.reliability)
            context={'method':method,'reference_subject':reference,'training_subjects':reference}
            for col,value in context.items():audit[col]=value
            audits.append(audit)
            screens.extend({**r,**context} for r in screen)
            duplicates.extend({**r,**context} for r in dups)
            schema.append({**context,'parameter_fields':fields,'error_column':error_col})
            coverage.append({**context,'expected_reference_runs':len(expected),
                             'candidate_configurations':len(audit),
                             'eligible_configurations':int(audit.eligible.sum()),
                             'selection_status':'SELECTED' if winner else 'NO_ELIGIBLE_CONFIGURATION'})
            if winner:
                selected.append({**context,**winner,'n_training_subjects':1,
                                 'selection_status':'SELECTED','score_reference_median_mm':winner['score_mean_subject_median_mm']})
            print(f'  {reference}: {int(audit.eligible.sum())}/{len(audit)} eligible',flush=True)
        later={path for m in mod.METHODS[mod.METHODS.index(method)+1:] for path in doc['methods'][m]}
        cache={path:frame for path,frame in cache.items() if path in later}
    pd.DataFrame(selected).to_csv(out/'01_single_reference_selected_parameters.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'02_reference_selection_coverage.csv',index=False)
    pd.concat(audits,ignore_index=True).to_csv(out/'03_reference_configuration_audit.csv',index=False)
    pd.DataFrame(screens).to_csv(out/'04_reference_primary_value_audit.csv',index=False)
    pd.DataFrame(duplicates,columns=['method','configuration_key','subject','run','rows','conflict','reference_subject','training_subjects']).to_csv(out/'05_reference_duplicate_audit.csv',index=False)
    (out/'06_parameter_fields.json').write_text(json.dumps(schema,indent=2)+'\n')
    # A run of a given test subject is paired with every OTHER reference.
    schedule=[{'reference_subject':reference,'test_subject':r.test_subject,'run':r.run}
              for reference in subjects for r in cases.itertuples() if r.test_subject!=reference]
    if len(schedule)!=366:raise AssertionError(f'Expected 366 directed run-reference pairs, found {len(schedule)}')
    pd.DataFrame(schedule).to_csv(out/'09_reference_test_schedule.csv',index=False)
    complete=len(selected)==len(mod.METHODS)*len(subjects)
    result={'status':'COMPLETE' if complete else 'INCOMPLETE','expected_selections':84,
            'actual_selections':len(selected),'source_manifest_sha256':mod.digest(manifest),
            'script02_sha256':mod.digest(args.script02),'script04_sha256':mod.digest(__file__),
            'reliability':args.reliability,'score':'reference-subject median localization error',
            'tie_breaks':['reference-subject mean','configuration key'],
            'reference_subjects':subjects,'methods':mod.METHODS,
            'excluded_test_outcomes':True,'parameter_source':'historical within-subject grids',
            'historical_montage_template':'fixed sub-01',
            'intended_reconstruction_montage_template':'reference subject',
            'inherited_hemisphere_peak_policy':True,
            'selected_csv_sha256':mod.digest(out/'01_single_reference_selected_parameters.csv'),
            'warnings':['Historical grid/search-space development involved the same dataset; not independent prospective validation.',
                        'Parameter selection based on historical montage policy, not newly optimized for reference-derived montage.']}
    (out/'07_selection_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copyfile(manifest,out/'08_source_manifest.json')
    (out/'README.txt').write_text(__doc__)
    print(f'Selected {len(selected)}/84 reference-method configurations; {len(schedule)} cross-subject run/reference jobs per method.')
    print('Outputs:',out)
    if not complete:raise SystemExit('INCOMPLETE: no fallback used. Inspect 02_reference_selection_coverage.csv before Script05.')

if __name__=='__main__':main()
