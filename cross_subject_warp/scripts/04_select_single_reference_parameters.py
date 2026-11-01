#!/usr/bin/env python3
# SCRIPT 04 — Select parameters using only each reference subject's own runs.
# Run from the project root with .venv active.
#
# Normal run, using the reviewed preparation audit:
# python cross_subject_warp/scripts/04_select_single_reference_parameters.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# Repeat into a NEW folder without overwriting earlier selections:
# python cross_subject_warp/scripts/04_select_single_reference_parameters.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --output-root cross_subject_warp/outputs/04_single_reference_parameters_review
#
# Help only (no selection):
# python cross_subject_warp/scripts/04_select_single_reference_parameters.py --help
#
# Expected: 84 reference/method selections and 366 reference/test/run cases.
# Exit 0: all selections complete; nonzero: inspect coverage/error output.
# No registration, inverse fitting, source-point import or anatomical evaluation.
"""Reference-only selection from pinned historical within-subject parameter grids.

Reuses Script 02's normalization, repair, duplicate handling, strict all-PASS
reliability and ranking. Only the reference subject's rows enter these steps.
With one subject, score is its median error; ties: mean error, configuration key.
Affine and SyN share parameters and native inverse fits. This stage is independent
of Script 03's reuse audit and does not resolve physical-coordinate validation.

Numbered artifacts 01–09 retain historical names under
cross_subject_warp/outputs/04_single_reference_parameters/.
Frozen historical grids/manifests and original outputs are never overwritten.
Primary and whole-brain sensitivity policies remain separate.
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

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'

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
    p.add_argument('--source-root',type=Path,default=EXP/'source/02_parameter_grids')
    p.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment')
    p.add_argument('--output-root',type=Path,default=EXP/'outputs/04_single_reference_parameters')
    p.add_argument('--reliability',choices=['primary-all-pass','configuration-all-pass'],default='primary-all-pass')
    p.add_argument('--error-column',action='append',default=[],help='METHOD=COLUMN if error column ambiguous (same as Script 02)')
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().resolve())
    out=args.output_root
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Output exists: {out}. Use a NEW --output-root; overwrites are refused.')
    for protected in [args.source_root,args.preparation_root,EXP/'source',
                      ROOT/'cross_subject_no_warp',ROOT/'cross_subject']:
        if out==protected or protected in out.parents or out in protected.parents:
            raise ValueError(f'Output overlaps protected input tree: {protected}')
    mod=import_shared(METHODS_02)
    root,doc=mod.snapshot(args)
    manifest=root/'source_manifest.json'
    preparation=mod.verify_preparation(args.preparation_root,root,doc)
    folds,cases=mod.load_plan(root,doc)
    subjects=sorted(set(cases.test_subject))
    if subjects != [f'sub-{i:02d}' for i in range(1,8)]:raise ValueError('Expected subjects sub-01..sub-07')
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'IN_PROGRESS','expected_selections':84,'source_root':str(root),
           'preparation':preparation,'reliability':args.reliability,
           'inverse_reconstruction_executed':False,'registration_fitting_executed':False,
           'peaks_imported':False,'anatomical_evaluation_ready':False}
    (out/'07_selection_manifest.json').write_text(json.dumps(state,indent=2)+'\n')
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
    if len({(r['reference_subject'],r['test_subject']) for r in schedule})!=42:
        raise AssertionError('Expected 42 directed non-self subject pairs')
    if len(schedule)!=366:raise AssertionError(f'Expected 366 directed run-reference pairs, found {len(schedule)}')
    pd.DataFrame(schedule).to_csv(out/'09_reference_test_schedule.csv',index=False)
    complete=len(selected)==len(mod.METHODS)*len(subjects)
    result={**state,'status':'COMPLETE' if complete else 'INCOMPLETE','expected_selections':84,
            'actual_selections':len(selected),'source_manifest_sha256':mod.digest(manifest),
            'script02_sha256':mod.digest(METHODS_02),'script04_sha256':mod.digest(__file__),
            'reliability':args.reliability,'score':'reference-subject median localization error',
            'tie_breaks':['reference-subject mean','configuration key'],
            'reference_subjects':subjects,'methods':mod.METHODS,
            'excluded_test_outcomes':True,'parameter_source':'historical within-subject grids',
            'historical_montage_template':'fixed sub-01',
            'registration_modes':['affine','syn'],
            'reference_test_schedule_sha256':mod.digest(out/'09_reference_test_schedule.csv'),
            'historical_script04_sha256':mod.digest(EXP/'source/00_no_warp_workflow/scripts/04_select_single_reference_parameters.py'),
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
