#!/usr/bin/env python3
# Registration QA sensitivity: separate pair tolerance .15 mm, source stays .1 mm.
# From project root, using the project's Python plus ANTs runtime:
#
# 1. Establish independent controls and investigate the six failures:
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_registration_qa_sensitivity.py --stage investigate
#
# 2. ONLY after attribution controls pass: rerun all 84 checks under both policies:
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_registration_qa_sensitivity.py --stage qa
#
# 3. ONLY after QA completes and reproduces primary flags: map sensitivity peaks:
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_registration_qa_sensitivity.py --stage evaluate
#
# 4. Compare matched rows, common cohort and newly eligible observations:
# python cross_subject_warp/scripts/06_registration_qa_sensitivity.py --stage compare
#
# Every output directory must be NEW; --output-root overrides this stage only.
# To consume alternate prior stages, also specify --attribution-root,
# --sensitivity-qa or --sensitivity-evaluation explicitly. No --overwrite exists.
# Exit 2 means a validation gate/execution check failed; inspect manifest.json.
# No EEG inverse reconstruction, registration fitting, threshold tuning or
# primary-output reclassification. Attribution may remain scientifically mixed.
"""Run the documented, post-hoc pair-QA015 sensitivity without changing primary.

Pair roundtrip <=.15 mm; source roundtrip <=.1 mm. All MRI, image reproduction,
coverage, coordinate and point-domain requirements remain unchanged. Existing
whole-brain peak sensitivity is a separate axis. No real-data analysis runs
unless a stage is explicitly requested. See REGISTRATION_QA_SENSITIVITY_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import tempfile
from qa_sensitivity import core,attribution,analysis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['investigate','qa','evaluate','compare'],required=True)
    outputs=core.EXP/'outputs'
    p.add_argument('--output-root',type=Path)
    p.add_argument('--primary-qa',type=Path,default=outputs/'06_registration_qa')
    p.add_argument('--primary-evaluation',type=Path,default=outputs/'06_evaluate_localization_mapped')
    p.add_argument('--diagnostics',type=Path,default=outputs/'06_roundtrip_diagnostics')
    p.add_argument('--bridge-root',type=Path,default=outputs/'06_coordinate_bridge')
    p.add_argument('--preparation-root',type=Path,default=outputs/'01_prepare_experiment_review')
    p.add_argument('--resource-root',type=Path,default=core.EXP/'source/01_registration_resources/outputs')
    p.add_argument('--attribution-root',type=Path,default=outputs/'06_registration_qa_attribution_domain_review')
    p.add_argument('--sensitivity-qa',type=Path,default=outputs/'06_registration_qa_sensitivity_rt015')
    p.add_argument('--sensitivity-evaluation',type=Path,default=outputs/'06_evaluate_localization_qa_rt015')
    args=p.parse_args()
    if args.output_root is None:
        args.output_root={'investigate':args.attribution_root,'qa':args.sensitivity_qa,'evaluate':args.sensitivity_evaluation,
                          'compare':outputs/'06_compare_registration_qa_thresholds'}[args.stage]
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.expanduser().resolve())
    return {'investigate':attribution.investigate,'qa':analysis.run_qa,'evaluate':analysis.evaluate,'compare':analysis.compare}[args.stage](args)


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='qa-sensitivity-config-') as config:
        os.environ['_MNE_FAKE_HOME_DIR']=config;os.environ['MPLCONFIGDIR']=config
        raise SystemExit(main())
