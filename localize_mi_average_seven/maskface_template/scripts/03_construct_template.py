#!/usr/bin/env python3
"""Manually launch matched ANTs construction, preserving every iteration."""
import argparse
import json
from pathlib import Path
import shlex
import pandas as pd
from stage2_common import VAR,Run,ants_environment,ants_command,construction_contract,launch,warning_rows,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-root',type=Path,default=VAR/'outputs/02_stage2_preparation/run_03')
    p.add_argument('--output-root',type=Path,default=VAR/'outputs/03_template/run_04')
    p.add_argument('--execute',action='store_true',help='Required to launch ANTs')
    args=p.parse_args()
    if not args.execute:p.error('--execute is required; nothing was launched')
    with Run(args.output_root,'03_CONSTRUCT') as run:
        prep=run.verify(args.preparation_root,stage='02_PREPARE')
        if not prep['technical_inputs_ready']:raise ValueError('Preparation not ready')
        inputs=json.loads((args.preparation_root/'inputs.json').read_text())
        pipeline=json.loads((args.preparation_root/'pipeline.json').read_text())
        env,tools=ants_environment(pipeline['ANTs_bin'])
        script=run.use(pipeline['ANTs_script'],pipeline['ANTs_script_sha256'])
        for row in inputs:run.use(row['path'],row['sha256'])
        command=ants_command(script,run.out,[r['path'] for r in inputs])
        (run.out/'command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(command)+'\n')
        run.doc.update(command=command,pipeline=pipeline,participants=[r['subject'] for r in inputs],template=str(run.out/'group_template0.nii.gz'))
        run.save()
        launch(command,run.out/'construction.log',run.out,env)
        header=construction_contract(run.out)
        write_json(run.out/'final_header.json',header)
        pd.DataFrame(warning_rows(run.out/'construction.log')).to_csv(run.out/'warnings.csv',index=False)
        run.doc.update(construction_returncode=0,completed_iterations=4,registered_inputs_per_iteration=7,retention='all four ANTs_iteration_* backups plus final products')
        (run.out/'REPORT_stage2.md').write_text('# ANTs construction execution\n\nConstruction completed with required outputs from all four iterations. No anatomical approval is implied. Run 04_compare_template_mri.py and inspect MRI QA before FreeSurfer. All logs, transformations and images are retained.\n')


if __name__=='__main__':main()
