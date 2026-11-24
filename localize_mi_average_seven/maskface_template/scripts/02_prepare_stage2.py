#!/usr/bin/env python3
"""Verify reviewed inputs and record original pipeline/settings; no reconstruction."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
from input_utils import image_header
from stage2_common import ROOT,EXP,VAR,OLD,OLD_FS,ANTS_BIN,STAMP,Run,ants_environment,small_command,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1',type=Path,default=VAR/'outputs/01_input_feasibility/run_01')
    p.add_argument('--intensity-review',type=Path,default=VAR/'outputs/01b_intensity_review/run_01')
    p.add_argument('--ants-bin',type=Path,default=ANTS_BIN)
    p.add_argument('--output-root',type=Path,default=VAR/'outputs/02_stage2_preparation/run_03')
    args=p.parse_args()
    with Run(args.output_root,'02_PREPARE') as run:
        run.verify(args.stage1);run.verify(args.intensity_review)
        for path in [VAR/'REVIEW_stage1_intensity_corrected.md',VAR/'REVIEW_stage1_run_01.md',VAR/'config/variant.json',EXP/'config/experiment.json',
            OLD/'01_build_template.sh',OLD/'03_construction.log',OLD/'04_execution_manifest.json',OLD_FS/'scripts/recon-all.log',OLD_FS/'scripts/recon-all.cmd',OLD_FS/'scripts/build-stamp.txt']:
            run.use(path)
        variant=json.loads((VAR/'config/variant.json').read_text());original=json.loads((EXP/'config/experiment.json').read_text())
        stage2=json.loads((VAR/'config/stage2.json').read_text())
        if not stage2.get('user_authorized') or stage2.get('stage')!='02_TEMPLATE_AND_ANATOMICAL_QA':
            raise ValueError('Stage 2 authorization/scope record is missing or invalid')
        if variant.get('stage2_authorized') is not False:
            raise ValueError('Stage 1 configuration was changed; retain it as historical evidence')
        if original['coregistration']['median_limit_mm']!=10 or original['coregistration']['p95_limit_mm']!=20 or original['coregistration']['scaling']:
            raise ValueError('Original alignment contract changed')
        env,tools=ants_environment(args.ants_bin)
        for path in tools.values():run.use(path)
        script=Path(tools['antsMultivariateTemplateConstruction2.sh'])
        (run.out/'source').mkdir();shutil.copy2(script,run.out/'source'/script.name)
        text=script.read_text()
        keys=['STATSMETHOD','SHARPENMETHOD','USEFLOAT','MAXITERATIONS','SMOOTHINGFACTORS','SHRINKFACTORS','N4CORRECT','DOLINEAR','GRADIENTSTEP','RIGID','AFFINE_UPDATE_FULL']
        defaults={key:re.search(r'^'+key+r'=(.+)$',text,re.M).group(1) for key in keys}
        expected=dict(STATSMETHOD='1',SHARPENMETHOD='1',USEFLOAT='1',MAXITERATIONS='100x100x70x20',SMOOTHINGFACTORS='3x2x1x0',SHRINKFACTORS='6x4x2x1',N4CORRECT='1',DOLINEAR='1',GRADIENTSTEP='0.25',RIGID='0',AFFINE_UPDATE_FULL='1')
        if defaults!=expected:raise ValueError(f'Installed ANTs script defaults differ from inspected baseline: {defaults}')
        inventory=pd.read_csv(args.stage1/'01_input_inventory.csv');rows=[]
        ants_input_dir=run.out/'ants_inputs';ants_input_dir.mkdir()
        for subject in variant['participants']:
            match=inventory[(inventory.subject==subject)&(inventory.version=='maskface')]
            if len(match)!=1:raise ValueError(f'Maskface input missing/duplicated: {subject}')
            row=match.iloc[0];path=run.use(row.path,row.sha256);image=nib.load(path);header=image_header(image)
            if not header['technical_header_valid'] or not np.isfinite(np.asarray(image.dataobj)).all():raise ValueError(f'Invalid maskface MRI: {subject}')
            py_match=inventory[(inventory.subject==subject)&(inventory.version=='Pydeface')]
            if len(py_match)!=1:raise ValueError(f'Paired Pydeface input missing/duplicated: {subject}')
            py_row=py_match.iloc[0]
            py_path=run.use(py_row.path,py_row.sha256)
            # The released dataset's parent directory contains spaces. The
            # ANTs shell pipeline does not quote every input path consistently,
            # so stage byte-identical copies under this whitespace-free path.
            staged=ants_input_dir/f'{subject}_maskface_T1w.nii'
            shutil.copy2(path,staged)
            staged_path=run.use(staged)
            staged_sha=run.inputs[str(staged_path)]
            if staged_sha!=row.sha256:raise ValueError(f'Staged ANTs input differs from source: {subject}')
            rows.append(dict(subject=subject,path=str(staged),sha256=staged_sha,
                source_path=str(path),source_sha256=row.sha256,header=header,
                processing_metadata=json.loads(row.processing_metadata_json),
                Pydeface_reference=dict(path=str(py_path),sha256=py_row.sha256,
                    processing_metadata=json.loads(py_row.processing_metadata_json)),
                staging='byte-identical copy to whitespace-free path for ANTs shell argument compatibility',
                intensity_policy='raw released input; NIfTI storage scaling applied by readers only; no fitted diagnostic calibration'))
        if (OLD_FS/'scripts/build-stamp.txt').read_text().strip()!=STAMP:raise ValueError('Original FreeSurfer build evidence changed')
        write_json(run.out/'inputs.json',rows)
        write_json(run.out/'pipeline.json',dict(original_command=(OLD/'01_build_template.sh').read_text(),
            original_FreeSurfer_command='recon-all -sd ORIGINAL_SUBJECTS -s average_seven -i ORIGINAL_TEMPLATE -all',
            new_FreeSurfer_command='recon-all -sd NEW_SUBJECTS -s average_seven_maskface -i NEW_TEMPLATE -all',
            ANTs_defaults=defaults,ANTs_script=str(script),ANTs_script_sha256=run.inputs[str(script.resolve())],ANTs_bin=str(args.ants_bin.resolve()),
            ANTs_version=small_command([tools['antsRegistration'],'--version'],env=env),
            retention_only_change='-b 1 retains every iteration; original default -b 0 only retained template updates and final inputs',
            averaging='mean of normalized input intensities (original internal ANTs procedure)',
            interpolation='LinearInterpolateImageFunction in original log; new log audited downstream',
            image_grid_and_FOV='original log: bigimage 4 maxSize [256,256,256], output physical space based on largest image; no explicit crop option in original command',
            bias_correction='ANTs N4Correct=1; preserve installed default and log its execution',
            FreeSurfer_build=STAMP,watershed='original MNE defaults: mri_watershed -useSRAS -surf PREFIX T1.mgz ws.mgz; no -atlas/-T1 correction variants',
            reproducibility_limit='Original ANTs executable/script hashes were not recorded then; installed script/defaults and original log are pinned now. Exact original executable identity is not independently proved.'))
        evidence=[]
        for path in [OLD/'03_construction.log',OLD_FS/'scripts/recon-all.log']:
            for i,line in enumerate(path.read_text(errors='replace').splitlines(),1):
                if re.search(r'Interpolation type:|Sharpening|Laplacian|Registration using|Averaging 7|shapeupdate|recon-all -|build-stamp|--conform|bigimage|maxSize|Setting physical space|Do N4|N4BiasFieldCorrection|cropp|padd',line,re.I):
                    evidence.append(dict(path=str(path),line=i,text=line.strip()))
        pd.DataFrame(evidence).to_csv(run.out/'original_pipeline_evidence.csv',index=False)
        run.doc.update(technical_inputs_ready=True,participants=variant['participants'],stage2_scope=stage2['scope'],raw_maskface_inputs=True)
        (run.out/'REPORT_stage2.md').write_text('# Stage 2 preparation\n\nAll seven reviewed maskface inputs verified and copied byte-for-byte to a whitespace-free staging directory because the ANTs shell script does not consistently preserve spaces in input paths. Source and staged SHA-256 values match. Original settings and installed ANTs script pinned. Only ANTs backup retention changes; no fitted intensity gains applied. No registration or reconstruction was executed. Read pipeline.json and follow the repository-root STAGE2_RUN.md.\n')


if __name__=='__main__':main()
