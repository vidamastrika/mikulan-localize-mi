#!/usr/bin/env python3
"""Manually run matched FreeSurfer recon-all and watershed surfaces."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from stage2_common import (VAR, Run, add_fs_arguments, fs_command, fs_runtime,
                           launch, write_json)


def launch_docker(command, log, cwd):
    # Stay in the foreground so sudo can prompt through the terminal, including
    # when its cached credentials expire between reconstruction and watershed.
    print('Manual stage executing:', shlex.join(command), flush=True)
    with log.open('w') as output:
        subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--construction-root', type=Path,
                        default=VAR/'outputs/03_template/run_04')
    parser.add_argument('--mri-qa-root', type=Path,
                        default=VAR/'outputs/04_template_MRI_QA/run_04')
    parser.add_argument('--output-root', type=Path,
                        default=VAR/'outputs/05_freesurfer/run_04')
    parser.add_argument('--reviewed-mri-qa', action='store_true',
                        help='Confirm that you reviewed the completed MRI QA report and figures')
    parser.add_argument('--execute', action='store_true',
                        help='Required to launch the expensive reconstruction')
    add_fs_arguments(parser)
    args = parser.parse_args()
    run_command = launch_docker if args.runtime == 'docker' else launch
    if not args.execute:
        parser.error('--execute is required; no reconstruction was launched')
    if not args.reviewed_mri_qa:
        parser.error('--reviewed-mri-qa is required after inspecting the MRI QA outputs')

    with Run(args.output_root, '05_FREESURFER_AND_WATERSHED') as run:
        construction = run.verify(args.construction_root, stage='03_CONSTRUCT')
        mri_qa = run.verify(args.mri_qa_root, stage='04_MRI_QA')
        if not mri_qa.get('pre_FreeSurfer_technical_ready'):
            raise ValueError('MRI QA did not mark the template technically ready')
        if construction.get('construction_returncode') != 0:
            raise ValueError('ANTs construction did not complete successfully')
        template = run.use(args.construction_root/'group_template0.nii.gz')
        if run.inputs[str(template)] != construction['output_sha256'].get('group_template0.nii.gz'):
            raise ValueError('Template hash differs from completed construction manifest')

        image, license_path = fs_runtime(args, run)
        subjects = run.out/'subjects'
        surfaces = run.out/'watershed_surfaces'
        subjects.mkdir()
        surfaces.mkdir()
        subject = 'average_seven_maskface'
        recon_inner = ['recon-all', '-sd', str(subjects), '-s', subject,
                       '-i', str(template), '-all']
        recon = fs_command(args.runtime, run.out, recon_inner,
                           args.docker_command, image, license_path)
        (run.out/'recon_all_command.sh').write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(recon)+'\n')
        run.doc.update(subject=subject, subject_directory=str(subjects/subject),
                       input_template=str(template),
                       recon_all_command=recon,
                       reconstruction_status='RUNNING',
                       FreeSurfer_completion_not_anatomical_validation=True)
        run.save()
        run_command(recon, run.out/'recon-all.log', run.out)

        t1 = subjects/subject/'mri/T1.mgz'
        if not t1.is_file():
            raise FileNotFoundError('recon-all returned zero without mri/T1.mgz')
        run.use(t1)
        prefix = surfaces/subject
        watershed_inner = ['mri_watershed', '-useSRAS', '-surf', str(prefix),
                           str(t1), str(surfaces/'watershed.mgz')]
        watershed = fs_command(args.runtime, run.out, watershed_inner,
                               args.docker_command, image, license_path)
        (run.out/'watershed_command.sh').write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(watershed)+'\n')
        run.doc.update(reconstruction_status='RECON_ALL_RETURNED_ZERO',
                       watershed_command=watershed,
                       surface_method='FreeSurfer mri_watershed -useSRAS -surf; original MNE-derived command had no -T1 or -atlas option',
                       surfaces_pending_review=True)
        run.save()
        run_command(watershed, run.out/'watershed.log', run.out)
        expected = [surfaces/f'{subject}_{name}_surface' for name in
                    ('outer_skin', 'outer_skull', 'inner_skull', 'brain')]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'Watershed did not produce expected surface files: {missing}')
        for path in expected:
            run.use(path)
        warnings = []
        for log in (run.out/'recon-all.log', run.out/'watershed.log'):
            for number, line in enumerate(log.read_text(errors='replace').splitlines(), 1):
                if any(term in line.lower() for term in ('warn', 'error', 'fail', 'cannot', 'nan', 'inf')):
                    warnings.append(dict(log=log.name, line=number, text=line.strip()))
        import pandas as pd
        pd.DataFrame(warnings, columns=['log', 'line', 'text']).to_csv(run.out/'warnings.csv', index=False)
        run.doc.update(reconstruction_status='COMPLETE_RETURN_ZERO',
                       watershed_status='COMPLETE_RETURN_ZERO',
                       outputs_exist=True,
                       anatomical_validity='NOT_ESTABLISHED; complete 06 MRI/surface QA and human review')
        (run.out/'REPORT_stage2.md').write_text(
            '# FreeSurfer reconstruction and watershed surface generation\n\n'
            'Both commands returned zero and required files were found. This is a processing status only; it does not establish anatomical validity. Four uncorrected watershed surfaces are retained. No BEM solution, source space, surface repair, EEG alignment, forward model or inverse localization was run. Inspect `warnings.csv`, `recon-all.log` and `watershed.log`, then run `06_review_surfaces.py`.\n')


if __name__ == '__main__':
    main()
