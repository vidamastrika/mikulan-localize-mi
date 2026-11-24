"""Stage 2 provenance, protected paths and manual external-process runners."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import traceback

import nibabel as nib
import numpy as np
import pandas as pd
from input_utils import image_header

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'localize_mi_average_seven'
VAR=EXP/'maskface_template'
OLD=EXP/'outputs/02_template_retry1'
OLD_FS=EXP/'work/freesurfer/average_seven'
SUBJECT='average_seven_maskface'
ANTS_BIN=Path.home()/'micromamba/envs/localizemi-registration/bin'
STAMP='freesurfer-linux-rocky8_x86_64-8.2.0-20260314-d932c45'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,obj):
    def default(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        return str(value)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=default)+'\n')


def protected_output(path):
    path=Path(path).expanduser().resolve()
    if (VAR/'outputs').resolve() not in path.parents:
        raise ValueError('Stage 2 outputs must be strictly inside maskface_template/outputs')
    if path.exists():raise FileExistsError(f'Existing output refused: {path}; select a fresh run')
    return path


class Run:
    def __init__(self,path,stage):
        self.out=protected_output(path);self.out.mkdir(parents=True,exist_ok=False)
        self.inputs={}
        self.doc=dict(status='RUNNING',stage=stage,created_utc=datetime.now(timezone.utc).isoformat(),
            argv=sys.argv,anatomy_review='PENDING',head_scaling=False,face_reconstruction=False,
            original_files_modified=False,EEG_alignment=False,forward=False,inverse=False)
        for p in [Path(__file__),Path(__file__).with_name('input_utils.py'),Path(__file__).with_name('stage2_qa.py'),Path(sys.argv[0]).resolve(),VAR/'config/stage2.json']:
            if p.is_file():self.use(p)
        self.save()
    def use(self,path,expected=None):
        p=Path(path).expanduser().resolve()
        if str(p) not in self.inputs:self.inputs[str(p)]=sha(p)
        if expected is not None and self.inputs[str(p)]!=expected:raise ValueError(f'Input hash mismatch: {p}')
        return p
    def verify(self,root,name='manifest.json',stage=None):
        root=Path(root).resolve();p=self.use(root/name);m=json.loads(p.read_text())
        if m['status']!='COMPLETE':raise ValueError(f'Incomplete upstream run: {p}')
        if stage is not None and m.get('stage')!=stage:raise ValueError(f'Wrong upstream stage: {p}')
        for k,v in m.get('input_sha256',{}).items():self.use(k,v)
        for k,v in m.get('output_sha256',{}).items():self.use(root/k,v)
        return m
    def save(self):
        self.doc['input_sha256']=self.inputs;write_json(self.out/'manifest.json',self.doc)
    def __enter__(self):return self
    def __exit__(self,kind,error,tb):
        if error is not None:
            self.doc.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=f'{kind.__name__}: {error}')
            (self.out/'failure.txt').write_text(''.join(traceback.format_exception(kind,error,tb)))
        else:
            try:
                for p,v in self.inputs.items():
                    if sha(p)!=v:raise ValueError(f'Input changed during execution: {p}')
                self.doc['status']='COMPLETE'
            except Exception as exc:
                self.doc.update(status='FAILED',error=str(exc));self.save();raise
        print('Hashing preserved outputs:',self.out,flush=True)
        self.doc['output_sha256']={str(p.relative_to(self.out)):sha(p) for p in self.out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        self.doc['finished_utc']=datetime.now(timezone.utc).isoformat();self.save()
        print(self.doc['status'],self.out,flush=True)
        return False


def launch(command,log,cwd,env=None):
    print('Manual stage executing:',shlex.join(map(str,command)),flush=True)
    with Path(log).open('w') as f:
        process=subprocess.Popen(list(map(str,command)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait()
        except BaseException:
            import signal
            os.killpg(process.pid,signal.SIGTERM);process.wait();raise
    if code:raise subprocess.CalledProcessError(code,command)


def small_command(command,env=None):
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,check=True)
    return result.stdout.strip()


def ants_environment(directory):
    directory=Path(directory).expanduser().resolve();env=os.environ.copy()
    env['PATH']=str(directory)+os.pathsep+env.get('PATH','')
    required=['antsMultivariateTemplateConstruction2.sh','antsRegistration','antsApplyTransforms','N4BiasFieldCorrection','AverageImages',
              'AverageAffineTransform','ImageMath','MultiplyImages','ANTSpexec.sh','waitForSGEQJobs.pl','waitForPBSQJobs.pl','waitForXGridJobs.pl','waitForSlurmJobs.pl']
    paths={name:shutil.which(name,path=env['PATH']) for name in required}
    if any(p is None for p in paths.values()):raise FileNotFoundError(f'ANTs environment missing tools: {[k for k,v in paths.items() if v is None]}')
    return env,paths


def ants_command(script,out,inputs):
    # Only -b 1 differs in options: retention, not numerical computation.
    return ['bash',str(script),'-d','3','-o',str(Path(out)/'group_'),'-i','4','-t','SyN','-m','CC','-b','1',*map(str,inputs)]


def construction_contract(root):
    root=Path(root)
    required=[root/'group_template0.nii.gz',root/'intermediateTemplates/initial_group_template0.nii.gz']
    required += [root/'intermediateTemplates'/f'SyN_iteration{i}_group_template0.nii.gz' for i in range(4)]
    for i in range(4):
        backup=root/f'ANTs_iteration_{i}'
        if len(list(backup.glob('group_input*-WarpedToTemplate.nii.gz')))!=7:
            raise ValueError(f'Iteration {i} does not preserve seven registered inputs')
    for path in required:
        if not path.is_file():raise FileNotFoundError(f'Missing completed construction output: {path}')
    for pattern in ['group_input*-WarpedToTemplate.nii.gz','group_input*-0GenericAffine.mat','group_input*-1Warp.nii.gz','group_input*-1InverseWarp.nii.gz']:
        if len(list(root.glob(pattern)))!=7:raise ValueError(f'Expected seven final {pattern}')
    header=image_header(nib.load(root/'group_template0.nii.gz'))
    if not header['technical_header_valid']:raise ValueError('Final template header failed')
    return header


def fs_command(runtime,out,inner_command,docker_command='sudo docker',image=None,license_path=None,write=True):
    """Original repo read-only; only the NEW stage output is writable in Docker."""
    if runtime=='local':return inner_command
    if image is None:raise ValueError('Pinned Docker image required')
    command=shlex.split(docker_command)+['run','--rm','--pull=never','--network=none','--user',f'{os.getuid()}:{os.getgid()}',
        '--volume',f'{ROOT}:{ROOT}:ro','--volume',f'{out}:{out}:{"rw" if write else "ro"}',
        '--workdir',str(out)]
    if license_path is not None:command += ['--volume',f'{license_path}:/license.txt:ro','--env','FS_LICENSE=/license.txt']
    command += ['--entrypoint',inner_command[0],image,*inner_command[1:]]
    return command


def fs_runtime(args,run):
    license_path=Path(args.license).expanduser().resolve()
    if not license_path.is_file():raise FileNotFoundError('FreeSurfer license missing; its contents are not logged/hashed')
    if args.runtime=='docker':
        inspection=json.loads(small_command(shlex.split(args.docker_command)+['image','inspect',args.image]))[0]
        image=inspection['Id'];write_json(run.out/'docker_image.json',dict(Id=image,RepoDigests=inspection.get('RepoDigests',[]),reference=args.image))
        command=shlex.split(args.docker_command)+['run','--rm','--pull=never','--network=none','--entrypoint','/bin/sh',image,'-c','cat "$FREESURFER_HOME/build-stamp.txt"']
        stamp=small_command(command)
    else:
        home=Path(os.environ.get('FREESURFER_HOME','/nonexistent'))
        stamp=run.use(home/'build-stamp.txt').read_text().strip();image=None
        if not shutil.which('recon-all') or not shutil.which('mri_watershed'):raise FileNotFoundError('Source matching FreeSurfer SetUpFreeSurfer.sh')
    if stamp.strip()!=STAMP:raise ValueError(f'FreeSurfer build differs from original: {stamp}; expected {STAMP}')
    run.doc.update(FreeSurfer_build=stamp,runtime=args.runtime,docker_image_id=image,license_path=str(license_path))
    return image,license_path


def add_fs_arguments(parser):
    parser.add_argument('--runtime',choices=['docker','local'],default='docker')
    parser.add_argument('--docker-command',default='sudo docker')
    parser.add_argument('--image',default='freesurfer/freesurfer:8.2.0')
    parser.add_argument('--license',type=Path,default=Path.home()/'.freesurfer/license.txt')


def warning_rows(path):
    return [dict(line=i,text=line.strip()) for i,line in enumerate(Path(path).read_text(errors='replace').splitlines(),1)
            if any(word in line.lower() for word in ['warn','error','fail','cannot','nan','inf intensity'])]
