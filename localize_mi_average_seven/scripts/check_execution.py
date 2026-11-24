"""Fast execution-adapter checks using tiny fixtures; no modelling or inverse fits."""
import ast
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import shlex

import numpy as np
from common import EXP, digest, config, template_plan
from execution import validate_mesh, verify_manifest, fingerprint


def main():
    for path in (EXP/'scripts').glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    rr=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    validate_mesh(rr,faces)
    for invalid in [faces[:-1],np.array([[0,0,1]])]:
        try: validate_mesh(rr,invalid)
        except ValueError: pass
        else: raise AssertionError('Invalid mesh accepted')
    with tempfile.TemporaryDirectory() as directory:
        tmp=Path(directory);data=tmp/'data';data.write_text('original')
        manifest=tmp/'manifest.json'
        manifest.write_text(json.dumps({'input_sha256':{str(data):digest(data)}}))
        verify_manifest(manifest)
        data.write_text('changed')
        try: verify_manifest(manifest)
        except ValueError: pass
        else: raise AssertionError('Changed frozen input accepted')
        template_plan(config(EXP/'config/experiment.json'), tmp/'template-plan')
        script = tmp/'template-plan/01_build_template.sh'
        command = shlex.split(script.read_text().splitlines()[-1])
        assert command[command.index('-m')+1] == 'CC'
        assert '-s' not in command  # Use ANTs default smoothing, never the metric string.
        subprocess.run(['bash','-n',str(script)],check=True)
    assert fingerprint({'x':1,'y':2})==fingerprint({'y':2,'x':1})
    for stage in ['02_execute_template','03_execute_anatomy','04a_coregister','04b_make_forward']:
        script=EXP/'scripts'/f'{stage}.py'
        proc=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,check=True)
        assert 'Plan only' in proc.stdout
    print('PASS: syntax, mesh gates, hash tampering, deterministic fingerprints, heavy-stage default guards')


if __name__=='__main__':
    main()
