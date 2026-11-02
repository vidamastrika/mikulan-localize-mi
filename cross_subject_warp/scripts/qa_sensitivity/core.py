"""Explicit sensitivity policies, provenance and unchanged MRI validation gates."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import re
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'cross_subject_warp'
KEY=['strategy','scenario','method','reference_subject','test_subject','run','condition']
PAIR=['reference_subject','test_subject','registration_mode']

@dataclass(frozen=True)
class Policy:
    name:str
    pair_tolerance_mm:float
    source_tolerance_mm:float=.1
    min_coverage:float=.75
    image_rmse:float=.015
    grid_mm:float=.02
    samples:int=200
    seed:int=2901

PRIMARY=Policy('primary_rt010',.1)
SENSITIVITY=Policy('pair_rt015_source_rt010',.15)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def modules():
    qa=load(EXP/'scripts/06_validate_registration_resources.py','sensitivity_original_qa')
    mapping,_=qa.historical_mapping()
    ev=load(EXP/'scripts/06_mapped_evaluation.py','sensitivity_original_evaluation')
    base=load(EXP/'scripts/06_evaluate_localization.py','sensitivity_original_base')
    return qa,mapping,ev,base


def read(path):return pd.read_csv(path,dtype=str,keep_default_na=False)
def frame(path):
    table=pd.read_csv(path,keep_default_na=False,low_memory=False)
    for column in ['localization_error_mm','roundtrip_mm','source_point_roundtrip_mm','raw_head_separation_mm']:
        if column in table:table[column]=pd.to_numeric(table[column].replace('',np.nan),errors='raise')
    if 'localization_error_valid' in table:
        values=table.localization_error_valid.astype(str).str.lower()
        if not values.isin(['true','false']).all():raise ValueError('Invalid metric-validity flag')
        table['localization_error_valid']=values.eq('true')
    return table
def yes(series):return series.astype(str).str.lower().eq('true')

class Ledger:
    def __init__(self):self.files={};self.stats={}
    def track(self,path,expected=None):
        path=Path(path).resolve();stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if str(path) in self.files and self.stats.get(str(path))==signature:
            value=self.files[str(path)]['sha256']
        else:value=sha(path)
        if expected is not None and value!=expected:raise ValueError(f'Changed input: {path}')
        self.files[str(path)]=dict(path=str(path),sha256=value,size_bytes=stat.st_size);self.stats[str(path)]=signature
        return value
    def verify(self):
        for path,item in self.files.items():
            if sha(path)!=item['sha256']:raise ValueError(f'Input changed during analysis: {path}')
    def receipt(self):return list(self.files.values())


def protect(args):
    out=args.output_root
    if out.exists() or out.is_symlink():raise FileExistsError(f'New destination required: {out}')
    paths=[EXP/'source',EXP/'scripts',ROOT/'cross_subject',ROOT/'cross_subject_no_warp',
           args.primary_qa,args.primary_evaluation,args.diagnostics,args.bridge_root,args.preparation_root,
           *([args.attribution_root] if args.stage!='investigate' else []),
           *([args.sensitivity_qa] if args.stage!='qa' else []),
           *([args.sensitivity_evaluation] if args.stage!='evaluate' else [])]
    paths += [EXP/'outputs'/n for n in ['03_reference_based_final','05_single_reference_final',
        '03_gamma_wholebrain_sensitivity','05_wholebrain_hemisphere_sensitivity','03_05_native_completion_audit',
        '02_select_loso_parameters','04_single_reference_parameters','06_evaluate_localization']]
    for path in paths:
        if out==path or path in out.parents or out in path.parents:raise ValueError(f'Output overlaps input: {path}')
    doc=json.loads((args.preparation_root/'10_source_provenance.json').read_text());dataset=Path(doc['locations']['dataset']).resolve()
    if out==dataset or dataset in out.parents or out in dataset.parents:raise ValueError('Output overlaps original dataset')
    return dataset


def context(args,full=False):
    qa,mapping,ev,base=modules();ledger=Ledger()
    primary_manifest=args.primary_qa/'05_registration_qa_manifest.json'
    primary_csv=args.primary_qa/'01_pair_mode_qa.csv'
    state=json.loads(primary_manifest.read_text());table=read(primary_csv)
    ev.validate_pair_qa(table,state,qa)
    sources=json.loads((args.primary_qa/'04_source_provenance.json').read_text())
    parent={str(Path(r['path']).resolve()):r for r in sources}
    for p in [primary_manifest,primary_csv,args.primary_qa/'04_source_provenance.json']:
        ledger.track(p)
    protocol=EXP/'docs/REGISTRATION_QA_SENSITIVITY_PROTOCOL.md'
    for relative,expected in re.findall(r'\| `([^`]+)` \| `([a-f0-9]{64})` \|',protocol.read_text()):
        folder,tail=relative.split('/',1)
        root={'06_registration_qa':args.primary_qa,'06_roundtrip_diagnostics':args.diagnostics,
              '06_evaluate_localization_mapped':args.primary_evaluation}.get(folder)
        if root is not None:ledger.track(root/tail,expected)
    for p in [args.preparation_root/'10_source_provenance.json',args.bridge_root/'05_bridge_manifest.json',args.bridge_root/'02_native_physical_bridges.json']:
        entry=parent.get(str(p.resolve()))
        if entry is None:raise ValueError(f'Missing parent QA receipt: {p}')
        ledger.track(p,entry['sha256'])
    bridge=json.loads((args.bridge_root/'05_bridge_manifest.json').read_text())
    if bridge.get('status')!='ALL_NATIVE_BRIDGES_PASS' or bridge.get('subjects_passed')!=7:raise ValueError('Native bridge validation incomplete')
    for p in [EXP/'scripts/06_validate_registration_resources.py',EXP/'source/01_registration_resources/scripts/32_all_reference_localization_evaluation.py']:
        ledger.track(p,parent[str(p.resolve())]['sha256'])
    exe=qa.locate_ants(Path(state['ants_executable']));ledger.track(exe,state['ants_executable_sha256'])
    if full:
        for i,r in enumerate(sources,1):
            ledger.track(r['path'],r['sha256'])
            if i%100==0:print(f'Verified primary source checksums: {i}/{len(sources)}',flush=True)
    for path in sorted((EXP/'scripts/qa_sensitivity').glob('*.py')):ledger.track(path)
    ledger.track(EXP/'scripts/06_registration_qa_sensitivity.py')
    ledger.track(EXP/'docs/REGISTRATION_QA_SENSITIVITY_PROTOCOL.md')
    return qa,mapping,ev,base,ledger,table,parent,str(exe)


def verify_pair_inputs(args,row,qa,ledger,parent):
    files=qa.files_for_pair(args.resource_root,row['reference_subject'],row['test_subject'])
    for path in [Path(row['fixed_mri']),Path(row['moving_mri']),*files.values()]:
        item=parent.get(str(path.resolve()))
        if item is None:raise ValueError(f'No primary source receipt: {path}')
        ledger.track(path,item['sha256'])
    return files


def roundtrip_metrics(point_map,exe,source,dest,forward,back,qa):
    pts=qa.foreground_points(source)
    mapped=point_map(exe,pts,forward)
    vox=nib.affines.apply_affine(np.linalg.inv(dest.affine),mapped)
    valid=np.isfinite(mapped).all(axis=1)&np.all((vox>=1)&(vox<np.asarray(dest.shape[:3])-2),axis=1)
    errors=np.full(len(pts),np.nan)
    if valid.any():errors[valid]=np.linalg.norm(point_map(exe,mapped[valid],back)-pts[valid],axis=1)
    return dict(tested=len(pts),valid=int(valid.sum()),coverage=float(valid.mean()),
                mri_destination_valid=int(valid.sum()),first_field_domain_valid=None,
                both_field_domains_valid=None,field_domain_excluded=0,field_domain_checked=False,
                maximum_mm=float(np.max(errors[valid])) if valid.any() else None,
                finite=bool(valid.any() and np.isfinite(errors[valid]).all()),
                p95_mm=float(np.percentile(errors[valid],95)) if valid.any() else None,
                over_rt010=int(np.sum(errors[valid]>.1)),over_rt015=int(np.sum(errors[valid]>.15)))


def accepted(metrics,policy):
    return bool(metrics['finite'] and metrics['coverage']>=policy.min_coverage and
                metrics['maximum_mm'] is not None and metrics['maximum_mm']<=policy.pair_tolerance_mm)


def validate_pair(mapping,exe,qa,resources,dataset,reference,test,field_basis=None):
    """Measure all gates before either policy is classified; no modified globals."""
    files=qa.files_for_pair(resources,reference,test)
    fixed_path=dataset/'derivatives/epochs'/test/'anat'/f'{test}_mod-T1w_maskface.nii'
    moving_path=dataset/'derivatives/epochs'/reference/'anat'/f'{reference}_mod-T1w_maskface.nii'
    fixed=nib.load(fixed_path);moving=nib.load(moving_path);terms=mapping.terms(files)
    if fixed.ndim!=3 or moving.ndim!=3:raise ValueError('Expected 3D MRIs')
    rows=[]
    for mode in ['affine','syn']:
        row=dict(reference_subject=reference,test_subject=test,registration_mode=mode,
            fixed_mri=str(fixed_path),moving_mri=str(moving_path),non_roundtrip_gates_pass=False,reason='')
        try:
            prefix='aff' if mode=='affine' else 'syn'
            if mode=='affine':qa.verify_log_identity(files['aff_log'].read_text(),fixed_path,moving_path)
            else:
                qa.verify_log_identity(files['syn_log'].read_text(),fixed_path,moving_path)
                q=json.loads(files['syn_quality'].read_text())
                if q['donor']!=reference or q['target']!=test or Path(q['fixed_mri']).resolve()!=fixed_path.resolve() or Path(q['moving_mri']).resolve()!=moving_path.resolve():
                    raise ValueError('Saved SyN input identities differ')
                qa.check_field(files['warp'],fixed);qa.check_field(files['invwarp'],fixed)
                qa.grid_error_mm(nib.load(files['syn_inverse_image']),moving)
            warped=files[f'{prefix}_image']
            row['grid_max_error_mm']=qa.grid_error_mm(nib.load(warped),fixed)
            im=mapping.image_qa(exe,fixed_path,moving_path,warped,terms[f'{prefix}_t2r'],qa.SAMPLES)
            row.update(image_reproduction_rmse=im['rmse'],image_valid_samples=im['n_valid'])
            if not np.isfinite(im['rmse']) or im['rmse']>PRIMARY.image_rmse or im['n_valid']<PRIMARY.min_coverage*PRIMARY.samples:raise ValueError('Forward saved image reproduction fails')
            row['inverse_image_reproduction_rmse']=None
            if mode=='syn':
                im2=mapping.image_qa(exe,moving_path,fixed_path,files['syn_inverse_image'],terms['syn_r2t'],qa.SAMPLES)
                row['inverse_image_reproduction_rmse']=im2['rmse']
                if not np.isfinite(im2['rmse']) or im2['rmse']>PRIMARY.image_rmse or im2['n_valid']<PRIMARY.min_coverage*PRIMARY.samples:raise ValueError('Inverse saved image reproduction fails')
            row['non_roundtrip_gates_pass']=True
            for direction,src,dst,forward,back in [('reference',moving,fixed,terms[f'{prefix}_r2t'],terms[f'{prefix}_t2r']),('test',fixed,moving,terms[f'{prefix}_t2r'],terms[f'{prefix}_r2t'])]:
                if mode=='syn':
                    if field_basis is None:raise ValueError('Controlled SyN vector basis is required for field-domain checks')
                    from .attribution import field_domain_roundtrip_metrics
                    metrics=field_domain_roundtrip_metrics(mapping,exe,src,dst,direction,files,qa,field_basis)
                else:metrics=roundtrip_metrics(mapping.point_map,exe,src,dst,forward,back,qa)
                for k,v in metrics.items():row[f'{direction}_{k}']=v
            ref={k:row[f'reference_{k}'] for k in ['finite','coverage','maximum_mm']}
            tst={k:row[f'test_{k}'] for k in ref}
            for policy in [PRIMARY,SENSITIVITY]:
                row[f'{policy.name}_pass']=accepted(ref,policy) and accepted(tst,policy)
            if not row[f'{PRIMARY.name}_pass']:row['reason']='Original coverage/roundtrip criterion not met; see directional metrics'
        except Exception as exc:row['reason']=f'{type(exc).__name__}: {exc}'
        row.setdefault(f'{PRIMARY.name}_pass',False);row.setdefault(f'{SENSITIVITY.name}_pass',False)
        rows.append(row)
    return rows


def require_manifest(root,name,status,ledger):
    path=root/name;state=json.loads(path.read_text());ledger.track(path)
    if state.get('status')!=status:raise ValueError(f'Incomplete or blocked stage: {path}')
    for item in state.get('artifacts',[]):ledger.track(root/item['file'],item['sha256'])
    if (root/'source_provenance.json').is_file():
        for item in json.loads((root/'source_provenance.json').read_text()):ledger.track(item['path'],item['sha256'])
    return state


def publish(args,ledger,manifest,tables=None,jsons=None,extra_files=None):
    tables=tables or {};jsons=jsons or {};ledger.verify()
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.qa_sensitivity_',dir=args.output_root.parent) as tmp:
        stage=Path(tmp)/'results';stage.mkdir();artifacts=[]
        for name,table in tables.items():table.to_csv(stage/name,index=False)
        for name,doc in jsons.items():(stage/name).write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
        validation=dict(stage=args.stage,runtime_checks=manifest.get('runtime_validation',{}),
            check_source='Historical artifacts and native-tool diagnostic controls as declared by this stage',
            unit_tests_rerun_by_analysis=False,unit_test_report='docs/REGISTRATION_QA_SENSITIVITY_VALIDATION.md',
            code_coverage_percentage_measured=False,scientific_accuracy_established=False,
            not_run=manifest.get('not_run',[]),limitations=manifest.get('limitations',[]))
        (stage/'validation_report.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
        for name,source in (extra_files or {}).items():
            destination=stage/name
            if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('Invalid artifact name')
            destination.parent.mkdir(parents=True,exist_ok=True)
            import shutil
            shutil.copyfile(source,destination)
        for p in sorted(stage.rglob('*')):
            if p.is_file():artifacts.append(dict(file=str(p.relative_to(stage)),sha256=sha(p)))
        (stage/'source_provenance.json').write_text(json.dumps(ledger.receipt(),indent=2)+'\n')
        artifacts.append(dict(file='source_provenance.json',sha256=sha(stage/'source_provenance.json')))
        manifest=dict(manifest,artifacts=artifacts,policies=[asdict(PRIMARY),asdict(SENSITIVITY)],
            inverse_reconstruction_executed=False,registration_fitting_executed=False,
            primary_outputs_modified=False,scientific_accuracy_established=False,
            versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','nibabel','scipy']})
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        if args.output_root.exists():raise FileExistsError(args.output_root)
        stage.rename(args.output_root)
    print('Written:',args.output_root,flush=True)
