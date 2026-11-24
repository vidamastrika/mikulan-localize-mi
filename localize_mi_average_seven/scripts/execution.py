"""Explicit-execution adapters for the within-cohort no-warp comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import time
import traceback
import importlib.metadata

import numpy as np
import pandas as pd
from common import ROOT, EXP, KEY, FINGERPRINTS, config, digest, fresh, schedule, write_json


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def fingerprint(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_manifest(path):
    doc = json.loads(path.read_text())
    for filename, expected in (doc['input_sha256'] | doc.get('forward_hashes', {})).items():
        if digest(Path(filename)) != expected:
            raise ValueError(f'Frozen input changed: {filename}')
    return doc


def run_template(args, c):
    from common import template_plan
    audit = json.loads((args.preparation_root/'04_manifest.json').read_text())
    if not audit['ready'] or audit['configuration'] != c:
        raise ValueError('Passing stage 01 audit with matching configuration required')
    inventory = json.loads((args.preparation_root/'01_mri_inventory.json').read_text())
    for row in inventory:
        if row['status'] != 'PASS' or digest(Path(row['path'])) != row['sha256']:
            raise ValueError('MRI input changed or did not pass inventory')
    if not shutil.which('antsMultivariateTemplateConstruction2.sh'):
        raise RuntimeError('ANTs template construction script must be on PATH')
    template_plan(c, args.output_root)
    log_path = args.output_root/'03_construction.log'
    try:
        with log_path.open('w') as log:
            subprocess.run(['bash', str((args.output_root/'01_build_template.sh').resolve())],
                           cwd=args.output_root, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as exc:
        tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-15:])
        write_json(args.output_root/'04_execution_manifest.json', dict(
            status='FAIL', returncode=exc.returncode, log=str(log_path),
            input_sha256={r['path']:r['sha256'] for r in inventory}, configuration=c,
            construction_only=True, registration_applied_to_estimates=False))
        raise SystemExit(f'ANTs construction failed (exit {exc.returncode}). Output preserved at {args.output_root}.\n'
                         f'Log: {log_path}\n{tail}\nChoose a fresh output directory for retry; an initial average is not a completed template.') from None
    write_json(args.output_root/'04_execution_manifest.json', dict(
        status='COMPLETE', input_sha256={r['path']:r['sha256'] for r in inventory}, configuration=c,
        construction_only=True, registration_applied_to_estimates=False))


def anatomy(args, c):
    import mne
    out = args.output_root
    subjects = args.subjects_dir.expanduser().resolve()
    if EXP not in subjects.parents:
        raise ValueError('--subjects-dir must be inside the new experiment')
    if args.phase == 'segment':
        if args.template is None or not args.template.is_file():
            raise ValueError('--template existing group-average NIfTI is required')
        construction_manifest=args.template.parent/'04_execution_manifest.json'
        if not construction_manifest.is_file() or verify_manifest(construction_manifest).get('status')!='COMPLETE':
            raise ValueError('Successful template-construction manifest required beside --template; an initial average from a failed run is not a completed template')
        if not shutil.which('recon-all'):
            raise RuntimeError('FreeSurfer recon-all must be on PATH; source SetUpFreeSurfer.sh')
        if (subjects/'average_seven').exists():
            raise FileExistsError('FreeSurfer subject already exists; resume manually after reviewing its logs')
        fresh(out); subjects.mkdir(parents=True,exist_ok=True)
        command = ['recon-all','-sd',str(subjects),'-s','average_seven','-i',str(args.template.resolve()),'-all']
        write_json(out/'01_segmentation_manifest.json', dict(command=command,input_sha256={str(args.template.resolve()):digest(args.template)}))
        with (out/'02_segmentation.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        return
    if not (subjects/'average_seven/scripts/recon-all.done').is_file():
        raise ValueError('Completed FreeSurfer segmentation required before watershed/BEM construction')
    fresh(out)
    # All writes stay in this experiment's subjects directory; prohibit historical FS roots.
    mne.bem.make_watershed_bem('average_seven',subjects_dir=subjects,overwrite=False,show=False)
    models = mne.make_bem_model('average_seven',ico=c['anatomy']['bem_ico'],
                              conductivity=c['anatomy']['conductivity_s_per_m'],subjects_dir=subjects)
    for surface in models:
        validate_mesh(surface['rr'],surface['tris'])
    solution = mne.make_bem_solution(models)  # MNE also checks topology/nesting.
    bem=out/'average_seven-bem-sol.fif'; src=out/'average_seven-src.fif'
    mne.write_bem_solution(bem,solution,overwrite=False)
    sources=mne.setup_source_space('average_seven',spacing=c['anatomy']['source_spacing'],subjects_dir=subjects,add_dist=False)
    mne.write_source_spaces(src,sources,overwrite=False)
    # Supply three manually reviewed anatomical fiducials in MRI surface-RAS metres.
    fid = subjects/'average_seven/bem/average_seven-fiducials.fif'
    scalp = subjects/'average_seven/bem/outer_skin.surf'
    write_json(out/'03_anatomy.json', dict(arm='average_seven',bem=str(bem.resolve()),src=str(src.resolve()),
        scalp=str(scalp),fiducials=str(fid),subject_dir=str(subjects/'average_seven'),
        input_sha256={str(p.resolve()):digest(p) for p in [bem,src,scalp]},
        review_required='Inspect segmentation, nested surfaces, intersections and containment. Save manually reviewed fiducials before coregistration.'))


def validate_mesh(rr, tris):
    rr, tris=np.asarray(rr),np.asarray(tris)
    if not np.isfinite(rr).all() or tris.ndim!=2 or tris.shape[1]!=3:
        raise ValueError('Invalid mesh')
    if tris.min()<0 or tris.max()>=len(rr): raise ValueError('Invalid triangle index')
    area=np.linalg.norm(np.cross(rr[tris[:,1]]-rr[tris[:,0]],rr[tris[:,2]]-rr[tris[:,0]]),axis=1)
    if (area<=1e-15).any(): raise ValueError('Degenerate mesh triangles')
    edges=np.concatenate([tris[:,[0,1]],tris[:,[1,2]],tris[:,[2,0]]])
    _, counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    if not (counts==2).all(): raise ValueError('Mesh is not closed/manifold')


def anatomy_paths(args):
    if args.arm=='fsaverage':
        root=args.fsaverage_root.expanduser().resolve()
        paths=dict(arm='fsaverage',bem=str(root/'bem/fsaverage-5120-5120-5120-bem-sol.fif'),
                   src=str(root/'bem/fsaverage-ico-5-src.fif'),scalp=str(root/'bem/outer_skin.surf'),
                   fiducials=str(root/'bem/fsaverage-fiducials.fif'))
    else:
        paths=verify_manifest(args.anatomy_manifest)
    for name in ['bem','src','scalp','fiducials']:
        if not Path(paths[name]).is_file(): raise FileNotFoundError(f'{name}: {paths[name]}')
    return paths


def coregister(args,c):
    import mne
    from mne.io.constants import FIFF
    helpers=module(ROOT/'cross_subject_fsaverage/scripts/02_coregister_eeg.py','average_seven_icp')
    paths=anatomy_paths(args)
    scalp,faces=mne.read_surface(paths['scalp']); validate_mesh(scalp/1000,faces); scalp=scalp/1000
    if not .05 < np.percentile(np.linalg.norm(scalp,axis=1),95) < .25:
        raise ValueError('Unexpected scalp radius/units')
    fid,frame=mne.io.read_fiducials(paths['fiducials'])
    if int(frame)!=int(FIFF.FIFFV_COORD_MRI): raise ValueError('Fiducials must be MRI surface-RAS metres')
    positions={int(d['ident']):d['r'] for d in fid}
    initial=np.linalg.inv(mne.transforms.get_ras_to_neuromag_trans(positions[2],positions[1],positions[3]))
    out=fresh(args.output_root);(out/'transforms').mkdir()
    rows=[]; hashes={str(Path(paths[k])):digest(Path(paths[k])) for k in ['bem','src','scalp','fiducials']}
    for subject in c['participants']:
        native=ROOT/c['dataset']/'derivatives/sourcemodelling'/subject/'fwd'/f'{subject}_fwd.fif'
        hashes[str(native)]=digest(native)
        f=mne.read_forward_solution(native,verbose=False)
        xyz=np.array([ch['loc'][:3] for ch in f['info']['chs']])
        if not np.isfinite(xyz).all() or any(int(ch['coord_frame'])!=int(FIFF.FIFFV_COORD_HEAD) for ch in f['info']['chs']):
            raise ValueError('Native sensors must be finite HEAD metres')
        cp=c['coregistration']
        mat,dist,it=helpers.icp(xyz,scalp,initial,cp['iterations'],cp['trim_fraction'])
        median,p95=np.percentile(dist*1000,[50,95])
        passed=median<=cp['median_limit_mm'] and p95<=cp['p95_limit_mm']
        path=out/'transforms'/f'{subject}-trans.fif'
        mne.write_trans(path,mne.transforms.Transform('head','mri',mat),overwrite=False)
        hashes[str(path.resolve())]=digest(path)
        rows.append(dict(test_subject=subject,status='PASS_NUMERICAL_REVIEW_REQUIRED' if passed else 'FAIL_ALIGNMENT',
                         median_mm=median,p95_mm=p95,transform=str(path.resolve()),native_forward=str(native),
                         reason='' if passed else 'Rigid scalp-fit thresholds failed'))
    pd.DataFrame(rows).to_csv(out/'01_alignment.csv',index=False)
    hashes[str((out/'01_alignment.csv').resolve())]=digest(out/'01_alignment.csv')
    write_json(out/'02_coregistration_manifest.json',dict(arm=args.arm,anatomy=paths,input_sha256=hashes,
        protocol_sha256=fingerprint(dict(coregistration=c['coregistration'],initialization='three anatomical fiducials, no scaling',script=digest(Path(__file__)))),
        configuration=c,mri_registration=False))
    write_json(out/'03_review_REQUIRED.json',dict(coregistration_manifest_sha256=digest(out/'02_coregistration_manifest.json'),
        anatomy_review='PENDING',subjects={s:'PENDING' for s in c['participants']},
        instructions='Review surfaces/nesting/self-intersections, sources, fiducials and all sensor overlays with MNE GUI. Set anatomy_review and each eligible subject to PASS only after independent visual review. Failed numerical alignments cannot be overridden.'))


def forwards(args,c):
    import mne
    doc=verify_manifest(args.coregistration_root/'02_coregistration_manifest.json')
    review=json.loads(args.review.read_text())
    if doc['arm']!=args.arm or doc['configuration']!=c: raise ValueError('Arm/configuration mismatch')
    if review['coregistration_manifest_sha256']!=digest(args.coregistration_root/'02_coregistration_manifest.json') or review['anatomy_review']!='PASS':
        raise ValueError('Matching independently reviewed anatomy/coregistration required')
    paths=doc['anatomy'];rows=[];out=fresh(args.output_root);(out/'forwards').mkdir()
    for row in pd.read_csv(args.coregistration_root/'01_alignment.csv',keep_default_na=False).to_dict('records'):
        subject=row['test_subject']; result=dict(test_subject=subject,status='FAIL',failure_reason='ALIGNMENT_OR_REVIEW_FAIL',forward_path='',forward_sha256='')
        if row['status']=='PASS_NUMERICAL_REVIEW_REQUIRED' and review['subjects'].get(subject)=='PASS':
            try:
                native=mne.read_forward_solution(row['native_forward'],verbose=False)
                names=native['info']['ch_names'];xyz=np.array([ch['loc'][:3] for ch in native['info']['chs']])
                info=mne.create_info(names,1.,'eeg');info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names,xyz)),coord_frame='head'))
                f=mne.make_forward_solution(info,trans=row['transform'],src=paths['src'],bem=paths['bem'],meg=False,eeg=True,mindist=0.,n_jobs=args.n_jobs)
                if f['sol']['row_names']!=names or not np.isfinite(f['sol']['data']).all() or f['nsource']<=0: raise ValueError('Invalid lead field')
                if not np.allclose(xyz,[ch['loc'][:3] for ch in f['info']['chs']],rtol=0,atol=1e-12): raise ValueError('Electrodes changed')
                path=out/'forwards'/f'{subject}-fwd.fif';mne.write_forward_solution(path,f,overwrite=False)
                result.update(status='PASS',failure_reason='',forward_path=str(path.resolve()),forward_sha256=digest(path))
            except Exception as exc: result['failure_reason']=f'{type(exc).__name__}: {exc}'
        rows.append(result)
        pd.DataFrame(rows).to_csv(out/'01_forward_status.csv',index=False)
    hashes=dict(doc['input_sha256']);hashes[str(args.review.resolve())]=digest(args.review)
    write_json(out/'02_forward_manifest.json',dict(arm=args.arm,configuration=c,input_sha256=hashes,
        coregistration_protocol_sha256=doc['protocol_sha256'],bem=paths['bem'],mri_registration=False))


def reconstruct(args,c):
    nw=ROOT/'cross_subject_no_warp'
    solver=module(nw/'scripts/03_run_localization.py','average_seven_solver')
    fm=verify_manifest(args.forward_root/'02_forward_manifest.json')
    preparation=json.loads((args.preparation_root/'04_manifest.json').read_text())
    if not preparation['ready'] or preparation['configuration']!=c:
        raise ValueError('Matching stage 01 preparation snapshot required')
    for filename, expected in preparation['inputs'].items():
        if digest(Path(filename))!=expected: raise ValueError('Prepared schedule/settings changed')
    if fm['arm']!=args.arm or fm['configuration']!=c: raise ValueError('Forward arm/config mismatch')
    status=pd.read_csv(args.forward_root/'01_forward_status.csv',keep_default_na=False).set_index('test_subject')
    if set(status.index)!=set(c['participants']) or not status.index.is_unique: raise ValueError('Seven-subject ledger required')
    paths={s:Path(r.forward_path) for s,r in status.iterrows() if r.status=='PASS'}
    for s,path in paths.items():
        if digest(path)!=status.loc[s,'forward_sha256']: raise ValueError('Forward changed')
    source=nw/'source/02_parameter_grids';code=nw/'source/03_solver_code'
    grids=json.loads((source/'source_manifest.json').read_text()); code_doc=json.loads((code/'code_manifest.json').read_text())
    hashes=dict(fm['input_sha256'])
    for filename in [source/'source_manifest.json',code/'code_manifest.json',Path(solver.__file__),Path(__file__),args.forward_root/'02_forward_manifest.json',args.forward_root/'01_forward_status.csv',ROOT/c['schedule'],args.config,args.preparation_root/'04_manifest.json']:
        hashes[str(filename.resolve())]=digest(filename)
    for item in grids['files']:
        filename=source/item['copy']
        if digest(filename)!=item['sha256']: raise ValueError('Frozen grid changed')
        hashes[str(filename)]=item['sha256']
    for item in code_doc['files']:
        filename=code/item['path']
        if digest(filename)!=item['sha256']: raise ValueError('Frozen solver code changed')
        hashes[str(filename)]=item['sha256']
    configs={}
    for strategy,filename in [('LOSO','loso_selection.csv'),('Single-reference','single_selection.csv')]:
        path=ROOT/c['frozen_settings']/filename;hashes[str(path)]=digest(path)
        selected=pd.read_csv(path,dtype=str,keep_default_na=False)
        if len(selected)!=84 or not selected.selection_status.eq('SELECTED').all(): raise ValueError('Frozen selection coverage differs')
        if strategy=='Single-reference': selected['test_subject']=selected.reference_subject
        resolved,errors=solver.enrich_configs(selected,source,grids)
        if errors or len(resolved)!=84: raise ValueError(f'Cannot resolve frozen settings: {errors}')
        configs.update({(strategy,*k):v for k,v in resolved.items()})
    # BEM injection retains the original ECD fit and transform semantics.
    original_bem=solver.find_bem
    solver.find_bem=lambda root,subject: Path(fm['bem']) if subject==args.arm else original_bem(root,subject)
    class ArmEngine(solver.Engine):
        def prepare(self,job,cfg):
            return super().prepare(dict(job,model_subject=args.arm+':'+job['test_subject']),cfg)
        def forward(self,subject):
            if subject.startswith(args.arm+':'):
                if subject not in self.cache: self.cache[subject]=self.mne.read_forward_solution(paths[subject.split(':',1)[1]],verbose=False)
                return self.cache[subject]
            return super().forward(subject)
    jobs=[]
    for row in schedule(c).to_dict('records'):
        owner=row['test_subject'] if row['strategy']=='LOSO' else row['reference_subject']
        cfg=configs[(row['strategy'],row['method'],owner)]
        jobs.append(dict(row,arm=args.arm,model_subject=args.arm,condition='no_warp_'+args.arm,
                         configuration_key=cfg['configuration_key'],parameter_owner=owner,
                         configuration_parameters=json.dumps(cfg['parameters'],sort_keys=True)))
    # Freeze EEG, metadata and all native reference models before any solver starts.
    data_root=ROOT/c['dataset']
    for subject in c['participants']:
        native=data_root/'derivatives/sourcemodelling'/subject/'fwd'/f'{subject}_fwd.fif';hashes[str(native)]=digest(native)
        for filename in (data_root/'derivatives/epochs'/subject).rglob('*'):
            if filename.is_file(): hashes[str(filename)]=digest(filename)
    versions={}
    for name in ['mne','numpy','pandas','scipy','sesameeg','invertmeeg']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='not_installed'
    frozen=dict(input_sha256=hashes,forward_hashes={str(p):digest(p) for p in paths.values()},configuration=c,arm=args.arm,versions=versions)
    out=args.output_root
    if out.exists() and not args.resume: raise FileExistsError('Choose fresh output or explicit --resume')
    with solver.lock_output(out):
        manifest=out/'01_run_manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=frozen: raise ValueError('Resume input fingerprint mismatch')
        else: solver.atomic_json(manifest,frozen)
        ledger=out/'02_results.csv'
        records=pd.read_csv(ledger,keep_default_na=False).to_dict('records') if ledger.exists() else [dict(j,status='PENDING',failure_reason='') for j in jobs]
        if len(records)!=len(jobs) or {tuple(r[k] for k in KEY) for r in records}!={tuple(j[k] for k in KEY) for j in jobs}: raise ValueError('Checkpoint schedule differs')
        engine=ArmEngine(SimpleNamespace(dataset=data_root,code_source_root=code,quiet=True,montage_template='reference',bem_root=Path(fm['bem']).parent)) if args.execute else None
        count=0
        for row in records:
            if row['status']!='PENDING': continue
            if args.strategy and row['strategy'] != args.strategy:
                continue
            subject=row['test_subject']
            if subject not in paths:
                row.update(status='FAIL',alignment_status='FAIL',failure_reason=status.loc[subject,'failure_reason']);continue
            if not args.execute or (args.method and row['method']!=args.method) or (args.test_subject and subject!=args.test_subject) or (args.run and row['run']!=args.run): continue
            if args.limit and count>=args.limit: continue
            try:
                cfg=configs[(row['strategy'],row['method'],row['parameter_owner'])]
                row.update(engine.solve(row,cfg['parameters']))
                row.update(status='PASS',alignment_status='PASS',peak_selection_policy='known_stimulation_hemisphere_original',
                    electrode_geometry_sha256=digest(data_root/'derivatives/sourcemodelling'/subject/'fwd'/f'{subject}_fwd.fif'),
                    coregistration_protocol_sha256=fm['coregistration_protocol_sha256'],
                    preprocessing_sha256=fingerprint({k:row.get(k) for k in ['eeg_data_sha256','epoch_times_sha256','baseline_json','reference_channel_scope','covariance_method_selected']}),
                    reference_policy_sha256=fingerprint(dict(channels=c['channels'],preprocessing=c['preprocessing'])),
                    localization_error_valid=False)
            except Exception as exc:
                row.update(status='FAIL',alignment_status='PASS',failure_reason=f'{type(exc).__name__}: {exc}')
                with (out/'03_failures.log').open('a') as log: log.write(str({k:row[k] for k in KEY})+'\n'+traceback.format_exc())
            count+=1;solver.atomic_csv(ledger,pd.DataFrame(records))
        solver.atomic_csv(ledger,pd.DataFrame(records))
    print(f'{out}: attempted {count}; full scheduled ledger retained')


def export(args,c):
    frames=[]
    snapshots={}
    for arm in c['arms']:
        directory=args.reconstruction_root/arm
        verify_manifest(directory/'01_run_manifest.json')
        snapshots[str(directory/'02_results.csv')]=digest(directory/'02_results.csv')
        frame=pd.read_csv(directory/'02_results.csv',keep_default_na=False)
        if not frame.arm.eq(arm).all(): raise ValueError('Arm ledger mismatch')
        frame['solver_status']=frame.status
        frame.loc[frame.status.eq('PENDING'),'failure_reason']='NOT_RUN'
        frame['status']=frame.status.where(frame.status.isin(['PASS','FAIL']),'FAIL')
        for field in ['alignment_status','coordinate_frame','coordinate_units',*[f'estimated_head_{a}_mm' for a in 'xyz'],*FINGERPRINTS]:
            if field not in frame: frame[field]=''
        frames.append(frame)
    gt=ROOT/'cross_subject_fsaverage/outputs/07_evaluate_localization/02_ground_truth_coordinates.csv'
    refs=pd.read_csv(gt)
    if not refs.coordinate_audit.eq('PASS_NUMERICAL').all() or refs.duplicated(['test_subject','run']).any(): raise ValueError('Original HEAD reference audit failed')
    planned=set(map(tuple,schedule(c)[['test_subject','run']].drop_duplicates().to_numpy()))
    if set(map(tuple,refs[['test_subject','run']].to_numpy()))!=planned:
        raise ValueError('Original stimulation reference coverage differs')
    for frame in frames:
        passed=frame[frame.status.eq('PASS')].merge(refs,on=['test_subject','run'],validate='many_to_one')
        for actual, expected in [('stimulation_pair','gt_stimulation_pair'),('hemisphere','gt_hemisphere')]:
            if not passed.empty and (actual not in passed or not passed[actual].eq(passed[expected]).all()):
                raise ValueError('Original stimulation identity differs from reconstructed run')
    refs['coordinate_frame']='test_head';refs['coordinate_units']='mm';refs['provenance']=str(gt)+' sha256='+digest(gt)
    for filename, expected in snapshots.items():
        if digest(Path(filename))!=expected: raise ValueError('Reconstruction is still changing; stop writers before export')
    out=fresh(args.output_root)
    pd.concat(frames,ignore_index=True).to_csv(out/'01_two_arm_head_results.csv',index=False)
    refs.to_csv(out/'02_original_test_head_references.csv',index=False)
    write_json(out/'03_export_manifest.json',dict(input_sha256=snapshots | {str(gt):digest(gt)},configuration=c))
    from common import evaluate
    evaluate(c,out/'05_evaluate_no_warp',out/'01_two_arm_head_results.csv',out/'02_original_test_head_references.csv')


def main(stage):
    p=argparse.ArgumentParser(description=f'{stage}: explicit heavy execution only with --execute; no MRI warp in comparison')
    p.add_argument('--config',type=Path,default=EXP/'config/experiment.json')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--arm',choices=['average_seven','fsaverage'],default='average_seven')
    p.add_argument('--subjects-dir',type=Path,default=EXP/'work/freesurfer')
    p.add_argument('--template',type=Path)
    p.add_argument('--phase',choices=['segment','bem'],default='segment')
    p.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment')
    p.add_argument('--fsaverage-root',type=Path,default=Path.home()/'mne_data/MNE-fsaverage-data/fsaverage')
    p.add_argument('--anatomy-manifest',type=Path,default=EXP/'outputs/03_bem/03_anatomy.json')
    p.add_argument('--coregistration-root',type=Path)
    p.add_argument('--review',type=Path)
    p.add_argument('--forward-root',type=Path)
    p.add_argument('--reconstruction-root',type=Path,default=EXP/'outputs/04c_reconstruct')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--n-jobs',type=int,default=1)
    p.add_argument('--method');p.add_argument('--test-subject');p.add_argument('--run')
    if stage == '04c_reconstruct':
        p.add_argument('--strategy', choices=['LOSO', 'Single-reference'],
                       help='Limit reconstruction to one parameter selection strategy')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();c=config(args.config)
    args.output_root=args.output_root.expanduser() if args.output_root else None
    args.output_root=(args.output_root or EXP/'outputs'/stage/args.arm).resolve()
    args.coregistration_root=args.coregistration_root or EXP/'outputs/04a_coregister'/args.arm
    args.forward_root=args.forward_root or EXP/'outputs/04b_make_forward'/args.arm
    args.review=args.review or args.coregistration_root/'03_review_REQUIRED.json'
    if args.n_jobs<1 or args.limit<0: p.error('n-jobs must be positive and limit nonnegative')
    if stage not in ['04c_reconstruct','04d_export_evaluate'] and not args.execute:
        print(f'Plan only: {stage}. Add --execute to launch this stage. See docs/RUN.md.');return
    if EXP not in args.output_root.parents: p.error('Output must be inside the new experiment')
    if stage=='02_execute_template': run_template(args,c)
    elif stage=='03_execute_anatomy': anatomy(args,c)
    elif stage=='04a_coregister': coregister(args,c)
    elif stage=='04b_make_forward': forwards(args,c)
    elif stage=='04c_reconstruct': reconstruct(args,c)
    elif stage=='04d_export_evaluate': export(args,c)
