#!/usr/bin/env python3
# OBJECTIVE / ASSUMPTIONS
# Reconstruct Localize-MI EEG with the common template/fsaverage forward from
# Script 02. No individual MRI, native forward, digitization, sensor TSV or
# stimulation coordinates are loaded. eN->EN remains ASSUMED_UNVERIFIED.
# Preserve 61 runs, 7 subjects, 12 methods, all frozen LOSO and Single Reference
# parameter/reference identities: 732 + 4,392 = 5,124 evaluations (588 sub-03).
# INPUTS
# Script 02 successful build/manifest and its pinned preparation snapshots;
# template nominal subsets; EEG epochs.npy, channels.tsv, epochs.tsv/json only.
# Channels TSV supplies identities/status/sampling rate, not coordinates.
# Frozen inverse helpers and solver dispatch are reused selectively via AST;
# their anatomy loaders, Engine.prepare/init and script entrypoints never run.
# OUTPUTS (new variant output root only)
# 01_execution_status.csv (all 5,124 jobs, PENDING/PASS/FAIL), full plan,
# frozen configuration JSON, immutable run fingerprint, summary, channel audit,
# EEG input inventory, failure log, source snapshots and README. Estimates in
# template HEAD mm and fsaverage MRI/surface RAS mm; no localization errors/STCs.
# PARAMETERS / VALIDATION
# --phase plan (default): validate hashes/design/settings and freeze inputs; no
# EEG array loading, covariance or inverse. All 61 EEG input files are SHA-pinned
# (streaming reads only) in both phases to prevent mixed inputs on resume.
# --phase reconstruct: --strategy/--method/--test-subject/--reference-subject/
# --run filters; --limit 0 means all pending matching jobs; --resume required
# for existing output. PASS and FAIL remain untouched; only PENDING is attempted.
# One writer enforced. Strict code/input/package fingerprint; never bypass a
# mismatch. Use a fresh output for changed code or diagnostic failure retries.
# EEG loading mirrors frozen io.py, baseline=None (released arrays already
# corrected); no resampling/filtering. Average reference BEFORE subset selection;
# bad flags from test EEG; nominal/replacement geometry from frozen template.
# Covariance -250..-50 ms auto, target -2..+2 ms; validate stored windows.
# Cortical peaks retain known stimulation hemisphere inferred from Description
# naming only (an additional informed prior); dipoles use max GOF. No truth point.
# SESAME retains prior capacity-only 100->256 compatibility adapter, original
# noise/seed/radius/probabilities; no installed package files are changed.
# DEPENDENCIES: existing NumPy/pandas/SciPy/MNE 1.12.1, sesameeg/invertmeeg.
# COMMANDS (repository root; manual execution):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/03_reconstruct.py --phase plan
# Pilot: all 12 methods, one LOSO run, same frozen full-plan checkpoint:
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/03_reconstruct.py \
#   --phase reconstruct --resume --strategy LOSO --test-subject sub-01 --run run-01 --limit 12 --quiet
# Full run AFTER inspecting pilot:
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/03_reconstruct.py \
#   --phase reconstruct --resume --quiet
# Exit 0 completed requested attempts/plan; 1 preflight or attempted-fit failure.
"""Frozen-parameter inverse reconstruction using template sensors and fsaverage."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

VARIANT = Path(__file__).resolve().parents[1]
KEY = ['strategy','method','reference_subject','test_subject','run']
METHODS = ['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC',
           'LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def atomic_csv(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    value.to_csv(temp,index=False)
    os.replace(temp,path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def selected_definitions(path, names, namespace):
    """Execute only explicit definitions, never module imports or entrypoints."""
    tree = ast.parse(path.read_text(),filename=str(path))
    nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes}==set(names),f'Missing frozen definitions in {path}')
    unit = ast.Module(body=nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit),str(path),'exec'),namespace)


def make_engine(sources, forward_path, bem_path, geometry, subsets, args):
    import mne
    env = dict(np=np,pd=pd,json=json,inspect=inspect,Path=Path,METHODS=METHODS,digest=digest)
    selected_definitions(sources['engine'],['boolean','peak_head'],env)
    # Include only the numerical dispatch methods: no original constructor,
    # data loader, prepare(), forward() or BEM search is even compiled here.
    tree = ast.parse(sources['engine'].read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
    methods = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'solve','sesame','champagne'}]
    require({n.name for n in methods}=={'solve','sesame','champagne'},'Frozen dispatch changed')
    base = ast.ClassDef(name='FrozenDispatch',bases=[],keywords=[],body=methods,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[base],type_ignores=[])),str(sources['engine']),'exec'),env)
    def template_bem(root, subject):
        require(subject=='fsaverage','Individual BEM lookup is forbidden')
        return bem_path
    env['find_bem'] = template_bem
    grid_env = dict(np=np)
    selected_definitions(sources['grid'],['positions_from_forward','fixed_spatial_subsets','replace_bad_nominal'],grid_env)
    pilot_env = dict(np=np,importlib=importlib)
    selected_definitions(sources['pilot'],['package_version','require_sesame','require_champagne_solver'],pilot_env)
    event_env = dict(np=np)
    selected_definitions(sources['io'],['_create_mne_events'],event_env)
    inverse = module(sources['inverse'],'template_frozen_inverse')
    capacity = module(sources['capacity'],'template_sesame_capacity')
    names = geometry.dataset_channel.tolist()
    positions = geometry[[f'template_head_{a}_m' for a in 'xyz']].to_numpy(float)
    positions_dict = dict(zip(names,positions))
    fake = {'info':{'ch_names':names,'chs':[{'loc':p} for p in positions]}}
    _,_,computed = grid_env['fixed_spatial_subsets'](fake)
    require(computed==subsets,'Frozen nominal subsets differ from original geometric algorithm')

    class TemplateEngine(env['FrozenDispatch']):
        def __init__(self):
            self.mne=mne;self.inverse=inverse;self.args=args
            self.common=mne.read_forward_solution(forward_path,verbose=False)
            self.last_run=None;self.last_loaded=None;self.capacity_info=None
            require(self.common['info']['ch_names']==names,'Common forward channel order changed')
            actual=np.array([c['loc'][:3] for c in self.common['info']['chs']])
            require(np.allclose(actual,positions,rtol=0,atol=1e-7),'Forward/template coordinates differ')
        def pilot19(self):
            return SimpleNamespace(**{k:pilot_env[k] for k in ['require_sesame','require_champagne_solver']})
        def sesame(self,evoked,forward,c):
            self.pilot19().require_sesame()
            if self.capacity_info is None:
                self.capacity_info=capacity.install()
            stc,extra=super().sesame(evoked,forward,c)
            extra['sesame_capacity_compatibility_json']=json.dumps(self.capacity_info,sort_keys=True)
            return stc,extra
        def load_eeg(self, job):
            key=(job['test_subject'],job['run'])
            if self.last_run==key:
                return self.last_loaded
            subject,run=key
            folder=args.dataset/'derivatives/epochs'/subject/'eeg'
            stem=f'{subject}_task-seegstim_{run}'
            data=np.load(folder/f'{stem}_epochs.npy',allow_pickle=False)
            channels=pd.read_csv(folder/f'{stem}_channels.tsv',sep='\t')
            events_table=pd.read_csv(folder/f'{stem}_epochs.tsv',sep='\t')
            metadata=json.loads((folder/f'{stem}_epochs.json').read_text())
            run_names=channels['name'].astype(str).tolist()
            require(run_names==names,'EEG channel identities/order differ from frozen template')
            require(data.ndim==3 and data.shape[:2]==(len(events_table),256),'EEG dimensions differ')
            require(np.isfinite(data).all(),'Nonfinite EEG samples')
            frequencies=channels['sampling_frequency'].astype(float).unique()
            zeros=events_table['zero_time'].astype(float).unique()
            require(len(frequencies)==len(zeros)==1 and np.isfinite(frequencies[0]) and frequencies[0]>0
                    and np.isfinite(zeros[0]),'Inconsistent EEG time metadata')
            info=mne.create_info(run_names,sfreq=float(frequencies[0]),ch_types=['eeg']*256,verbose=False)
            info.set_montage(mne.channels.make_dig_montage(ch_pos=positions_dict,coord_frame='head'),verbose=False)
            info['bads']=channels.loc[channels.status.astype(str).str.lower().eq('bad'),'name'].astype(str).tolist()
            info['description']=metadata.get('Description')
            events,event_id=event_env['_create_mne_events'](events_table)
            eeg=mne.EpochsArray(data,info,events=events,event_id=event_id,tmin=-float(zeros[0]),baseline=None,verbose=False)
            # Same naming convention as metrics.parse_stimulation_description;
            # no contacts/coordinates table is opened to obtain the hemisphere.
            match=re.search(r"([A-Za-z]+['’]?)(\d+)-(\d+)",str(metadata.get('Description')))
            require(match is not None,'Cannot infer known hemisphere from EEG Description')
            prefix,first,second=match.groups();prefix=prefix.replace('’',"'")
            pair=f'{prefix}{first}-{second}'
            self.last_run=key;self.last_loaded=(eeg,pair,'lh' if "'" in pair else 'rh')
            return self.last_loaded
        def prepare(self,job,cfg):
            loaded,pair,hemisphere=self.load_eeg(job)
            eeg=loaded.copy()
            good=[c for c in eeg.ch_names if c not in eeg.info['bads']]
            montage=str(cfg['montage'])
            nominal=good if montage=='all_good' else subsets[montage]
            actual,replacement=(good,[]) if montage=='all_good' else grid_env['replace_bad_nominal'](nominal,names,positions,good)
            eeg=self.inverse.apply_average_reference(eeg)
            chosen=eeg.copy().pick(actual)
            require(not chosen.info['bads'],'Bad channel survived selection')
            raw=np.ascontiguousarray(chosen.get_data(copy=True),dtype=np.float64)
            cov=self.inverse.estimate_noise_covariance(chosen,-.250,-.050,method='auto',verbose=not args.quiet)
            evoked=self.inverse.create_target_evoked(chosen,-.002,.002)
            meta=dict(stimulation_pair=pair,hemisphere=hemisphere,channels=len(actual),
                actual_channels_json=json.dumps(actual),nominal_channels_json=json.dumps(nominal),
                replacement_details_json=json.dumps(replacement,sort_keys=True),
                original_bad_channels_json=json.dumps(list(loaded.info['bads'])),
                reference_channel_scope='all loaded channels; MNE average-ref excludes marked bads',
                eeg_data_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                epoch_times_sha256=hashlib.sha256(np.ascontiguousarray(chosen.times).tobytes()).hexdigest(),
                sampling_frequency_hz=float(chosen.info['sfreq']),highpass_hz=float(chosen.info['highpass']),
                lowpass_hz=float(chosen.info['lowpass']),baseline_json=json.dumps(chosen.baseline),
                montage_template_subject='GSN-HydroCel-256_common_template',n_epochs=len(chosen),
                covariance_method_selected=str(cov.get('method','auto')))
            # Some solvers convert forward orientation in place: isolate jobs.
            return self.common.copy(),chosen,cov,evoked,meta
    return TemplateEngine()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['plan','reconstruct'],default='plan')
    p.add_argument('--forward-root',type=Path,default=VARIANT/'outputs/02_template_forward_assumed_v1')
    p.add_argument('--output-root',type=Path,default=VARIANT/'outputs/03_reconstruct_assumed_v1')
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI')
    p.add_argument('--strategy',choices=['LOSO','Single-reference'],action='append')
    p.add_argument('--method',choices=METHODS,action='append')
    for name in ['test-subject','reference-subject','run']:
        p.add_argument('--'+name,action='append')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--quiet',action='store_true')
    args=p.parse_args(argv)
    require(args.limit>=0,'Negative --limit')
    args.dataset=args.dataset.expanduser().resolve()
    froot=args.forward_root.expanduser().resolve();out=args.output_root.expanduser().resolve()
    require((VARIANT/'outputs').resolve() in out.parents,'Output must stay within this variant/outputs')
    require(VARIANT in froot.parents and out!=froot and froot not in out.parents and out not in froot.parents,'Output overlaps forward inputs')
    require(not out.exists() or args.resume,'Existing output requires --resume')
    if out.exists():
        require((out/'06_frozen_run_manifest.json').is_file(),'Existing output has no frozen run; use a fresh directory')
    hashes={}
    def verify(path,expected=None):
        path=Path(path).resolve();value=digest(path)
        require(expected is None or value==expected,f'Frozen input changed: {path}')
        hashes[str(path)]=value
        return path
    fm=json.loads(verify(froot/'01_forward_manifest.json').read_text())
    require(fm['status']=='PASS_EXPLORATORY_FORWARD_NUMERICAL_ONLY' and fm['phase']=='build','Successful Script 02 build required')
    require(importlib.metadata.version('mne')=='1.12.1','Reviewed MNE 1.12.1 environment required')
    for path,expected in fm['input_sha256'].items():
        verify(path,expected)
    forward_path=verify(fm['forward_path'],fm['forward_sha256'])
    prep_paths=[Path(x) for x in fm['input_sha256'] if Path(x).name=='01_experiment_manifest.json']
    require(len(prep_paths)==1,'Ambiguous preparation manifest')
    prep=json.loads(prep_paths[0].read_text())
    inventory=prep['source_inventory']
    def frozen_suffix(suffix):
        items=[x for x in inventory if x['copy_path'].endswith(suffix)]
        require(len(items)==1,f'Ambiguous frozen source: {suffix}')
        return verify(items[0]['copy_path'],items[0]['sha256'])
    sources=dict(engine=frozen_suffix('/policy_code/cross_subject_no_warp/scripts/03_run_localization.py'))
    base='/policy_code/cross_subject_no_warp/source/03_solver_code/'
    for role,suffix in [('grid','scripts/09_full_parameter_grid.py'),('pilot','scripts/19_next_method_pilot.py'),
                        ('io','src/localize_mi/io.py'),('inverse','src/localize_mi/inverse.py')]:
        sources[role]=frozen_suffix(base+suffix)
    sources['capacity']=verify(Path(__file__).with_name('sesame_capacity_compat.py'))
    verify(Path(__file__))
    anatomy={r['role']:r for r in prep['fsaverage_resources']}
    bem=verify(anatomy['bem_solution']['path'],anatomy['bem_solution']['sha256'])
    coords_path=verify(froot/'04_template_sensor_coordinates.csv')
    require(digest(coords_path)==digest(prep_paths[0].parent/'06_sensor_scalp_geometry_qa.csv'),'Copied sensor coordinates differ')
    geometry=pd.read_csv(coords_path)
    require(geometry.dataset_channel.tolist()==[f'e{i}' for i in range(1,257)],'Template channel order differs')
    coordinate_snapshot=frozen_suffix('/01_montage/fsaverage_geometry/fsaverage-trans.fif')
    subsets_path=verify(coordinate_snapshot.parents[1]/'template_nominal_subsets.json')
    subsets=json.loads(subsets_path.read_text())['subsets']
    require(set(subsets)=={'32','64','128'} and all(len(v)==int(k) and len(set(v))==len(v)
            and set(v)<=set(geometry.dataset_channel) for k,v in subsets.items()),'Invalid nominal subsets')
    configs=json.loads(frozen_suffix('/02_frozen_configurations/resolved_configurations.json').read_text())
    require(len(configs)==168,'Expected 168 resolved frozen configurations')
    plan_path=verify(froot/'02_frozen_reconstruction_schedule.csv')
    require(digest(plan_path)==digest(prep_paths[0].parent/'02_planned_reconstruction_schedule.csv'),'Frozen schedule copy differs')
    frame=pd.read_csv(plan_path,keep_default_na=False)
    require(len(frame)==5124 and not frame.duplicated(KEY).any()
            and frame.groupby('strategy').size().to_dict()=={'LOSO':732,'Single-reference':4392},'Invalid full plan')
    jobs=frame.to_dict('records')
    for job in jobs:
        owner=job['test_subject'] if job['strategy']=='LOSO' else job['reference_subject']
        require(job['parameter_owner']==owner and job['model_subject']=='fsaverage','Parameter/anatomy owner changed')
        cfg=configs['|'.join([job['strategy'],job['method'],owner])]
        require(cfg['configuration_key']==job['configuration_key'] and cfg['parameters']==json.loads(job['configuration_parameters']),'Frozen configuration differs from job schedule')
        for k,v in [('covariance_tmin_s',-.25),('covariance_tmax_s',-.05),('target_tmin_s',-.002),('target_tmax_s',.002)]:
            require(np.isclose(float(cfg['parameters'][k]),v),f'Unexpected preprocessing window: {k}')
        job.update(condition='template_montage_no_digitization',mapping_status='ASSUMED_UNVERIFIED')
    requested=[j for j in jobs if all(not getattr(args,k) or j[k] in getattr(args,k)
        for k in ['strategy','method','test_subject','reference_subject','run'])]
    require(requested,'Filters match no jobs')
    print('Hashing EEG inputs for all 61 runs (no array loading in plan mode)...',flush=True)
    eeg_inventory=[]
    for subject,run in sorted({(j['test_subject'],j['run']) for j in jobs}):
        for suffix in ['epochs.npy','channels.tsv','epochs.tsv','epochs.json']:
            path=verify(args.dataset/'derivatives/epochs'/subject/'eeg'/f'{subject}_task-seegstim_{run}_{suffix}')
            eeg_inventory.append(dict(test_subject=subject,run=run,path=str(path),sha256=hashes[str(path)]))
    versions={}
    for package in ['numpy','pandas','scipy','mne','sesameeg','invertmeeg']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not_installed'
    fingerprint=dict(input_sha256=hashes,versions=versions,dataset=str(args.dataset),
        mapping='ASSUMED_UNVERIFIED',montage='fixed template; frozen subsets; original replacement algorithm',
        peak_policy='original known hemisphere from Description; no truth coordinates',
        individual_geometry_used=False,localization_error_calculated=False,
        SESAME_capacity=256,phase_not_part_of_fingerprint=True)
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.writer.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process is writing this output')
        frozen=out/'06_frozen_run_manifest.json'
        if frozen.exists():
            require(json.loads(frozen.read_text())==fingerprint,'Resume inputs/code/environment differ from frozen run; use a fresh directory')
        else:
            atomic_json(frozen,fingerprint)
            for role,path in sources.items():
                target=out/'source_snapshots'/f'{role}_{path.name}'
                target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
            shutil.copy2(Path(__file__),out/'source_snapshots/03_reconstruct.py.txt')
        atomic_csv(out/'02_reconstruction_plan.csv',pd.DataFrame(jobs))
        atomic_json(out/'05_resolved_configurations.json',configs)
        atomic_csv(out/'08_eeg_input_inventory.csv',pd.DataFrame(eeg_inventory))
        path=out/'01_execution_status.csv'
        records=pd.read_csv(path,keep_default_na=False).to_dict('records') if path.exists() else [dict(j,status='PENDING',failure_reason='',runtime_seconds=0.0) for j in jobs]
        recorded={tuple(r[k] for k in KEY):r for r in records}
        require(len(recorded)==len(records)==5124 and set(recorded)=={tuple(j[k] for k in KEY) for j in jobs},'Checkpoint job identities differ')
        require(all(r['status'] in {'PENDING','PASS','FAIL'} for r in records),'Unknown checkpoint status')
        def checkpoint():
            result=pd.DataFrame(recorded.values())
            atomic_csv(path,result)
            atomic_csv(out/'07_execution_summary.csv',result.groupby(['strategy','method','status']).size().reset_index(name='count'))
            fields=KEY+['status','configuration_key','configuration_parameters','channels','actual_channels_json',
                'nominal_channels_json','original_bad_channels_json','replacement_details_json','eeg_data_sha256',
                'epoch_times_sha256','sampling_frequency_hz','highpass_hz','lowpass_hz','baseline_json','n_epochs',
                'covariance_method_selected','reference_channel_scope','montage_template_subject','failure_reason']
            atomic_csv(out/'09_channel_selection_audit.csv',result.reindex(columns=fields))
        checkpoint()
        (out/'README.md').write_text('Template/fsaverage inverse reconstruction\n\n'
            '01_execution_status.csv: all 5,124 jobs; only PENDING rows are attempted.\n'
            '02_reconstruction_plan.csv and 05_resolved_configurations.json: frozen design.\n'
            '06_frozen_run_manifest.json: strict resume input/code/environment fingerprint.\n'
            '07_execution_summary.csv: per-strategy/method/status counts.\n'
            '08_eeg_input_inventory.csv: all 61 runs, four EEG input files per run, SHA-256.\n'
            '09_channel_selection_audit.csv: actual/nominal/bad/replacement channels and EEG fingerprints.\n'
            'source_snapshots/: reused numerical code; anatomy loaders are excluded via AST.\n'
            '04_failure_tracebacks.log: attempted-fit failures, preserved on resume.\n\n'
            'PASS means solver completion only. Coordinates are template HEAD and fsaverage MRI RAS mm.\n'
            'No anatomical localization errors or individual sensor/anatomy inputs. Cortical peaks use known hemisphere.\n'
            'ASSUMED_UNVERIFIED physical channel correspondence; anatomy and sensor geometry both differ from no-warp.\n')
        if args.phase=='plan':
            print(f'PLAN_READY: {len(jobs)} jobs; no EEG epochs loaded or inverse computed. {out}')
            return 0
        engine_args=SimpleNamespace(dataset=args.dataset,quiet=args.quiet,bem_root=bem.parent)
        engine=make_engine(sources,forward_path,bem,geometry,subsets,engine_args)
        matrix=np.asarray(json.loads((prep_paths[0].parent/'07_coordinate_transform_provenance.json').read_text())['head_to_fsaverage_mri'],float)
        attempted=failed=0
        for job in requested:
            key=tuple(job[k] for k in KEY)
            if recorded[key]['status']!='PENDING':continue
            started=time.perf_counter();row=dict(job,status='FAIL',failure_reason='')
            try:
                # Detect EEG file changes since preflight before loading this run.
                if engine.last_run!=(job['test_subject'],job['run']):
                    for item in eeg_inventory:
                        if item['test_subject']==job['test_subject'] and item['run']==job['run']:
                            require(digest(item['path'])==item['sha256'],f'EEG input changed: {item["path"]}')
                cfg=configs['|'.join([job['strategy'],job['method'],job['parameter_owner']])]
                with open(os.devnull,'w') as sink, (contextlib.redirect_stdout(sink) if args.quiet else contextlib.nullcontext()):
                    result=engine.solve(job,cfg['parameters'])
                row.update(result)
                head=np.array([row[f'estimated_head_{a}_mm'] for a in 'xyz'])/1000
                mapped=(matrix[:3,:3]@head+matrix[:3,3])*1000
                for i,a in enumerate('xyz'):row[f'estimated_fsaverage_mri_{a}_mm']=float(mapped[i])
                row.update(status='PASS',coordinate_frame='template_head',
                    fsaverage_coordinate_frame='MRI surface RAS',localization_error_valid=False,
                    interpretation='EXPLORATORY_TEMPLATE_GEOMETRY_NO_ANATOMICAL_ERROR_EVALUATION',
                    peak_selection_policy='original_known_hemisphere_for_cortical_STCs_max_GOF_for_dipoles')
            except Exception as exc:
                failed+=1;row['failure_reason']=f'{type(exc).__name__}: {exc}'
                with (out/'04_failure_tracebacks.log').open('a') as log:
                    log.write(str(key)+'\n'+traceback.format_exc()+'\n')
            row.update(runtime_seconds=time.perf_counter()-started,forward_sha256=fm['forward_sha256'])
            recorded[key]=row;attempted+=1;checkpoint()
            print(f'{key}: {row["status"]} ({row["runtime_seconds"]:.1f}s) {row["failure_reason"]}',flush=True)
            if args.limit and attempted>=args.limit:break
        print(f'Attempted {attempted}; failures {failed}. Existing PASS/FAIL preserved. {out}')
        return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
