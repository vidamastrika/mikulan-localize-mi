#!/usr/bin/env python3
"""Read-only audit for Experiment C (target EEG + donor original forward).

Never reads target MRI, target forward, target anatomy, ground-truth locations,
or electrode-coordinate TSVs. Searches *channel metadata* TSV/JSON only.
Does not perform source localization or assert unverified reference/units.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import mne

DEFAULT_ROOT = Path.home() / 'Documents/DATASETS SOURCE LOCALIZATION/Localize-MI'

def write_csv(path, rows, headers):
    with path.open('w', newline='') as h:
        wr = csv.DictWriter(h, fieldnames=headers, extrasaction='ignore')
        wr.writeheader(); wr.writerows(rows)

def load_channel_order(path):
    if path is None: return None
    if path.suffix.lower() == '.json':
        val = json.loads(path.read_text())
        if isinstance(val, dict): val = val.get('channels', val.get('channel_names'))
        if not isinstance(val, list): raise ValueError('Channel-order JSON must be a list or {"channels": [...] }')
        return [str(v).strip() for v in val]
    if path.suffix.lower() == '.tsv':
        with path.open(newline='') as h: rows = list(csv.DictReader(h, delimiter='\t'))
        if not rows or 'name' not in rows[0]: raise ValueError('Channel-order TSV must have a name column in NPY array order')
        return [r['name'].strip() for r in rows]
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith('#')]

def load_bad_list(path):
    if path is None: return None
    if path.suffix.lower() == '.json':
        val = json.loads(path.read_text())
        if isinstance(val, dict): val = val.get('bads', val.get('bad_channels'))
        if not isinstance(val, list): raise ValueError('Bad-channel JSON must be a list or {"bads": [...] }')
        return [str(x).strip() for x in val]
    return [v.strip() for v in re.split(r'[,\s]+', path.read_text()) if v.strip() and not v.startswith('#')]

def find_sidecars(root, target, run):
    """Search only permitted EEG metadata, never *_electrodes.tsv."""
    folder = root / 'derivatives' / 'epochs' / target / 'eeg'
    if not folder.exists(): return []
    candidate = list(folder.glob('*.json')) + list(folder.glob('*channels.tsv'))
    candidate += list((root / target / 'eeg').glob('*.json')) if (root / target / 'eeg').exists() else []
    candidate += list((root / target / 'eeg').glob('*channels.tsv')) if (root / target / 'eeg').exists() else []
    relevant = []
    for f in candidate:
        name = f.name
        if 'electrodes' in name.lower() or 'coordsystem' in name.lower(): continue
        if f'run-' in name and f'run-{run}' not in name: continue
        if f'task-' in name and 'seegstim' not in name: continue
        relevant.append(f)
    return sorted(set(relevant))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--target', default='sub-02')
    p.add_argument('--donor', default='sub-01')
    p.add_argument('--run', default='01')
    p.add_argument('--output-dir', type=Path, default=Path('cross_subject/outputs/04b_preprocessing_validation'))
    p.add_argument('--target-bads-file', type=Path, help='Verified bad channel names (one per line or comma separated, or JSON list)')
    p.add_argument('--channel-order-file', type=Path, help='Verified NPY axis-1 order, one name per line / JSON / TSV')
    p.add_argument('--preprocessing-root', type=Path, help='Optional original preprocessing code folder to scan for relevant operations')
    p.add_argument('--known-units', choices=['V','mV','uV'], help='Set ONLY after independently verifying NPY signal units')
    p.add_argument('--known-reference', help='Reference method verified from dataset documentation or preprocessing code')
    args=p.parse_args()
    root=args.dataset_root.expanduser().resolve(); run=f'{int(args.run):02d}'
    eeg=root/'derivatives'/'epochs'/args.target/'eeg'/f'{args.target}_task-seegstim_run-{run}_epochs.npy'
    ff=root/'derivatives'/'sourcemodelling'/args.donor/'fwd'/f'{args.donor}_fwd.fif'
    for f in [eeg,ff]:
        if not f.is_file(): p.error(f'Missing: {f}')
    data=np.load(eeg, mmap_mode='r',allow_pickle=False)
    if data.ndim!=3: p.error('Expected epochs array [n_epochs,n_channels,n_samples]')
    fwd=mne.read_forward_solution(ff,verbose='ERROR')
    names=list(fwd['info']['ch_names']); donor_bads=set(fwd['info']['bads'])
    order=load_channel_order(args.channel_order_file)
    if order is not None and (len(order)!=data.shape[1] or len(set(order))!=len(order)):
        p.error('Channel-order file must uniquely label every NPY column')
    target_bads=load_bad_list(args.target_bads_file)
    if target_bads is not None and (order is None):
        p.error('--target-bads-file requires --channel-order-file to map labels to NPY channels safely')
    if target_bads is not None and (set(target_bads)-set(order)):
        p.error(f'Bad-channel names not in channel order: {sorted(set(target_bads)-set(order))}')
    checks=[]
    def record(name, status, evidence):
        checks.append({'check':name,'status':status,'evidence':str(evidence)})
    record('EEG dimensions','PASS',f'{list(data.shape)}')
    record('Channel count compatible','PASS' if data.shape[1]==len(names) else 'FAIL',f'EEG={data.shape[1]}, donor={len(names)}')
    record('Donor leadfield dimensions','PASS' if fwd['sol']['data'].shape[0]==len(names) else 'FAIL',str(fwd['sol']['data'].shape))
    record('Finite EEG signals','PASS' if np.isfinite(data).all() else 'FAIL','All epoch values finite' if np.isfinite(data).all() else 'NaN or infinity present')
    record('Verified NPY channel order','PASS' if order is not None else 'UNVERIFIED',str(args.channel_order_file) if order is not None else 'NPY has no channel names; supply preprocessing-derived --channel-order-file')
    record('Channel names match donor','PASS' if order==names else ('FAIL' if order is not None else 'UNVERIFIED'), 'Same order' if order==names else 'Mapping not established or differs; do not assume direct indexing')
    record('Verified target run bad channels','PASS' if target_bads is not None else 'UNVERIFIED',str(args.target_bads_file) if target_bads is not None else 'Supply target run-specific list from original processing')
    record('Verified EEG units','PASS' if args.known_units else 'UNVERIFIED',args.known_units or 'Amplitude alone cannot prove units')
    record('Verified reference','PASS' if args.known_reference else 'UNVERIFIED',args.known_reference or 'Requires original recording/preprocessing evidence')
    # Explicitly do not import donor bads into the target's good-channel definition.
    selection=[]
    if order is not None:
        for i,n in enumerate(order):
            bad_target=(n in set(target_bads)) if target_bads is not None else None
            selection.append({'npy_channel_index_0based':i,'npy_channel_name':n,
                              'in_donor_forward':n in names,
                              'bad_in_target':bad_target if bad_target is not None else 'UNVERIFIED',
                              'bad_in_donor_historical':n in donor_bads,
                              'use_in_experiment_c': (not bad_target and n in names) if bad_target is not None else 'UNVERIFIED'})
    q=np.nanpercentile(np.abs(data), [50,95,99,99.9,100]).tolist()
    amplitude={'abs_value_quantiles_native_units':dict(zip(['p50','p95','p99','p99_9','max'],map(float,q))),
               'data_units_verified':args.known_units,
               'scale_to_volts_if_verified':{'V':1.0,'mV':1e-3,'uV':1e-6}.get(args.known_units),
               'caution':'Distribution is a sanity check, not evidence of units.'}
    metadata=[]
    for f in find_sidecars(root,args.target,run):
        entry={'path':str(f),'kind':f.suffix}
        if f.suffix=='.json':
            try:
                obj=json.loads(f.read_text())
                if isinstance(obj,dict):
                    entry['relevant_fields']={k:v for k,v in obj.items() if any(x in k.lower() for x in ['reference','unit','sampling','channel','filter','bad'])}
            except (ValueError,OSError) as ex: entry['error']=str(ex)
        elif f.name.endswith('channels.tsv'):
            try:
                with f.open(newline='') as h: rows=list(csv.DictReader(h,delimiter='\t'))
                entry['rows']=len(rows)
                entry['columns']=list(rows[0]) if rows else []
                if rows and 'status' in rows[0]:
                    entry['marked_bad']=[r.get('name') for r in rows if r.get('status','').strip().lower()=='bad']
                entry['warning']='Sidecar may be applicable, but is not proven to match NPY order/processing without code inspection.'
            except (ValueError,OSError) as ex: entry['error']=str(ex)
        metadata.append(entry)
    code_hits=[]
    if args.preprocessing_root:
        base=args.preprocessing_root.expanduser().resolve()
        if not base.is_dir():p.error(f'Not a directory: {base}')
        pattern=re.compile(r'set_eeg_reference|custom_ref_applied|average.ref|reference|\bbads\b|\bchannels\b|\.get_data\(|\.save\(|np\.save\(|\b1e-6\b|\b1e6\b|\bmicrovolt\b|\bunits\b',re.I)
        for path in sorted(base.rglob('*.py')):
            if len(code_hits)>=120:break
            try:
                for lineno,line in enumerate(path.read_text(errors='replace').splitlines(),1):
                    if pattern.search(line):
                        code_hits.append({'file':str(path),'line':lineno,'text':line.strip()[:250]})
                        if len(code_hits)>=120:break
            except OSError:continue
    out=args.output_dir.expanduser().resolve()/f'{args.target}_with_{args.donor}_run-{run}'
    out.mkdir(parents=True,exist_ok=True)
    report={'experiment':'C MRI-free digitizer-free audit','target':args.target,'donor':args.donor,'run':run,
            'data_inputs':{'target_epoch_array':str(eeg),'donor_forward':str(ff)},
            'forbidden_inputs_not_used':['target MRI','target forward FIF','target anatomical meshes','target electrode TSV','target ground truth'],
            'target_epoch_shape':list(data.shape),'donor_forward_channels':len(names),
            'donor_historical_bads':sorted(donor_bads),
            'target_bads':target_bads,
            'channel_order_status':'verified-file-supplied' if order else 'UNVERIFIED',
            'amplitudes':amplitude,'metadata_candidates':metadata,'preprocessing_code_hits':code_hits,
            'checks':checks,
            'ready_for_source_localization':all(x['status']=='PASS' for x in checks),
            'interpretation':'Donor historical bad channels are not automatically excluded. With verified target bad channels and a valid channel mapping, select target good channels and subset/reorder donor forward consistently.',
            'important':'Supplying manual options records user-supplied claims; check source evidence before treating them as verified.'}
    (out/'preprocessing_validation.json').write_text(json.dumps(report,indent=2,default=str))
    write_csv(out/'validation_checks.csv',checks,['check','status','evidence'])
    if selection:write_csv(out/'channel_selection.csv',selection,['npy_channel_index_0based','npy_channel_name','in_donor_forward','bad_in_target','bad_in_donor_historical','use_in_experiment_c'])
    if code_hits:write_csv(out/'preprocessing_code_hits.csv',code_hits,['file','line','text'])
    print(f'Experiment C 04b: {args.target} run-{run}, donor={args.donor}')
    for item in checks: print(f'  {item["status"]:10s} {item["check"]}: {item["evidence"]}')
    print(f'Output: {out}')
    print('UNVERIFIED items require dataset/preprocessing evidence; this script does not infer channel metadata from NPY values.')

if __name__=='__main__':main()
