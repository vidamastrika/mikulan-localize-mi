#!/usr/bin/env python3
"""Safely replace only Script-35 LCMV CSV rows with validated reproduction data.

Dry run by default. --apply writes both merged CSVs after backing up originals.
No inverse solutions are recomputed. Other method rows are preserved exactly.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, shutil, tempfile
from pathlib import Path
from collections import Counter

METHODS = {'Continuous-ECD', 'RAP-MUSIC', 'LCMV', 'MxNE'}
EXEC_KEY = ('method', 'test_subject', 'reference_subject', 'run')
EVAL_KEY = EXEC_KEY
FILES = ('01_execution_status.csv', '02_localization_errors.csv')

def read_csv(path):
    if not path.is_file(): raise ValueError(f'Missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        if not r.fieldnames: raise ValueError(f'Empty CSV: {path}')
        rows=list(r)
        if any(None in row for row in rows): raise ValueError(f'Malformed CSV: {path}')
        return list(r.fieldnames),rows

def key(row):return tuple(row[c] for c in EXEC_KEY)
def assert_unique(rows,name):
    keys=[key(r) for r in rows]
    if len(set(keys))!=len(keys):
        duplicates=[k for k,n in Counter(keys).items() if n!=1]
        raise ValueError(f'{name} duplicate keys: {duplicates[:4]}')

def assert_schedule(rows,name,expected_per_method):
    assert_unique(rows,name)
    counts=Counter(r['method'] for r in rows)
    if set(counts)!=METHODS or any(counts[m]!=expected_per_method for m in METHODS):
        raise ValueError(f'{name} must contain all four methods with {expected_per_method} each; got {dict(counts)}')
    if any(r.get('test_subject','')=='' or r.get('reference_subject','')=='' or r.get('run','')=='' for r in rows):
        raise ValueError(f'{name}: blank subject/run in key')

def atomic_write(path,columns,rows):
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=columns,extrasaction='raise')
            writer.writeheader();writer.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-root',type=Path,default=Path('cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne'))
    p.add_argument('--repro-root',type=Path,default=Path('cross_subject/outputs/35_lcmv_reproduction_test'))
    p.add_argument('--apply',action='store_true',help='Execute the merge; without this, check only')
    a=p.parse_args()
    main_root=a.main_root.resolve();repro_root=a.repro_root.resolve()
    if main_root==repro_root:raise ValueError('Main and reproduction roots must differ')
    combined={}
    for filename, count in [(FILES[0],427),(FILES[1],366)]:
        columns,old=read_csv(main_root/filename)
        new_columns,new=read_csv(repro_root/filename)
        if columns!=new_columns:
            raise ValueError(f'{filename}: column layout differs between main and reproduction. Stop; manual schema audit needed.\nMain: {columns}\nRepro: {new_columns}')
        assert_schedule(old,'main '+filename,count)
        assert_unique(new,'repro '+filename)
        if len(new)!=count or any(r['method']!='LCMV' for r in new):
            raise ValueError(f'repro {filename}: expected exactly {count} LCMV rows, got {len(new)}')
        original_lcmv={key(r) for r in old if r['method']=='LCMV'}
        replacement_lcmv={key(r) for r in new}
        if original_lcmv!=replacement_lcmv:
            raise ValueError(f'{filename}: LCMV keys differ; missing={list(original_lcmv-replacement_lcmv)[:3]}, extras={list(replacement_lcmv-original_lcmv)[:3]}')
        if filename==FILES[0]:
            if not all(r['status']=='PASS' for r in new):raise ValueError('Reproduction reconstruction has non-PASS rows')
            within=sum(str(r.get('is_within','')).lower()=='true' for r in new)
            if within!=61:raise ValueError(f'Expected 61 LCMV within rows, got {within}')
            if any(r.get('lcmv_reg') not in ('0','0.0','0.00') or r.get('lcmv_data_covariance_method')!='diagonal_fixed' for r in new):
                raise ValueError('LCMV selected config differs (reg / covariance)')
        else:
            if not all(r['status']=='PASS_PROVISIONAL' for r in new):
                raise ValueError('Expected PASS_PROVISIONAL evaluation statuses')
        bykey={key(r):r for r in new}
        merged=[bykey[key(row)] if row['method']=='LCMV' else row for row in old]
        assert_schedule(merged,'merged '+filename,count)
        if [r for r in old if r['method']!='LCMV'] != [r for r in merged if r['method']!='LCMV']:
            raise ValueError(f'{filename}: other-method rows changed')
        combined[filename]=(columns, merged)
        print(f'{filename}: main={len(old)}; replace LCMV={len(new)}; retain other methods={len(old)-len(new)}; PASS')
    if not a.apply:
        print('\nDRY RUN ONLY. Nothing was modified. Add --apply to merge.')
        return
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    backup=main_root/'backups'/f'before_lcmv_merge_{stamp}'
    backup.mkdir(parents=True,exist_ok=False)
    metadata={'timestamp':stamp,'source_reproduction_root':str(repro_root),'original_sha256':{},'replacement_sha256':{}}
    for filename in FILES:
        src=main_root/filename
        shutil.copy2(src,backup/filename)
        metadata['original_sha256'][filename]=digest(src)
        metadata['replacement_sha256'][filename]=digest(repro_root/filename)
    (backup/'merge_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    try:
        for filename,(columns,merged) in combined.items():
            atomic_write(main_root/filename,columns,merged)
        for filename,(_,planned) in combined.items():
            _,actual=read_csv(main_root/filename)
            if actual!=planned:raise RuntimeError(f'Post-write verification failed: {filename}')
    except BaseException:
        for filename in FILES:
            shutil.copy2(backup/filename,main_root/filename)
        print('MERGE ERROR: original CSVs restored from backup.')
        raise
    print(f'\nMERGE SUCCESS. Backup: {backup}')
    print('Only 01_execution_status.csv and 02_localization_errors.csv changed.')
    print('NEXT: run Script 35 --phase analyze to refresh summaries and plots.')

if __name__=='__main__':main()
