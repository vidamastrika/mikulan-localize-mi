#!/usr/bin/env python3
"""Optional Script 03 sensitivity: one unrestricted-hemisphere Gamma-MAP peak.

This is NOT a correction to the primary LOSO benchmark. Its only purpose is to
reconstruct the original Gamma-MAP sub-05/run-08/reference-sub-01 CROSS job and
report the strongest absolute source across BOTH hemispheres. The inverse
parameters, EEG, reference forward, montage, and solver are unchanged.

Commands (from project root; .venv active):
  python cross_subject_no_warp/scripts/03_wholebrain_gamma_sensitivity.py reconstruct
  python cross_subject_no_warp/scripts/03_wholebrain_gamma_sensitivity.py merge

Source: cross_subject_no_warp/scripts/03_run_localization.py (patched Script 03).
Primary results: cross_subject_no_warp/outputs/03_reference_based_final/
Separate output: cross_subject_no_warp/outputs/03_gamma_wholebrain_sensitivity/

Reconstruction may warn 'Convergence NOT reached'; a PASS here means a valid
nonzero peak was found, NOT that the Gamma-MAP optimizer converged. The resulting
coordinate remains in reference model HEAD space, not registered to test anatomy.
The merged 1464-row variant is clearly marked and MUST NOT replace the primary.
No solver or parameter changes are applied to any other experiment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path('cross_subject_no_warp')
ORIGINAL = HERE / 'scripts/03_run_localization.py'
PRIMARY = HERE / 'outputs/03_reference_based_final/01_execution_status.csv'
OUT = HERE / 'outputs/03_gamma_wholebrain_sensitivity'
KEY = ('Gamma-MAP', 'sub-05', 'run-08', 'single_reference_no_warp')
KEYS = ['method','test_subject','run','condition']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def key(row):
    return tuple(row[k] for k in KEYS)

def reconstruct():
    if not ORIGINAL.is_file():raise FileNotFoundError(ORIGINAL)
    if not PRIMARY.is_file():raise FileNotFoundError(PRIMARY)
    baseline=read_csv(PRIMARY)
    hits=[r for r in baseline if key(r)==KEY]
    if len(hits)!=1 or hits[0]['status']!='FAIL':
        raise ValueError('Primary dataset must contain exactly one matching FAIL')
    if (OUT/'01_execution_status.csv').exists():
        raise FileExistsError('Sensitivity reconstruction already exists; refusing to overwrite')
    spec=importlib.util.spec_from_file_location('gamma_source03',ORIGINAL)
    src=importlib.util.module_from_spec(spec);spec.loader.exec_module(src)
    def unrestricted_peak(stc,fwd,hemisphere,mne):
        import numpy as np
        from mne.io.constants import FIFF
        vertices=[np.asarray(v,dtype=int) for v in stc.vertices]
        if len(vertices)!=2 or sum(map(len,vertices))!=stc.data.shape[0]:
            raise ValueError('Unexpected cortical vertex layout')
        if stc.data.ndim!=2 or not stc.data.size or not np.isfinite(stc.data).all():
            raise ValueError('Empty/non-finite source estimate')
        if not np.any(np.abs(stc.data)>0):raise ValueError('All-zero source estimate')
        row,tidx=np.unravel_index(np.abs(stc.data).argmax(),stc.data.shape)
        hi=0 if row<len(vertices[0]) else 1
        idx=row if hi==0 else row-len(vertices[0])
        vertex=int(vertices[hi][idx]); tm=float(stc.times[tidx]); found='lh' if hi==0 else 'rh'
        source=fwd['src'][hi]
        xyz=np.asarray(source['rr'][vertex],float)
        frame=int(source['coord_frame'])
        if frame==int(FIFF.FIFFV_COORD_MRI):
            transform=fwd['mri_head_t']
            if int(transform['from'])==int(FIFF.FIFFV_COORD_HEAD):
                transform=mne.transforms.invert_transform(transform)
            if int(transform['from'])!=int(FIFF.FIFFV_COORD_MRI) or int(transform['to'])!=int(FIFF.FIFFV_COORD_HEAD):
                raise ValueError('Invalid MRI-to-HEAD transform')
            xyz=mne.transforms.apply_trans(transform,xyz)
        elif frame!=int(FIFF.FIFFV_COORD_HEAD):
            raise ValueError('Unsupported source coordinate frame')
        print(f'WHOLE-BRAIN PEAK: requested={hemisphere}; found={found}; vertex={vertex}; time={tm:.6f}s; '
              f'active_vertices={[len(v) for v in vertices]}',flush=True)
        return xyz,vertex,tm
    src.peak_head=unrestricted_peak
    sys.argv=[str(ORIGINAL),'--phase','reconstruct','--montage-template','reference',
              '--method','Gamma-MAP','--test-subject','sub-05','--run','run-08',
              '--condition','single_reference_no_warp','--output-root',str(OUT),'--quiet']
    src.main()
    rows=read_csv(OUT/'01_execution_status.csv')
    if len(rows)!=1 or key(rows[0])!=KEY or rows[0]['status']!='PASS':
        raise RuntimeError('No valid whole-brain result; do not merge')
    (OUT/'12_wholebrain_override_manifest.json').write_text(json.dumps({
       'policy':'WHOLE_BRAIN_PEAK_SENSITIVITY_ONLY',
       'only_job':dict(zip(KEYS,KEY)),
       'primary_status_sha256':digest(PRIMARY),
       'base_script_sha256':digest(ORIGINAL),
       'wrapper_script_sha256':digest(__file__),
       'primary_hemisphere':hits[0].get('hemisphere'),
       'method':'same selected LOSO configuration and source solver; unrestricted absolute peak from BOTH hemispheres',
       'important':'Optimizer convergence warning must be reviewed; PASS is not convergence or anatomical validity',
    },indent=2)+'\n')

def merge():
    result=OUT/'01_execution_status.csv'
    manifest=OUT/'12_wholebrain_override_manifest.json'
    if not result.is_file() or not manifest.is_file():raise FileNotFoundError('Run reconstruct first')
    source=json.loads(manifest.read_text())
    if source['primary_status_sha256']!=digest(PRIMARY):raise ValueError('Primary CSV changed since reconstruction')
    primary=read_csv(PRIMARY); supplemental=read_csv(result)
    if len(primary)!=1464 or len({key(r) for r in primary})!=1464:
        raise ValueError('Primary result must contain 1464 unique records')
    if len(supplemental)!=1 or key(supplemental[0])!=KEY or supplemental[0]['status']!='PASS':
        raise ValueError('Exactly one successful targeted supplemental result required')
    old=next(r for r in primary if key(r)==KEY)
    new=supplemental[0]
    if old['status']!='FAIL' or old.get('reference_subject')!='sub-01':raise ValueError('Unexpected old result')
    # FAIL rows can lack fields populated only after successful reconstruction.
    # Never accept a conflicting non-empty field; verify the reconstructed
    # stimulation hemisphere against the source ground-truth convention.
    for fld in ('configuration_key','configuration_parameters','reference_subject','model_subject'):
        if old.get(fld) not in (None, '') and old.get(fld)!=new.get(fld):
            raise ValueError(f'Mismatched retained setting {fld}: {old.get(fld)!r} vs {new.get(fld)!r}')
    old_hemi=(old.get('hemisphere') or '').strip()
    new_hemi=(new.get('hemisphere') or '').strip()
    if new_hemi!='lh':
        raise ValueError(f'Unexpected stimulation hemisphere for sub-05 run-08: {new_hemi!r}; expected lh')
    if old_hemi and old_hemi!=new_hemi:
        raise ValueError(f'Conflicting recorded stimulation hemisphere: {old_hemi!r} vs {new_hemi!r}')
    if (new.get('hemisphere_constraint_applied') or '').strip().lower() not in ('true','1'):
        # The wrapper changes peak extraction only; base reconstruction metadata
        # may still say constrained=True. This is overwritten below.
        raise ValueError('Unexpected base reconstruction hemisphere metadata')
    result_rows=[]
    for r in primary:
        out=dict(r)
        out['peak_selection_policy']='known_stimulation_hemisphere'
        out['sensitivity_override']='false'
        if key(r)==KEY:
            out.update(new)
            out['hemisphere_constraint_applied']='False'
            out['selection_criterion']='maximum_absolute_amplitude_whole_brain'
            out['peak_selection_policy']='whole_brain_absolute_peak'
            out['sensitivity_override']='true'
        result_rows.append(out)
    if sum(r['status']=='PASS' for r in result_rows)!=1464:raise ValueError('Unexpected PASS count')
    columns=list(primary[0])+['peak_selection_policy','sensitivity_override']
    target=OUT/'01_execution_status_wholebrain_variant.csv'
    if target.exists():raise FileExistsError(target)
    write_csv(target,result_rows,columns)
    (OUT/'13_variant_README.txt').write_text(
        'SENSITIVITY ONLY. 1463 original records plus ONE Gamma-MAP sub-05 run-08 cross '
        'record obtained using unrestricted whole-brain peak selection. All 1464 rows '
        'have PASS status but ONE row uses a different peak-selection rule. This must '
        'not replace the primary 1463 PASS / 1 FAIL result. Do not interpret as '
        'equivalent comparison or evidence of optimization convergence.\n')
    (OUT/'14_variant_provenance.json').write_text(json.dumps({
       'primary_csv':str(PRIMARY),'primary_sha256':digest(PRIMARY),
       'supplement_csv':str(result),'supplement_sha256':digest(result),
       'variant_csv':str(target),'variant_sha256':digest(target),
       'override_key':dict(zip(KEYS,KEY)),
       'original_failure_reason':old.get('failure_reason',''),
       'new_peak_selection_policy':'whole_brain_absolute_peak',
       'warning':'Non-convergence may persist; no registration/warping performed',
    },indent=2)+'\n')
    print('Wrote sensitivity-only variant:',target,'(1464 PASS; 1 nonstandard peak policy)')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['reconstruct','merge'])
    args=p.parse_args()
    if args.action=='reconstruct':reconstruct()
    else:merge()
