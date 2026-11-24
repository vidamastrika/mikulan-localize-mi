#!/usr/bin/env python3
"""Supplement Stage 1: intensity-unit audit, not spatial scaling or reconstruction."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from input_utils import MRI,apply,compare_samples,region_labels,support_states

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage1_original',HERE/'01_audit_inputs.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VAR=HERE.parent


def calibration(a,b,vertices):
    """Fit Pydeface = gain*maskface+offset on alternating vertex IDs.

    Entire vertices are split, retaining both depths together; held-out samples
    are not used in fitting. This aligns intensity conventions for diagnostics,
    not anatomy. It does not establish unchanged tissue or a valid segmentation.
    """
    a=np.asarray(a,float);b=np.asarray(b,float);vertices=np.asarray(vertices)
    finite=np.isfinite(a)&np.isfinite(b)
    parity=np.searchsorted(np.unique(vertices),vertices)%2
    train=finite&(parity==0);test=finite&(parity==1)
    if train.sum()<32 or test.sum()<32 or np.std(b[train])<1e-12:
        raise ValueError('Insufficient independent intensity calibration controls')
    gain,offset=np.linalg.lstsq(np.column_stack([b[train],np.ones(train.sum())]),a[train],rcond=None)[0]
    if not np.isfinite(gain) or not np.isfinite(offset) or gain<=0:raise ValueError('Invalid intensity calibration')
    return float(gain),float(offset),train,test


def adjusted_regions(py,mask,bridge,py_scale,mask_scale,gain,offset):
    to_surface=np.linalg.inv(bridge)@py.image.affine
    chunks={label:[[],[]] for label in ['facial_proxy','inferior_lateral_proxy','cranial_proxy','other_or_transition']}
    nx,ny,nz=py.data.shape
    xx,yy=np.meshgrid(np.arange(0,nx,2),np.arange(0,ny,2),indexing='ij')
    for z in range(0,nz,2):
        vox=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,z)])
        labels=region_labels(apply(to_surface,vox))
        a=py.data[::2,::2,z].ravel();b=mask.data[::2,::2,z].ravel()
        for label in chunks:
            keep=labels==label;chunks[label][0].append(a[keep]);chunks[label][1].append(b[keep])
    rows=[]
    for label,parts in chunks.items():
        a,b=(np.concatenate(part) for part in parts)
        finite=np.isfinite(a)&np.isfinite(b)
        for fraction in [.01,.05,.10]:
            states=support_states(a,b,py_scale,mask_scale,fraction)
            common=(states=='COMMON_SIGNAL')&finite
            metrics=compare_samples(a,gain*b+offset,common,common,py_scale)
            for state in ['COMMON_SIGNAL','MASKFACE_ONLY_SIGNAL_UNVERIFIED','PYDEFACE_ONLY_SIGNAL','BACKGROUND_OR_LOW_SIGNAL']:
                rows.append(dict(region_proxy=label,intensity_fraction_p99=fraction,state=state,
                    n_samples=int(((states==state)&finite).sum()),n_region_samples=len(a),sample_stride_voxels=2,
                    adjusted_common_absdiff_p95_fraction_py_p99=metrics['normalized_absdiff_p95'],
                    adjusted_common_pearson=metrics['pearson'],
                    interpretation='support counts use original within-image cutoffs; common-signal differences use diagnostic intensity calibration'))
    return rows


class IntensityView:
    """In-memory display adapter only. Does not mutate/write an MRI array."""
    def __init__(self,mri,gain,offset):
        self.mri=mri;self.gain=gain;self.offset=offset
        self.vmax=gain*mri.vmax+offset
    def slice(self,*args,**kwargs):
        values,extent=self.mri.slice(*args,**kwargs)
        return self.gain*values+self.offset,extent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1',type=Path,default=VAR/'outputs/01_input_feasibility/run_01')
    parser.add_argument('--output-root',type=Path,default=VAR/'outputs/01b_intensity_review/run_01')
    parser.add_argument('--dpi',type=int,default=220)
    args=parser.parse_args();out=args.output_root.resolve()
    if (VAR/'outputs').resolve() not in out.parents or out.exists():raise ValueError('Use a fresh directory within maskface_template/outputs')
    if args.dpi<180:parser.error('Use at least 180 DPI')
    hashes={}
    def use(path):
        p=Path(path).resolve()
        if str(p) not in hashes:hashes[str(p)]=base.sha(p)
        return p
    for path in [Path(__file__),HERE/'01_audit_inputs.py',HERE/'input_utils.py']:use(path)
    prior=json.loads(use(args.stage1/'manifest.json').read_text())
    if prior['status']!='COMPLETE':raise ValueError('Stage 1 must be complete')
    for k,v in prior['input_sha256'].items():
        if base.sha(k)!=v:raise ValueError(f'Stage 1 input changed: {k}')
        use(k)
    for k,v in prior['output_sha256'].items():
        if base.sha(args.stage1/k)!=v:raise ValueError(f'Stage 1 output changed: {k}')
        use(args.stage1/k)
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    manifest=dict(status='RUNNING',created_utc=datetime.now(timezone.utc).isoformat(),stage2_authorized=False,
        intensity_calibration='diagnostic display and metric adapter only',head_scaling=False,
        MRI_arrays_modified=False,MRI_files_written=False,registration=False,construction=False,FreeSurfer=False,forward=False,inverse=False)
    def save():manifest['input_sha256']=hashes;base.write_json(out/'manifest.json',manifest)
    save()
    inventory=pd.read_csv(use(args.stage1/'01_input_inventory.csv'))
    pairs=pd.read_csv(use(args.stage1/'02_pair_geometry.csv'))
    profiles=pd.read_csv(use(args.stage1/'04_layer_radial_samples.csv'))
    calibrations=[];layers=[];regions=[];storage=[]
    for subject in inventory.subject.unique():
        print('Supplemental intensity audit:',subject,flush=True)
        data=profiles[profiles.subject==subject]
        controls=data[(data.layer=='brain')&data.offset_mm.isin([-20,-10])&data.both_in_fov]
        a=controls.Pydeface_intensity.to_numpy(float);b=controls.maskface_intensity.to_numpy(float)
        gain,offset,train,test=calibration(a,b,controls.vertex.to_numpy())
        pair=pairs[pairs.subject==subject].iloc[0]
        scale=float(pair.Pydeface_positive_p99)
        fitted=gain*b+offset
        metrics=compare_samples(a,fitted,test,test,scale)
        fixed=compare_samples(a,b*255,test,test,scale)
        calibrations.append(dict(subject=subject,gain_maskface_to_Pydeface=gain,offset_Pydeface_units=offset,
            training_samples=int(train.sum()),heldout_samples=int(test.sum()),
            raw_heldout_pearson=compare_samples(a,b,test,test,scale)['pearson'],
            heldout_normalized_absdiff_p95=metrics['normalized_absdiff_p95'],
            fixed_255_heldout_normalized_absdiff_p95=fixed['normalized_absdiff_p95'],
            interpretation='Intensity-only convention model on brain inward proxies; held-out vertices; not proof of unchanged anatomy'))
        bridge=np.array(json.loads(pair.surface_to_scanner_ras_mm_json))
        paths={r.version:Path(r.path) for _,r in inventory[inventory.subject==subject].iterrows()}
        for version,path in paths.items():
            image=nib.load(use(path))
            storage.append(dict(subject=subject,version=version,stored_dtype=str(image.get_data_dtype()),
                dataobj_slope=float(image.dataobj.slope),dataobj_intercept=float(image.dataobj.inter),
                interpretation='nibabel already applies NIfTI slope/intercept; do not apply twice'))
        py=MRI(paths['Pydeface'],bridge);mask=MRI(paths['maskface'],bridge)
        view=IntensityView(mask,gain,offset)
        # Read native reference surfaces from already hashed Stage 1 inputs.
        candidates=[Path(p) for p in prior['input_sha256'] if Path(p).name==f'{subject}_outer_skin.surf.gii']
        if len(candidates)!=1:raise ValueError('Expected one native scalp input')
        folder=candidates[0].parent
        meshes=[]
        for layer,color in [('outer_skin','cyan'),('outer_skull','yellow'),('inner_skull','red'),('brain','lime')]:
            v,f,_,_,_=base.surface(use(folder/f'{subject}_{layer}.surf.gii'))
            meshes.append((layer,v,f,color))
        for name,focus in [('facial',np.array([0.,60.,-20.])),('cranial',np.array([-45.,0.,45.]))]:
            base.compare_figure(subject,py,view,meshes,focus,out/'figures'/f'{subject}_{name}_intensity_corrected.png',args.dpi,
                f'{name}; diagnostic intensity convention: Py ≈ {gain:.4f} × mask {offset:+.4f}; NO spatial scaling')
        for (layer,region),g in data.groupby(['layer','region_proxy']):
            for band,part in [('inward',g[g.offset_mm.between(-20,-5)]),('near_surface',g[g.offset_mm.between(-4,4)]),('outward',g[g.offset_mm.between(5,20)])]:
                aa=part.Pydeface_intensity.to_numpy(float);bb=gain*part.maskface_intensity.to_numpy(float)+offset
                valid=part.both_in_fov.to_numpy(bool)
                layers.append(dict(subject=subject,layer=layer,region_proxy=region,band=band,
                    **compare_samples(aa,bb,valid,valid,scale),interpretation='intensity-adjusted signal comparison; modifications and raw quantization may remain'))
        # Region support labels stay defined in original per-image intensity units.
        # Only paired common-signal DIFFERENCE metrics are recalculated.
        if bool(pair.same_voxel_grid):
            samples=adjusted_regions(py,mask,bridge,scale,float(pair.maskface_positive_p99),gain,offset)
            for r in samples:regions.append(dict(subject=subject,**r))
        fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
        for ax,layer in zip(axes.flat,['outer_skin','outer_skull','inner_skull','brain']):
            for region,color in [('facial_proxy','tab:orange'),('cranial_proxy','tab:blue'),('inferior_lateral_proxy','tab:green')]:
                g=data[(data.layer==layer)&(data.region_proxy==region)&data.both_in_fov].groupby('offset_mm')[['Pydeface_intensity','maskface_intensity']].mean()
                if len(g):
                    ax.plot(g.index,g.Pydeface_intensity,'--',color=color,label=region+' Pydeface')
                    ax.plot(g.index,gain*g.maskface_intensity+offset,color=color,label=region+' maskface adjusted')
            ax.axvline(0,color='gray',lw=.7);ax.set(title=layer,xlabel='radial offset (mm; negative inward)',ylabel='Pydeface intensity convention (diagnostic)')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'{subject}: intensity-adjusted profiles; original MRI files/geometry untouched',fontsize=14)
        fig.savefig(out/'figures'/f'{subject}_corrected_layer_profiles.png',dpi=args.dpi);plt.close(fig)
        del py,mask,view
        save()
    pd.DataFrame(calibrations).to_csv(out/'01_intensity_calibration_controls.csv',index=False)
    pd.DataFrame(storage).to_csv(out/'02_NIfTI_storage_conventions.csv',index=False)
    pd.DataFrame(layers).to_csv(out/'03_intensity_adjusted_layer_agreement.csv',index=False)
    # Keep the valid original support counts; explicitly label invalid raw-unit metrics.
    pd.DataFrame(regions).to_csv(out/'04_intensity_adjusted_regional_agreement.csv',index=False)
    report=['# Stage 1 supplemental intensity-convention review','',
        'Status: diagnostic correction complete; Stage 2 remains unauthorized. Original Stage 1 outputs are preserved.','',
        '## Why the supplement was required','',
        'Original Stage 1 compared raw MRI amplitudes with an approximately 255-fold convention difference. Its shared display window makes maskface panels nearly black; raw difference maps, raw exact equality and raw absolute-difference metrics cannot be interpreted as anatomical changes. Pair geometry, finite-input checks, within-image relative signal-support counts and correlations remain useful.','',
        '## Calibration and held-out controls','',pd.DataFrame(calibrations).to_string(index=False),'',
        'A single gain/offset is fitted using brain-boundary inward proxies from alternating reference vertices; the other vertices are held out, including both sampled depths. The fixed factor 255 is also reported as a separate diagnostic comparison. No calibration uses facial or EEG residual outcomes. Calibration is a display/metric adapter, not a spatial transform or intensity preprocessing for construction. It does not prove original anatomy or exclude masking effects.','',
        '## What to review','',
        'Corrected facial and cranial panels share a meaningful intensity window. Layer profiles and inward/near-surface/outward signal agreement retain actual local differences after accounting for intensity convention. Naturally dark skull/CSF, original surface uncertainty and possible masking modifications remain confounds.','',
        'NIfTI dataobj slope/intercept are inventoried; nibabel has already applied those conversions. Additional fitted gains are descriptive observed image relationships, not proof of a particular export operation.','',
        'Additional maskface-only signal still lacks an unmodified reference. Better coverage is not automatically authentic scalp or skull anatomy. The cited surface-masking method intentionally modifies boundary signal. FreeSurfer and BEM suitability still require review and subsequent experimental validation.','',
        'No MRI file or array was modified, no head was scaled, and no construction, registration, FreeSurfer, forward or inverse calculation was performed. Stage 2 is not opened by this supplement.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    for path,value in hashes.items():
        if base.sha(path)!=value:raise ValueError(f'Input changed: {path}')
    manifest.update(status='COMPLETE',software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,nibabel=nib.__version__),
        output_sha256={str(p.relative_to(out)):base.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    save();print(out,flush=True)


if __name__=='__main__':main()
