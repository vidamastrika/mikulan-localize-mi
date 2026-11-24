#!/usr/bin/env python3
"""Evaluate completed new group MRI BEFORE FreeSurfer; no new registration."""
import argparse
import json
from pathlib import Path
import re
import nibabel as nib
import numpy as np
import pandas as pd
from input_utils import MRI,image_header,native_bridge
from stage2_common import ROOT,EXP,VAR,OLD,OLD_FS,Run,construction_contract,write_json
from stage2_qa import fs_bridge,overlay_figure,roi_metrics,warp_metrics,read_surface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--construction-root',type=Path,default=VAR/'outputs/03_template/run_04')
    p.add_argument('--output-root',type=Path,default=VAR/'outputs/04_template_MRI_QA/run_04')
    p.add_argument('--dpi',type=int,default=220)
    args=p.parse_args()
    if args.dpi<180:p.error('Use at least 180 DPI')
    with Run(args.output_root,'04_MRI_QA') as run:
        run.verify(args.construction_root,stage='03_CONSTRUCT');header=construction_contract(args.construction_root)
        (run.out/'figures').mkdir()
        original=run.use(OLD/'group_template0.nii.gz');new=run.use(args.construction_root/'group_template0.nii.gz')
        bridge=fs_bridge(run.use(OLD_FS/'mri/T1.mgz'))
        old_mri=MRI(original,bridge);new_mri=MRI(new,bridge)
        same_grid=old_mri.data.shape==new_mri.data.shape and np.allclose(old_mri.image.affine,new_mri.image.affine,atol=1e-4)
        if not same_grid:
            raise ValueError('Original and maskface average grids differ; common-coordinate overlays are withheld')
        entries=[('original Pydeface template',old_mri),('new maskface template',new_mri)]
        reference=[]
        for layer,color in [('outer_skin','cyan'),('outer_skull','yellow'),('inner_skull','red'),('brain','lime')]:
            v,f,meta,warn=read_surface(run.use(OLD_FS/'bem'/f'{layer}.surf'));reference.append(('original '+layer,v,f,color))
        for region,focus in [('facial',(0,55,-15)),('left_cranial',(-50,0,40)),('brain_control',(0,0,45))]:
            overlay_figure(entries,{entries[0][0]:reference,entries[1][0]:reference},run.out/'figures'/f'templates_{region}.png',
                'Original versus maskface template on common scanner-RAS plotting grid\nAll contours are ORIGINAL reference surfaces; grid correspondence does not establish anatomical registration.',args.dpi,focus)
        pd.DataFrame(roi_metrics(entries)).to_csv(run.out/'01_regional_MRI_cues.csv',index=False)
        intensity_rows=[]
        for label,path in [('original_pydeface',original),('new_maskface',new)]:
            image=nib.load(path);data=np.asarray(image.dataobj,dtype=np.float32)
            finite=data[np.isfinite(data)];positive=finite[finite>0]
            intensity_rows.append(dict(image=label,stored_dtype=str(image.get_data_dtype()),
                finite_fraction=float(len(finite)/data.size),positive_voxels=int(len(positive)),
                positive_p50=float(np.percentile(positive,50)) if len(positive) else None,
                positive_p99=float(np.percentile(positive,99)) if len(positive) else None,
                positive_p999=float(np.percentile(positive,99.9)) if len(positive) else None,
                maximum=float(np.max(finite)) if len(finite) else None,
                note='NIfTI data scaling is applied by nibabel; compare ranges before FreeSurfer import; no fitted gain applied'))
        pd.DataFrame(intensity_rows).to_csv(run.out/'05_intensity_ranges.csv',index=False)
        headers=[]
        for label,path in [('original',original),('new',new)]:headers.append(dict(image=label,path=str(path),**image_header(nib.load(path))))
        for path in sorted((args.construction_root/'intermediateTemplates').glob('*template0.nii.gz')):
            headers.append(dict(image=path.name,path=str(run.use(path)),**image_header(nib.load(path))))
        write_json(run.out/'MRI_headers.json',headers)
        warps=[warp_metrics(run.use(path)) for path in sorted(args.construction_root.glob('*Warp.nii.gz'))]
        pd.DataFrame(warps).to_csv(run.out/'02_coarse_warp_QA.csv',index=False)
        evidence=[]
        for i,line in enumerate((args.construction_root/'construction.log').read_text(errors='replace').splitlines(),1):
            if re.search(r'Interpolation type:|Laplacian|Sharpening|Averaging 7|Iteration .*completed|shapeupdate|[Cc]rop|[Pp]add',line):evidence.append(dict(line=i,text=line.strip()))
        pd.DataFrame(evidence).to_csv(run.out/'03_processing_evidence.csv',index=False)
        # Registered native RASTERS have known construction reference grids; no
        # raw native scalp is overlaid as if it were already in template space.
        warped=sorted(args.construction_root.glob('group_input*-WarpedToTemplate.nii.gz'))
        warped_rows=[]
        for path in warped:
            mri=MRI(run.use(path),bridge);label=path.name.split('modality0-')[-1].split('_mod-')[0]
            if len(label)>35:label=path.name[:35]
            overlay_figure([('final maskface template',new_mri),(label+' registered raster',mri)],{},run.out/'figures'/f'{path.name[:45]}_registered.png',
                'Construction-grid raster comparison; retained input may precede final shape update\nNo native-surface/electrode transform inferred.',args.dpi,(-45,35,15))
            for row in roi_metrics([('final maskface template',new_mri),(label,mri)]):warped_rows.append(row)
        pd.DataFrame(warped_rows).to_csv(run.out/'04_registered_raster_cues.csv',index=False)
        # Native comparisons use only established same-participant scanner bridges.
        inputs=json.loads((VAR/'config/variant.json').read_text())
        for subject in inputs['participants']:
            dataset=ROOT/inputs['dataset'];skin=run.use(dataset/'derivatives/sourcemodelling'/subject/'anat'/f'{subject}_outer_skin.surf.gii')
            v,f,meta,_=read_surface(skin);native_to_scanner,scanner,size=native_bridge(meta)
            py=run.use(dataset/subject/'anat'/f'{subject}_T1w.nii');mf=run.use(dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii')
            if tuple(size)!=nib.load(py).shape or not np.allclose(scanner,nib.load(py).affine,atol=1e-3):raise ValueError('Native coordinate bridge failed')
            overlay_figure([('native Pydeface',MRI(py,native_to_scanner)),('native maskface',MRI(mf,native_to_scanner))],
                {'native Pydeface':[('original native scalp',v,f,'cyan')],'native maskface':[('original native scalp',v,f,'cyan')]},
                run.out/'figures'/f'{subject}_native_reference.png','Same-participant native MRI/scalp reference only; no native-to-group correspondence assumed',args.dpi,(-45,55,15))
        nonfinite=int((~np.isfinite(new_mri.data)).sum())
        technical=bool(header['technical_header_valid'] and nonfinite==0 and np.any(new_mri.data>0)
                       and len(warped)==7 and same_grid)
        run.doc.update(pre_FreeSurfer_technical_ready=technical,original_and_new_same_grid=bool(same_grid),
            anatomical_validity='UNDETERMINED_REVIEW_REQUIRED',nonpositive_coarse_jacobian_samples=sum(r['nonpositive_samples'] for r in warps),
            comparison_frame='original FS surface RAS -> scanner RAS; no fitted intertemplate registration',template=str(new))
        write_json(run.out/'pre_FreeSurfer_QA.json',dict(technical_ready=technical,anatomical_review='PENDING',
            same_grid_as_original=bool(same_grid),notes='Inspect figures, regional cues, logs, FOV and warp screens before manually invoking FreeSurfer. No anatomical pass is granted.'))
        (run.out/'REPORT_stage2.md').write_text('# Pre-FreeSurfer template MRI QA\n\n'
            f'Technical readiness: {technical}. Same grid as original: {same_grid}.\n\n'
            'MRI figures use independent intensity windows; no Stage 1 fitted gains enter construction. Regional brain/face/left-boundary measurements are common-grid review cues, not verified voxelwise anatomical correspondence. Registered rasters may precede the final shape update. Native scalp overlays are confined to same-participant MRIs with checked volume geometry.\n\n'
            'Inspect templates_facial.png, templates_left_cranial.png, templates_brain_control.png and registered raster panels. Coarse field-only Jacobians and interpolation/averaging excerpts are retained; neither proves full registration validity. Successful ANTs completion does not establish anatomical validity.\n\n'
            'If the result is suitable for an exploratory reconstruction after review, follow root `STAGE2_RUN.md` and invoke 05_run_freesurfer.py manually. No FreeSurfer, scalp segmentation, surface correction or EEG computation was run here.\n')


if __name__=='__main__':main()
