#!/usr/bin/env python3
"""Localize-MI SyN point-transform audit and *provisional* paired SESAME comparison.

Run at repository root. Reuses finished Script 19 SyN registration and Script 17
paired audit; does not rerun either registration or SESAME. Point direction is
verified against the ANTs-warped MRI (independent of source/ground truth).

Dependencies: numpy, pandas, scipy, nibabel, matplotlib, h5py and
antsApplyTransformsToPoints (available through micromamba env by default).

Scientific limitation: T1w electrode coordinates are provisionally assumed to
be NiBabel NIfTI RAS world coordinates, and image registration is an imperfect
estimate of correspondence. Do not describe error as publication-validated.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

FLIP = np.diag([-1.0, -1.0, 1.0])


def find_exe(name: str, env: str) -> str:
    direct = shutil.which(name)
    if direct:
        return direct
    micromamba = shutil.which('micromamba')
    if micromamba:
        r = subprocess.run([micromamba, 'run', '-n', env, 'sh', '-c', f'command -v {name}'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            lines = [line.strip() for line in r.stdout.splitlines() if line.strip()]
            if lines and Path(lines[-1]).is_file():
                return lines[-1]
    raise RuntimeError(f'Cannot find {name}. Check micromamba environment {env!r}.')


def ants_points(exe: str, ras_points_mm: np.ndarray, transforms: list[str]) -> np.ndarray:
    points = np.asarray(ras_points_mm, float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 RAS points in mm')
    lps = points @ FLIP.T
    with tempfile.TemporaryDirectory(prefix='localizemi_syn20_') as td:
        inp, out = Path(td)/'in.csv', Path(td)/'out.csv'
        pd.DataFrame(dict(x=lps[:, 0], y=lps[:, 1], z=lps[:, 2], t=np.zeros(len(lps)))).to_csv(
            inp, index=False, float_format='%.12f')
        cmd = [exe, '-d', '3', '-i', str(inp), '-o', str(out)]
        for tr in transforms:
            cmd += ['-t', str(tr)]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0 or not out.exists():
            raise RuntimeError(f'ANTs point transform failed: {completed.stderr}\n{completed.stdout}\n{cmd}')
        mapped = pd.read_csv(out)[['x', 'y', 'z']].to_numpy(float)
        if mapped.shape != points.shape or not np.isfinite(mapped).all():
            raise ValueError('ANTs returned unexpected number of finite points')
        return mapped @ FLIP.T


def h5_transform(dataset: Path, sub: str) -> np.ndarray:
    file = dataset/'derivatives'/'sourcemodelling'/sub/'xfm'/f'{sub}_from-head_to-surface.h5'
    with h5py.File(file) as h5:
        matrix = np.array(h5['trans'], float)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Unexpected head-to-surface transform')
    return matrix


def point_transform_mm(point: np.ndarray, affine_m: np.ndarray) -> np.ndarray:
    return nib.affines.apply_affine(affine_m, np.asarray(point, float)/1000.)*1000.


def get_contacts(dataset: Path, sub: str) -> tuple[list[str], np.ndarray]:
    path = dataset/'derivatives'/'epochs'/sub/'ieeg'/f'{sub}_task-seegstim_space-T1w_electrodes.tsv'
    table = pd.read_csv(path, sep='\t')
    points = table[['x', 'y', 'z']].to_numpy(float)*1000.
    if table.name.duplicated().any():
        raise ValueError('Duplicated electrode names')
    return table.name.astype(str).tolist(), points


def quality_from_fixed_points(fixed_ras: np.ndarray, warped_img, moving_img, mapped_ras: np.ndarray):
    fixed_vox = nib.affines.apply_affine(np.linalg.inv(warped_img.affine), fixed_ras)
    moving_vox = nib.affines.apply_affine(np.linalg.inv(moving_img.affine), mapped_ras)
    src = np.asarray(moving_img.dataobj, dtype=np.float32)
    warped = np.asarray(warped_img.dataobj, dtype=np.float32)
    valid = np.all((moving_vox >= 1) & (moving_vox < np.array(src.shape)-2), axis=1)
    if int(valid.sum()) < max(20, len(fixed_ras)//2):
        return {'n_valid': int(valid.sum()), 'rmse': None, 'mae': None, 'correlation': None}
    sampled = map_coordinates(src, moving_vox[valid].T, order=1, mode='constant', cval=0)
    expected = map_coordinates(warped, fixed_vox[valid].T, order=1, mode='nearest')
    rmse = float(np.sqrt(np.mean((sampled-expected)**2)))
    corr = float(np.corrcoef(sampled, expected)[0, 1]) if np.std(sampled) and np.std(expected) else None
    return {'n_valid': int(valid.sum()), 'rmse': rmse,
            'mae': float(np.mean(abs(sampled-expected))), 'correlation': corr}


def figure(target_img, truth_native, within_native, affine_native, syn_native,
           target_head_to_surface, errors, dest):
    native = np.vstack([truth_native, within_native, affine_native, syn_native])
    t1w = nib.affines.apply_affine(np.linalg.inv(target_head_to_surface), native/1000.)*1000.
    vox = nib.affines.apply_affine(np.linalg.inv(target_img.affine), t1w)
    arr = np.asarray(target_img.dataobj, dtype=np.float32)
    titles = ['Ground truth', f'Within: {errors[0]:.2f} mm',
              f'Affine: {errors[1]:.2f} mm', f'SyN: {errors[2]:.2f} mm']
    colors = ['#18b953', '#269dff', '#e73fd1', '#ff9b19']
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, axis in zip(axs, range(3)):
        sl = int(np.clip(round(vox[0, axis]), 0, arr.shape[axis]-1))
        other = [a for a in range(3) if a != axis]
        ax.imshow(np.take(arr, sl, axis=axis).T, origin='lower', cmap='gray')
        for idx, xyz in enumerate(vox):
            gap = abs(xyz[axis] - sl)
            # Projection onto common GT-centered slice: all shown with circular
            # markers and explicit signed out-of-plane offset in millimetres.
            ax.scatter(xyz[other[0]], xyz[other[1]], c=colors[idx], s=74,
                       edgecolors='black', linewidths=.8,
                       label=f'{titles[idx]} (out-of-plane {gap:.1f} mm)')
        ax.set_xlim(0, arr.shape[other[0]] - 1)
        ax.set_ylim(0, arr.shape[other[1]] - 1)
        ax.set_title(f'MRI axis {axis}, slice index {sl}\nAll markers projected onto this slice')
        ax.set_xlabel(f'Voxel axis {other[0]} (1 mm/voxel approx.)')
        ax.set_ylabel(f'Voxel axis {other[1]}')
    h, l = axs[-1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle('SESAME cross-subject: native 3D distances in mm; points are 2D projections\n'
                 'SyN and affine distances PROVISIONAL (RAS/anatomical registration assumptions)', fontsize=12)
    fig.tight_layout(rect=[0, .13, 1, .86])
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    return vox.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--donor', default='sub-01')
    ap.add_argument('--target', default='sub-02')
    ap.add_argument('--run', default='run-01')
    ap.add_argument('--dataset', type=Path, default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    ap.add_argument('--affine-audit', type=Path)
    ap.add_argument('--cross-pilot', type=Path)
    ap.add_argument('--syn-root', type=Path, default=Path('cross_subject/outputs/19_syn_registration'))
    ap.add_argument('--output-root', type=Path, default=Path('cross_subject/outputs/20_syn_point_validation'))
    ap.add_argument('--micromamba-env', default='localizemi-registration')
    ap.add_argument('--samples', type=int, default=500)
    args = ap.parse_args()
    if args.donor == args.target: ap.error('Donor and target must be different')
    if args.samples < 100: ap.error('--samples must be >= 100')
    pair = f'{args.donor}_to_{args.target}'
    key = f'{args.target}_with_{args.donor}_{args.run}'
    affine_audit_path = args.affine_audit or Path('cross_subject/outputs/17_paired_sesame_validation')/key/'paired_sesame_audit.json'
    cross_path = args.cross_pilot or Path('cross_subject/outputs/05_sesame_pilot')/key/'sesame_pilot_result.json'
    reg = args.syn_root/pair
    stem = f'{pair}_syn_'
    aff = reg/f'{stem}0GenericAffine.mat'
    warp = reg/f'{stem}1Warp.nii.gz'
    invwarp = reg/f'{stem}1InverseWarp.nii.gz'
    warped = reg/'donor_syn_in_target.nii.gz'
    quality_path = reg/'syn_registration_quality.json'
    moving = args.dataset/'derivatives'/'epochs'/args.donor/'anat'/f'{args.donor}_mod-T1w_maskface.nii'
    fixed = args.dataset/'derivatives'/'epochs'/args.target/'anat'/f'{args.target}_mod-T1w_maskface.nii'
    for p in [affine_audit_path, cross_path, aff, warp, invwarp, warped, moving, fixed, quality_path]:
        if not p.is_file(): raise FileNotFoundError(f'Missing input: {p}')
    audit = json.loads(affine_audit_path.read_text())
    pilot = json.loads(cross_path.read_text())
    reg_report = json.loads(quality_path.read_text())
    if audit['metrics']['donor']!=args.donor or audit['metrics']['target']!=args.target or audit['metrics']['run']!=args.run:
        raise ValueError('Script 17 donor/target/run does not match')
    if pilot.get('status')!='PASS' or pilot.get('ground_truth_used') is not False or pilot.get('target_forward_used') is not False:
        raise ValueError('Cross-subject pilot not confirmed PASS/ground-truth-independent')
    if reg_report.get('donor')!=args.donor or reg_report.get('target')!=args.target:
        raise ValueError('SyN QA is for a different donor/target')
    gt = np.asarray(audit['coordinate_chain']['ground_truth_target_surface_mm'], float)
    within = np.asarray(audit['coordinate_chain']['within_subject_source_target_surface_mm'], float)
    aff_src = np.asarray(audit['coordinate_chain']['cross_subject_source_target_surface_mm'], float)
    head = np.asarray(pilot['peak_coordinate_donor_src_rr_m'], float)*1000.
    donor_h5 = h5_transform(args.dataset, args.donor)
    target_h5 = h5_transform(args.dataset, args.target)
    donor_surface = point_transform_mm(head, donor_h5)
    donor_t1w = point_transform_mm(donor_surface, np.linalg.inv(donor_h5))
    if np.linalg.norm(donor_surface - np.asarray(audit['coordinate_chain']['cross_subject_source_donor_surface_mm']))>.001:
        raise ValueError('Donor source differs from Script 17 audit')
    if np.linalg.norm(donor_t1w-np.asarray(audit['coordinate_chain']['cross_subject_source_donor_T1w_provisional_RAS_mm']))>.001:
        raise ValueError('Donor T1w differs from Script 17 audit')
    baseline=float(np.linalg.norm(gt-within))
    afferr=float(np.linalg.norm(gt-aff_src))
    if abs(baseline-float(audit['metrics']['within_subject_error_mm']))>.001 or abs(afferr-float(audit['metrics']['cross_subject_affine_error_mm']))>.001:
        raise ValueError('Within/affine previous distances failed consistency checks')

    exe = find_exe('antsApplyTransformsToPoints', args.micromamba_env)
    fixed_img = nib.load(str(fixed)); moving_img=nib.load(str(moving)); warped_img=nib.load(str(warped))
    if warped_img.shape != fixed_img.shape or not np.allclose(warped_img.affine, fixed_img.affine, atol=.001):
        raise ValueError('SyN warped donor is not in target image grid')
    vol = np.asarray(warped_img.dataobj, dtype=np.float32)
    positives = vol[np.isfinite(vol) & (vol>0)]
    if len(positives)<args.samples: raise ValueError('Too few donor tissue voxels')
    mask=np.argwhere(vol>np.quantile(positives,.4))
    rng=np.random.default_rng(20261021)
    chosen=mask[rng.choice(len(mask), args.samples, replace=False)]
    target_world=nib.affines.apply_affine(fixed_img.affine,chosen)
    # ANTs image resampling uses fixed->moving mapping for each fixed voxel.
    # For point tool the validated candidate should reproduce the saved warped image.
    target_to_donor_terms=[str(warp),str(aff)]
    donor_to_target_terms=[f'[{aff},1]',str(invwarp)]
    mapped_moving=ants_points(exe,target_world,target_to_donor_terms)
    consistency=quality_from_fixed_points(target_world,warped_img,moving_img,mapped_moving)
    if consistency['rmse'] is None or consistency['n_valid']<.85*args.samples:
        raise RuntimeError('Insufficient valid mapped MRI points; inspect SyN direction')
    # Floating-point interpolation and warp-field boundary behavior can cause
    # small deviations; image intensity values are usually ~0..1.
    if consistency['rmse']>.015 or consistency['correlation'] is None or consistency['correlation']<.995:
        raise RuntimeError(f'NONLINEAR POINT MAPPING NOT VALIDATED: {consistency}. '
                           'Do not report SESAME distances until warp direction is resolved.')
    names, contacts=get_contacts(args.dataset,args.donor)
    points=np.vstack([contacts, donor_t1w[None,:]])
    mapped=ants_points(exe,points,donor_to_target_terms)
    recovered=ants_points(exe,mapped,target_to_donor_terms)
    errors=np.linalg.norm(recovered-points,axis=1)
    if float(np.max(errors))>.5:
        raise RuntimeError(f'SyN source/contact round-trip exceeds 0.5 mm: {np.max(errors):.4f} mm')
    syn_t1w=mapped[-1]
    syn_surface=point_transform_mm(syn_t1w,target_h5)
    syn_error=float(np.linalg.norm(syn_surface-gt))
    result={
        'target':args.target, 'donor':args.donor, 'run':args.run, 'method':'SESAME',
        'within_subject_error_mm':baseline,
        'cross_subject_affine_error_mm':afferr,
        'cross_subject_syn_error_mm':syn_error,
        'syn_minus_affine_mm':syn_error-afferr,
        'syn_minus_within_mm':syn_error-baseline,
        'syn_roundtrip_source_mm':float(errors[-1]),
        'syn_roundtrip_max_all_points_mm':float(np.max(errors)),
        'syn_image_reproduction_rmse':consistency['rmse'],
        'syn_image_reproduction_corr':consistency['correlation'],
        'status':'PROVISIONAL_SYN_AFFINE_COMPARISON_NOT_PUBLICATION_VALIDATED'
    }
    out=args.output_root/key
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([result]).to_csv(out/'paired_sesame_registration_comparison.csv',index=False,float_format='%.9f')
    mapped_csv=pd.DataFrame({'name':names+['SESAME_source']})
    for i,lab in enumerate('xyz'):
        mapped_csv[f'donor_T1w_{lab}_candidate_RAS_mm']=points[:,i]
        mapped_csv[f'target_T1w_{lab}_candidate_RAS_mm']=mapped[:,i]
        mapped_csv[f'roundtrip_{lab}_RAS_mm']=recovered[:,i]
    mapped_csv['roundtrip_error_mm']=errors
    mapped_csv.to_csv(out/'syn_transformed_donor_points.csv',index=False,float_format='%.9f')
    vox=figure(fixed_img,gt,within,aff_src,syn_surface,target_h5,
               [baseline,afferr,syn_error],out/'sesame_affine_vs_syn_target_mri.png')
    report={
        'status':result['status'],'metrics':result,
        'source_coordinates_mm':{
            'donor_head':head.tolist(),'donor_surface':donor_surface.tolist(),
            'donor_T1w_candidate_RAS':donor_t1w.tolist(),
            'target_T1w_candidate_RAS_via_syn':syn_t1w.tolist(),
            'target_surface_via_syn':syn_surface.tolist(),
            'target_surface_via_affine':aff_src.tolist(),
            'ground_truth_target_surface':gt.tolist(),
            'within_subject_target_surface':within.tolist(),
            'target_MRI_plot_voxels_gt_within_affine_syn':vox},
        'point_mapping_validation':{
            'target_to_donor_terms_as_passed_to_antsApplyTransformsToPoints':target_to_donor_terms,
            'donor_to_target_terms_as_passed_to_antsApplyTransformsToPoints':donor_to_target_terms,
            'sampled_fixed_mri_points':args.samples,
            'warped_MRI_reproduction':consistency,
            'roundtrip_max_mm':float(np.max(errors)),
            'roundtrip_median_mm':float(np.median(errors)),
            'source_roundtrip_mm':float(errors[-1])},
        'registration_quality':reg_report.get('syn_quality'),
        'provenance':{'affine_audit':str(affine_audit_path),'cross_pilot':str(cross_path),
                      'syn_quality':str(quality_path),'syn_affine':str(aff),
                      'syn_forward_warp':str(warp),'syn_inverse_warp':str(invwarp),
                      'syn_warped_donor':str(warped)},
        'limitations':['T1w native coordinates assumed to equal NIfTI RAS physical coordinates.',
                       'Round-trip checks implementation, not anatomical correctness.',
                       'SyN image correspondence is not guaranteed to be true cortical correspondence.',
                       'Registration and source localization errors are inseparable in these numbers.',
                       'Registration must never be selected/tuned based on localization error.',
                       'Source/ground truth not used for registration or point direction selection.']}
    (out/'syn_paired_sesame_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print(f'Output directory: {out}')


if __name__=='__main__':
    main()
