#!/usr/bin/env python3
# Focused diagnostic for the single near-edge field sample in the attribution report.
# From project root, with the existing ANTs runtime:
#
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_diagnose_field_boundary.py \
#   --output-root cross_subject_warp/outputs/06_registration_qa_boundary_diagnostic_review
#
# Writes a NEW output directory: outputs/06_registration_qa_boundary_diagnostic_review/.
# To keep a prior report, choose another absent directory with --output-root.
# This repeats native point mapping only for one saved sample and diagnostic
# perturbations. It does not change QA, registration files or primary outputs.
# Outside-field probes remain invalid and cannot pass a QA gate.
"""Inspect the near-boundary SyN field point that blocked attribution.

The source sample is the recorded sample 196 for sub-06 -> sub-04,
test_to_reference_to_test. Measure its MRI/field physical geometry and compare
native ANTs versus an independently controlled physical-vector reader at the
original point and small inward/outward perturbations along the nearest field
boundary axis. Every probe is diagnostic-only; no QA status is recalculated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from qa_sensitivity import core as c
from qa_sensitivity import attribution as a

STEPS_VOXELS=(0.0,1e-6,1e-4,1e-2,0.25)


def load_field_pair(warp_path, inverse_path, basis):
    """Read each saved field independently; their vector values are not interchangeable."""
    return a.Field(warp_path,basis),a.Field(inverse_path,basis)


def boundary_probes(point, affine, shape):
    """Return original plus symmetric inward/outward probes on nearest face."""
    point=np.asarray(point,dtype=float).reshape(1,3)
    vox=nib.affines.apply_affine(np.linalg.inv(affine),point)[0]
    upper=np.asarray(shape[:3],float)-1
    distances=np.r_[vox,upper-vox]
    face=int(np.argmin(distances))
    axis=face%3
    inward=1.0 if face<3 else -1.0
    vector=np.asarray(affine,float)[:3,axis]
    vector=vector/np.linalg.norm(vector)
    result=[dict(label='original',step_vox=0.0,point_ras_mm=point[0].tolist())]
    for step in STEPS_VOXELS[1:]:
        result.append(dict(label='toward_interior',step_vox=step,
            point_ras_mm=(point[0]+inward*step*vector).tolist()))
        result.append(dict(label='toward_exterior',step_vox=step,
            point_ras_mm=(point[0]-inward*step*vector).tolist()))
    return axis,face,vox,result


def mri_geometry(image, point):
    point=np.asarray(point,float).reshape(1,3)
    q=image.get_qform();s=image.get_sform()
    return dict(shape=list(image.shape[:3]),units=image.header.get_xyzt_units()[0],
        image_affine=image.affine.tolist(),qform=q.tolist(),sform=s.tolist(),
        qform_code=int(image.header['qform_code']),sform_code=int(image.header['sform_code']),
        point_index_selected_affine=nib.affines.apply_affine(np.linalg.inv(image.affine),point)[0].tolist(),
        point_index_qform=nib.affines.apply_affine(np.linalg.inv(q),point)[0].tolist(),
        point_index_sform=nib.affines.apply_affine(np.linalg.inv(s),point)[0].tolist())


def run(args):
    qa,mapping,ev,base,ledger,parent_table,parent,exe=c.context(args,full=False)
    if args.attribution_root.exists() is False:raise FileNotFoundError(args.attribution_root)
    parent_manifest=c.require_manifest(args.attribution_root,'manifest.json','ATTRIBUTION_BLOCKED',ledger)
    if parent_manifest.get('sensitivity_acceptance_unblocked') is not False:
        raise ValueError('Expected a blocked attribution report; inspect its status')
    closure_path=args.attribution_root/'02_point_closure.csv'
    closure=c.read(closure_path);ledger.track(closure_path)
    selected=closure.loc[(closure.reference_subject=='sub-06')&
        (closure.test_subject=='sub-04')&
        (closure.direction=='test_to_reference_to_test')&
        (closure.sample_index=='196')]
    if len(selected)!=1:raise ValueError(f'Expected one saved sample 196, found {len(selected)}')
    sample=selected.iloc[0].to_dict()
    qa_row=parent_table.loc[(parent_table.reference_subject=='sub-06')&
        (parent_table.test_subject=='sub-04')&(parent_table.registration_mode=='syn')]
    if len(qa_row)!=1:raise ValueError('Expected one parent SyN pair row')
    files=c.verify_pair_inputs(args,qa_row.iloc[0].to_dict(),qa,ledger,parent)
    fixed_path=Path(qa_row.iloc[0]['fixed_mri']);fixed=nib.load(fixed_path)
    warp=nib.load(files['warp']);basis=np.asarray(
        json.loads((args.attribution_root/'05_precision_and_vector_convention.json').read_text())['component_to_ras'])
    forward_field,inverse_field=load_field_pair(files['warp'],files['invwarp'],basis)
    p=np.asarray([sample[f'forward_field_input_{axis}_ras_mm'] for axis in 'xyz'],float)
    axis,face,index,probes=boundary_probes(p,warp.affine,warp.shape)
    terms=mapping.terms(files)
    rows=[]
    print('Checking sub-06 -> sub-04 SyN boundary probes',flush=True)
    for probe in probes:
        point=np.asarray(probe['point_ras_mm'],float).reshape(1,3)
        qvox,qvalid,qedge=forward_field.inspect(point)
        try:
            native_w=mapping.point_map(exe,point,[str(files['warp'])])
            independent_w,indvalid=forward_field.mapped(point)
            native_aff=mapping.point_map(exe,native_w,[str(files['syn_aff'])])
            native_inv_aff=mapping.point_map(exe,native_aff,[f'[{files["syn_aff"]},1]'])
            native_back=mapping.point_map(exe,native_inv_aff,[str(files['invwarp'])])
            full=mapping.point_map(exe,point,terms['syn_t2r'])
            full_back=mapping.point_map(exe,full,terms['syn_r2t'])
            vvox,vvalid,vedge=inverse_field.inspect(native_inv_aff)
            independent_v,indvalid_v=inverse_field.mapped(native_inv_aff)
            row=dict(probe=probe['label'],step_vox=probe['step_vox'],point_ras_mm=point[0].tolist(),
                initial_field_index=qvox[0].tolist(),initial_field_inside=bool(qvalid[0]),initial_field_edge_vox=float(qedge[0]),
                native_warp_displacement_mm=(native_w-point)[0].tolist(),
                independent_warp_inside=bool(indvalid[0]),
                independent_warp_mapped_ras_mm=independent_w[0].tolist() if indvalid[0] else None,
                native_reader_warp_discrepancy_mm=float(np.linalg.norm(native_w-independent_w)) if indvalid[0] else None,
                after_syn_affine_ras_mm=native_aff[0].tolist(),
                after_inverse_syn_affine_ras_mm=native_inv_aff[0].tolist(),
                inverse_field_index=vvox[0].tolist(),inverse_field_inside=bool(vvalid[0]),
                inverse_field_edge_vox=float(vedge[0]),
                native_inverse_displacement_mm=(native_back-native_inv_aff)[0].tolist(),
                independent_inverse_inside=bool(indvalid_v[0]),
                independent_inverse_displacement_mm=(independent_v-native_inv_aff)[0].tolist() if indvalid_v[0] else None,
                native_reader_inverse_discrepancy_mm=float(np.linalg.norm(native_back-independent_v)) if indvalid_v[0] else None,
                staged_cycle_mm=float(np.linalg.norm(native_back-point)),
                composite_cycle_mm=float(np.linalg.norm(full_back-point)),
                composite_vs_staged_return_mm=float(np.linalg.norm(full_back-native_back)),
                diagnostic_status='MEASURED',diagnostic_error='',diagnostic_only=True,eligible_for_qa=False)
        except Exception as exc:
            row=dict(probe=probe['label'],step_vox=probe['step_vox'],point_ras_mm=point[0].tolist(),
                initial_field_index=qvox[0].tolist(),initial_field_inside=bool(qvalid[0]),
                initial_field_edge_vox=float(qedge[0]),diagnostic_status='ERROR',
                diagnostic_error=f'{type(exc).__name__}: {exc}',diagnostic_only=True,eligible_for_qa=False)
        rows.append(row)
        print(f"  {probe['label']} {probe['step_vox']:g} voxel: field_inside={qvalid[0]}, status={row['diagnostic_status']}",flush=True)

    all_files=[Path(qa_row.iloc[0][k]) for k in ['fixed_mri','moving_mri']]
    geometry=dict(reference_subject='sub-06',test_subject='sub-04',registration_mode='syn',
        sample_index=196,original_roundtrip_mm=float(sample['full_roundtrip_mm']),
        original_inverse_field_edge_vox=float(sample['inverse_field_input_edge_vox']),
        original_forward_field_edge_vox=float(sample['forward_field_input_edge_vox']),
        nearest_boundary_axis_index=axis,nearest_boundary_face_index=face,
        nearest_boundary_axis_name='xyz'[axis],source_point_ras_mm=p.tolist(),
        mri=mri_geometry(fixed,p),field=mri_geometry(warp,p),
        max_abs_mri_field_affine_difference=float(np.max(np.abs(fixed.affine-warp.affine))),
        field_point_indices_from_parent_report=sample,
        interpretation='All boundary probes are diagnostic. Any field-outside point remains invalid; no registration QA status is changed.')
    ledger.track(Path(__file__))
    summary=dict(status='BOUNDARY_DIAGNOSTIC_ONLY',sensitivity_acceptance_unblocked=False,
        parent_attribution_manifest_sha256=c.sha(args.attribution_root/'manifest.json'),
        pair='sub-06_to_sub-04',mode='syn',sample_index=196,probes=len(rows),
        diagnostic_errors=sum(r['diagnostic_status']!='MEASURED' for r in rows),
        probes_with_initial_field_inside=sum(r['initial_field_inside'] for r in rows),
        probes_with_inverse_field_inside=sum(r.get('inverse_field_inside',False) for r in rows),
        no_qa_reclassification=True,registration_fitting_executed=False,
        inverse_reconstruction_executed=False,scientific_accuracy_established=False,
        next_step='Review voxel indices and ANTs-vs-reader displacement at the boundary. Keep QA blocked if the original point is outside either required field domain.')
    c.publish(args,ledger,summary,tables={'01_boundary_probe_results.csv':pd.DataFrame(rows)},
        jsons={'02_boundary_geometry.json':geometry})
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    out=c.EXP/'outputs'
    p.add_argument('--output-root',type=Path,default=out/'06_registration_qa_boundary_diagnostic')
    p.add_argument('--attribution-root',type=Path,default=out/'06_registration_qa_attribution')
    p.add_argument('--primary-qa',type=Path,default=out/'06_registration_qa')
    p.add_argument('--primary-evaluation',type=Path,default=out/'06_evaluate_localization_mapped')
    p.add_argument('--diagnostics',type=Path,default=out/'06_roundtrip_diagnostics')
    p.add_argument('--bridge-root',type=Path,default=out/'06_coordinate_bridge')
    p.add_argument('--preparation-root',type=Path,default=out/'01_prepare_experiment_review')
    p.add_argument('--resource-root',type=Path,default=c.EXP/'source/01_registration_resources/outputs')
    p.add_argument('--sensitivity-qa',type=Path,default=out/'06_registration_qa_sensitivity_rt015')
    p.add_argument('--sensitivity-evaluation',type=Path,default=out/'06_evaluate_localization_qa_rt015')
    args=p.parse_args()
    for name,value in vars(args).items():setattr(args,name,value.expanduser().resolve())
    args.stage='boundary'
    return run(args)


if __name__=='__main__':
    import os,tempfile
    with tempfile.TemporaryDirectory(prefix='qa-boundary-config-') as config:
        os.environ['_MNE_FAKE_HOME_DIR']=config;os.environ['MPLCONFIGDIR']=config
        raise SystemExit(main())
