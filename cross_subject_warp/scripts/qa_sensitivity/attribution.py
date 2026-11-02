"""Independent physical-vector closure and native composite/staged diagnostics."""
from __future__ import annotations
from decimal import Decimal
import itertools
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
from scipy.ndimage import map_coordinates

from . import core as c

FLIP=np.diag([-1.,-1.,1.])
CONTROL_MAX_MM=.01
BUDGET_MAX_MM=.02


def affine_ras(path,points,inverse=False):
    doc=loadmat(path);keys=[k for k in doc if k.startswith('AffineTransform_') and k.endswith('_3_3')]
    if len(keys)!=1 or 'fixed' not in doc:raise ValueError('Expected one centred ITK affine in MATLAB format')
    params=np.asarray(doc[keys[0]],float).ravel();centre=np.asarray(doc['fixed'],float).ravel()
    if len(params)!=12 or len(centre)!=3:raise ValueError('Unexpected affine parameters')
    mat=np.eye(4);mat[:3,:3]=params[:9].reshape(3,3);mat[:3,3]=params[9:]+centre-mat[:3,:3]@centre
    if inverse:mat=np.linalg.inv(mat)
    lps=np.asarray(points)@FLIP.T
    return (lps@mat[:3,:3].T+mat[:3,3])@FLIP.T

class Field:
    def __init__(self,path,basis,order=1):
        self.image=nib.load(path);self.stored_dtype=str(self.image.header.get_data_dtype());self.data=np.asarray(self.image.dataobj,dtype=np.float64)
        if self.data.shape[3:]==(1,3):self.data=self.data[:,:,:,0,:]
        if self.data.shape[3:]!=(3,) or not np.isfinite(self.data).all():raise ValueError('Invalid vector field')
        self.basis=np.asarray(basis);self.order=order
    def inspect(self,points):
        vox=nib.affines.apply_affine(np.linalg.inv(self.image.affine),np.asarray(points))
        # Linear interpolation requires each contributing node inside the lattice.
        valid=np.isfinite(vox).all(axis=1)&np.all((vox>=0)&(vox<=np.array(self.data.shape[:3])-1),axis=1)
        edge=np.min(np.c_[vox,np.array(self.data.shape[:3])-1-vox],axis=1)
        return vox,valid,edge
    def displacement(self,points,order=None):
        vox,valid,_=self.inspect(points);result=np.full_like(np.asarray(points,float),np.nan)
        if valid.any():result[valid]=np.column_stack([map_coordinates(self.data[:,:,:,i],vox[valid].T,order=self.order if order is None else order,mode='constant',cval=np.nan,prefilter=False) for i in range(3)])@self.basis.T
        return result,valid
    def mapped(self,points,order=None):
        values,valid=self.displacement(points,order);return np.asarray(points)+values,valid


def write_field(path,vector,affine,shape=(64,64,64)):
    values=np.zeros((*shape,1,3),np.float32);values[:,:,:,0,:]=vector
    im=nib.Nifti1Image(values,affine);im.header.set_xyzt_units('mm');im.header.set_intent('vector');nib.save(im,path)


def native_controls(directory,mapping,exe):
    """Resolve component convention only on synthetic data, before any real field."""
    directory=Path(directory);directory.mkdir()
    affine=np.diag([8.,8.,8.,1.]);affine[:3,3]=-256
    probe=np.array([[4.12345,5.23456,6.34567],[31.2345,-14.1234,11.2345]])
    observed=[];controls=[];all_outputs=[];point_results=[]
    for i in range(3):
        raw=np.zeros(3);raw[i]=.375
        path=directory/f'basis_{i}.nii.gz';write_field(path,raw,affine)
        result=mapping.point_map(exe,probe,[str(path)]);all_outputs.append(result)
        observed.append(np.mean(result-probe,axis=0)/.375)
    measured=np.column_stack(observed)
    candidates=[np.eye(3),FLIP]
    basis=min(candidates,key=lambda b:np.max(np.abs(b-measured)))
    convention_error=float(np.max(np.abs(basis-measured))*.375)
    controls.append(dict(control='constant_component_convention',max_error_mm=convention_error,passed=convention_error<=CONTROL_MAX_MM))
    theta=np.deg2rad(19);rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    oblique=affine.copy();oblique[:3,:3]=rot@affine[:3,:3];oblique[:3,3]=-oblique[:3,:3]@np.array([32.,32.,32.])
    oblique_path=directory/'oblique.nii.gz';raw=np.array([.3,-.4,.7]);write_field(oblique_path,raw,oblique)
    got=mapping.point_map(exe,probe,[str(oblique_path)]);all_outputs.append(got)
    error=float(np.linalg.norm(got-(probe+raw@basis.T),axis=1).max())
    controls.append(dict(control='physical_vector_oblique_grid',max_error_mm=error,passed=error<=CONTROL_MAX_MM))
    mat=rot@np.array([[1.03,.02,0],[0,.97,.01],[0,0,1.01]]);trans=np.array([3.2,-4.1,5.3]);centre=np.array([1.,-2.,3.])
    af=directory/'noncommuting.mat';savemat(af,{'AffineTransform_double_3_3':np.r_[mat.ravel(),trans].reshape(-1,1),'fixed':centre.reshape(-1,1)},format='4')
    pts=np.array([[x+.123456789,y+.234567891,z+.345678912] for x,y,z in itertools.product([-140.,0.,140.],repeat=3)])
    delta=np.array([.7,-.4,.3]);w=directory/'translation.nii.gz';v=directory/'inverse_translation.nii.gz'
    write_field(w,delta@basis,affine);write_field(v,-delta@basis,affine)
    cases=[('affine_centre_and_lps',[str(af)],affine_ras(af,pts)),
           ('affine_inverse',[f'[{af},1]'],affine_ras(af,pts,True)),
           ('noncommuting_forward',[str(w),str(af)],affine_ras(af,pts+delta)),
           ('noncommuting_inverse',[f'[{af},1]',str(v)],affine_ras(af,pts,True)-delta)]
    for label,terms,expected in cases:
        got=mapping.point_map(exe,pts,terms);all_outputs.append(got)
        err=float(np.linalg.norm(got-expected,axis=1).max())
        controls.append(dict(control=label,max_error_mm=err,passed=err<=CONTROL_MAX_MM))
        for index,(source,target,result) in enumerate(zip(pts,expected,got)):
            point_results.append(dict(control=label,index=index,source_ras_mm=source.tolist(),expected_ras_mm=target.tolist(),
                                      observed_ras_mm=result.tolist(),difference_mm=float(np.linalg.norm(result-target))))
    # Parse reported decimal values to estimate writer precision; establish the
    # diagnostic agreement budget from controls, never the six real residuals.
    digits=max(len(Decimal(str(float(x))).normalize().as_tuple().digits) for out in all_outputs for x in out.ravel() if x)
    magnitude=250.;quantum=10.**(np.floor(np.log10(magnitude))-digits+1)
    io_bound=float(np.sqrt(3)*quantum/2)
    budget=float(max(1e-8,4*max(r['max_error_mm'] for r in controls),4*io_bound*(2+np.linalg.norm(mat,2))))
    controls_pass=bool(all(r['passed'] for r in controls) and budget<=BUDGET_MAX_MM)
    return dict(passed=controls_pass,component_to_ras=basis.tolist(),writer_significant_digits_observed=digits,
        probe_range_mm=250.,estimated_single_csv_rounding_bound_mm=io_bound,agreement_budget_mm=budget,
        budget_formula='max(1e-8,4*max_control_error,4*sqrt(3)*quantum/2*(2+affine_operator_norm))',
        budget_cap_mm=BUDGET_MAX_MM,controls=controls,point_results=point_results)


def real_direction(points,direction,files,mapping,exe,basis):
    p=np.asarray(points,float);A=files['syn_aff'];W=Field(files['warp'],basis);V=Field(files['invwarp'],basis)
    terms=mapping.terms(files)
    if direction=='reference_to_test_to_reference':
        z=mapping.point_map(exe,p,[f'[{A},1]'])
        staged=mapping.point_map(exe,z,[str(files['invwarp'])])
        stage_w=mapping.point_map(exe,staged,[str(files['warp'])])
        stage_back=mapping.point_map(exe,stage_w,[str(A)])
        ind_z=affine_ras(A,p,True);ind_forward,dom1=V.mapped(ind_z)
        ind_w,dom2=W.mapped(ind_forward);ind_back=affine_ras(A,ind_w)
        field_closure=ind_w-ind_z
        origin=affine_ras(A,ind_z);closure_end=affine_ras(A,ind_z+field_closure)
        amplified=closure_end-origin
        inspections=[('inverse_field_input',V,z),('forward_field_input',W,staged)]
        native_field=staged;ind_field=ind_forward
        forward_terms=terms['syn_r2t'];back_terms=terms['syn_t2r']
    else:
        z=mapping.point_map(exe,p,[str(files['warp'])]);staged=mapping.point_map(exe,z,[str(A)])
        inv_a=mapping.point_map(exe,staged,[f'[{A},1]']);stage_back=mapping.point_map(exe,inv_a,[str(files['invwarp'])])
        ind_z,dom1=W.mapped(p);ind_forward=affine_ras(A,ind_z)
        restored=affine_ras(A,ind_forward,True);ind_back,dom2=V.mapped(restored)
        field_closure=ind_back-p;amplified=field_closure
        inspections=[('forward_field_input',W,p),('inverse_field_input',V,inv_a)]
        native_field=z;ind_field=ind_z
        forward_terms=terms['syn_t2r'];back_terms=terms['syn_r2t']
    full=mapping.point_map(exe,p,forward_terms);back=mapping.point_map(exe,full,back_terms)
    aff_native=mapping.point_map(exe,p,[str(A)])
    aff_back=mapping.point_map(exe,aff_native,[f'[{A},1]'])
    aff_independent=affine_ras(A,p)
    if max(np.max(np.abs(x)) for x in [p,z,staged,stage_back,full,back,aff_native])>250:
        raise ValueError('Intermediate coordinates exceed independently calibrated precision range')
    rows=[]
    for i in range(len(p)):
        row=dict(sample_index=i,forward_field_stored_dtype=W.stored_dtype,inverse_field_stored_dtype=V.stored_dtype,
            independent_arithmetic='float64; original field values unchanged',affine_only_roundtrip_mm=float(np.linalg.norm(aff_back[i]-p[i])),
            independent_affine_error_mm=float(np.linalg.norm(aff_native[i]-aff_independent[i])),full_roundtrip_mm=float(np.linalg.norm(back[i]-p[i])),
            full_staged_forward_mm=float(np.linalg.norm(full[i]-staged[i])),
            full_staged_cycle_mm=float(np.linalg.norm(back[i]-stage_back[i])),
            independent_native_field_mm=float(np.linalg.norm(native_field[i]-ind_field[i])) if dom1[i] else None,
            independent_cycle_mm=float(np.linalg.norm(ind_back[i]-p[i])) if dom1[i] and dom2[i] else None,
            closure_norm_mm=float(np.linalg.norm(field_closure[i])) if dom1[i] and dom2[i] else None,
            closure_predicted_cycle_mm=float(np.linalg.norm(amplified[i])) if dom1[i] and dom2[i] else None,
            closure_observed_vector_discrepancy_mm=float(np.linalg.norm((back[i]-p[i])-amplified[i])) if dom1[i] and dom2[i] else None)
        for label,field,coords in inspections:
            vox,valid,edge=field.inspect(coords[i:i+1]);row[f'{label}_inside']=bool(valid[0]);row[f'{label}_edge_vox']=float(edge[0])
            for axis,j in zip('xyz',range(3)):
                row[f'{label}_{axis}_ras_mm']=float(coords[i,j]);row[f'{label}_{axis}_fractional_voxel']=float(vox[0,j]-np.floor(vox[0,j]))
        rows.append(row)
    return pd.DataFrame(rows),full,back


def field_domain_roundtrip_metrics(mapping,exe,source,destination,direction,files,qa,basis,points=None):
    """Exclude field-outside samples before roundtrip scoring; retain full denominator."""
    points=qa.foreground_points(source) if points is None else np.asarray(points,float)
    points=np.asarray(points,float).reshape(-1,3)
    if not np.isfinite(points).all():raise ValueError('Nonfinite QA source samples')
    terms=mapping.terms(files);warp=Field(files['warp'],basis);invwarp=Field(files['invwarp'],basis)
    A=files['syn_aff']
    if direction=='reference':
        first_field_input=affine_ras(A,points,True);first_field=invwarp
        first_terms=terms['syn_r2t'];back_terms=terms['syn_t2r']
    elif direction=='test':
        first_field_input=points;first_field=warp
        first_terms=terms['syn_t2r'];back_terms=terms['syn_r2t']
    else:raise ValueError(f'Unknown direction: {direction}')
    _,first_ok,_=first_field.inspect(first_field_input)
    mapped=np.full_like(points,np.nan)
    if first_ok.any():mapped[first_ok]=mapping.point_map(exe,points[first_ok],first_terms)
    destination_vox=nib.affines.apply_affine(np.linalg.inv(destination.affine),mapped[first_ok]) if first_ok.any() else np.empty((0,3))
    destination_ok=np.zeros(len(points),bool)
    if first_ok.any():
        destination_ok[first_ok]=np.isfinite(destination_vox).all(axis=1)&np.all(
            (destination_vox>=1)&(destination_vox<np.asarray(destination.shape[:3])-2),axis=1)
    return_field_input=np.full_like(points,np.nan)
    first_and_destination=first_ok&destination_ok
    if first_and_destination.any():
        if direction=='reference':
            return_field_input[first_and_destination]=mapped[first_and_destination]
            return_field=warp
        else:
            return_field_input[first_and_destination]=affine_ras(A,mapped[first_and_destination],True)
            return_field=invwarp
    _,return_ok_local,_=return_field.inspect(return_field_input[first_and_destination]) if first_and_destination.any() else (None,np.zeros(0,bool),None)
    return_ok=np.zeros(len(points),bool)
    if first_and_destination.any():return_ok[first_and_destination]=return_ok_local
    valid=first_and_destination&return_ok
    field_domain_excluded=int((~first_ok).sum()+((first_and_destination)&~return_ok).sum())
    errors=np.full(len(points),np.nan)
    if valid.any():
        back=mapping.point_map(exe,mapped[valid],back_terms)
        errors[valid]=np.linalg.norm(back-points[valid],axis=1)
    finite=errors[valid]
    return dict(tested=len(points),valid=int(valid.sum()),coverage=float(valid.mean()),
        mri_destination_valid=int(destination_ok.sum()),first_field_domain_valid=int(first_ok.sum()),
        return_field_domain_assessed=int(first_and_destination.sum()),
        return_field_domain_valid=int((first_and_destination&return_ok).sum()),
        both_field_domains_valid=int(valid.sum()),field_domain_excluded=field_domain_excluded,
        field_domain_checked=True,
        maximum_mm=float(finite.max()) if len(finite) else None,
        p95_mm=float(np.percentile(finite,95)) if len(finite) else None,
        finite=bool(len(finite)>0 and np.isfinite(finite).all()),
        over_rt010=int(np.sum(finite>.1)),over_rt015=int(np.sum(finite>.15)))


def interpolation_probes(points,direction,files,basis):
    W=Field(files['warp'],basis);V=Field(files['invwarp'],basis);A=files['syn_aff']
    rows=[]
    for index,p in enumerate(np.asarray(points)):
        q=affine_ras(A,p.reshape(1,3),True)[0] if direction.startswith('reference') else p
        field=V if direction.startswith('reference') else W
        vox,_,_=field.inspect(q.reshape(1,3));centre=nib.affines.apply_affine(field.image.affine,np.rint(vox[0]))
        probes=[('original',q),('nearest_field_voxel_centre',centre)]
        for axis in range(3):
            for sign in [-1,1]:probes.append((f'quarter_voxel_axis{axis}_{sign}',q+sign*.25*field.image.affine[:3,axis]))
        for location,probe in probes:
            for order,label in [(1,'PRODUCTION_LINEAR'),(0,'DIAGNOSTIC_NEAREST_ONLY')]:
                first,ok1=(V if direction.startswith('reference') else W).mapped(probe.reshape(1,3),order)
                second,ok2=(W if direction.startswith('reference') else V).mapped(first,order)
                residual=second-probe
                if direction.startswith('reference'):residual=affine_ras(A,second)-affine_ras(A,probe.reshape(1,3))
                valid=bool(ok1[0] and ok2[0])
                rows.append(dict(offender_index=index,probe=location,interpolant=label,valid=valid,
                    closure_cycle_mm=float(np.linalg.norm(residual[0])) if valid else None,
                    used_for_registration_acceptance=False))
    return rows


def investigate(args):
    dataset=c.protect(args)
    qa,mapping,ev,base,ledger,parent_table,parent,exe=c.context(args)
    original=args.diagnostics/'02_point_diagnostics.csv';ledger.track(original)
    saved=pd.read_csv(original);failed=parent_table[parent_table.status.eq('FAIL')]
    if not failed.registration_mode.eq('syn').all():raise ValueError('Attribution currently supports the historical SyN failures only')
    expected=set(map(tuple,failed[c.PAIR].values))
    if set(map(tuple,saved[c.PAIR].drop_duplicates().values))!=expected:raise ValueError('Diagnostic subset differs from failed pairs')
    for row in failed.to_dict('records'):c.verify_pair_inputs(args,row,qa,ledger,parent)
    summaries=[];details=[];replay=[];gates=[];probes=[]
    with tempfile.TemporaryDirectory(prefix='attribution_controls_') as tmp:
        fixtures=Path(tmp)/'controls';controls=native_controls(fixtures,mapping,exe)
        budget=controls['agreement_budget_mm']
        if controls['passed']:
            basis=np.array(controls['component_to_ras'])
            for row in failed.to_dict('records'):
                files=qa.files_for_pair(args.resource_root,row['reference_subject'],row['test_subject'])
                for direction in ['reference_to_test_to_reference','test_to_reference_to_test']:
                    part=saved[(saved.reference_subject==row['reference_subject'])&(saved.test_subject==row['test_subject'])&(saved.direction==direction)].sort_values('sample_index')
                    source=nib.load(row['moving_mri'] if direction.startswith('reference') else row['fixed_mri'])
                    destination=nib.load(row['fixed_mri'] if direction.startswith('reference') else row['moving_mri'])
                    points=qa.foreground_points(source)
                    xyz=part[[f'source_ras_{a}_mm' for a in 'xyz']].to_numpy(float)
                    if len(part)!=qa.SAMPLES or not np.allclose(points,xyz,atol=1e-9,rtol=0):raise ValueError('Original sample reproduction failed')
                    if np.max(np.abs(points))>controls['probe_range_mm']:raise ValueError('Real points exceed precision-control range')
                    result,full,back=real_direction(points,direction,files,mapping,exe,basis)
                    vox=nib.affines.apply_affine(np.linalg.inv(destination.affine),full)
                    valid=np.all((vox>=1)&(vox<np.array(destination.shape[:3])-2),axis=1)
                    if not np.array_equal(valid,c.yes(part.destination_interior).to_numpy()):raise ValueError('Original coverage mask changed')
                    old=part.roundtrip_mm.to_numpy(float);errors=np.linalg.norm(back-points,axis=1)
                    difference=float(np.max(np.abs(errors[valid]-old[valid])))
                    # Replay is the same executable/files/batch and should agree to CSV storage precision.
                    replay_ok=difference<=1e-9
                    result['destination_interior']=valid;result['original_over_rt010']=valid&(errors>.1)
                    for field,value in {**{k:row[k] for k in c.PAIR},'direction':direction}.items():result[field]=value
                    details.append(result)
                    field_valid=result.forward_field_input_inside.to_numpy(bool)&result.inverse_field_input_inside.to_numpy(bool)
                    eligible_mask=valid&field_valid
                    eligible=result[eligible_mask]
                    field_coverage=float(eligible_mask.sum()/qa.SAMPLES)
                    field_coverage_pass=field_coverage>=qa.MIN_COVERAGE
                    exceedance_domain_pass=not bool((result.original_over_rt010.to_numpy(bool)&~field_valid).any())
                    domain_ok=field_coverage_pass and exceedance_domain_pass
                    if eligible.empty:staged_max=field_max=closure_max=affine_max=None
                    else:
                        staged_max=float(eligible[['full_staged_forward_mm','full_staged_cycle_mm']].max().max())
                        field_max=float(eligible.independent_native_field_mm.max())
                        closure_max=float(eligible.closure_observed_vector_discrepancy_mm.max())
                        affine_max=float(eligible[['affine_only_roundtrip_mm','independent_affine_error_mm']].max().max())
                    numerical_agreement_pass=bool(eligible_mask.any() and all(x is not None and x<=budget for x in [staged_max,field_max,closure_max,affine_max]))
                    passing=replay_ok and domain_ok and numerical_agreement_pass
                    if passing and (valid&(errors>.1)).any():
                        probes.extend(dict(reference_subject=row['reference_subject'],test_subject=row['test_subject'],direction=direction,**r)
                            for r in interpolation_probes(points[valid&(errors>.1)],direction,files,basis))
                    verdict='STORED_INTERPOLATED_FIELD_CLOSURE_REPRODUCES_CYCLE' if passing else 'UNRESOLVED_OR_SOFTWARE_DOMAIN_DISCREPANCY'
                    summaries.append(dict(reference_subject=row['reference_subject'],test_subject=row['test_subject'],direction=direction,
                        replay_max_difference_mm=difference,replay_pass=replay_ok,intermediate_domains_pass=domain_ok,
                        field_domain_coverage=field_coverage,field_domain_coverage_pass=field_coverage_pass,
                        field_domain_excluded_samples=int((valid&~field_valid).sum()),
                        roundtrip_exceedances_in_domain=exceedance_domain_pass,
                        affine_control_max_mm=affine_max,composite_staged_max_mm=staged_max,independent_native_field_max_mm=field_max,closure_vector_discrepancy_max_mm=closure_max,
                        numerical_agreement_pass=numerical_agreement_pass,
                        agreement_budget_mm=budget,validation_gate_pass=passing,assessment=verdict,
                        inference_limit='Saved interpolant inverse inconsistency; interpolation versus original inverse estimation is not uniquely identifiable.'))
                    replay.append(dict(reference_subject=row['reference_subject'],test_subject=row['test_subject'],direction=direction,
                        mri_destination_valid=int(valid.sum()),field_domain_valid=int(eligible_mask.sum()),
                        field_domain_excluded=int((valid&~field_valid).sum()),coverage_after_field_domain=field_coverage,
                        above_rt010=int((errors[eligible_mask]>.1).sum()),above_rt015=int((errors[eligible_mask]>.15).sum()),
                        maximum_mm=float(errors[eligible_mask].max()) if eligible_mask.any() else None))
                    gates.append(passing)
        probe_table=pd.DataFrame(probes)
        if len(probe_table):
            for summary in summaries:
                selected=probe_table[(probe_table.reference_subject==summary['reference_subject'])&(probe_table.test_subject==summary['test_subject'])&(probe_table.direction==summary['direction'])]
                if len(selected):
                    pair=selected.pivot(index=['offender_index','probe'],columns='interpolant',values='closure_cycle_mm')
                    summary['diagnostic_interpolation_max_difference_mm']=float((pair['PRODUCTION_LINEAR']-pair['DIAGNOSTIC_NEAREST_ONLY']).abs().max())
                    summary['interpolation_claim']='Sensitivity of stored interpolants only; not proof of the unavailable continuous inverse.'
        ready=controls['passed'] and len(gates)==2*len(failed) and all(gates)
        manifest=dict(status='ATTRIBUTION_CONTROLS_PASS' if ready else 'ATTRIBUTION_BLOCKED',
            real_direction_checks=len(gates),passed_direction_checks=sum(gates),controls_passed=controls['passed'],
            sensitivity_acceptance_unblocked=ready,
            runtime_validation=dict(synthetic_native_controls_passed=sum(r['passed'] for r in controls['controls']),
                synthetic_native_controls_failed=sum(not r['passed'] for r in controls['controls']),
                historical_direction_checks_passed=sum(gates),historical_direction_checks_failed=len(gates)-sum(gates),
                historical_direction_checks_not_run=2*len(failed)-len(gates),
                field_domain_excluded_samples=int(sum(r.get('field_domain_excluded_samples',0) for r in summaries))),
            causal_attribution='OPERATIONAL_FIELD_CLOSURE_OR_UNRESOLVED; no claim of unique historical cause',
            not_run=['Original continuous inverse is unavailable; no refitting or regenerated real inverse fields.',
                     'SimpleITK/Python ITK are not required; the physical-vector reader is controlled against native ANTs single-field calls.'],
            limitations=['Synthetic controls validate mechanics, not scientific/anatomical accuracy.',
                         'The diagnostic agreement budget is derived from synthetic controls before inspecting real residuals.'])
        c.publish(args,ledger,manifest,
            {'01_cause_summary.csv':pd.DataFrame(summaries),'02_point_closure.csv':pd.concat(details,ignore_index=True) if details else pd.DataFrame(),
             '03_original_sample_replay.csv':pd.DataFrame(replay),'04_synthetic_controls.csv':pd.DataFrame(controls['controls']),
             '06_diagnostic_interpolation_probes.csv':pd.DataFrame(probes)},
            {'05_precision_and_vector_convention.json':controls},
            {f'synthetic_controls/{p.name}':p for p in fixtures.iterdir()})
    return 0 if ready else 2
