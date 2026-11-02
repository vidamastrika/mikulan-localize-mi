"""Policy/independent-reader tests plus read-only real historical regressions.

No real-data mapping, registration fitting, EEG inverse fitting or external
commands are run. Historical checks are skipped explicitly if local inputs are
unavailable. Synthetic cases test mechanics, not scientific accuracy.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import savemat

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'scripts'))
from qa_sensitivity import core as c,attribution as a,analysis as s
_boundary_spec=importlib.util.spec_from_file_location('field_boundary_diagnostic',EXP/'scripts/06_diagnose_field_boundary.py')
boundary=importlib.util.module_from_spec(_boundary_spec);_boundary_spec.loader.exec_module(boundary)


class SensitivityMechanicsTests(unittest.TestCase):
    def test_boundary_reader_loads_forward_and_inverse_fields_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            affine=np.diag([2.,2.,2.,1.]);affine[:3,3]=[-10.,-10.,-10.]
            warp=Path(tmp)/'warp.nii.gz';inverse=Path(tmp)/'inverse.nii.gz'
            a.write_field(warp,np.array([1.,0.,0.]),affine,shape=(10,10,10))
            a.write_field(inverse,np.array([0.,2.,0.]),affine,shape=(10,10,10))
            forward_field,inverse_field=boundary.load_field_pair(warp,inverse,a.FLIP)
            point=np.array([[0.,0.,0.]])
            forward,valid_forward=forward_field.mapped(point)
            backward,valid_inverse=inverse_field.mapped(point)
            self.assertTrue(valid_forward.all());self.assertTrue(valid_inverse.all())
            np.testing.assert_allclose(forward-point,[[-1.,0.,0.]])
            np.testing.assert_allclose(backward-point,[[0.,-2.,0.]])

    def test_syn_roundtrip_excludes_field_outside_points_and_keeps_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);aff=np.eye(4)
            source_path=root/'source.nii.gz';dest_path=root/'dest.nii.gz'
            for path in [source_path,dest_path]:nib.save(nib.Nifti1Image(np.ones((10,10,10)),aff),path)
            syn_aff=root/'syn.mat'
            savemat(syn_aff,{'AffineTransform_double_3_3':np.r_[np.eye(3).ravel(),np.zeros(3)].reshape(-1,1),
                             'fixed':np.zeros((3,1))},format='4')
            warp=root/'warp.nii.gz';invwarp=root/'inverse.nii.gz'
            a.write_field(warp,np.zeros(3),aff,shape=(10,10,10));a.write_field(invwarp,np.zeros(3),aff,shape=(10,10,10))
            points=np.tile([4.,4.,4.],(200,1));points[0,0]=-1e-7
            class Mapper:
                calls=[]
                @staticmethod
                def terms(files):return {'syn_t2r':[str(files['warp']),str(files['syn_aff'])],
                    'syn_r2t':[f'[{files["syn_aff"]},1]',str(files['invwarp'])]}
                @staticmethod
                def point_map(exe,xyz,terms):
                    Mapper.calls.append(len(xyz));return np.asarray(xyz,float).copy()
            files=dict(syn_aff=syn_aff,warp=warp,invwarp=invwarp)
            qa=object();basis=a.FLIP
            for direction in ['test','reference']:
                result=a.field_domain_roundtrip_metrics(Mapper(),'mock',nib.load(source_path),nib.load(dest_path),
                    direction,files,qa,basis,points)
                self.assertEqual(result['tested'],200);self.assertEqual(result['valid'],199)
                self.assertEqual(result['field_domain_excluded'],1);self.assertAlmostEqual(result['coverage'],.995)
                self.assertEqual(result['maximum_mm'],0.)
            self.assertEqual(Mapper.calls,[199,199,199,199])

    def test_boundary_probes_step_symmetrically_along_nearest_grid_face(self):
        affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[-10.,-20.,-30.]
        point=nib.affines.apply_affine(affine,[[4.,5.,9.+1e-7]])
        axis,face,index,probes=boundary.boundary_probes(point[0],affine,(10,10,10))
        self.assertEqual((axis,face),(2,5))
        self.assertAlmostEqual(index[2],9.+1e-7,places=8)
        inside=np.asarray(probes[1]['point_ras_mm'])
        outside=np.asarray(probes[2]['point_ras_mm'])
        inside_idx=nib.affines.apply_affine(np.linalg.inv(affine),inside)
        outside_idx=nib.affines.apply_affine(np.linalg.inv(affine),outside)
        self.assertLess(inside_idx[2],9.)
        self.assertGreater(outside_idx[2],9.)
        self.assertEqual(probes[0]['label'],'original')

    def test_pair_and_source_policies_are_separate_immutable_and_exact(self):
        self.assertEqual(c.SENSITIVITY.pair_tolerance_mm,.15)
        self.assertEqual(c.SENSITIVITY.source_tolerance_mm,.1)
        metrics=dict(finite=True,coverage=.75,maximum_mm=.1)
        self.assertTrue(c.accepted(metrics,c.PRIMARY))
        metrics['maximum_mm']=np.nextafter(.1,np.inf)
        self.assertFalse(c.accepted(metrics,c.PRIMARY));self.assertTrue(c.accepted(metrics,c.SENSITIVITY))
        metrics['maximum_mm']=.15;self.assertTrue(c.accepted(metrics,c.SENSITIVITY))
        metrics['maximum_mm']=np.nextafter(.15,np.inf);self.assertFalse(c.accepted(metrics,c.SENSITIVITY))
        with self.assertRaises(FrozenInstanceError):c.PRIMARY.pair_tolerance_mm=.15
        metrics.update(maximum_mm=.12,coverage=.74)
        self.assertFalse(c.accepted(metrics,c.SENSITIVITY))

    def test_independent_affine_uses_fixed_centre_and_lps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'aff.mat'
            savemat(path,{'AffineTransform_double_3_3':np.r_[np.diag([2.,3.,4.]).ravel(),[.5,1.,-1.]].reshape(-1,1),
                          'fixed':np.array([1.,2.,3.]).reshape(-1,1)},format='4')
            result=a.affine_ras(path,np.array([[3.,4.,5.]]))
            np.testing.assert_allclose(result,[[6.5,15.,10.]])
            np.testing.assert_allclose(a.affine_ras(path,result,True),[[3.,4.,5.]])

    def test_physical_vector_reader_and_approximate_inverse_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'field.nii.gz';inverse=Path(tmp)/'inverse.nii.gz'
            transform=np.diag([2.,3.,4.,1.]);transform[:3,3]=[-20,-30,-40]
            a.write_field(path,np.array([1.,2.,3.]),transform,shape=(24,24,24))
            basis=np.diag([-1.,-1.,1.]);field=a.Field(path,basis)
            points=np.array([[0.1,0.2,0.3],[2.,3.,4.]])
            mapped,valid=field.mapped(points)
            self.assertTrue(valid.all());np.testing.assert_allclose(mapped,points+[-1,-2,3])
            a.write_field(inverse,np.array([-1.,-2.,-3.]),transform,shape=(24,24,24))
            back,_=a.Field(inverse,basis).mapped(mapped);np.testing.assert_allclose(back,points,atol=1e-12)
            a.write_field(inverse,np.array([-1.12,-2.,-3.]),transform,shape=(24,24,24))
            back,_=a.Field(inverse,basis).mapped(mapped)
            np.testing.assert_allclose(np.linalg.norm(back-points,axis=1),.12,atol=1e-6)
            _,valid=field.mapped(np.array([[10000.,0.,0.]]));self.assertFalse(valid[0])

    def test_native_control_orchestration_with_mocked_tool_is_serializable(self):
        def mock_points(exe,points,terms):
            result=np.asarray(points,float).copy()
            for term in terms:
                if term.startswith('['):result=a.affine_ras(term[1:-3],result,True)
                elif term.endswith('.mat'):result=a.affine_ras(term,result)
                else:
                    result,valid=a.Field(term,a.FLIP).mapped(result)
                    self.assertTrue(valid.all())
            return np.array([[float(format(x,'.6g')) for x in row] for row in result])
        with tempfile.TemporaryDirectory() as tmp:
            result=a.native_controls(Path(tmp)/'fixtures',type('Mapper',(),{'point_map':staticmethod(mock_points)})(),'mocked')
            self.assertTrue(result['passed'],result)
            self.assertLess(result['agreement_budget_mm'],.02)
            json.dumps(result,allow_nan=False)

    def test_common_cohort_detects_changed_errors_and_lost_primary_validity(self):
        data={k:['x'] for k in c.KEY};data.update(registration_mode=['affine'],status=['PASS'],configuration_key=['k'],
            configuration_parameters=['{}'],failure_reason=[''],estimated_head_x_mm=['1'],estimated_head_y_mm=['2'],
            estimated_head_z_mm=['3'],localization_error_valid=[True],localization_error_mm=[5.],
            mapped_test_ras_x_mm=[1.],mapped_test_ras_y_mm=[2.],mapped_test_ras_z_mm=[3.])
        primary=pd.DataFrame(data);s.same_common(primary,primary.copy())
        changed=primary.copy();changed.loc[0,'localization_error_mm']=5.01
        with self.assertRaisesRegex(ValueError,'numerical regression'):s.same_common(primary,changed)
        changed=primary.copy();changed.loc[0,'localization_error_valid']=False
        with self.assertRaisesRegex(ValueError,'lost previously valid'):s.same_common(primary,changed)

    def test_new_pair_pipeline_relaxes_only_roundtrip_and_preserves_image_gate(self):
        from types import SimpleNamespace
        qa,mapping,ev,base=c.modules()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'data';resources=root/'resources'
            values=np.zeros((20,20,20),np.float32);values[3:-3,3:-3,3:-3]=1
            for subject in ['sub-01','sub-02']:
                path=dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
                path.parent.mkdir(parents=True);im=nib.Nifti1Image(values,np.eye(4));im.header.set_xyzt_units('mm');nib.save(im,path)
            files=qa.files_for_pair(resources,'sub-01','sub-02')
            for path in files.values():path.parent.mkdir(parents=True,exist_ok=True)
            savemat(files['syn_aff'],{'AffineTransform_double_3_3':np.r_[np.eye(3).ravel(),np.zeros(3)].reshape(-1,1),
                                      'fixed':np.zeros((3,1))},format='4')
            for name in ['aff_image','syn_image','syn_inverse_image']:
                im=nib.Nifti1Image(values,np.eye(4));im.header.set_xyzt_units('mm');nib.save(im,files[name])
            for name in ['warp','invwarp']:a.write_field(files[name],np.zeros(3),np.eye(4),shape=(20,20,20))
            fixed=dataset/'derivatives/epochs/sub-02/anat/sub-02_mod-T1w_maskface.nii'
            moving=dataset/'derivatives/epochs/sub-01/anat/sub-01_mod-T1w_maskface.nii'
            for name in ['aff_log','syn_log']:files[name].write_text(f'--initial-moving-transform [{fixed},{moving},1]')
            files['syn_quality'].write_text(json.dumps(dict(donor='sub-01',target='sub-02',fixed_mri=str(fixed),moving_mri=str(moving))))
            def points(exe,xyz,terms):
                return np.asarray(xyz)+([.12,0,0] if str(files['invwarp']) in terms else [0,0,0])
            fake=SimpleNamespace(terms=mapping.terms,point_map=points,image_qa=lambda *args:dict(rmse=.001,n_valid=200,n_tested=200))
            rows=c.validate_pair(fake,'mocked',qa,resources,dataset,'sub-01','sub-02',a.FLIP)
            self.assertTrue(rows[0]['primary_rt010_pass']);self.assertTrue(rows[0]['pair_rt015_source_rt010_pass'])
            self.assertFalse(rows[1]['primary_rt010_pass']);self.assertTrue(rows[1]['pair_rt015_source_rt010_pass'],rows[1])
            fake.image_qa=lambda *args:dict(rmse=.1,n_valid=200,n_tested=200)
            rows=c.validate_pair(fake,'mocked',qa,resources,dataset,'sub-01','sub-02',a.FLIP)
            self.assertTrue(all(not r['pair_rt015_source_rt010_pass'] for r in rows))
            self.assertEqual(qa.MAX_ROUNDTRIP_MM,.1)

    def test_manifest_hashes_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'original';path.write_bytes(b'original')
            ledger=c.Ledger();ledger.track(path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):ledger.verify()
            args=type('Args',(),dict(output_root=root))()
            with self.assertRaises(FileExistsError):c.protect(args)


class HistoricalSensitivityRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=EXP/'outputs';cls.qa=cls.out/'06_registration_qa';cls.eval=cls.out/'06_evaluate_localization_mapped'
        if not (cls.qa/'01_pair_mode_qa.csv').is_file() or not (cls.eval/'03_primary_localization.csv').is_file():
            raise unittest.SkipTest('Real historical registration/evaluation outputs are unavailable')
        cls.pairs=c.read(cls.qa/'01_pair_mode_qa.csv')
        cls.points=c.frame(cls.out/'06_roundtrip_diagnostics/02_point_diagnostics.csv')
        cls.rows=pd.concat([c.frame(cls.eval/name) for name in ['03_primary_localization.csv','04_sensitivity_localization.csv']],ignore_index=True)

    def test_saved_diagnostics_preserve_counts_and_only_propose_candidates(self):
        self.assertEqual(len(self.points),2400)
        valid=self.points[c.yes(self.points.destination_interior)]
        self.assertEqual(len(valid),2366)
        self.assertEqual(int((valid.roundtrip_mm>.1).sum()),6)
        self.assertEqual(int((valid.roundtrip_mm>.15).sum()),0)
        self.assertEqual(int(self.pairs.status.eq('PASS_NUMERICAL_MAPPING_QA').sum()),78)
        self.assertEqual(int(self.pairs[self.pairs.registration_mode.eq('affine')].status.eq('PASS_NUMERICAL_MAPPING_QA').sum()),42)

    def test_real_metrics_classify_both_thresholds_without_promoting_primary(self):
        diagnostic=c.frame(self.out/'06_roundtrip_diagnostics/01_direction_summary.csv')
        counts={c.PRIMARY.name:0,c.SENSITIVITY.name:0}
        for row in self.pairs.to_dict('records'):
            if row['status']=='PASS_NUMERICAL_MAPPING_QA':
                metrics=[dict(finite=True,coverage=float(row[f'{d}_coverage']),maximum_mm=float(row[f'{d}_roundtrip_max_mm'])) for d in ['reference','test']]
            else:
                part=diagnostic[(diagnostic.reference_subject==row['reference_subject'])&(diagnostic.test_subject==row['test_subject'])]
                self.assertEqual(len(part),2)
                metrics=[dict(finite=True,coverage=float(r.coverage),maximum_mm=float(r.maximum_mm)) for r in part.itertuples()]
            for policy in [c.PRIMARY,c.SENSITIVITY]:counts[policy.name]+=all(c.accepted(m,policy) for m in metrics)
        self.assertEqual(counts,{c.PRIMARY.name:78,c.SENSITIVITY.name:84})
        self.assertEqual(int(self.pairs.status.eq('FAIL').sum()),6)  # Still FAIL, no reclassification.

    def test_primary_tables_and_real_matched_cohort_parity(self):
        self.assertEqual(len(self.rows),21960)
        _,common=s.same_common(self.rows,self.rows.copy())
        self.assertEqual(int(common.sum()),20520)
        qa,mapping,ev,base=c.modules()
        _,matches=ev.summaries_and_matches(self.rows,base.KEY)
        saved=c.frame(self.eval/'12_matched_affine_syn.csv')
        keys=c.KEY
        joined=matches.merge(saved,on=keys,suffixes=('_replayed','_saved'),validate='one_to_one')
        self.assertEqual(len(joined),len(saved));self.assertEqual(len(saved),8834)
        for name in ['localization_error_mm_affine','localization_error_mm_syn','syn_minus_affine_mm']:
            np.testing.assert_allclose(joined[f'{name}_replayed'],joined[f'{name}_saved'],atol=1e-9,rtol=0)
        self.assertEqual(qa.MAX_ROUNDTRIP_MM,.1)

    def test_real_no_warp_within_consistency_both_peak_scenarios(self):
        oldroot=c.ROOT/'cross_subject_no_warp/outputs/06_evaluate_localization'
        for scenario,file in [('primary','03_primary_localization.csv'),('sensitivity','04_sensitivity_localization.csv')]:
            path=oldroot/file
            if not path.is_file():self.skipTest(f'No historical no-warp table: {path}')
            old=c.frame(path);old=old[old.condition.eq('within')]
            native=self.rows[self.rows.condition.eq('within')&self.rows.scenario.eq(scenario)]
            keys=[k for k in c.KEY if k!='scenario']
            join=native.merge(old,on=keys,suffixes=('_warp','_old'),validate='one_to_one')
            self.assertEqual(len(join),732)
            np.testing.assert_allclose(pd.to_numeric(join.localization_error_mm_warp),pd.to_numeric(join.localization_error_mm_old),atol=1e-9,rtol=0)

    def test_primary_artifact_receipts_from_reviewed_protocol(self):
        protocol=(EXP/'docs/REGISTRATION_QA_SENSITIVITY_PROTOCOL.md').read_text()
        for relative in ['06_registration_qa/01_pair_mode_qa.csv','06_registration_qa/05_registration_qa_manifest.json',
                         '06_roundtrip_diagnostics/02_point_diagnostics.csv','06_evaluate_localization_mapped/07_evaluation_manifest.json',
                         '06_evaluate_localization_mapped/12_matched_affine_syn.csv']:
            value=c.sha(self.out/relative)
            self.assertIn(f'`{relative}` | `{value}`',protocol)


if __name__=='__main__':unittest.main()
