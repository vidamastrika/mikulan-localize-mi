"""Synthetic MRI-only registration QA tests; no ANTs execution or subject QA."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/06_validate_registration_resources.py'
spec=importlib.util.spec_from_file_location('registration_qa_checks',SCRIPT)
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


def image(data,affine=None):
    result=nib.Nifti1Image(data,np.eye(4) if affine is None else affine)
    result.header.set_xyzt_units('mm')
    return result


class RegistrationQATests(unittest.TestCase):
    def test_pinned_point_transform_order_and_ras_lps_conversion(self):
        mapping,_=stage.historical_mapping()
        files=dict(aff=Path('/tmp/aff.mat'),syn_aff=Path('/tmp/syn.mat'),warp=Path('/tmp/warp.nii.gz'),invwarp=Path('/tmp/inv.nii.gz'))
        terms=mapping.terms(files)
        self.assertEqual(terms['aff_r2t'],['[/tmp/aff.mat,1]'])
        self.assertEqual(terms['syn_r2t'],['[/tmp/syn.mat,1]','/tmp/inv.nii.gz'])
        self.assertEqual(terms['syn_t2r'],['/tmp/warp.nii.gz','/tmp/syn.mat'])
        np.testing.assert_array_equal(mapping.FLIP@mapping.FLIP,np.eye(3))
        np.testing.assert_array_equal(np.array([[1,2,3]])@mapping.FLIP.T,[[-1,-2,3]])

    def test_grid_checks_physical_corner_coordinates_and_units(self):
        source=image(np.zeros((16,16,16),np.float32))
        self.assertEqual(stage.grid_error_mm(source,source),0)
        changed=np.eye(4);changed[0,3]=.1
        with self.assertRaises(ValueError):stage.grid_error_mm(image(np.zeros((16,16,16),np.float32),changed),source)
        source.header.set_xyzt_units('meter')
        with self.assertRaises(ValueError):stage.grid_error_mm(source,source)

    def test_field_checks_require_finite_three_component_fixed_domain(self):
        fixed=image(np.zeros((8,8,8),np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'field.nii'
            field=image(np.zeros((8,8,8,1,3),np.float32));field.header.set_intent('vector');nib.save(field,path)
            self.assertTrue(stage.check_field(path,fixed)['finite'])
            values=np.zeros((8,8,8,1,3),np.float32);values[0,0,0,0,0]=np.nan
            field=image(values);field.header.set_intent('vector');nib.save(field,path)
            with self.assertRaisesRegex(ValueError,'Nonfinite'):stage.check_field(path,fixed)
            field=image(np.zeros((8,8,8),np.float32));nib.save(field,path)
            with self.assertRaisesRegex(ValueError,'three-component'):stage.check_field(path,fixed)

    def test_roundtrip_coverage_and_error_are_enforced(self):
        values=np.zeros((20,20,20),np.float32);values[3:-3,3:-3,3:-3]=1
        source=image(values)
        calls=[]
        def identity(exe,points,terms):calls.append(list(terms));return np.asarray(points).copy()
        qa=stage.roundtrip_qa(identity,'synthetic',source,source,['forward'],['inverse'])
        self.assertEqual(qa['max_roundtrip_mm'],0)
        self.assertEqual(calls,[['forward'],['inverse']])
        def bad_inverse(exe,points,terms):return np.asarray(points)+(1 if terms==['inverse'] else 0)
        with self.assertRaisesRegex(ValueError,'roundtrip'):stage.roundtrip_qa(bad_inverse,'synthetic',source,source,['forward'],['inverse'])
        def outside(*args):return np.full_like(args[1],10000.)
        with self.assertRaisesRegex(ValueError,'coverage'):stage.roundtrip_qa(outside,'synthetic',source,source,['forward'],['inverse'])

    def test_log_identifies_ordered_fixed_and_moving_inputs(self):
        stage.verify_log_identity('fixed image: /tmp/test.nii and moving image: /tmp/ref.nii','/tmp/test.nii','/tmp/ref.nii')
        stage.verify_log_identity('--initial-moving-transform [/tmp/test.nii,/tmp/ref.nii,1]','/tmp/test.nii','/tmp/ref.nii')
        with self.assertRaises(ValueError):stage.verify_log_identity('fixed image: /tmp/ref.nii moving image: /tmp/test.nii','/tmp/test.nii','/tmp/ref.nii')

    def test_pair_pipeline_uses_saved_images_and_reports_modes_independently(self):
        mapping,_=stage.historical_mapping()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);resources=root/'resources';dataset=root/'dataset'
            files=stage.files_for_pair(resources,'sub-01','sub-02')
            values=np.zeros((20,20,20),np.float32);values[3:-3,3:-3,3:-3]=1
            for subject in ['sub-01','sub-02']:
                path=dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
                path.parent.mkdir(parents=True);nib.save(image(values),path)
            fixed=dataset/'derivatives/epochs/sub-02/anat/sub-02_mod-T1w_maskface.nii'
            moving=dataset/'derivatives/epochs/sub-01/anat/sub-01_mod-T1w_maskface.nii'
            for path in files.values():path.parent.mkdir(parents=True,exist_ok=True)
            for name in ['aff_image','syn_image','syn_inverse_image']:nib.save(image(values),files[name])
            for name in ['warp','invwarp']:
                field=image(np.zeros((20,20,20,1,3),np.float32));field.header.set_intent('vector');nib.save(field,files[name])
            for name in ['aff_log','syn_log']:files[name].write_text(f'--initial-moving-transform [{fixed},{moving},1]')
            files['syn_quality'].write_text(json.dumps(dict(donor='sub-01',target='sub-02',fixed_mri=str(fixed),moving_mri=str(moving))))
            calls=[]
            def points(exe,xyz,terms):calls.append(list(terms));return np.asarray(xyz).copy()
            def qa(exe,fixed,moving,warped,terms,n):
                return dict(rmse=0.,n_valid=n,n_tested=n)
            fake=SimpleNamespace(terms=mapping.terms,image_qa=qa,point_map=points)
            with patch.object(stage.subprocess,'run',side_effect=AssertionError('no external calls in synthetic test')):
                rows,fields=stage.validate_pair(fake,'synthetic',resources,dataset,'sub-01','sub-02')
            self.assertEqual([row['status'] for row in rows],['PASS_NUMERICAL_MAPPING_QA']*2)
            self.assertEqual(len(fields),2);self.assertEqual(len(calls),8)
            self.assertEqual(json.loads(rows[1]['reference_to_test_terms']),mapping.terms(files)['syn_r2t'])
            def bad_affine(exe,fixed,moving,warped,terms,n):
                return dict(rmse=.1 if warped==files['aff_image'] else 0.,n_valid=n,n_tested=n)
            fake.image_qa=bad_affine
            rows,_=stage.validate_pair(fake,'synthetic',resources,dataset,'sub-01','sub-02')
            self.assertEqual(rows[0]['status'],'FAIL');self.assertEqual(rows[1]['status'],'PASS_NUMERICAL_MAPPING_QA')
            self.assertIn('RMSE',rows[0]['reason'])

    def test_existing_outputs_refused_before_mapping_or_external_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys,'argv',[str(SCRIPT),'--output-root',tmp]), \
                 patch.object(stage,'historical_mapping',side_effect=AssertionError('must not load')), \
                 patch.object(stage.subprocess,'run',side_effect=AssertionError('no external command')):
                with self.assertRaises(FileExistsError):stage.main()


if __name__=='__main__':unittest.main()
