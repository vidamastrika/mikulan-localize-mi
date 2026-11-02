"""Synthetic saved-peak evaluation; no local data, EEG, ANTs or fitting."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pandas as pd

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/06_mapped_evaluation.py'
spec=importlib.util.spec_from_file_location('mapped_evaluation_tests',SCRIPT)
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
KEY=['method','reference_subject','test_subject','run','condition']


def matrix(translation):
    m=np.eye(4);m[:3,3]=translation;return m


class MappedEvaluationTests(unittest.TestCase):
    def fixture(self,blocked_syn=True):
        original=dict(method='MNE',reference_subject='sub-01',test_subject='sub-02',run='run-01',
                      condition='cross_reference',status='PASS',failure_reason='',stimulation_pair='K1-2',hemisphere='rh',
                      estimated_head_x_mm='5',estimated_head_y_mm='6',estimated_head_z_mm='7')
        within=dict(original,condition='within',estimated_head_x_mm='19',estimated_head_y_mm='3',estimated_head_z_mm='7')
        data={'primary':pd.DataFrame([within,original]),'sensitivity':pd.DataFrame([within,original])}
        specs={'primary':('LOSO','primary'),'sensitivity':('LOSO','sensitivity')}
        truth={('sub-02','run-01'):dict(head_mm=np.array([20.,3.,7.]),ras_mm=np.array([20.,6.,7.]),stimulation_pair='K1-2',hemisphere='rh')}
        bridges={'sub-01':dict(head_to_scanner_ras_mm=matrix([10,0,0])),
                 'sub-02':dict(scanner_ras_to_surface_mm=matrix([0,0,-4]))}
        pairs={('sub-01','sub-02',mode):dict(status='FAIL' if mode=='syn' and blocked_syn else 'PASS_NUMERICAL_MAPPING_QA',
               reason='prior QA fail' if mode=='syn' and blocked_syn else '',moving_mri='/synthetic/reference',fixed_mri='/synthetic/test')
               for mode in ['affine','syn']}
        files=dict(syn_aff='/synthetic/syn.mat',warp='/synthetic/warp',invwarp='/synthetic/invwarp')
        qa=SimpleNamespace(files_for_pair=lambda *args:files,MAX_ROUNDTRIP_MM=.1)
        calls=[]
        def points(exe,xyz,terms):
            calls.append(list(terms));xyz=np.asarray(xyz)
            if terms==['aff_forward']:return xyz+[2.,0.,0.]
            if terms==['aff_inverse']:return xyz-[2.,0.,0.]
            if terms==['syn_forward'] or terms==['[/synthetic/syn.mat,1]']:return xyz+[3.,0.,0.]
            if terms==['syn_inverse']:return xyz-[3.,0.,0.]
            self.fail(f'Unexpected terms {terms}')
        mapping=SimpleNamespace(terms=lambda f:dict(aff_r2t=['aff_forward'],aff_t2r=['aff_inverse'],
                                                    syn_r2t=['syn_forward'],syn_t2r=['syn_inverse']),point_map=points)
        image=nib.Nifti1Image(np.zeros((64,64,64),np.float32),np.eye(4))
        return data,specs,truth,bridges,pairs,qa,mapping,lambda path:image,calls,files

    def evaluate(self,fixture):
        data,specs,truth,bridges,pairs,qa,mapping,load,calls,files=fixture
        with contextlib.redirect_stdout(io.StringIO()):
            return stage.evaluate_records(data,specs,truth,bridges,pairs,qa,mapping,'synthetic',Path('/synthetic'),load)

    def test_failed_pair_retained_never_mapped_and_source_peak_not_reselected(self):
        fixture=self.fixture();rows,raw=self.evaluate(fixture)
        self.assertEqual(len(rows),6);self.assertEqual(len(raw),2)
        blocked=rows[rows.registration_mode.eq('syn')]
        self.assertTrue(blocked.evaluation_status.eq('PAIR_QA_FAIL').all())
        self.assertTrue(blocked.localization_error_mm.isna().all());self.assertTrue(blocked.status.eq('PASS').all())
        self.assertEqual(fixture[8],[['aff_forward'],['aff_inverse']])
        self.assertTrue(rows.loc[rows.condition.eq('cross_reference'),'estimated_head_x_mm'].eq('5').all())
        self.assertEqual(len(rows[rows.registration_mode.eq('within')]),2)
        self.assertTrue(rows.loc[rows.registration_mode.eq('within'),'localization_error_mm'].eq(1.).all())
        affine=rows[rows.registration_mode.eq('affine')]
        self.assertTrue(affine.localization_error_mm.eq(3.).all())
        self.assertTrue(affine.mapped_test_ras_x_mm.eq(17.).all())
        self.assertTrue(affine.mapped_test_surface_z_mm.eq(3.).all())
        self.assertTrue(affine.gt_test_ras_x_mm.eq(20.).all())  # Truth was not reference-warped.
        self.assertTrue(raw.valid_localization_error.eq(False).all())
        summary,matches=stage.summaries_and_matches(rows,KEY)
        self.assertEqual(len(matches),0)
        self.assertEqual(int(summary.scheduled.sum()),6)
        self.assertEqual(int(summary.pair_qa_excluded.sum()),2)

    def test_matched_comparison_uses_exact_valid_mode_pairs_and_separate_scenarios(self):
        fixture=self.fixture(blocked_syn=False);rows,_=self.evaluate(fixture)
        summary,matches=stage.summaries_and_matches(rows,KEY)
        self.assertEqual(len(matches),2);self.assertEqual(set(matches.scenario),{'primary','sensitivity'})
        self.assertTrue(matches.localization_error_mm_affine.eq(3.).all())
        self.assertTrue(matches.localization_error_mm_syn.eq(2.).all())
        self.assertTrue(matches.syn_minus_affine_mm.eq(-1.).all())
        self.assertEqual(int(summary.valid_localizations.sum()),6)

    def test_reconstruction_failure_retains_rows_with_blank_errors(self):
        fixture=list(self.fixture());data=fixture[0]
        for key in data:
            table=data[key].copy();table.loc[1,'status']='FAIL';table.loc[1,'failure_reason']='no vertices'
            for axis in 'xyz':table.loc[1,f'estimated_head_{axis}_mm']=''
            data[key]=table
        rows,_=self.evaluate(fixture)
        failed=rows[rows.condition.eq('cross_reference')]
        self.assertEqual(len(failed),4);self.assertTrue(failed.evaluation_status.eq('RECONSTRUCTION_FAIL').all())
        self.assertTrue(failed.failure_reason.eq('no vertices').all());self.assertTrue(failed.localization_error_mm.isna().all())
        self.assertEqual(fixture[8],[])

    def test_point_roundtrip_or_domain_failures_never_get_distances(self):
        fixture=self.fixture();_,_,_,_,_,qa,mapping,load,_,files=fixture
        source=load(None)
        bad=stage.map_group(np.array([[0.,5.,5.]]),'affine',files,mapping,'synthetic',source,source,load,.1)
        self.assertEqual(bad[0]['mapping_status'],'POINT_DOMAIN_FAIL')
        def points(exe,xyz,terms):return np.asarray(xyz)+([.2,0,0] if terms==['aff_inverse'] else [0,0,0])
        mapping.point_map=points
        result=stage.map_group(np.array([[5.,5.,5.]]),'affine',files,mapping,'synthetic',source,source,load,.1)
        self.assertEqual(result[0]['mapping_status'],'POINT_ROUNDTRIP_FAIL');self.assertIsNone(result[0]['mapped'])
        def inverse_outside(exe,xyz,terms):return np.asarray(xyz)+[1000,0,0]
        mapping.point_map=inverse_outside
        result=stage.map_group(np.array([[5.,5.,5.]]),'syn',files,mapping,'synthetic',source,source,load,.1)
        self.assertEqual(result[0]['mapping_status'],'POINT_DOMAIN_FAIL')
        self.assertIn('inverse-field',result[0]['reason'])
        result=stage.map_group(np.array([[5.,5.,5.]]),'affine',files,mapping,'synthetic',source,source,load,.1)
        self.assertEqual(result[0]['mapping_status'],'POINT_DOMAIN_FAIL')
        self.assertIn('Mapped point',result[0]['reason'])

    def test_mapping_execution_errors_are_retained_as_failures(self):
        fixture=self.fixture();fixture[6].point_map=Mock(side_effect=RuntimeError('synthetic tool failure'))
        rows,_=self.evaluate(fixture)
        aff=rows[rows.registration_mode.eq('affine')]
        self.assertTrue(aff.evaluation_status.eq('MAPPING_EXECUTION_FAIL').all())
        self.assertTrue(aff.localization_error_mm.isna().all())
        self.assertTrue(aff.evaluation_failure_reason.str.contains('synthetic tool failure').all())

    def test_pair_qa_policy_accepts_partial_results_but_rejects_false_pass(self):
        qa=SimpleNamespace(MAX_ROUNDTRIP_MM=.1,MAX_RMSE=.015,MIN_COVERAGE=.75,SAMPLES=200,GRID_TOLERANCE_MM=.02)
        rows=[]
        for r in stage.SUBJECTS:
            for t in stage.SUBJECTS:
                if r==t:continue
                for mode in ['affine','syn']:
                    rows.append(dict(reference_subject=r,test_subject=t,registration_mode=mode,status='PASS_NUMERICAL_MAPPING_QA',reason='',
                        grid_max_error_mm='0',image_reproduction_rmse='.001',inverse_image_reproduction_rmse='.001',
                        reference_roundtrip_max_mm='.05',test_roundtrip_max_mm='.05',reference_coverage='1',test_coverage='1',image_valid_samples='200'))
        table=pd.DataFrame(rows);failed=table.index[table.registration_mode.eq('syn')][:6]
        table.loc[failed,'status']='FAIL';table.loc[failed,'reason']='roundtrip exceedance'
        manifest=dict(mode_pair_checks_passed=78,max_roundtrip_mm=.1,max_image_rmse=.015,min_coverage=.75,samples=200,foreground_seed=2901,grid_tolerance_mm=.02)
        result=stage.validate_pair_qa(table,manifest,qa)
        self.assertEqual(len(result),84)
        table.loc[0,'reference_roundtrip_max_mm']='.2'
        with self.assertRaisesRegex(ValueError,'contradicts'):stage.validate_pair_qa(table,manifest,qa)


if __name__=='__main__':unittest.main()
