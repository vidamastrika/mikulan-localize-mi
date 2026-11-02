"""Synthetic-only Script 06 tests; never run an experiment phase on local data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/06_evaluate_localization.py'
spec=importlib.util.spec_from_file_location('evaluation_preflight',SCRIPT)
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


class EvaluationPreflightTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for condition,model in [('within','sub-01'),('cross_reference','sub-02')]:
            rows.append(dict(method='MNE',test_subject='sub-01',reference_subject='sub-02',run='run-01',
                condition=condition,model_subject=model,configuration_key='key',configuration_parameters='{"montage":"32"}',
                status='PASS',failure_reason='',native_input_validation='PASS_RECOMPUTED',
                actual_channels_json='["e1","e2"]',nominal_channels_json='["e1","e2"]',original_bad_channels_json='[]',
                eeg_data_sha256='a'*64,epoch_times_sha256='b'*64,montage_template_subject='sub-02',n_epochs='2',
                covariance_method_selected='shrunk',coordinate_frame='model_head',coordinate_units='mm',
                estimated_head_x_mm='1',estimated_head_y_mm='2',estimated_head_z_mm='3',peak_vertex='10',peak_time_s='0.001'))
        return pd.DataFrame(rows)

    def validate(self,table):
        keys=set(map(tuple,self.frame()[stage.KEY].values))
        configs={('MNE','sub-01'):dict(configuration_key='key',configuration_parameters='{"montage":"32"}')}
        return stage.validate_table(table,keys,'synthetic',configs,'LOSO',2,2)

    def test_native_validation_accepts_matched_pairs_and_rejects_input_or_peak_changes(self):
        frame=self.frame();self.assertEqual(self.validate(frame)['status'],'PASS')
        altered=frame.copy();altered.loc[1,'eeg_data_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'paired'):self.validate(altered)
        altered=frame.copy();altered.loc[1,'estimated_head_x_mm']='nan'
        with self.assertRaisesRegex(ValueError,'invalid peak'):self.validate(altered)
        altered=frame.copy();altered.loc[1,'configuration_key']='different'
        with self.assertRaisesRegex(ValueError,'configuration'):self.validate(altered)

    def test_sensitivity_preserves_other_rows_and_requires_exact_input_match(self):
        primary=self.frame();primary.loc[1,'status']='FAIL';primary.loc[1,'failure_reason']='no vertices'
        for col in stage.PEAKS:primary.loc[1,col]=''
        variant=primary.copy();variant['sensitivity_override']='false';variant['peak_selection_policy']='known_stimulation_hemisphere'
        variant['hemisphere_constraint_applied']='True'
        variant.loc[1,'status']='PASS';variant.loc[1,'failure_reason']=''
        for col in stage.PEAKS:variant.loc[1,col]='1'
        variant.loc[1,'sensitivity_override']='true';variant.loc[1,'peak_selection_policy']='whole_brain_absolute_peak'
        variant.loc[1,'hemisphere_constraint_applied']='False'
        self.assertEqual(stage.validate_variant(primary,variant,1,'synthetic')['overrides'],1)
        changed=variant.copy();changed.loc[1,'eeg_data_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'inputs changed'):stage.validate_variant(primary,changed,1,'synthetic')
        changed=variant.copy();changed.loc[0,'estimated_head_x_mm']='99'
        with self.assertRaisesRegex(ValueError,'nonoverride'):stage.validate_variant(primary,changed,1,'synthetic')

    def test_evaluation_existing_output_refused_before_reading_any_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys,'argv',[str(SCRIPT),'--phase','evaluate','--output-root',tmp]), \
                 patch.object(stage,'read',side_effect=AssertionError('must not read inputs')):
                with self.assertRaises(FileExistsError):stage.main()

    def test_existing_output_refused_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys,'argv',[str(SCRIPT),'--output-root',tmp]), \
                 patch.object(stage,'read',side_effect=AssertionError('must not read inputs')):
                with self.assertRaises(FileExistsError):stage.main()

    def test_complete_case_schedule_has_expected_counts(self):
        subjects=[f'sub-{i:02d}' for i in range(1,8)]
        cases=pd.DataFrame([dict(test_subject=s,reference_subject='sub-02' if s=='sub-01' else 'sub-01',run=f'run-{r:02d}')
            for s,n in zip(subjects,[11,6,7,9,14,7,7]) for r in range(1,n+1)])
        methods=[f'm{i}' for i in range(12)]
        self.assertEqual(len(stage.expected_keys(cases,methods,'LOSO')),1464)
        self.assertEqual(len(stage.expected_keys(cases,methods,'Single-reference')),4392)

    def test_header_rigidity_never_certifies_physical_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);subject='sub-01'
            h5=root/'derivatives/sourcemodelling'/subject/'xfm'/f'{subject}_from-head_to-surface.h5'
            mri=root/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
            for path in [h5,mri]:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic')
            for space in ['surface','T1w']:
                path=root/'derivatives/epochs'/subject/'ieeg'/f'{subject}_task-seegstim_space-{space}_coordsystem.json'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'iEEGCoordinateUnits':'m'}))
            class Header(dict):
                def get_xyzt_units(self):return 'mm','sec'
            image=SimpleNamespace(affine=np.eye(4),shape=(4,4,4),header=Header(qform_code=1,sform_code=1))
            class File:
                def __enter__(self):return {'trans':np.eye(4)}
                def __exit__(self,*args):pass
            with patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda *args:File()),
                'nibabel':SimpleNamespace(load=lambda *args:image,aff2axcodes=lambda *args:('R','A','S'))}):
                row,paths=stage.coordinate_preflight(root,subject)
            self.assertEqual(row['numerical_status'],'PASS_NUMERICAL_ONLY')
            self.assertEqual(row['physical_bridge_status'],'UNRESOLVED')
            self.assertEqual(len(paths),4)


if __name__=='__main__':unittest.main()
