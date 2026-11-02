"""Completion must preserve solver outcomes and reject unsupported metadata."""
import ast
import tempfile
import sys
from unittest.mock import patch
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/03_05_finalize_native_datasets.py'
spec=importlib.util.spec_from_file_location('native_completion',SCRIPT)
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


class NativeCompletionTests(unittest.TestCase):
    def record(self,status='FAIL'):
        return dict(method='Gamma-MAP',reference_subject='sub-01',test_subject='sub-05',run='run-08',
                    condition='cross_reference',status=status,failure_reason='no vertices' if status=='FAIL' else '',
                    estimated_head_x_mm='',estimated_head_y_mm='',estimated_head_z_mm='',peak_vertex='',peak_time_s='',
                    montage_template_subject='',actual_channels_json='')

    def test_missing_input_metadata_is_added_with_evidence_without_changing_failure(self):
        original=self.record();updated,changes=stage.complete_metadata(original,
            dict(montage_template_subject='sub-01',actual_channels_json='["e1","e2"]'),'recomputed')
        self.assertEqual(original['actual_channels_json'],'')
        self.assertEqual(updated['status'],'FAIL')
        self.assertEqual(updated['failure_reason'],'no vertices')
        self.assertTrue(all(updated[c]=='' for c in stage.PEAKS))
        self.assertEqual(len(changes),2)
        self.assertTrue(all(c['evidence']=='recomputed' for c in changes))

    def test_existing_completion_destination_is_refused_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'03_reference_based_final';target.mkdir()
            sentinel=target/'keep';sentinel.write_text('original')
            with patch.object(sys,'argv',[str(SCRIPT),'--output-base',tmp]), \
                 patch.object(stage,'module',side_effect=AssertionError('must not load inputs')):
                with self.assertRaises(FileExistsError):stage.main()
            self.assertEqual(sentinel.read_text(),'original')

    def test_existing_metadata_conflicts_are_not_overwritten(self):
        original=self.record();original['montage_template_subject']='sub-02'
        with self.assertRaisesRegex(ValueError,'Input metadata mismatch'):
            stage.complete_metadata(original,{'montage_template_subject':'sub-01'},'recomputed')

    def test_semantically_equal_metadata_does_not_get_rewritten(self):
        original=self.record();original['channels']='32.0'
        updated,changes=stage.complete_metadata(original,{'channels':'32'},'recomputed')
        self.assertEqual(changes,[]);self.assertEqual(updated['channels'],'32.0')

    def test_sensitivity_requires_exact_supplement_and_preserves_primary(self):
        original=self.record();replacement=dict(original,status='PASS',failure_reason='',estimated_head_x_mm='1')
        primary=pd.DataFrame([original]);variant=pd.DataFrame([replacement]);supplement=pd.DataFrame([replacement])
        keys=stage.verify_variant(primary,variant,1,supplement)
        self.assertEqual(len(keys),1);self.assertEqual(primary.iloc[0].status,'FAIL')
        bad=dict(replacement,estimated_head_x_mm='2')
        with self.assertRaises(ValueError):stage.verify_variant(primary,variant,1,pd.DataFrame([bad]))
        with self.assertRaises(ValueError):stage.verify_variant(variant,pd.DataFrame([bad]),1,pd.DataFrame([bad]))

    def test_documented_whole_brain_annotations_do_not_allow_peak_changes(self):
        primary=self.record()
        source=dict(primary,status='PASS',failure_reason='',estimated_head_x_mm='1',
                    hemisphere_constraint_applied='True',selection_criterion='maximum_absolute_amplitude')
        variant=dict(source,hemisphere_constraint_applied='False',
                     selection_criterion='maximum_absolute_amplitude_whole_brain')
        self.assertEqual(len(stage.verify_variant(pd.DataFrame([primary]),pd.DataFrame([variant]),1,
            pd.DataFrame([source]),annotated_whole_brain=True)),1)
        changed=dict(variant,estimated_head_x_mm='2')
        with self.assertRaises(ValueError):
            stage.verify_variant(pd.DataFrame([primary]),pd.DataFrame([changed]),1,
                                 pd.DataFrame([source]),annotated_whole_brain=True)

    def test_recomputation_has_no_inverse_or_covariance_fit_entry_points(self):
        tree=ast.parse(SCRIPT.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recompute_inputs')
        calls={n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls & {'solve','prepare','estimate_noise_covariance','make_inverse_operator','fit_dipole'})


if __name__=='__main__':unittest.main()
