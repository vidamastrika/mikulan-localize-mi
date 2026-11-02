"""Synthetic-only Script 03 checks. Never load local EEG or invoke a solver."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/03_run_localization.py'
spec=importlib.util.spec_from_file_location('warp_localization',SCRIPT)
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class LocalizationStageTests(unittest.TestCase):
    def test_solver_engine_matches_frozen_patched_implementation(self):
        original=ast.parse((EXP/'source/00_no_warp_workflow/scripts/03_run_localization.py').read_text())
        current=ast.parse(SCRIPT.read_text())
        for name in ['Engine','peak_head','enrich_configs','run_jobs']:
            old=next(n for n in original.body if getattr(n,'name',None)==name)
            new=next(n for n in current.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(old),ast.dump(new),name)

    def test_existing_outputs_are_refused_before_input_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'existing';out.mkdir()
            sentinel=out/'keep';sentinel.write_text('unchanged')
            with patch.object(sys,'argv',[str(SCRIPT),'--output-root',str(out)]), \
                 patch.object(stage,'load_inputs',side_effect=AssertionError('must not load inputs')):
                with self.assertRaises(FileExistsError):stage.main()
            self.assertEqual(sentinel.read_text(),'unchanged')

    def test_reuse_audit_keeps_matching_records_unvalidated_and_detects_changed_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['canonical','preparation','selection']:(root/name).mkdir()
            cases=pd.DataFrame([dict(test_subject='sub-01',run='run-01')])
            cases.to_csv(root/'preparation/03_single_reference_cases.csv',index=False)
            (root/'selection/01_loso_selected_parameters.csv').write_text('synthetic selection')
            args=SimpleNamespace(canonical_root=root/'canonical',preparation_root=root/'preparation',
                                 selection_root=root/'selection',montage_template='reference')
            jobs=[];records=[]
            for condition,model in [('within','sub-01'),('cross_reference','sub-02')]:
                job=dict(method='MNE',test_subject='sub-01',reference_subject='sub-02',run='run-01',
                         fold_id='sub-01',condition=condition,model_subject=model,configuration_key='key',
                         source_condition='within' if condition=='within' else 'single_reference_no_warp')
                jobs.append(job)
                row=dict(job,status='PASS',failure_reason='',configuration_parameters='{"montage":"32"}',
                         coordinate_frame='model_head',coordinate_units='mm',estimated_head_x_mm='10',
                         estimated_head_y_mm='20',estimated_head_z_mm='30',eeg_data_sha256='a'*64,
                         epoch_times_sha256='b'*64,actual_channels_json='["e1","e2"]',
                         nominal_channels_json='["e1","e2"]',montage_template_subject='sub-02',
                         hemisphere_constraint_applied='True')
                row['condition']=row['source_condition'];records.append(row)
            configs={('MNE','sub-01'):dict(parameters={'montage':'32'})}
            status=root/'canonical/01_execution_status.csv'
            pd.DataFrame(records).to_csv(status,index=False)
            before=status.read_bytes()
            with patch.object(stage,'METHODS',['MNE']),patch.object(stage.Engine,'solve',side_effect=AssertionError('no solver')):
                rows,report=stage.audit_reuse(args,jobs,configs)
                self.assertEqual(report['record_mismatches'],0)
                self.assertFalse(report['peaks_imported'])
                self.assertTrue(all(r['reuse_status']=='PENDING_INDEPENDENT_PROVENANCE_VALIDATION' for r in rows))
                self.assertEqual(status.read_bytes(),before)
                records[0]['estimated_head_x_mm']='nan'
                pd.DataFrame(records).to_csv(status,index=False)
                rows,report=stage.audit_reuse(args,jobs,configs)
                self.assertGreater(report['record_mismatches'],0)
                self.assertIn('invalid head peak',rows[0]['issues'])


if __name__=='__main__':unittest.main()
