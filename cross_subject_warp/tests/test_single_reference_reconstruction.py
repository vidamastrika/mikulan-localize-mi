"""Synthetic Script 05 checks; no experiment inputs or solver execution."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/05_run_single_reference.py'
spec=importlib.util.spec_from_file_location('single_reference_reconstruction',SCRIPT)
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class SingleReferenceReconstructionTests(unittest.TestCase):
    def test_existing_output_refused_before_loading_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys,'argv',[str(SCRIPT),'--output-root',tmp]), \
                 patch.object(stage,'load_module',side_effect=AssertionError('must not load')):
                with self.assertRaises(FileExistsError):stage.main()

    def test_full_cross_schedule_has_4392_unique_reference_keys(self):
        subjects=[f'sub-{i:02d}' for i in range(1,8)]
        cases=[(s,f'run-{r:02d}') for s,n in zip(subjects,[11,6,7,9,14,7,7]) for r in range(1,n+1)]
        schedule=pd.DataFrame([dict(reference_subject=ref,test_subject=s,run=r)
            for ref in subjects for s,r in cases if ref!=s])
        selected=pd.DataFrame([dict(method=m,reference_subject=s,configuration_key=m+s)
            for m in stage.METHODS for s in subjects])
        jobs=stage.build_jobs(selected,schedule)
        self.assertEqual(len(jobs),4392)
        self.assertEqual(len({tuple(j[k] for k in stage.KEYS) for j in jobs}),4392)
        self.assertTrue(all(j['condition']=='cross_reference' and j['model_subject']==j['reference_subject']
                            and j['reference_subject']!=j['test_subject'] for j in jobs))

    def test_configuration_resolution_uses_reference_subject_not_test(self):
        selected=pd.DataFrame([dict(method='MNE',reference_subject=f'sub-{i:02d}',
                                   training_subjects=f'sub-{i:02d}') for i in range(1,8)])
        def enrich(shaped,source,doc):
            self.assertTrue(shaped.test_subject.eq(shaped.reference_subject).all())
            self.assertTrue(shaped.training_subjects.eq(shaped.reference_subject).all())
            return {('synthetic',i):{} for i in range(84)},[]
        stage.resolve_configurations(selected,SimpleNamespace(enrich_configs=enrich),None,{})

    def test_plan_and_reuse_main_paths_never_construct_engine(self):
        real=stage.load_module(EXP/'scripts/03_run_localization.py')
        for phase in ['plan','reuse-audit']:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as tmp:
                solver=SimpleNamespace(digest=lambda path:'synthetic-hash',
                    snapshot_code=lambda *args:{},check_audit=lambda *args:None,
                    lock_output=real.lock_output,atomic_csv=real.atomic_csv,atomic_json=real.atomic_json)
                solver.Engine=lambda *args: self.fail('Engine must not be constructed')
                out=Path(tmp)/'output'
                job=dict(method='MNE',reference_subject='sub-01',test_subject='sub-02',run='run-01')
                with patch.object(sys,'argv',[str(SCRIPT),'--phase',phase,'--output-root',str(out)]), \
                     patch.object(stage,'load_module',return_value=solver), \
                     patch.object(stage,'verify_selection',return_value=(None,None,{},{})), \
                     patch.object(stage,'resolve_configurations',return_value=({('MNE','sub-01'):{}},[])), \
                     patch.object(stage,'build_jobs',return_value=[job]), \
                     patch.object(stage,'audit_reuse',return_value=([{'synthetic':True}],
                         {'record_mismatches':0,'incomplete_failure_metadata':0})), \
                     contextlib.redirect_stdout(io.StringIO()):
                    stage.main()
                self.assertTrue((out/'02_reconstruction_plan.csv').is_file())
                self.assertFalse((out/'01_execution_status.csv').exists())

    def test_reuse_handles_multiple_references_and_preserves_incomplete_failure(self):
        solver=stage.load_module(EXP/'scripts/03_run_localization.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);canonical=root/'canonical';canonical.mkdir()
            selection=root/'selection';selection.mkdir()
            (selection/'01_single_reference_selected_parameters.csv').write_text('synthetic')
            args=SimpleNamespace(canonical_root=canonical,selection_root=selection,
                                 script03=EXP/'scripts/03_run_localization.py')
            jobs=[];records=[];configs={}
            for ref in ['sub-02','sub-03']:
                job=dict(method='MNE',reference_subject=ref,test_subject='sub-01',run='run-01',
                         condition='cross_reference',source_condition='single_reference_no_warp',
                         fold_id=ref,model_subject=ref,configuration_key=ref,parameter_selection='single_reference')
                jobs.append(job);configs[('MNE',ref)]={'parameters':{'montage':'32'}}
                records.append(dict(job,condition_override='unused',status='PASS',failure_reason='',
                    configuration_parameters='{"montage":"32"}',coordinate_frame='model_head',coordinate_units='mm',
                    estimated_head_x_mm='10',estimated_head_y_mm='20',estimated_head_z_mm='30',
                    montage_template_subject=ref,channels='2',actual_channels_json='["e1","e2"]',
                    nominal_channels_json='["e1","e2"]',original_bad_channels_json='[]',
                    eeg_data_sha256='a'*64,epoch_times_sha256='b'*64,hemisphere_constraint_applied='True'))
            for row in records:row['condition']='single_reference_no_warp'
            path=canonical/'01_execution_status.csv'
            pd.DataFrame(records).to_csv(path,index=False)
            before=path.read_bytes()
            with patch.object(solver.Engine,'solve',side_effect=AssertionError('no solver')):
                rows,report=stage.audit_reuse(args,solver,jobs,jobs,configs)
                self.assertEqual(report['record_mismatches'],0)
                self.assertEqual(report['incomplete_failure_metadata'],0)
                self.assertFalse(report['peaks_imported'])
                self.assertEqual(path.read_bytes(),before)
                failed=records[0];failed['status']='FAIL';failed['failure_reason']='no vertices'
                for field in ['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm',
                    'montage_template_subject','channels','actual_channels_json','nominal_channels_json',
                    'original_bad_channels_json','eeg_data_sha256','epoch_times_sha256']:failed[field]=''
                pd.DataFrame(records).to_csv(path,index=False)
                before=path.read_bytes()
                rows,report=stage.audit_reuse(args,solver,jobs,jobs,configs)
                self.assertEqual(report['record_mismatches'],0)
                self.assertEqual(report['incomplete_failure_metadata'],1)
                self.assertEqual(rows[0]['canonical_status'],'FAIL')
                self.assertEqual(path.read_bytes(),before)
                records[1]['estimated_head_x_mm']='nan'
                pd.DataFrame(records).to_csv(path,index=False)
                rows,report=stage.audit_reuse(args,solver,jobs,jobs,configs)
                self.assertEqual(report['record_mismatches'],1)
                self.assertIn('invalid native head peak',rows[1]['issues'])


if __name__=='__main__':unittest.main()
