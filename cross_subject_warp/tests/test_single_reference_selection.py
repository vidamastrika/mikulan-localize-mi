"""Script 04 synthetic orchestration checks; no real grids or experiment inputs."""
import contextlib
import hashlib
import importlib.util
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
SCRIPT=EXP/'scripts/04_select_single_reference_parameters.py'
spec=importlib.util.spec_from_file_location('single_reference_stage',SCRIPT)
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class SingleReferenceStageTests(unittest.TestCase):
    def test_existing_output_refused_before_shared_code_or_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys,'argv',[str(SCRIPT),'--output-root',tmp]), \
                 patch.object(stage,'import_shared',side_effect=AssertionError('must not import')):
                with self.assertRaises(FileExistsError):stage.main()

    def exercise_synthetic_selection(self,missing_winner=False):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); source=base/'source';source.mkdir()
            (source/'source_manifest.json').write_text('{}\n')
            cases=pd.DataFrame([dict(test_subject=f'sub-{i:02d}',run=f'run-{run:02d}')
                for i,count in enumerate([11,6,7,9,14,7,7],1) for run in range(1,count+1)])
            methods=['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC',
                     'LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
            calls=[]
            def select(data,training,expected,method,reliability):
                self.assertEqual(len(training),1)
                reference=training[0]
                self.assertTrue(all(subject==reference for subject,run in expected))
                self.assertEqual(reliability,'primary-all-pass')
                calls.append((method,reference))
                winner=dict(configuration_key=method+reference,configuration_parameters='{}',
                            score_mean_subject_median_mm=1.)
                if missing_winner and method=='MNE' and reference=='sub-01':winner=None
                return winner,pd.DataFrame([dict(eligible=winner is not None)]),[],[],['montage']
            mod=SimpleNamespace(METHODS=methods,
                digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                snapshot=lambda args:(source,{'methods':{m:['synthetic.csv'] for m in methods}}),
                verify_preparation=lambda *args:{'readiness':'BLOCKED','plan_matches_frozen':True},
                load_plan=lambda *args:(pd.DataFrame(),cases), kv=lambda args:{},
                load_method=lambda *args:(None,'synthetic_error'),select_fold=select)
            output=base/'output'
            with patch.object(sys,'argv',[str(SCRIPT),'--source-root',str(source),
                    '--preparation-root',str(base/'preparation'),'--output-root',str(output)]), \
                 patch.object(stage,'import_shared',return_value=mod),contextlib.redirect_stdout(io.StringIO()):
                if missing_winner:
                    with self.assertRaises(SystemExit):stage.main()
                else:stage.main()
            self.assertEqual(len(calls),84)
            selected=pd.read_csv(output/'01_single_reference_selected_parameters.csv')
            self.assertEqual(len(selected),83 if missing_winner else 84)
            schedule=pd.read_csv(output/'09_reference_test_schedule.csv')
            self.assertEqual(len(schedule),366)
            self.assertTrue(schedule.reference_subject.ne(schedule.test_subject).all())
            self.assertEqual(len(schedule[['reference_subject','test_subject']].drop_duplicates()),42)
            manifest=json.loads((output/'07_selection_manifest.json').read_text())
            self.assertEqual(manifest['status'],'INCOMPLETE' if missing_winner else 'COMPLETE')
            self.assertFalse(manifest['inverse_reconstruction_executed'])
            self.assertFalse(manifest['registration_fitting_executed'])
            self.assertFalse(manifest['anatomical_evaluation_ready'])
            self.assertEqual((source/'source_manifest.json').read_text(),'{}\n')

    def test_reference_only_calls_and_complete_schedule(self):
        self.exercise_synthetic_selection()

    def test_missing_winner_records_incomplete_without_fallback(self):
        self.exercise_synthetic_selection(missing_winner=True)


if __name__=='__main__':unittest.main()
