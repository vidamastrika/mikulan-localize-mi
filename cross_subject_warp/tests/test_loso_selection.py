"""Regression checks for training-only selection and historical policy preservation.

Run: python -m unittest discover -s cross_subject_warp/tests -v
"""
import ast
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/02_select_loso_parameters.py'
spec=importlib.util.spec_from_file_location('warp_selection',SCRIPT)
selection=importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.training=[f'sub-{i:02d}' for i in range(2,8)]
        self.expected={(s,'run-01') for s in self.training}
        self.rows=[dict(subject=s,run='run-01',status='PASS',montage='32',
                        loose=.6,depth=.2,snr=1,lambda2=1,_raw_error=10)
                   for s in self.training]

    def test_held_out_failures_missing_settings_and_conflicts_do_not_affect_selection(self):
        baseline=selection.select_fold(pd.DataFrame(self.rows),self.training,self.expected,'MNE')[0]
        heldout=[dict(self.rows[0],subject='sub-01',status='FAIL',loose=None,_raw_error=None),
                 dict(self.rows[0],subject='sub-01',_raw_error=-1)]
        observed=selection.select_fold(pd.DataFrame(self.rows+heldout),self.training,self.expected,'MNE')[0]
        self.assertEqual(baseline,observed)

    def test_training_failure_or_conflicting_duplicate_prevents_selection(self):
        failed=[dict(r) for r in self.rows]
        failed[0]['status']='FAIL'
        self.assertIsNone(selection.select_fold(pd.DataFrame(failed),self.training,self.expected,'MNE')[0])
        conflicting=self.rows+[dict(self.rows[0],_raw_error=11)]
        winner,_,_,duplicates,_=selection.select_fold(pd.DataFrame(conflicting),self.training,self.expected,'MNE')
        self.assertIsNone(winner)
        self.assertTrue(duplicates[0]['conflict'])

    def test_selection_functions_and_policy_constants_match_frozen_implementation(self):
        old=ast.parse((EXP/'source/00_no_warp_workflow/scripts/02_select_loso_parameters.py').read_text())
        new=ast.parse(SCRIPT.read_text())
        for name in ['canonical','norm','load_plan','load_method','prepare_training','select_fold']:
            original=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            current=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(original),ast.dump(current),name)
        for name in ['METHODS','PARAMS','REQUIRED','PRIMARY','ALLOWED']:
            def statements(tree):
                return [ast.dump(n) for n in tree.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
                        or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                        and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
                        and n.value.func.value.id==name]
            self.assertEqual(statements(old),statements(new),name)


if __name__=='__main__':
    unittest.main()
