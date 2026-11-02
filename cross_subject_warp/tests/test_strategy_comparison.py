"""Synthetic checks for Script 07 matching and validity handling."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_compare_strategies.py"
spec = importlib.util.spec_from_file_location("strategy_comparison", SCRIPT)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class StrategyComparisonTests(unittest.TestCase):
    def test_matched_delta_keeps_invalid_pairs_and_reports_only_valid_delta(self):
        keys = ["test_subject", "run", "method"]
        a = pd.DataFrame([
            dict(test_subject="sub-01", run="run-01", method="MNE", localization_error_mm=4., localization_error_valid=True),
            dict(test_subject="sub-02", run="run-02", method="MNE", localization_error_mm=None, localization_error_valid=False),
        ])
        b = pd.DataFrame([
            dict(test_subject="sub-01", run="run-01", method="MNE", localization_error_mm=3., localization_error_valid=True),
            dict(test_subject="sub-02", run="run-02", method="MNE", localization_error_mm=9., localization_error_valid=False),
        ])
        out = stage.matched_delta(a, b, keys, "a", "b")
        self.assertEqual(len(out), 2)
        self.assertEqual(int(out.matched_valid.sum()), 1)
        self.assertAlmostEqual(float(out.loc[out.matched_valid, "b_minus_a_mm"].iloc[0]), -1.)
        self.assertTrue(pd.isna(out.loc[~out.matched_valid, "b_minus_a_mm"]).all())

    def test_duplicate_keys_are_rejected(self):
        row = dict(test_subject="sub-01", run="run-01", method="MNE", localization_error_mm=4., localization_error_valid=True)
        with self.assertRaises(pd.errors.MergeError):
            stage.matched_delta(pd.DataFrame([row, row]), pd.DataFrame([row]), ["test_subject", "run", "method"], "a", "b")

    def test_summary_reports_denominator_and_valid_count(self):
        frame = pd.DataFrame([
            dict(localization_error_mm=2., localization_error_valid=True, status="PASS", group="affine"),
            dict(localization_error_mm=None, localization_error_valid=False, status="PASS", group="affine"),
        ])
        result = stage.summarize(frame, ["group"])
        self.assertEqual(int(result.n_rows.iloc[0]), 2)
        self.assertEqual(int(result.n_valid.iloc[0]), 1)
        self.assertEqual(float(result.median_error_mm.iloc[0]), 2.)

    def test_loso_and_single_reference_match_despite_different_strategy_labels(self):
        common = dict(scenario="primary", method="MNE", reference_subject="sub-02",
                      test_subject="sub-01", run="run-01", condition="cross_reference",
                      registration_mode="affine", qa_policy="primary_rt010", peak_scenario="primary",
                      localization_error_valid=True)
        loso = pd.DataFrame([dict(common, strategy="LOSO", localization_error_mm=12.)])
        single = pd.DataFrame([dict(common, strategy="Single-reference", localization_error_mm=9.)])
        keys = ["scenario", "method", "reference_subject", "test_subject", "run", "condition",
                "registration_mode", "qa_policy", "peak_scenario"]
        out = stage.matched_delta(loso, single, keys, "loso", "single_reference")
        self.assertEqual(len(out), 1)
        self.assertTrue(out.matched_valid.iloc[0])
        self.assertEqual(float(out.single_reference_minus_loso_mm.iloc[0]), -3.)


if __name__ == "__main__":
    unittest.main()
