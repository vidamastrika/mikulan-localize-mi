"""Synthetic unit tests for the warped reference sensitivity summaries."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_reference_sensitivity.py"
spec = importlib.util.spec_from_file_location("reference_sensitivity", SCRIPT)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class ReferenceSensitivityTests(unittest.TestCase):
    def test_stats_uses_sample_sd_and_ignores_missing_values(self):
        result = stage.stats(pd.Series([1., 3., np.nan]))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["median_mm"], 2.)
        self.assertAlmostEqual(result["std_mm"], np.sqrt(2.))

    def test_per_run_keeps_scheduled_invalid_reference_in_denominator(self):
        frame = pd.DataFrame([
            dict(qa_policy="p", peak_scenario="primary", registration_mode="affine", method="MNE",
                 test_subject="sub-01", run="run-01", reference_subject="sub-02", reconstruction_status="PASS",
                 localization_error_valid=True, localization_error_mm=10.),
            dict(qa_policy="p", peak_scenario="primary", registration_mode="affine", method="MNE",
                 test_subject="sub-01", run="run-01", reference_subject="sub-03", reconstruction_status="PASS",
                 localization_error_valid=False, localization_error_mm=np.nan),
        ])
        result = stage.sensitivity_per_run(frame).iloc[0]
        self.assertEqual(result.scheduled_references, 6)
        self.assertEqual(result.reference_count, 2)
        self.assertEqual(result.mapped_valid, 1)
        self.assertEqual(result.mapped_invalid, 1)
        self.assertAlmostEqual(result.mapping_coverage, .5)

    def test_peak_sensitivity_pairing_retains_validity_and_delta(self):
        keys = dict(method="MNE", test_subject="sub-01", run="run-01", reference_subject="sub-02",
                    registration_mode="syn", qa_policy="pair_rt015_source_rt010")
        frame = pd.DataFrame([
            dict(**keys, peak_scenario="primary", localization_error_mm=12., localization_error_valid=True,
                 reconstruction_status="PASS"),
            dict(**keys, peak_scenario="peak_sensitivity", localization_error_mm=9., localization_error_valid=True,
                 reconstruction_status="PASS"),
        ])
        result = stage.compare_peak_scenarios(frame).iloc[0]
        self.assertTrue(result.valid_both)
        self.assertEqual(result.sensitivity_minus_primary_mm, -3.)
        self.assertFalse(result.status_changed)

    def test_invalid_scenario_pair_has_blank_difference(self):
        keys = dict(method="MNE", test_subject="sub-01", run="run-01", reference_subject="sub-02",
                    registration_mode="syn", qa_policy="primary_rt010")
        frame = pd.DataFrame([
            dict(**keys, peak_scenario="primary", localization_error_mm=np.nan, localization_error_valid=False,
                 reconstruction_status="PAIR_QA_FAIL"),
            dict(**keys, peak_scenario="peak_sensitivity", localization_error_mm=8., localization_error_valid=True,
                 reconstruction_status="PASS_MAPPED"),
        ])
        result = stage.compare_peak_scenarios(frame).iloc[0]
        self.assertFalse(result.valid_both)
        self.assertTrue(np.isnan(result.sensitivity_minus_primary_mm))
        self.assertTrue(result.status_changed)


if __name__ == "__main__":
    unittest.main()
