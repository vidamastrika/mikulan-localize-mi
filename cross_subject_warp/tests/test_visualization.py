"""Synthetic checks for Script 09 plot inputs and output generation."""
import importlib.util
import tempfile
from pathlib import Path
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_visualize_results.py"
spec = importlib.util.spec_from_file_location("warp_visualization", SCRIPT)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class VisualizationTests(unittest.TestCase):
    def test_valid_metric_excludes_invalid_and_missing_errors(self):
        frame = pd.DataFrame({"localization_error_valid": [True, False, True],
                              "localization_error_mm": [4., 6., None]})
        self.assertEqual(stage.valid_metric(frame).tolist(), [True, False, False])

    def test_matched_plot_renders_synthetic_data(self):
        data = []
        for policy in stage.POLICIES:
            for mode in stage.MODES:
                data.append(dict(_merge="both", peak_scenario="primary", qa_policy=policy,
                                 registration_mode=mode, method="MNE", matched_valid=True,
                                 single_reference_minus_loso_mm=-2.5))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "synthetic.png"
            stage.plot_loso_sr(pd.DataFrame(data), Path(temp))
            out = Path(temp) / "06_loso_vs_single_reference_matched.png"
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)

    def test_within_plot_renders_three_labeled_sources(self):
        rows = [
            dict(comparison_source="historical_baseline", peak_scenario="primary", condition="historical_within",
                 qa_policy="not_applicable", method="MNE", localization_error_valid=True, localization_error_mm=8.),
            dict(comparison_source="no_warp", peak_scenario="primary", condition="within",
                 qa_policy="not_applicable", method="MNE", localization_error_valid=True, localization_error_mm=8.),
            dict(comparison_source="warp_primary", peak_scenario="primary", condition="within",
                 qa_policy="primary_rt010", method="MNE", localization_error_valid=True, localization_error_mm=8.),
        ]
        with tempfile.TemporaryDirectory() as temp:
            stage.plot_within(pd.DataFrame(rows), Path(temp))
            self.assertTrue((Path(temp) / "01_within_localization_error.png").is_file())

    def test_master_figure_uses_common_baseline_affine_syn_cases_by_policy(self):
        rows = [dict(comparison_source="historical_baseline", peak_scenario="primary", scenario="primary",
                     strategy="Historical-selected", condition="historical_within", qa_policy="not_applicable",
                     registration_mode="none", method="MNE", test_subject="sub-01", run="run-01",
                     reference_subject=None, localization_error_valid=True, localization_error_mm=10.)]
        for policy, source in [("primary_rt010", "warp_primary"), ("pair_rt015_source_rt010", "warp_sensitivity")]:
            for mode, error in [("affine", 12.), ("syn", 13.)]:
                rows.append(dict(comparison_source=source, peak_scenario="primary", scenario="primary",
                                 strategy="LOSO", condition="cross_reference", qa_policy=policy,
                                 registration_mode=mode, method="MNE", test_subject="sub-01", run="run-01",
                                 reference_subject="sub-02", localization_error_valid=True,
                                 localization_error_mm=error))
        data, coverage = stage.master_affine_syn_rows(pd.DataFrame(rows))
        self.assertEqual(len(data), 2)
        self.assertTrue(coverage.common_valid.eq(1).all())
        self.assertEqual(set(data.qa_policy), set(stage.POLICIES))
        with tempfile.TemporaryDirectory() as temp:
            stage.plot_master_baseline_affine_syn(data, Path(temp))
            plot = Path(temp) / "00_master_baseline_affine_syn.png"
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 1000)

    def test_master_figure_statistics_match_the_plotted_series(self):
        data = pd.DataFrame([
            dict(qa_policy=policy, method="MNE", baseline_error_mm=10., affine_error_mm=12., syn_error_mm=14.)
            for policy in stage.POLICIES
        ])
        stats = stage.master_figure_statistics(data)
        mne = stats[stats.method.eq("MNE")]
        self.assertEqual(len(mne), 6)
        self.assertEqual(set(mne.figure_series), {"Baseline", "Affine", "Syn"})
        self.assertTrue(mne.N.eq(1).all())
        self.assertEqual(mne.loc[mne.figure_series.eq("Syn"), "Mean_mm"].tolist(), [14., 14.])

    def test_method_strategy_statistics_include_registration_policy_and_denominators(self):
        rows = [dict(comparison_source="historical_baseline", strategy="Historical-selected", condition="historical_within",
                     peak_scenario="primary", registration_mode="none", qa_policy="not_applicable", method="MNE",
                     localization_error_valid=True, localization_error_mm=10.)]
        for strategy, source in [("LOSO", "warp_primary"), ("Single-reference", "warp_primary")]:
            rows.append(dict(comparison_source=source, strategy=strategy, condition="cross_reference",
                             peak_scenario="primary", registration_mode="affine", qa_policy="primary_rt010",
                             method="MNE", localization_error_valid=True, localization_error_mm=12.))
        result = stage.method_strategy_statistics(pd.DataFrame(rows))
        self.assertEqual(set(result.strategy), {"Baseline", "LOSO", "Single-reference"})
        self.assertTrue(result.Valid_N.eq(1).all())
        self.assertTrue(result.Invalid_N.eq(0).all())


if __name__ == "__main__":
    unittest.main()
