"""Synthetic-only tail diagnostics; never rerun subject QA or fit registrations."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/06_diagnose_registration_roundtrips.py'
spec=importlib.util.spec_from_file_location('roundtrip_diagnostics',SCRIPT)
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


class RoundtripDiagnosticTests(unittest.TestCase):
    def setup_data(self):
        points=np.array([[3.,4.,5.],[6.,7.,8.],[8.,8.,8.],[10.,10.,10.]])
        source=nib.Nifti1Image(np.zeros((20,20,20),np.float32),np.eye(4))
        qa=SimpleNamespace(foreground_points=lambda img:points.copy(),MIN_COVERAGE=.75,MAX_ROUNDTRIP_MM=.1)
        return points,source,qa

    def test_single_tail_exceedance_keeps_fixed_threshold_and_all_points(self):
        points,source,qa=self.setup_data()
        def mapping(exe,xyz,terms):
            result=np.asarray(xyz).copy()
            if terms==['inverse']:result[-1,0]+=.12
            return result
        summary,rows=stage.measure(mapping,'synthetic',source,source,['forward'],['inverse'],qa)
        self.assertEqual(summary['valid'],4);self.assertEqual(summary['over_threshold'],1)
        self.assertAlmostEqual(summary['maximum_mm'],.12)
        self.assertEqual(summary['median_mm'],0);self.assertFalse(summary['roundtrip_pass'])
        self.assertEqual(len(rows),4);self.assertTrue(rows[-1]['over_threshold'])
        self.assertAlmostEqual(rows[-1]['returned_ras_x_mm']-rows[-1]['source_ras_x_mm'],.12)
        self.assertEqual(qa.MAX_ROUNDTRIP_MM,.1)

    def test_outside_points_are_explicit_not_silently_counted_as_pass(self):
        points,source,qa=self.setup_data()
        def mapping(exe,xyz,terms):
            result=np.asarray(xyz).copy()
            if terms==['forward']:result[:2]=1000.
            return result
        summary,rows=stage.measure(mapping,'synthetic',source,source,['forward'],['inverse'],qa)
        self.assertEqual(summary['valid'],2);self.assertFalse(summary['coverage_pass'])
        self.assertEqual(sum(r['destination_interior'] for r in rows),2)
        self.assertIsNone(rows[0]['roundtrip_mm'])

    def test_invalid_returned_points_are_rejected(self):
        points,source,qa=self.setup_data()
        def mapping(exe,xyz,terms):return np.asarray(xyz).copy() if terms==['forward'] else np.full_like(xyz,np.nan)
        with self.assertRaisesRegex(ValueError,'returned'):stage.measure(mapping,'synthetic',source,source,['forward'],['inverse'],qa)

    def test_existing_output_is_refused_before_loading_any_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys,'argv',[str(SCRIPT),'--output-root',tmp]), \
                 patch.object(stage,'module',side_effect=AssertionError('must not load QA')):
                with self.assertRaises(FileExistsError):stage.main()


if __name__=='__main__':unittest.main()
