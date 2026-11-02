"""Synthetic native-frame geometry checks; no subject data or fitting."""
import importlib.util
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

EXP=Path(__file__).resolve().parents[1]
SCRIPT=EXP/'scripts/06_validate_coordinate_bridge.py'
spec=importlib.util.spec_from_file_location('native_bridge_checks',SCRIPT)
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


class CoordinateBridgeTests(unittest.TestCase):
    def meta(self,centre=(2.,-3.,4.),zooms=(1.,1.,1.),dimensions=(256,256,256)):
        meta={f'VolGeom{name}':str(v) for name,v in zip(['Width','Height','Depth'],dimensions)}
        meta.update({f'VolGeom{x}size':str(v) for x,v in zip('XYZ',zooms)})
        directions=np.array([[-1,0,0],[0,0,1],[0,-1,0]],float)
        meta.update({f'VolGeom{x}_{a}':str(directions[j,i]) for i,x in enumerate('XYZ') for j,a in enumerate('RAS')})
        meta.update({f'VolGeomC_{a}':str(v) for a,v in zip('RAS',centre)})
        return meta

    def test_conformed_native_surface_to_scanner_is_derived_centre_translation(self):
        g=stage.volume_geometry(self.meta())
        expected=np.eye(4);expected[:3,3]=[2.,-3.,4.]
        np.testing.assert_allclose(g['surface_to_scanner'],expected,atol=1e-12)
        np.testing.assert_allclose(stage.apply(g['norig'],[[128.,128.,128.]]),[[2.,-3.,4.]])
        self.assertFalse(np.allclose(stage.apply(g['norig'],[[127.5,127.5,127.5]]),[[2.,-3.,4.]]))

    def test_geometry_handles_anisotropic_voxels_and_nonconformed_dimensions(self):
        g=stage.volume_geometry(self.meta(zooms=(.7,1.2,2.),dimensions=(180,200,120)))
        expected=np.eye(4);expected[:3,3]=[2.,-3.,4.]
        np.testing.assert_allclose(g['surface_to_scanner'],expected,atol=1e-12)
        np.testing.assert_allclose(stage.apply(g['norig'],[[90,100,60]]),[[2.,-3.,4.]])

    def test_metre_h5_bridge_composition_and_inverse_roundtrip(self):
        head_to_surface=np.eye(4);head_to_surface[:3,3]=[-.002,.003,-.004]
        surface_to_world=stage.volume_geometry(self.meta())['surface_to_scanner']
        composed=surface_to_world@stage.h5_metres_to_mm(head_to_surface)
        np.testing.assert_allclose(composed,np.eye(4),atol=1e-12)
        points=np.array([[10.,20.,30.],[-20.,0.,40.]])
        np.testing.assert_allclose(stage.apply(np.linalg.inv(surface_to_world),stage.apply(surface_to_world,points)),points)

    def test_grid_corner_check_detects_voxel_axis_or_origin_mismatch(self):
        a=stage.volume_geometry(self.meta())['norig'];b=a.copy();b[0,3]+=.5
        self.assertAlmostEqual(stage.grid_discrepancy_mm(a,b,[256]*3),.5)
        b=a.copy();b[:3,0]*=-1
        self.assertGreater(stage.grid_discrepancy_mm(a,b,[256]*3),100)

    def test_contacts_are_checked_without_fitting_and_mismatched_names_rejected(self):
        surface=pd.DataFrame(dict(name=['a','b','c','d'],x=[.01,.02,.03,.04],y=[0.,.01,.02,.03],z=[.02,.02,.02,.02]))
        target=surface.copy();target['x']+=.002;target['y']-=.003;target['z']+=.004
        bridge=stage.volume_geometry(self.meta())['surface_to_scanner']
        names,errors=stage.contacts_in_mm(surface,target,bridge)
        self.assertEqual(len(names),4);np.testing.assert_allclose(errors,0,atol=1e-12)
        broken=target.copy();broken.loc[0,'name']='unknown'
        with self.assertRaises(ValueError):stage.contacts_in_mm(surface,broken,bridge)
        np.testing.assert_allclose(bridge[:3,3],[2.,-3.,4.])

    def test_missing_geometry_or_nonrigid_h5_is_not_accepted(self):
        with self.assertRaises(KeyError):stage.volume_geometry({})
        matrix=np.eye(4);matrix[0,0]=2
        with self.assertRaises(ValueError):stage.h5_metres_to_mm(matrix)

    def test_subject_audit_publishes_only_consistent_synthetic_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);subject='sub-01';model=root/'derivatives/sourcemodelling'/subject
            ieeg=root/'derivatives/epochs'/subject/'ieeg'
            mri=root/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
            h5=model/'xfm'/f'{subject}_from-head_to-surface.h5'
            fwd=model/'fwd'/f'{subject}_fwd.fif'
            for path in [mri,h5,fwd]:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic')
            contacts=pd.DataFrame(dict(name=['a','b','c','d'],x=[.01,.02,.03,.04],y=[0.,.01,.02,.03],z=[.02]*4))
            target=contacts.copy();target['x']+=.002;target['y']-=.003;target['z']+=.004
            ieeg.mkdir(parents=True)
            for space,table in [('surface',contacts),('T1w',target)]:
                table.to_csv(ieeg/f'{subject}_task-seegstim_space-{space}_electrodes.tsv',sep='\t',index=False)
                (ieeg/f'{subject}_task-seegstim_space-{space}_coordsystem.json').write_text(json.dumps({'iEEGCoordinateUnits':'m'}))
            surface=ieeg/f'{subject}_task-seegstim_space-surface_electrodes.tsv'
            audited=dict(forward_path=str(fwd),forward_sha256=stage.sha(fwd),head_to_surface_path=str(h5),
                         h5_sha256=stage.sha(h5),contacts_path=str(surface),contacts_sha256=stage.sha(surface))
            g=stage.volume_geometry(self.meta())
            h2s=np.eye(4);h2s[:3,3]=[-.002,.003,-.004]
            class Header(dict):
                def get_xyzt_units(self):return 'mm','sec'
            image=SimpleNamespace(affine=g['norig'].copy(),shape=(256,256,256),ndim=3,header=Header(),
                                  get_qform=lambda coded:(g['norig'],1),get_sform=lambda coded:(g['norig'],1))
            def load(path):
                if str(path).endswith('.gii'):
                    hemi='L' if 'hemi-L_' in str(path) else 'R'
                    meta=self.meta();meta.update(GeometricType='Anatomical',AnatomicalStructureSecondary='Pial',
                        AnatomicalStructurePrimary='CortexLeft' if hemi=='L' else 'CortexRight',
                        Name='/synthetic/'+('lh.pial' if hemi=='L' else 'rh.pial'))
                    return SimpleNamespace(darrays=[SimpleNamespace(intent=1008,data=np.zeros((4,3)),meta=meta,
                        coordsys=SimpleNamespace(dataspace=0,xformspace=3,xform=np.eye(4)))])
                return image
            class File:
                def __enter__(self):return {'trans':h2s}
                def __exit__(self,*args):pass
            fake_mne=SimpleNamespace(read_forward_solution=lambda *args,**kwargs:
                {'mri_head_t':{'from':5,'to':4,'trans':np.linalg.inv(h2s)}})
            modules={'h5py':SimpleNamespace(File=lambda *args:File()),'nibabel':SimpleNamespace(load=load),
                     'mne':fake_mne,'mne.io':SimpleNamespace(),
                     'mne.io.constants':SimpleNamespace(FIFF=SimpleNamespace(FIFFV_COORD_HEAD=4,FIFFV_COORD_MRI=5))}
            with patch.dict(sys.modules,modules):
                row,bridge,_,_=stage.inspect_subject(root,subject,audited)
                self.assertEqual(row['status'],'PASS_HEADER_GEOMETRY_AND_RELEASED_CONTACTS',row['reason'])
                np.testing.assert_allclose(bridge['head_to_scanner_ras_mm'],np.eye(4),atol=1e-12)
                self.assertFalse(bridge['registration_point_mapping_validated'])
                image.affine=image.affine.copy();image.affine[0,3]+=1.
                row,bridge,_,_=stage.inspect_subject(root,subject,audited)
                self.assertEqual(row['status'],'FAIL');self.assertIsNone(bridge)

    def test_existing_output_refused_before_input_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys,'argv',[str(SCRIPT),'--output-root',tmp]), \
                 patch.object(stage,'inspect_subject',side_effect=AssertionError('must not inspect')):
                with self.assertRaises(FileExistsError):stage.main()


if __name__=='__main__':unittest.main()
