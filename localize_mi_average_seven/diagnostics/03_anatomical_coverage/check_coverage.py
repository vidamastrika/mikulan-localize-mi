"""Synthetic tests only; no participant data, MNE, or pipeline execution."""
import ast
from pathlib import Path
import numpy as np
from coverage_math import frame,region_proxy,patch_points,support,summary,mirrored_pairs

for direction in [[1.,2.,3.],[0.,0.,1.],[1.,0.,0.],[0.,-1.,0.]]:
    b=frame(direction)
    np.testing.assert_allclose(b@b.T,np.eye(3),atol=1e-12)
    assert np.linalg.det(b)>0
    q=np.array([-65.,35.,-20.]);p=q+np.array(direction)
    # Both endpoints lie on BOTH declared residual planes; no projection error.
    for bb in [b,np.array([b[0],b[2],-b[1]])]:
        np.testing.assert_allclose(((p-q)@bb.T)[2],0,atol=1e-12)
        np.testing.assert_allclose(bb@bb.T,np.eye(3),atol=1e-12)
try:frame([0,0,0])
except ValueError:pass
else:raise AssertionError('Zero residual direction accepted')

points=np.array([[0.,70,-20],[-65,0,-20],[65,0,-20],[-60,0,40],[0,-60,-20]])
assert list(region_proxy(points))==['facial_proxy','inferior_lateral_proxy','inferior_lateral_proxy','cranial_proxy','transition_or_other_uncertain']
patch=patch_points(points,np.array([0.,0.,0.]))
assert patch.shape==(5,27,3)
for p,probe in zip(points,patch):
    radial=p/np.linalg.norm(p)
    np.testing.assert_allclose(np.sort(np.unique(np.round((p-probe)@radial,8))),[4,8,12])
values=np.array([[100.]*27,[0.]*27,[100.]*14+[0.]*13,[100.]*27])
valid=np.ones_like(values,dtype=bool);valid[-1,0]=False
labels,fractions=support(values,valid,100.)
assert list(labels)==['MRI_SUPPORTED_PROXY','LOW_MRI_SUPPORT_PROXY','UNCERTAIN','OUT_OF_FOV_OR_NONFINITE']
values[0,0]=np.nan
assert support(values,valid,100.)[0][0]=='OUT_OF_FOV_OR_NONFINITE'
assert summary([])['n']==0
np.testing.assert_allclose(summary([1.,2.,3.])['p95_mm'],2.9)
try:summary([np.nan])
except ValueError:pass
else:raise AssertionError('Nonfinite residual accepted')
q=np.array([[-60.,0,40],[60.,0,40],[90,0,40]])
pairs=mirrored_pairs(q,np.array(['cranial_proxy']*3),np.array(['MRI_SUPPORTED_PROXY']*3))
assert pairs==[(0,1,0.)]
# Coverages and geometric labels depend on no EEG outcome input.
assert list(support(values,valid,100.)[0])==list(support(values.copy(),valid.copy(),100.)[0])
for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text())
print('PASS: anatomy proxies, MRI support ambiguity, inward patch geometry, residual plane coplanarity, mirror pairing, summary edge cases and syntax. No dataset analysis executed.')

# Exercise the plotting path with a synthetic MRI, isolated from MNE/dataset.
import importlib.util
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
previous=Path(__file__).resolve().parents[1]/'02_scalp_geometry/geometry.py'
spec=importlib.util.spec_from_file_location('tested_surface_geometry',previous)
geom=importlib.util.module_from_spec(spec);spec.loader.exec_module(geom)
tree=ast.parse(Path(__file__).with_name('diagnose.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['plane_panel','residual_figure','stage_figure']]
ns={'np':np,'plt':plt,'LineCollection':LineCollection,'frame':frame,'plane_segments':geom.plane_segments}
exec(compile(ast.Module(body=functions,type_ignores=[]),'isolated_plots','exec'),ns)
class SyntheticMRI:
    vmax=1.
    def sample(self,points):
        values=np.exp(-np.sum(points**2,axis=-1)/500.)
        return values,np.ones_like(values,dtype=bool)
v=np.array([[0.,0,0],[30.,0,0],[0,30.,0],[0,0,30.]])
f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
q=np.array([30.,0,0]);p=np.array([60.,10,5.])
with tempfile.TemporaryDirectory(prefix='coverage-plane-check-') as temp:
    path=Path(temp)/'residual.png'
    records=ns['residual_figure']('synthetic','synthetic','test','e1',p,q,SyntheticMRI(),[('test surface',v,f,'cyan')],path,50)
    assert path.is_file()
    assert all(max(abs(r['electrode_off_plane_mm']),abs(r['scalp_off_plane_mm']))<1e-8 for r in records[:2])
    assert all(r['positive_intensity_fraction']>0 for r in records)
print('PASS: synthetic MRI/mesh rendering and recorded residual endpoint coplanarity; no dataset analysis executed.')
