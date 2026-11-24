"""Synthetic geometry checks only; never reads the participant dataset or MNE."""
import ast
from pathlib import Path
import numpy as np
from geometry import triangle_distance, topology, signed_distance, plane_segments, apply

# Closed tetrahedron with consistent outward winding.
v=np.array([[0.,0,0],[10,0,0],[0,10,0],[0,0,10]])
f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
p=np.array([[1.,1,1],[12,1,1],[0,0,0]])
t=topology(v,f)
assert t['sign_topology_eligible'] and t['euler']==2
assert not topology(v,f[:-1])['sign_topology_eligible']
broken=f.copy();broken[0]=broken[0,::-1]
assert not topology(v,broken)['sign_topology_eligible']
d,q=triangle_distance(p,v,f)
s,state,w=signed_distance(p,v,f,d,t)
assert list(state)==['INSIDE_CONDITIONAL','OUTSIDE_CONDITIONAL','UNRESOLVED'],state
assert s[0]<0 and s[1]>0 and np.isnan(s[2])
# Reversing all faces does not reverse geometric inside/outside.
s2,state2,w2=signed_distance(p,v,f[:,::-1],d,topology(v,f[:,::-1]))
np.testing.assert_allclose(s,s2,equal_nan=True)
sopen,stopen,_=signed_distance(p,v,f[:-1],d,topology(v,f[:-1]))
assert np.isnan(sopen).all()
segments=plane_segments(v,f,2,2.)
np.testing.assert_allclose(segments.reshape(-1,2).min(0),[0,0])
np.testing.assert_allclose(segments.reshape(-1,2).max(0),[8,8])
# Interior, edge, vertex and degenerate triangle closest-point cases.
tri=np.array([[0,1,2]])
a=np.array([[0.,0,0],[1,0,0],[0,1,0]])
d,q=triangle_distance(np.array([[.2,.2,2],[2,0,0],[-1,-1,0]]),a,tri)
np.testing.assert_allclose(d,[2,1,np.sqrt(2)])
d,q=triangle_distance(np.array([[.5,1,0]]),np.array([[0.,0,0],[1,0,0],[2,0,0]]),tri)
np.testing.assert_allclose(d,[1])
# Extract bridge function without importing the dataset runner or MNE.
script=Path(__file__).with_name('diagnose.py')
tree=ast.parse(script.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_bridge')
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(script),'exec'),ns)
meta={**{f'VolGeom{k}':'256' for k in ['Width','Height','Depth']},
      **{f'VolGeom{k}size':'1' for k in 'XYZ'}}
axes=np.array([[-1,0,0],[0,0,1],[0,-1,0]])
for j,a in enumerate('XYZ'):
    for i,r in enumerate('RAS'):meta[f'VolGeom{a}_{r}']=str(axes[i,j])
for r,value in zip('RAS',[2,3,4]):meta[f'VolGeomC_{r}']=str(value)
bridge,scanner,size=ns['native_bridge'](meta)
np.testing.assert_allclose(bridge[:3,:3],np.eye(3))
np.testing.assert_allclose(bridge[:3,3],[2,3,4])
np.testing.assert_allclose(apply(scanner,np.array([[128,128,128]])),[[2,3,4]])
# Syntax of every new Python file.
for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text())
print('PASS: closest points, closed/open topology, orientation, winding/rays, unsigned ambiguity, plane intersections, volume bridge and syntax. No dataset analysis executed.')

# Physical MRI sampling with anisotropic voxels and a nonzero surface bridge.
# Temporary synthetic volume only; no participant data is accessed.
import tempfile
import nibabel as nib
from scipy.ndimage import map_coordinates
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MRI')
ns={'np':np,'nib':nib,'apply':apply,'map_coordinates':map_coordinates}
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(script),'exec'),ns)
with tempfile.TemporaryDirectory(prefix='scalp-geometry-test-') as temp:
    indices=np.indices((9,9,9))
    data=(indices[0]+10*indices[1]+100*indices[2]).astype(np.float32)
    affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[10,-20,30]
    path=Path(temp)/'synthetic.nii.gz';nib.save(nib.Nifti1Image(data,affine),path)
    bridge=np.eye(4);bridge[:3,3]=[5,6,7]
    mri=ns['MRI'](path,bridge)
    voxel=np.array([[2.25,3.5,4.75]])
    surface=apply(np.linalg.inv(bridge),apply(affine,voxel))
    values,valid=mri.sample(surface)
    np.testing.assert_allclose(values,[2.25+35+475],atol=1e-4)
    assert valid.all()
    values,valid=mri.sample(np.array([[10000,0,0]]))
    assert not valid.any() and np.isnan(values).all()
print('PASS: physical MRI sampling, anisotropic voxel dimensions, nonzero bridge, and out-of-FOV handling.')

# Regression: the diagnostic reader retains FIF topology warnings, without
# accepting the defective mesh for signed classification or changing arrays.
import warnings as python_warnings
from types import SimpleNamespace
reader=next(n for n in ast.parse(script.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='load_mesh')
calls=[]
def fake_read(path,**kwargs):
    calls.append(kwargs)
    assert kwargs['on_defects']=='warn'
    python_warnings.warn('Surface outer skin has topological defects',RuntimeWarning)
    return [dict(id=4,coord_frame=5,rr=v*.01,tris=f)]
ns={'np':np,'mne':SimpleNamespace(read_bem_surfaces=fake_read),'python_warnings':python_warnings}
exec(compile(ast.Module(body=[reader],type_ignores=[]),str(script),'exec'),ns)
rv,rf,meta=ns['load_mesh'](Path('synthetic-head.fif'))
np.testing.assert_array_equal(rv,v*10)
np.testing.assert_array_equal(rf,f)
assert len(meta['reader_warnings'])==1 and 'topological defects' in meta['reader_warnings'][0]
# Unreferenced and low-incidence vertices must block geometric sign eligibility.
assert not topology(np.vstack([v,[[50,50,50]]]),f)['sign_topology_eligible']
double_triangle=np.array([[0,1,2],[2,1,0]])
tlow=topology(v[:3],double_triangle)
assert tlow['vertices_with_fewer_than_three_faces']==3 and not tlow['sign_topology_eligible']
print('PASS: defective FIF warning ingestion preserves geometry; low-incidence topology withholds sign eligibility.')
