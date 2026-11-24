"""Synthetic tests of intensity audit; no participant images."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
from input_utils import MRI

p=Path(__file__).with_name('01b_review_intensity_conventions.py')
spec=importlib.util.spec_from_file_location('intensity_supplement_test',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
vertices=np.repeat(np.arange(64),2)
b=np.linspace(.1,.8,len(vertices));a=255*b+.2
k,c,tr,te=m.calibration(a,b,vertices)
np.testing.assert_allclose([k,c],[255,.2],atol=1e-10)
a_heldout=a.copy();a_heldout[te]+=10
kk,cc,_,_=m.calibration(a_heldout,b,vertices)
np.testing.assert_allclose([kk,cc],[k,c],atol=1e-10)
assert tr.sum()==64 and te.sum()==64
with tempfile.TemporaryDirectory(prefix='maskface-intensity-check-') as folder:
    data=np.arange(8**3,dtype=np.float32).reshape(8,8,8)/1000
    img=nib.Nifti1Image(data,np.eye(4));path=Path(folder)/'mask.nii.gz';nib.save(img,path)
    mask=MRI(path,np.eye(4));raw=mask.data.copy();view=m.IntensityView(mask,255,.2)
    x,extent=mask.slice(2,3,[0,7,0,7]);y,_=view.slice(2,3,[0,7,0,7])
    np.testing.assert_allclose(y,x*255+.2);np.testing.assert_array_equal(mask.data,raw)
    py_path=Path(folder)/'py.nii.gz';nib.save(nib.Nifti1Image(data*255+.2,np.eye(4)),py_path)
    py=MRI(py_path,np.eye(4))
    rows=m.adjusted_regions(py,mask,np.eye(4),130,.51,255,.2)
    values=[r['adjusted_common_absdiff_p95_fraction_py_p99'] for r in rows if r['adjusted_common_absdiff_p95_fraction_py_p99'] is not None]
    assert values and max(values)<1e-6
    # Exercise corrected shared-window plotting without touching source data.
    v=np.array([[0.,0,0],[7,0,0],[0,7,0],[0,0,7]]);f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    plot=Path(folder)/'corrected.png'
    m.base.compare_figure('synthetic',py,view,[('surface',v,f,'cyan')],np.array([3.,3.,3.]),plot,50,'intensity-corrected test')
    assert plot.is_file();np.testing.assert_array_equal(mask.data,raw)
print('PASS: known gain/offset, held-out vertex independence, immutable image adapter, adjusted regional metrics and corrected rendering. No participant analysis executed.')
