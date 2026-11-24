"""Synthetic Stage 1 checks; no participant data or reconstruction."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
from input_utils import compare_samples,support_states,region_labels,image_header,MRI

values=np.array([0.,10.,20.,30.]);valid=np.ones(4,bool)
m=compare_samples(values,values,valid,valid,30.)
assert m['exact_equal_fraction']==1 and m['normalized_absdiff_p95']==0
assert compare_samples(values,values,np.zeros(4,bool),valid,30.)['n_valid']==0
states=support_states([0,20,20,0],[20,0,20,0],20,20)
assert list(states)==['MASKFACE_ONLY_SIGNAL_UNVERIFIED','PYDEFACE_ONLY_SIGNAL','COMMON_SIGNAL','BACKGROUND_OR_LOW_SIGNAL']
assert list(region_labels([[0,60,-10],[-60,0,-10],[0,0,50],[0,-40,-10]]))==['facial_proxy','inferior_lateral_proxy','cranial_proxy','other_or_transition']
with tempfile.TemporaryDirectory(prefix='maskface-stage1-test-') as folder:
    data=np.arange(8**3,dtype=np.float32).reshape(8,8,8)
    affine=np.diag([1.,2.,3.,1.]);image=nib.Nifti1Image(data,affine)
    image.header.set_xyzt_units('mm');image.set_qform(affine,code=1);image.set_sform(affine,code=1)
    assert image_header(image)['technical_header_valid']
    changed=affine.copy();changed[0,3]=3;image.set_sform(changed,code=1)
    assert not image_header(image)['technical_header_valid']
    image.set_sform(affine,code=1)
    path=Path(folder)/'synthetic.nii.gz';nib.save(image,path)
    mri=MRI(path,np.eye(4));sample,inside=mri.sample(np.array([[2.,6.,12.]]))
    np.testing.assert_allclose(sample,[data[2,3,4]]);assert inside.all()
    # Import runner without main; no MNE import or data loading occurs.
    script=Path(__file__).with_name('01_audit_inputs.py')
    spec=importlib.util.spec_from_file_location('maskface_stage1_test',script)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    rows=runner.paired_sample_table(mri,mri,np.eye(4),500,500,stride=2)
    assert all(r['n_samples']==0 for r in rows if r['state']=='MASKFACE_ONLY_SIGNAL_UNVERIFIED')
    assert all(r['common_signal_normalized_absdiff_p95'] in [None,0] for r in rows)
    v=np.array([[0.,0,0],[6,0,0],[0,12,0],[0,0,18]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    plot=Path(folder)/'synthetic_pair.png'
    runner.compare_figure('synthetic',mri,mri,[('test reference',v,f,'cyan')],np.array([3.,6.,9.]),plot,50,'synthetic validation')
    assert plot.is_file()
for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
print('PASS: header/affine ambiguity, anisotropic physical sampling, signal states, paired-grid metrics, region rules, module import, paired MRI plotting and syntax. No participant analysis executed.')
