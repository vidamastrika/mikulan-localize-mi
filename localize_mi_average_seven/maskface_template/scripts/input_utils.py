"""Read-only MRI utilities in physical RAS mm, derived from tested diagnostic code."""
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

def apply(matrix, points):
    return np.asarray(points) @ matrix[:3,:3].T + matrix[:3,3]

def native_bridge(meta):
    """Reconstruct FS volume geometry; scannerRAS @ inverse(tkregisterRAS)."""
    size=np.array([int(meta['VolGeom'+k]) for k in ['Width','Height','Depth']])
    zoom=np.array([float(meta['VolGeom'+k+'size']) for k in 'XYZ'])
    directions=np.array([[float(meta[f'VolGeom{a}_{r}']) for a in 'XYZ'] for r in 'RAS'])
    center=np.array([float(meta[f'VolGeomC_{r}']) for r in 'RAS'])
    scanner=np.eye(4);scanner[:3,:3]=directions@np.diag(zoom)
    scanner[:3,3]=center-scanner[:3,:3]@(size/2)
    tkr=np.array([[-zoom[0],0,0,zoom[0]*size[0]/2],
                  [0,0,zoom[2],-zoom[2]*size[2]/2],
                  [0,-zoom[1],0,zoom[1]*size[1]/2],[0,0,0,1.]])
    return scanner@np.linalg.inv(tkr),scanner,size

class MRI:
    def __init__(self, path, bridge):
        self.path=path;self.image=nib.load(path)
        self.data=np.asarray(self.image.dataobj,dtype=np.float32)
        if self.data.ndim!=3:raise ValueError(f'Expected scalar MRI: {path}')
        self.bridge=np.asarray(bridge,float)
        self.to_voxel=np.linalg.inv(self.image.affine)@self.bridge
        positive=self.data[np.isfinite(self.data)&(self.data>0)]
        self.vmax=float(np.percentile(positive,99.5)) if len(positive) else 1
    def sample(self, points):
        points=np.asarray(points);vox=apply(self.to_voxel,points.reshape(-1,3))
        valid=((vox>=0)&(vox<=np.array(self.data.shape)-1)).all(1)
        values=map_coordinates(self.data,vox.T,order=1,mode='constant',cval=np.nan,prefilter=False)
        return values.reshape(points.shape[:-1]),valid.reshape(points.shape[:-1])
    def slice(self,axis,position,limits,resolution=1.):
        remaining=[i for i in range(3) if i!=axis]
        x=np.arange(limits[0],limits[1]+resolution/2,resolution)
        y=np.arange(limits[2],limits[3]+resolution/2,resolution)
        xx,yy=np.meshgrid(x,y);p=np.zeros((*xx.shape,3))
        p[:,:,axis]=position;p[:,:,remaining[0]]=xx;p[:,:,remaining[1]]=yy
        values,valid=self.sample(p)
        return values,[x[0],x[-1],y[0],y[-1]]


def compare_samples(py,mask,valid_py,valid_mask,scale):
    """Raw-intensity comparisons; no input intensity/geometry modification."""
    valid=np.asarray(valid_py)&np.asarray(valid_mask)&np.isfinite(py)&np.isfinite(mask)
    a=np.asarray(py)[valid];b=np.asarray(mask)[valid]
    if not len(a):return dict(n_valid=0,exact_equal_fraction=None,normalized_absdiff_p95=None,pearson=None)
    corr=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None
    return dict(n_valid=len(a),exact_equal_fraction=float(np.mean(a==b)),
        normalized_absdiff_p95=float(np.percentile(np.abs(a-b),95)/max(scale,1e-12)),pearson=corr)


def support_states(py,mask,py_scale,mask_scale,fraction=.05):
    """Signal-support states, explicitly not tissue truth or an anonymization mask."""
    a=np.asarray(py)>py_scale*fraction;b=np.asarray(mask)>mask_scale*fraction
    states=np.full(a.shape,'BACKGROUND_OR_LOW_SIGNAL',dtype=object)
    states[a&b]='COMMON_SIGNAL'
    states[~a&b]='MASKFACE_ONLY_SIGNAL_UNVERIFIED'
    states[a&~b]='PYDEFACE_ONLY_SIGNAL'
    return states


def region_labels(points):
    """Fixed surface-RAS geometric regions; not image-derived tissue labels."""
    x,y,z=np.asarray(points).T
    labels=np.full(len(x),'other_or_transition',dtype=object)
    labels[z>=15]='cranial_proxy'
    labels[(z<15)&(np.abs(x)>=40)]='inferior_lateral_proxy'
    labels[(z<15)&(y>=40)&(y>=.65*np.abs(x))]='facial_proxy'
    return labels


def image_header(image):
    shape=np.array(image.shape[:3]);affine=np.asarray(image.affine,float)
    finite=bool(np.isfinite(affine).all());invertible=finite and abs(np.linalg.det(affine[:3,:3]))>1e-10
    zooms=np.array(image.header.get_zooms()[:3],float)
    units=image.header.get_xyzt_units()[0]
    q,qc=image.get_qform(coded=True);s,sc=image.get_sform(coded=True)
    corners=apply(affine,np.array([[x,y,z] for x in [-.5,shape[0]-.5] for y in [-.5,shape[1]-.5] for z in [-.5,shape[2]-.5]]))
    return dict(shape=shape.tolist(),voxel_spacing_mm=zooms.tolist(),axis_codes=list(nib.aff2axcodes(affine)),units=units,
        affine_ras_mm=affine.tolist(),qform_code=int(qc),sform_code=int(sc),
        qform_sform_agree=None if q is None or s is None else bool(np.allclose(q,s,atol=1e-3)),
        voxel_FOV_mm=(shape*zooms).tolist(),scanner_ras_min_mm=corners.min(0).tolist(),scanner_ras_max_mm=corners.max(0).tolist(),
        technical_header_valid=bool(len(image.shape)==3 and units=='mm' and invertible and (zooms>0).all()
            and (qc>0 or sc>0) and not(q is not None and s is not None and not np.allclose(q,s,atol=1e-3))))
