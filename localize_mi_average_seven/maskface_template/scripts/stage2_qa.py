"""Read-only physical MRI and uncorrected candidate surface QA."""
from pathlib import Path
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from input_utils import MRI,apply,region_labels

GEO=Path(__file__).resolve().parents[2]/'diagnostics/02_scalp_geometry'
sys.path.insert(0,str(GEO))
from geometry import topology,plane_segments,triangle_distance


def read_surface(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if str(path).endswith('.gii'):
            g=nib.load(path);p=next(a for a in g.darrays if a.intent==1008);f=next(a for a in g.darrays if a.intent==1009)
            v,faces,meta=np.array(p.data,float),np.array(f.data,int),dict(p.meta)
        else:v,faces,meta=nib.freesurfer.read_geometry(path,read_metadata=True)
    if not np.isfinite(v).all() or faces.min()<0 or faces.max()>=len(v):raise ValueError(f'Invalid surface {path}')
    return np.asarray(v,float),np.asarray(faces,int),meta,[str(w.message) for w in caught]


def fs_bridge(path):
    img=nib.load(path)
    return img.affine@np.linalg.inv(img.header.get_vox2ras_tkr())


def geometry_row(model,layer,v,f):
    return dict(model=model,layer=layer,**topology(v,f),width_mm=float(np.ptp(v[:,0])),ap_length_mm=float(np.ptp(v[:,1])),height_mm=float(np.ptp(v[:,2])))


def overlay_figure(entries,meshes,path,title,dpi=220,focus=(0,35,10)):
    """Each MRI entry uses an explicit common plotting-frame -> scanner bridge."""
    focus=np.asarray(focus,float);fig,axes=plt.subplots(3,len(entries),figsize=(7*len(entries),17),squeeze=False,layout='constrained')
    for row,(axis,name) in enumerate([(2,'axial'),(1,'coronal'),(0,'sagittal')]):
        rem=[i for i in range(3) if i!=axis];limits=[-125,125,-135,140]
        for col,(label,mri) in enumerate(entries):
            ax=axes[row,col];values,extent=mri.slice(axis,focus[axis],limits,.8)
            ax.imshow(values,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=mri.vmax)
            for name_mesh,v,f,color in meshes.get(label,[]):
                segments=plane_segments(v,f,axis,focus[axis])
                if len(segments):ax.add_collection(LineCollection(segments,colors=color,linewidths=.9,label=name_mesh))
            ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='XYZ'[rem[0]]+' (mm)',ylabel='XYZ'[rem[1]]+' (mm)',
                title=f'{label}: {name}, {"XYZ"[axis]}={focus[axis]:g} mm\nindependent positive-intensity p99.5 window')
            ax.set_aspect('equal')
            if row==0 and meshes.get(label):ax.legend(fontsize=7,loc='upper left')
    fig.suptitle(title,fontsize=14);fig.savefig(path,dpi=dpi);plt.close(fig)


def roi_metrics(entries):
    """Header common-grid cues only; not voxelwise anatomical correspondence."""
    rois={'facial_proxy':([-50,50],[40,100],[-60,10]),'left_lateral_proxy':([-90,-40],[-30,60],[-15,75]),
          'right_lateral_proxy':([40,90],[-30,60],[-15,75]),'brain_core_proxy':([-40,40],[-40,30],[25,75])}
    rows=[]
    for region,bounds in rois.items():
        points=np.array(np.meshgrid(*[np.arange(lo,hi+1,3.) for lo,hi in bounds],indexing='ij')).reshape(3,-1).T
        reference=None
        for label,mri in entries:
            values,valid=mri.sample(points);normalized=values/max(mri.vmax,1e-12)
            okay=valid&np.isfinite(values);signal=okay&(normalized>.05)
            row=dict(image=label,region_proxy=region,n=len(points),in_fov_fraction=float(okay.mean()),
                signal_fraction_p995_05=float(signal.mean()),mean_normalized_intensity=float(np.mean(normalized[okay])) if okay.any() else None,
                interpretation='fixed original surface-RAS grid boxes; no intertemplate anatomical registration; intensity support is not tissue segmentation')
            if reference is None:reference=(normalized,okay)
            else:
                a,av=reference;common=av&okay&(a>.05)&(normalized>.05)
                row['common_signal_n']=int(common.sum())
                row['common_signal_correlation']=float(np.corrcoef(a[common],normalized[common])[0,1]) if common.sum()>10 and np.std(a[common])>0 and np.std(normalized[common])>0 else None
            rows.append(row)
    return rows


def warp_metrics(path):
    img=nib.load(path);u=np.asarray(img.dataobj,dtype=np.float32).squeeze()[::4,::4,::4].astype(float)
    if u.ndim!=4 or u.shape[-1]!=3:raise ValueError(f'Unrecognized ANTs displacement layout: {path}')
    u*=np.array([-1,-1,1]) # LPS displacement -> RAS; nibabel exposes affine in RAS.
    jac=np.stack([np.stack(np.gradient(u[...,c],axis=(0,1,2)),axis=-1) for c in range(3)],axis=-2)
    jac=jac@np.linalg.inv(img.affine[:3,:3]*4)+np.eye(3)
    det=np.linalg.det(jac[1:-1,1:-1,1:-1]);norm=np.linalg.norm(u,axis=-1)
    return dict(path=str(path),sample_stride=4,jacobian_min=float(det.min()),nonpositive_samples=int((det<=0).sum()),
        displacement_p99_mm=float(np.percentile(norm,99)),interpretation='coarse displacement-field-only samples, not full composite or full-resolution fold certification')


def vtk_poly(v,f):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.ascontiguousarray(v),deep=True))
    cells=vtk.vtkCellArray()
    for triangle in f:cells.InsertNextCell(3,list(map(int,triangle)))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    return poly


def contains(points,v,f,eligible):
    """Conditional watertight-surface ray test; do not classify invalid topology."""
    if not eligible:return None
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
    cloud=vtk.vtkPolyData();pp=vtk.vtkPoints();pp.SetData(numpy_to_vtk(np.ascontiguousarray(points),deep=True));cloud.SetPoints(pp)
    enclosed=vtk.vtkSelectEnclosedPoints();enclosed.SetInputData(cloud);enclosed.SetSurfaceData(vtk_poly(v,f));enclosed.SetTolerance(1e-6);enclosed.Update()
    return vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray('SelectedPoints')).astype(bool)


def intersections(v,f,w=None,g=None):
    """Full AABB candidates + VTK triangle test; shared-vertex self pairs excluded.

    Contact/coplanar behavior is implementation/tolerance dependent. This is a
    numerical screen, never certification of absent anatomical intersections.
    """
    import vtk
    self_test=w is None
    if self_test:w,g=v,f
    poly=vtk_poly(w,g);locator=vtk.vtkStaticCellLocator();locator.SetDataSet(poly);locator.BuildLocator()
    ids=vtk.vtkIdList();hits=[];count=0;tested=0
    for i,face in enumerate(f):
        t=v[face];lo=t.min(0)-1e-7;hi=t.max(0)+1e-7
        locator.FindCellsWithinBounds([lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]],ids)
        for k in range(ids.GetNumberOfIds()):
            j=ids.GetId(k)
            if self_test and (j<=i or np.intersect1d(face,g[j]).size):continue
            tested+=1
            if vtk.vtkTriangle.TrianglesIntersect(*t,*w[g[j]]):
                count+=1
                if len(hits)<100:hits.append([int(i),int(j)])
    return dict(candidate_pairs_tested=tested,intersection_pairs=count,first_pairs=hits,
        interpretation='AABB+VTK triangle contacts; self pairs sharing vertices excluded; no full geometric/anatomical certification')


def distance_stats(values):
    return dict(n=len(values),min_mm=float(np.min(values)),median_mm=float(np.median(values)),p95_mm=float(np.percentile(values,95)),max_mm=float(np.max(values)))
