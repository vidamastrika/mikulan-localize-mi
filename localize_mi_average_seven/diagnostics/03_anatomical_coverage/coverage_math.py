"""Coverage proxies and physical plane geometry; no outcome-dependent selection."""
import numpy as np


def region_proxy(points, face_y=40., inferior_z=15.):
    """Unregistered surface RAS mm; broad geometric proxies, not net labels."""
    points=np.asarray(points)
    x,y,z=points.T
    labels=np.full(len(points),'transition_or_other_uncertain',dtype=object)
    labels[z>=inferior_z]='cranial_proxy'
    inferior=z<inferior_z
    labels[inferior&(np.abs(x)>=40)]='inferior_lateral_proxy'
    labels[inferior&(y>=face_y)&(y>=.65*np.abs(x))]='facial_proxy'
    return labels


def frame(direction):
    """Rows define residual axis and two mutually orthogonal transverse axes."""
    direction=np.asarray(direction,float)
    norm=np.linalg.norm(direction)
    if norm<1e-8:raise ValueError('Residual direction is undefined')
    u=direction/norm
    trial=np.array([0.,0,1.]) if abs(u[2])<.9 else np.array([0.,1.,0])
    v=trial-u*np.dot(trial,u);v/=np.linalg.norm(v)
    w=np.cross(u,v)
    return np.array([u,v,w])


def patch_points(points, center):
    """27 inward probes: 3 depths x 3x3 tangent patch, mm.

    Center is native scalp vertex median; radial is a proxy for the inward
    direction. MRI support at these probes is not a verified surface boundary.
    """
    points=np.asarray(points,float);output=[]
    for p in points:
        r=p-center
        if np.linalg.norm(r)<1e-8:raise ValueError('Radial patch direction undefined')
        axes=frame(r)
        output.append(np.array([p-depth*axes[0]+a*axes[1]+b*axes[2]
            for depth in [4.,8.,12.] for a in [-4.,0.,4.] for b in [-4.,0.,4.]]))
    return np.asarray(output)


def support(values, valid, positive_p99):
    """Coverage labels from fixed MRI intensity cutoffs, never EEG residuals.

    Conservative consensus across .01/.05/.10 of positive p99; >=80% probe
    support is MRI-supported, <=20% low-support, otherwise uncertain. All 27
    probes must be in FOV and finite to classify. These are review/sensitivity
    rules, not alignment criteria or a validated tissue segmentation.
    """
    values=np.asarray(values);valid=np.asarray(valid)&np.isfinite(values)
    fractions=np.array([(valid&(values>fraction*positive_p99)).mean(axis=1) for fraction in [.01,.05,.10]]).T
    reliable=valid.all(axis=1)&(positive_p99>0)
    labels=np.full(len(values),'UNCERTAIN',dtype=object)
    labels[reliable&(fractions.min(1)>=.8)]='MRI_SUPPORTED_PROXY'
    labels[reliable&(fractions.max(1)<=.2)]='LOW_MRI_SUPPORT_PROXY'
    labels[~valid.all(1)]='OUT_OF_FOV_OR_NONFINITE'
    return labels,fractions


def summary(distances):
    d=np.asarray(distances,float)
    if not len(d):return dict(n=0,median_mm=None,p95_mm=None,max_mm=None,mean_mm=None)
    if not np.isfinite(d).all():raise ValueError('Nonfinite residual')
    return dict(n=len(d),median_mm=float(np.median(d)),p95_mm=float(np.percentile(d,95)),
                max_mm=float(d.max()),mean_mm=float(d.mean()))


def mirrored_pairs(points, labels, coverage):
    """Descriptive nearest right q for mirrored left q, no fitted transform.

    Pairing depends only on native q positions/coverage, not residuals.
    It is not verified physical net correspondence, and may reuse right points.
    """
    from scipy.spatial import cKDTree
    points=np.asarray(points);coverage=np.asarray(coverage);labels=np.asarray(labels)
    left=np.flatnonzero((points[:,0]<-40)&(labels=='cranial_proxy')&(coverage=='MRI_SUPPORTED_PROXY'))
    right=np.flatnonzero((points[:,0]>40)&(labels=='cranial_proxy')&(coverage=='MRI_SUPPORTED_PROXY'))
    if not len(left) or not len(right):return []
    mirrored=points[left].copy();mirrored[:,0]*=-1
    d,idx=cKDTree(points[right]).query(mirrored)
    return [(int(i),int(right[j]),float(distance)) for i,j,distance in zip(left,idx,d) if distance<=15]
