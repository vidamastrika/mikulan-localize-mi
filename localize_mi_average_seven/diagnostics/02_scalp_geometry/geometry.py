"""Pure numerical geometry, in millimetres; no file writes or MNE imports."""
import numpy as np

def triangle_distance(points, vertices, faces):
    """Exact unsigned point-to-triangle minimum; all faces, bounded memory.

    Project onto triangle plane when inside, otherwise onto its three segments.
    Degenerate faces contribute their edges. Returned closest points are in mm.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    n2 = np.einsum('ij,ij->i', normal, normal)
    valid = n2 > 1e-20
    closest, distances = [], []
    for point in points:
        projection = point - normal * (np.einsum('ij,ij->i', point-a, normal)
                                        / np.where(valid, n2, 1))[:, None]
        ap = projection-a
        d00 = np.einsum('ij,ij->i', ab, ab)
        d01 = np.einsum('ij,ij->i', ab, ac)
        d11 = np.einsum('ij,ij->i', ac, ac)
        d20 = np.einsum('ij,ij->i', ap, ab)
        d21 = np.einsum('ij,ij->i', ap, ac)
        denom = np.where(valid, n2, 1)
        u = (d11*d20-d01*d21)/denom
        v = (d00*d21-d01*d20)/denom
        inside = valid & (u >= -1e-12) & (v >= -1e-12) & (u+v <= 1+1e-12)
        best_d = np.where(inside, np.sum((projection-point)**2, axis=1), np.inf)
        best_p = projection.copy()
        for start, end in [(a, b), (b, c), (c, a)]:
            edge = end-start
            e2 = np.einsum('ij,ij->i', edge, edge)
            t = np.clip(np.einsum('ij,ij->i', point-start, edge)/np.where(e2 > 0, e2, 1), 0, 1)
            q = start+t[:, None]*edge
            d = np.sum((q-point)**2, axis=1)
            mask = d < best_d
            best_p[mask], best_d[mask] = q[mask], d[mask]
        index = np.argmin(best_d)
        closest.append(best_p[index]); distances.append(np.sqrt(best_d[index]))
    return np.asarray(distances), np.asarray(closest)


def apply(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def topology(v, f):
    edges = np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    canonical = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    direction = np.where(edges[:,0] < edges[:,1], 1, -1)
    sums = np.bincount(inverse, weights=direction)
    cross = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
    area = np.linalg.norm(cross, axis=1)/2
    # Connected components among referenced vertices.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(len(v),len(v)))
    components, _ = connected_components(adjacency, directed=False)
    incident_faces=np.bincount(f.ravel(),minlength=len(v))
    result = dict(vertices=len(v), faces=len(f), vertices_with_fewer_than_three_faces=int(np.sum(incident_faces<3)), boundary_edges=int(np.sum(counts==1)),
        nonmanifold_edges=int(np.sum(counts>2)), inconsistent_oriented_edges=int(np.sum((counts==2)&(sums!=0))),
        degenerate_faces=int(np.sum(area<1e-10)), connected_components=int(components),
        unused_vertices=int(len(v)-len(np.unique(f))), euler=int(len(v)-len(unique)+len(f)),
        area_mm2=float(area.sum()), signed_volume_mm3=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6))
    result['sign_topology_eligible'] = bool(result['boundary_edges']==0 and result['nonmanifold_edges']==0
        and result['inconsistent_oriented_edges']==0 and result['degenerate_faces']==0
        and result['vertices_with_fewer_than_three_faces']==0 and components==1)
    result['self_intersection_certified'] = False
    return result


def winding(points, v, f):
    values=[]
    for p in points:
        a,b,c=(v[f[:,i]]-p for i in range(3))
        la,lb,lc=(np.linalg.norm(x,axis=1) for x in (a,b,c))
        num=np.einsum('ij,ij->i',a,np.cross(b,c))
        den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
        values.append(np.sum(2*np.arctan2(num,den))/(4*np.pi))
    return np.asarray(values)


def ray_inside(points, v, f, direction):
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    a,b,c=(v[f[:,i]] for i in range(3))
    edge1,edge2=b-a,c-a
    h=np.cross(np.broadcast_to(direction,edge2.shape),edge2)
    det=np.einsum('ij,ij->i',edge1,h)
    inv=np.where(np.abs(det)>1e-12,1/np.where(np.abs(det)>1e-12,det,1),0)
    states=[]; ambiguous=[]
    for p in points:
        s=p-a;u=inv*np.einsum('ij,ij->i',s,h)
        q=np.cross(s,edge1);vv=inv*(q@direction)
        t=inv*np.einsum('ij,ij->i',edge2,q)
        hit=(np.abs(det)>1e-12)&(u>=-1e-10)&(vv>=-1e-10)&(u+vv<=1+1e-10)&(t>1e-7)
        ts=np.sort(t[hit]); count=0 if not len(ts) else 1+np.sum(np.diff(ts)>1e-6)
        states.append(count%2==1)
        ambiguous.append(bool(np.any(hit&((np.abs(u)<1e-8)|(np.abs(vv)<1e-8)|(np.abs(1-u-vv)<1e-8)))))
    return np.asarray(states),np.asarray(ambiguous)


def signed_distance(points, v, f, distance, topo):
    """Positive outside, negative inside; conditional geometry, not anatomy.

    Requires closed consistent topology, absolute winding near 0/1 and agreement
    with three oblique rays; within 0.1 mm left unsigned. Self-intersections are
    not certified absent. No anatomical or alignment acceptance gate is changed.
    """
    w=winding(points,v,f)
    votes=[]; edge=[]
    for direction in [(1,.371,.173),(.213,1,.419),(.317,.227,1)]:
        s,a=ray_inside(points,v,f,direction);votes.append(s);edge.append(a)
    votes=np.array(votes);edge=np.array(edge)
    reliable=(topo['sign_topology_eligible'] & (distance>.1) & ~edge.any(0)
        & (np.max(votes,axis=0)==np.min(votes,axis=0))
        & ((np.abs(np.abs(w)-1)<1e-3)|(np.abs(w)<1e-3))
        & ((np.abs(w)>.5)==votes[0]))
    signed=np.where(reliable,np.where(votes[0],-distance,distance),np.nan)
    status=np.where(reliable,np.where(votes[0],'INSIDE_CONDITIONAL','OUTSIDE_CONDITIONAL'),'UNRESOLVED')
    return signed,status,w


def plane_segments(v, f, axis, position):
    """Exact triangle-plane intersections, in the other two coordinate axes."""
    a=v[f]; b=np.roll(a,-1,axis=1)
    cross=(a[:,:,axis]<position)!=(b[:,:,axis]<position)
    valid=cross.sum(1)==2
    a,b,cross=a[valid],b[valid],cross[valid]
    a,b=a[cross],b[cross]
    fraction=(position-a[:,axis])/(b[:,axis]-a[:,axis])
    points=a+(b-a)*fraction[:,None]
    remaining=[i for i in range(3) if i!=axis]
    return points[:,remaining].reshape(-1,2,2)


def region(points, center):
    p=np.asarray(points)-center
    az=np.arctan2(p[:,0],p[:,1])
    sectors=np.array(['posterior']*len(p),dtype=object)
    sectors[np.abs(az)<=np.pi/4]='anterior'
    sectors[(az>np.pi/4)&(az<3*np.pi/4)]='right'
    sectors[(az<-np.pi/4)&(az>-3*np.pi/4)]='left'
    return sectors
