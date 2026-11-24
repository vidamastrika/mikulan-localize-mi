"""Generate an unapproved local inner-skull repair candidate; preserve originals."""
import hashlib
import json
from pathlib import Path
import shutil

import mne
import numpy as np
from scipy.spatial import cKDTree
from mne.surface import _project_onto_surface
from execution import validate_mesh


def main():
    exp = Path(__file__).resolve().parents[1]
    source = exp/'work/bem_candidates/atlas_mode'
    target = exp/'work/bem_candidates/local_inner_1mm'
    audit = exp/'outputs/03b_nesting_atlas_candidate/01_nesting.json'
    seeds = np.array(json.loads(audit.read_text())['pairs'][
        'inner_skull_inside_outer_skull']['outside_surface_ras_mm'])
    if seeds.shape != (9, 3):
        raise ValueError('Expected audited atlas candidate with nine outside vertices')
    rr, tris, metadata = mne.read_surface(source/'average_seven_inner_skull_surface', read_metadata=True)
    outer, faces = mne.read_surface(source/'average_seven_outer_skull_surface')
    original = rr.copy()
    distances = cKDTree(seeds).query(rr)[0]
    indices = np.flatnonzero(distances < 15)
    _, _, projected, normals = _project_onto_surface(
        rr[indices], dict(rr=outer, tris=faces), project_rrs=True, return_nn=True)
    signed = np.einsum('ij,ij->i', rr[indices]-projected, normals)
    # Full 1-mm clearance within 6 mm of an outside vertex, cosine taper to 15 mm.
    weight = .5*(1+np.cos(np.pi*np.clip((distances[indices]-6)/9, 0, 1)))
    move = np.maximum(signed+weight, 0)
    rr[indices] -= move[:, None]*normals
    displacement = np.linalg.norm(rr-original, axis=1)
    if displacement.max() > 3:
        raise ValueError('Repair exceeds 3-mm candidate displacement limit')
    validate_mesh(rr, tris)
    def face_normals(vertices):
        return np.cross(vertices[tris[:, 1]]-vertices[tris[:, 0]],
                        vertices[tris[:, 2]]-vertices[tris[:, 0]])
    if (np.einsum('ij,ij->i', face_normals(rr), face_normals(original)) <= 0).any():
        raise ValueError('Repair reverses a triangle normal')
    target.mkdir(exist_ok=False)
    hashes = {}
    for name in ['brain', 'inner_skull', 'outer_skull', 'outer_skin']:
        path = source/f'average_seven_{name}_surface'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = target/path.name
        if name == 'inner_skull':
            mne.write_surface(dest, rr, tris, volume_info=metadata, overwrite=False)
        else:
            shutil.copy2(path, dest)
    changed = np.flatnonzero(displacement > 1e-10)
    report = dict(anatomy_approved=False, input_sha256=hashes,
                  audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                  method='Nearest outer-skull triangle projection inward; 1 mm clearance; local cosine taper',
                  support_radius_mm=15, plateau_radius_mm=6, clearance_mm=1,
                  changed_vertices=changed.tolist(),
                  original_surface_ras_mm=original[changed].tolist(),
                  repaired_surface_ras_mm=rr[changed].tolist(),
                  displacement_mm=displacement[changed].tolist(),
                  max_displacement_mm=float(displacement.max()),
                  vertex_and_triangle_counts_unchanged=True,
                  review_required='MRI fit, nested mesh intersections, self intersections, cortical containment, BEM solve')
    (target/'repair.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(changed)} vertices moved; maximum {displacement.max():.3f} mm; candidate: {target}')


if __name__ == '__main__':
    main()
