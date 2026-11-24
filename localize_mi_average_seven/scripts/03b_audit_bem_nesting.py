"""Read-only surface audit and MRI views; never changes or approves anatomy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
from mne.surface import _CheckInside


def main():
    exp = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-dir', type=Path, default=exp/'work/freesurfer/average_seven')
    parser.add_argument('--output-root', type=Path, default=exp/'outputs/03b_nesting_audit')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    surfaces = {}
    for name in ('inner_skull', 'outer_skull', 'outer_skin'):
        rr, tris = mne.read_surface(args.subject_dir/'bem'/f'{name}.surf')
        surfaces[name] = dict(rr=rr, tris=tris)
    report = {'units': 'surface RAS mm', 'anatomy_approved': False, 'pairs': {}}
    failing = []
    for inner, outer in [('inner_skull', 'outer_skull'), ('outer_skull', 'outer_skin')]:
        rr = surfaces[inner]['rr']
        indices = np.flatnonzero(~_CheckInside(surfaces[outer])(rr))
        report['pairs'][f'{inner}_inside_{outer}'] = dict(
            vertices=len(rr), outside_vertices=len(indices),
            outside_indices=indices.tolist(), outside_surface_ras_mm=rr[indices].tolist())
        if len(indices):
            failing.append(rr[indices])
    (args.output_root/'01_nesting.json').write_text(json.dumps(report, indent=2)+'\n')
    if not failing:
        print('Vertex containment passes; anatomical review and full BEM validation still required.')
    bad = np.concatenate(failing) if failing else np.empty((0, 3))
    img = nib.load(args.subject_dir/'mri/T1.mgz')
    data = img.get_fdata()
    to_voxel = np.linalg.inv(img.header.get_vox2ras_tkr())
    voxels = {name: nib.affines.apply_affine(to_voxel, surf['rr']) for name, surf in surfaces.items()}
    bad_vox = nib.affines.apply_affine(to_voxel, bad)
    center = bad_vox.mean(axis=0) if len(bad) else nib.affines.apply_affine(to_voxel, [0, 60, 10])
    # FreeSurfer conformed voxel axes: sagittal=0, axial=1, coronal=2.
    for axis, label in [(0, 'sagittal'), (1, 'axial'), (2, 'coronal')]:
        remaining = [i for i in range(3) if i != axis]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, offset in zip(axes, [-4, 0, 4]):
            index = int(np.clip(round(center[axis])+offset, 0, data.shape[axis]-1))
            ax.imshow(np.take(data, index, axis=axis).T, origin='lower', cmap='gray')
            for (name, surf), color in zip(surfaces.items(), ['red', 'yellow', 'cyan']):
                v = voxels[name]
                ax.tricontour(v[:, remaining[0]], v[:, remaining[1]], surf['tris'],
                              v[:, axis], levels=[index], colors=[color], linewidths=.7)
            near = np.abs(bad_vox[:, axis]-index) < 2
            ax.scatter(bad_vox[near, remaining[0]], bad_vox[near, remaining[1]],
                       s=22, c='magenta', marker='x')
            ax.set_xlim(center[remaining[0]]-45, center[remaining[0]]+45)
            ax.set_ylim(center[remaining[1]]-45, center[remaining[1]]+45)
            ax.set_title(f'{label}, voxel {index}')
        fig.suptitle('Inner skull: red; outer skull: yellow; skin: cyan; failing vertices: magenta')
        fig.tight_layout()
        fig.savefig(args.output_root/f'02_{label}.png', dpi=150)
        plt.close(fig)
    print(f'Audit saved to {args.output_root}; no anatomy changed.')


if __name__ == '__main__':
    main()
