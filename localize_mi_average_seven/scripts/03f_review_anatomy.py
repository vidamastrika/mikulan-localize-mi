"""Compare atlas and repaired anatomy on MRI; interactive or static export.

No files are adopted and no review status is changed by this viewer.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import mne


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, help='Save both review regions without opening a GUI')
    args = parser.parse_args()
    import matplotlib
    if args.export:
        matplotlib.use('Agg')
    else:
        matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import RadioButtons, Slider

    exp = Path(__file__).resolve().parents[1]
    original = exp/'work/freesurfer/average_seven'
    candidates = exp/'work/bem_candidates'
    image = nib.load(original/'mri/T1.mgz')
    volume = image.get_fdata()
    inverse = np.linalg.inv(image.header.get_vox2ras_tkr())
    regions = {
        'Repair': nib.affines.apply_affine(inverse, [2, 60, 10]),
        'Pial outlier': nib.affines.apply_affine(inverse, [38.47557, 25.83610, -.05174]),
        'Whole head': np.array(volume.shape)/2,
    }
    meshes = []
    for folder in ['atlas_mode', 'local_inner_1mm']:
        row = []
        for name, color in [('inner_skull', 'red'), ('outer_skull', 'yellow'), ('outer_skin', 'cyan')]:
            rr, tris = mne.read_surface(candidates/folder/f'average_seven_{name}_surface')
            row.append((nib.affines.apply_affine(inverse, rr)[tris], color))
        for hemi in ['lh', 'rh']:
            rr, tris = mne.read_surface(original/'surf'/f'{hemi}.pial')
            row.append((nib.affines.apply_affine(inverse, rr)[tris], 'lime'))
        meshes.append(row)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(bottom=.24, top=.88, hspace=.24)
    fig.suptitle('Atlas candidate (top) / local repair (bottom) — anatomy review pending\n'
                 'Inner skull: red | outer skull: yellow | skin: cyan | pial: green')
    labels = ['Sagittal', 'Axial', 'Coronal']
    sliders = []
    center = regions['Repair'].copy()
    zoom = 40
    vmax = float(np.percentile(volume[volume > 0], 99.5))

    def segments(triangles, axis, position, remaining):
        coord = triangles[:, :, axis]
        hit = (coord.min(1) <= position) & (coord.max(1) >= position)
        a = triangles[hit]
        b = np.roll(a, -1, axis=1)
        cross = (a[:, :, axis] < position) != (b[:, :, axis] < position)
        # A generic triangle-plane crossing has two edges, in face-major order.
        valid = cross.sum(1) == 2
        a, b, cross = a[valid], b[valid], cross[valid]
        a, b = a[cross], b[cross]
        fraction = (position-a[:, axis])/(b[:, axis]-a[:, axis])
        points = a+(b-a)*fraction[:, None]
        return points[:, remaining].reshape(-1, 2, 2)

    def redraw(_=None):
        for axis in range(3):
            index = int(sliders[axis].val)
            remaining = [i for i in range(3) if i != axis]
            for row in range(2):
                ax = axes[row, axis]
                ax.clear()
                ax.imshow(np.take(volume, index, axis=axis).T, origin='lower',
                          cmap='gray', vmin=0, vmax=vmax)
                for triangles, color in meshes[row]:
                    ax.add_collection(LineCollection(segments(triangles, axis, index, remaining),
                                                     colors=color, linewidths=.7))
                marker = regions['Pial outlier']
                if abs(marker[axis]-index) < 1:
                    ax.scatter(marker[remaining[0]], marker[remaining[1]],
                               c='magenta', marker='x', s=45)
                ax.set_xlim(center[remaining[0]]-zoom, center[remaining[0]]+zoom)
                ax.set_ylim(center[remaining[1]]-zoom, center[remaining[1]]+zoom)
                ax.set_title(f'{labels[axis]}, voxel {index}')
                ax.set_aspect('equal')
        fig.canvas.draw_idle()

    for axis in range(3):
        slider = Slider(fig.add_axes([.25, .14-axis*.045, .68, .025]), labels[axis],
                        0, volume.shape[axis]-1, valinit=round(center[axis]), valstep=1)
        slider.on_changed(redraw)
        sliders.append(slider)
    radio = RadioButtons(fig.add_axes([.015, .025, .15, .15]), list(regions))

    def preset(label):
        nonlocal center, zoom
        center = regions[label].copy()
        zoom = 135 if label == 'Whole head' else (22 if label == 'Pial outlier' else 40)
        for axis, slider in enumerate(sliders):
            slider.eventson = False
            slider.set_val(round(center[axis]))
            slider.eventson = True
        redraw()
    radio.on_clicked(preset)
    if args.export:
        args.export.mkdir(parents=True, exist_ok=False)
        for label in ['Repair', 'Pial outlier']:
            radio.set_active(list(regions).index(label))
            fig.savefig(args.export/(label.lower().replace(' ', '_')+'.png'), dpi=150)
        print(f'Review images saved to {args.export}; anatomy approval unchanged.')
    else:
        redraw()
        print('Select a region, then scroll through slices using the sliders. Close the window to exit.')
        print('Inspect skull/cortex fit and the magenta pial outlier; this viewer does not approve anatomy.')
        plt.show()


if __name__ == '__main__':
    main()
