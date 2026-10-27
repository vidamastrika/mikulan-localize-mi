#!/usr/bin/env python3
"""
Script 23b — Compare original Localize-MI MRI anatomy and registration results.

REFERENCE sub-01 (ANTs moving) -> TEST sub-02 (ANTs fixed).

READ-ONLY: Uses already-generated affine and SyN warped images from Scripts 09/19/31.
Does NOT rerun registration, rewrite transforms, or read EEG/ground-truth locations.

Usage from project root:
  python cross_subject/scripts/23b_compare_affine_syn_mri.py

  python cross_subject/scripts/23b_compare_affine_syn_mri.py \
    --reference sub-03 --test sub-06

Requires: numpy, nibabel, matplotlib. Optional --no-difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ORIENTATIONS = [
    ("Axial", 2),
    ("Coronal", 1),
    ("Sagittal", 0),
]
PANELS = [
    "A. Reference MRI (original)",
    "B. Test MRI (original)",
    "C. Affine: reference in test space",
    "D. SyN: reference in test space",
    "E. Affine overlay",
    "F. SyN overlay",
]


def read_image(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing MRI: {path}\n"
            "Use the existing output from Scripts 09/19/31; this script does not register images."
        )
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D MRI, got shape {image.shape}: {path}")
    array = image.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite intensities: {path}")
    return image, array


def check_target_grid(fixed, transformed, label: str):
    if transformed.shape != fixed.shape or not np.allclose(
        transformed.affine, fixed.affine, rtol=0, atol=0.02
    ):
        raise ValueError(
            f"{label} does not match TEST MRI voxel/world grid.\n"
            f"Test: shape={fixed.shape}, affine={fixed.affine}\n"
            f"{label}: shape={transformed.shape}, affine={transformed.affine}\n"
            "Do not compare unaligned voxel slices."
        )


def normalized(volume):
    vals = volume[np.isfinite(volume) & (volume > 0)]
    if vals.size < 100:
        raise ValueError("MRI has insufficient positive intensity values")
    low, high = np.percentile(vals, [3, 99])
    if high <= low:
        raise ValueError("MRI has no useful intensity contrast")
    return np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)


def center_of_foreground(volume):
    """Robust display center, NOT an anatomical landmark."""
    foreground = volume[np.isfinite(volume) & (volume > 0)]
    threshold = np.percentile(foreground, 45)
    coords = np.argwhere(volume > threshold)
    if len(coords) == 0:
        return np.asarray(volume.shape) // 2
    return np.rint(np.median(coords, axis=0)).astype(int)


def slice_at(volume, axis, index):
    # Display all images identically in voxel-index orientation.
    # Do not interpret panel left/right as anatomical hemisphere labels.
    return np.rot90(np.take(volume, int(index), axis=axis))


def make_overlay(test_slice, warped_slice):
    # test image = magenta (red+blue); warped reference = green.
    rgb = np.zeros((*test_slice.shape, 3), dtype=np.float32)
    rgb[..., 0] = test_slice
    rgb[..., 2] = test_slice
    rgb[..., 1] = warped_slice
    return rgb


def panel_arrays(ref, test, affine, syn, axis, idx_ref, idx_test):
    a = slice_at(ref, axis, idx_ref)
    b = slice_at(test, axis, idx_test)
    c = slice_at(affine, axis, idx_test)
    d = slice_at(syn, axis, idx_test)
    return [a, b, c, d, make_overlay(b, c), make_overlay(b, d)]


def draw_six(axes, arrays, *, names=True):
    for i, (ax, arr) in enumerate(zip(axes, arrays)):
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1, origin="lower")
        else:
            ax.imshow(arr, origin="lower")
        ax.axis("off")
        if names:
            ax.set_title(PANELS[i], fontsize=10)


def save_view(ref, test, aff, syn, axis, title, idx_ref, idx_test, output, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    draw_six(axes.ravel(), panel_arrays(ref, test, aff, syn, axis, idx_ref, idx_test))
    fig.suptitle(
        f"{title}: reference MRI → test MRI\n"
        f"Reference slice {idx_ref} (own native grid); test-space slice {idx_test}",
        fontsize=14,
    )
    fig.text(
        0.5, 0.012,
        "Overlays: TEST = magenta; registered REFERENCE = green. "
        "Light/white = similar intensity (not validated anatomical correspondence).",
        ha="center", fontsize=9,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_all(ref, test, aff, syn, refcenter, testcenter, output, dpi):
    fig, axes = plt.subplots(3, 6, figsize=(24, 13), layout="constrained")
    for row, (title, axis) in enumerate(ORIENTATIONS):
        draw_six(
            axes[row, :],
            panel_arrays(ref, test, aff, syn, axis, refcenter[axis], testcenter[axis]),
            names=row == 0,
        )
        axes[row, 0].set_ylabel(title, fontsize=13)
    fig.suptitle(
        "MRI registration comparison | Original reference and test; "
        "affine and SyN in the test MRI space",
        fontsize=15,
    )
    fig.text(
        0.5, 0.005,
        "TEST = magenta; transformed REFERENCE = green. "
        "Display normalization is per volume. All transformed slices use the test grid.",
        ha="center", fontsize=10,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_difference(test, aff, syn, center, output, dpi):
    """Illustrative normalized-intensity differences; not registration accuracy metrics."""
    # Both already independently normalized for display. Raw intensity difference
    # cannot be interpreted as true anatomical displacement.
    da = np.abs(test - aff)
    ds = np.abs(test - syn)
    improvement = da - ds  # positive: SyN lower display-scaled intensity mismatch
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), layout="constrained")
    names = ("|Test − Affine|", "|Test − SyN|", "Affine mismatch − SyN mismatch")
    for i, (title, axis) in enumerate(ORIENTATIONS):
        idx = center[axis]
        arrays = [slice_at(v, axis, idx) for v in (da, ds, improvement)]
        for j, arr in enumerate(arrays):
            if j == 2:
                im = axes[i, j].imshow(arr, origin="lower", cmap="coolwarm",
                                       vmin=-0.5, vmax=0.5)
            else:
                im = axes[i, j].imshow(arr, origin="lower", cmap="magma",
                                       vmin=0, vmax=1)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(names[j], fontsize=11)
            fig.colorbar(im, ax=axes[i, j], fraction=0.045, pad=0.02)
        axes[i, 0].set_ylabel(title, fontsize=12)
    fig.suptitle(
        "MRI intensity differences (illustrative only)\n"
        "Positive improvement: SyN has smaller normalized-intensity difference",
        fontsize=13,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reference", default="sub-01", help="Reference anatomy (moving)")
    p.add_argument("--test", default="sub-02", help="Test anatomy (fixed)")
    p.add_argument(
        "--dataset-root", type=Path,
        default=Path.home() / "Documents" / "DATASETS SOURCE LOCALIZATION" / "Localize-MI"
    )
    p.add_argument(
        "--affine-root", type=Path,
        default=Path("cross_subject/outputs/09_anatomical_registration")
    )
    p.add_argument(
        "--syn-root", type=Path,
        default=Path("cross_subject/outputs/19_syn_registration")
    )
    p.add_argument(
        "--output-root", type=Path,
        default=Path("cross_subject/outputs/23b_mri_registration_comparison")
    )
    p.add_argument("--no-difference", action="store_true")
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    if args.reference == args.test:
        p.error("--reference must differ from --test")

    reference, test = args.reference, args.test
    pair = f"{reference}_to_{test}"

    reference_path = (args.dataset_root / "derivatives" / "epochs" / reference
                      / "anat" / f"{reference}_mod-T1w_maskface.nii")
    test_path = (args.dataset_root / "derivatives" / "epochs" / test
                 / "anat" / f"{test}_mod-T1w_maskface.nii")
    affine_path = args.affine_root / pair / f"{reference}_in_{test}.nii.gz"
    syn_path = args.syn_root / pair / "donor_syn_in_target.nii.gz"

    paths = {
        "reference_mri": reference_path,
        "test_mri": test_path,
        "affine_registered_reference": affine_path,
        "syn_registered_reference": syn_path,
    }
    images, arrays = {}, {}
    for label, path in paths.items():
        images[label], arrays[label] = read_image(path)
    fixed = images["test_mri"]
    for label in ("affine_registered_reference", "syn_registered_reference"):
        check_target_grid(fixed, images[label], label)

    original_ref = arrays["reference_mri"]
    original_test = arrays["test_mri"]
    ref_center = center_of_foreground(original_ref)
    test_center = center_of_foreground(original_test)

    r = normalized(original_ref)
    t = normalized(original_test)
    aff = normalized(arrays["affine_registered_reference"])
    syn = normalized(arrays["syn_registered_reference"])

    output = args.output_root / pair
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for n, (label, axis) in enumerate(ORIENTATIONS, start=1):
        destination = output / f"{n:02d}_{label.lower()}_comparison.png"
        save_view(r, t, aff, syn, axis, label, ref_center[axis],
                  test_center[axis], destination, args.dpi)
        generated.append(destination)

    destination = output / "04_all_views_comparison.png"
    save_all(r, t, aff, syn, ref_center, test_center, destination, args.dpi)
    generated.append(destination)

    if not args.no_difference:
        destination = output / "05_affine_vs_syn_difference.png"
        save_difference(t, aff, syn, test_center, destination, args.dpi)
        generated.append(destination)

    report = {
        "reference": reference,
        "test": test,
        "registration_direction": "reference (moving) to test (fixed)",
        "input_images": {k: str(v) for k, v in paths.items()},
        "all_transformed_images_on_test_grid": True,
        "original_reference_own_grid": True,
        "reference_native_slice_indices": ref_center.tolist(),
        "test_grid_slice_indices": test_center.tolist(),
        "overlay_colors": {"test": "magenta", "transformed_reference": "green"},
        "outputs": [str(path) for path in generated],
        "note": (
            "MRI-only descriptive visualization. Contrast normalized separately "
            "per MRI for display; normalized intensity differences do not prove "
            "registration quality, source localization accuracy, or anatomical "
            "correspondence. Reference original and test original use each "
            "participant's native slice center; registered images use identical "
            "test-space slice indices."
        ),
    }
    (output / "06_visualization_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Read-only MRI visualization complete: {pair}")
    for path in generated:
        print(" ", path)
    print(" Manifest:", output / "06_visualization_manifest.json")


if __name__ == "__main__":
    main()
