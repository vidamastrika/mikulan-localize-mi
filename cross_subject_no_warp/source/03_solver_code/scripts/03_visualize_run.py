#!/usr/bin/env python3
"""
Create quality-control figures for one Localize-MI EEG run.

The script creates four sensor-space quality-control figures:

1. participant-specific EEG sensor alignment;
2. the averaged EEG butterfly response;
3. global field power;
4. scalp voltage topographies around stimulation.

The large signal close to 0 ms is likely dominated by the immediate
electrical stimulation transient and should not automatically be
interpreted as a physiological cortical response.

Example
-------
Run from the repository root:

    python scripts/03_visualize_run.py \
        --subject sub-01 \
        --run run-01
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SOURCE_DIR))

from localize_mi import load_run  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "Localize-MI"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "figures"


def compact_number(value):
    """Format a number without unnecessary trailing zeros."""

    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def signed_milliseconds(seconds):
    """Format seconds as signed milliseconds."""

    milliseconds = float(seconds) * 1000
    if np.isclose(milliseconds, 0):
        milliseconds = 0.0
    return f"{milliseconds:+g}"


def make_figure_title(title, subject, run, *context_lines):
    """Create a consistent title containing run and analysis context."""

    lines = [title, f"{subject} | {run}"]
    lines.extend(line for line in context_lines if line)
    return "\n".join(lines)


def parse_arguments():
    """Read the participant, run, plot window, and output location."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject", default="sub-01",
        help="Participant ID, for example sub-01.",
    )
    parser.add_argument(
        "--task", default="seegstim",
        help="Task name. The released task is seegstim.",
    )
    parser.add_argument(
        "--run", default="run-01",
        help="Run ID, for example run-01.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="Path to the Localize-MI dataset.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for saved figures.",
    )
    parser.add_argument(
        "--plot-tmin", type=float, default=-0.020,
        help="Beginning of the main EEG plots in seconds.",
    )
    parser.add_argument(
        "--plot-tmax", type=float, default=0.010,
        help="End of the main EEG plots in seconds.",
    )
    parser.add_argument(
        "--post-tmin", type=float, default=0.001,
        help="Beginning of the enlarged post-stimulation panel in seconds.",
    )
    return parser.parse_args()


def require_files(paths):
    """Raise an informative error when required files are missing."""

    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required visualization files are missing:\n"
            f"{formatted}"
        )


def load_surface(surface_file):
    """Load a GIFTI surface and return vertices in metres and triangles."""

    surface = nib.load(surface_file)
    if len(surface.darrays) < 2:
        raise ValueError(
            "The surface file does not contain both vertices and triangles: "
            f"{surface_file}"
        )

    vertices = np.asarray(surface.darrays[0].data, dtype=float)
    triangles = np.asarray(surface.darrays[1].data, dtype=int)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "Expected scalp vertices with shape (number of vertices, 3), "
            f"found {vertices.shape}."
        )
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            "Expected triangular faces with shape (number of triangles, 3), "
            f"found {triangles.shape}."
        )

    maximum_radius = np.max(np.linalg.norm(vertices, axis=1))
    if maximum_radius > 1:
        vertices = vertices / 1000

    return vertices, triangles


def load_head_to_surface_transform(transform_file):
    """Load the 4 x 4 head-to-anatomical-surface transform."""

    with h5py.File(transform_file, "r") as file:
        if "trans" not in file:
            raise KeyError(
                "No transformation named 'trans' was found in "
                f"{transform_file}."
            )
        transform = np.asarray(file["trans"][()], dtype=float)

    if transform.shape != (4, 4):
        raise ValueError(
            "Expected a 4 x 4 transformation matrix, "
            f"found {transform.shape}."
        )
    return transform


def get_plot_window(epochs, requested_tmin, requested_tmax):
    """Restrict a requested plot window to available EEG samples."""

    plot_tmin = max(float(requested_tmin), float(epochs.tmin))
    plot_tmax = min(float(requested_tmax), float(epochs.tmax))
    if plot_tmin >= plot_tmax:
        raise ValueError(
            "The requested plotting window does not overlap the available "
            f"interval {epochs.tmin} to {epochs.tmax} seconds."
        )
    return plot_tmin, plot_tmax


def eeg_run_metadata(epochs):
    """Return EEG-specific counts used in titles and terminal output."""

    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    eeg_names = [epochs.ch_names[index] for index in eeg_picks]
    bad_names = set(epochs.info["bads"])
    bad_eeg_names = [name for name in eeg_names if name in bad_names]
    good_eeg_names = [name for name in eeg_names if name not in bad_names]

    return {
        "eeg_picks": np.asarray(eeg_picks, dtype=int),
        "eeg_names": eeg_names,
        "bad_eeg_names": bad_eeg_names,
        "good_eeg_names": good_eeg_names,
        "epochs": len(epochs),
        "sampling_frequency_hz": float(epochs.info["sfreq"]),
    }


def acquisition_context(metadata, include_bad=True):
    """Create a compact acquisition-summary line for figure titles."""

    if include_bad:
        channel_text = (
            f"EEG: {len(metadata['good_eeg_names'])} good + "
            f"{len(metadata['bad_eeg_names'])} bad"
        )
    else:
        channel_text = f"Good EEG channels: {len(metadata['good_eeg_names'])}"

    return (
        f"{metadata['epochs']} epochs | {channel_text} | "
        f"Sampling: {compact_number(metadata['sampling_frequency_hz'])} Hz"
    )


def create_sensor_alignment_figure(
    epochs,
    surface_vertices,
    surface_triangles,
    head_to_surface,
    output_file,
    subject,
    run,
    metadata,
):
    """Plot participant-specific good and bad EEG sensors on the scalp."""

    eeg_picks = metadata["eeg_picks"]
    head_positions = np.array([
        epochs.info["chs"][index]["loc"][:3]
        for index in eeg_picks
    ])
    eeg_names = [epochs.ch_names[index] for index in eeg_picks]

    surface_positions = mne.transforms.apply_trans(
        head_to_surface,
        head_positions,
    )

    bad_names = set(metadata["bad_eeg_names"])
    bad_mask = np.array([name in bad_names for name in eeg_names])
    good_mask = ~bad_mask

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    axis.plot_trisurf(
        surface_vertices[:, 0],
        surface_vertices[:, 1],
        surface_vertices[:, 2],
        triangles=surface_triangles,
        color="lightgray",
        alpha=0.20,
        linewidth=0,
        shade=True,
    )
    axis.scatter(
        surface_positions[good_mask, 0],
        surface_positions[good_mask, 1],
        surface_positions[good_mask, 2],
        color="#2166ac",
        s=16,
        label=f"Good EEG sensors ({int(good_mask.sum())})",
        depthshade=False,
    )
    axis.scatter(
        surface_positions[bad_mask, 0],
        surface_positions[bad_mask, 1],
        surface_positions[bad_mask, 2],
        color="#b2182b",
        s=20,
        label=f"Bad EEG sensors ({int(bad_mask.sum())})",
        depthshade=False,
    )

    all_positions = np.vstack([surface_vertices, surface_positions])
    centre = all_positions.mean(axis=0)
    radius = np.ptp(all_positions, axis=0).max() / 2
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(
        make_figure_title(
            "Participant-specific EEG sensor alignment",
            subject,
            run,
            acquisition_context(metadata, include_bad=True),
        ),
        pad=18,
    )
    axis.legend(loc="upper right")
    axis.view_init(elev=20, azim=110)

    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_evoked_butterfly_figure(
    evoked,
    plot_tmin,
    plot_tmax,
    post_tmin,
    output_file,
    subject,
    run,
    metadata,
):
    """Plot the full and enlarged averaged response of good EEG channels."""

    good_evoked = evoked.copy().pick("eeg", exclude="bads")
    times_ms = good_evoked.times * 1000
    data_uv = good_evoked.data * 1e6

    full_mask = (
        (good_evoked.times >= plot_tmin)
        & (good_evoked.times <= plot_tmax)
    )
    valid_post_tmin = max(
        float(post_tmin), float(plot_tmin), float(good_evoked.tmin)
    )
    post_mask = (
        (good_evoked.times >= valid_post_tmin)
        & (good_evoked.times <= plot_tmax)
    )
    if not np.any(post_mask):
        raise ValueError(
            "The enlarged post-stimulation interval contains no EEG samples."
        )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 9),
        constrained_layout=True,
    )
    full_axis, post_axis = axes

    full_axis.plot(
        times_ms[full_mask],
        data_uv[:, full_mask].T,
        color="#2166ac",
        alpha=0.15,
        linewidth=0.6,
    )
    full_axis.axvspan(
        -1, 1, color="#f4a582", alpha=0.15,
        label="Immediate stimulation interval",
    )
    full_axis.axvline(
        0, color="#b2182b", linestyle="--", linewidth=1.2,
        label="Stimulation",
    )
    full_axis.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    full_axis.set_title("Full averaged stimulation-locked response")
    full_axis.set_xlabel("Time relative to stimulation (ms)")
    full_axis.set_ylabel("Amplitude (µV)")
    full_axis.legend(loc="upper right")

    post_axis.plot(
        times_ms[post_mask],
        data_uv[:, post_mask].T,
        color="#1b7837",
        alpha=0.18,
        linewidth=0.7,
    )
    post_axis.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    post_axis.set_title("Enlarged post-stimulation response")
    post_axis.set_xlabel("Time relative to stimulation (ms)")
    post_axis.set_ylabel("Amplitude (µV)")

    interval_context = (
        f"Displayed: {signed_milliseconds(plot_tmin)} to "
        f"{signed_milliseconds(plot_tmax)} ms | Enlarged: "
        f"{signed_milliseconds(valid_post_tmin)} to "
        f"{signed_milliseconds(plot_tmax)} ms"
    )
    figure.suptitle(
        make_figure_title(
            "Averaged EEG response across good channels",
            subject,
            run,
            acquisition_context(metadata, include_bad=False),
            interval_context,
        ),
        fontsize=15,
    )

    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_global_field_power_figure(
    evoked,
    plot_tmin,
    plot_tmax,
    output_file,
    subject,
    run,
    metadata,
):
    """Plot global field power calculated across good EEG channels."""

    good_evoked = evoked.copy().pick("eeg", exclude="bads")
    times_ms = good_evoked.times * 1000
    global_field_power_uv = np.std(good_evoked.data, axis=0) * 1e6
    time_mask = (
        (good_evoked.times >= plot_tmin)
        & (good_evoked.times <= plot_tmax)
    )
    displayed_gfp = global_field_power_uv[time_mask]
    displayed_times = times_ms[time_mask]
    peak_index = int(np.argmax(displayed_gfp))
    peak_time = displayed_times[peak_index]
    peak_value = displayed_gfp[peak_index]

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(
        displayed_times,
        displayed_gfp,
        color="#542788",
        linewidth=1.8,
    )
    axis.axvspan(
        -1, 1, color="#f4a582", alpha=0.15,
        label="Immediate stimulation interval",
    )
    axis.axvline(
        0, color="#b2182b", linestyle="--", linewidth=1.2,
        label="Stimulation",
    )
    axis.scatter(
        peak_time,
        peak_value,
        color="#f46d43",
        s=45,
        zorder=3,
        label=f"Largest GFP: {peak_time:.2f} ms, {peak_value:.2f} µV",
    )

    interval_context = (
        f"Displayed interval: {signed_milliseconds(plot_tmin)} to "
        f"{signed_milliseconds(plot_tmax)} ms"
    )
    axis.set_title(
        make_figure_title(
            "Global field power of the averaged EEG response",
            subject,
            run,
            acquisition_context(metadata, include_bad=False),
            interval_context,
        ),
        pad=14,
    )
    axis.set_xlabel("Time relative to stimulation (ms)")
    axis.set_ylabel("Global field power (µV)")
    axis.legend(loc="upper right")

    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_topography_figure(
    evoked,
    output_file,
    subject,
    run,
    metadata,
):
    """
    Plot sensor-space scalp voltage patterns around stimulation.

    The subject and run are displayed above the topographies. Acquisition
    information is placed below the maps to prevent overlap with the
    individual topography time labels.
    """

    requested_times = np.array([
        -0.0005,
        -0.00025,
        0.0000,
        0.000125,
        0.0005,
        0.0010,
    ])

    valid_times = requested_times[
        (requested_times >= evoked.tmin)
        & (requested_times <= evoked.tmax)
    ]

    if len(valid_times) == 0:
        raise ValueError(
            "None of the requested topography times "
            "exist in the EEG interval."
        )

    figure = evoked.plot_topomap(
        times=valid_times,
        ch_type="eeg",
        scalings=1e6,
        units="µV",
        time_unit="ms",
        time_format="%0.3f ms",
        contours=6,
        sensors=True,
        show=False,
    )

    # Disable automatic layout so MNE does not undo our positions.
    if hasattr(figure, "set_layout_engine"):
        figure.set_layout_engine(None)

    figure.set_size_inches(16, 6.5)

    figure.subplots_adjust(
        left=0.03,
        right=0.87,
        bottom=0.08,
        top=1.00,
        wspace=0.20,
        )

    figure.suptitle(
        make_figure_title(
            "Scalp voltage patterns around stimulation",
            subject,
            run,
            acquisition_context(
                metadata,
                include_bad=False,
            ),
        ),
        fontsize=14,
        y=0.97,
        linespacing=1.25,
    )

    # Move the colour-scale legend to the right.
    if len(figure.axes) > len(valid_times):
        colorbar_axis = figure.axes[-1]
        colorbar_axis.set_position([
            0.925,  # left
            0.35,   # bottom
            0.012,  # width
            0.38,   # height
        ])

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    """Load one run and create all four quality-control figures."""

    args = parse_arguments()
    dataset = args.dataset.expanduser().resolve()
    output_directory = (
        args.output_root.expanduser().resolve() / args.subject / args.run
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    surface_file = (
        dataset / "derivatives" / "sourcemodelling" / args.subject
        / "anat" / f"{args.subject}_outer_skin.surf.gii"
    )
    transform_file = (
        dataset / "derivatives" / "sourcemodelling" / args.subject
        / "xfm" / f"{args.subject}_from-head_to-surface.h5"
    )
    require_files([surface_file, transform_file])

    loaded = load_run(
        dataset=dataset,
        subject=args.subject,
        task=args.task,
        run=args.run,
    )
    epochs = loaded.epochs
    metadata = eeg_run_metadata(epochs)
    plot_tmin, plot_tmax = get_plot_window(
        epochs=epochs,
        requested_tmin=args.plot_tmin,
        requested_tmax=args.plot_tmax,
    )
    surface_vertices, surface_triangles = load_surface(surface_file)
    head_to_surface = load_head_to_surface_transform(transform_file)
    evoked = epochs.average()

    sensor_file = output_directory / "01_sensor_alignment.png"
    butterfly_file = output_directory / "02_evoked_butterfly.png"
    gfp_file = output_directory / "03_global_field_power.png"
    topography_file = output_directory / "04_scalp_topographies.png"

    create_sensor_alignment_figure(
        epochs=epochs,
        surface_vertices=surface_vertices,
        surface_triangles=surface_triangles,
        head_to_surface=head_to_surface,
        output_file=sensor_file,
        subject=args.subject,
        run=args.run,
        metadata=metadata,
    )
    create_evoked_butterfly_figure(
        evoked=evoked,
        plot_tmin=plot_tmin,
        plot_tmax=plot_tmax,
        post_tmin=args.post_tmin,
        output_file=butterfly_file,
        subject=args.subject,
        run=args.run,
        metadata=metadata,
    )
    create_global_field_power_figure(
        evoked=evoked,
        plot_tmin=plot_tmin,
        plot_tmax=plot_tmax,
        output_file=gfp_file,
        subject=args.subject,
        run=args.run,
        metadata=metadata,
    )
    create_topography_figure(
        evoked=evoked,
        output_file=topography_file,
        subject=args.subject,
        run=args.run,
        metadata=metadata,
    )

    print("\nQUALITY-CONTROL FIGURES")
    print("-----------------------")
    print(f"Subject             : {args.subject}")
    print(f"Run                 : {args.run}")
    print(f"Epochs averaged     : {metadata['epochs']}")
    print(f"Good EEG channels   : {len(metadata['good_eeg_names'])}")
    print(f"Bad EEG channels    : {len(metadata['bad_eeg_names'])}")
    print(
        f"Sampling frequency  : "
        f"{compact_number(metadata['sampling_frequency_hz'])} Hz"
    )
    print(
        f"Displayed interval  : {plot_tmin * 1000:.1f} to "
        f"{plot_tmax * 1000:.1f} ms"
    )
    print(
        f"Post-stimulus view  : {max(args.post_tmin, plot_tmin) * 1000:.1f} "
        f"to {plot_tmax * 1000:.1f} ms"
    )
    print(f"Sensor alignment    : {sensor_file}")
    print(f"Averaged response   : {butterfly_file}")
    print(f"Global field power  : {gfp_file}")
    print(f"Scalp topographies  : {topography_file}")


if __name__ == "__main__":
    main()
