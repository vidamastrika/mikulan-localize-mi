#!/usr/bin/env python3
"""
Create quality-control figures for one Localize-MI EEG run.

Purpose
-------
This script visually checks the EEG data before source localization.

It uses the participant-specific EEG positions stored in the forward
model. It does not use the duplicated EEG electrode coordinates as the
source of sensor geometry.

The figures help us answer four questions:

1. Do the EEG sensors align sensibly with the participant's scalp?
2. Is there a consistent stimulation-locked response across epochs?
3. When is the overall scalp voltage pattern strongest?
4. How does the scalp voltage pattern change around stimulation?

Important interpretation
------------------------
The very large signal close to 0 ms is likely dominated by the immediate
electrical stimulation transient. It should not automatically be
interpreted as a physiological cortical response.

Inputs
------
The script reads:

1. One preprocessed EEG epoch array.
2. Its channel and event metadata.
3. The participant-specific forward model.
4. The participant's outer-scalp surface.
5. The head-to-surface transformation.

Outputs
-------
Four PNG files are saved under:

    outputs/figures/<subject>/<run>/

The files are:

1. 01_sensor_alignment.png
2. 02_evoked_butterfly.png
3. 03_global_field_power.png
4. 04_scalp_topographies.png

No dataset files are modified.

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


# Find the repository and modern source package automatically.
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SOURCE_DIR))

from localize_mi import load_run  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "Localize-MI"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "figures"


def parse_arguments():
    """
    Read the participant, run, plot window, and output location.

    Returns
    -------
    argparse.Namespace
        The selections provided through the command line.
    """

    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Participant ID, for example sub-01.",
    )

    parser.add_argument(
        "--task",
        default="seegstim",
        help="Task name. The released task is seegstim.",
    )

    parser.add_argument(
        "--run",
        default="run-01",
        help="Run ID, for example run-01.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the Localize-MI dataset.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for saved figures.",
    )

    parser.add_argument(
        "--plot-tmin",
        type=float,
        default=-0.020,
        help=(
            "Beginning of the main EEG plots in seconds. "
            "The default is -0.020 seconds."
        ),
    )

    parser.add_argument(
        "--plot-tmax",
        type=float,
        default=0.010,
        help=(
            "End of the main EEG plots in seconds. "
            "The default is 0.010 seconds."
        ),
    )

    parser.add_argument(
        "--post-tmin",
        type=float,
        default=0.001,
        help=(
            "Beginning of the enlarged post-stimulation "
            "panel in seconds."
        ),
    )

    return parser.parse_args()


def require_files(paths):
    """
    Check that all files needed for visualization exist.

    Parameters
    ----------
    paths : list of pathlib.Path
        Files required to create the figures.

    Returns
    -------
    None
        Nothing is returned when all required files exist.

    Raises
    ------
    FileNotFoundError
        Raised with a list of missing files.
    """

    missing = [
        path
        for path in paths
        if not path.is_file()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required visualization files are missing:\n"
            f"{formatted}"
        )


def load_surface(surface_file):
    """
    Load the participant's outer-scalp surface.

    Parameters
    ----------
    surface_file : pathlib.Path
        GIFTI file containing scalp vertices and triangles.

    Returns
    -------
    vertices : numpy.ndarray
        Scalp coordinates in metres.
    triangles : numpy.ndarray
        Indices defining the triangular surface mesh.
    """

    surface = nib.load(surface_file)

    if len(surface.darrays) < 2:
        raise ValueError(
            "The surface file does not contain both "
            f"vertices and triangles: {surface_file}"
        )

    vertices = np.asarray(
        surface.darrays[0].data,
        dtype=float,
    )

    triangles = np.asarray(
        surface.darrays[1].data,
        dtype=int,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "Expected scalp vertices with shape "
            f"(number of vertices, 3), found "
            f"{vertices.shape}."
        )

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            "Expected triangular faces with shape "
            f"(number of triangles, 3), found "
            f"{triangles.shape}."
        )

    # Some surface formats store coordinates in millimetres.
    # Values larger than one indicate that conversion to metres
    # is needed for compatibility with MNE coordinates.
    maximum_radius = np.max(
        np.linalg.norm(
            vertices,
            axis=1,
        )
    )

    if maximum_radius > 1:
        vertices = vertices / 1000

    return vertices, triangles


def load_head_to_surface_transform(transform_file):
    """
    Load the transform connecting EEG head and anatomical coordinates.

    Parameters
    ----------
    transform_file : pathlib.Path
        HDF5 transformation file for the selected participant.

    Returns
    -------
    numpy.ndarray
        A 4 × 4 transformation matrix.
    """

    with h5py.File(transform_file, "r") as file:
        if "trans" not in file:
            raise KeyError(
                "No transformation named 'trans' was found "
                f"in {transform_file}."
            )

        transform = np.asarray(
            file["trans"][()],
            dtype=float,
        )

    if transform.shape != (4, 4):
        raise ValueError(
            "Expected a 4 × 4 transformation matrix, "
            f"found {transform.shape}."
        )

    return transform


def get_plot_window(
    epochs,
    requested_tmin,
    requested_tmax,
):
    """
    Restrict a requested plot window to available EEG samples.

    Parameters
    ----------
    epochs : mne.Epochs
        Loaded EEG epochs.
    requested_tmin : float
        Requested beginning in seconds.
    requested_tmax : float
        Requested ending in seconds.

    Returns
    -------
    plot_tmin : float
        Valid beginning time.
    plot_tmax : float
        Valid ending time.
    """

    plot_tmin = max(
        float(requested_tmin),
        float(epochs.tmin),
    )

    plot_tmax = min(
        float(requested_tmax),
        float(epochs.tmax),
    )

    if plot_tmin >= plot_tmax:
        raise ValueError(
            "The requested plotting window does not overlap "
            f"the available interval {epochs.tmin} to "
            f"{epochs.tmax} seconds."
        )

    return plot_tmin, plot_tmax


def create_sensor_alignment_figure(
    epochs,
    surface_vertices,
    surface_triangles,
    head_to_surface,
    output_file,
):
    """
    Plot participant-specific EEG sensors around the scalp.

    Good sensors are blue and channels marked bad are red. This figure
    checks whether the sensor geometry used by the forward model is
    anatomically plausible.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs containing participant-specific sensor positions.
    surface_vertices : numpy.ndarray
        Outer-scalp coordinates.
    surface_triangles : numpy.ndarray
        Triangles forming the scalp surface.
    head_to_surface : numpy.ndarray
        Transformation from head to anatomical surface coordinates.
    output_file : pathlib.Path
        Destination PNG file.

    Returns
    -------
    None
        The figure is saved and closed.
    """

    head_positions = np.array([
        channel["loc"][:3]
        for channel in epochs.info["chs"]
    ])

    # The sensors and scalp must be represented in the same coordinate
    # space before they can be displayed together.
    surface_positions = mne.transforms.apply_trans(
        head_to_surface,
        head_positions,
    )

    bad_names = set(epochs.info["bads"])

    bad_mask = np.array([
        channel_name in bad_names
        for channel_name in epochs.ch_names
    ])

    good_mask = ~bad_mask

    figure = plt.figure(
        figsize=(9, 8),
        constrained_layout=True,
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

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
        label=(
            f"Good EEG sensors "
            f"({int(good_mask.sum())})"
        ),
        depthshade=False,
    )

    axis.scatter(
        surface_positions[bad_mask, 0],
        surface_positions[bad_mask, 1],
        surface_positions[bad_mask, 2],
        color="#b2182b",
        s=20,
        label=(
            f"Bad EEG sensors "
            f"({int(bad_mask.sum())})"
        ),
        depthshade=False,
    )

    # Equal axis ranges prevent the head from appearing artificially
    # stretched along one direction.
    all_positions = np.vstack([
        surface_vertices,
        surface_positions,
    ])

    centre = all_positions.mean(axis=0)
    ranges = np.ptp(
        all_positions,
        axis=0,
    )
    radius = ranges.max() / 2

    axis.set_xlim(
        centre[0] - radius,
        centre[0] + radius,
    )
    axis.set_ylim(
        centre[1] - radius,
        centre[1] + radius,
    )
    axis.set_zlim(
        centre[2] - radius,
        centre[2] + radius,
    )

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(
        "Participant-specific EEG sensor alignment"
    )
    axis.legend(loc="upper right")
    axis.view_init(
        elev=20,
        azim=110,
    )

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_evoked_butterfly_figure(
    evoked,
    plot_tmin,
    plot_tmax,
    post_tmin,
    output_file,
):
    """
    Plot the averaged response of all good EEG channels.

    The upper panel shows the complete selected interval, including the
    large stimulation transient. The lower panel enlarges the later
    interval so that smaller post-stimulation activity remains visible.

    Parameters
    ----------
    evoked : mne.Evoked
        EEG response averaged across epochs.
    plot_tmin : float
        Beginning of the full displayed interval.
    plot_tmax : float
        End of the full displayed interval.
    post_tmin : float
        Beginning of the enlarged post-stimulation panel.
    output_file : pathlib.Path
        Destination PNG file.

    Returns
    -------
    None
        The figure is saved and closed.
    """

    good_evoked = evoked.copy().pick(
        "eeg",
        exclude="bads",
    )

    times_ms = good_evoked.times * 1000
    data_uv = good_evoked.data * 1e6

    full_mask = (
        (good_evoked.times >= plot_tmin)
        & (good_evoked.times <= plot_tmax)
    )

    valid_post_tmin = max(
        float(post_tmin),
        float(plot_tmin),
        float(good_evoked.tmin),
    )

    post_mask = (
        (good_evoked.times >= valid_post_tmin)
        & (good_evoked.times <= plot_tmax)
    )

    if not np.any(post_mask):
        raise ValueError(
            "The enlarged post-stimulation interval "
            "contains no EEG samples."
        )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 9),
        constrained_layout=True,
    )

    full_axis, post_axis = axes

    # Full response, including the immediate stimulation transient.
    full_axis.plot(
        times_ms[full_mask],
        data_uv[:, full_mask].T,
        color="#2166ac",
        alpha=0.15,
        linewidth=0.6,
    )

    full_axis.axvspan(
        -1,
        1,
        color="#f4a582",
        alpha=0.15,
        label="Immediate stimulation interval",
    )

    full_axis.axvline(
        0,
        color="#b2182b",
        linestyle="--",
        linewidth=1.2,
        label="Stimulation",
    )

    full_axis.axhline(
        0,
        color="black",
        linewidth=0.6,
        alpha=0.5,
    )

    full_axis.set_title(
        "Full averaged stimulation-locked response"
    )
    full_axis.set_xlabel(
        "Time relative to stimulation (ms)"
    )
    full_axis.set_ylabel("Amplitude (µV)")
    full_axis.legend(loc="upper right")

    # Enlarged post-stimulation view. Its vertical scale is determined
    # only by this later interval, making smaller responses visible.
    post_axis.plot(
        times_ms[post_mask],
        data_uv[:, post_mask].T,
        color="#1b7837",
        alpha=0.18,
        linewidth=0.7,
    )

    post_axis.axhline(
        0,
        color="black",
        linewidth=0.6,
        alpha=0.5,
    )

    post_axis.set_title(
        "Enlarged post-stimulation response"
    )
    post_axis.set_xlabel(
        "Time relative to stimulation (ms)"
    )
    post_axis.set_ylabel("Amplitude (µV)")

    figure.suptitle(
        "Averaged EEG response across good channels",
        fontsize=15,
    )

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_global_field_power_figure(
    evoked,
    plot_tmin,
    plot_tmax,
    output_file,
):
    """
    Plot the overall strength of the scalp voltage pattern.

    Global field power is calculated as the standard deviation across
    good EEG channels at every time point. A large value means the scalp
    pattern is strong, but it does not prove that the signal is a
    physiological brain response.

    Parameters
    ----------
    evoked : mne.Evoked
        EEG response averaged across epochs.
    plot_tmin : float
        Beginning of the displayed interval.
    plot_tmax : float
        End of the displayed interval.
    output_file : pathlib.Path
        Destination PNG file.

    Returns
    -------
    None
        The figure is saved and closed.
    """

    good_evoked = evoked.copy().pick(
        "eeg",
        exclude="bads",
    )

    times_ms = good_evoked.times * 1000

    global_field_power_uv = (
        np.std(
            good_evoked.data,
            axis=0,
        )
        * 1e6
    )

    time_mask = (
        (good_evoked.times >= plot_tmin)
        & (good_evoked.times <= plot_tmax)
    )

    displayed_gfp = global_field_power_uv[
        time_mask
    ]

    displayed_times = times_ms[
        time_mask
    ]

    peak_index = int(
        np.argmax(displayed_gfp)
    )

    peak_time = displayed_times[
        peak_index
    ]

    peak_value = displayed_gfp[
        peak_index
    ]

    figure, axis = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True,
    )

    axis.plot(
        displayed_times,
        displayed_gfp,
        color="#542788",
        linewidth=1.8,
    )

    axis.axvspan(
        -1,
        1,
        color="#f4a582",
        alpha=0.15,
        label="Immediate stimulation interval",
    )

    axis.axvline(
        0,
        color="#b2182b",
        linestyle="--",
        linewidth=1.2,
        label="Stimulation",
    )

    axis.scatter(
        peak_time,
        peak_value,
        color="#f46d43",
        s=45,
        zorder=3,
        label=(
            f"Largest GFP: {peak_time:.2f} ms, "
            f"{peak_value:.2f} µV"
        ),
    )

    axis.set_title(
        "Global field power of the averaged EEG response"
    )
    axis.set_xlabel(
        "Time relative to stimulation (ms)"
    )
    axis.set_ylabel(
        "Global field power (µV)"
    )
    axis.legend(loc="upper right")

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_topography_figure(
    evoked,
    output_file,
):
    """
    Plot scalp voltage patterns around stimulation.

    The maps show positive and negative voltage distributions measured
    across the scalp. These are sensor-space maps, not estimated brain
    sources.

    A shared colour scale is used so amplitudes can be compared across
    time. This means later low-amplitude maps may appear faint when the
    immediate stimulation transient is very large.

    Parameters
    ----------
    evoked : mne.Evoked
        EEG response averaged across epochs.
    output_file : pathlib.Path
        Destination PNG file.

    Returns
    -------
    None
        The figure is saved and closed.
    """

    requested_times = np.array([
        -0.0010,
        -0.0005,
        0.0000,
        0.0005,
        0.0010,
        0.0020,
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
        time_format="%0.1f ms",
        contours=6,
        sensors=True,
        show=False,
    )

    figure.suptitle(
        "Scalp voltage patterns around stimulation",
        fontsize=14,
    )

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    """
    Load one run and create all quality-control figures.

    Returns
    -------
    None
        The saved figure locations are printed in the terminal.
    """

    args = parse_arguments()

    dataset = (
        args.dataset
        .expanduser()
        .resolve()
    )

    output_directory = (
        args.output_root
        .expanduser()
        .resolve()
        / args.subject
        / args.run
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    surface_file = (
        dataset
        / "derivatives"
        / "sourcemodelling"
        / args.subject
        / "anat"
        / f"{args.subject}_outer_skin.surf.gii"
    )

    transform_file = (
        dataset
        / "derivatives"
        / "sourcemodelling"
        / args.subject
        / "xfm"
        / (
            f"{args.subject}"
            "_from-head_to-surface.h5"
        )
    )

    require_files([
        surface_file,
        transform_file,
    ])

    # Load the EEG using the participant-specific geometry recovered
    # from the corresponding forward model.
    loaded = load_run(
        dataset=dataset,
        subject=args.subject,
        task=args.task,
        run=args.run,
    )

    epochs = loaded.epochs

    plot_tmin, plot_tmax = (
        get_plot_window(
            epochs=epochs,
            requested_tmin=args.plot_tmin,
            requested_tmax=args.plot_tmax,
        )
    )

    surface_vertices, surface_triangles = (
        load_surface(surface_file)
    )

    head_to_surface = (
        load_head_to_surface_transform(
            transform_file
        )
    )

    # Averaging produces one response representing activity that occurs
    # consistently at the same time relative to stimulation.
    evoked = epochs.average()

    sensor_file = (
        output_directory
        / "01_sensor_alignment.png"
    )

    butterfly_file = (
        output_directory
        / "02_evoked_butterfly.png"
    )

    gfp_file = (
        output_directory
        / "03_global_field_power.png"
    )

    topography_file = (
        output_directory
        / "04_scalp_topographies.png"
    )

    create_sensor_alignment_figure(
        epochs=epochs,
        surface_vertices=surface_vertices,
        surface_triangles=surface_triangles,
        head_to_surface=head_to_surface,
        output_file=sensor_file,
    )

    create_evoked_butterfly_figure(
        evoked=evoked,
        plot_tmin=plot_tmin,
        plot_tmax=plot_tmax,
        post_tmin=args.post_tmin,
        output_file=butterfly_file,
    )

    create_global_field_power_figure(
        evoked=evoked,
        plot_tmin=plot_tmin,
        plot_tmax=plot_tmax,
        output_file=gfp_file,
    )

    create_topography_figure(
        evoked=evoked,
        output_file=topography_file,
    )

    print("\nQUALITY-CONTROL FIGURES")
    print("-----------------------")
    print(f"Subject             : {args.subject}")
    print(f"Run                 : {args.run}")
    print(f"Epochs averaged     : {len(epochs)}")
    print(
        f"Good EEG channels   : "
        f"{len(epochs.ch_names) - len(epochs.info['bads'])}"
    )
    print(
        f"Bad EEG channels    : "
        f"{len(epochs.info['bads'])}"
    )
    print(
        f"Displayed interval  : "
        f"{plot_tmin * 1000:.1f} to "
        f"{plot_tmax * 1000:.1f} ms"
    )
    print(
        f"Post-stimulus view  : "
        f"{args.post_tmin * 1000:.1f} to "
        f"{plot_tmax * 1000:.1f} ms"
    )
    print(f"Sensor alignment    : {sensor_file}")
    print(f"Averaged response   : {butterfly_file}")
    print(f"Global field power  : {gfp_file}")
    print(f"Scalp topographies  : {topography_file}")


if __name__ == "__main__":
    main()