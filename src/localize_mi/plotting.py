"""
Shared source-localization plotting functions.

Purpose
-------
This module creates figures that are common to all inverse methods:

1. Averaged scalp EEG inside the localized interval.
2. Source-estimate time course at the maximum location.
3. Spatial comparison between the estimated maximum and known
   stimulation midpoint.

Keeping these functions separate ensures that MNE, dSPM, sLORETA, and
eLORETA results use the same visual presentation.

Each function saves one PNG file and closes the figure afterward.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


def create_target_evoked_figure(
    evoked,
    method,
    output_file,
):
    """
    Plot the scalp EEG interval supplied to the inverse method.

    Each line represents one good EEG channel. This figure shows the
    sensor-space signal used to estimate cortical activity.

    Parameters
    ----------
    evoked : mne.Evoked
        Averaged EEG cropped to the target interval.
    method : str
        Name of the inverse method.
    output_file : path-like
        Destination PNG file.

    Returns
    -------
    pathlib.Path
        Saved figure path.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    good_evoked = evoked.copy().pick(
        "eeg",
        exclude="bads",
    )

    times_ms = good_evoked.times * 1000
    amplitudes_uv = good_evoked.data * 1e6

    figure, axis = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    axis.plot(
        times_ms,
        amplitudes_uv.T,
        color="#2166ac",
        alpha=0.18,
        linewidth=0.7,
    )

    axis.axvline(
        0,
        color="#b2182b",
        linestyle="--",
        linewidth=1.2,
        label="Stimulation",
    )

    axis.axhline(
        0,
        color="black",
        linewidth=0.6,
        alpha=0.5,
    )

    axis.set_title(
        f"Averaged EEG supplied to {method}"
    )
    axis.set_xlabel(
        "Time relative to stimulation (ms)"
    )
    axis.set_ylabel("Amplitude (µV)")
    axis.legend()

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_file


def create_peak_time_course_figure(
    source_estimate,
    peak_row,
    peak_time,
    method,
    output_file,
):
    """
    Plot source activity at the strongest estimated location.

    Parameters
    ----------
    source_estimate : mne.SourceEstimate
        Estimated cortical activity across the target interval.
    peak_row : int
        Row containing the maximum source location.
    peak_time : float
        Time of the maximum in seconds.
    method : str
        Name of the inverse method.
    output_file : path-like
        Destination PNG file.

    Returns
    -------
    pathlib.Path
        Saved figure path.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    times_ms = source_estimate.times * 1000

    source_values = source_estimate.data[
        peak_row,
        :
    ]

    peak_time_ms = peak_time * 1000

    figure, axis = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    axis.plot(
        times_ms,
        source_values,
        color="#542788",
        linewidth=1.8,
    )

    axis.axvline(
        0,
        color="#b2182b",
        linestyle="--",
        linewidth=1.2,
        label="Stimulation",
    )

    axis.axvline(
        peak_time_ms,
        color="#f46d43",
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Maximum at "
            f"{peak_time_ms:.3f} ms"
        ),
    )

    axis.set_title(
        f"{method} estimate at the maximum location"
    )
    axis.set_xlabel(
        "Time relative to stimulation (ms)"
    )

    # Different inverse methods use different numerical scales.
    # Therefore, the vertical axis is described generically rather than
    # assigning the same physical unit to every method.
    axis.set_ylabel(
        f"{method} source estimate"
    )

    axis.legend()

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_file


def set_equal_3d_axes(
    axis,
    coordinates,
):
    """
    Give all spatial axes the same scale.

    Equal scaling prevents the cortical surface and localization
    distance from appearing stretched.

    Parameters
    ----------
    axis : matplotlib 3D axis
        Axis containing the cortical surface.
    coordinates : numpy.ndarray
        Coordinates that must fit inside the displayed region.

    Returns
    -------
    None
        The supplied axis is modified directly.
    """

    centre = coordinates.mean(axis=0)

    ranges = np.ptp(
        coordinates,
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


def create_localization_figure(
    forward,
    transform,
    peak_coordinate,
    stimulation_midpoint,
    localization_distance_mm,
    nearest_source_coordinate,
    nearest_source_distance_mm,
    method,
    output_file,
):
    """
    Compare estimated, known, and nearest-possible source locations.

    The markers represent:

    - Blue: maximum estimated by the inverse method.
    - Green: known midpoint of the stimulating SEEG contacts.
    - Orange: nearest available vertex in the cortical source grid.

    The orange point shows whether a large localization error could be
    explained by the cortical grid itself.

    Parameters
    ----------
    forward : mne.Forward
        Participant-specific forward model.
    transform : numpy.ndarray
        Transformation to surface coordinates.
    peak_coordinate : numpy.ndarray
        Estimated source maximum.
    stimulation_midpoint : numpy.ndarray
        Known stimulation midpoint.
    localization_distance_mm : float
        Distance from estimated maximum to stimulation midpoint.
    nearest_source_coordinate : numpy.ndarray
        Closest available cortical source to the stimulation midpoint.
    nearest_source_distance_mm : float
        Distance from that source to the stimulation midpoint.
    method : str
        Name of the inverse method.
    output_file : path-like
        Destination PNG file.

    Returns
    -------
    pathlib.Path
        Saved figure path.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Convert both cortical hemispheres into the coordinate space used
    # by the released SEEG stimulation coordinates.
    surface_coordinates = [
        mne.transforms.apply_trans(
            transform,
            source_space["rr"],
        )
        for source_space in forward["src"]
    ]

    surface_triangles = [
        source_space["tris"]
        for source_space in forward["src"]
    ]

    figure = plt.figure(
        figsize=(10, 8),
        constrained_layout=True,
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    for coordinates, triangles in zip(
        surface_coordinates,
        surface_triangles,
    ):
        axis.plot_trisurf(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            triangles=triangles,
            color="lightgray",
            alpha=0.16,
            linewidth=0,
            shade=True,
        )

    axis.scatter(
        peak_coordinate[0],
        peak_coordinate[1],
        peak_coordinate[2],
        color="#2166ac",
        s=85,
        label=f"{method} maximum",
        depthshade=False,
    )

    axis.scatter(
        stimulation_midpoint[0],
        stimulation_midpoint[1],
        stimulation_midpoint[2],
        color="#1a9850",
        s=85,
        label="Stimulation midpoint",
        depthshade=False,
    )

    axis.scatter(
        nearest_source_coordinate[0],
        nearest_source_coordinate[1],
        nearest_source_coordinate[2],
        color="#fdae61",
        edgecolor="black",
        linewidth=0.5,
        s=65,
        label=(
            "Nearest cortical source "
            f"({nearest_source_distance_mm:.2f} mm)"
        ),
        depthshade=False,
    )

    axis.plot(
        [
            peak_coordinate[0],
            stimulation_midpoint[0],
        ],
        [
            peak_coordinate[1],
            stimulation_midpoint[1],
        ],
        [
            peak_coordinate[2],
            stimulation_midpoint[2],
        ],
        color="#d73027",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Localization distance: "
            f"{localization_distance_mm:.2f} mm"
        ),
    )

    all_coordinates = np.vstack([
        *surface_coordinates,
        peak_coordinate.reshape(1, 3),
        stimulation_midpoint.reshape(1, 3),
        nearest_source_coordinate.reshape(1, 3),
    ])

    set_equal_3d_axes(
        axis,
        all_coordinates,
    )

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")

    axis.set_title(
        f"{method}: estimated and known locations"
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

    return output_file