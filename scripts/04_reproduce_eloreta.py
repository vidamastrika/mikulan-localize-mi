#!/usr/bin/env python3
"""
Reproduce the Localize-MI eLORETA demonstration for one EEG run.

Purpose
-------
This script follows the source-localization workflow in the authors'
example notebook:

1. Load the released preprocessed EEG epochs.
2. Use the matching participant-specific forward model.
3. Apply an average EEG reference.
4. Estimate noise covariance from -250 to -50 ms.
5. Average the epochs and retain -2 to +2 ms.
6. Create the inverse operator.
7. Apply eLORETA.
8. Find the strongest estimated source.
9. Compare it with the known stimulation midpoint.

Important interpretation
------------------------
The -2 to +2 ms interval contains the immediate electrical stimulation
transient. This analysis evaluates whether that stimulation location can
be recovered from scalp EEG. It does not necessarily localize a later
physiological response.

Outputs
-------
Results are saved under:

    outputs/source_localization/<subject>/<run>/

The files are:

1. eloreta_summary.json
2. 01_evoked_target_window.png
3. 02_peak_source_time_course.png
4. 03_localization_comparison.png

No dataset files are modified.

Example
-------
Run from the repository root:

    python scripts/04_reproduce_eloreta.py \
        --subject sub-01 \
        --run run-01
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import sys
import warnings

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


# Locate the repository and modern source package.
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SOURCE_DIR))

from localize_mi import load_run  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "Localize-MI"

DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "source_localization"
)


def parse_arguments():
    """
    Read the participant, run, dataset, and output selections.

    Returns
    -------
    argparse.Namespace
        Command-line selections. The defaults use sub-01/run-01.
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
        help="Root directory for saved results.",
    )

    return parser.parse_args()


def require_files(paths):
    """
    Confirm that all files needed for localization exist.

    Parameters
    ----------
    paths : list of pathlib.Path
        Required files for the selected participant.

    Returns
    -------
    None
        Nothing is returned when all files exist.

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
            "Required source-localization files are missing:\n"
            f"{formatted}"
        )


def parse_stimulation(description):
    """
    Identify the stimulation pair and its hemisphere.

    In the authors' code, labels containing an apostrophe are assigned
    to the left hemisphere. Labels without an apostrophe are assigned
    to the right hemisphere.

    Parameters
    ----------
    description : str
        Description such as
        ``Stimulation of channel K13-14 1mA``.

    Returns
    -------
    stimulation_pair : str
        Bipolar stimulation label, such as K13-14.
    contacts : list of str
        Two individual contacts, such as K13 and K14.
    hemisphere : str
        ``lh`` for left or ``rh`` for right.
    """

    match = re.search(
        r"([A-Za-z]+['’]?)(\d+)-(\d+)",
        description,
    )

    if match is None:
        raise ValueError(
            "Could not identify the stimulation pair in "
            f"description: {description!r}"
        )

    prefix, first_number, second_number = match.groups()

    # Normalize curly apostrophes to match the electrode table.
    prefix = prefix.replace("’", "'")

    first_contact = f"{prefix}{first_number}"
    second_contact = f"{prefix}{second_number}"

    stimulation_pair = (
        f"{first_contact}-{second_number}"
    )

    hemisphere = (
        "lh"
        if "'" in stimulation_pair
        else "rh"
    )

    return (
        stimulation_pair,
        [first_contact, second_contact],
        hemisphere,
    )


def load_stimulation_midpoint(
    electrode_file,
    contact_names,
):
    """
    Calculate the known location of the electrical stimulation.

    Stimulation was delivered between two neighbouring SEEG contacts.
    Their midpoint is used as the known target location.

    Parameters
    ----------
    electrode_file : pathlib.Path
        Surface-space SEEG electrode table.
    contact_names : list of str
        Names of the two stimulating contacts.

    Returns
    -------
    numpy.ndarray
        Three-dimensional midpoint in metres.
    """

    electrodes = pd.read_csv(
        electrode_file,
        sep="\t",
    )

    selected = electrodes.loc[
        electrodes["name"].isin(contact_names),
        ["name", "x", "y", "z"],
    ]

    if len(selected) != 2:
        found = (
            selected["name"]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Could not find both stimulating contacts. "
            f"Expected {contact_names}; found {found}."
        )

    # Restore the order used in the stimulation label.
    selected = (
        selected
        .set_index("name")
        .loc[contact_names]
    )

    midpoint = (
        selected[["x", "y", "z"]]
        .mean()
        .to_numpy(dtype=float)
    )

    return midpoint


def load_transform(transform_file):
    """
    Load the participant's head-to-surface transformation.

    This transformation places the reconstructed cortical source and
    stimulation contacts into the same coordinate space.

    Parameters
    ----------
    transform_file : pathlib.Path
        Participant-specific HDF5 transformation file.

    Returns
    -------
    numpy.ndarray
        A 4 × 4 transformation matrix.
    """

    with h5py.File(transform_file, "r") as file:
        if "trans" not in file:
            raise KeyError(
                f"No 'trans' matrix found in {transform_file}."
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


def find_peak_row(
    source_estimate,
    hemisphere,
    peak_vertex,
):
    """
    Find the data-array row corresponding to the peak cortical vertex.

    SourceEstimate stores left-hemisphere rows first, followed by
    right-hemisphere rows.

    Parameters
    ----------
    source_estimate : mne.SourceEstimate
        eLORETA result.
    hemisphere : str
        ``lh`` or ``rh``.
    peak_vertex : int
        Cortical vertex containing the strongest estimate.

    Returns
    -------
    int
        Row index in ``source_estimate.data``.
    """

    hemisphere_index = (
        0
        if hemisphere == "lh"
        else 1
    )

    hemisphere_vertices = (
        source_estimate.vertices[
            hemisphere_index
        ]
    )

    matches = np.flatnonzero(
        hemisphere_vertices == peak_vertex
    )

    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely locate the peak vertex "
            "inside the source estimate."
        )

    peak_row = int(matches[0])

    if hemisphere == "rh":
        peak_row += len(
            source_estimate.vertices[0]
        )

    return peak_row


def create_target_evoked_figure(
    evoked,
    output_file,
):
    """
    Plot the EEG interval supplied to eLORETA.

    Each line represents one good EEG channel. The displayed interval
    is the authors' -2 to +2 ms localization window.

    Parameters
    ----------
    evoked : mne.Evoked
        Averaged EEG cropped to the target interval.
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
        "Averaged EEG used for source localization"
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


def create_peak_time_course_figure(
    source_estimate,
    peak_row,
    peak_time,
    output_file,
):
    """
    Plot eLORETA activity at the strongest estimated source.

    Parameters
    ----------
    source_estimate : mne.SourceEstimate
        eLORETA result across the target interval.
    peak_row : int
        Data row corresponding to the peak vertex.
    peak_time : float
        Time of the maximum in seconds.
    output_file : pathlib.Path
        Destination PNG file.

    Returns
    -------
    None
        The figure is saved and closed.
    """

    times_ms = source_estimate.times * 1000

    source_values = (
        source_estimate.data[
            peak_row,
            :
        ]
    )

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
        "eLORETA estimate at the maximum source location"
    )
    axis.set_xlabel(
        "Time relative to stimulation (ms)"
    )
    axis.set_ylabel(
        "eLORETA source estimate"
    )
    axis.legend()

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def set_equal_3d_axes(
    axis,
    coordinates,
):
    """
    Give the three spatial axes equal scales.

    Equal scaling prevents the brain and localization distance from
    appearing stretched.

    Parameters
    ----------
    axis : matplotlib 3D axis
        Axis containing the cortical surface.
    coordinates : numpy.ndarray
        Coordinates that must fit in the displayed region.

    Returns
    -------
    None
        The axis is updated directly.
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
    distance_mm,
    output_file,
):
    """
    Compare the estimated source with the known stimulation location.

    The cortical surface is grey, the eLORETA maximum is blue, and the
    known stimulation midpoint is green. The dashed line represents
    their Euclidean distance.

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
    distance_mm : float
        Distance between the two locations in millimetres.
    output_file : pathlib.Path
        Destination PNG file.

    Returns
    -------
    None
        The figure is saved and closed.
    """

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
            alpha=0.18,
            linewidth=0,
            shade=True,
        )

    axis.scatter(
        peak_coordinate[0],
        peak_coordinate[1],
        peak_coordinate[2],
        color="#2166ac",
        s=80,
        label="eLORETA maximum",
        depthshade=False,
    )

    axis.scatter(
        stimulation_midpoint[0],
        stimulation_midpoint[1],
        stimulation_midpoint[2],
        color="#1a9850",
        s=80,
        label="Stimulation midpoint",
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
        label=f"Distance: {distance_mm:.2f} mm",
    )

    all_coordinates = np.vstack([
        *surface_coordinates,
        peak_coordinate.reshape(1, 3),
        stimulation_midpoint.reshape(1, 3),
    ])

    set_equal_3d_axes(
        axis,
        all_coordinates,
    )

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(
        "Estimated and known stimulation locations"
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


def main():
    """
    Run the complete eLORETA reproduction for one run.

    Returns
    -------
    None
        Figures and a JSON summary are saved. The main numerical results
        are also printed in the terminal.
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

    task_basename = (
        f"{args.subject}_task-{args.task}"
    )

    seeg_electrode_file = (
        dataset
        / "derivatives"
        / "epochs"
        / args.subject
        / "ieeg"
        / (
            f"{task_basename}"
            "_space-surface_electrodes.tsv"
        )
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
        seeg_electrode_file,
        transform_file,
    ])

    loaded = load_run(
        dataset=dataset,
        subject=args.subject,
        task=args.task,
        run=args.run,
    )

    epochs = loaded.epochs.copy()
    forward = loaded.forward

    description = (
        loaded.metadata["Description"]
    )

    (
        stimulation_pair,
        stimulation_contacts,
        hemisphere,
    ) = parse_stimulation(description)

    stimulation_midpoint = (
        load_stimulation_midpoint(
            electrode_file=seeg_electrode_file,
            contact_names=stimulation_contacts,
        )
    )

    transform = load_transform(
        transform_file
    )

    # Match the authors' notebook by applying the average reference.
    # Channels marked bad are not used to calculate the reference.
    epochs.set_eeg_reference(
        ref_channels="average",
        projection=True,
        verbose=False,
    )

    epochs.apply_proj(
        verbose=False,
    )

    covariance_tmin = -0.250
    covariance_tmax = -0.050

    # The released values were already baseline-corrected before export.
    # The full original baseline is no longer present, so the modern
    # loader correctly records epochs.baseline as None. We suppress only
    # MNE's metadata warning; no additional baseline correction occurs.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Epochs are not baseline corrected.*"
            ),
            category=RuntimeWarning,
        )

        covariance = mne.compute_covariance(
            epochs,
            method="auto",
            tmin=covariance_tmin,
            tmax=covariance_tmax,
            verbose=True,
        )

    loose = 1.0
    depth = 0.1
    snr = 1.0
    lambda2 = 1.0 / snr**2

    # Combine the forward model and background covariance to create the
    # inverse mapping from scalp voltages to cortical source estimates.
    inverse_operator = (
        mne.minimum_norm.make_inverse_operator(
            info=epochs.info,
            forward=forward,
            noise_cov=covariance,
            loose=loose,
            depth=depth,
            verbose=True,
        )
    )

    target_tmin = -0.002
    target_tmax = 0.002

    # Average aligned epochs, then retain the same target interval used
    # in the authors' demonstration.
    evoked = epochs.average()

    evoked.crop(
        tmin=target_tmin,
        tmax=target_tmax,
    )

    source_estimate = (
        mne.minimum_norm.apply_inverse(
            evoked=evoked,
            inverse_operator=inverse_operator,
            lambda2=lambda2,
            method="eLORETA",
            pick_ori=None,
            verbose=True,
        )
    )

    # Follow the authors' approach by searching only within the
    # hemisphere containing the stimulating contacts.
    peak_vertex, peak_time = (
        source_estimate.get_peak(
            hemi=hemisphere,
            mode="abs",
        )
    )

    peak_vertex = int(peak_vertex)
    peak_time = float(peak_time)

    hemisphere_index = (
        0
        if hemisphere == "lh"
        else 1
    )

    peak_row = find_peak_row(
        source_estimate=source_estimate,
        hemisphere=hemisphere,
        peak_vertex=peak_vertex,
    )

    # Convert the estimated vertex into the same surface coordinate
    # system used by the released stimulation-contact coordinates.
    peak_coordinate_native = (
        forward["src"][
            hemisphere_index
        ]["rr"][peak_vertex]
    )

    peak_coordinate = (
        mne.transforms.apply_trans(
            transform,
            peak_coordinate_native,
        )
    )

    distance_mm = float(
        np.linalg.norm(
            peak_coordinate
            - stimulation_midpoint
        )
        * 1000
    )

    target_figure = (
        output_directory
        / "01_evoked_target_window.png"
    )

    time_course_figure = (
        output_directory
        / "02_peak_source_time_course.png"
    )

    localization_figure = (
        output_directory
        / "03_localization_comparison.png"
    )

    summary_file = (
        output_directory
        / "eloreta_summary.json"
    )

    create_target_evoked_figure(
        evoked=evoked,
        output_file=target_figure,
    )

    create_peak_time_course_figure(
        source_estimate=source_estimate,
        peak_row=peak_row,
        peak_time=peak_time,
        output_file=time_course_figure,
    )

    create_localization_figure(
        forward=forward,
        transform=transform,
        peak_coordinate=peak_coordinate,
        stimulation_midpoint=stimulation_midpoint,
        distance_mm=distance_mm,
        output_file=localization_figure,
    )

    summary = {
        "dataset": str(dataset),
        "subject": args.subject,
        "task": args.task,
        "run": args.run,
        "epochs": len(epochs),
        "good_channels": (
            len(epochs.ch_names)
            - len(epochs.info["bads"])
        ),
        "bad_channels": len(
            epochs.info["bads"]
        ),
        "sampling_frequency_hz": float(
            epochs.info["sfreq"]
        ),
        "average_reference": True,
        "covariance_tmin_s": covariance_tmin,
        "covariance_tmax_s": covariance_tmax,
        "covariance_method": "auto",
        "covariance_nfree": int(
            covariance["nfree"]
        ),
        "target_tmin_s": target_tmin,
        "target_tmax_s": target_tmax,
        "inverse_method": "eLORETA",
        "loose": loose,
        "depth": depth,
        "snr": snr,
        "lambda2": lambda2,
        "stimulation_pair": stimulation_pair,
        "stimulation_contacts": (
            stimulation_contacts
        ),
        "hemisphere": hemisphere,
        "peak_vertex": peak_vertex,
        "peak_time_s": peak_time,
        "peak_coordinate_m": (
            peak_coordinate.tolist()
        ),
        "stimulation_midpoint_m": (
            stimulation_midpoint.tolist()
        ),
        "localization_distance_mm": (
            distance_mm
        ),
    }

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nELORETA REPRODUCTION")
    print("--------------------")
    print(f"Subject             : {args.subject}")
    print(f"Run                 : {args.run}")
    print(f"Epochs              : {len(epochs)}")
    print(
        f"Good channels       : "
        f"{summary['good_channels']}"
    )
    print(
        f"Bad channels        : "
        f"{summary['bad_channels']}"
    )
    print(
        f"Covariance window   : "
        f"{covariance_tmin:.3f} to "
        f"{covariance_tmax:.3f} s"
    )
    print(
        f"Covariance samples  : "
        f"{covariance['nfree']}"
    )
    print(
        f"Target window       : "
        f"{target_tmin * 1000:.1f} to "
        f"{target_tmax * 1000:.1f} ms"
    )
    print("Inverse method      : eLORETA")
    print(f"Loose               : {loose}")
    print(f"Depth               : {depth}")
    print(f"SNR                 : {snr}")
    print(f"Lambda²             : {lambda2}")
    print(f"Stimulation pair    : {stimulation_pair}")
    print(f"Hemisphere          : {hemisphere}")
    print(f"Peak vertex         : {peak_vertex}")
    print(
        f"Peak time           : "
        f"{peak_time * 1000:.3f} ms"
    )
    print(
        "Estimated coordinate: "
        f"[{peak_coordinate[0]:.5f}, "
        f"{peak_coordinate[1]:.5f}, "
        f"{peak_coordinate[2]:.5f}] m"
    )
    print(
        "Known midpoint      : "
        f"[{stimulation_midpoint[0]:.5f}, "
        f"{stimulation_midpoint[1]:.5f}, "
        f"{stimulation_midpoint[2]:.5f}] m"
    )
    print(
        f"Localization distance: "
        f"{distance_mm:.2f} mm"
    )
    print(f"Summary             : {summary_file}")
    print(f"Target EEG figure   : {target_figure}")
    print(f"Peak time course    : {time_course_figure}")
    print(
        f"Localization figure : "
        f"{localization_figure}"
    )


if __name__ == "__main__":
    main()