"""
Localization measurements for the Localize-MI dataset.

Purpose
-------
This module compares an estimated cortical source with the known
intracranial stimulation location.

It provides functions to:

1. Read the bipolar stimulation pair from run metadata.
2. Calculate the midpoint between the two stimulating SEEG contacts.
3. Find the maximum source estimate in the expected hemisphere.
4. Calculate localization distance.
5. Find the nearest available cortical source to the stimulation point.

The nearest-source distance is important because the inverse solution is
restricted to a cortical grid. It tells us the smallest localization
distance that the source space could theoretically represent.

This module does not run an inverse method, save files, or create figures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import h5py
import mne
import numpy as np
import pandas as pd


@dataclass
class StimulationInfo:
    """
    Store the known stimulation information for one run.

    Attributes
    ----------
    pair : str
        Bipolar stimulation label, such as K13-14.
    contacts : list of str
        Two individual contacts, such as K13 and K14.
    hemisphere : str
        Expected hemisphere: ``lh`` or ``rh``.
    midpoint : numpy.ndarray
        Midpoint between the contacts in surface coordinates.
    """

    pair: str
    contacts: list[str]
    hemisphere: str
    midpoint: np.ndarray


@dataclass
class LocalizationMetrics:
    """
    Store the main localization measurements for one result.

    Attributes
    ----------
    peak_vertex : int
        Cortical vertex containing the largest source estimate.
    peak_row : int
        Row of that vertex in the SourceEstimate data array.
    peak_time : float
        Time of the maximum in seconds.
    peak_coordinate : numpy.ndarray
        Estimated source coordinate in surface space.
    stimulation_midpoint : numpy.ndarray
        Known midpoint of the stimulation contacts.
    localization_distance_mm : float
        Distance between the estimate and known midpoint.
    nearest_source_vertex : int
        Cortical vertex closest to the stimulation midpoint.
    nearest_source_coordinate : numpy.ndarray
        Coordinate of the closest available cortical source.
    nearest_source_distance_mm : float
        Smallest distance representable by the cortical source grid.
    hemisphere : str
        Hemisphere used for the peak and nearest-source searches.
    """

    peak_vertex: int
    peak_row: int
    peak_time: float
    peak_coordinate: np.ndarray
    stimulation_midpoint: np.ndarray
    localization_distance_mm: float
    nearest_source_vertex: int
    nearest_source_coordinate: np.ndarray
    nearest_source_distance_mm: float
    hemisphere: str


def parse_stimulation_description(description):
    """
    Extract the stimulation pair, contacts, and hemisphere.

    In the authors' naming convention, contact labels containing an
    apostrophe belong to the left hemisphere. Labels without an
    apostrophe belong to the right hemisphere.

    Parameters
    ----------
    description : str
        Description such as
        ``Stimulation of channel X'1-2 1mA``.

    Returns
    -------
    pair : str
        Bipolar stimulation pair.
    contacts : list of str
        Two individual contact names.
    hemisphere : str
        ``lh`` for left or ``rh`` for right.
    """

    match = re.search(
        r"([A-Za-z]+['’]?)(\d+)-(\d+)",
        str(description),
    )

    if match is None:
        raise ValueError(
            "Could not identify the stimulation pair in "
            f"description: {description!r}"
        )

    prefix, first_number, second_number = match.groups()

    # Normalize curly apostrophes to match the TSV electrode names.
    prefix = prefix.replace("’", "'")

    first_contact = f"{prefix}{first_number}"
    second_contact = f"{prefix}{second_number}"
    pair = f"{first_contact}-{second_number}"

    hemisphere = (
        "lh"
        if "'" in pair
        else "rh"
    )

    return (
        pair,
        [first_contact, second_contact],
        hemisphere,
    )


def load_stimulation_midpoint(
    electrode_file,
    contact_names,
):
    """
    Calculate the midpoint between two stimulating SEEG contacts.

    Parameters
    ----------
    electrode_file : path-like
        Surface-space SEEG electrode table.
    contact_names : list of str
        Names of the two stimulating contacts.

    Returns
    -------
    numpy.ndarray
        Three-dimensional midpoint in metres.
    """

    electrode_file = Path(
        electrode_file
    ).expanduser().resolve()

    if not electrode_file.is_file():
        raise FileNotFoundError(
            "SEEG electrode file was not found: "
            f"{electrode_file}"
        )

    electrodes = pd.read_csv(
        electrode_file,
        sep="\t",
    )

    required_columns = {
        "name",
        "x",
        "y",
        "z",
    }

    missing_columns = (
        required_columns
        - set(electrodes.columns)
    )

    if missing_columns:
        raise ValueError(
            "SEEG electrode table is missing columns: "
            f"{sorted(missing_columns)}"
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

    if not np.isfinite(midpoint).all():
        raise ValueError(
            "The stimulation midpoint contains invalid values."
        )

    return midpoint


def load_surface_transform(transform_file):
    """
    Load the transformation used for surface-space comparison.

    Parameters
    ----------
    transform_file : path-like
        Participant-specific HDF5 transformation file.

    Returns
    -------
    numpy.ndarray
        A 4 × 4 transformation matrix.
    """

    transform_file = Path(
        transform_file
    ).expanduser().resolve()

    if not transform_file.is_file():
        raise FileNotFoundError(
            "Surface transformation file was not found: "
            f"{transform_file}"
        )

    with h5py.File(
        transform_file,
        "r",
    ) as file:
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


def load_stimulation_info(
    description,
    electrode_file,
):
    """
    Load all known stimulation information for one run.

    Parameters
    ----------
    description : str
        Run description from the epoch metadata.
    electrode_file : path-like
        Surface-space SEEG electrode table.

    Returns
    -------
    StimulationInfo
        Pair, contacts, hemisphere, and midpoint.
    """

    (
        pair,
        contacts,
        hemisphere,
    ) = parse_stimulation_description(
        description
    )

    midpoint = load_stimulation_midpoint(
        electrode_file=electrode_file,
        contact_names=contacts,
    )

    return StimulationInfo(
        pair=pair,
        contacts=contacts,
        hemisphere=hemisphere,
        midpoint=midpoint,
    )


def find_peak_row(
    source_estimate,
    hemisphere,
    peak_vertex,
):
    """
    Locate a cortical vertex inside the source-estimate data array.

    SourceEstimate stores all left-hemisphere rows first, followed by
    all right-hemisphere rows.

    Parameters
    ----------
    source_estimate : mne.SourceEstimate
        Cortical source estimate.
    hemisphere : str
        ``lh`` or ``rh``.
    peak_vertex : int
        Vertex number returned by MNE.

    Returns
    -------
    int
        Matching row in ``source_estimate.data``.
    """

    if hemisphere not in {"lh", "rh"}:
        raise ValueError(
            "Hemisphere must be 'lh' or 'rh'."
        )

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
        hemisphere_vertices
        == int(peak_vertex)
    )

    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely locate vertex "
            f"{peak_vertex} in hemisphere {hemisphere}."
        )

    peak_row = int(matches[0])

    if hemisphere == "rh":
        peak_row += len(
            source_estimate.vertices[0]
        )

    return peak_row


def source_vertex_coordinate(
    forward,
    transform,
    hemisphere,
    vertex,
):
    """
    Convert one source vertex into surface coordinates.

    Parameters
    ----------
    forward : mne.Forward
        Participant-specific forward model.
    transform : numpy.ndarray
        Head-to-surface transformation.
    hemisphere : str
        ``lh`` or ``rh``.
    vertex : int
        Cortical vertex number.

    Returns
    -------
    numpy.ndarray
        Three-dimensional coordinate in metres.
    """

    if hemisphere not in {"lh", "rh"}:
        raise ValueError(
            "Hemisphere must be 'lh' or 'rh'."
        )

    hemisphere_index = (
        0
        if hemisphere == "lh"
        else 1
    )

    source_space = forward["src"][
        hemisphere_index
    ]

    vertex = int(vertex)

    if vertex < 0 or vertex >= len(
        source_space["rr"]
    ):
        raise IndexError(
            f"Vertex {vertex} is outside the "
            f"{hemisphere} source surface."
        )

    native_coordinate = (
        source_space["rr"][vertex]
    )

    surface_coordinate = (
        mne.transforms.apply_trans(
            transform,
            native_coordinate,
        )
    )

    return np.asarray(
        surface_coordinate,
        dtype=float,
    )


def find_nearest_source(
    forward,
    transform,
    hemisphere,
    stimulation_midpoint,
):
    """
    Find the cortical source nearest to the stimulation midpoint.

    This measurement indicates the smallest error that the cortical
    source grid could theoretically represent.

    Parameters
    ----------
    forward : mne.Forward
        Participant-specific forward model.
    transform : numpy.ndarray
        Head-to-surface transformation.
    hemisphere : str
        Hemisphere containing the stimulating contacts.
    stimulation_midpoint : numpy.ndarray
        Known stimulation midpoint.

    Returns
    -------
    nearest_vertex : int
        Closest active cortical vertex.
    nearest_coordinate : numpy.ndarray
        Coordinate of that vertex.
    nearest_distance_mm : float
        Distance from the stimulation midpoint in millimetres.
    """

    if hemisphere not in {"lh", "rh"}:
        raise ValueError(
            "Hemisphere must be 'lh' or 'rh'."
        )

    hemisphere_index = (
        0
        if hemisphere == "lh"
        else 1
    )

    source_space = forward["src"][
        hemisphere_index
    ]

    active_vertices = np.asarray(
        source_space["vertno"],
        dtype=int,
    )

    active_coordinates = (
        mne.transforms.apply_trans(
            transform,
            source_space["rr"][
                active_vertices
            ],
        )
    )

    distances_mm = np.linalg.norm(
        active_coordinates
        - stimulation_midpoint,
        axis=1,
    ) * 1000

    nearest_index = int(
        np.argmin(distances_mm)
    )

    nearest_vertex = int(
        active_vertices[nearest_index]
    )

    nearest_coordinate = np.asarray(
        active_coordinates[nearest_index],
        dtype=float,
    )

    nearest_distance_mm = float(
        distances_mm[nearest_index]
    )

    return (
        nearest_vertex,
        nearest_coordinate,
        nearest_distance_mm,
    )


def calculate_localization_metrics(
    source_estimate,
    forward,
    transform,
    stimulation,
):
    """
    Calculate peak localization and geometric lower-bound measurements.

    Parameters
    ----------
    source_estimate : mne.SourceEstimate
        Cortical source estimate produced by an inverse method.
    forward : mne.Forward
        Participant-specific forward model.
    transform : numpy.ndarray
        Head-to-surface transformation.
    stimulation : StimulationInfo
        Known stimulation pair, hemisphere, and midpoint.

    Returns
    -------
    LocalizationMetrics
        Peak location, localization distance, and nearest-source
        distance.
    """

    peak_vertex, peak_time = (
        source_estimate.get_peak(
            hemi=stimulation.hemisphere,
            mode="abs",
        )
    )

    peak_vertex = int(peak_vertex)
    peak_time = float(peak_time)

    peak_row = find_peak_row(
        source_estimate=source_estimate,
        hemisphere=stimulation.hemisphere,
        peak_vertex=peak_vertex,
    )

    peak_coordinate = (
        source_vertex_coordinate(
            forward=forward,
            transform=transform,
            hemisphere=stimulation.hemisphere,
            vertex=peak_vertex,
        )
    )

    localization_distance_mm = float(
        np.linalg.norm(
            peak_coordinate
            - stimulation.midpoint
        )
        * 1000
    )

    (
        nearest_source_vertex,
        nearest_source_coordinate,
        nearest_source_distance_mm,
    ) = find_nearest_source(
        forward=forward,
        transform=transform,
        hemisphere=stimulation.hemisphere,
        stimulation_midpoint=stimulation.midpoint,
    )

    return LocalizationMetrics(
        peak_vertex=peak_vertex,
        peak_row=peak_row,
        peak_time=peak_time,
        peak_coordinate=peak_coordinate,
        stimulation_midpoint=(
            stimulation.midpoint.copy()
        ),
        localization_distance_mm=(
            localization_distance_mm
        ),
        nearest_source_vertex=(
            nearest_source_vertex
        ),
        nearest_source_coordinate=(
            nearest_source_coordinate
        ),
        nearest_source_distance_mm=(
            nearest_source_distance_mm
        ),
        hemisphere=stimulation.hemisphere,
    )