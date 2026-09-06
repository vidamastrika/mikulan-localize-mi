#!/usr/bin/env python3
"""
Inspect one EEG stimulation run from the Localize-MI dataset.

Purpose
-------
This script checks whether one released run can be safely used for source
source-localization analysis. It combines the EEG data with the matching
participant-specific forward model.

The released EEG electrode TSV files are identical across participants.
Therefore, the modern loader uses the sensor positions stored inside each
participant's forward model.

Inputs
------
The script reads:

1. Preprocessed EEG epochs stored as a NumPy array.
2. Channel and event information stored in TSV files.
3. Baseline and coordinate descriptions stored in JSON files.
4. The participant-specific forward model.
5. Intracranial electrode coordinates for the stimulation pair.

Outputs
-------
The script prints:

1. EEG dimensions, timing, and channel quality.
2. The baseline information applied before data release.
3. A comparison between the released EEG coordinates and forward model.
4. A check of the participant-specific geometry used by our loader.
5. Forward-model dimensions.
6. The stimulating SEEG contacts and their midpoint.
7. A final validation summary.

No files are created or modified by this script.

Examples
--------
Run the default example:

    python scripts/01_inspect_dataset.py

Select another participant and run:

    python scripts/01_inspect_dataset.py \
        --subject sub-07 \
        --run run-07
"""

from argparse import ArgumentParser
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# Locate the repository and its src directory automatically.
# This allows the script to work without using an absolute project path.
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SOURCE_DIR))

from localize_mi import load_run  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "Localize-MI"

# Differences below 0.02 mm are treated as numerical rounding.
GEOMETRY_TOLERANCE_MM = 0.02


def parse_arguments():
    """
    Read the participant, task, run, and dataset path.

    Returns
    -------
    argparse.Namespace
        The selections provided on the command line. Defaults are used
        when no selections are provided.
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

    return parser.parse_args()


def require_files(paths):
    """
    Check that all files needed for the report exist.

    Parameters
    ----------
    paths : list of pathlib.Path
        Files required for the selected participant and run.

    Returns
    -------
    None
        The function returns nothing when all files exist.

    Raises
    ------
    FileNotFoundError
        Raised with a list of missing files when the selected run is
        incomplete or the dataset path is incorrect.
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
            "The following required files were not found:\n"
            f"{formatted}"
        )


def parse_stimulation_contacts(description):
    """
    Find the two contacts involved in bipolar stimulation.

    For example, the description ``K13-14 1mA`` refers to contacts K13
    and K14. Their midpoint represents the known stimulation location
    used later to evaluate source-localization accuracy.

    Parameters
    ----------
    description : str
        Description from the run's epoch JSON file.

    Returns
    -------
    stimulation_pair : str
        Bipolar stimulation label, such as K13-14.
    contacts : list of str
        Two individual contact names, such as K13 and K14.
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

    # The electrode files use straight apostrophes.
    prefix = prefix.replace("’", "'")

    first_contact = f"{prefix}{first_number}"
    second_contact = f"{prefix}{second_number}"

    stimulation_pair = (
        f"{first_contact}-{second_number}"
    )

    return stimulation_pair, [
        first_contact,
        second_contact,
    ]


def calculate_position_differences(
    first_positions,
    second_positions,
):
    """
    Calculate the distance between two versions of each sensor position.

    Parameters
    ----------
    first_positions : numpy.ndarray
        First set of sensor coordinates in metres.
    second_positions : numpy.ndarray
        Second set of sensor coordinates in metres.

    Returns
    -------
    numpy.ndarray
        One distance per EEG sensor in millimetres.
    """

    if first_positions.shape != second_positions.shape:
        raise ValueError(
            "The two coordinate arrays have different shapes: "
            f"{first_positions.shape} and "
            f"{second_positions.shape}."
        )

    differences_m = np.linalg.norm(
        first_positions - second_positions,
        axis=1,
    )

    return differences_m * 1000


def status_count(channels, requested_status):
    """
    Count channels marked good or bad.

    Parameters
    ----------
    channels : pandas.DataFrame
        Released channel-information table.
    requested_status : str
        Status to count, normally ``good`` or ``bad``.

    Returns
    -------
    int
        Number of channels with the requested status.
    """

    statuses = (
        channels["status"]
        .astype(str)
        .str.lower()
    )

    return int(
        statuses.eq(requested_status.lower()).sum()
    )


def main():
    """
    Load the selected run, validate it, and print its summary.

    Returns
    -------
    None
        Results are printed to the terminal. No output file is created.
    """

    args = parse_arguments()

    dataset = args.dataset.expanduser().resolve()
    subject = args.subject
    task = args.task
    run = args.run

    if not dataset.is_dir():
        raise NotADirectoryError(
            f"Dataset directory was not found: {dataset}"
        )

    eeg_dir = (
        dataset
        / "derivatives"
        / "epochs"
        / subject
        / "eeg"
    )

    ieeg_dir = (
        dataset
        / "derivatives"
        / "epochs"
        / subject
        / "ieeg"
    )

    task_basename = f"{subject}_task-{task}"

    eeg_electrodes_file = (
        eeg_dir
        / f"{task_basename}_electrodes.tsv"
    )

    seeg_electrodes_file = (
        ieeg_dir
        / (
            f"{task_basename}"
            "_space-surface_electrodes.tsv"
        )
    )

    require_files([
        eeg_electrodes_file,
        seeg_electrodes_file,
    ])

    # Load the EEG epochs and participant-specific forward model.
    #
    # The loader obtains sensor geometry from the forward model rather
    # than the duplicated electrode TSV file.
    loaded = load_run(
        dataset=dataset,
        subject=subject,
        task=task,
        run=run,
    )

    epochs = loaded.epochs
    forward = loaded.forward
    channels = loaded.channels
    events = loaded.events
    metadata = loaded.metadata
    coordinate_metadata = loaded.coordinate_metadata

    epoch_data = epochs.get_data(copy=False)

    n_epochs, n_channels, n_samples = (
        epoch_data.shape
    )

    sfreq = float(epochs.info["sfreq"])
    duration = (n_samples - 1) / sfreq

    # Read the released EEG coordinate table separately.
    #
    # We do not use these coordinates for source localization. They are
    # loaded only to document whether they agree with the participant's
    # forward model.
    eeg_electrodes = pd.read_csv(
        eeg_electrodes_file,
        sep="\t",
    ).set_index("name")

    forward_channel_names = (
        forward["info"]["ch_names"]
    )

    if epochs.ch_names != forward_channel_names:
        raise ValueError(
            "The loaded epochs and forward model have "
            "different channel orders."
        )

    if not set(forward_channel_names).issubset(
        eeg_electrodes.index
    ):
        missing = sorted(
            set(forward_channel_names)
            - set(eeg_electrodes.index)
        )

        raise ValueError(
            "The released electrode table is missing "
            f"channels: {missing}"
        )

    released_positions = (
        eeg_electrodes.loc[
            forward_channel_names,
            ["x", "y", "z"],
        ]
        .to_numpy(dtype=float)
    )

    forward_positions = np.array([
        channel["loc"][:3]
        for channel in forward["info"]["chs"]
    ])

    loaded_positions = np.array([
        channel["loc"][:3]
        for channel in epochs.info["chs"]
    ])

    # This comparison reveals whether the shared EEG TSV happens to
    # match the selected participant's forward model.
    released_differences_mm = (
        calculate_position_differences(
            released_positions,
            forward_positions,
        )
    )

    # This comparison verifies the geometry actually used by our loader.
    loaded_differences_mm = (
        calculate_position_differences(
            loaded_positions,
            forward_positions,
        )
    )

    released_geometry_matches = bool(
        released_differences_mm.max()
        <= GEOMETRY_TOLERANCE_MM
    )

    loaded_geometry_matches = bool(
        loaded_differences_mm.max()
        <= GEOMETRY_TOLERANCE_MM
    )

    # Identify the known intracranial stimulation location.
    description = metadata["Description"]

    stimulation_pair, stimulation_contacts = (
        parse_stimulation_contacts(description)
    )

    seeg_electrodes = pd.read_csv(
        seeg_electrodes_file,
        sep="\t",
    )

    selected_contacts = seeg_electrodes.loc[
        seeg_electrodes["name"].isin(
            stimulation_contacts
        ),
        ["name", "x", "y", "z"],
    ]

    if len(selected_contacts) != 2:
        found_contacts = (
            selected_contacts["name"]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Could not find both stimulation contacts. "
            f"Expected {stimulation_contacts}; "
            f"found {found_contacts}."
        )

    # Restore the intended ordering before calculating the midpoint.
    selected_contacts = (
        selected_contacts
        .set_index("name")
        .loc[stimulation_contacts]
    )

    stimulation_midpoint = (
        selected_contacts[["x", "y", "z"]]
        .mean()
        .to_numpy()
    )

    good_channels = status_count(
        channels,
        "good",
    )

    bad_channels = status_count(
        channels,
        "bad",
    )

    # Each hemisphere is one source space. The active vertices from
    # both hemispheres give the total number of cortical locations.
    source_locations = sum(
        len(source_space["vertno"])
        for source_space in forward["src"]
    )

    leadfield_shape = (
        forward["sol"]["data"].shape
    )

    orientations_per_source = (
        leadfield_shape[1] // source_locations
    )

    print("\nLOCALIZE-MI RUN INSPECTION")
    print("--------------------------")
    print(f"Dataset             : {dataset}")
    print(f"Subject             : {subject}")
    print(f"Task                : {task}")
    print(f"Run                 : {run}")

    print("\nEEG EPOCHS")
    print("----------")
    print(f"Epochs              : {n_epochs}")
    print(f"EEG channels        : {n_channels}")
    print(f"Good channels       : {good_channels}")
    print(f"Bad channels        : {bad_channels}")
    print(f"Samples per epoch   : {n_samples}")
    print(f"Sampling frequency  : {sfreq:.1f} Hz")
    print(
        f"Time range          : "
        f"{epochs.tmin:.5f} to {epochs.tmax:.5f} s"
    )
    print(f"Duration            : {duration:.5f} s")
    print(f"Data type           : {epoch_data.dtype}")
    print(
        f"Data units          : "
        f"{channels['units'].iloc[0]}"
    )
    print(
        "Baseline corrected  : "
        f"{metadata.get('BaselineCorrection')}"
    )
    print(
        "Original baseline   : "
        f"{loaded.original_baseline} s"
    )
    print(
        "MNE baseline field  : "
        f"{epochs.baseline}"
    )

    print("\nEVENTS")
    print("------")
    print(f"Event rows          : {len(events)}")
    print(
        f"Event description   : "
        f"{events['trial_type'].iloc[0]}"
    )

    print("\nSENSOR GEOMETRY")
    print("---------------")
    print(
        "Sidecar system      : "
        f"{coordinate_metadata.get('EEGCoordinateSystem')}"
    )
    print(
        "Sidecar units       : "
        f"{coordinate_metadata.get('EEGCoordinateUnits')}"
    )
    print(
        "Loaded montage frame: "
        f"{epochs.get_montage().get_positions()['coord_frame']}"
    )
    print(
        "Released TSV mean   : "
        f"{released_differences_mm.mean():.6f} mm"
    )
    print(
        "Released TSV maximum: "
        f"{released_differences_mm.max():.6f} mm"
    )
    print(
        "Released TSV matches: "
        f"{'YES' if released_geometry_matches else 'NO'}"
    )
    print(
        "Loaded mean error   : "
        f"{loaded_differences_mm.mean():.6f} mm"
    )
    print(
        "Loaded maximum error: "
        f"{loaded_differences_mm.max():.6f} mm"
    )
    print(
        "Loaded model matches: "
        f"{'YES' if loaded_geometry_matches else 'NO'}"
    )

    if not released_geometry_matches:
        print(
            "Geometry note       : The released EEG TSV "
            "contains shared coordinates. The loader instead "
            "uses this participant's forward-model geometry."
        )

    print("\nFORWARD MODEL")
    print("-------------")
    print(f"EEG channels        : {forward['nchan']}")
    print(f"Source spaces       : {len(forward['src'])}")
    print(f"Source locations    : {source_locations}")
    print(
        f"Orientations/source : "
        f"{orientations_per_source}"
    )
    print(f"Lead-field shape    : {leadfield_shape}")

    print("\nSTIMULATION")
    print("-----------")
    print(f"Description         : {description}")
    print(f"Bipolar pair        : {stimulation_pair}")
    print(
        "Contacts            : "
        f"{', '.join(stimulation_contacts)}"
    )
    print(
        "Surface midpoint    : "
        f"[{stimulation_midpoint[0]:.5f}, "
        f"{stimulation_midpoint[1]:.5f}, "
        f"{stimulation_midpoint[2]:.5f}] m"
    )

    print("\nVALIDATION")
    print("----------")
    print("Epoch/event counts  : PASS")
    print("Epoch/channel counts: PASS")
    print("Channel names/order : PASS")
    print(
        "Loaded EEG geometry : "
        f"{'PASS' if loaded_geometry_matches else 'FAIL'}"
    )
    print(
        "Released TSV geometry: "
        f"{'PASS' if released_geometry_matches else 'WARNING'}"
    )

    if not loaded_geometry_matches:
        raise RuntimeError(
            "The epochs do not match the participant-specific "
            "forward-model geometry."
        )


if __name__ == "__main__":
    main()