#!/usr/bin/env python3
"""Inspect one released Localize-MI EEG run and its forward model."""

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import sys

import mne
import numpy as np
import pandas as pd


# Repository root:
# mikulan-localize-mi/scripts/01_inspect_dataset.py
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Allow importing the authors' fx_bids.py from the repository root.
sys.path.insert(0, str(PROJECT_DIR))

from fx_bids import load_bids  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "Localize-MI"
GEOMETRY_TOLERANCE_MM = 0.02


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "--subject",
        default="sub-01",
        help="BIDS participant ID, for example sub-01.",
    )
    parser.add_argument(
        "--run",
        default="run-01",
        help="BIDS run ID, for example run-01.",
    )
    parser.add_argument(
        "--task",
        default="seegstim",
        help="BIDS task name.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the Localize-MI BIDS dataset.",
    )

    return parser.parse_args()


def parse_stimulation_contacts(description):
    """
    Extract monopolar contacts from a bipolar stimulation description.

    Examples
    --------
    "Stimulation of channel K13-14 1mA"
        becomes ("K13-14", ["K13", "K14"]).

    "Stimulation of channel A'1-2 1mA"
        becomes ("A'1-2", ["A'1", "A'2"]).
    """

    match = re.search(
        r"([A-Za-z]+['’]?)(\d+)-(\d+)",
        description,
    )

    if match is None:
        raise ValueError(
            "Could not parse the stimulation pair from "
            f"description: {description!r}"
        )

    prefix, first_number, second_number = match.groups()

    # Normalize curly apostrophes to match the electrode tables.
    prefix = prefix.replace("’", "'")

    first_contact = f"{prefix}{first_number}"
    second_contact = f"{prefix}{second_number}"

    stimulation_pair = (
        f"{prefix}{first_number}-{second_number}"
    )

    return stimulation_pair, [
        first_contact,
        second_contact,
    ]


def require_files(paths):
    """Check that every required file exists."""

    missing = [path for path in paths if not path.is_file()]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "The following required files were not found:\n"
            f"{formatted}"
        )


def load_json(path):
    """Load and return a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_channel_status(channels, status):
    """Count channels with the requested status."""

    normalized_status = (
        channels["status"]
        .astype(str)
        .str.lower()
    )

    return int(normalized_status.eq(status.lower()).sum())


def main():
    """Inspect one Localize-MI run."""

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

    run_basename = f"{subject}_task-{task}_{run}"
    task_basename = f"{subject}_task-{task}"

    array_file = (
        eeg_dir
        / f"{run_basename}_epochs.npy"
    )

    channels_file = (
        eeg_dir
        / f"{run_basename}_channels.tsv"
    )

    events_file = (
        eeg_dir
        / f"{run_basename}_epochs.tsv"
    )

    metadata_file = (
        eeg_dir
        / f"{run_basename}_epochs.json"
    )

    eeg_electrodes_file = (
        eeg_dir
        / f"{task_basename}_electrodes.tsv"
    )

    eeg_coordinates_file = (
        eeg_dir
        / f"{task_basename}_coordsystem.json"
    )

    seeg_electrodes_file = (
        ieeg_dir
        / (
            f"{task_basename}"
            "_space-surface_electrodes.tsv"
        )
    )

    forward_file = (
        dataset
        / "derivatives"
        / "sourcemodelling"
        / subject
        / "fwd"
        / f"{subject}_fwd.fif"
    )

    required_files = [
        array_file,
        channels_file,
        events_file,
        metadata_file,
        eeg_electrodes_file,
        eeg_coordinates_file,
        seeg_electrodes_file,
        forward_file,
    ]

    require_files(required_files)

    # Load the released data and metadata.
    data = np.load(
        array_file,
        mmap_mode="r",
    )

    channels = pd.read_csv(
        channels_file,
        sep="\t",
    )

    events = pd.read_csv(
        events_file,
        sep="\t",
    )

    eeg_electrodes = pd.read_csv(
        eeg_electrodes_file,
        sep="\t",
    )

    seeg_electrodes = pd.read_csv(
        seeg_electrodes_file,
        sep="\t",
    )

    metadata = load_json(metadata_file)
    coordinate_metadata = load_json(
        eeg_coordinates_file
    )

    # Validate the basic epoch-array structure.
    if data.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional array with shape "
            "epochs × channels × samples, but found "
            f"{data.shape}."
        )

    if data.shape[0] != len(events):
        raise ValueError(
            "Epoch count does not match the number of "
            f"event rows: {data.shape[0]} versus "
            f"{len(events)}."
        )

    if data.shape[1] != len(channels):
        raise ValueError(
            "Channel count does not match the channel "
            f"table: {data.shape[1]} versus "
            f"{len(channels)}."
        )

    channel_names = (
        channels["name"]
        .astype(str)
        .tolist()
    )

    electrode_names = (
        eeg_electrodes["name"]
        .astype(str)
        .tolist()
    )

    if set(channel_names) != set(electrode_names):
        raise ValueError(
            "EEG channel and electrode name sets do not match."
        )

    sampling_frequencies = (
        channels["sampling_frequency"]
        .astype(float)
        .unique()
    )

    if len(sampling_frequencies) != 1:
        raise ValueError(
            "The channel table contains inconsistent "
            "sampling frequencies."
        )

    sfreq = float(sampling_frequencies[0])
    zero_time = float(events["zero_time"].iloc[0])

    tmin = -zero_time
    duration = (data.shape[-1] - 1) / sfreq
    tmax = tmin + duration

    # Load epochs through the authors' original loader.
    # This is used here for inspection and comparison.
    with mne.use_log_level("WARNING"):
        epochs = load_bids(
            str(dataset),
            subject,
            task,
            run,
        )

    # Load the participant-specific released forward model.
    forward = mne.read_forward_solution(
        forward_file,
        verbose=False,
    )

    forward_channel_names = (
        forward["info"]["ch_names"]
    )

    if channel_names != forward_channel_names:
        channel_set = set(channel_names)
        forward_set = set(forward_channel_names)

        raise ValueError(
            "Epoch and forward-model channel orders do not "
            "match.\n"
            f"Only in epochs: "
            f"{sorted(channel_set - forward_set)}\n"
            f"Only in forward: "
            f"{sorted(forward_set - channel_set)}"
        )

    # Compare the montage reconstructed by the authors' loader
    # against the sensor geometry embedded in the forward model.
    epoch_positions = np.array([
        channel["loc"][:3]
        for channel in epochs.info["chs"]
    ])

    forward_positions = np.array([
        channel["loc"][:3]
        for channel in forward["info"]["chs"]
    ])

    position_differences = np.linalg.norm(
        epoch_positions - forward_positions,
        axis=1,
    )

    mean_difference_mm = (
        position_differences.mean() * 1000
    )

    maximum_difference_mm = (
        position_differences.max() * 1000
    )

    geometry_matches = (
        maximum_difference_mm
        <= GEOMETRY_TOLERANCE_MM
    )

    # Locate the two intracranial stimulation contacts.
    description = metadata["Description"]

    stimulation_pair, stimulation_contacts = (
        parse_stimulation_contacts(description)
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

    # Restore the contact order from the stimulation label.
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

    good_channels = count_channel_status(
        channels,
        "good",
    )

    bad_channels = count_channel_status(
        channels,
        "bad",
    )

    source_locations = sum(
        len(source_space["vertno"])
        for source_space in forward["src"]
    )

    leadfield_shape = (
        forward["sol"]["data"].shape
    )

    if source_locations == 0:
        raise ValueError(
            "The forward model contains no active "
            "source locations."
        )

    orientation_components = (
        leadfield_shape[1] // source_locations
    )

    # Print the inspection report.
    print("\nLOCALIZE-MI RUN INSPECTION")
    print("--------------------------")
    print(f"Dataset             : {dataset}")
    print(f"Subject             : {subject}")
    print(f"Task                : {task}")
    print(f"Run                 : {run}")

    print("\nEEG EPOCHS")
    print("----------")
    print(f"Epochs              : {data.shape[0]}")
    print(f"EEG channels        : {data.shape[1]}")
    print(f"Good channels       : {good_channels}")
    print(f"Bad channels        : {bad_channels}")
    print(f"Samples per epoch   : {data.shape[2]}")
    print(f"Sampling frequency  : {sfreq:.1f} Hz")
    print(
        f"Time range          : "
        f"{tmin:.5f} to {tmax:.5f} s"
    )
    print(f"Duration            : {duration:.5f} s")
    print(f"Data type           : {data.dtype}")
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
        f"{metadata.get('BaselinePeriod')} s"
    )
    print(
        "MNE baseline field  : "
        f"{epochs.baseline}"
    )

    print("\nCOORDINATES")
    print("-----------")
    print(
        "Sidecar system      : "
        f"{coordinate_metadata.get('EEGCoordinateSystem')}"
    )
    print(
        "Sidecar units       : "
        f"{coordinate_metadata.get('EEGCoordinateUnits')}"
    )
    print(
        "MNE montage frame   : "
        f"{epochs.get_montage().get_positions()['coord_frame']}"
    )
    print(
        "Mean EEG difference : "
        f"{mean_difference_mm:.6f} mm"
    )
    print(
        "Max EEG difference  : "
        f"{maximum_difference_mm:.6f} mm"
    )
    print(
        "TSV matches forward : "
        f"{'YES' if geometry_matches else 'NO'}"
    )

    if not geometry_matches:
        print(
            "Geometry note       : Released EEG TSV "
            "geometry does not match this participant's "
            "forward model."
        )

    print("\nFORWARD MODEL")
    print("-------------")
    print(f"EEG channels        : {forward['nchan']}")
    print(f"Source spaces       : {len(forward['src'])}")
    print(f"Source locations    : {source_locations}")
    print(
        f"Orientations/source : "
        f"{orientation_components}"
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
    print("Channel name sets   : PASS")
    print("Channel names/order : PASS")
    print("Required files      : PASS")
    print(
        "EEG geometry        : "
        f"{'PASS' if geometry_matches else 'WARNING'}"
    )


if __name__ == "__main__":
    main()