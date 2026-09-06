"""Input/output utilities for the Localize-MI dataset."""

from dataclasses import dataclass
from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd


@dataclass
class LocalizeMIRun:
    """Data and metadata for one Localize-MI run."""

    epochs: mne.EpochsArray
    forward: mne.Forward
    events: pd.DataFrame
    channels: pd.DataFrame
    metadata: dict
    coordinate_metadata: dict
    dataset: Path
    subject: str
    task: str
    run: str

    @property
    def original_baseline(self):
        """Baseline interval applied before data release."""

        return self.metadata.get("BaselinePeriod")


def _load_json(path):
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _require_files(paths):
    """Raise an error when required files are missing."""

    missing = [path for path in paths if not path.is_file()]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required Localize-MI files are missing:\n"
            f"{formatted}"
        )


def _create_mne_events(event_table):
    """Create an MNE event array from the released event table."""

    descriptions = (
        event_table["trial_type"]
        .astype(str)
        .tolist()
    )

    unique_descriptions = list(
        dict.fromkeys(descriptions)
    )

    event_id = {
        description: index + 1
        for index, description
        in enumerate(unique_descriptions)
    }

    event_codes = np.array([
        event_id[description]
        for description in descriptions
    ])

    events = np.column_stack([
        np.arange(len(event_table)),
        np.zeros(len(event_table), dtype=int),
        event_codes,
    ]).astype(int)

    return events, event_id


def _create_forward_montage(forward, channel_names):
    """
    Reconstruct a participant-specific montage from a forward model.

    The released EEG electrode TSV files are identical across
    participants. The forward models retain the sensor positions
    originally used to calculate each participant's lead field.
    """

    forward_names = forward["info"]["ch_names"]

    if channel_names != forward_names:
        channel_set = set(channel_names)
        forward_set = set(forward_names)

        raise ValueError(
            "Channel order differs between the epochs and "
            "forward model.\n"
            f"Only in epochs: "
            f"{sorted(channel_set - forward_set)}\n"
            f"Only in forward model: "
            f"{sorted(forward_set - channel_set)}"
        )

    channel_positions = {
        channel["ch_name"]: np.asarray(
            channel["loc"][:3],
            dtype=float,
        )
        for channel in forward["info"]["chs"]
    }

    invalid = [
        name
        for name, position in channel_positions.items()
        if (
            position.shape != (3,)
            or not np.isfinite(position).all()
        )
    ]

    if invalid:
        raise ValueError(
            "Invalid sensor positions in the forward model: "
            f"{invalid}"
        )

    return mne.channels.make_dig_montage(
        ch_pos=channel_positions,
        coord_frame="head",
    )


def load_run(
    dataset,
    subject="sub-01",
    task="seegstim",
    run="run-01",
):
    """
    Load one Localize-MI EEG run with participant-specific geometry.

    Parameters
    ----------
    dataset : path-like
        Path to the Localize-MI BIDS root.
    subject : str
        BIDS participant identifier, such as ``sub-01``.
    task : str
        BIDS task name.
    run : str
        BIDS run identifier, such as ``run-01``.

    Returns
    -------
    LocalizeMIRun
        Epochs, forward solution, tabular metadata, and sidecars.

    Notes
    -----
    The released arrays were baseline-corrected before export. The
    original baseline interval is retained in ``metadata`` but is not
    assigned to ``epochs.baseline`` because part of that interval was
    removed when the released epochs were cropped.
    """

    dataset = Path(dataset).expanduser().resolve()

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

    coordinate_file = (
        eeg_dir
        / f"{task_basename}_coordsystem.json"
    )

    forward_file = (
        dataset
        / "derivatives"
        / "sourcemodelling"
        / subject
        / "fwd"
        / f"{subject}_fwd.fif"
    )

    _require_files([
        array_file,
        channels_file,
        events_file,
        metadata_file,
        coordinate_file,
        forward_file,
    ])

    data = np.load(array_file)

    channels = pd.read_csv(
        channels_file,
        sep="\t",
    )

    events_table = pd.read_csv(
        events_file,
        sep="\t",
    )

    metadata = _load_json(metadata_file)
    coordinate_metadata = _load_json(
        coordinate_file
    )

    if data.ndim != 3:
        raise ValueError(
            "Expected an array with dimensions "
            "epochs × channels × samples, but found "
            f"{data.shape}."
        )

    if data.shape[0] != len(events_table):
        raise ValueError(
            "Epoch and event counts differ: "
            f"{data.shape[0]} versus {len(events_table)}."
        )

    if data.shape[1] != len(channels):
        raise ValueError(
            "Array and channel-table counts differ: "
            f"{data.shape[1]} versus {len(channels)}."
        )

    channel_names = (
        channels["name"]
        .astype(str)
        .tolist()
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

    zero_times = (
        events_table["zero_time"]
        .astype(float)
        .unique()
    )

    if len(zero_times) != 1:
        raise ValueError(
            "The event table contains inconsistent "
            "zero-time values."
        )

    sfreq = float(sampling_frequencies[0])
    tmin = -float(zero_times[0])

    forward = mne.read_forward_solution(
        forward_file,
        verbose=False,
    )

    montage = _create_forward_montage(
        forward,
        channel_names,
    )

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(channel_names),
        verbose=False,
    )

    info.set_montage(
        montage,
        on_missing="raise",
        verbose=False,
    )

    bad_channels = (
        channels.loc[
            channels["status"]
            .astype(str)
            .str.lower()
            .eq("bad"),
            "name",
        ]
        .astype(str)
        .tolist()
    )

    info["bads"] = bad_channels
    info["description"] = metadata.get("Description")

    mne_events, event_id = _create_mne_events(
        events_table
    )

    epochs = mne.EpochsArray(
        data,
        info,
        events=mne_events,
        event_id=event_id,
        tmin=tmin,
        baseline=None,
        verbose=False,
    )

    return LocalizeMIRun(
        epochs=epochs,
        forward=forward,
        events=events_table,
        channels=channels,
        metadata=metadata,
        coordinate_metadata=coordinate_metadata,
        dataset=dataset,
        subject=subject,
        task=task,
        run=run,
    )