#!/usr/bin/env python3
"""
Validate every released EEG run in the Localize-MI dataset.

Purpose
-------
Before running source localization, this script checks that every EEG run
can be combined with the correct participant-specific forward model.

For each run, it checks:

1. The EEG array can be loaded.
2. Epoch, event, and channel counts are consistent.
3. The channel order and forward model use the same channel order.
4. Our reconstructed EEG geometry matches the forward model.
5. Timing, baseline, stimulation, and forward-model information exist.

Output
------
The script saves:

    outputs/tables/dataset_validation.csv

Each row represents one participant/run combination. A final ``status``
column reports either ``PASS`` or ``FAIL``.

The script also prints a short whole-dataset summary in the terminal.

No dataset files are modified.

Example
-------
Run from the repository root:

    python scripts/02_validate_dataset.py
"""

from pathlib import Path
import gc
import re
import sys

import numpy as np
import pandas as pd


# Find the repository and src folder automatically.
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SOURCE_DIR))

from localize_mi import load_run  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "Localize-MI"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "outputs"
    / "tables"
    / "dataset_validation.csv"
)

GEOMETRY_TOLERANCE_MM = 0.02


def find_runs(dataset):
    """
    Find all released EEG epoch arrays.

    Parameters
    ----------
    dataset : pathlib.Path
        Root directory of the Localize-MI dataset.

    Returns
    -------
    list of dict
        One dictionary per run containing its participant, task,
        run identifier, and epoch-array path.
    """

    pattern = re.compile(
        r"^(sub-\d+)_task-(.+?)_(run-\d+)_epochs\.npy$"
    )

    epoch_root = (
    dataset
    / "derivatives"
    / "epochs"
)

    runs = []

    for path in sorted(
        epoch_root.glob(
            "sub-*/eeg/*_epochs.npy"
        )
    ):
        match = pattern.match(path.name)

        if match is None:
            print(
                "Warning: unrecognized epoch filename:",
                path.name,
            )
            continue

        subject, task, run = match.groups()

        runs.append({
            "subject": subject,
            "task": task,
            "run": run,
            "path": path,
        })

    return runs


def calculate_geometry_error(epochs, forward):
    """
    Compare the loaded EEG positions with the forward-model positions.

    The returned distances show whether the EEG data use the same sensor
    geometry as the lead field. Values close to zero indicate that the
    correct participant-specific geometry was attached.

    Parameters
    ----------
    epochs : m.Epochs
        Loaded EEG epochs.
    forward : mne.Forward
        Participant-specific forward model.

    Returns
    -------
    mean_geometry_mean_mm : float
        Mean difference across EEG sensors, in millimetres.
    geometry_maximum_mm : float
        Largest difference across EEG sensors, in millimetres.
    """

    epoch_positions = np.array([
        channel["loc"][:3]
        for channel in epochs.info["chs"]
    ])

    forward_positions = np.array([
        channel["loc"][:3]
        for channel in forward["info"]["chs"]
    ])

    distances_mm = np.linalg.norm(
        epoch_positions - forward_positions,
        axis=1,
    ) * 1000

    return (
        float(distances_mm.mean()),
        float(distances_mm.max()),
    )


def calculate_released_tsv_error(
    dataset,
    subject,
    task,
    forward,
):
    """
    Compare the released EEG TSV coordinates with the forward model.

    This comparison documents the shared-coordinate issue in the
    released dataset. These TSV coordinates are not used by our modern
    loader.

    Parameters
    ----------
    dataset : pathlib.Path
        Dataset root.
    subject : str
        Participant identifier.
    task : str
        Task name.
    forward : mne.Forward
        Participant-specific forward model.

    Returns
    -------
    mean_error_mm : float
        Mean TSV-to-forward difference in millimetres.
    maximum_error_mm : float
        Maximum TSV-to-forward difference in millimetres.
    """

    electrode_file = (
        dataset
        / "derivatives"
        / "epochs"
        / subject
        / "eeg"
        / f"{subject}_task-{task}_electrodes.tsv"
    )

    electrodes = pd.read_csv(
        electrode_file,
        sep="\t",
    ).set_index("name")

    channel_names = forward["info"]["ch_names"]

    released_positions = (
        electrodes.loc[
            channel_names,
            ["x", "y", "z"],
        ]
        .to_numpy(dtype=float)
    )

    forward_positions = np.array([
        channel["loc"][:3]
        for channel in forward["info"]["chs"]
    ])

    distances_mm = np.linalg.norm(
        released_positions - forward_positions,
        axis=1,
    ) * 1000

    return (
        float(distances_mm.mean()),
        float(distances_mm.max()),
    )


def extract_stimulation(description):
    """
    Extract the stimulation pair and intensity from a description.

    Parameters
    ----------
    description : str
        Description such as
        ``Stimulation of channel K13-14 1mA``.

    Returns
    -------
    stimulation_pair : str
        Bipolar contact pair, such as K13-14.
    stimulation_intensity : str
        Stimulation intensity, such as 1mA.
    """

    pair_match = re.search(
        r"([A-Za-z]+['’]?\d+-\d+)",
        description,
    )

    intensity_match = re.search(
        r"(\d+(?:\.\d+)?\s*mA)",
        description,
        flags=re.IGNORECASE,
    )

    stimulation_pair = (
        pair_match.group(1).replace("’", "'")
        if pair_match
        else "unknown"
    )

    stimulation_intensity = (
        intensity_match.group(1).replace(" ", "")
        if intensity_match
        else "unknown"
    )

    return (
        stimulation_pair,
        stimulation_intensity,
    )


def validate_run(dataset, subject, task, run):
    """
    Validate one participant/run combination.

    Parameters
    ----------
    dataset : pathlib.Path
        Dataset root.
    subject : str
        Participant identifier.
    task : str
        Task name.
    run : str
        Run identifier.

    Returns
    -------
    dict
        One table row containing run characteristics and validation
        results.
    """

    result = {
        "subject": subject,
        "task": task,
        "run": run,
        "status": "FAIL",
        "error": "",
    }

    try:
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

        data = epochs.get_data(copy=False)
        n_epochs, n_channels, n_samples = data.shape

        good_channels = int(
            channels["status"]
            .astype(str)
            .str.lower()
            .eq("good")
            .sum()
        )

        bad_channels = int(
            channels["status"]
            .astype(str)
            .str.lower()
            .eq("bad")
            .sum()
        )

        geometry_mean_mm, geometry_maximum_mm = (
            calculate_geometry_error(
                epochs,
                forward,
            )
        )

        released_mean_mm, released_maximum_mm = (
            calculate_released_tsv_error(
                dataset,
                subject,
                task,
                forward,
            )
        )

        source_locations = sum(
            len(source_space["vertno"])
            for source_space in forward["src"]
        )

        leadfield_rows, leadfield_columns = (
            forward["sol"]["data"].shape
        )

        orientations_per_source = (
            leadfield_columns // source_locations
        )

        description = metadata.get(
            "Description",
            "",
        )

        stimulation_pair, stimulation_intensity = (
            extract_stimulation(description)
        )

        duration = (
            (n_samples - 1)
            / float(epochs.info["sfreq"])
        )

        loaded_geometry_pass = (
            geometry_maximum_mm
            <= GEOMETRY_TOLERANCE_MM
        )

        event_count_pass = (
            n_epochs == len(events)
        )

        channel_count_pass = (
            n_channels == len(channels)
        )

        channel_order_pass = (
            epochs.ch_names
            == forward["info"]["ch_names"]
        )

        overall_pass = all([
            loaded_geometry_pass,
            event_count_pass,
            channel_count_pass,
            channel_order_pass,
        ])

        result.update({
            "epochs": n_epochs,
            "events": len(events),
            "channels": n_channels,
            "good_channels": good_channels,
            "bad_channels": bad_channels,
            "samples": n_samples,
            "sampling_frequency_hz": float(
                epochs.info["sfreq"]
            ),
            "tmin_s": float(epochs.tmin),
            "tmax_s": float(epochs.tmax),
            "duration_s": float(duration),
            "baseline_corrected": metadata.get(
                "BaselineCorrection"
            ),
            "original_baseline": str(
                loaded.original_baseline
            ),
            "mne_baseline": str(
                epochs.baseline
            ),
            "stimulation_pair": stimulation_pair,
            "stimulation_intensity": (
                stimulation_intensity
            ),
            "source_spaces": len(
                forward["src"]
            ),
            "source_locations": source_locations,
            "orientations_per_source": (
                orientations_per_source
            ),
            "leadfield_rows": leadfield_rows,
            "leadfield_columns": leadfield_columns,
            "loaded_geometry_mean_mm": (
                geometry_mean_mm
            ),
            "loaded_geometry_maximum_mm": (
                geometry_maximum_mm
            ),
            "loaded_geometry_pass": (
                loaded_geometry_pass
            ),
            "released_tsv_mean_mm": (
                released_mean_mm
            ),
            "released_tsv_maximum_mm": (
                released_maximum_mm
            ),
            "released_tsv_matches_forward": (
                released_maximum_mm
                <= GEOMETRY_TOLERANCE_MM
            ),
            "event_count_pass": event_count_pass,
            "channel_count_pass": (
                channel_count_pass
            ),
            "channel_order_pass": (
                channel_order_pass
            ),
            "status": (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
        })

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def main():
    """
    Validate all discovered runs and save the summary table.

    Returns
    -------
    None
        The validation results are printed and saved as CSV.
    """

    dataset = DEFAULT_DATASET.expanduser().resolve()
    output_file = DEFAULT_OUTPUT.resolve()

    if not dataset.is_dir():
        raise NotADirectoryError(
            f"Dataset directory was not found: {dataset}"
        )

    runs = find_runs(dataset)

    if not runs:
        raise FileNotFoundError(
            "No released EEG epoch arrays were found."
        )

    print("\nLOCALIZE-MI DATASET VALIDATION")
    print("------------------------------")
    print(f"Dataset    : {dataset}")
    print(f"Runs found : {len(runs)}")
    print()

    results = []

    for index, run_information in enumerate(
        runs,
        start=1,
    ):
        subject = run_information["subject"]
        task = run_information["task"]
        run = run_information["run"]

        print(
            f"[{index:02d}/{len(runs):02d}] "
            f"{subject} {run}",
            end=" ... ",
            flush=True,
        )

        result = validate_run(
            dataset=dataset,
            subject=subject,
            task=task,
            run=run,
        )

        results.append(result)

        print(result["status"])

        # Release the previous EEG array before loading the next run.
        gc.collect()

    table = pd.DataFrame(results)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_file,
        index=False,
    )

    passed = int(
        table["status"].eq("PASS").sum()
    )

    failed = int(
        table["status"].eq("FAIL").sum()
    )

    participants = int(
        table["subject"].nunique()
    )

    total_epochs = int(
        pd.to_numeric(
            table.get("epochs"),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print("\nVALIDATION SUMMARY")
    print("------------------")
    print(f"Participants       : {participants}")
    print(f"Runs checked       : {len(table)}")
    print(f"Runs passed        : {passed}")
    print(f"Runs failed        : {failed}")
    print(f"Total epochs       : {total_epochs}")
    print(f"Saved table        : {output_file}")

    if failed:
        print("\nFAILED RUNS")
        print("-----------")

        failed_columns = [
            "subject",
            "run",
            "error",
        ]

        print(
            table.loc[
                table["status"].eq("FAIL"),
                failed_columns,
            ].to_string(index=False)
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()