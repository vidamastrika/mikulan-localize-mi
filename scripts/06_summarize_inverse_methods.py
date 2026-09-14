#!/usr/bin/env python3
"""
Summarize and compare inverse-method localization results.

Purpose
-------
This script reads the results created by Script 05 and compares MNE,
dSPM, sLORETA, and eLORETA across all Localize-MI runs.

It creates:

1. A descriptive summary for each method.
2. Participant-level summaries.
3. Tie-aware counts of the best method for each run.
4. Paired differences between methods.
5. A list of unusually large localization errors.
6. Figures showing method distributions and participant variation.

This script does not rerun source localization.

Input
-----
    outputs/tables/inverse_method_results.csv

Outputs
-------
    outputs/method_comparison/method_summary.csv
    outputs/method_comparison/participant_summary.csv
    outputs/method_comparison/run_winners.csv
    outputs/method_comparison/pairwise_differences.csv
    outputs/method_comparison/outlier_runs.csv
    outputs/method_comparison/01_distance_distribution.png
    outputs/method_comparison/02_paired_differences.png
    outputs/method_comparison/03_participant_medians.png
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = [
    "MNE",
    "dSPM",
    "sLORETA",
    "eLORETA",
]

METHOD_COLORS = {
    "MNE": "#377eb8",
    "dSPM": "#e41a1c",
    "sLORETA": "#4daf4a",
    "eLORETA": "#984ea3",
}

REQUIRED_COLUMNS = {
    "subject",
    "run",
    "method",
    "status",
    "localization_distance_mm",
    "nearest_source_distance_mm",
    "geometric_excess_mm",
    "peak_time_ms",
}


# Read the command-line options.
#
# Output:
#   The input CSV, output directory, and tolerance used when identifying
#   methods with effectively equal localization distances.
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Localize-MI inverse-method results."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "tables"
            / "inverse_method_results.csv"
        ),
        help="CSV created by Script 05.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "method_comparison"
        ),
        help="Directory used for tables and figures.",
    )

    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Distances differing by no more than this value in "
            "millimetres are treated as tied. Default: 1e-6."
        ),
    )

    return parser.parse_args()


# Load the batch-results table and check its basic structure.
#
# Output:
#   A clean table containing successful results in the expected method
#   order.
def load_and_validate_results(input_file):
    input_file = input_file.expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Batch result table was not found: {input_file}"
        )

    table = pd.read_csv(input_file)

    missing_columns = REQUIRED_COLUMNS - set(table.columns)

    if missing_columns:
        raise ValueError(
            "The result table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_count = table.duplicated(
        ["subject", "run", "method"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            "The result table contains "
            f"{duplicate_count} duplicate run-method rows."
        )

    failed = table.loc[
        table["status"].astype(str) != "PASS"
    ]

    if not failed.empty:
        failed_keys = failed[
            ["subject", "run", "method", "status"]
        ]

        raise ValueError(
            "The table contains unsuccessful results:\n"
            f"{failed_keys.to_string(index=False)}"
        )

    if table[
        "localization_distance_mm"
    ].isna().any():
        raise ValueError(
            "Some localization distances are missing."
        )

    available_methods = set(
        table["method"].astype(str)
    )

    unexpected_methods = (
        available_methods - set(METHOD_ORDER)
    )

    if unexpected_methods:
        raise ValueError(
            "Unexpected inverse methods were found: "
            f"{sorted(unexpected_methods)}"
        )

    run_method_counts = (
        table.groupby(["subject", "run"])["method"]
        .nunique()
    )

    expected_method_count = len(available_methods)

    incomplete = run_method_counts.loc[
        run_method_counts != expected_method_count
    ]

    if not incomplete.empty:
        raise ValueError(
            "Some runs do not contain all methods:\n"
            f"{incomplete.to_string()}"
        )

    table["method"] = pd.Categorical(
        table["method"],
        categories=[
            method
            for method in METHOD_ORDER
            if method in available_methods
        ],
        ordered=True,
    )

    table = table.sort_values(
        ["subject", "run", "method"]
    ).reset_index(drop=True)

    return table


# Calculate the main descriptive measurements for each inverse method.
#
# Output:
#   One row per method containing the number of runs, average, median,
#   spread, and range of localization errors.
def summarize_methods(table):
    summary = (
        table.groupby(
            "method",
            observed=True,
        )["localization_distance_mm"]
        .agg(
            runs="count",
            mean_mm="mean",
            standard_deviation_mm="std",
            median_mm="median",
            minimum_mm="min",
            maximum_mm="max",
        )
        .reset_index()
    )

    quartiles = (
        table.groupby(
            "method",
            observed=True,
        )["localization_distance_mm"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(
            columns={
                0.25: "first_quartile_mm",
                0.75: "third_quartile_mm",
            }
        )
        .reset_index()
    )

    summary = summary.merge(
        quartiles,
        on="method",
        how="left",
    )

    summary["interquartile_range_mm"] = (
        summary["third_quartile_mm"]
        - summary["first_quartile_mm"]
    )

    summary = summary[
        [
            "method",
            "runs",
            "mean_mm",
            "standard_deviation_mm",
            "median_mm",
            "first_quartile_mm",
            "third_quartile_mm",
            "interquartile_range_mm",
            "minimum_mm",
            "maximum_mm",
        ]
    ]

    return summary


# Summarize localization performance separately for each participant.
#
# Output:
#   One row for every participant-method combination containing the
#   number of runs and the participant's mean and median errors.
def summarize_participants(table):
    summary = (
        table.groupby(
            ["subject", "method"],
            observed=True,
        )["localization_distance_mm"]
        .agg(
            runs="count",
            mean_mm="mean",
            median_mm="median",
            standard_deviation_mm="std",
            minimum_mm="min",
            maximum_mm="max",
        )
        .reset_index()
    )

    return summary


# Identify every method tied for the smallest error in each run.
#
# Output:
#   One row per run containing its minimum distance, number of winning
#   methods, and the names of all winning methods.
def identify_run_winners(
    table,
    tie_tolerance,
):
    rows = []

    for (subject, run), group in table.groupby(
        ["subject", "run"],
        sort=True,
        observed=True,
    ):
        distances = group[
            "localization_distance_mm"
        ].astype(float)

        minimum_distance = float(
            distances.min()
        )

        winning_mask = np.isclose(
            distances,
            minimum_distance,
            atol=tie_tolerance,
            rtol=0,
        )

        winning_methods = (
            group.loc[winning_mask, "method"]
            .astype(str)
            .tolist()
        )

        rows.append({
            "subject": subject,
            "run": run,
            "minimum_distance_mm": minimum_distance,
            "number_of_winners": len(winning_methods),
            "is_tie": len(winning_methods) > 1,
            "winning_methods": ", ".join(
                winning_methods
            ),
            "MNE_winner": "MNE" in winning_methods,
            "dSPM_winner": "dSPM" in winning_methods,
            "sLORETA_winner": (
                "sLORETA" in winning_methods
            ),
            "eLORETA_winner": (
                "eLORETA" in winning_methods
            ),
        })

    return pd.DataFrame(rows)


# Compare two methods using the same runs.
#
# A negative difference means method A produced a smaller localization
# error. A positive difference means method B produced a smaller error.
#
# Output:
#   One summary row describing the paired differences between two
#   methods.
def compare_method_pair(
    wide_table,
    method_a,
    method_b,
    tie_tolerance,
):
    difference = (
        wide_table[method_a]
        - wide_table[method_b]
    )

    method_a_better = int(
        (difference < -tie_tolerance).sum()
    )

    method_b_better = int(
        (difference > tie_tolerance).sum()
    )

    ties = int(
        np.isclose(
            difference,
            0,
            atol=tie_tolerance,
            rtol=0,
        ).sum()
    )

    return {
        "method_a": method_a,
        "method_b": method_b,
        "runs": len(difference),
        "mean_a_minus_b_mm": float(
            difference.mean()
        ),
        "median_a_minus_b_mm": float(
            difference.median()
        ),
        "standard_deviation_mm": float(
            difference.std()
        ),
        "minimum_difference_mm": float(
            difference.min()
        ),
        "maximum_difference_mm": float(
            difference.max()
        ),
        "method_a_better_runs": method_a_better,
        "method_b_better_runs": method_b_better,
        "tied_runs": ties,
    }


# Calculate comparisons for every pair of inverse methods.
#
# Output:
#   One row for each method pair with paired differences and counts of
#   which method produced the smaller error.
def calculate_pairwise_differences(
    table,
    tie_tolerance,
):
    wide_table = table.pivot(
        index=["subject", "run"],
        columns="method",
        values="localization_distance_mm",
    )

    methods = [
        method
        for method in METHOD_ORDER
        if method in wide_table.columns
    ]

    rows = []

    for method_a, method_b in itertools.combinations(
        methods,
        2,
    ):
        rows.append(
            compare_method_pair(
                wide_table=wide_table,
                method_a=method_a,
                method_b=method_b,
                tie_tolerance=tie_tolerance,
            )
        )

    return pd.DataFrame(rows)


# Identify unusually large errors separately within each method.
#
# An outlier is defined as a value above the third quartile plus
# 1.5 times the interquartile range. This is a descriptive flag, not a
# reason to delete the run.
#
# Output:
#   Rows whose localization errors are unusually large relative to the
#   other results from the same method.
def identify_outliers(table):
    outlier_tables = []

    for method, group in table.groupby(
        "method",
        observed=True,
    ):
        distances = group[
            "localization_distance_mm"
        ]

        first_quartile = distances.quantile(0.25)
        third_quartile = distances.quantile(0.75)

        interquartile_range = (
            third_quartile - first_quartile
        )

        upper_limit = (
            third_quartile
            + 1.5 * interquartile_range
        )

        method_outliers = group.loc[
            distances > upper_limit
        ].copy()

        method_outliers[
            "outlier_upper_limit_mm"
        ] = upper_limit

        outlier_tables.append(
            method_outliers
        )

    if not outlier_tables:
        return pd.DataFrame()

    outliers = pd.concat(
        outlier_tables,
        ignore_index=True,
    )

    selected_columns = [
        column
        for column in [
            "subject",
            "run",
            "method",
            "stimulation_pair",
            "hemisphere",
            "localization_distance_mm",
            "nearest_source_distance_mm",
            "geometric_excess_mm",
            "peak_time_ms",
            "outlier_upper_limit_mm",
        ]
        if column in outliers.columns
    ]

    return outliers[selected_columns]


# Plot the complete distribution of localization errors.
#
# The box shows the median and middle half of the results. Each dot is
# one observed run, so unusually large errors remain visible without
# estimating a smoothed distribution shape.
#
# Output:
#   A boxplot with individual run points for each inverse method.
def create_distance_distribution_figure(
    table,
    output_file,
):
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    sns.boxplot(
        data=table,
        x="method",
        y="localization_distance_mm",
        order=METHOD_ORDER,
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        width=0.48,
        dodge=False,
        saturation=0.65,
        showfliers=False,
        legend=False,
        ax=axis,
    )

    sns.stripplot(
        data=table,
        x="method",
        y="localization_distance_mm",
        order=METHOD_ORDER,
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        #color="black",
        dodge=False,
        alpha=0.70,
        size=4,
        jitter=0.18,
        edgecolor="black",
        linewidth=1.0,
        legend=False,
        ax=axis,
    )

    axis.set_title(
        "Localization error across all Localize-MI runs"
    )
    axis.set_xlabel("Inverse method")
    axis.set_ylabel("Localization distance (mm)")
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# Plot median paired differences between all method combinations.
#
# Each cell compares two methods using the same EEG runs. A negative
# value means the reference method on that row produced the smaller
# median localization error. A positive value means the comparison
# method in that column produced the smaller median error.
#
# Output:
#   A heatmap showing the direction and size of paired differences.
def create_paired_difference_figure(
    table,
    output_file,
):
    # Place each method in a separate column while keeping participants
    # and runs aligned. This ensures that every comparison uses the same
    # recordings.
    wide_table = table.pivot(
        index=["subject", "run"],
        columns="method",
        values="localization_distance_mm",
    )

    methods = [
        method
        for method in METHOD_ORDER
        if method in wide_table.columns
    ]

    # Create a square table containing the median difference for every
    # pair of methods.
    difference_matrix = pd.DataFrame(
        index=methods,
        columns=methods,
        dtype=float,
    )

    for reference_method in methods:
        for comparison_method in methods:
            paired_differences = (
                wide_table[reference_method]
                - wide_table[comparison_method]
            )

            difference_matrix.loc[
                reference_method,
                comparison_method,
            ] = paired_differences.median()

    # Use equal colour limits around zero so that improvements in either
    # direction have comparable colour intensity.
    maximum_absolute = float(
        np.abs(
            difference_matrix.to_numpy()
        ).max()
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    sns.heatmap(
        difference_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-maximum_absolute,
        vmax=maximum_absolute,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": (
                "Median difference: reference − comparison (mm)"
            ),
            "shrink": 0.82,
        },
        ax=axis,
    )

    axis.set_title(
        "Median paired difference in localization error"
    )

    axis.set_xlabel("Comparison method")
    axis.set_ylabel("Reference method")

    # Keep method names horizontal so they can be read quickly.
    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=0,
    )

    axis.set_yticklabels(
        axis.get_yticklabels(),
        rotation=0,
    )

    # Explain how the sign of a cell should be interpreted.
    figure.text(
        0.5,
        0.02,
        (
            "Negative values favour the reference method; "
            "positive values favour the comparison method."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4d4d4d",
    )

    # Reserve space below the axes for the interpretation note.
    figure.tight_layout(
        rect=[0, 0.06, 1, 1]
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# Plot each participant's median localization error by method.
#
# Output:
#   A grouped bar chart showing whether method performance changes
#   between participants.
def create_participant_median_figure(
    participant_summary,
    output_file,
):
    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    sns.barplot(
        data=participant_summary,
        x="subject",
        y="median_mm",
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        errorbar=None,
        ax=axis,
    )

    axis.set_title(
        "Median localization error by participant"
    )
    axis.set_xlabel("Participant")
    axis.set_ylabel("Median localization distance (mm)")
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        title="Inverse method",
        frameon=True,
    )

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# Print a short summary that can be checked immediately in the terminal.
#
# Output:
#   Dataset counts, method summaries, winner counts, and saved paths.
def print_terminal_summary(
    table,
    method_summary,
    run_winners,
    output_directory,
):
    unique_runs = table[
        ["subject", "run"]
    ].drop_duplicates()

    sole_winners = run_winners.loc[
        ~run_winners["is_tie"]
    ]

    print()
    print("INVERSE-METHOD COMPARISON")
    print("-------------------------")
    print(
        f"Participants        : "
        f"{table['subject'].nunique()}"
    )
    print(
        f"Runs                : {len(unique_runs)}"
    )
    print(
        f"Run-method results  : {len(table)}"
    )

    print()
    print("LOCALIZATION DISTANCE BY METHOD")
    print("-------------------------------")

    display_summary = method_summary[
        [
            "method",
            "mean_mm",
            "median_mm",
            "standard_deviation_mm",
            "minimum_mm",
            "maximum_mm",
        ]
    ].copy()

    numeric_columns = display_summary.select_dtypes(
        include="number"
    ).columns

    display_summary[numeric_columns] = (
        display_summary[numeric_columns]
        .round(2)
    )

    print(
        display_summary.to_string(index=False)
    )

    print()
    print("RUN WINNERS")
    print("-----------")
    print(
        "Runs with one winner : "
        f"{(~run_winners['is_tie']).sum()}"
    )
    print(
        "Runs with ties       : "
        f"{run_winners['is_tie'].sum()}"
    )

    sole_counts = (
        sole_winners["winning_methods"]
        .value_counts()
        .reindex(
            METHOD_ORDER,
            fill_value=0,
        )
    )

    print()
    print("Sole wins:")
    print(sole_counts.to_string())

    print()
    print(
        f"Saved outputs       : {output_directory}"
    )


# Run the complete descriptive comparison.
#
# Output:
#   Five CSV tables, three figures, and a terminal summary.
def main():
    arguments = parse_arguments()

    input_file = (
        arguments.input
        .expanduser()
        .resolve()
    )

    output_directory = (
        arguments.output_dir
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    table = load_and_validate_results(
        input_file
    )

    method_summary = summarize_methods(
        table
    )

    participant_summary = summarize_participants(
        table
    )

    run_winners = identify_run_winners(
        table=table,
        tie_tolerance=arguments.tie_tolerance,
    )

    pairwise_differences = (
        calculate_pairwise_differences(
            table=table,
            tie_tolerance=arguments.tie_tolerance,
        )
    )

    outlier_runs = identify_outliers(
        table
    )

    method_summary.to_csv(
        output_directory
        / "method_summary.csv",
        index=False,
    )

    participant_summary.to_csv(
        output_directory
        / "participant_summary.csv",
        index=False,
    )

    run_winners.to_csv(
        output_directory
        / "run_winners.csv",
        index=False,
    )

    pairwise_differences.to_csv(
        output_directory
        / "pairwise_differences.csv",
        index=False,
    )

    outlier_runs.to_csv(
        output_directory
        / "outlier_runs.csv",
        index=False,
    )

    create_distance_distribution_figure(
        table=table,
        output_file=(
            output_directory
            / "01_distance_distribution.png"
        ),
    )

    create_paired_difference_figure(
        table=table,
        output_file=(
            output_directory
            / "02_paired_differences.png"
        ),
    )

    create_participant_median_figure(
        participant_summary=participant_summary,
        output_file=(
            output_directory
            / "03_participant_medians.png"
        ),
    )

    print_terminal_summary(
        table=table,
        method_summary=method_summary,
        run_winners=run_winners,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()
