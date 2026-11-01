#!/usr/bin/env python3
"""Explore the full inverse-parameter grid created by Script 09.

The script treats runs, rather than individual parameter solutions, as the
independent observational units. Main-effect summaries first collapse nuisance
parameter combinations within each subject/run/method/factor level and then
summarize those paired run-level values across runs.

This is an exploratory sensitivity analysis. A configuration selected as
"best" on these same 61 runs is descriptive and should not be presented as an
independently validated optimum.

Example
-------
Run from the repository root:

    python scripts/10_explore_parameter_grid.py \
        --input outputs/grid_all_runs/grid_results.csv \
        --output-dir outputs/parameter_grid_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = ["MNE", "dSPM", "sLORETA", "eLORETA"]
MONTAGE_ORDER = ["all_good", "128", "64", "32"]
METHOD_COLORS = {
    "MNE": "#377EB8",
    "dSPM": "#E41A1C",
    "sLORETA": "#4DAF4A",
    "eLORETA": "#984EA3",
}
AUTHOR_LEVELS = {0.1, 0.5, 1.0}

REQUIRED_COLUMNS = {
    "subject",
    "run",
    "method",
    "method_origin",
    "montage",
    "channels",
    "replacement_count",
    "loose",
    "depth",
    "snr",
    "lambda2",
    "target_tmin_s",
    "target_tmax_s",
    "peak_time_ms",
    "localization_distance_mm",
    "nearest_source_distance_mm",
    "geometric_excess_mm",
    "script05_baseline_mm",
    "change_from_baseline_mm",
    "status",
}

GRID_KEY = [
    "subject",
    "run",
    "method",
    "montage",
    "loose",
    "depth",
    "snr",
]

CONFIGURATION_KEY = [
    "method",
    "method_origin",
    "montage",
    "loose",
    "depth",
    "snr",
    "lambda2",
]


def parse_arguments():
    """Read input, output, and reporting options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "grid_all_runs" / "grid_results.csv",
        help="Complete grid_results.csv created by Script 09.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "parameter_grid_analysis",
        help="Directory for Script 10 tables and figures.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of lowest-median configurations retained per method.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of saved figures.",
    )
    return parser.parse_args()


def compact_number(value):
    """Format a numeric parameter without unnecessary trailing zeros."""

    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def signed_milliseconds(seconds):
    """Format seconds as signed milliseconds."""

    milliseconds = float(seconds) * 1000
    if np.isclose(milliseconds, 0):
        milliseconds = 0.0
    return f"{milliseconds:+g}"


def quantile_25(values):
    return values.quantile(0.25)


def quantile_75(values):
    return values.quantile(0.75)


quantile_25.__name__ = "q25"
quantile_75.__name__ = "q75"


def load_and_validate_grid(path):
    """Load Script 09 output and validate completeness and uniqueness."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Script 09 result table was not found: {path}")

    table = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"Script 09 result table is missing columns: {missing}")

    non_pass = table.loc[table["status"].astype(str) != "PASS"]
    if not non_pass.empty:
        examples = non_pass[
            ["subject", "run", "method", "montage", "loose", "depth", "snr", "status"]
        ].head()
        raise ValueError(
            "Script 10 requires successful grid rows only. Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    if table.duplicated(GRID_KEY).any():
        count = int(table.duplicated(GRID_KEY).sum())
        raise ValueError(f"The grid contains {count} duplicate parameter rows.")

    numeric_columns = [
        "channels",
        "replacement_count",
        "loose",
        "depth",
        "snr",
        "lambda2",
        "target_tmin_s",
        "target_tmax_s",
        "peak_time_ms",
        "localization_distance_mm",
        "nearest_source_distance_mm",
        "geometric_excess_mm",
        "script05_baseline_mm",
        "change_from_baseline_mm",
    ]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    essential_numeric = [
        "loose",
        "depth",
        "snr",
        "peak_time_ms",
        "localization_distance_mm",
        "script05_baseline_mm",
        "change_from_baseline_mm",
    ]
    if table[essential_numeric].isna().any().any():
        missing_counts = table[essential_numeric].isna().sum()
        missing_counts = missing_counts.loc[missing_counts > 0]
        raise ValueError(
            "Essential numeric values are missing:\n"
            f"{missing_counts.to_string()}"
        )

    expected_methods = [method for method in METHOD_ORDER if method in set(table["method"])]
    unexpected_methods = sorted(set(table["method"]) - set(METHOD_ORDER))
    if unexpected_methods:
        raise ValueError(f"Unexpected inverse methods: {unexpected_methods}")
    if not expected_methods:
        raise ValueError("No recognized inverse methods were found.")

    table["method"] = pd.Categorical(
        table["method"], categories=expected_methods, ordered=True
    )
    montage_levels = [level for level in MONTAGE_ORDER if level in set(table["montage"])]
    montage_levels += sorted(set(table["montage"]) - set(montage_levels))
    table["montage"] = pd.Categorical(
        table["montage"], categories=montage_levels, ordered=True
    )

    counts = table.groupby(["subject", "run"], observed=True).size()
    if counts.nunique() != 1:
        raise ValueError(
            "Runs do not contain equal numbers of grid solutions:\n"
            f"{counts.describe().to_string()}"
        )

    return table.sort_values(GRID_KEY).reset_index(drop=True)


def create_analysis_context(table):
    """Create a concise description used in figure titles."""

    runs = table[["subject", "run"]].drop_duplicates().shape[0]
    participants = table["subject"].nunique()
    target_start = table["target_tmin_s"].dropna().unique()
    target_stop = table["target_tmax_s"].dropna().unique()
    text = f"{runs} runs | {participants} participants | {len(table):,} solutions"
    if len(target_start) == 1 and len(target_stop) == 1:
        text += (
            f" | Target: {signed_milliseconds(target_start[0])} to "
            f"{signed_milliseconds(target_stop[0])} ms"
        )
    return text


def create_quality_summary(table):
    """Create a compact audit table describing grid coverage."""

    run_count = table[["subject", "run"]].drop_duplicates().shape[0]
    rows_per_run = table.groupby(["subject", "run"], observed=True).size()
    return pd.DataFrame([
        {
            "rows": len(table),
            "participants": table["subject"].nunique(),
            "runs": run_count,
            "methods": table["method"].nunique(),
            "montages": table["montage"].nunique(),
            "loose_levels": table["loose"].nunique(),
            "depth_levels": table["depth"].nunique(),
            "snr_levels": table["snr"].nunique(),
            "rows_per_run": int(rows_per_run.iloc[0]),
            "duplicate_grid_keys": int(table.duplicated(GRID_KEY).sum()),
            "failed_rows": int((table["status"].astype(str) != "PASS").sum()),
            "missing_localization_distances": int(
                table["localization_distance_mm"].isna().sum()
            ),
        }
    ])


def create_method_summary(table):
    """Summarize solution-level errors and run-level typical errors."""

    solution_summary = (
        table.groupby("method", observed=True)["localization_distance_mm"]
        .agg(
            solutions="size",
            solution_mean_mm="mean",
            solution_median_mm="median",
            solution_standard_deviation_mm="std",
            solution_q25_mm=quantile_25,
            solution_q75_mm=quantile_75,
            solution_minimum_mm="min",
            solution_maximum_mm="max",
        )
        .reset_index()
    )

    run_typical = (
        table.groupby(["subject", "run", "method"], observed=True)
        .agg(
            run_median_mm=("localization_distance_mm", "median"),
            run_mean_mm=("localization_distance_mm", "mean"),
        )
        .reset_index()
    )
    run_summary = (
        run_typical.groupby("method", observed=True)
        .agg(
            runs=("run_median_mm", "size"),
            mean_of_run_medians_mm=("run_median_mm", "mean"),
            median_of_run_medians_mm=("run_median_mm", "median"),
            standard_deviation_of_run_medians_mm=("run_median_mm", "std"),
            mean_of_run_means_mm=("run_mean_mm", "mean"),
            median_of_run_means_mm=("run_mean_mm", "median"),
        )
        .reset_index()
    )
    return solution_summary.merge(run_summary, on="method", validate="one_to_one")


def create_factor_level_summary(table):
    """Summarize paired run-level main effects for each varied factor."""

    records = []
    for factor in ["montage", "loose", "depth", "snr"]:
        run_level = (
            table.groupby(
                ["subject", "run", "method", factor], observed=True
            )
            .agg(
                run_median_mm=("localization_distance_mm", "median"),
                run_mean_mm=("localization_distance_mm", "mean"),
                run_median_change_from_baseline_mm=(
                    "change_from_baseline_mm", "median"
                ),
            )
            .reset_index()
        )
        summary = (
            run_level.groupby(["method", factor], observed=True)
            .agg(
                runs=("run_median_mm", "size"),
                mean_of_run_medians_mm=("run_median_mm", "mean"),
                median_of_run_medians_mm=("run_median_mm", "median"),
                standard_deviation_of_run_medians_mm=("run_median_mm", "std"),
                q25_of_run_medians_mm=("run_median_mm", quantile_25),
                q75_of_run_medians_mm=("run_median_mm", quantile_75),
                mean_of_run_means_mm=("run_mean_mm", "mean"),
                median_change_from_baseline_mm=(
                    "run_median_change_from_baseline_mm", "median"
                ),
            )
            .reset_index()
            .rename(columns={factor: "level"})
        )
        summary.insert(1, "factor", factor)
        summary["level"] = summary["level"].astype(str)
        records.append(summary)
    return pd.concat(records, ignore_index=True)


def create_configuration_summary(table):
    """Summarize every full parameter combination across runs."""

    summary = (
        table.groupby(CONFIGURATION_KEY, observed=True)
        .agg(
            runs=("localization_distance_mm", "size"),
            mean_localization_distance_mm=("localization_distance_mm", "mean"),
            median_localization_distance_mm=("localization_distance_mm", "median"),
            standard_deviation_mm=("localization_distance_mm", "std"),
            q25_mm=("localization_distance_mm", quantile_25),
            q75_mm=("localization_distance_mm", quantile_75),
            minimum_mm=("localization_distance_mm", "min"),
            maximum_mm=("localization_distance_mm", "max"),
            mean_change_from_baseline_mm=("change_from_baseline_mm", "mean"),
            median_change_from_baseline_mm=("change_from_baseline_mm", "median"),
            improvement_rate=("change_from_baseline_mm", lambda x: (x < 0).mean()),
            pre_zero_peak_rate=("peak_time_ms", lambda x: (x < 0).mean()),
            immediate_peak_rate=(
                "peak_time_ms", lambda x: ((x >= -1) & (x <= 1)).mean()
            ),
            median_peak_time_ms=("peak_time_ms", "median"),
            median_channels=("channels", "median"),
            mean_replacement_count=("replacement_count", "mean"),
        )
        .reset_index()
    )
    summary["author_loose_depth_levels"] = (
        summary["loose"].round(10).isin(AUTHOR_LEVELS)
        & summary["depth"].round(10).isin(AUTHOR_LEVELS)
    )
    summary["rank_by_median_within_method"] = (
        summary.groupby("method", observed=True)["median_localization_distance_mm"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    summary["rank_by_mean_within_method"] = (
        summary.groupby("method", observed=True)["mean_localization_distance_mm"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return summary.sort_values(
        ["method", "rank_by_median_within_method", "rank_by_mean_within_method"]
    ).reset_index(drop=True)


def create_best_configuration_tables(configuration_summary, top_n):
    """Return top full-grid and author-level configurations per method."""

    top_full = (
        configuration_summary.sort_values(
            ["method", "median_localization_distance_mm", "mean_localization_distance_mm"]
        )
        .groupby("method", observed=True, group_keys=False)
        .head(top_n)
        .copy()
    )
    top_full["reported_rank"] = (
        top_full.groupby("method", observed=True).cumcount() + 1
    )

    author = configuration_summary.loc[
        configuration_summary["author_loose_depth_levels"]
    ].copy()
    top_author = (
        author.sort_values(
            ["method", "median_localization_distance_mm", "mean_localization_distance_mm"]
        )
        .groupby("method", observed=True, group_keys=False)
        .head(top_n)
        .copy()
    )
    top_author["reported_rank"] = (
        top_author.groupby("method", observed=True).cumcount() + 1
    )
    return top_full, top_author


def classify_peak_time(values):
    """Classify peak times relative to the immediate stimulation interval."""

    return pd.cut(
        values,
        bins=[-np.inf, -1.0, 0.0, 1.0, np.inf],
        right=False,
        labels=["before -1 ms", "-1 to <0 ms", "0 to <1 ms", "+1 ms or later"],
    )


def create_peak_timing_summary(table):
    """Summarize where grid solutions place their maximum in time."""

    working = table[["method", "montage", "peak_time_ms"]].copy()
    working["peak_interval"] = classify_peak_time(working["peak_time_ms"])
    summary = (
        working.groupby(
            ["method", "montage", "peak_interval"], observed=False
        )
        .size()
        .rename("solutions")
        .reset_index()
    )
    totals = summary.groupby(["method", "montage"], observed=True)["solutions"].transform("sum")
    summary["percentage"] = summary["solutions"] / totals * 100
    return summary


def create_winner_frequency_summary(table):
    """Calculate tie-aware parameter frequencies among run-level minima.

    Every subject-run-method contributes total weight one. When several full
    configurations share the minimum error, that weight is divided equally
    across the tied configurations before parameter levels are counted.
    """

    working = table.copy()
    minimum_by_run = working.groupby(
        ["subject", "run", "method"], observed=True
    )["localization_distance_mm"].transform("min")
    winners = working.loc[
        np.isclose(
            working["localization_distance_mm"],
            minimum_by_run,
            rtol=0,
            atol=1e-9,
        )
    ].copy()
    tie_sizes = winners.groupby(
        ["subject", "run", "method"], observed=True
    )["localization_distance_mm"].transform("size")
    winners["tie_weight"] = 1.0 / tie_sizes

    records = []
    run_count_by_method = (
        table[["subject", "run", "method"]]
        .drop_duplicates()
        .groupby("method", observed=True)
        .size()
    )
    for factor in ["montage", "loose", "depth", "snr"]:
        summary = (
            winners.groupby(["method", factor], observed=True)["tie_weight"]
            .sum()
            .rename("weighted_wins")
            .reset_index()
            .rename(columns={factor: "level"})
        )
        summary.insert(1, "factor", factor)
        summary["runs"] = summary["method"].map(run_count_by_method).astype(int)
        summary["weighted_percentage"] = (
            summary["weighted_wins"] / summary["runs"] * 100
        )
        summary["level"] = summary["level"].astype(str)
        records.append(summary)
    return pd.concat(records, ignore_index=True)


def run_level_typical_errors(table):
    """Return one typical grid error per run and method."""

    return (
        table.groupby(["subject", "run", "method"], observed=True)
        .agg(
            median_grid_error_mm=("localization_distance_mm", "median"),
            mean_grid_error_mm=("localization_distance_mm", "mean"),
        )
        .reset_index()
    )


def select_best_configuration_rows(table, configuration_summary):
    """Select rows belonging to the lowest-median configuration per method."""

    best = (
        configuration_summary.sort_values(
            ["method", "median_localization_distance_mm", "mean_localization_distance_mm"]
        )
        .groupby("method", observed=True, group_keys=False)
        .head(1)[CONFIGURATION_KEY]
        .copy()
    )
    selected = table.merge(best, on=CONFIGURATION_KEY, how="inner", validate="many_to_one")
    return selected


def add_context_title(axis, title, context):
    axis.set_title(f"{title}\n{context}", pad=14)


def save_method_distribution_figure(table, output_file, dpi, context):
    """Plot run-level typical errors for each inverse method."""

    run_level = run_level_typical_errors(table)
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=run_level,
        x="method",
        y="median_grid_error_mm",
        order=METHOD_ORDER,
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        dodge=False,
        width=0.50,
        showfliers=False,
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=run_level,
        x="method",
        y="median_grid_error_mm",
        order=METHOD_ORDER,
        color="black",
        alpha=0.55,
        size=3.5,
        jitter=0.16,
        ax=axis,
    )
    add_context_title(axis, "Typical localization error across the parameter grid", context)
    axis.set_xlabel("Inverse method")
    axis.set_ylabel("Within-run median localization distance (mm)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_main_effects_figure(factor_summary, output_file, dpi, context):
    """Plot mean and median run-level errors across each factor level."""

    factors = [
        ("montage", "EEG montage"),
        ("loose", "Loose orientation constraint"),
        ("depth", "Depth weighting"),
        ("snr", "Signal-to-noise ratio"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))

    for axis, (factor, title) in zip(axes.flat, factors):
        subset = factor_summary.loc[factor_summary["factor"] == factor].copy()
        if factor == "montage":
            ordered_levels = [level for level in MONTAGE_ORDER if level in set(subset["level"])]
            x_positions = np.arange(len(ordered_levels))
            level_to_x = dict(zip(ordered_levels, x_positions))
            subset["x"] = subset["level"].map(level_to_x)
            axis.set_xticks(x_positions, ordered_levels)
        else:
            subset["x"] = pd.to_numeric(subset["level"])

        for method in METHOD_ORDER:
            method_data = subset.loc[subset["method"] == method].sort_values("x")
            if method_data.empty:
                continue
            axis.plot(
                method_data["x"],
                method_data["median_of_run_medians_mm"],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.0,
            )
            axis.plot(
                method_data["x"],
                method_data["mean_of_run_medians_mm"],
                color=METHOD_COLORS[method],
                marker="x",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
        axis.set_title(title)
        axis.set_xlabel("Level")
        axis.set_ylabel("Localization distance (mm)")
        axis.grid(axis="y", alpha=0.25)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker="o", label=m)
        for m in METHOD_ORDER
    ]
    statistic_handles = [
        Line2D([0], [0], color="#333333", marker="o", label="Median", linewidth=2),
        Line2D([0], [0], color="#333333", marker="x", linestyle="--", label="Mean"),
    ]
    figure.legend(
        handles=method_handles + statistic_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(f"Marginal parameter effects\n{context}", y=0.995)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_loose_depth_heatmaps(table, output_file, dpi, context):
    """Plot marginalized loose-by-depth interactions for each method."""

    interaction = (
        table.groupby(["method", "loose", "depth"], observed=True)[
            "localization_distance_mm"
        ]
        .median()
        .rename("median_mm")
        .reset_index()
    )
    vmin = float(interaction["median_mm"].min())
    vmax = float(interaction["median_mm"].max())
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))

    for panel_index, (axis, method) in enumerate(zip(axes.flat, METHOD_ORDER)):
        subset = interaction.loc[interaction["method"] == method]
        matrix = subset.pivot(index="depth", columns="loose", values="median_mm")
        matrix = matrix.sort_index(ascending=False).sort_index(axis=1)
        sns.heatmap(
            matrix,
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 8},
            linewidths=0.3,
            linecolor="white",
            cbar=method == METHOD_ORDER[-1],
            cbar_kws={"label": "Median localization distance (mm)"},
            ax=axis,
        )
        axis.set_title(method)
        axis.set_xlabel("Loose")
        axis.set_ylabel("Depth")

    figure.suptitle(
        "Loose–depth interaction, marginalized over montage and SNR\n"
        f"{context}",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_montage_sensitivity_figure(table, output_file, dpi, context):
    """Plot paired run-level montage sensitivity after nuisance collapsing."""

    run_level = (
        table.groupby(["subject", "run", "method", "montage"], observed=True)
        .agg(run_median_mm=("localization_distance_mm", "median"))
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(13, 8))
    sns.boxplot(
        data=run_level,
        x="montage",
        y="run_median_mm",
        order=MONTAGE_ORDER,
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        showfliers=False,
        width=0.72,
        ax=axis,
    )
    add_context_title(axis, "Sensitivity to EEG montage size", context)
    axis.set_xlabel("EEG montage")
    axis.set_ylabel("Within-run median localization distance (mm)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        title="Inverse method",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_best_vs_baseline_figure(
    table, configuration_summary, output_file, dpi, context
):
    """Plot paired change from baseline for the selected best configuration."""

    selected = select_best_configuration_rows(table, configuration_summary)
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=selected,
        x="method",
        y="change_from_baseline_mm",
        order=METHOD_ORDER,
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        dodge=False,
        width=0.5,
        showfliers=False,
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=selected,
        x="method",
        y="change_from_baseline_mm",
        order=METHOD_ORDER,
        color="black",
        alpha=0.55,
        size=3.5,
        jitter=0.16,
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=1)
    add_context_title(
        axis,
        "Selected best full-grid configuration versus Script 05 baseline",
        context,
    )
    axis.set_xlabel("Inverse method")
    axis.set_ylabel("Localization-distance change (mm)")
    axis.text(
        0.01,
        0.02,
        "Negative values indicate a smaller error than baseline. Selection and evaluation use the same runs.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4d4d4d",
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_peak_timing_figure(table, output_file, dpi, context):
    """Plot the proportion of solutions peaking in four temporal intervals."""

    working = table[["method", "peak_time_ms"]].copy()
    working["peak_interval"] = classify_peak_time(working["peak_time_ms"])
    counts = (
        working.groupby(["method", "peak_interval"], observed=False)
        .size()
        .rename("solutions")
        .reset_index()
    )
    counts["percentage"] = (
        counts["solutions"]
        / counts.groupby("method", observed=True)["solutions"].transform("sum")
        * 100
    )
    matrix = counts.pivot(index="method", columns="peak_interval", values="percentage")
    matrix = matrix.reindex(METHOD_ORDER)
    colors = ["#4575B4", "#91BFDB", "#FC8D59", "#D73027"]
    figure, axis = plt.subplots(figsize=(11, 7))
    matrix.plot(kind="bar", stacked=True, color=colors, width=0.65, ax=axis)
    add_context_title(axis, "Timing of the maximum source estimate", context)
    axis.set_xlabel("Inverse method")
    axis.set_ylabel("Grid solutions (%)")
    axis.set_ylim(0, 100)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=0)
    axis.legend(
        title="Peak interval",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_top_configurations_figure(
    top_configurations, output_file, dpi, context
):
    """Compare mean and median error for leading configurations."""

    montage_colors = {
        "all_good": "#4C78A8",
        "128": "#59A14F",
        "64": "#F28E2B",
        "32": "#E15759",
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 13))
    for panel_index, (axis, method) in enumerate(zip(axes.flat, METHOD_ORDER)):
        subset = top_configurations.loc[
            top_configurations["method"] == method
        ].sort_values(
            ["median_localization_distance_mm", "mean_localization_distance_mm"],
            ascending=False,
        )
        if subset.empty:
            axis.set_visible(False)
            continue
        labels = [
            (
                f"{row.montage} | L={compact_number(row.loose)}, "
                f"D={compact_number(row.depth)}, SNR={compact_number(row.snr)}"
            )
            for row in subset.itertuples()
        ]
        y = np.arange(len(subset))
        for position, (_, row) in enumerate(subset.iterrows()):
            axis.plot(
                [row["median_localization_distance_mm"], row["mean_localization_distance_mm"]],
                [position, position],
                color="#B0B0B0",
                linewidth=1.2,
                zorder=1,
            )
        axis.scatter(
            subset["median_localization_distance_mm"],
            y,
            marker="o",
            s=55,
            color=[montage_colors.get(str(x), "#777777") for x in subset["montage"]],
            edgecolor="black",
            linewidth=0.4,
            label="Median",
            zorder=3,
        )
        axis.scatter(
            subset["mean_localization_distance_mm"],
            y,
            marker="x",
            s=55,
            color="black",
            linewidth=1.4,
            label="Mean",
            zorder=4,
        )
        axis.set_yticks(y, labels)
        axis.set_title(method)
        axis.set_xlabel("Localization distance (mm)")
        axis.set_ylabel("")
        axis.grid(axis="x", alpha=0.25)

    legend_handles = [
        Patch(facecolor=color, label=montage)
        for montage, color in montage_colors.items()
    ] + [
        Line2D([0], [0], marker="o", color="black", linestyle="none", label="Median"),
        Line2D([0], [0], marker="x", color="black", linestyle="none", label="Mean"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        frameon=False,
        title="Montage and summary statistic",
    )
    figure.suptitle(
        f"Lowest-median full-grid configurations\n{context}", y=0.995
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_accuracy_stability_figure(
    configuration_summary, output_file, dpi, context
):
    """Plot typical error against across-run interquartile spread."""

    working = configuration_summary.copy()
    working["interquartile_range_mm"] = working["q75_mm"] - working["q25_mm"]
    montage_colors = {
        "all_good": "#4C78A8",
        "128": "#59A14F",
        "64": "#F28E2B",
        "32": "#E15759",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    for panel_index, (axis, method) in enumerate(zip(axes.flat, METHOD_ORDER)):
        subset = working.loc[working["method"] == method]
        for montage in MONTAGE_ORDER:
            points = subset.loc[subset["montage"].astype(str) == montage]
            axis.scatter(
                points["median_localization_distance_mm"],
                points["interquartile_range_mm"],
                s=18,
                alpha=0.40,
                color=montage_colors[montage],
                edgecolor="none",
            )
        best = subset.sort_values(
            ["median_localization_distance_mm", "interquartile_range_mm"]
        ).iloc[0]
        axis.scatter(
            best["median_localization_distance_mm"],
            best["interquartile_range_mm"],
            marker="*",
            s=190,
            color="#FFD92F",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        axis.set_title(method)
        axis.set_xlabel("Median localization distance (mm)")
        axis.set_ylabel("")
        axis.grid(alpha=0.25)

    handles = [
        Patch(facecolor=montage_colors[m], label=m) for m in MONTAGE_ORDER
    ] + [
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor="#FFD92F",
            markeredgecolor="black", markersize=13, label="Lowest-median configuration"
        )
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        title="EEG montage",
    )
    figure.supylabel("Interquartile range across runs (mm)", x=0.015)
    figure.suptitle(
        "Configuration accuracy and stability across runs\n"
        f"{context}\nLower-left indicates smaller typical error and less variability",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.90))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_winner_frequency_figure(
    winner_summary, output_file, dpi, context
):
    """Plot tie-aware parameter frequencies among run-level minima."""

    factors = [
        ("montage", "EEG montage"),
        ("loose", "Loose"),
        ("depth", "Depth"),
        ("snr", "SNR"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    for panel_index, (axis, (factor, title)) in enumerate(
        zip(axes.flat, factors)
    ):
        subset = winner_summary.loc[winner_summary["factor"] == factor].copy()
        if factor == "montage":
            level_order = [x for x in MONTAGE_ORDER if x in set(subset["level"])]
        else:
            level_order = [
                str(x)
                for x in sorted(pd.to_numeric(subset["level"]).unique())
            ]
        sns.barplot(
            data=subset,
            x="level",
            y="weighted_percentage",
            order=level_order,
            hue="method",
            hue_order=METHOD_ORDER,
            palette=METHOD_COLORS,
            errorbar=None,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Level")
        axis.set_ylabel("")
        axis.grid(axis="y", alpha=0.25)
        if axis.legend_ is not None:
            axis.legend_.remove()

    handles = [Patch(facecolor=METHOD_COLORS[m], label=m) for m in METHOD_ORDER]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        title="Inverse method",
    )
    figure.supylabel("Weighted winner share (%)", x=0.015)
    figure.suptitle(
        "Parameter levels among run-level minimum-error configurations\n"
        f"{context}\nTied configurations divide each run-method's weight equally",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.90))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_improvement_consistency_figure(
    configuration_summary, output_file, dpi, context
):
    """Plot median baseline change against the proportion of improved runs."""

    montage_colors = {
        "all_good": "#4C78A8",
        "128": "#59A14F",
        "64": "#F28E2B",
        "32": "#E15759",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    for panel_index, (axis, method) in enumerate(zip(axes.flat, METHOD_ORDER)):
        subset = configuration_summary.loc[
            configuration_summary["method"] == method
        ]
        for montage in MONTAGE_ORDER:
            points = subset.loc[subset["montage"].astype(str) == montage]
            axis.scatter(
                points["median_change_from_baseline_mm"],
                points["improvement_rate"] * 100,
                s=18,
                alpha=0.40,
                color=montage_colors[montage],
                edgecolor="none",
            )
        axis.axvline(0, color="black", linewidth=0.9)
        axis.axhline(50, color="#666666", linewidth=0.8, linestyle="--")
        axis.set_title(method)
        axis.set_xlabel("Median change from baseline (mm)")
        axis.set_ylabel("Runs improved (%)" if panel_index % 2 == 0 else "")
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)

    handles = [Patch(facecolor=montage_colors[m], label=m) for m in MONTAGE_ORDER]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        title="EEG montage",
    )
    figure.suptitle(
        "Configuration improvement relative to the Script 05 baseline\n"
        f"{context}\nMore reliable improvements appear toward the upper-left",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.90))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def print_summary(table, configuration_summary, output_dir):
    """Print a concise completion and best-configuration summary."""

    best = (
        configuration_summary.sort_values(
            ["method", "median_localization_distance_mm", "mean_localization_distance_mm"]
        )
        .groupby("method", observed=True, group_keys=False)
        .head(1)
    )
    columns = [
        "method",
        "montage",
        "loose",
        "depth",
        "snr",
        "median_localization_distance_mm",
        "mean_localization_distance_mm",
        "median_change_from_baseline_mm",
        "improvement_rate",
    ]
    display = best[columns].copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(3)

    print()
    print("PARAMETER-GRID EXPLORATION COMPLETE")
    print("-----------------------------------")
    print(f"Participants        : {table['subject'].nunique()}")
    print(f"Runs                : {table[['subject', 'run']].drop_duplicates().shape[0]}")
    print(f"Grid rows           : {len(table):,}")
    print(f"Configurations      : {len(configuration_summary):,}")
    print()
    print("Lowest-median configuration within each method")
    print(display.to_string(index=False))
    print()
    print("Caution: configurations were selected and evaluated on the same runs.")
    print(f"Saved outputs       : {output_dir}")


def main():
    """Run validation, tabular summaries, and exploratory figures."""

    args = parse_arguments()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")

    input_file = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    table = load_and_validate_grid(input_file)
    context = create_analysis_context(table)

    quality_summary = create_quality_summary(table)
    method_summary = create_method_summary(table)
    factor_summary = create_factor_level_summary(table)
    configuration_summary = create_configuration_summary(table)
    top_full, top_author = create_best_configuration_tables(
        configuration_summary, args.top_n
    )
    peak_summary = create_peak_timing_summary(table)
    winner_summary = create_winner_frequency_summary(table)

    quality_summary.to_csv(table_dir / "01_grid_quality_summary.csv", index=False)
    method_summary.to_csv(table_dir / "02_method_summary.csv", index=False)
    factor_summary.to_csv(table_dir / "03_factor_level_summary.csv", index=False)
    configuration_summary.to_csv(
        table_dir / "04_configuration_summary.csv", index=False
    )
    top_full.to_csv(table_dir / "05_top_full_grid_configurations.csv", index=False)
    top_author.to_csv(
        table_dir / "06_top_author_level_configurations.csv", index=False
    )
    peak_summary.to_csv(table_dir / "07_peak_timing_summary.csv", index=False)
    winner_summary.to_csv(
        table_dir / "08_run_winner_parameter_frequencies.csv", index=False
    )

    save_method_distribution_figure(
        table, figure_dir / "01_method_distribution.png", args.dpi, context
    )
    save_main_effects_figure(
        factor_summary, figure_dir / "02_parameter_main_effects.png", args.dpi, context
    )
    save_loose_depth_heatmaps(
        table, figure_dir / "03_loose_depth_interaction.png", args.dpi, context
    )
    save_montage_sensitivity_figure(
        table, figure_dir / "04_montage_sensitivity.png", args.dpi, context
    )
    save_best_vs_baseline_figure(
        table,
        configuration_summary,
        figure_dir / "05_best_configuration_vs_baseline.png",
        args.dpi,
        context,
    )
    save_peak_timing_figure(
        table, figure_dir / "06_peak_timing.png", args.dpi, context
    )
    save_top_configurations_figure(
        top_full,
        figure_dir / "07_top_configurations.png",
        args.dpi,
        context,
    )
    save_accuracy_stability_figure(
        configuration_summary,
        figure_dir / "08_accuracy_stability.png",
        args.dpi,
        context,
    )
    save_winner_frequency_figure(
        winner_summary,
        figure_dir / "09_run_winner_parameter_frequencies.png",
        args.dpi,
        context,
    )
    save_improvement_consistency_figure(
        configuration_summary,
        figure_dir / "10_improvement_consistency.png",
        args.dpi,
        context,
    )

    print_summary(table, configuration_summary, output_dir)


if __name__ == "__main__":
    main()
