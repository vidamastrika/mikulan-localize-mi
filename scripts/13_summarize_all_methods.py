#!/usr/bin/env python3
"""Summarize the fixed-configuration baseline for eight inverse methods.

Script 13 combines the standard-method results from Script 05 with the
additional-method results from Script 12.  It does not rerun localization and
does not select parameters.  All outputs are explicitly labelled as baseline
results because the additional-method parameter pilot and grid will follow in
later scripts.

Methods
-------
MNE, dSPM, sLORETA, eLORETA, ECD-grid, LCMV, MxNE, and irMxNE.

Default inputs
--------------
outputs/tables/inverse_method_results.csv
outputs/additional_methods/additional_method_results.csv

Default output
--------------
outputs/all_method_baseline_summary/
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

STANDARD_METHODS = ("MNE", "dSPM", "sLORETA", "eLORETA")
ADDITIONAL_METHODS = ("ECD-grid", "LCMV", "MxNE", "irMxNE")
METHOD_ORDER = STANDARD_METHODS + ADDITIONAL_METHODS

METHOD_COLORS = {
    "MNE": "#377eb8",
    "dSPM": "#e41a1c",
    "sLORETA": "#4daf4a",
    "eLORETA": "#984ea3",
    "ECD-grid": "#ff7f00",
    "LCMV": "#00a6a6",
    "MxNE": "#a65628",
    "irMxNE": "#f781bf",
}

REQUIRED_COLUMNS = {
    "subject", "run", "method", "status", "peak_time_ms",
    "localization_distance_mm", "nearest_source_distance_mm",
    "geometric_excess_mm", "estimated_x_m", "estimated_y_m",
    "estimated_z_m", "known_x_m", "known_y_m", "known_z_m",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-results", type=Path,
        default=PROJECT / "outputs/tables/inverse_method_results.csv",
        help="Script 05 results.",
    )
    parser.add_argument(
        "--additional-results", type=Path,
        default=(PROJECT / "outputs/additional_methods/"
                 "additional_method_results.csv"),
        help="Script 12 results.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT / "outputs/all_method_baseline_summary",
    )
    parser.add_argument(
        "--tie-tolerance", type=float, default=1e-6,
        help="Distances within this many millimetres are tied.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def validate_args(args):
    if args.tie_tolerance < 0:
        raise ValueError("--tie-tolerance must be non-negative.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")


def read_result_table(path, expected_methods, source_name):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} results not found: {path}")
    table = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(
            f"{source_name} results lack required columns: {sorted(missing)}"
        )
    duplicates = table.duplicated(["subject", "run", "method"])
    if duplicates.any():
        rows = table.loc[duplicates, ["subject", "run", "method"]]
        raise ValueError(
            f"{source_name} results contain duplicate solutions:\n"
            f"{rows.to_string(index=False)}"
        )
    observed = set(table["method"].astype(str))
    missing_methods = set(expected_methods) - observed
    unexpected = observed - set(expected_methods)
    if missing_methods or unexpected:
        raise ValueError(
            f"{source_name} method mismatch. Missing: "
            f"{sorted(missing_methods)}; unexpected: {sorted(unexpected)}"
        )
    table = table.loc[table["method"].isin(expected_methods)].copy()
    table["result_source"] = source_name
    if "montage" not in table:
        table["montage"] = "all_good"
    else:
        table["montage"] = table["montage"].fillna("all_good")
    if "n_good_channels" not in table and "good_channels" in table:
        table["n_good_channels"] = table["good_channels"]
    return table


def compare_run_sets(standard, additional):
    standard_runs = set(map(tuple, standard[["subject", "run"]].drop_duplicates().values))
    additional_runs = set(map(tuple, additional[["subject", "run"]].drop_duplicates().values))
    if standard_runs != additional_runs:
        only_standard = sorted(standard_runs - additional_runs)
        only_additional = sorted(additional_runs - standard_runs)
        raise ValueError(
            "The two inputs do not contain the same runs. "
            f"Only standard: {only_standard}; only additional: {only_additional}"
        )
    return sorted(standard_runs)


def check_run_geometry(table):
    """Confirm all methods use the same known geometry within every run."""
    columns = (
        "known_x_m", "known_y_m", "known_z_m",
        "nearest_source_distance_mm",
    )
    problems = []
    for (subject, run), group in table.groupby(["subject", "run"], sort=True):
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy()
            if len(values) != len(group) or not np.allclose(
                values, values[0], atol=1e-9, rtol=0
            ):
                problems.append((subject, run, column))
    if problems:
        raise ValueError(
            "Known-source geometry differs between methods: "
            + ", ".join("/".join(problem) for problem in problems[:20])
        )


def load_and_validate(args):
    standard = read_result_table(
        args.standard_results, STANDARD_METHODS, "Script 05"
    )
    additional = read_result_table(
        args.additional_results, ADDITIONAL_METHODS, "Script 12"
    )
    runs = compare_run_sets(standard, additional)
    table = pd.concat([standard, additional], ignore_index=True, sort=False)
    table["method"] = pd.Categorical(
        table["method"], categories=METHOD_ORDER, ordered=True
    )
    table = table.sort_values(["subject", "run", "method"]).reset_index(drop=True)
    if table.duplicated(["subject", "run", "method"]).any():
        raise ValueError("Combined results contain duplicate run-method rows.")
    check_run_geometry(table)
    return table, runs


def successful(table):
    result = table.loc[table["status"].astype(str).eq("PASS")].copy()
    result["localization_distance_mm"] = pd.to_numeric(
        result["localization_distance_mm"], errors="coerce"
    )
    result["peak_time_ms"] = pd.to_numeric(
        result["peak_time_ms"], errors="coerce"
    )
    if result[["localization_distance_mm", "peak_time_ms"]].isna().any().any():
        raise ValueError("Successful rows contain missing error or peak-time values.")
    return result


def method_summary(table):
    rows = []
    for method in METHOD_ORDER:
        all_rows = table.loc[table["method"].astype(str).eq(method)]
        passed = all_rows.loc[all_rows["status"].astype(str).eq("PASS")]
        values = pd.to_numeric(passed["localization_distance_mm"], errors="coerce")
        q1 = float(values.quantile(0.25)) if len(values) else np.nan
        q3 = float(values.quantile(0.75)) if len(values) else np.nan
        rows.append({
            "method": method,
            "method_group": (
                "standard" if method in STANDARD_METHODS else "additional"
            ),
            "attempted_runs": len(all_rows),
            "successful_runs": len(passed),
            "failed_runs": int(len(all_rows) - len(passed)),
            "success_rate_percent": 100 * len(passed) / len(all_rows),
            "mean_mm": values.mean(),
            "standard_deviation_mm": values.std(),
            "median_mm": values.median(),
            "first_quartile_mm": q1,
            "third_quartile_mm": q3,
            "interquartile_range_mm": q3 - q1,
            "minimum_mm": values.min(),
            "maximum_mm": values.max(),
        })
    return pd.DataFrame(rows)


def participant_summary(passed):
    return (
        passed.groupby(["subject", "method"], observed=True)
        ["localization_distance_mm"]
        .agg(
            runs="count", mean_mm="mean", median_mm="median",
            standard_deviation_mm="std", minimum_mm="min", maximum_mm="max",
        )
        .reset_index()
    )


def peak_timing_summary(passed):
    rows = []
    for method in METHOD_ORDER:
        group = passed.loc[passed["method"].astype(str).eq(method)]
        peak = group["peak_time_ms"].astype(float)
        rows.append({
            "method": method,
            "runs": len(group),
            "mean_peak_time_ms": peak.mean(),
            "median_peak_time_ms": peak.median(),
            "minimum_peak_time_ms": peak.min(),
            "maximum_peak_time_ms": peak.max(),
            "pre_zero_runs": int((peak < 0).sum()),
            "at_zero_runs": int(np.isclose(peak, 0, atol=1e-12).sum()),
            "post_zero_runs": int((peak > 0).sum()),
            "pre_zero_percent": 100 * (peak < 0).mean(),
            "at_zero_percent": 100 * np.isclose(peak, 0, atol=1e-12).mean(),
            "post_zero_percent": 100 * (peak > 0).mean(),
        })
    return pd.DataFrame(rows)


def paired_table(passed, tolerance):
    wide = passed.pivot(
        index=["subject", "run"], columns="method",
        values="localization_distance_mm",
    ).reindex(columns=METHOD_ORDER)
    rows = []
    for method_a, method_b in itertools.combinations(METHOD_ORDER, 2):
        pair = wide[[method_a, method_b]].dropna()
        difference = pair[method_a] - pair[method_b]
        ties = np.isclose(difference, 0, atol=tolerance, rtol=0)
        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "paired_runs": len(pair),
            "mean_a_minus_b_mm": difference.mean(),
            "median_a_minus_b_mm": difference.median(),
            "standard_deviation_mm": difference.std(),
            "first_quartile_mm": difference.quantile(0.25),
            "third_quartile_mm": difference.quantile(0.75),
            "method_a_better_runs": int((difference < -tolerance).sum()),
            "tied_runs": int(ties.sum()),
            "method_b_better_runs": int((difference > tolerance).sum()),
        })
    return pd.DataFrame(rows)


def run_winners(passed, tolerance):
    rows = []
    for (subject, run), group in passed.groupby(["subject", "run"], sort=True):
        minimum = float(group["localization_distance_mm"].min())
        mask = np.isclose(
            group["localization_distance_mm"], minimum,
            atol=tolerance, rtol=0,
        )
        winners = group.loc[mask, "method"].astype(str).tolist()
        row = {
            "subject": subject,
            "run": run,
            "minimum_distance_mm": minimum,
            "number_of_winners": len(winners),
            "is_tie": len(winners) > 1,
            "winning_methods": ", ".join(winners),
        }
        row.update({f"{method}_winner": method in winners for method in METHOD_ORDER})
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_winner_summary(winners):
    rows = []
    for method in METHOD_ORDER:
        flag = winners[f"{method}_winner"].astype(bool)
        weighted = (flag / winners["number_of_winners"]).sum()
        rows.append({
            "method": method,
            "runs_including_ties": int(flag.sum()),
            "weighted_wins": float(weighted),
            "weighted_win_percent": 100 * float(weighted) / len(winners),
        })
    return pd.DataFrame(rows)


def additional_vs_best_standard(passed, tolerance):
    standard = passed.loc[
        passed["method"].astype(str).isin(STANDARD_METHODS)
    ].copy()
    standard_best = (
        standard.sort_values("localization_distance_mm")
        .groupby(["subject", "run"], as_index=False, sort=True)
        .first()[["subject", "run", "method", "localization_distance_mm"]]
        .rename(columns={
            "method": "best_standard_method",
            "localization_distance_mm": "best_standard_distance_mm",
        })
    )
    additional = passed.loc[
        passed["method"].astype(str).isin(ADDITIONAL_METHODS)
    ].merge(standard_best, on=["subject", "run"], how="left")
    additional["change_from_best_standard_mm"] = (
        additional["localization_distance_mm"]
        - additional["best_standard_distance_mm"]
    )
    change = additional["change_from_best_standard_mm"]
    additional["comparison"] = np.select(
        [change < -tolerance, change > tolerance],
        ["better", "worse"], default="tied",
    )
    return additional[[
        "subject", "run", "method", "localization_distance_mm",
        "best_standard_method", "best_standard_distance_mm",
        "change_from_best_standard_mm", "comparison",
    ]]


def additional_comparison_summary(comparison):
    rows = []
    for method in ADDITIONAL_METHODS:
        group = comparison.loc[comparison["method"].astype(str).eq(method)]
        change = group["change_from_best_standard_mm"]
        rows.append({
            "method": method,
            "runs": len(group),
            "mean_change_mm": change.mean(),
            "median_change_mm": change.median(),
            "standard_deviation_mm": change.std(),
            "first_quartile_mm": change.quantile(0.25),
            "third_quartile_mm": change.quantile(0.75),
            "better_runs": int(group["comparison"].eq("better").sum()),
            "tied_runs": int(group["comparison"].eq("tied").sum()),
            "worse_runs": int(group["comparison"].eq("worse").sum()),
        })
    return pd.DataFrame(rows)


def diagnostic_summary(passed):
    rows = []
    metrics = (
        "active_sources", "ecd_goodness_of_fit_percent",
        "explained_variance_percent", "runtime_seconds",
    )
    for method in METHOD_ORDER:
        group = passed.loc[passed["method"].astype(str).eq(method)]
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "method": method,
                "metric": metric,
                "observations": len(values),
                "mean": values.mean(),
                "standard_deviation": values.std(),
                "median": values.median(),
                "minimum": values.min(),
                "maximum": values.max(),
            })
    return pd.DataFrame(rows)


def identify_outliers(passed):
    pieces = []
    for method in METHOD_ORDER:
        group = passed.loc[passed["method"].astype(str).eq(method)].copy()
        values = group["localization_distance_mm"]
        q1, q3 = values.quantile([0.25, 0.75])
        limit = q3 + 1.5 * (q3 - q1)
        selected = group.loc[values > limit].copy()
        selected["method_outlier_upper_limit_mm"] = limit
        pieces.append(selected)
    if not pieces:
        return passed.iloc[0:0].copy()
    result = pd.concat(pieces, ignore_index=True)
    columns = [
        "subject", "run", "method", "peak_time_ms",
        "localization_distance_mm", "method_outlier_upper_limit_mm",
        "nearest_source_distance_mm", "geometric_excess_mm",
    ]
    return result[columns].sort_values(
        "localization_distance_mm", ascending=False
    )


def analysis_context(passed):
    runs = passed[["subject", "run"]].drop_duplicates().shape[0]
    participants = passed["subject"].nunique()
    return (
        f"{runs} runs | {participants} participants | all-good EEG montage | "
        "Target: −2 to +2 ms\n"
        "Fixed baseline configurations; parameter-grid comparison follows later"
    )


def finish_figure(figure, output, dpi, bottom=0.12, top=0.86):
    figure.subplots_adjust(left=0.08, right=0.98, bottom=bottom, top=top)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def deterministic_strip(axis, values, position, color, seed):
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.12, 0.12, len(values))
    axis.scatter(
        position + jitter, values, s=18, color="black", alpha=0.55,
        linewidths=0, zorder=3,
    )


def error_distribution_figure(passed, output, dpi, context):
    figure, axis = plt.subplots(figsize=(13.5, 7.5))
    data = [
        passed.loc[passed["method"].astype(str).eq(method),
                   "localization_distance_mm"].to_numpy()
        for method in METHOD_ORDER
    ]
    boxes = axis.boxplot(
        data, positions=np.arange(len(METHOD_ORDER)), widths=0.58,
        patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for box, method in zip(boxes["boxes"], METHOD_ORDER):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_alpha(0.72)
    for index, (values, method) in enumerate(zip(data, METHOD_ORDER)):
        deterministic_strip(axis, values, index, METHOD_COLORS[method], 100 + index)
    axis.set_xticks(range(len(METHOD_ORDER)), METHOD_ORDER, rotation=20, ha="right")
    axis.set_ylabel("Localization error (mm)")
    axis.set_title("Localization error across eight inverse methods\n" + context)
    axis.grid(axis="y", alpha=0.25)
    finish_figure(figure, output, dpi, bottom=0.18, top=0.80)


def participant_heatmap_figure(participants, output, dpi, context):
    matrix = participants.pivot(
        index="subject", columns="method", values="median_mm"
    ).reindex(columns=METHOD_ORDER)
    figure, axis = plt.subplots(figsize=(13.5, 6.8))
    image = axis.imshow(matrix.to_numpy(), cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(METHOD_ORDER)), METHOD_ORDER, rotation=20, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            axis.text(
                column, row, f"{value:.1f}", ha="center", va="center",
                fontsize=9, color=("white" if value > 0.65 * np.nanmax(matrix) else "black"),
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Median localization error (mm)")
    axis.set_title("Participant median localization error\n" + context)
    axis.set_xlabel("Inverse method")
    axis.set_ylabel("Participant")
    finish_figure(figure, output, dpi, bottom=0.18, top=0.78)


def additional_change_figure(comparison, output, dpi, context):
    figure, axis = plt.subplots(figsize=(11.5, 7.5))
    data = [
        comparison.loc[comparison["method"].astype(str).eq(method),
                       "change_from_best_standard_mm"].to_numpy()
        for method in ADDITIONAL_METHODS
    ]
    boxes = axis.boxplot(
        data, positions=range(len(ADDITIONAL_METHODS)), widths=0.58,
        patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
    )
    for index, (box, method, values) in enumerate(
        zip(boxes["boxes"], ADDITIONAL_METHODS, data)
    ):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_alpha(0.72)
        deterministic_strip(axis, values, index, METHOD_COLORS[method], 200 + index)
    axis.axhline(0, color="#333333", linestyle="--", linewidth=1.3)
    axis.set_xticks(range(len(ADDITIONAL_METHODS)), ADDITIONAL_METHODS)
    axis.set_ylabel("Additional method − best standard method (mm)")
    axis.set_title(
        "Paired change from the best standard method in each run\n" + context
    )
    axis.text(
        0.01, 0.02, "Negative values favour the additional method",
        transform=axis.transAxes, fontsize=9, color="#444444",
    )
    axis.grid(axis="y", alpha=0.25)
    finish_figure(figure, output, dpi, bottom=0.14, top=0.80)


def peak_timing_figure(passed, output, dpi, context):
    figure, axis = plt.subplots(figsize=(13.5, 7.5))
    data = [
        passed.loc[passed["method"].astype(str).eq(method),
                   "peak_time_ms"].to_numpy()
        for method in METHOD_ORDER
    ]
    boxes = axis.boxplot(
        data, positions=range(len(METHOD_ORDER)), widths=0.58,
        patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
    )
    for index, (box, method, values) in enumerate(
        zip(boxes["boxes"], METHOD_ORDER, data)
    ):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_alpha(0.72)
        deterministic_strip(axis, values, index, METHOD_COLORS[method], 300 + index)
    axis.axhline(0, color="#b2182b", linestyle="--", linewidth=1.4,
                 label="Stimulation")
    axis.set_xticks(range(len(METHOD_ORDER)), METHOD_ORDER, rotation=20, ha="right")
    axis.set_ylabel("Peak time relative to stimulation (ms)")
    axis.set_title("Selected peak timing\n" + context)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    axis.grid(axis="y", alpha=0.25)
    finish_figure(figure, output, dpi, bottom=0.26, top=0.80)


def winner_figure(weighted, output, dpi, context):
    figure, axis = plt.subplots(figsize=(12.5, 7.2))
    x = np.arange(len(METHOD_ORDER))
    values = weighted.set_index("method").reindex(METHOD_ORDER)[
        "weighted_wins"
    ].to_numpy()
    bars = axis.bar(
        x, values, color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.82
    )
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3)
    axis.set_xticks(x, METHOD_ORDER, rotation=20, ha="right")
    axis.set_ylabel("Tie-weighted run wins")
    axis.set_title("Best method frequency across runs\n" + context)
    axis.grid(axis="y", alpha=0.25)
    finish_figure(figure, output, dpi, bottom=0.18, top=0.80)


def accuracy_stability_figure(summary, output, dpi, context):
    figure, axis = plt.subplots(figsize=(11.5, 7.5))
    for _, row in summary.iterrows():
        method = row["method"]
        axis.scatter(
            row["median_mm"], row["interquartile_range_mm"],
            s=115, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.7,
        )
    axis.set_xlabel("Median localization error (mm)")
    axis.set_ylabel("Interquartile range (mm)")
    axis.set_title("Baseline accuracy and between-run stability\n" + context)
    axis.grid(alpha=0.25)
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=METHOD_COLORS[method], markeredgecolor="black",
            label=method,
        )
        for method in METHOD_ORDER
    ]
    axis.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=4, frameon=False,
    )
    finish_figure(figure, output, dpi, bottom=0.24, top=0.80)


def save_manifest(output_dir, table, passed, summary, winners):
    payload = {
        "analysis_scope": "fixed baseline configurations only",
        "parameter_optimized": False,
        "participants": int(table["subject"].nunique()),
        "runs": int(table[["subject", "run"]].drop_duplicates().shape[0]),
        "rows": int(len(table)),
        "successful_rows": int(len(passed)),
        "failed_rows": int(len(table) - len(passed)),
        "methods": list(METHOD_ORDER),
        "runs_with_tied_winners": int(winners["is_tie"].sum()),
        "median_error_mm": {
            str(row["method"]): float(row["median_mm"])
            for _, row in summary.iterrows()
        },
    }
    path = output_dir / "baseline_summary_manifest.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def main():
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table, runs = load_and_validate(args)
    passed = successful(table)
    summary = method_summary(table)
    participants = participant_summary(passed)
    timing = peak_timing_summary(passed)
    pairs = paired_table(passed, args.tie_tolerance)
    winners = run_winners(passed, args.tie_tolerance)
    weighted = weighted_winner_summary(winners)
    comparison = additional_vs_best_standard(passed, args.tie_tolerance)
    comparison_summary = additional_comparison_summary(comparison)
    diagnostics = diagnostic_summary(passed)
    outliers = identify_outliers(passed)
    context = analysis_context(passed)

    table.to_csv(table_dir / "01_combined_baseline_results.csv", index=False)
    summary.to_csv(table_dir / "02_method_summary.csv", index=False)
    participants.to_csv(table_dir / "03_participant_summary.csv", index=False)
    pairs.to_csv(table_dir / "04_pairwise_differences.csv", index=False)
    winners.to_csv(table_dir / "05_run_winners.csv", index=False)
    weighted.to_csv(table_dir / "06_weighted_winner_summary.csv", index=False)
    timing.to_csv(table_dir / "07_peak_timing_summary.csv", index=False)
    comparison.to_csv(
        table_dir / "08_additional_vs_best_standard.csv", index=False
    )
    comparison_summary.to_csv(
        table_dir / "09_additional_vs_standard_summary.csv", index=False
    )
    diagnostics.to_csv(
        table_dir / "10_method_diagnostics_summary.csv", index=False
    )
    outliers.to_csv(table_dir / "11_outlier_runs.csv", index=False)

    error_distribution_figure(
        passed, figure_dir / "01_localization_error_distributions.png",
        args.dpi, context,
    )
    participant_heatmap_figure(
        participants, figure_dir / "02_participant_median_heatmap.png",
        args.dpi, context,
    )
    additional_change_figure(
        comparison, figure_dir / "03_additional_vs_best_standard.png",
        args.dpi, context,
    )
    peak_timing_figure(
        passed, figure_dir / "04_peak_time_distributions.png",
        args.dpi, context,
    )
    winner_figure(
        weighted, figure_dir / "05_weighted_winner_frequency.png",
        args.dpi, context,
    )
    accuracy_stability_figure(
        summary, figure_dir / "06_accuracy_stability.png",
        args.dpi, context,
    )
    save_manifest(output_dir, table, passed, summary, winners)

    print("EIGHT-METHOD BASELINE SUMMARY COMPLETE")
    print("--------------------------------------")
    print(f"Participants       : {table['subject'].nunique()}")
    print(f"Runs               : {len(runs)}")
    print(f"Methods            : {len(METHOD_ORDER)}")
    print(f"Rows               : {len(table)}")
    print(f"Successful rows    : {len(passed)}")
    print(f"Failed rows        : {len(table) - len(passed)}")
    print(f"Tied winner runs   : {int(winners['is_tie'].sum())}")
    print("Scope              : fixed baseline configurations only")
    print(f"Tables             : {table_dir}")
    print(f"Figures            : {figure_dir}")
    print("\nMedian localization error (mm):")
    print(summary[["method", "median_mm"]].to_string(index=False))


if __name__ == "__main__":
    main()
