#!/usr/bin/env python3
"""
13_summarize_all_methods.py

Summarize the matched all-run baseline localization results for eight
Localize-MI inverse methods.

INPUTS
------
Original four methods (61 runs x 4 methods):
    outputs/tables/inverse_method_results.csv

Additional methods from Script 12:
    outputs/12_additional_methods_all_runs/12_continuous_ecd_all_runs.csv
    outputs/12_additional_methods_all_runs/12_mxne_all_runs.csv
    outputs/12_additional_methods_all_runs/12_rap_music_all_runs.csv
    outputs/12_additional_methods_all_runs/12_lcmv_all_runs.csv

The script accepts alternative paths through terminal arguments.

METHODS
-------
    MNE
    dSPM
    sLORETA
    eLORETA
    Continuous-ECD
    MxNE
    RAP-MUSIC
    LCMV

IMPORTANT INTERPRETATION
------------------------
This script compares ONE baseline result per method per stimulation run.
It does NOT compare the 97,600 Script-10 parameter-grid solutions against
the 61 Script-12 baseline solutions.

All method comparisons are paired by subject + run. By default, the script
requires a complete 8-method set for every included run.

For every method the script reports BOTH:
    * mean +/- sample SD
    * median [Q1, Q3]

The median/IQR are especially useful when localization-error distributions
contain large outliers.

"Lowest-error method" means only the method with the smallest observed
localization error for that particular run. It is descriptive and should
not be interpreted as a universally superior method.

TERMINAL EXAMPLES
-----------------

Run with the default project paths:

    python scripts/13_summarize_all_methods.py --overwrite

Use explicit input files:

    python scripts/13_summarize_all_methods.py \
        --inverse-results outputs/tables/inverse_method_results.csv \
        --ecd-results outputs/12_additional_methods_all_runs/12_continuous_ecd_all_runs.csv \
        --mxne-results outputs/12_additional_methods_all_runs/12_mxne_all_runs.csv \
        --rap-music-results outputs/12_additional_methods_all_runs/12_rap_music_all_runs.csv \
        --lcmv-results outputs/12_additional_methods_all_runs/12_lcmv_all_runs.csv \
        --overwrite

Allow incomplete method/run combinations instead of requiring all eight
methods for every run:

    python scripts/13_summarize_all_methods.py \
        --allow-incomplete --overwrite

Change the output directory:

    python scripts/13_summarize_all_methods.py \
        --output-root outputs/13_my_summary --overwrite

DEFAULT OUTPUT
--------------
    outputs/13_additional_methods_summary/

Tables:
    01_method_summary.csv
    02_pairwise_differences.csv
    03_participant_summary.csv
    04_lowest_error_method_per_run.csv
    05_lowest_error_counts.csv
    06_outlier_runs.csv
    07_matched_results_long.csv
    08_matched_results_wide.csv

Figures:
    figures/01_localization_error_distribution.png
    figures/02_pairwise_median_differences.png
    figures/03_median_error_by_participant.png
    figures/04_lowest_error_counts.png

Figure 1:
    Overall localization-error distributions for all eight methods.

Figure 2:
    Median paired error difference for every method pair.
    Difference = Method A error - Method B error.
    Negative -> A has lower median paired error.
    Positive -> B has lower median paired error.

Figure 3:
    Median localization error for each participant and method.

Figure 4:
    Number of runs on which each method has the lowest observed error.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

DEFAULT_INVERSE = PROJECT / "outputs" / "tables" / "inverse_method_results.csv"
DEFAULT_SCRIPT12 = PROJECT / "outputs" / "12_additional_methods_all_runs"
DEFAULT_OUTPUT = PROJECT / "outputs" / "13_additional_methods_summary"

METHOD_ORDER = [
    "MNE",
    "dSPM",
    "sLORETA",
    "eLORETA",
    "Continuous-ECD",
    "MxNE",
    "RAP-MUSIC",
    "LCMV",
]

ORIGINAL_METHODS = {"MNE", "dSPM", "sLORETA", "eLORETA"}

DISPLAY_LABELS = {
    "MNE": "MNE",
    "dSPM": "dSPM",
    "sLORETA": "sLORETA",
    "eLORETA": "eLORETA",
    "Continuous-ECD": "Continuous-ECD",
    "MxNE": "MxNE",
    "RAP-MUSIC": "RAP-MUSIC",
    "LCMV": "LCMV",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--inverse-results", type=Path, default=DEFAULT_INVERSE)
    parser.add_argument(
        "--ecd-results",
        type=Path,
        default=DEFAULT_SCRIPT12 / "12_continuous_ecd_all_runs.csv",
    )
    parser.add_argument(
        "--mxne-results",
        type=Path,
        default=DEFAULT_SCRIPT12 / "12_mxne_all_runs.csv",
    )
    parser.add_argument(
        "--rap-music-results",
        type=Path,
        default=DEFAULT_SCRIPT12 / "12_rap_music_all_runs.csv",
    )
    parser.add_argument(
        "--lcmv-results",
        type=Path,
        default=DEFAULT_SCRIPT12 / "12_lcmv_all_runs.csv",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Keep runs lacking one or more of the eight methods.",
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="Per-method upper outlier threshold: Q3 + k*IQR. Default: 1.5.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def require_columns(table, columns, path):
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def first_existing(columns, candidates, description, path):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not identify {description} in {path}. "
        f"Tried: {', '.join(candidates)}"
    )


def clean_status(table):
    if "status" not in table.columns:
        return table.copy(), 0

    status = table["status"].astype(str).str.upper()
    failed = int((status != "PASS").sum())
    return table.loc[status == "PASS"].copy(), failed


def standardize_original(path):
    table = pd.read_csv(path)
    require_columns(table, ["subject", "run", "method"], path)

    error_column = first_existing(
        table.columns,
        [
            "localization_distance_mm",
            "selected_localization_error_mm",
            "localization_error_mm",
            "distance_mm",
        ],
        "localization-error column",
        path,
    )

    table, failed = clean_status(table)
    table = table.loc[table["method"].isin(ORIGINAL_METHODS)].copy()

    result = pd.DataFrame(
        {
            "subject": table["subject"].astype(str),
            "run": table["run"].astype(str),
            "method": table["method"].astype(str),
            "localization_error_mm": pd.to_numeric(
                table[error_column], errors="coerce"
            ),
            "source_file": str(path),
        }
    )

    return result, failed


def standardize_additional(path, expected_method):
    table = pd.read_csv(path)
    require_columns(table, ["subject", "run"], path)

    error_column = first_existing(
        table.columns,
        [
            "selected_localization_error_mm",
            "localization_distance_mm",
            "localization_error_mm",
            "distance_mm",
        ],
        "localization-error column",
        path,
    )

    table, failed = clean_status(table)

    if "method" in table.columns:
        observed = set(table["method"].dropna().astype(str))
        if observed and observed != {expected_method}:
            raise ValueError(
                f"{path} was expected to contain only {expected_method}, "
                f"but contains: {sorted(observed)}"
            )

    result = pd.DataFrame(
        {
            "subject": table["subject"].astype(str),
            "run": table["run"].astype(str),
            "method": expected_method,
            "localization_error_mm": pd.to_numeric(
                table[error_column], errors="coerce"
            ),
            "source_file": str(path),
        }
    )

    return result, failed


def validate_long_table(table, allow_incomplete):
    if table["localization_error_mm"].isna().any():
        bad = table.loc[
            table["localization_error_mm"].isna(),
            ["subject", "run", "method"],
        ]
        raise ValueError(
            "Missing/non-numeric localization errors were found:\n"
            + bad.to_string(index=False)
        )

    if (table["localization_error_mm"] < 0).any():
        raise ValueError("Localization error cannot be negative.")

    duplicated = table.duplicated(["subject", "run", "method"], keep=False)
    if duplicated.any():
        bad = table.loc[
            duplicated,
            ["subject", "run", "method", "localization_error_mm"],
        ].sort_values(["subject", "run", "method"])
        raise ValueError(
            "Duplicate subject/run/method results were found:\n"
            + bad.to_string(index=False)
        )

    counts = (
        table.groupby(["subject", "run"])["method"]
        .nunique()
        .sort_values()
    )

    if not allow_incomplete and not (counts == len(METHOD_ORDER)).all():
        bad = counts.loc[counts != len(METHOD_ORDER)]
        raise ValueError(
            "The comparison is not fully matched. These runs do not contain "
            f"all {len(METHOD_ORDER)} methods:\n{bad.to_string()}\n"
            "Use --allow-incomplete only if this is intentional."
        )

    if not allow_incomplete:
        observed_methods = set(table["method"])
        expected_methods = set(METHOD_ORDER)
        if observed_methods != expected_methods:
            raise ValueError(
                "Method set mismatch. "
                f"Expected {sorted(expected_methods)}, "
                f"found {sorted(observed_methods)}."
            )

    return counts


def method_summary(table):
    rows = []

    for method in METHOD_ORDER:
        values = (
            table.loc[
                table["method"] == method,
                "localization_error_mm",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            continue

        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])

        rows.append(
            {
                "method": method,
                "runs": len(values),
                "mean_mm": float(np.mean(values)),
                "standard_deviation_mm": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else np.nan
                ),
                "median_mm": float(median),
                "q25_mm": float(q25),
                "q75_mm": float(q75),
                "iqr_mm": float(q75 - q25),
                "minimum_mm": float(np.min(values)),
                "maximum_mm": float(np.max(values)),
            }
        )

    return pd.DataFrame(rows)


def pairwise_summary(wide):
    rows = []

    for method_a, method_b in combinations(METHOD_ORDER, 2):
        pair = wide[[method_a, method_b]].dropna()

        if pair.empty:
            continue

        differences = (
            pair[method_a].to_numpy(dtype=float)
            - pair[method_b].to_numpy(dtype=float)
        )

        q25, median, q75 = np.quantile(
            differences,
            [0.25, 0.50, 0.75],
        )

        lower_a = int(np.sum(differences < 0))
        lower_b = int(np.sum(differences > 0))
        ties = int(np.sum(np.isclose(differences, 0.0)))

        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "paired_runs": len(differences),
                "mean_a_minus_b_mm": float(np.mean(differences)),
                "standard_deviation_a_minus_b_mm": (
                    float(np.std(differences, ddof=1))
                    if len(differences) > 1
                    else np.nan
                ),
                "median_a_minus_b_mm": float(median),
                "q25_a_minus_b_mm": float(q25),
                "q75_a_minus_b_mm": float(q75),
                "method_a_lower_error_runs": lower_a,
                "method_b_lower_error_runs": lower_b,
                "ties": ties,
            }
        )

    return pd.DataFrame(rows)


def participant_summary(table):
    grouped = (
        table.groupby(["subject", "method"], as_index=False)
        .agg(
            runs=("localization_error_mm", "size"),
            mean_mm=("localization_error_mm", "mean"),
            standard_deviation_mm=("localization_error_mm", "std"),
            median_mm=("localization_error_mm", "median"),
            minimum_mm=("localization_error_mm", "min"),
            maximum_mm=("localization_error_mm", "max"),
        )
    )

    grouped["method"] = pd.Categorical(
        grouped["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return grouped.sort_values(["subject", "method"]).reset_index(drop=True)


def lowest_error_tables(wide):
    rows = []

    for (subject, run), values in wide.iterrows():
        available = values.dropna()

        if available.empty:
            continue

        minimum = float(available.min())
        methods = [
            method
            for method, value in available.items()
            if np.isclose(float(value), minimum)
        ]

        rows.append(
            {
                "subject": subject,
                "run": run,
                "lowest_error_mm": minimum,
                "lowest_error_method": ";".join(methods),
                "tie": len(methods) > 1,
                "methods_available": len(available),
            }
        )

    per_run = pd.DataFrame(rows)

    count_rows = []
    for method in METHOD_ORDER:
        sole = 0
        tied = 0

        for methods_text in per_run["lowest_error_method"]:
            methods = methods_text.split(";")
            if method not in methods:
                continue
            if len(methods) == 1:
                sole += 1
            else:
                tied += 1

        count_rows.append(
            {
                "method": method,
                "sole_lowest_error_runs": sole,
                "tied_lowest_error_runs": tied,
                "total_lowest_error_appearances": sole + tied,
            }
        )

    counts = pd.DataFrame(count_rows)

    return per_run, counts


def outlier_table(table, multiplier):
    rows = []

    for method in METHOD_ORDER:
        subset = table.loc[table["method"] == method].copy()

        if subset.empty:
            continue

        q25 = float(subset["localization_error_mm"].quantile(0.25))
        q75 = float(subset["localization_error_mm"].quantile(0.75))
        iqr = q75 - q25
        threshold = q75 + multiplier * iqr

        flagged = subset.loc[
            subset["localization_error_mm"] > threshold
        ].copy()

        for record in flagged.itertuples(index=False):
            rows.append(
                {
                    "subject": record.subject,
                    "run": record.run,
                    "method": method,
                    "localization_error_mm": float(
                        record.localization_error_mm
                    ),
                    "q25_mm": q25,
                    "q75_mm": q75,
                    "iqr_mm": iqr,
                    "upper_outlier_threshold_mm": threshold,
                    "iqr_multiplier": multiplier,
                }
            )

    return pd.DataFrame(rows)


def prepare_output(output_root, overwrite):
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_root}\n"
                "Use --overwrite to replace Script 13 outputs."
            )
        shutil.rmtree(output_root)

    (output_root / "figures").mkdir(parents=True, exist_ok=True)


def save_figure_1(table, path, dpi):
    data = [
        table.loc[
            table["method"] == method,
            "localization_error_mm",
        ].to_numpy(dtype=float)
        for method in METHOD_ORDER
    ]

    figure, axis = plt.subplots(figsize=(12, 6))

    axis.violinplot(
        data,
        positions=np.arange(1, len(METHOD_ORDER) + 1),
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    axis.boxplot(
        data,
        positions=np.arange(1, len(METHOD_ORDER) + 1),
        widths=0.18,
        showfliers=True,
    )

    axis.set_xticks(
        np.arange(1, len(METHOD_ORDER) + 1),
        [DISPLAY_LABELS[m] for m in METHOD_ORDER],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Localization error (mm)")
    axis.set_title("Localization error across matched Localize-MI runs")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_figure_2(pairwise, path, dpi):
    plot = pairwise.copy()

    labels = (
        plot["method_a"].astype(str)
        + " − "
        + plot["method_b"].astype(str)
    )

    values = plot["median_a_minus_b_mm"].to_numpy(dtype=float)

    height = max(7, 0.30 * len(plot))
    figure, axis = plt.subplots(figsize=(10, height))

    y = np.arange(len(plot))
    axis.barh(y, values)
    axis.axvline(0, linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Median paired difference (mm): Method A − Method B")
    axis.set_title("Pairwise median localization-error differences")
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_figure_3(participant, path, dpi):
    pivot = participant.pivot(
        index="subject",
        columns="method",
        values="median_mm",
    ).reindex(columns=METHOD_ORDER)

    figure, axis = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pivot.index))

    for method in METHOD_ORDER:
        if method in pivot.columns:
            axis.plot(
                x,
                pivot[method].to_numpy(dtype=float),
                marker="o",
                label=DISPLAY_LABELS[method],
            )

    axis.set_xticks(x, pivot.index)
    axis.set_ylabel("Median localization error (mm)")
    axis.set_xlabel("Participant")
    axis.set_title("Median localization error by participant")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_figure_4(counts, path, dpi):
    plot = counts.set_index("method").reindex(METHOD_ORDER)

    figure, axis = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(METHOD_ORDER))
    values = plot["sole_lowest_error_runs"].to_numpy(dtype=float)

    axis.bar(x, values)
    axis.set_xticks(
        x,
        [DISPLAY_LABELS[m] for m in METHOD_ORDER],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Runs")
    axis.set_title("Method with the lowest observed localization error per run")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()

    if args.outlier_iqr_multiplier < 0:
        raise ValueError("--outlier-iqr-multiplier must be non-negative.")

    inputs = {
        "inverse": args.inverse_results.expanduser().resolve(),
        "ecd": args.ecd_results.expanduser().resolve(),
        "mxne": args.mxne_results.expanduser().resolve(),
        "rap_music": args.rap_music_results.expanduser().resolve(),
        "lcmv": args.lcmv_results.expanduser().resolve(),
    }

    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.expanduser().resolve()
    prepare_output(output_root, args.overwrite)

    original, original_failed = standardize_original(inputs["inverse"])
    ecd, ecd_failed = standardize_additional(
        inputs["ecd"], "Continuous-ECD"
    )
    mxne, mxne_failed = standardize_additional(inputs["mxne"], "MxNE")
    rap_music, rap_failed = standardize_additional(
        inputs["rap_music"], "RAP-MUSIC"
    )
    lcmv, lcmv_failed = standardize_additional(inputs["lcmv"], "LCMV")

    long = pd.concat(
        [original, ecd, mxne, rap_music, lcmv],
        ignore_index=True,
    )

    long["method"] = pd.Categorical(
        long["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    long = long.sort_values(
        ["subject", "run", "method"]
    ).reset_index(drop=True)

    counts = validate_long_table(long, args.allow_incomplete)

    if args.allow_incomplete:
        complete_keys = counts.loc[
            counts == len(METHOD_ORDER)
        ].index
        complete_index = pd.MultiIndex.from_tuples(
            complete_keys,
            names=["subject", "run"],
        )
        long_index = pd.MultiIndex.from_frame(long[["subject", "run"]])
        paired_long = long.loc[long_index.isin(complete_index)].copy()
    else:
        paired_long = long.copy()

    wide = (
        paired_long.pivot(
            index=["subject", "run"],
            columns="method",
            values="localization_error_mm",
        )
        .reindex(columns=METHOD_ORDER)
        .sort_index()
    )

    summary = method_summary(paired_long)
    pairwise = pairwise_summary(wide)
    participants = participant_summary(paired_long)
    per_run, lowest_counts = lowest_error_tables(wide)
    outliers = outlier_table(
        paired_long,
        args.outlier_iqr_multiplier,
    )

    long.to_csv(
        output_root / "07_matched_results_long.csv",
        index=False,
    )

    wide.reset_index().to_csv(
        output_root / "08_matched_results_wide.csv",
        index=False,
    )

    summary.to_csv(
        output_root / "01_method_summary.csv",
        index=False,
    )

    pairwise.to_csv(
        output_root / "02_pairwise_differences.csv",
        index=False,
    )

    participants.to_csv(
        output_root / "03_participant_summary.csv",
        index=False,
    )

    per_run.to_csv(
        output_root / "04_lowest_error_method_per_run.csv",
        index=False,
    )

    lowest_counts.to_csv(
        output_root / "05_lowest_error_counts.csv",
        index=False,
    )

    outliers.to_csv(
        output_root / "06_outlier_runs.csv",
        index=False,
    )

    figures = output_root / "figures"

    save_figure_1(
        paired_long,
        figures / "01_localization_error_distribution.png",
        args.dpi,
    )

    save_figure_2(
        pairwise,
        figures / "02_pairwise_median_differences.png",
        args.dpi,
    )

    save_figure_3(
        participants,
        figures / "03_median_error_by_participant.png",
        args.dpi,
    )

    save_figure_4(
        lowest_counts,
        figures / "04_lowest_error_counts.png",
        args.dpi,
    )

    print("=" * 78)
    print("SCRIPT 13 — ALL-METHOD BASELINE SUMMARY")
    print("=" * 78)
    print(f"Methods                 : {len(METHOD_ORDER)}")
    print(f"Rows loaded             : {len(long)}")
    print(f"Unique runs             : {counts.size}")
    print(
        "Complete 8-method runs : "
        f"{int((counts == len(METHOD_ORDER)).sum())}"
    )
    print(f"Paired rows analyzed    : {len(paired_long)}")
    print(
        "Filtered FAIL rows     : "
        f"{original_failed + ecd_failed + mxne_failed + rap_failed + lcmv_failed}"
    )
    print(f"Output                  : {output_root}")

    print("\nMETHOD SUMMARY")
    print(
        summary[
            [
                "method",
                "runs",
                "mean_mm",
                "standard_deviation_mm",
                "median_mm",
                "q25_mm",
                "q75_mm",
                "minimum_mm",
                "maximum_mm",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nLOWEST-ERROR COUNTS")
    print(lowest_counts.to_string(index=False))

    print("\nFigures:")
    for path in sorted(figures.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
