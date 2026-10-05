#!/usr/bin/env python3
"""Analyze the full Script-15 grids for the four additional inverse methods.

This script is the analysis companion to ``15_full_additional_method_grid.py``.
It borrows the run-level analysis philosophy from Script 10: subject-runs, not
individual grid solutions, are treated as the observational units when
summarizing marginal parameter effects. Nuisance parameter combinations are
first collapsed within each run and factor level, then the resulting run-level
values are summarized across runs.

Scientific design
-----------------
The Script-15 grids have deliberately unequal sizes:

    Continuous-ECD   6 configurations x 4 montages x 61 runs
    MxNE           300 configurations x 4 montages x 61 runs
    RAP-MUSIC        1 configuration  x 4 montages x 61 runs
    LCMV            45 configurations x 4 montages x 61 runs

Therefore raw solution rows must NOT be treated as independent observations
when comparing factors or methods. Script 16 keeps solution-level summaries for
quality control, but parameter-effect summaries use run-level collapsing.

Failures
--------
Failed Script-15 rows are retained and summarized. They are NOT silently
removed from the quality analysis. Localization-error summaries use successful
solutions only, while success/failure rates are reported separately.

For configuration selection, the default rule is conservative: a configuration
must have a 100% solution success rate across the included runs to be eligible
for the descriptive "lowest-median observed configuration". This prevents a
configuration from looking artificially accurate because difficult failed runs
are absent from its localization-error median. The threshold can be changed
with ``--min-selection-success-rate``.

Selection caution
-----------------
The lowest-median configuration is selected and evaluated on the same runs.
It is descriptive and must NOT be presented as an independently validated
optimum.

Default input root
------------------
    outputs/15_full_additional_method_grid/
        continuous_ecd/grid_results.csv
        mxne/grid_results.csv
        rap_music/grid_results.csv
        lcmv/grid_results.csv

Default output root
-------------------
    outputs/16_additional_method_grid_analysis/

Main tables
-----------
    01_grid_quality_summary.csv
    02_method_grid_summary.csv
    03_factor_level_summary.csv
    04_interaction_summary.csv
    05_configuration_summary.csv
    06_top_configurations.csv
    07_lowest_median_by_montage.csv
    08_failure_factor_summary.csv
    09_failure_details.csv
    10_selected_configurations.csv
    11_selected_config_results_long.csv
    12_selected_config_results_wide.csv
    13_selected_config_method_summary.csv
    14_selected_config_pairwise_differences.csv
    15_selected_config_participant_summary.csv
    16_selected_config_lowest_error_per_run.csv
    17_selected_config_lowest_error_counts.csv
    18_selected_config_outliers.csv
    19_factor_sensitivity_summary.csv
    20_selected_vs_script12_baseline_runs.csv
    21_selected_vs_script12_baseline_summary.csv

Figures
-------
Figures are intentionally exploratory and descriptive. Parameter-effect and
interaction figures are generated separately for each method/factor so the
unequal grid dimensions do not imply equal solution counts across methods.

Examples
--------
Run the complete analysis:

    python scripts/16_analyze_additional_method_grid.py --overwrite

Analyze only MxNE while debugging:

    python scripts/16_analyze_additional_method_grid.py \
        --method mxne --overwrite

Use an explicit MxNE result file:

    python scripts/16_analyze_additional_method_grid.py \
        --method mxne \
        --mxne-results outputs/15_full_additional_method_grid/mxne/grid_results.csv \
        --overwrite

Keep the ten lowest-median eligible configurations per method:

    python scripts/16_analyze_additional_method_grid.py \
        --top-n 10 --overwrite

Allow configurations with at least 99% successful runs to be eligible for the
selected descriptive configuration:

    python scripts/16_analyze_additional_method_grid.py \
        --min-selection-success-rate 0.99 --overwrite
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_ROOT = PROJECT / "outputs" / "15_full_additional_method_grid"
DEFAULT_OUTPUT_ROOT = PROJECT / "outputs" / "16_additional_method_grid_analysis"

METHOD_ORDER = ["Continuous-ECD", "MxNE", "RAP-MUSIC", "LCMV"]
METHOD_SLUG = {
    "Continuous-ECD": "continuous_ecd",
    "MxNE": "mxne",
    "RAP-MUSIC": "rap_music",
    "LCMV": "lcmv",
}
METHOD_ALIASES = {
    "ecd": "Continuous-ECD",
    "continuous_ecd": "Continuous-ECD",
    "continuousecd": "Continuous-ECD",
    "mxne": "MxNE",
    "rap_music": "RAP-MUSIC",
    "rapmusic": "RAP-MUSIC",
    "music": "RAP-MUSIC",
    "lcmv": "LCMV",
}
MONTAGE_ORDER = ["all_good", "128", "64", "32"]

BASE_REQUIRED_COLUMNS = {
    "subject",
    "run",
    "method",
    "configuration_id",
    "montage",
    "channels",
    "replacement_count",
    "sampling_frequency_hz",
    "target_tmin_s",
    "target_tmax_s",
    "selected_time_ms",
    "localization_distance_mm",
    "status",
    "error_type",
    "error_message",
    "runtime_seconds",
}

GRID_KEY = ["subject", "run", "method", "montage", "configuration_id"]

PARAMETERS_BY_METHOD = {
    "Continuous-ECD": ["montage", "ecd_min_dist_mm"],
    "MxNE": ["montage", "mxne_alpha", "mxne_loose", "mxne_depth"],
    "RAP-MUSIC": ["montage"],
    "LCMV": [
        "montage",
        "lcmv_reg",
        "lcmv_window_half_ms",
        "lcmv_data_covariance_method",
    ],
}

CONFIG_PARAMETER_COLUMNS = [
    "ecd_min_dist_mm",
    "mxne_alpha",
    "mxne_loose",
    "mxne_depth",
    "rap_music_n_dipoles",
    "lcmv_reg",
    "lcmv_window_half_ms",
    "lcmv_data_covariance_method",
    "lcmv_weight_norm",
]

NUMERIC_COLUMNS = [
    "channels",
    "replacement_count",
    "sampling_frequency_hz",
    "target_tmin_s",
    "target_tmax_s",
    "selected_time_ms",
    "localization_distance_mm",
    "nearest_source_distance_mm",
    "geometric_excess_mm",
    "script12_baseline_mm",
    "change_from_script12_baseline_mm",
    "runtime_seconds",
    "ecd_min_dist_mm",
    "ecd_selected_gof_percent",
    "ecd_minimum_observed_error_mm",
    "mxne_alpha",
    "mxne_loose",
    "mxne_depth",
    "mxne_active_sources",
    "mxne_explained_variance_percent",
    "rap_music_n_dipoles",
    "rap_music_gof_percent",
    "lcmv_reg",
    "lcmv_window_half_ms",
]


def normalize_method(value: str) -> str:
    key = str(value).strip().replace("-", "_").lower()
    if key in ("all", "*"):
        return "ALL"
    if key not in METHOD_ALIASES:
        raise argparse.ArgumentTypeError(
            "Choose ecd, mxne, rap_music, lcmv, or all."
        )
    return METHOD_ALIASES[key]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument(
        "--method",
        action="append",
        type=normalize_method,
        help="Repeat to analyze selected methods; default is all four.",
    )

    parser.add_argument("--ecd-results", type=Path)
    parser.add_argument("--mxne-results", type=Path)
    parser.add_argument("--rap-music-results", type=Path)
    parser.add_argument("--lcmv-results", type=Path)

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of eligible lowest-median configurations retained per method.",
    )
    parser.add_argument(
        "--min-selection-success-rate",
        type=float,
        default=1.0,
        help=(
            "Minimum successful-run fraction required for configuration selection. "
            "Default: 1.0 (all included runs must succeed)."
        ),
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="Upper outlier threshold for selected configurations: Q3 + k*IQR.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--allow-incomplete-grid",
        action="store_true",
        help="Allow missing expected rows when a Script-15 manifest is available.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def selected_methods(values):
    if not values or "ALL" in values:
        return list(METHOD_ORDER)
    return list(dict.fromkeys(values))


def quantile25(values):
    return values.quantile(0.25)


def quantile75(values):
    return values.quantile(0.75)


quantile25.__name__ = "q25"
quantile75.__name__ = "q75"


def safe_rate(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def compact_level(value):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number):
        return f"{number:.6g}"
    return str(value)


def sort_levels(values, factor):
    values = [value for value in values if not pd.isna(value)]
    if factor == "montage":
        order = [item for item in MONTAGE_ORDER if item in set(map(str, values))]
        leftovers = sorted(set(map(str, values)) - set(order))
        return order + leftovers
    if factor == "lcmv_data_covariance_method":
        preferred = ["empirical", "shrunk", "diagonal_fixed"]
        order = [item for item in preferred if item in set(map(str, values))]
        leftovers = sorted(set(map(str, values)) - set(order))
        return order + leftovers
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        return list(np.sort(numeric.unique()))
    return sorted(set(map(str, values)))


def resolve_input_paths(args, methods):
    root = args.grid_root.expanduser().resolve()
    overrides = {
        "Continuous-ECD": args.ecd_results,
        "MxNE": args.mxne_results,
        "RAP-MUSIC": args.rap_music_results,
        "LCMV": args.lcmv_results,
    }
    paths = {}
    for method in methods:
        override = overrides[method]
        if override is not None:
            path = override.expanduser().resolve()
        else:
            path = root / METHOD_SLUG[method] / "grid_results.csv"
        paths[method] = path
    return paths


def manifest_path_for(result_path):
    return result_path.with_name("grid_manifest.json")


def parse_parameters_from_configuration_id(table):
    """Fill method-specific parameter columns even when a failed row left them blank."""
    table = table.copy()
    for column in CONFIG_PARAMETER_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan

    ecd_mask = table["method"].eq("Continuous-ECD")
    for idx, cid in table.loc[ecd_mask, "configuration_id"].items():
        match = re.fullmatch(r"ecd_min_dist_([-+0-9.eE]+)", str(cid))
        if match and pd.isna(table.at[idx, "ecd_min_dist_mm"]):
            table.at[idx, "ecd_min_dist_mm"] = float(match.group(1))

    mxne_mask = table["method"].eq("MxNE")
    for idx, cid in table.loc[mxne_mask, "configuration_id"].items():
        match = re.fullmatch(
            r"mxne_a([-+0-9.eE]+)_l([-+0-9.eE]+)_d([-+0-9.eE]+)", str(cid)
        )
        if not match:
            continue
        values = [float(match.group(i)) for i in range(1, 4)]
        for column, value in zip(
            ["mxne_alpha", "mxne_loose", "mxne_depth"], values
        ):
            if pd.isna(table.at[idx, column]):
                table.at[idx, column] = value

    rap_mask = table["method"].eq("RAP-MUSIC")
    for idx, cid in table.loc[rap_mask, "configuration_id"].items():
        match = re.fullmatch(r"rap_music_n(\d+)", str(cid))
        if match and pd.isna(table.at[idx, "rap_music_n_dipoles"]):
            table.at[idx, "rap_music_n_dipoles"] = int(match.group(1))

    lcmv_mask = table["method"].eq("LCMV")
    for idx, cid in table.loc[lcmv_mask, "configuration_id"].items():
        match = re.fullmatch(
            r"lcmv_r([-+0-9.eE]+)_w([-+0-9.eE]+)_(.+)", str(cid)
        )
        if not match:
            continue
        if pd.isna(table.at[idx, "lcmv_reg"]):
            table.at[idx, "lcmv_reg"] = float(match.group(1))
        if pd.isna(table.at[idx, "lcmv_window_half_ms"]):
            table.at[idx, "lcmv_window_half_ms"] = float(match.group(2))
        current = table.at[idx, "lcmv_data_covariance_method"]
        if pd.isna(current) or str(current).strip() == "":
            table.at[idx, "lcmv_data_covariance_method"] = match.group(3)

    return table


def load_manifest(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def expected_keys_from_manifest(manifest, method):
    if not manifest:
        return None
    cases = [tuple(item) for item in manifest.get("cases", [])]
    montages = [str(item) for item in manifest.get("montages", [])]
    configs = [
        str(item.get("configuration_id"))
        for item in manifest.get("configurations", [])
        if item.get("configuration_id") is not None
    ]
    if not cases or not montages or not configs:
        return None
    return {
        (str(subject), str(run), method, montage, cid)
        for subject, run in cases
        for montage in montages
        for cid in configs
    }


def load_method_grid(path, expected_method, allow_incomplete_grid):
    if not path.is_file():
        raise FileNotFoundError(path)

    table = pd.read_csv(path, low_memory=False)
    missing = sorted(BASE_REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    observed_methods = set(table["method"].dropna().astype(str))
    if observed_methods != {expected_method}:
        raise ValueError(
            f"{path} should contain only {expected_method}; found {sorted(observed_methods)}"
        )

    for column in NUMERIC_COLUMNS:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    table["subject"] = table["subject"].astype(str)
    table["run"] = table["run"].astype(str)
    table["method"] = table["method"].astype(str)
    table["montage"] = table["montage"].astype(str)
    table["configuration_id"] = table["configuration_id"].astype(str)
    table["status"] = table["status"].astype(str).str.upper()

    invalid_status = sorted(set(table["status"]) - {"PASS", "FAIL"})
    if invalid_status:
        raise ValueError(f"Unexpected status values in {path}: {invalid_status}")

    duplicated = table.duplicated(GRID_KEY, keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, GRID_KEY].head(10)
        raise ValueError(
            f"Duplicate Script-15 keys in {path}:\n{examples.to_string(index=False)}"
        )

    table = parse_parameters_from_configuration_id(table)

    pass_rows = table["status"].eq("PASS")
    if table.loc[pass_rows, "localization_distance_mm"].isna().any():
        examples = table.loc[
            pass_rows & table["localization_distance_mm"].isna(), GRID_KEY
        ].head(10)
        raise ValueError(
            "PASS rows with missing localization distance were found:\n"
            + examples.to_string(index=False)
        )
    if (table.loc[pass_rows, "localization_distance_mm"] < 0).any():
        raise ValueError(f"Negative localization error found in {path}")

    manifest_path = manifest_path_for(path)
    manifest = load_manifest(manifest_path)
    expected_keys = expected_keys_from_manifest(manifest, expected_method)

    observed_keys = {
        tuple(row)
        for row in table[GRID_KEY].itertuples(index=False, name=None)
    }
    missing_expected = set()
    unexpected = set()
    if expected_keys is not None:
        missing_expected = expected_keys - observed_keys
        unexpected = observed_keys - expected_keys
        if unexpected:
            raise ValueError(
                f"{path} contains {len(unexpected)} rows outside its manifest."
            )
        if missing_expected and not allow_incomplete_grid:
            example = list(sorted(missing_expected))[:10]
            raise ValueError(
                f"{path} is incomplete relative to its manifest: "
                f"{len(missing_expected)} expected rows are missing. Examples: {example}. "
                "Use --allow-incomplete-grid only if intentional."
            )

    metadata = {
        "result_path": path,
        "manifest_path": manifest_path if manifest_path.is_file() else None,
        "manifest": manifest,
        "expected_rows": len(expected_keys) if expected_keys is not None else len(table),
        "missing_expected_rows": len(missing_expected),
    }

    return table.sort_values(GRID_KEY).reset_index(drop=True), metadata


def load_all_grids(paths, allow_incomplete_grid):
    tables = []
    metadata = {}
    for method, path in paths.items():
        table, info = load_method_grid(path, method, allow_incomplete_grid)
        tables.append(table)
        metadata[method] = info
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined["method"] = pd.Categorical(
        combined["method"], categories=[m for m in METHOD_ORDER if m in paths], ordered=True
    )
    combined["montage"] = pd.Categorical(
        combined["montage"], categories=MONTAGE_ORDER, ordered=True
    )
    return combined, metadata


def create_quality_summary(table, metadata):
    rows = []
    for method in [m for m in METHOD_ORDER if m in set(table["method"].astype(str))]:
        subset = table.loc[table["method"].astype(str).eq(method)]
        passes = int(subset["status"].eq("PASS").sum())
        fails = int(subset["status"].eq("FAIL").sum())
        expected_rows = int(metadata[method]["expected_rows"])
        rows.append(
            {
                "method": method,
                "participants": subset["subject"].nunique(),
                "runs": subset[["subject", "run"]].drop_duplicates().shape[0],
                "montages": subset["montage"].nunique(),
                "configurations_per_montage": subset["configuration_id"].nunique(),
                "expected_solutions": expected_rows,
                "observed_rows": len(subset),
                "pass_solutions": passes,
                "fail_solutions": fails,
                "solution_success_rate": safe_rate(passes, passes + fails),
                "missing_expected_rows": int(metadata[method]["missing_expected_rows"]),
                "duplicate_grid_keys": int(subset.duplicated(GRID_KEY).sum()),
                "missing_pass_localization_errors": int(
                    subset.loc[subset["status"].eq("PASS"), "localization_distance_mm"]
                    .isna()
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def create_method_grid_summary(table):
    records = []
    for method in [m for m in METHOD_ORDER if m in set(table["method"].astype(str))]:
        subset = table.loc[table["method"].astype(str).eq(method)].copy()
        passed = subset.loc[subset["status"].eq("PASS")].copy()
        run_level = (
            passed.groupby(["subject", "run"], observed=True)
            .agg(
                run_median_grid_error_mm=("localization_distance_mm", "median"),
                run_mean_grid_error_mm=("localization_distance_mm", "mean"),
            )
            .reset_index()
        )
        solution_values = passed["localization_distance_mm"].dropna()
        run_values = run_level["run_median_grid_error_mm"].dropna()
        records.append(
            {
                "method": method,
                "solutions_total": len(subset),
                "solutions_pass": len(passed),
                "solutions_fail": int(subset["status"].eq("FAIL").sum()),
                "solution_success_rate": safe_rate(len(passed), len(subset)),
                "solution_mean_mm": solution_values.mean(),
                "solution_standard_deviation_mm": solution_values.std(ddof=1),
                "solution_median_mm": solution_values.median(),
                "solution_q25_mm": solution_values.quantile(0.25),
                "solution_q75_mm": solution_values.quantile(0.75),
                "solution_minimum_mm": solution_values.min(),
                "solution_maximum_mm": solution_values.max(),
                "runs_with_successful_solutions": len(run_values),
                "mean_of_run_medians_mm": run_values.mean(),
                "standard_deviation_of_run_medians_mm": run_values.std(ddof=1),
                "median_of_run_medians_mm": run_values.median(),
                "q25_of_run_medians_mm": run_values.quantile(0.25),
                "q75_of_run_medians_mm": run_values.quantile(0.75),
                "mean_of_run_means_mm": run_level["run_mean_grid_error_mm"].mean(),
            }
        )
    return pd.DataFrame(records)


def factor_run_level(table, method, factor):
    subset = table.loc[table["method"].astype(str).eq(method)].copy()
    if factor not in subset.columns:
        return pd.DataFrame()
    subset = subset.loc[subset[factor].notna()].copy()
    if subset.empty:
        return pd.DataFrame()

    keys = ["subject", "run", factor]
    counts = (
        subset.groupby(keys, observed=True)
        .agg(
            total_solutions=("status", "size"),
            pass_solutions=("status", lambda x: int((x == "PASS").sum())),
            fail_solutions=("status", lambda x: int((x == "FAIL").sum())),
        )
        .reset_index()
    )

    passed = subset.loc[subset["status"].eq("PASS")]
    errors = (
        passed.groupby(keys, observed=True)
        .agg(
            run_median_mm=("localization_distance_mm", "median"),
            run_mean_mm=("localization_distance_mm", "mean"),
        )
        .reset_index()
    )
    result = counts.merge(errors, on=keys, how="left", validate="one_to_one")
    result["run_solution_success_rate"] = (
        result["pass_solutions"] / result["total_solutions"]
    )
    result["method"] = method
    result["factor"] = factor
    result = result.rename(columns={factor: "level"})
    return result


def create_factor_level_summary(table):
    records = []
    methods = [m for m in METHOD_ORDER if m in set(table["method"].astype(str))]
    for method in methods:
        for factor in PARAMETERS_BY_METHOD[method]:
            run_level = factor_run_level(table, method, factor)
            if run_level.empty:
                continue
            for level, group in run_level.groupby("level", observed=True, dropna=False):
                values = group["run_median_mm"].dropna()
                records.append(
                    {
                        "method": method,
                        "factor": factor,
                        "level": level,
                        "runs": group[["subject", "run"]].drop_duplicates().shape[0],
                        "runs_with_valid_error": len(values),
                        "total_solutions": int(group["total_solutions"].sum()),
                        "pass_solutions": int(group["pass_solutions"].sum()),
                        "fail_solutions": int(group["fail_solutions"].sum()),
                        "solution_success_rate": safe_rate(
                            group["pass_solutions"].sum(), group["total_solutions"].sum()
                        ),
                        "mean_of_run_medians_mm": values.mean(),
                        "standard_deviation_of_run_medians_mm": values.std(ddof=1),
                        "median_of_run_medians_mm": values.median(),
                        "q25_of_run_medians_mm": values.quantile(0.25),
                        "q75_of_run_medians_mm": values.quantile(0.75),
                        "mean_of_run_means_mm": group["run_mean_mm"].mean(),
                        "median_run_solution_success_rate": group[
                            "run_solution_success_rate"
                        ].median(),
                    }
                )
    result = pd.DataFrame(records)
    if not result.empty:
        result["level"] = result["level"].astype(str)
    return result


def interaction_run_level(table, method, factor_a, factor_b):
    subset = table.loc[table["method"].astype(str).eq(method)].copy()
    if factor_a not in subset.columns or factor_b not in subset.columns:
        return pd.DataFrame()
    subset = subset.loc[subset[factor_a].notna() & subset[factor_b].notna()].copy()
    if subset.empty:
        return pd.DataFrame()

    keys = ["subject", "run", factor_a, factor_b]
    counts = (
        subset.groupby(keys, observed=True)
        .agg(
            total_solutions=("status", "size"),
            pass_solutions=("status", lambda x: int((x == "PASS").sum())),
            fail_solutions=("status", lambda x: int((x == "FAIL").sum())),
        )
        .reset_index()
    )
    passed = subset.loc[subset["status"].eq("PASS")]
    errors = (
        passed.groupby(keys, observed=True)
        .agg(run_median_mm=("localization_distance_mm", "median"))
        .reset_index()
    )
    result = counts.merge(errors, on=keys, how="left", validate="one_to_one")
    result["method"] = method
    result["factor_a"] = factor_a
    result["factor_b"] = factor_b
    result = result.rename(columns={factor_a: "level_a", factor_b: "level_b"})
    return result


def create_interaction_summary(table):
    records = []
    methods = [m for m in METHOD_ORDER if m in set(table["method"].astype(str))]
    for method in methods:
        factors = PARAMETERS_BY_METHOD[method]
        for factor_a, factor_b in combinations(factors, 2):
            run_level = interaction_run_level(table, method, factor_a, factor_b)
            if run_level.empty:
                continue
            for (level_a, level_b), group in run_level.groupby(
                ["level_a", "level_b"], observed=True, dropna=False
            ):
                values = group["run_median_mm"].dropna()
                records.append(
                    {
                        "method": method,
                        "factor_a": factor_a,
                        "level_a": level_a,
                        "factor_b": factor_b,
                        "level_b": level_b,
                        "runs": group[["subject", "run"]].drop_duplicates().shape[0],
                        "runs_with_valid_error": len(values),
                        "total_solutions": int(group["total_solutions"].sum()),
                        "pass_solutions": int(group["pass_solutions"].sum()),
                        "fail_solutions": int(group["fail_solutions"].sum()),
                        "solution_success_rate": safe_rate(
                            group["pass_solutions"].sum(), group["total_solutions"].sum()
                        ),
                        "mean_of_run_medians_mm": values.mean(),
                        "median_of_run_medians_mm": values.median(),
                        "standard_deviation_of_run_medians_mm": values.std(ddof=1),
                        "q25_of_run_medians_mm": values.quantile(0.25),
                        "q75_of_run_medians_mm": values.quantile(0.75),
                    }
                )
    result = pd.DataFrame(records)
    if not result.empty:
        result["level_a"] = result["level_a"].astype(str)
        result["level_b"] = result["level_b"].astype(str)
    return result


def create_configuration_summary(table, min_selection_success_rate):
    records = []
    methods = [m for m in METHOD_ORDER if m in set(table["method"].astype(str))]
    for method in methods:
        subset = table.loc[table["method"].astype(str).eq(method)].copy()
        for (montage, cid), group in subset.groupby(
            ["montage", "configuration_id"], observed=True
        ):
            passed = group.loc[group["status"].eq("PASS")]
            values = passed["localization_distance_mm"].dropna()
            total = len(group)
            n_pass = len(passed)
            n_fail = int(group["status"].eq("FAIL").sum())
            success_rate = safe_rate(n_pass, total)
            record = {
                "method": method,
                "montage": str(montage),
                "configuration_id": cid,
                "runs_total": total,
                "runs_pass": n_pass,
                "runs_fail": n_fail,
                "success_rate": success_rate,
                "eligible_for_selection": bool(
                    success_rate >= min_selection_success_rate
                    and n_pass > 0
                    and np.isfinite(values.median())
                ),
                "mean_localization_distance_mm": values.mean(),
                "standard_deviation_mm": values.std(ddof=1),
                "median_localization_distance_mm": values.median(),
                "q25_mm": values.quantile(0.25),
                "q75_mm": values.quantile(0.75),
                "iqr_mm": values.quantile(0.75) - values.quantile(0.25),
                "minimum_mm": values.min(),
                "maximum_mm": values.max(),
                "mean_runtime_seconds": pd.to_numeric(
                    group["runtime_seconds"], errors="coerce"
                ).mean(),
            }
            for column in CONFIG_PARAMETER_COLUMNS:
                if column in group.columns:
                    non_null = group[column].dropna()
                    record[column] = non_null.iloc[0] if len(non_null) else np.nan
            records.append(record)

    summary = pd.DataFrame(records)
    if summary.empty:
        return summary

    summary["rank_by_median_within_method"] = np.nan
    summary["rank_by_mean_within_method"] = np.nan
    for method, index in summary.groupby("method").groups.items():
        eligible_index = summary.loc[index].index[
            summary.loc[index, "eligible_for_selection"].astype(bool)
        ]
        summary.loc[eligible_index, "rank_by_median_within_method"] = (
            summary.loc[eligible_index, "median_localization_distance_mm"]
            .rank(method="min", ascending=True)
        )
        summary.loc[eligible_index, "rank_by_mean_within_method"] = (
            summary.loc[eligible_index, "mean_localization_distance_mm"]
            .rank(method="min", ascending=True)
        )

    return summary.sort_values(
        ["method", "eligible_for_selection", "median_localization_distance_mm"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def create_top_configurations(configuration_summary, top_n):
    eligible = configuration_summary.loc[
        configuration_summary["eligible_for_selection"].astype(bool)
    ].copy()
    if eligible.empty:
        return eligible
    top = (
        eligible.sort_values(
            ["method", "median_localization_distance_mm", "mean_localization_distance_mm"]
        )
        .groupby("method", group_keys=False)
        .head(top_n)
        .copy()
    )
    top["reported_rank"] = top.groupby("method").cumcount() + 1
    return top


def create_lowest_median_by_montage(configuration_summary):
    eligible = configuration_summary.loc[
        configuration_summary["eligible_for_selection"].astype(bool)
    ].copy()
    if eligible.empty:
        return eligible
    result = (
        eligible.sort_values(
            [
                "method",
                "montage",
                "median_localization_distance_mm",
                "mean_localization_distance_mm",
            ]
        )
        .groupby(["method", "montage"], observed=True, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    return result


def create_failure_factor_summary(factor_summary):
    columns = [
        "method",
        "factor",
        "level",
        "runs",
        "total_solutions",
        "pass_solutions",
        "fail_solutions",
        "solution_success_rate",
    ]
    if factor_summary.empty:
        return pd.DataFrame(columns=columns)
    return factor_summary[columns].copy()


def create_failure_details(table):
    columns = [
        "subject",
        "run",
        "method",
        "montage",
        "configuration_id",
        *CONFIG_PARAMETER_COLUMNS,
        "channels",
        "replacement_count",
        "error_type",
        "error_message",
        "runtime_seconds",
    ]
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan
    return table.loc[table["status"].eq("FAIL"), columns].copy()



def create_factor_sensitivity_summary(factor_summary):
    """Rank how strongly each varied factor shifts run-level typical error.

    The range is descriptive: maximum minus minimum median-of-run-medians across
    factor levels after nuisance parameters were collapsed within run.
    """
    columns = [
        "method",
        "factor",
        "levels",
        "best_level",
        "worst_level",
        "best_median_of_run_medians_mm",
        "worst_median_of_run_medians_mm",
        "sensitivity_range_mm",
        "relative_range_percent",
        "minimum_solution_success_rate",
        "maximum_solution_success_rate",
        "total_fail_solutions",
    ]
    if factor_summary.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for (method, factor), group in factor_summary.groupby(
        ["method", "factor"], observed=True
    ):
        working = group.copy()
        working["median_value"] = pd.to_numeric(
            working["median_of_run_medians_mm"], errors="coerce"
        )
        working = working.loc[working["median_value"].notna()].copy()
        if working.empty:
            continue

        best = working.loc[working["median_value"].idxmin()]
        worst = working.loc[working["median_value"].idxmax()]
        best_value = float(best["median_value"])
        worst_value = float(worst["median_value"])
        span = worst_value - best_value
        records.append(
            {
                "method": method,
                "factor": factor,
                "levels": len(working),
                "best_level": str(best["level"]),
                "worst_level": str(worst["level"]),
                "best_median_of_run_medians_mm": best_value,
                "worst_median_of_run_medians_mm": worst_value,
                "sensitivity_range_mm": span,
                "relative_range_percent": (
                    span / best_value * 100 if best_value > 0 else np.nan
                ),
                "minimum_solution_success_rate": pd.to_numeric(
                    working["solution_success_rate"], errors="coerce"
                ).min(),
                "maximum_solution_success_rate": pd.to_numeric(
                    working["solution_success_rate"], errors="coerce"
                ).max(),
                "total_fail_solutions": int(
                    pd.to_numeric(working["fail_solutions"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=columns)
    return result.sort_values(
        ["method", "sensitivity_range_mm"],
        ascending=[True, False],
    ).reset_index(drop=True)


def extract_script12_baselines(table):
    """Return one Script-12 baseline localization error per subject-run-method."""
    if "script12_baseline_mm" not in table.columns:
        return pd.DataFrame(
            columns=["subject", "run", "method", "script12_baseline_mm"]
        )

    working = table.loc[
        pd.to_numeric(table["script12_baseline_mm"], errors="coerce").notna(),
        ["subject", "run", "method", "script12_baseline_mm"],
    ].copy()
    if working.empty:
        return working

    working["script12_baseline_mm"] = pd.to_numeric(
        working["script12_baseline_mm"], errors="coerce"
    )

    grouped = working.groupby(["subject", "run", "method"], observed=True)
    conflicts = grouped["script12_baseline_mm"].nunique(dropna=True)
    conflicts = conflicts.loc[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting Script-12 baseline values were found for subject/run/method: "
            + str(list(conflicts.index[:10]))
        )

    return (
        grouped["script12_baseline_mm"]
        .first()
        .reset_index()
        .sort_values(["subject", "run", "method"])
        .reset_index(drop=True)
    )


def compare_selected_with_script12(selected_rows, baseline_rows):
    """Create paired run-level selected-configuration versus Script-12 comparisons."""
    run_columns = [
        "subject",
        "run",
        "method",
        "selected_localization_distance_mm",
        "script12_baseline_mm",
        "selected_minus_script12_mm",
        "improved_vs_script12",
    ]
    summary_columns = [
        "method",
        "paired_runs",
        "selected_mean_mm",
        "selected_median_mm",
        "script12_mean_mm",
        "script12_median_mm",
        "mean_selected_minus_script12_mm",
        "standard_deviation_selected_minus_script12_mm",
        "median_selected_minus_script12_mm",
        "q25_selected_minus_script12_mm",
        "q75_selected_minus_script12_mm",
        "improved_runs",
        "worsened_runs",
        "ties",
        "improvement_rate",
    ]
    if selected_rows.empty or baseline_rows.empty:
        return pd.DataFrame(columns=run_columns), pd.DataFrame(columns=summary_columns)

    chosen = selected_rows[
        ["subject", "run", "method", "localization_distance_mm"]
    ].copy()
    chosen = chosen.rename(
        columns={"localization_distance_mm": "selected_localization_distance_mm"}
    )
    paired = chosen.merge(
        baseline_rows,
        on=["subject", "run", "method"],
        how="inner",
        validate="one_to_one",
    )
    paired["selected_minus_script12_mm"] = (
        paired["selected_localization_distance_mm"] - paired["script12_baseline_mm"]
    )
    paired["improved_vs_script12"] = paired["selected_minus_script12_mm"] < 0

    records = []
    for method, group in paired.groupby("method", observed=True):
        diff = group["selected_minus_script12_mm"].to_numpy(float)
        selected_values = group["selected_localization_distance_mm"].to_numpy(float)
        baseline_values = group["script12_baseline_mm"].to_numpy(float)
        records.append(
            {
                "method": method,
                "paired_runs": len(group),
                "selected_mean_mm": np.mean(selected_values),
                "selected_median_mm": np.median(selected_values),
                "script12_mean_mm": np.mean(baseline_values),
                "script12_median_mm": np.median(baseline_values),
                "mean_selected_minus_script12_mm": np.mean(diff),
                "standard_deviation_selected_minus_script12_mm": (
                    np.std(diff, ddof=1) if len(diff) > 1 else np.nan
                ),
                "median_selected_minus_script12_mm": np.median(diff),
                "q25_selected_minus_script12_mm": np.quantile(diff, 0.25),
                "q75_selected_minus_script12_mm": np.quantile(diff, 0.75),
                "improved_runs": int(np.sum(diff < 0)),
                "worsened_runs": int(np.sum(diff > 0)),
                "ties": int(np.sum(np.isclose(diff, 0.0))),
                "improvement_rate": float(np.mean(diff < 0)),
            }
        )

    summary = pd.DataFrame(records)
    return paired[run_columns], summary[summary_columns]


def select_configurations(configuration_summary):
    eligible = configuration_summary.loc[
        configuration_summary["eligible_for_selection"].astype(bool)
    ].copy()
    if eligible.empty:
        raise ValueError(
            "No configuration meets the selection success-rate threshold. "
            "Lower --min-selection-success-rate only if scientifically justified."
        )

    selected = (
        eligible.sort_values(
            ["method", "median_localization_distance_mm", "mean_localization_distance_mm"]
        )
        .groupby("method", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )

    total_counts = (
        configuration_summary.groupby("method", observed=True)
        .size()
        .rename("candidate_configurations_total")
    )
    eligible_counts = (
        eligible.groupby("method", observed=True)
        .size()
        .rename("eligible_configurations")
    )
    selected["candidate_configurations_total"] = (
        selected["method"].map(total_counts).astype(int)
    )
    selected["eligible_configurations"] = (
        selected["method"].map(eligible_counts).fillna(0).astype(int)
    )

    expected_methods = set(configuration_summary["method"].astype(str))
    observed_methods = set(selected["method"].astype(str))
    missing = expected_methods - observed_methods
    if missing:
        raise ValueError(
            "No eligible selected configuration for methods: " + ", ".join(sorted(missing))
        )
    return selected


def selected_configuration_rows(table, selected):
    keys = selected[["method", "montage", "configuration_id"]].copy()
    result = table.merge(
        keys,
        on=["method", "montage", "configuration_id"],
        how="inner",
        validate="many_to_one",
    )
    result = result.loc[result["status"].eq("PASS")].copy()
    if result.duplicated(["subject", "run", "method"]).any():
        raise ValueError("Selected configuration produced duplicate subject/run/method rows.")
    return result


def create_selected_wide(selected_rows, methods):
    wide = (
        selected_rows.pivot(
            index=["subject", "run"], columns="method", values="localization_distance_mm"
        )
        .reindex(columns=methods)
        .sort_index()
    )
    return wide


def summarize_selected_methods(selected_rows):
    records = []
    for method in [
        m for m in METHOD_ORDER if m in set(selected_rows["method"].astype(str))
    ]:
        values = selected_rows.loc[
            selected_rows["method"].astype(str).eq(method), "localization_distance_mm"
        ].dropna()
        records.append(
            {
                "method": method,
                "runs": len(values),
                "mean_mm": values.mean(),
                "standard_deviation_mm": values.std(ddof=1),
                "median_mm": values.median(),
                "q25_mm": values.quantile(0.25),
                "q75_mm": values.quantile(0.75),
                "iqr_mm": values.quantile(0.75) - values.quantile(0.25),
                "minimum_mm": values.min(),
                "maximum_mm": values.max(),
            }
        )
    return pd.DataFrame(records)


def pairwise_selected_summary(wide, methods):
    records = []
    for method_a, method_b in combinations(methods, 2):
        pair = wide[[method_a, method_b]].dropna()
        if pair.empty:
            continue
        differences = pair[method_a].to_numpy(float) - pair[method_b].to_numpy(float)
        records.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "paired_runs": len(differences),
                "mean_a_minus_b_mm": np.mean(differences),
                "standard_deviation_a_minus_b_mm": (
                    np.std(differences, ddof=1) if len(differences) > 1 else np.nan
                ),
                "median_a_minus_b_mm": np.median(differences),
                "q25_a_minus_b_mm": np.quantile(differences, 0.25),
                "q75_a_minus_b_mm": np.quantile(differences, 0.75),
                "method_a_lower_error_runs": int(np.sum(differences < 0)),
                "method_b_lower_error_runs": int(np.sum(differences > 0)),
                "ties": int(np.sum(np.isclose(differences, 0.0))),
            }
        )
    return pd.DataFrame(records)


def participant_selected_summary(selected_rows):
    if selected_rows.empty:
        return pd.DataFrame()
    return (
        selected_rows.groupby(["subject", "method"], observed=True)
        .agg(
            runs=("localization_distance_mm", "size"),
            mean_mm=("localization_distance_mm", "mean"),
            standard_deviation_mm=("localization_distance_mm", "std"),
            median_mm=("localization_distance_mm", "median"),
            minimum_mm=("localization_distance_mm", "min"),
            maximum_mm=("localization_distance_mm", "max"),
        )
        .reset_index()
        .sort_values(["subject", "method"])
        .reset_index(drop=True)
    )


def lowest_error_selected_tables(wide, methods):
    rows = []
    for (subject, run), values in wide.iterrows():
        available = values.dropna()
        if available.empty:
            continue
        minimum = float(available.min())
        lowest = [
            method
            for method, value in available.items()
            if np.isclose(float(value), minimum)
        ]
        rows.append(
            {
                "subject": subject,
                "run": run,
                "lowest_error_mm": minimum,
                "lowest_error_method": ";".join(lowest),
                "tie": len(lowest) > 1,
                "methods_available": len(available),
            }
        )
    per_run = pd.DataFrame(rows)

    count_rows = []
    for method in methods:
        sole = 0
        tied = 0
        if not per_run.empty:
            for text in per_run["lowest_error_method"]:
                items = str(text).split(";")
                if method not in items:
                    continue
                if len(items) == 1:
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
    return per_run, pd.DataFrame(count_rows)


def selected_outliers(selected_rows, multiplier):
    records = []
    for method in [
        m for m in METHOD_ORDER if m in set(selected_rows["method"].astype(str))
    ]:
        subset = selected_rows.loc[selected_rows["method"].astype(str).eq(method)].copy()
        if subset.empty:
            continue
        q25 = subset["localization_distance_mm"].quantile(0.25)
        q75 = subset["localization_distance_mm"].quantile(0.75)
        iqr = q75 - q25
        threshold = q75 + multiplier * iqr
        flagged = subset.loc[subset["localization_distance_mm"] > threshold]
        for row in flagged.itertuples(index=False):
            records.append(
                {
                    "subject": row.subject,
                    "run": row.run,
                    "method": method,
                    "localization_distance_mm": float(row.localization_distance_mm),
                    "q25_mm": q25,
                    "q75_mm": q75,
                    "iqr_mm": iqr,
                    "upper_outlier_threshold_mm": threshold,
                    "iqr_multiplier": multiplier,
                }
            )
    return pd.DataFrame(records)


def prepare_output(output_root, overwrite):
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_root}\nUse --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    (output_root / "figures" / "factor_effects").mkdir(parents=True, exist_ok=True)
    (output_root / "figures" / "interactions").mkdir(parents=True, exist_ok=True)
    (output_root / "figures" / "failures").mkdir(parents=True, exist_ok=True)
    (output_root / "figures" / "top_configurations").mkdir(
        parents=True, exist_ok=True
    )
    (output_root / "figures" / "accuracy_stability").mkdir(
        parents=True, exist_ok=True
    )


def save_method_grid_distribution(table, path, dpi):
    """Plot Script-10-style colored boxplots of run-level grid medians.

    Each displayed point is one subject-run. Within each subject-run and method,
    the median localization error is first taken across successful grid solutions.
    This is therefore a descriptive within-grid summary, not a fair comparison of
    optimization opportunity across methods because the parameter grids differ in size.
    """
    passed = table.loc[table["status"].eq("PASS")].copy()
    run_level = (
        passed.groupby(["subject", "run", "method"], observed=True)
        .agg(run_median_mm=("localization_distance_mm", "median"))
        .reset_index()
    )
    methods = [m for m in METHOD_ORDER if m in set(run_level["method"].astype(str))]
    data = [
        run_level.loc[run_level["method"].astype(str).eq(method), "run_median_mm"].to_numpy(float)
        for method in methods
    ]

    method_colors = {
        "Continuous-ECD": "#4C78A8",
        "MxNE": "#E45756",
        "RAP-MUSIC": "#54A24B",
        "LCMV": "#B279A2",
    }

    fig, ax = plt.subplots(figsize=(12, 7.5))
    boxes = ax.boxplot(
        data,
        tick_labels=methods,
        patch_artist=True,
        showfliers=False,
        widths=0.50,
        medianprops={"color": "#333333", "linewidth": 1.5},
        whiskerprops={"color": "#444444", "linewidth": 1.2},
        capprops={"color": "#444444", "linewidth": 1.2},
        boxprops={"edgecolor": "#444444", "linewidth": 1.2},
    )
    for patch, method in zip(boxes["boxes"], methods):
        patch.set_facecolor(method_colors.get(method, "#999999"))
        patch.set_alpha(0.90)

    # Show all run-level observations, as in Script 10's stripplot.
    rng = np.random.default_rng(42)
    for xpos, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(
            np.full(len(values), xpos, dtype=float) + jitter,
            values,
            s=18,
            c="black",
            alpha=0.50,
            linewidths=0,
            zorder=3,
        )

    runs = table[["subject", "run"]].drop_duplicates().shape[0]
    participants = table["subject"].nunique()
    attempted = len(table)
    passed_count = int(table["status"].eq("PASS").sum())
    failed_count = int(table["status"].eq("FAIL").sum())

    target_start = pd.to_numeric(table["target_tmin_s"], errors="coerce").dropna().unique()
    target_stop = pd.to_numeric(table["target_tmax_s"], errors="coerce").dropna().unique()
    context = f"{runs} runs | {participants} participants | {attempted:,} attempted solutions"
    if failed_count:
        context += f" ({passed_count:,} PASS; {failed_count:,} FAIL)"
    if len(target_start) == 1 and len(target_stop) == 1:
        context += (
            f" | Target: {target_start[0] * 1000:+g} to "
            f"{target_stop[0] * 1000:+g} ms"
        )

    ax.set_ylabel("Within-run median localization distance (mm)")
    ax.set_xlabel("Inverse method")
    ax.set_title(
        "Typical localization error across the parameter grid\n" + context,
        pad=14,
    )
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.012,
        "Descriptive within-grid summary: parameter grids differ in size and design; failures are summarized separately.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_failure_rate_by_method(quality, path, dpi):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(quality))
    rates = (1.0 - quality["solution_success_rate"].to_numpy(float)) * 100
    ax.bar(x, rates)
    ax.set_xticks(x, quality["method"], rotation=20, ha="right")
    ax.set_ylabel("Failed solutions (%)")
    ax.set_title("Script-15 solution failure rate by method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_factor_effect_figures(factor_summary, output_dir, dpi):
    if factor_summary.empty:
        return
    for (method, factor), subset in factor_summary.groupby(["method", "factor"]):
        subset = subset.copy()
        levels = sort_levels(subset["level"].tolist(), factor)
        level_strings = [str(item) for item in levels]
        subset["level_string"] = subset["level"].astype(str)
        subset = subset.set_index("level_string").reindex(level_strings).reset_index()

        x = np.arange(len(subset))
        y = subset["median_of_run_medians_mm"].to_numpy(float)
        q25 = subset["q25_of_run_medians_mm"].to_numpy(float)
        q75 = subset["q75_of_run_medians_mm"].to_numpy(float)
        lower = y - q25
        upper = q75 - y

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=4)
        ax.set_xticks(x, level_strings, rotation=25, ha="right")
        ax.set_xlabel(factor)
        ax.set_ylabel("Median of run-level median localization error (mm)")
        ax.set_title(f"{method}: marginal effect of {factor}")
        ax.grid(axis="y", alpha=0.25)

        failures = subset["fail_solutions"].fillna(0).to_numpy(int)
        for xpos, ypos, fail in zip(x, y, failures):
            if fail > 0 and np.isfinite(ypos):
                ax.annotate(f"{fail} fail", (xpos, ypos), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=8)

        fig.tight_layout()
        name = f"{METHOD_SLUG[str(method)]}__{factor}.png"
        fig.savefig(output_dir / name, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def interaction_matrix(subset, factor_a, factor_b, value_column):
    working = subset.copy()
    order_a = [str(v) for v in sort_levels(working["level_a"].tolist(), factor_a)]
    order_b = [str(v) for v in sort_levels(working["level_b"].tolist(), factor_b)]
    working["level_a"] = working["level_a"].astype(str)
    working["level_b"] = working["level_b"].astype(str)
    matrix = working.pivot(index="level_a", columns="level_b", values=value_column)
    matrix = matrix.reindex(index=order_a, columns=order_b)
    return matrix


def save_heatmap(matrix, title, xlabel, ylabel, cbar_label, path, dpi, fmt=".1f"):
    if matrix.empty:
        return
    values = matrix.to_numpy(float)
    fig_width = max(7, 0.75 * matrix.shape[1] + 3)
    fig_height = max(5, 0.55 * matrix.shape[0] + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(cbar_label)

    finite = np.isfinite(values)
    if finite.sum() <= 120:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_interaction_figures(interaction_summary, output_dir, dpi):
    if interaction_summary.empty:
        return
    for (method, factor_a, factor_b), subset in interaction_summary.groupby(
        ["method", "factor_a", "factor_b"]
    ):
        matrix = interaction_matrix(
            subset, factor_a, factor_b, "median_of_run_medians_mm"
        )
        name = (
            f"{METHOD_SLUG[str(method)]}__{factor_a}__{factor_b}.png"
            .replace("/", "_")
        )
        save_heatmap(
            matrix,
            f"{method}: {factor_a} × {factor_b}",
            factor_b,
            factor_a,
            "Median of run-level median error (mm)",
            output_dir / name,
            dpi,
            fmt=".1f",
        )


def save_failure_interaction_figures(interaction_summary, output_dir, dpi):
    if interaction_summary.empty:
        return
    failed = interaction_summary.loc[interaction_summary["fail_solutions"] > 0].copy()
    if failed.empty:
        return
    failed["failure_rate_percent"] = (
        1.0 - failed["solution_success_rate"].astype(float)
    ) * 100
    for (method, factor_a, factor_b), subset in failed.groupby(
        ["method", "factor_a", "factor_b"]
    ):
        complete_subset = interaction_summary.loc[
            (interaction_summary["method"] == method)
            & (interaction_summary["factor_a"] == factor_a)
            & (interaction_summary["factor_b"] == factor_b)
        ].copy()
        complete_subset["failure_rate_percent"] = (
            1.0 - complete_subset["solution_success_rate"].astype(float)
        ) * 100
        matrix = interaction_matrix(
            complete_subset, factor_a, factor_b, "failure_rate_percent"
        )
        name = (
            f"{METHOD_SLUG[str(method)]}__{factor_a}__{factor_b}__failure_rate.png"
            .replace("/", "_")
        )
        save_heatmap(
            matrix,
            f"{method}: failure rate for {factor_a} × {factor_b}",
            factor_b,
            factor_a,
            "Failed solutions (%)",
            output_dir / name,
            dpi,
            fmt=".2f",
        )


def save_top_configuration_figures(top, output_dir, dpi):
    if top.empty:
        return
    for method, subset in top.groupby("method"):
        subset = subset.sort_values(
            ["median_localization_distance_mm", "mean_localization_distance_mm"],
            ascending=True,
        ).copy()
        labels = [
            f"{row.montage} | {row.configuration_id}"
            for row in subset.itertuples(index=False)
        ]
        y = np.arange(len(subset))
        fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(subset) + 2)))
        ax.scatter(subset["median_localization_distance_mm"], y, label="Median")
        ax.scatter(
            subset["mean_localization_distance_mm"], y, marker="x", label="Mean"
        )
        for i, row in enumerate(subset.itertuples(index=False)):
            ax.plot(
                [row.median_localization_distance_mm, row.mean_localization_distance_mm],
                [i, i],
                linewidth=1,
                alpha=0.6,
            )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Localization error (mm)")
        ax.set_title(f"{method}: lowest-median eligible configurations")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{METHOD_SLUG[str(method)]}__top_configurations.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_accuracy_stability_figures(configuration_summary, output_dir, dpi):
    eligible = configuration_summary.loc[
        configuration_summary["eligible_for_selection"].astype(bool)
    ].copy()
    if eligible.empty:
        return
    for method, subset in eligible.groupby("method"):
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for montage in MONTAGE_ORDER:
            points = subset.loc[subset["montage"].astype(str).eq(montage)]
            if points.empty:
                continue
            ax.scatter(
                points["median_localization_distance_mm"],
                points["iqr_mm"],
                label=montage,
                alpha=0.6,
            )
        best = subset.sort_values(
            ["median_localization_distance_mm", "mean_localization_distance_mm"]
        ).iloc[0]
        ax.scatter(
            [best["median_localization_distance_mm"]],
            [best["iqr_mm"]],
            marker="*",
            s=180,
            label="selected lowest median",
        )
        ax.set_xlabel("Median localization error across runs (mm)")
        ax.set_ylabel("Interquartile range across runs (mm)")
        ax.set_title(f"{method}: accuracy–stability landscape")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{METHOD_SLUG[str(method)]}__accuracy_stability.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_selected_distribution(selected_rows, path, dpi):
    methods = [m for m in METHOD_ORDER if m in set(selected_rows["method"].astype(str))]
    data = [
        selected_rows.loc[
            selected_rows["method"].astype(str).eq(method), "localization_distance_mm"
        ].to_numpy()
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data, tick_labels=methods, showfliers=True)
    ax.set_ylabel("Localization error (mm)")
    ax.set_xlabel("Inverse method")
    ax.set_title("Selected lowest-median observed configurations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_selected_participant_figure(participant, path, dpi):
    if participant.empty:
        return
    pivot = participant.pivot(index="subject", columns="method", values="median_mm")
    methods = [m for m in METHOD_ORDER if m in pivot.columns]
    x = np.arange(len(pivot.index))
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        ax.plot(x, pivot[method].to_numpy(float), marker="o", label=method)
    ax.set_xticks(x, pivot.index)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Median localization error (mm)")
    ax.set_title("Selected-configuration localization error by participant")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_lowest_count_figure(counts, path, dpi):
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(counts))
    ax.bar(x, counts["sole_lowest_error_runs"])
    ax.set_xticks(x, counts["method"], rotation=20, ha="right")
    ax.set_ylabel("Runs")
    ax.set_title("Lowest observed error among selected fixed configurations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def save_selected_vs_script12_figure(paired, path, dpi):
    """Plot paired change for selected configurations relative to Script 12."""
    if paired.empty:
        return
    methods = [
        m for m in METHOD_ORDER if m in set(paired["method"].astype(str))
    ]
    data = [
        paired.loc[
            paired["method"].astype(str).eq(method),
            "selected_minus_script12_mm",
        ].to_numpy(float)
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data, tick_labels=methods, showfliers=True)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Selected configuration - Script 12 baseline (mm)")
    ax.set_xlabel("Inverse method")
    ax.set_title(
        "Selected lowest-median configuration versus predefined Script-12 baseline"
    )
    ax.text(
        0.01,
        0.02,
        "Negative values indicate lower localization error than the Script-12 baseline.",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_tables(
    table_dir,
    quality,
    method_summary,
    factor_summary,
    interaction_summary,
    configuration_summary,
    top,
    best_by_montage,
    failure_factor,
    failure_details,
    selected,
    selected_rows,
    selected_wide,
    selected_method_summary,
    pairwise,
    participants,
    lowest_per_run,
    lowest_counts,
    outliers,
    factor_sensitivity,
    selected_vs_baseline_runs,
    selected_vs_baseline_summary,
):
    quality.to_csv(table_dir / "01_grid_quality_summary.csv", index=False)
    method_summary.to_csv(table_dir / "02_method_grid_summary.csv", index=False)
    factor_summary.to_csv(table_dir / "03_factor_level_summary.csv", index=False)
    interaction_summary.to_csv(table_dir / "04_interaction_summary.csv", index=False)
    configuration_summary.to_csv(
        table_dir / "05_configuration_summary.csv", index=False
    )
    top.to_csv(table_dir / "06_top_configurations.csv", index=False)
    best_by_montage.to_csv(
        table_dir / "07_lowest_median_by_montage.csv", index=False
    )
    failure_factor.to_csv(table_dir / "08_failure_factor_summary.csv", index=False)
    failure_details.to_csv(table_dir / "09_failure_details.csv", index=False)
    selected.to_csv(table_dir / "10_selected_configurations.csv", index=False)
    selected_rows.to_csv(
        table_dir / "11_selected_config_results_long.csv", index=False
    )
    selected_wide.reset_index().to_csv(
        table_dir / "12_selected_config_results_wide.csv", index=False
    )
    selected_method_summary.to_csv(
        table_dir / "13_selected_config_method_summary.csv", index=False
    )
    pairwise.to_csv(
        table_dir / "14_selected_config_pairwise_differences.csv", index=False
    )
    participants.to_csv(
        table_dir / "15_selected_config_participant_summary.csv", index=False
    )
    lowest_per_run.to_csv(
        table_dir / "16_selected_config_lowest_error_per_run.csv", index=False
    )
    lowest_counts.to_csv(
        table_dir / "17_selected_config_lowest_error_counts.csv", index=False
    )
    outliers.to_csv(table_dir / "18_selected_config_outliers.csv", index=False)
    factor_sensitivity.to_csv(
        table_dir / "19_factor_sensitivity_summary.csv", index=False
    )
    selected_vs_baseline_runs.to_csv(
        table_dir / "20_selected_vs_script12_baseline_runs.csv", index=False
    )
    selected_vs_baseline_summary.to_csv(
        table_dir / "21_selected_vs_script12_baseline_summary.csv", index=False
    )


def print_summary(quality, selected, output_root, min_success):
    print("=" * 78)
    print("SCRIPT 16 — ADDITIONAL-METHOD GRID ANALYSIS")
    print("=" * 78)
    print("\nGRID QUALITY")
    show = quality[
        [
            "method",
            "expected_solutions",
            "observed_rows",
            "pass_solutions",
            "fail_solutions",
            "solution_success_rate",
        ]
    ].copy()
    print(
        show.to_string(
            index=False,
            formatters={"solution_success_rate": lambda x: f"{x * 100:.3f}%"},
        )
    )

    print("\nSELECTED LOWEST-MEDIAN OBSERVED CONFIGURATIONS")
    columns = [
        "method",
        "montage",
        "configuration_id",
        "candidate_configurations_total",
        "eligible_configurations",
        "runs_pass",
        "runs_fail",
        "success_rate",
        "median_localization_distance_mm",
        "mean_localization_distance_mm",
        "standard_deviation_mm",
    ]
    print(
        selected[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print()
    print(
        f"Selection eligibility required success_rate >= {min_success:.3f}. "
        "Selection and evaluation use the same runs; these are descriptive, "
        "not independently validated optima."
    )
    print(
        "Caution: methods were searched over unequal numbers of candidate "
        "configurations; selected-configuration cross-method comparisons are descriptive."
    )
    print(f"Saved outputs: {output_root}")


def main():
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if not 0 < args.min_selection_success_rate <= 1:
        raise ValueError("--min-selection-success-rate must be in (0, 1].")
    if args.outlier_iqr_multiplier < 0:
        raise ValueError("--outlier-iqr-multiplier must be non-negative.")
    if args.dpi < 50:
        raise ValueError("--dpi must be at least 50.")

    methods = selected_methods(args.method)
    paths = resolve_input_paths(args, methods)
    output_root = args.output_root.expanduser().resolve()
    prepare_output(output_root, args.overwrite)

    table, metadata = load_all_grids(paths, args.allow_incomplete_grid)

    quality = create_quality_summary(table, metadata)
    method_summary = create_method_grid_summary(table)
    factor_summary = create_factor_level_summary(table)
    interaction_summary = create_interaction_summary(table)
    configuration_summary = create_configuration_summary(
        table, args.min_selection_success_rate
    )
    top = create_top_configurations(configuration_summary, args.top_n)
    best_by_montage = create_lowest_median_by_montage(configuration_summary)
    failure_factor = create_failure_factor_summary(factor_summary)
    failure_details = create_failure_details(table)
    factor_sensitivity = create_factor_sensitivity_summary(factor_summary)

    selected = select_configurations(configuration_summary)
    selected_rows = selected_configuration_rows(table, selected)
    selected_wide = create_selected_wide(selected_rows, methods)
    selected_method_summary = summarize_selected_methods(selected_rows)
    pairwise = pairwise_selected_summary(selected_wide, methods)
    participants = participant_selected_summary(selected_rows)
    lowest_per_run, lowest_counts = lowest_error_selected_tables(selected_wide, methods)
    outliers = selected_outliers(selected_rows, args.outlier_iqr_multiplier)
    script12_baselines = extract_script12_baselines(table)
    selected_vs_baseline_runs, selected_vs_baseline_summary = (
        compare_selected_with_script12(selected_rows, script12_baselines)
    )

    table_dir = output_root / "tables"
    write_tables(
        table_dir,
        quality,
        method_summary,
        factor_summary,
        interaction_summary,
        configuration_summary,
        top,
        best_by_montage,
        failure_factor,
        failure_details,
        selected,
        selected_rows,
        selected_wide,
        selected_method_summary,
        pairwise,
        participants,
        lowest_per_run,
        lowest_counts,
        outliers,
        factor_sensitivity,
        selected_vs_baseline_runs,
        selected_vs_baseline_summary,
    )

    if not args.no_figures:
        figure_root = output_root / "figures"
        save_method_grid_distribution(
            table, figure_root / "01_within_grid_typical_error_descriptive.png", args.dpi
        )
        save_failure_rate_by_method(
            quality, figure_root / "02_failure_rate_by_method.png", args.dpi
        )
        save_factor_effect_figures(
            factor_summary, figure_root / "factor_effects", args.dpi
        )
        save_interaction_figures(
            interaction_summary, figure_root / "interactions", args.dpi
        )
        save_failure_interaction_figures(
            interaction_summary, figure_root / "failures", args.dpi
        )
        save_top_configuration_figures(
            top, figure_root / "top_configurations", args.dpi
        )
        save_accuracy_stability_figures(
            configuration_summary, figure_root / "accuracy_stability", args.dpi
        )
        save_selected_distribution(
            selected_rows,
            figure_root / "03_selected_configuration_distribution.png",
            args.dpi,
        )
        save_selected_participant_figure(
            participants,
            figure_root / "04_selected_configuration_by_participant.png",
            args.dpi,
        )
        save_lowest_count_figure(
            lowest_counts,
            figure_root / "05_selected_configuration_lowest_error_counts.png",
            args.dpi,
        )
        save_selected_vs_script12_figure(
            selected_vs_baseline_runs,
            figure_root / "06_selected_vs_script12_baseline.png",
            args.dpi,
        )

    print_summary(
        quality,
        selected,
        output_root,
        args.min_selection_success_rate,
    )


if __name__ == "__main__":
    main()
