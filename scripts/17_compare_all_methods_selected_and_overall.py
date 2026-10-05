#!/usr/bin/env python3
"""Compare all eight Localize-MI inverse methods using two complementary views.

SCRIPT 17
=========
This script is the final descriptive comparison companion to Scripts 10 and 16.
It does NOT run source localization and it does NOT search for new parameters.

Two main values are retained for every inverse method:

1) SELECTED CONFIGURATION VALUE
   One fixed "lowest-median observed configuration" selected previously by
   Script 10 (MNE/dSPM/sLORETA/eLORETA) or Script 16
   (Continuous-ECD/MxNE/RAP-MUSIC/LCMV). The fixed configuration is then
   represented by one localization error per subject-run (61 values/method).

2) OVERALL-RUN / WITHIN-GRID TYPICAL VALUE
   For each subject-run-method, all successful solutions in that method's
   tested parameter grid are collapsed to the within-run median localization
   error. This again yields one value per subject-run (61 values/method).

The second quantity describes typical behavior over the *tested grid*; it is
not the performance of a fixed configuration. Parameter grids differ markedly
in size and design, so cross-method comparisons of these overall-grid values
are descriptive only.

Selection caution
-----------------
The selected configurations were selected and evaluated on the same 61 runs.
They must be described as "lowest-median observed configurations", not as
independently validated optima or universally best configurations.

Default inputs
--------------
Original-method full grid (Script 09):
    outputs/grid_all_runs/grid_results.csv

Script-10 selected configuration table:
    outputs/parameter_grid_analysis/tables/05_top_full_grid_configurations.csv

Additional-method full grids (Script 15):
    outputs/15_full_additional_method_grid/continuous_ecd/grid_results.csv
    outputs/15_full_additional_method_grid/mxne/grid_results.csv
    outputs/15_full_additional_method_grid/rap_music/grid_results.csv
    outputs/15_full_additional_method_grid/lcmv/grid_results.csv

Script-16 selected configuration table:
    outputs/16_additional_method_grid_analysis/tables/10_selected_configurations.csv

Default output
--------------
    outputs/17_all_methods_comparison/

Main tables
-----------
    01_selected_configuration_definitions.csv
    02_selected_configuration_method_summary.csv
    03_overall_grid_method_summary.csv
    04_selected_vs_overall_summary.csv
    05_selected_configuration_pairwise_differences.csv
    06_overall_grid_pairwise_differences.csv
    07_selected_configuration_participant_summary.csv
    08_overall_grid_participant_summary.csv
    09_selected_configuration_run_summary.csv
    10_overall_grid_run_summary.csv
    11_selected_configuration_lowest_error_counts.csv
    12_overall_grid_lowest_typical_counts.csv
    13_selected_configuration_results_long.csv
    14_selected_configuration_results_wide.csv
    15_overall_grid_run_values_long.csv
    16_overall_grid_run_values_wide.csv
    17_selected_configuration_outliers.csv
    18_overall_grid_outliers.csv
    19_dataset_summary.csv

Figures
-------
    01_selected_configuration_distribution.png
    02_overall_grid_run_distribution.png
    03_selected_vs_overall_method_medians.png
    04_selected_pairwise_median_differences.png
    05_selected_configuration_by_participant.png
    06_selected_configuration_lowest_error_counts.png
    07_overall_grid_by_participant.png
    08_selected_run_difficulty.png

Example
-------
    python scripts/17_compare_all_methods_selected_and_overall.py --overwrite
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - figures still work without seaborn
    sns = None


PROJECT = Path(__file__).resolve().parents[1]

DEFAULT_ORIGINAL_GRID = PROJECT / "outputs" / "grid_all_runs" / "grid_results.csv"
DEFAULT_SCRIPT10_TOP = (
    PROJECT
    / "outputs"
    / "parameter_grid_analysis"
    / "tables"
    / "05_top_full_grid_configurations.csv"
)
DEFAULT_SCRIPT15_ROOT = PROJECT / "outputs" / "15_full_additional_method_grid"
DEFAULT_SCRIPT16_SELECTED = (
    PROJECT
    / "outputs"
    / "16_additional_method_grid_analysis"
    / "tables"
    / "10_selected_configurations.csv"
)
DEFAULT_OUTPUT = PROJECT / "outputs" / "17_all_methods_comparison"

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
ORIGINAL_METHODS = ["MNE", "dSPM", "sLORETA", "eLORETA"]
ADDITIONAL_METHODS = ["Continuous-ECD", "MxNE", "RAP-MUSIC", "LCMV"]
MONTAGE_ORDER = ["all_good", "128", "64", "32"]

METHOD_COLORS = {
    "MNE": "#4C78A8",
    "dSPM": "#E45756",
    "sLORETA": "#54A24B",
    "eLORETA": "#8E5C99",
    "Continuous-ECD": "#F58518",
    "MxNE": "#72B7B2",
    "RAP-MUSIC": "#B279A2",
    "LCMV": "#ECA82C",
}

ORIGINAL_CONFIG_COLUMNS = [
    "method",
    "method_origin",
    "montage",
    "loose",
    "depth",
    "snr",
    "lambda2",
]
ADDITIONAL_CONFIG_COLUMNS = ["method", "montage", "configuration_id"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--original-grid", type=Path, default=DEFAULT_ORIGINAL_GRID)
    parser.add_argument("--script10-top", type=Path, default=DEFAULT_SCRIPT10_TOP)
    parser.add_argument("--script15-root", type=Path, default=DEFAULT_SCRIPT15_ROOT)
    parser.add_argument(
        "--script16-selected", type=Path, default=DEFAULT_SCRIPT16_SELECTED
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="Upper outlier threshold per method: Q3 + k*IQR.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Allow missing method/run values. Cross-method tables still use only "
            "available values; default requires all 8 methods on all matched runs."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def require_columns(table: pd.DataFrame, columns, label: str):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def first_existing(columns, candidates, description, label):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not identify {description} in {label}. Tried {candidates}."
    )


def prepare_output(path: Path, overwrite: bool):
    path = path.expanduser().resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}\nUse --overwrite.")
        shutil.rmtree(path)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def normalize_status(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if "status" not in table.columns:
        table["status"] = "PASS"
    table["status"] = table["status"].astype(str).str.upper()
    invalid = sorted(set(table["status"]) - {"PASS", "FAIL"})
    if invalid:
        raise ValueError(f"Unexpected status values: {invalid}")
    return table


def standardize_grid_error_column(table: pd.DataFrame, label: str) -> pd.DataFrame:
    error_col = first_existing(
        table.columns,
        [
            "localization_distance_mm",
            "selected_localization_error_mm",
            "localization_error_mm",
            "distance_mm",
        ],
        "localization-error column",
        label,
    )
    table = table.copy()
    table["localization_distance_mm"] = pd.to_numeric(
        table[error_col], errors="coerce"
    )
    return table


def load_original_grid(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    require_columns(table, ["subject", "run", "method", "montage"], str(path))
    table = standardize_grid_error_column(table, str(path))
    table = normalize_status(table)
    table = table.loc[table["method"].astype(str).isin(ORIGINAL_METHODS)].copy()
    table["subject"] = table["subject"].astype(str)
    table["run"] = table["run"].astype(str)
    table["method"] = table["method"].astype(str)
    table["montage"] = table["montage"].astype(str)

    pass_mask = table["status"].eq("PASS")
    if table.loc[pass_mask, "localization_distance_mm"].isna().any():
        raise ValueError("Original grid contains PASS rows with missing localization errors.")
    return table


def additional_grid_paths(root: Path):
    return {
        "Continuous-ECD": root / "continuous_ecd" / "grid_results.csv",
        "MxNE": root / "mxne" / "grid_results.csv",
        "RAP-MUSIC": root / "rap_music" / "grid_results.csv",
        "LCMV": root / "lcmv" / "grid_results.csv",
    }


def load_additional_grids(root: Path) -> pd.DataFrame:
    tables = []
    for method, path in additional_grid_paths(root).items():
        path = require_file(path, f"{method} Script-15 grid")
        table = pd.read_csv(path, low_memory=False)
        require_columns(
            table,
            ["subject", "run", "method", "montage", "configuration_id"],
            str(path),
        )
        table = standardize_grid_error_column(table, str(path))
        table = normalize_status(table)
        observed = set(table["method"].dropna().astype(str))
        if observed != {method}:
            raise ValueError(f"Expected only {method} in {path}, found {observed}")
        table["subject"] = table["subject"].astype(str)
        table["run"] = table["run"].astype(str)
        table["method"] = table["method"].astype(str)
        table["montage"] = table["montage"].astype(str)
        table["configuration_id"] = table["configuration_id"].astype(str)
        pass_mask = table["status"].eq("PASS")
        if table.loc[pass_mask, "localization_distance_mm"].isna().any():
            raise ValueError(f"{method} grid has PASS rows with missing localization errors.")
        tables.append(table)
    return pd.concat(tables, ignore_index=True, sort=False)


def choose_script10_selected(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    require_columns(
        table,
        ["method", "montage", "loose", "depth", "snr"],
        str(path),
    )
    table = table.loc[table["method"].astype(str).isin(ORIGINAL_METHODS)].copy()

    if "reported_rank" in table.columns:
        selected = table.loc[pd.to_numeric(table["reported_rank"], errors="coerce").eq(1)].copy()
    else:
        sort_cols = ["method", "median_localization_distance_mm"]
        if "mean_localization_distance_mm" in table.columns:
            sort_cols.append("mean_localization_distance_mm")
        selected = (
            table.sort_values(sort_cols)
            .groupby("method", group_keys=False)
            .head(1)
            .copy()
        )

    counts = selected.groupby("method").size()
    missing = set(ORIGINAL_METHODS) - set(selected["method"].astype(str))
    duplicates = counts.loc[counts != 1]
    if missing or not duplicates.empty:
        raise ValueError(
            f"Script-10 selected configuration table is ambiguous. Missing={missing}; "
            f"non-single counts={duplicates.to_dict()}"
        )
    return selected


def choose_script16_selected(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    require_columns(table, ADDITIONAL_CONFIG_COLUMNS, str(path))
    table = table.loc[table["method"].astype(str).isin(ADDITIONAL_METHODS)].copy()
    counts = table.groupby("method").size()
    missing = set(ADDITIONAL_METHODS) - set(table["method"].astype(str))
    duplicates = counts.loc[counts != 1]
    if missing or not duplicates.empty:
        raise ValueError(
            f"Script-16 selected configuration table must have one row/method. "
            f"Missing={missing}; non-single counts={duplicates.to_dict()}"
        )
    return table


def _close_series(series, value):
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        target = float(value)
    except (TypeError, ValueError):
        return series.astype(str).eq(str(value))
    return np.isclose(numeric.to_numpy(float), target, rtol=0, atol=1e-10)


def extract_original_selected_rows(grid: pd.DataFrame, selected: pd.DataFrame):
    pieces = []
    definition_rows = []

    for method in ORIGINAL_METHODS:
        cfg = selected.loc[selected["method"].astype(str).eq(method)].iloc[0]
        subset = grid.loc[grid["method"].eq(method)].copy()

        for column in ["method_origin", "montage", "loose", "depth", "snr", "lambda2"]:
            if column not in selected.columns or pd.isna(cfg.get(column, np.nan)):
                continue
            if column not in subset.columns:
                raise ValueError(f"Original grid lacks selected-configuration column {column}")
            if column in {"loose", "depth", "snr", "lambda2"}:
                subset = subset.loc[_close_series(subset[column], cfg[column])]
            else:
                subset = subset.loc[subset[column].astype(str).eq(str(cfg[column]))]

        subset = subset.loc[subset["status"].eq("PASS")].copy()
        if subset.duplicated(["subject", "run"]).any():
            raise ValueError(f"Selected {method} configuration has duplicate rows per run.")
        if subset.empty:
            raise ValueError(f"No rows matched the Script-10 selected configuration for {method}.")

        result = subset[["subject", "run"]].copy()
        result["method"] = method
        result["localization_error_mm"] = subset["localization_distance_mm"].to_numpy()
        result["view"] = "selected_configuration"
        pieces.append(result)

        # Candidate count = unique tested full configurations including montage.
        config_cols = [c for c in ORIGINAL_CONFIG_COLUMNS if c in grid.columns and c != "method"]
        candidate_count = (
            grid.loc[grid["method"].eq(method), ["method", *config_cols]]
            .drop_duplicates()
            .shape[0]
        )
        definition_rows.append(
            make_definition_row(method, "Script 10", cfg, candidate_count)
        )

    return pd.concat(pieces, ignore_index=True), pd.DataFrame(definition_rows)


def extract_additional_selected_rows(grid: pd.DataFrame, selected: pd.DataFrame):
    pieces = []
    definition_rows = []

    for method in ADDITIONAL_METHODS:
        cfg = selected.loc[selected["method"].astype(str).eq(method)].iloc[0]
        subset = grid.loc[
            grid["method"].eq(method)
            & grid["montage"].astype(str).eq(str(cfg["montage"]))
            & grid["configuration_id"].astype(str).eq(str(cfg["configuration_id"]))
        ].copy()
        subset = subset.loc[subset["status"].eq("PASS")].copy()
        if subset.duplicated(["subject", "run"]).any():
            raise ValueError(f"Selected {method} configuration has duplicate rows per run.")
        if subset.empty:
            raise ValueError(f"No rows matched Script-16 selected configuration for {method}.")

        result = subset[["subject", "run"]].copy()
        result["method"] = method
        result["localization_error_mm"] = subset["localization_distance_mm"].to_numpy()
        result["view"] = "selected_configuration"
        pieces.append(result)

        candidate_count = (
            grid.loc[grid["method"].eq(method), ["montage", "configuration_id"]]
            .drop_duplicates()
            .shape[0]
        )
        definition_rows.append(
            make_definition_row(method, "Script 16", cfg, candidate_count)
        )

    return pd.concat(pieces, ignore_index=True), pd.DataFrame(definition_rows)


def make_definition_row(method, selection_source, cfg, candidate_count):
    fields = {
        "method": method,
        "selection_source": selection_source,
        "selection_label": "lowest-median observed configuration",
        "candidate_configurations_searched": int(candidate_count),
        "montage": cfg.get("montage", np.nan),
        "configuration_id": cfg.get("configuration_id", np.nan),
        "method_origin": cfg.get("method_origin", np.nan),
        "loose": cfg.get("loose", np.nan),
        "depth": cfg.get("depth", np.nan),
        "snr": cfg.get("snr", np.nan),
        "lambda2": cfg.get("lambda2", np.nan),
        "ecd_min_dist_mm": cfg.get("ecd_min_dist_mm", np.nan),
        "mxne_alpha": cfg.get("mxne_alpha", np.nan),
        "mxne_loose": cfg.get("mxne_loose", np.nan),
        "mxne_depth": cfg.get("mxne_depth", np.nan),
        "rap_music_n_dipoles": cfg.get("rap_music_n_dipoles", np.nan),
        "lcmv_reg": cfg.get("lcmv_reg", np.nan),
        "lcmv_window_half_ms": cfg.get("lcmv_window_half_ms", np.nan),
        "lcmv_data_covariance_method": cfg.get(
            "lcmv_data_covariance_method", np.nan
        ),
        "lcmv_weight_norm": cfg.get("lcmv_weight_norm", np.nan),
        "selected_median_mm_from_source": cfg.get(
            "median_localization_distance_mm", np.nan
        ),
        "selected_mean_mm_from_source": cfg.get(
            "mean_localization_distance_mm", np.nan
        ),
        "selected_success_rate": cfg.get("success_rate", 1.0),
    }
    fields["parameter_description"] = describe_parameters(fields)
    return fields


def describe_parameters(row):
    parts = []
    montage = row.get("montage")
    if pd.notna(montage):
        parts.append(f"montage={montage}")

    mapping = [
        ("loose", "loose"),
        ("depth", "depth"),
        ("snr", "SNR"),
        ("lambda2", "lambda2"),
        ("ecd_min_dist_mm", "min_dist_mm"),
        ("mxne_alpha", "alpha"),
        ("mxne_loose", "loose"),
        ("mxne_depth", "depth"),
        ("rap_music_n_dipoles", "n_dipoles"),
        ("lcmv_reg", "reg"),
        ("lcmv_window_half_ms", "window_half_ms"),
        ("lcmv_data_covariance_method", "data_cov"),
        ("lcmv_weight_norm", "weight_norm"),
    ]
    for key, label in mapping:
        value = row.get(key)
        if pd.notna(value):
            parts.append(f"{label}={compact(value)}")
    return ", ".join(parts)


def compact(value):
    if pd.isna(value):
        return "NA"
    try:
        number = float(value)
        if np.isfinite(number):
            return f"{number:.6g}"
    except (TypeError, ValueError):
        pass
    return str(value)


def create_overall_run_values(original_grid, additional_grid):
    table = pd.concat([original_grid, additional_grid], ignore_index=True, sort=False)
    keys = ["subject", "run", "method"]

    counts = (
        table.groupby(keys, observed=True)
        .agg(
            attempted_solutions=("status", "size"),
            pass_solutions=("status", lambda x: int((x == "PASS").sum())),
            fail_solutions=("status", lambda x: int((x == "FAIL").sum())),
        )
        .reset_index()
    )
    passed = table.loc[table["status"].eq("PASS")].copy()
    errors = (
        passed.groupby(keys, observed=True)
        .agg(
            localization_error_mm=("localization_distance_mm", "median"),
            within_run_grid_mean_mm=("localization_distance_mm", "mean"),
        )
        .reset_index()
    )
    result = counts.merge(errors, on=keys, how="left", validate="one_to_one")
    result["solution_success_rate"] = (
        result["pass_solutions"] / result["attempted_solutions"]
    )
    result["view"] = "overall_grid_run_median"
    return result


def validate_matched(long_table, label, allow_incomplete=False):
    if long_table["localization_error_mm"].isna().any():
        bad = long_table.loc[
            long_table["localization_error_mm"].isna(), ["subject", "run", "method"]
        ].head(20)
        raise ValueError(f"{label} contains missing errors:\n{bad.to_string(index=False)}")
    if (long_table["localization_error_mm"] < 0).any():
        raise ValueError(f"{label} contains negative localization errors.")
    dup = long_table.duplicated(["subject", "run", "method"], keep=False)
    if dup.any():
        bad = long_table.loc[dup, ["subject", "run", "method"]].head(20)
        raise ValueError(f"{label} contains duplicate run/method rows:\n{bad.to_string(index=False)}")

    counts = long_table.groupby(["subject", "run"])["method"].nunique()
    methods = set(long_table["method"].astype(str))
    if not allow_incomplete:
        if methods != set(METHOD_ORDER):
            raise ValueError(f"{label}: method set mismatch. Found {sorted(methods)}")
        bad = counts.loc[counts != len(METHOD_ORDER)]
        if not bad.empty:
            raise ValueError(
                f"{label} is not fully matched across 8 methods:\n{bad.to_string()}"
            )
    return counts


def complete_matched_subset(long_table):
    counts = long_table.groupby(["subject", "run"])["method"].nunique()
    complete = counts.loc[counts == len(METHOD_ORDER)].index
    index = pd.MultiIndex.from_frame(long_table[["subject", "run"]])
    complete_index = pd.MultiIndex.from_tuples(complete, names=["subject", "run"])
    return long_table.loc[index.isin(complete_index)].copy()


def to_wide(long_table):
    return (
        long_table.pivot(
            index=["subject", "run"], columns="method", values="localization_error_mm"
        )
        .reindex(columns=METHOD_ORDER)
        .sort_index()
    )


def method_summary(long_table):
    rows = []
    for method in METHOD_ORDER:
        values = long_table.loc[
            long_table["method"].astype(str).eq(method), "localization_error_mm"
        ].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append(
            {
                "method": method,
                "runs": len(values),
                "mean_mm": float(np.mean(values)),
                "standard_deviation_mm": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
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
    for a, b in combinations(METHOD_ORDER, 2):
        pair = wide[[a, b]].dropna()
        if pair.empty:
            continue
        diff = pair[a].to_numpy(float) - pair[b].to_numpy(float)
        q25, median, q75 = np.quantile(diff, [0.25, 0.5, 0.75])
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "paired_runs": len(diff),
                "mean_a_minus_b_mm": float(np.mean(diff)),
                "standard_deviation_a_minus_b_mm": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
                "median_a_minus_b_mm": float(median),
                "q25_a_minus_b_mm": float(q25),
                "q75_a_minus_b_mm": float(q75),
                "method_a_lower_error_runs": int(np.sum(diff < 0)),
                "method_b_lower_error_runs": int(np.sum(diff > 0)),
                "ties": int(np.sum(np.isclose(diff, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def participant_summary(long_table):
    result = (
        long_table.groupby(["subject", "method"], observed=True)
        .agg(
            runs=("localization_error_mm", "size"),
            mean_mm=("localization_error_mm", "mean"),
            standard_deviation_mm=("localization_error_mm", "std"),
            median_mm=("localization_error_mm", "median"),
            minimum_mm=("localization_error_mm", "min"),
            maximum_mm=("localization_error_mm", "max"),
        )
        .reset_index()
    )
    result["method"] = pd.Categorical(result["method"], METHOD_ORDER, ordered=True)
    return result.sort_values(["subject", "method"]).reset_index(drop=True)


def run_summary(wide):
    rows = []
    for (subject, run), values in wide.iterrows():
        available = values.dropna().astype(float)
        if available.empty:
            continue
        minimum = float(available.min())
        maximum = float(available.max())
        min_methods = [m for m, v in available.items() if np.isclose(v, minimum)]
        max_methods = [m for m, v in available.items() if np.isclose(v, maximum)]
        rows.append(
            {
                "subject": subject,
                "run": run,
                "methods_available": len(available),
                "mean_across_methods_mm": float(available.mean()),
                "median_across_methods_mm": float(available.median()),
                "standard_deviation_across_methods_mm": float(available.std(ddof=1)) if len(available) > 1 else np.nan,
                "minimum_error_mm": minimum,
                "minimum_error_method": ";".join(min_methods),
                "maximum_error_mm": maximum,
                "maximum_error_method": ";".join(max_methods),
                "range_across_methods_mm": maximum - minimum,
            }
        )
    return pd.DataFrame(rows)


def lowest_counts(run_table):
    rows = []
    for method in METHOD_ORDER:
        sole = 0
        tied = 0
        for text in run_table["minimum_error_method"]:
            methods = str(text).split(";")
            if method not in methods:
                continue
            if len(methods) == 1:
                sole += 1
            else:
                tied += 1
        rows.append(
            {
                "method": method,
                "sole_lowest_error_runs": sole,
                "tied_lowest_error_runs": tied,
                "total_lowest_error_appearances": sole + tied,
            }
        )
    return pd.DataFrame(rows)


def outlier_table(long_table, multiplier):
    rows = []
    for method in METHOD_ORDER:
        subset = long_table.loc[long_table["method"].astype(str).eq(method)].copy()
        if subset.empty:
            continue
        q25 = subset["localization_error_mm"].quantile(0.25)
        q75 = subset["localization_error_mm"].quantile(0.75)
        iqr = q75 - q25
        threshold = q75 + multiplier * iqr
        for row in subset.loc[subset["localization_error_mm"] > threshold].itertuples(index=False):
            rows.append(
                {
                    "subject": row.subject,
                    "run": row.run,
                    "method": method,
                    "localization_error_mm": float(row.localization_error_mm),
                    "q25_mm": q25,
                    "q75_mm": q75,
                    "iqr_mm": iqr,
                    "upper_outlier_threshold_mm": threshold,
                    "iqr_multiplier": multiplier,
                }
            )
    return pd.DataFrame(rows)


def selected_vs_overall_summary(selected_summary, overall_summary):
    left = selected_summary.add_prefix("selected_").rename(columns={"selected_method": "method"})
    right = overall_summary.add_prefix("overall_").rename(columns={"overall_method": "method"})
    result = left.merge(right, on="method", validate="one_to_one")
    result["selected_minus_overall_median_mm"] = (
        result["selected_median_mm"] - result["overall_median_mm"]
    )
    result["selected_minus_overall_mean_mm"] = (
        result["selected_mean_mm"] - result["overall_mean_mm"]
    )
    result["median_reduction_from_overall_mm"] = -result[
        "selected_minus_overall_median_mm"
    ]
    result["mean_reduction_from_overall_mm"] = -result[
        "selected_minus_overall_mean_mm"
    ]
    result["median_reduction_percent_of_overall"] = np.where(
        result["overall_median_mm"].ne(0),
        result["median_reduction_from_overall_mm"] / result["overall_median_mm"] * 100,
        np.nan,
    )
    result["method"] = pd.Categorical(result["method"], METHOD_ORDER, ordered=True)
    return result.sort_values("method").reset_index(drop=True)


def dataset_summary(selected_long, overall_long, original_grid, additional_grid):
    all_grid = pd.concat([original_grid, additional_grid], ignore_index=True, sort=False)
    run_keys = selected_long[["subject", "run"]].drop_duplicates()
    return pd.DataFrame(
        [
            {
                "participants": selected_long["subject"].nunique(),
                "matched_runs": len(run_keys),
                "methods": len(METHOD_ORDER),
                "selected_method_run_values": len(selected_long),
                "overall_grid_method_run_values": len(overall_long),
                "attempted_grid_solutions_all_methods": len(all_grid),
                "pass_grid_solutions_all_methods": int(all_grid["status"].eq("PASS").sum()),
                "fail_grid_solutions_all_methods": int(all_grid["status"].eq("FAIL").sum()),
            }
        ]
    )


def setup_plot_style():
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")


def save_distribution(long_table, path, title, subtitle, ylabel, dpi):
    fig, ax = plt.subplots(figsize=(15, 7.5))
    if sns is not None:
        sns.boxplot(
            data=long_table,
            x="method",
            y="localization_error_mm",
            order=METHOD_ORDER,
            hue="method",
            hue_order=METHOD_ORDER,
            palette=METHOD_COLORS,
            dodge=False,
            showfliers=False,
            width=0.58,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=long_table,
            x="method",
            y="localization_error_mm",
            order=METHOD_ORDER,
            color="black",
            alpha=0.52,
            size=3.6,
            jitter=0.18,
            ax=ax,
        )
    else:
        data = [
            long_table.loc[long_table["method"].eq(m), "localization_error_mm"].to_numpy()
            for m in METHOD_ORDER
        ]
        boxes = ax.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showfliers=False)
        for patch, method in zip(boxes["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.8)
        rng = np.random.default_rng(0)
        for i, values in enumerate(data, start=1):
            x = i + rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(x, values, s=12, alpha=0.5)

    ax.set_title(f"{title}\n{subtitle}", pad=14)
    ax.set_xlabel("Inverse method")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_selected_vs_overall(summary, path, dpi):
    working = summary.copy()
    working["method"] = working["method"].astype(str)
    x = np.arange(len(METHOD_ORDER))
    selected_map = working.set_index("method")["selected_median_mm"]
    overall_map = working.set_index("method")["overall_median_mm"]
    selected = np.array([selected_map[m] for m in METHOD_ORDER], dtype=float)
    overall = np.array([overall_map[m] for m in METHOD_ORDER], dtype=float)

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, method in enumerate(METHOD_ORDER):
        ax.plot([i, i], [overall[i], selected[i]], color="#999999", linewidth=1.4, zorder=1)
        ax.scatter(i, overall[i], marker="o", s=80, facecolor="white", edgecolor=METHOD_COLORS[method], linewidth=2, zorder=3)
        ax.scatter(i, selected[i], marker="D", s=65, color=METHOD_COLORS[method], zorder=4)
    ax.set_xticks(x, METHOD_ORDER, rotation=25, ha="right")
    ax.set_ylabel("Median localization error across 61 runs (mm)")
    ax.set_title(
        "Selected configuration versus typical within-grid run value\n"
        "Diamonds: selected fixed configuration | Open circles: within-run grid median"
    )
    ax.grid(axis="y", alpha=0.25)
    # lightweight legend
    ax.scatter([], [], marker="D", s=65, color="#666666", label="Selected configuration")
    ax.scatter([], [], marker="o", s=80, facecolor="white", edgecolor="#666666", linewidth=2, label="Overall-run grid median")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_heatmap(pairwise, path, dpi):
    matrix = pd.DataFrame(np.nan, index=METHOD_ORDER, columns=METHOD_ORDER, dtype=float)
    for i in range(len(METHOD_ORDER)):
        matrix.iat[i, i] = 0.0
    for row in pairwise.itertuples(index=False):
        matrix.loc[row.method_a, row.method_b] = row.median_a_minus_b_mm
        matrix.loc[row.method_b, row.method_a] = -row.median_a_minus_b_mm

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    values = matrix.to_numpy(float)
    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER)
    ax.set_title(
        "Selected configurations: median paired error difference\n"
        "Cell = row method - column method; negative means row method is lower"
    )
    for i in range(len(METHOD_ORDER)):
        for j in range(len(METHOD_ORDER)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Median paired difference (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_participant(participant, path, title, dpi):
    pivot = participant.pivot(index="subject", columns="method", values="median_mm")
    pivot = pivot.reindex(columns=METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(pivot.index))
    for method in METHOD_ORDER:
        if method not in pivot.columns:
            continue
        ax.plot(x, pivot[method].to_numpy(float), marker="o", linewidth=1.6, label=method, color=METHOD_COLORS[method])
    ax.set_xticks(x, pivot.index)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Median localization error (mm)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.16), loc="upper center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_lowest_counts(counts, path, dpi):
    plot = counts.set_index("method").reindex(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(METHOD_ORDER))
    bars = ax.bar(x, plot["sole_lowest_error_runs"].to_numpy(float), color=[METHOD_COLORS[m] for m in METHOD_ORDER])
    ax.set_xticks(x, METHOD_ORDER, rotation=25, ha="right")
    ax.set_ylabel("Runs")
    ax.set_title("Selected fixed configuration with the lowest observed error per run")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_run_difficulty(run_table, path, dpi):
    working = run_table.copy().sort_values("median_across_methods_mm").reset_index(drop=True)
    labels = working["subject"].astype(str) + " " + working["run"].astype(str)
    x = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.plot(x, working["median_across_methods_mm"], marker="o", markersize=3.5, linewidth=1.3)
    ax.fill_between(
        x,
        working["minimum_error_mm"].to_numpy(float),
        working["maximum_error_mm"].to_numpy(float),
        alpha=0.15,
        label="min-max across 8 methods",
    )
    ax.set_xlabel("Runs ordered from lower to higher cross-method median error")
    ax.set_ylabel("Localization error (mm)")
    ax.set_title("Run difficulty across the eight selected fixed configurations")
    tick_step = max(1, len(working) // 12)
    ticks = x[::tick_step]
    ax.set_xticks(ticks, labels.iloc[::tick_step], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_tables(output_root, tables):
    table_dir = output_root / "tables"
    for filename, table in tables.items():
        table.to_csv(table_dir / filename, index=False)


def print_summary(selected_summary, overall_summary, comparison, definitions, dataset, output_root):
    print("=" * 78)
    print("SCRIPT 17 — EIGHT-METHOD SELECTED + OVERALL-RUN COMPARISON")
    print("=" * 78)
    row = dataset.iloc[0]
    print(f"Participants             : {int(row['participants'])}")
    print(f"Matched runs             : {int(row['matched_runs'])}")
    print(f"Methods                  : {int(row['methods'])}")
    print(f"Attempted grid solutions : {int(row['attempted_grid_solutions_all_methods']):,}")
    print(f"Grid PASS / FAIL         : {int(row['pass_grid_solutions_all_methods']):,} / {int(row['fail_grid_solutions_all_methods']):,}")

    print("\nSELECTED LOWEST-MEDIAN OBSERVED CONFIGURATIONS")
    show_defs = definitions[[
        "method", "selection_source", "candidate_configurations_searched", "parameter_description"
    ]].copy()
    print(show_defs.to_string(index=False))

    print("\nTWO MAIN METHOD VALUES")
    show = comparison[[
        "method",
        "selected_median_mm",
        "selected_mean_mm",
        "selected_standard_deviation_mm",
        "overall_median_mm",
        "overall_mean_mm",
        "overall_standard_deviation_mm",
        "median_reduction_from_overall_mm",
    ]].copy()
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nInterpretation:")
    print("  selected_* = one fixed lowest-median observed configuration per method")
    print("  overall_*  = median across that method's tested grid within each run, then summarized across runs")
    print("  Both are descriptive; grids differ in size/design, and selected configurations were chosen on these same runs.")
    print(f"\nSaved outputs: {output_root}")


def main():
    args = parse_args()
    if args.outlier_iqr_multiplier < 0:
        raise ValueError("--outlier-iqr-multiplier must be non-negative.")
    if args.dpi < 50:
        raise ValueError("--dpi must be at least 50.")

    original_grid_path = require_file(args.original_grid, "Original Script-09 grid")
    script10_top_path = require_file(args.script10_top, "Script-10 top configurations")
    script16_selected_path = require_file(args.script16_selected, "Script-16 selected configurations")
    script15_root = args.script15_root.expanduser().resolve()

    output_root = prepare_output(args.output_root, args.overwrite)

    original_grid = load_original_grid(original_grid_path)
    additional_grid = load_additional_grids(script15_root)
    script10_selected = choose_script10_selected(script10_top_path)
    script16_selected = choose_script16_selected(script16_selected_path)

    selected_original, definitions_original = extract_original_selected_rows(
        original_grid, script10_selected
    )
    selected_additional, definitions_additional = extract_additional_selected_rows(
        additional_grid, script16_selected
    )

    selected_long = pd.concat(
        [selected_original, selected_additional], ignore_index=True, sort=False
    )
    selected_long["method"] = pd.Categorical(
        selected_long["method"], METHOD_ORDER, ordered=True
    )
    selected_long = selected_long.sort_values(["subject", "run", "method"]).reset_index(drop=True)

    overall_long = create_overall_run_values(original_grid, additional_grid)
    overall_long["method"] = pd.Categorical(
        overall_long["method"], METHOD_ORDER, ordered=True
    )
    overall_long = overall_long.sort_values(["subject", "run", "method"]).reset_index(drop=True)

    selected_counts = validate_matched(selected_long, "Selected-configuration comparison", args.allow_incomplete)
    overall_counts = validate_matched(overall_long, "Overall-grid run comparison", args.allow_incomplete)

    if args.allow_incomplete:
        selected_analysis = complete_matched_subset(selected_long)
        overall_analysis = complete_matched_subset(overall_long)
    else:
        selected_analysis = selected_long.copy()
        overall_analysis = overall_long.copy()

    # Ensure both views use the same matched run set.
    selected_keys = set(map(tuple, selected_analysis[["subject", "run"]].drop_duplicates().to_numpy()))
    overall_keys = set(map(tuple, overall_analysis[["subject", "run"]].drop_duplicates().to_numpy()))
    common_keys = selected_keys & overall_keys
    if not common_keys:
        raise ValueError("Selected and overall-grid views have no common complete runs.")
    if not args.allow_incomplete and selected_keys != overall_keys:
        raise ValueError("Selected and overall-grid analyses do not contain the same run set.")

    common_index = pd.MultiIndex.from_tuples(sorted(common_keys), names=["subject", "run"])
    for name, frame in [("selected", selected_analysis), ("overall", overall_analysis)]:
        idx = pd.MultiIndex.from_frame(frame[["subject", "run"]])
        frame.drop(frame.index[~idx.isin(common_index)], inplace=True)

    selected_wide = to_wide(selected_analysis)
    overall_wide = to_wide(overall_analysis)

    selected_method = method_summary(selected_analysis)
    overall_method = method_summary(overall_analysis)
    comparison = selected_vs_overall_summary(selected_method, overall_method)

    selected_pairwise = pairwise_summary(selected_wide)
    overall_pairwise = pairwise_summary(overall_wide)
    selected_participants = participant_summary(selected_analysis)
    overall_participants = participant_summary(overall_analysis)
    selected_runs = run_summary(selected_wide)
    overall_runs = run_summary(overall_wide)
    selected_lowest = lowest_counts(selected_runs)
    overall_lowest = lowest_counts(overall_runs)
    selected_outliers = outlier_table(selected_analysis, args.outlier_iqr_multiplier)
    overall_outliers = outlier_table(overall_analysis, args.outlier_iqr_multiplier)

    definitions = pd.concat(
        [definitions_original, definitions_additional], ignore_index=True, sort=False
    )
    definitions["method"] = pd.Categorical(definitions["method"], METHOD_ORDER, ordered=True)
    definitions = definitions.sort_values("method").reset_index(drop=True)

    dataset = dataset_summary(
        selected_analysis, overall_analysis, original_grid, additional_grid
    )

    tables = {
        "01_selected_configuration_definitions.csv": definitions,
        "02_selected_configuration_method_summary.csv": selected_method,
        "03_overall_grid_method_summary.csv": overall_method,
        "04_selected_vs_overall_summary.csv": comparison,
        "05_selected_configuration_pairwise_differences.csv": selected_pairwise,
        "06_overall_grid_pairwise_differences.csv": overall_pairwise,
        "07_selected_configuration_participant_summary.csv": selected_participants,
        "08_overall_grid_participant_summary.csv": overall_participants,
        "09_selected_configuration_run_summary.csv": selected_runs,
        "10_overall_grid_run_summary.csv": overall_runs,
        "11_selected_configuration_lowest_error_counts.csv": selected_lowest,
        "12_overall_grid_lowest_typical_counts.csv": overall_lowest,
        "13_selected_configuration_results_long.csv": selected_analysis,
        "14_selected_configuration_results_wide.csv": selected_wide.reset_index(),
        "15_overall_grid_run_values_long.csv": overall_analysis,
        "16_overall_grid_run_values_wide.csv": overall_wide.reset_index(),
        "17_selected_configuration_outliers.csv": selected_outliers,
        "18_overall_grid_outliers.csv": overall_outliers,
        "19_dataset_summary.csv": dataset,
    }
    write_tables(output_root, tables)

    if not args.no_figures:
        setup_plot_style()
        figures = output_root / "figures"
        context = (
            f"{int(dataset.iloc[0]['matched_runs'])} runs | "
            f"{int(dataset.iloc[0]['participants'])} participants | 8 methods"
        )
        save_distribution(
            selected_analysis,
            figures / "01_selected_configuration_distribution.png",
            "Localization error for selected lowest-median observed configurations",
            context,
            "Localization error (mm)",
            args.dpi,
        )
        save_distribution(
            overall_analysis,
            figures / "02_overall_grid_run_distribution.png",
            "Typical localization error across each method's tested parameter grid",
            context + " | one within-run grid median per method",
            "Within-run median localization error across grid (mm)",
            args.dpi,
        )
        save_selected_vs_overall(
            comparison,
            figures / "03_selected_vs_overall_method_medians.png",
            args.dpi,
        )
        save_pairwise_heatmap(
            selected_pairwise,
            figures / "04_selected_pairwise_median_differences.png",
            args.dpi,
        )
        save_participant(
            selected_participants,
            figures / "05_selected_configuration_by_participant.png",
            "Selected fixed configurations: median localization error by participant",
            args.dpi,
        )
        save_lowest_counts(
            selected_lowest,
            figures / "06_selected_configuration_lowest_error_counts.png",
            args.dpi,
        )
        save_participant(
            overall_participants,
            figures / "07_overall_grid_by_participant.png",
            "Typical within-grid run values: median localization error by participant",
            args.dpi,
        )
        save_run_difficulty(
            selected_runs,
            figures / "08_selected_run_difficulty.png",
            args.dpi,
        )

    print_summary(
        selected_method,
        overall_method,
        comparison,
        definitions,
        dataset,
        output_root,
    )


if __name__ == "__main__":
    main()
