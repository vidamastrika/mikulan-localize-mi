#!/usr/bin/env python3
"""
18_make_presentation_figures.py

TERMINAL INPUT OPTIONS
----------------------
Default representative run (sub-04 run-08):
    python scripts/18_make_presentation_figures.py --overwrite

Choose another subject and run:
    python scripts/18_make_presentation_figures.py \
        --subject sub-01 \
        --run run-01 \
        --overwrite

Choose another output root:
    python scripts/18_make_presentation_figures.py \
        --output-root outputs/18_presentation_figures \
        --subject sub-04 \
        --run run-08 \
        --overwrite

Optional input-path overrides:
    --dataset PATH
    --original-grid PATH
    --script10-top PATH
    --script15-root PATH
    --script16-selected PATH

Figure options:
    --dpi INTEGER             PNG resolution; default 400
    --surface-alpha FLOAT     Brain-surface transparency; default 0.10
    --overwrite               Replace only this subject-run output folder

OUTPUT STRUCTURE
----------------
Each subject-run gets its own child folder:
    outputs/18_presentation_figures/
        sub-04_run-08/
            figures/
            csv/
        sub-01_run-01/
            figures/
            csv/

The script does NOT modify Script 07 and does NOT rerun source localization.
It reconstructs the same selected fixed configurations used by Script 17 while
preserving full-grid coordinate columns for the presentation figures.

Figures
-------
1. Anatomical geometry of the known source.
2. Spatial comparison of all eight inverse methods.
3. Localization-error bars for the representative run.
4. Colorful boxplots of selected fixed configurations across 61 runs.
5. Colorful boxplots of overall within-grid run medians across 61 runs.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import shutil

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mne
import nibabel as nib
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


PROJECT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = PROJECT / "data" / "Localize-MI"
DEFAULT_ORIGINAL_GRID = PROJECT / "outputs" / "grid_all_runs" / "grid_results.csv"
DEFAULT_SCRIPT10_TOP = (
    PROJECT / "outputs" / "parameter_grid_analysis" / "tables"
    / "05_top_full_grid_configurations.csv"
)
DEFAULT_SCRIPT15_ROOT = PROJECT / "outputs" / "15_full_additional_method_grid"
DEFAULT_SCRIPT16_SELECTED = (
    PROJECT / "outputs" / "16_additional_method_grid_analysis" / "tables"
    / "10_selected_configurations.csv"
)
DEFAULT_OUTPUT = PROJECT / "outputs" / "18_presentation_figures"

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
METHOD_MARKERS = {
    "MNE": "o",
    "dSPM": "o",
    "sLORETA": "o",
    "eLORETA": "o",
    "Continuous-ECD": "s",
    "MxNE": "D",
    "RAP-MUSIC": "^",
    "LCMV": "P",
}
ORIGINAL_CONFIG_COLUMNS = [
    "method_origin",
    "montage",
    "loose",
    "depth",
    "snr",
    "lambda2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create presentation figures for Localize-MI."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--original-grid", type=Path, default=DEFAULT_ORIGINAL_GRID)
    parser.add_argument("--script10-top", type=Path, default=DEFAULT_SCRIPT10_TOP)
    parser.add_argument("--script15-root", type=Path, default=DEFAULT_SCRIPT15_ROOT)
    parser.add_argument("--script16-selected", type=Path, default=DEFAULT_SCRIPT16_SELECTED)
    parser.add_argument("--subject", default="sub-04")
    parser.add_argument("--run", default="run-08")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--surface-alpha", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


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



def standardize_error_column(table: pd.DataFrame, label: str) -> pd.DataFrame:
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
    table["localization_error_mm"] = pd.to_numeric(table[error_col], errors="coerce")
    return table



def normalize_status(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if "status" not in table.columns:
        table["status"] = "PASS"
    table["status"] = table["status"].astype(str).str.upper()
    return table



def close_series(series, value):
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        target = float(value)
    except (TypeError, ValueError):
        return series.astype(str).eq(str(value))
    return np.isclose(numeric.to_numpy(float), target, rtol=0, atol=1e-10)



def prepare_output(output_root: Path, subject: str, run: str, overwrite: bool):
    output_root = output_root.expanduser().resolve()
    run_root = output_root / f"{subject}_{run}"
    if run_root.exists() and overwrite:
        shutil.rmtree(run_root)
    (run_root / "figures").mkdir(parents=True, exist_ok=True)
    (run_root / "csv").mkdir(parents=True, exist_ok=True)
    return output_root, run_root, run_root / "figures", run_root / "csv"


# -----------------------------------------------------------------------------
# Reconstruct selected rows with preserved coordinates
# -----------------------------------------------------------------------------


def choose_script10_selected(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    require_columns(table, ["method", "montage", "loose", "depth", "snr"], str(path))
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
    if missing or (counts != 1).any():
        raise ValueError("Script-10 selected configuration table is ambiguous.")
    return selected



def choose_script16_selected(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    require_columns(table, ["method", "montage", "configuration_id"], str(path))
    table = table.loc[table["method"].astype(str).isin(ADDITIONAL_METHODS)].copy()
    counts = table.groupby("method").size()
    missing = set(ADDITIONAL_METHODS) - set(table["method"].astype(str))
    if missing or (counts != 1).any():
        raise ValueError("Script-16 selected configuration table is ambiguous.")
    return table



def load_original_grid(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    require_columns(table, ["subject", "run", "method", "montage"], str(path))
    table = standardize_error_column(table, str(path))
    table = normalize_status(table)
    table["subject"] = table["subject"].astype(str)
    table["run"] = table["run"].astype(str)
    table["method"] = table["method"].astype(str)
    table["montage"] = table["montage"].astype(str)
    return table.loc[table["method"].isin(ORIGINAL_METHODS)].copy()



def load_additional_grids(root: Path) -> pd.DataFrame:
    paths = {
        "Continuous-ECD": root / "continuous_ecd" / "grid_results.csv",
        "MxNE": root / "mxne" / "grid_results.csv",
        "RAP-MUSIC": root / "rap_music" / "grid_results.csv",
        "LCMV": root / "lcmv" / "grid_results.csv",
    }
    tables = []
    for method, path in paths.items():
        path = require_file(path, f"{method} Script-15 grid")
        table = pd.read_csv(path, low_memory=False)
        require_columns(table, ["subject", "run", "method", "montage", "configuration_id"], str(path))
        table = standardize_error_column(table, str(path))
        table = normalize_status(table)
        table["subject"] = table["subject"].astype(str)
        table["run"] = table["run"].astype(str)
        table["method"] = table["method"].astype(str)
        table["montage"] = table["montage"].astype(str)
        table["configuration_id"] = table["configuration_id"].astype(str)
        tables.append(table)
    return pd.concat(tables, ignore_index=True, sort=False)



def extract_original_selected_full_rows(grid: pd.DataFrame, selected: pd.DataFrame):
    pieces = []
    for method in ORIGINAL_METHODS:
        cfg = selected.loc[selected["method"].astype(str).eq(method)].iloc[0]
        subset = grid.loc[grid["method"].eq(method)].copy()
        for column in ORIGINAL_CONFIG_COLUMNS:
            if column not in selected.columns:
                continue
            value = cfg.get(column, np.nan)
            if pd.isna(value):
                continue
            if column not in subset.columns:
                raise ValueError(f"Original grid lacks selected-configuration column {column}")
            if column in {"loose", "depth", "snr", "lambda2"}:
                subset = subset.loc[close_series(subset[column], value)]
            else:
                subset = subset.loc[subset[column].astype(str).eq(str(value))]
        subset = subset.loc[subset["status"].eq("PASS")].copy()
        if subset.duplicated(["subject", "run"]).any():
            raise ValueError(f"Selected {method} configuration has duplicate rows per run.")
        if subset.empty:
            raise ValueError(f"No selected rows found for {method}.")
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True, sort=False)



def extract_additional_selected_full_rows(grid: pd.DataFrame, selected: pd.DataFrame):
    pieces = []
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
            raise ValueError(f"No selected rows found for {method}.")
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True, sort=False)



def _coalesce_coordinate(table: pd.DataFrame, candidates, target_name: str, label: str):
    result = pd.Series(np.nan, index=table.index, dtype=float)
    used = []
    for column, scale in candidates:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce") * float(scale)
        fill_mask = result.isna() & values.notna()
        if fill_mask.any():
            result.loc[fill_mask] = values.loc[fill_mask]
            used.append(column)
    if result.notna().sum() == 0:
        tried = [name for name, _ in candidates]
        raise ValueError(f"Could not identify {target_name} in {label}. Tried {tried}")
    return result, used



def identify_coordinate_columns(table: pd.DataFrame, label: str):
    coordinate_candidates = {
        "estimated_x_m": [
            ("estimated_x_m", 1.0), ("selected_x_m", 1.0), ("peak_x_m", 1.0),
            ("source_x_m", 1.0), ("estimated_x", 1.0), ("selected_x", 1.0),
            ("selected_x_mm", 1e-3), ("estimated_x_mm", 1e-3),
            ("peak_x_mm", 1e-3), ("source_x_mm", 1e-3),
        ],
        "estimated_y_m": [
            ("estimated_y_m", 1.0), ("selected_y_m", 1.0), ("peak_y_m", 1.0),
            ("source_y_m", 1.0), ("estimated_y", 1.0), ("selected_y", 1.0),
            ("selected_y_mm", 1e-3), ("estimated_y_mm", 1e-3),
            ("peak_y_mm", 1e-3), ("source_y_mm", 1e-3),
        ],
        "estimated_z_m": [
            ("estimated_z_m", 1.0), ("selected_z_m", 1.0), ("peak_z_m", 1.0),
            ("source_z_m", 1.0), ("estimated_z", 1.0), ("selected_z", 1.0),
            ("selected_z_mm", 1e-3), ("estimated_z_mm", 1e-3),
            ("peak_z_mm", 1e-3), ("source_z_mm", 1e-3),
        ],
        "known_x_m": [
            ("known_x_m", 1.0), ("target_x_m", 1.0), ("stimulation_x_m", 1.0),
            ("true_x_m", 1.0), ("known_x_mm", 1e-3), ("target_x_mm", 1e-3),
            ("stimulation_x_mm", 1e-3), ("true_x_mm", 1e-3),
        ],
        "known_y_m": [
            ("known_y_m", 1.0), ("target_y_m", 1.0), ("stimulation_y_m", 1.0),
            ("true_y_m", 1.0), ("known_y_mm", 1e-3), ("target_y_mm", 1e-3),
            ("stimulation_y_mm", 1e-3), ("true_y_mm", 1e-3),
        ],
        "known_z_m": [
            ("known_z_m", 1.0), ("target_z_m", 1.0), ("stimulation_z_m", 1.0),
            ("true_z_m", 1.0), ("known_z_mm", 1e-3), ("target_z_mm", 1e-3),
            ("stimulation_z_mm", 1e-3), ("true_z_mm", 1e-3),
        ],
    }
    out = table.copy()
    mapping = {}
    for target, candidates in coordinate_candidates.items():
        out[target], used = _coalesce_coordinate(out, candidates, target, label)
        mapping[target] = used
    return out, mapping



def fill_known_coordinates_from_original_methods(selected_all: pd.DataFrame):
    table = selected_all.copy()
    known_cols = ["known_x_m", "known_y_m", "known_z_m"]
    original = table.loc[
        table["method"].astype(str).isin(ORIGINAL_METHODS),
        ["subject", "run", *known_cols],
    ].dropna(subset=known_cols)

    if original.empty:
        raise ValueError("No complete known stimulation coordinates were found.")

    run_known_records = []
    for (subject, run), group in original.groupby(["subject", "run"], sort=False):
        xyz = group[known_cols].to_numpy(dtype=float)
        reference = xyz[0]
        spread_mm = np.linalg.norm(xyz - reference, axis=1) * 1000.0
        if spread_mm.max() > 0.05:
            raise ValueError(
                f"Known coordinates disagree among original methods for {subject} {run}."
            )
        run_known_records.append({
            "subject": subject,
            "run": run,
            "run_known_x_m": reference[0],
            "run_known_y_m": reference[1],
            "run_known_z_m": reference[2],
        })

    run_known = pd.DataFrame(run_known_records)
    table = table.merge(run_known, on=["subject", "run"], how="left", validate="many_to_one")
    for target, fallback in {
        "known_x_m": "run_known_x_m",
        "known_y_m": "run_known_y_m",
        "known_z_m": "run_known_z_m",
    }.items():
        table[target] = pd.to_numeric(table[target], errors="coerce")
        table[target] = table[target].fillna(table[fallback])
    return table.drop(columns=["run_known_x_m", "run_known_y_m", "run_known_z_m"])



def create_overall_run_values(original_grid: pd.DataFrame, additional_grid: pd.DataFrame):
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
            localization_error_mm=("localization_error_mm", "median"),
            within_run_grid_mean_mm=("localization_error_mm", "mean"),
        )
        .reset_index()
    )
    result = counts.merge(errors, on=keys, how="left", validate="one_to_one")
    result["solution_success_rate"] = result["pass_solutions"] / result["attempted_solutions"]
    result["view"] = "overall_grid_run_median"
    return result



def validate_selected_run(table: pd.DataFrame, subject: str, run: str):
    selected = table.loc[
        table["subject"].astype(str).eq(subject)
        & table["run"].astype(str).eq(run)
    ].copy()
    if selected.empty:
        raise ValueError(f"No selected rows found for {subject} {run}.")
    counts = selected.groupby("method").size()
    duplicates = counts.loc[counts != 1]
    if not duplicates.empty:
        raise ValueError(f"Expected one row per method for {subject} {run}, found {duplicates.to_dict()}")
    missing = [method for method in METHOD_ORDER if method not in set(selected["method"])]
    if missing:
        raise ValueError(f"Selected run is missing methods: {missing}")
    selected["method"] = pd.Categorical(selected["method"], categories=METHOD_ORDER, ordered=True)
    return selected.sort_values("method").reset_index(drop=True)



def validate_coordinate_completeness(run_table: pd.DataFrame):
    coordinate_cols = [
        "estimated_x_m", "estimated_y_m", "estimated_z_m",
        "known_x_m", "known_y_m", "known_z_m",
    ]
    bad_mask = run_table[coordinate_cols].isna().any(axis=1)
    if bad_mask.any():
        bad = run_table.loc[bad_mask, ["method", *coordinate_cols]]
        raise ValueError(
            "Some selected methods still have incomplete spatial coordinates:\n"
            + bad.to_string(index=False)
        )
    values = run_table[coordinate_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = run_table.loc[~np.isfinite(values).all(axis=1), ["method", *coordinate_cols]]
        raise ValueError(
            "Some selected methods contain non-finite spatial coordinates:\n"
            + bad.to_string(index=False)
        )


# -----------------------------------------------------------------------------
# Geometry helpers for Figure 1
# -----------------------------------------------------------------------------


def load_surface(path: Path):
    image = nib.load(path)
    coordinates = np.asarray(image.darrays[0].data, dtype=float)
    triangles = np.asarray(image.darrays[1].data, dtype=int)
    if np.nanmax(np.abs(coordinates)) > 1:
        coordinates = coordinates / 1000.0
    return coordinates, triangles



def load_subject_surfaces(dataset: Path, subject: str):
    anat_dir = dataset / "derivatives" / "sourcemodelling" / subject / "anat"
    paths = {
        "lh": anat_dir / f"{subject}_hemi-L_pial.surf.gii",
        "rh": anat_dir / f"{subject}_hemi-R_pial.surf.gii",
        "scalp": anat_dir / f"{subject}_outer_skin.surf.gii",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required anatomical surfaces were not found:\n" + "\n".join(f"  - {p}" for p in missing)
        )
    return {name: load_surface(path) for name, path in paths.items()}



def load_surface_transform_matrix(dataset: Path, subject: str):
    path = (
        dataset / "derivatives" / "sourcemodelling" / subject / "xfm"
        / f"{subject}_from-head_to-surface.h5"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Surface transform was not found: {path}")
    with h5py.File(path, "r") as file:
        transform = np.asarray(file["trans"][()], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4 x 4 transform in {path}")
    return transform



def load_subject_sensor_coordinates(dataset: Path, subject: str):
    forward_path = (
        dataset / "derivatives" / "sourcemodelling" / subject / "fwd"
        / f"{subject}_fwd.fif"
    )
    if not forward_path.is_file():
        raise FileNotFoundError(f"Forward model was not found: {forward_path}")
    forward = mne.read_forward_solution(forward_path, verbose=False)
    transform = load_surface_transform_matrix(dataset, subject)
    names = [channel["ch_name"] for channel in forward["info"]["chs"]]
    coordinates = np.asarray([channel["loc"][:3] for channel in forward["info"]["chs"]], dtype=float)
    coordinates = mne.transforms.apply_trans(transform, coordinates)
    return dict(zip(names, coordinates))



def nearest_good_sensor_distance(dataset: Path, subject: str, run: str, stimulation_midpoint, sensor_coordinates):
    channel_path = (
        dataset / "derivatives" / "epochs" / subject / "eeg"
        / f"{subject}_task-seegstim_{run}_channels.tsv"
    )
    if not channel_path.is_file():
        raise FileNotFoundError(f"Channel table is missing: {channel_path}")
    channels = pd.read_csv(channel_path, sep="\t")
    require_columns(channels, ["name", "status"], str(channel_path))
    good_names = set(
        channels.loc[channels["status"].astype(str).str.lower() != "bad", "name"].astype(str)
    )
    good_items = [
        (name, coordinate) for name, coordinate in sensor_coordinates.items() if name in good_names
    ]
    good_coordinates = np.asarray([coordinate for _, coordinate in good_items], dtype=float)
    if good_coordinates.size == 0:
        raise ValueError(f"No good EEG sensor coordinates found for {subject} {run}.")
    distances_mm = np.linalg.norm(good_coordinates - stimulation_midpoint, axis=1) * 1000.0
    nearest_index = int(np.argmin(distances_mm))
    return (
        float(distances_mm[nearest_index]),
        good_items[nearest_index][0],
        good_coordinates[nearest_index].copy(),
    )



def point_to_mesh_distance(point, coordinates, triangles, chunk_size=50000):
    point = np.asarray(point, dtype=float)
    minimum_squared_distance = np.inf
    minimum_point = None

    for start in range(0, len(triangles), chunk_size):
        face_indices = triangles[start:start + chunk_size]
        a = coordinates[face_indices[:, 0]]
        b = coordinates[face_indices[:, 1]]
        c = coordinates[face_indices[:, 2]]

        edge_distances = []
        edge_points = []
        for edge_start, edge_end in ((a, b), (b, c), (c, a)):
            edge = edge_end - edge_start
            denominator = np.einsum("ij,ij->i", edge, edge)
            valid = denominator > np.finfo(float).eps
            fraction = np.zeros(len(edge), dtype=float)
            fraction[valid] = (
                np.einsum("ij,ij->i", point - edge_start[valid], edge[valid])
                / denominator[valid]
            )
            fraction = np.clip(fraction, 0.0, 1.0)
            closest = edge_start + fraction[:, None] * edge
            edge_points.append(closest)
            edge_distances.append(np.einsum("ij,ij->i", point - closest, point - closest))
        stacked_edge_distances = np.stack(edge_distances, axis=0)
        stacked_edge_points = np.stack(edge_points, axis=0)
        closest_edge_index = np.argmin(stacked_edge_distances, axis=0)
        face_indices_in_chunk = np.arange(len(a))
        squared_distance = stacked_edge_distances[closest_edge_index, face_indices_in_chunk]
        closest_points = stacked_edge_points[closest_edge_index, face_indices_in_chunk].copy()

        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        normal_squared = np.einsum("ij,ij->i", normal, normal)
        valid_plane = normal_squared > np.finfo(float).eps
        signed_numerator = np.einsum("ij,ij->i", point - a, normal)
        projection = np.empty_like(a)
        projection[:] = np.nan
        projection[valid_plane] = (
            point - (signed_numerator[valid_plane] / normal_squared[valid_plane])[:, None] * normal[valid_plane]
        )

        v0 = ab
        v1 = ac
        v2 = projection - a
        dot00 = np.einsum("ij,ij->i", v0, v0)
        dot01 = np.einsum("ij,ij->i", v0, v1)
        dot11 = np.einsum("ij,ij->i", v1, v1)
        dot20 = np.einsum("ij,ij->i", v2, v0)
        dot21 = np.einsum("ij,ij->i", v2, v1)
        barycentric_denominator = dot00 * dot11 - dot01 * dot01
        valid_barycentric = valid_plane & (np.abs(barycentric_denominator) > np.finfo(float).eps)
        barycentric_u = np.full(len(a), np.nan)
        barycentric_v = np.full(len(a), np.nan)
        barycentric_u[valid_barycentric] = (
            dot11[valid_barycentric] * dot20[valid_barycentric]
            - dot01[valid_barycentric] * dot21[valid_barycentric]
        ) / barycentric_denominator[valid_barycentric]
        barycentric_v[valid_barycentric] = (
            dot00[valid_barycentric] * dot21[valid_barycentric]
            - dot01[valid_barycentric] * dot20[valid_barycentric]
        ) / barycentric_denominator[valid_barycentric]
        inside = (
            valid_barycentric
            & (barycentric_u >= -1e-12)
            & (barycentric_v >= -1e-12)
            & (barycentric_u + barycentric_v <= 1.0 + 1e-12)
        )
        plane_squared_distance = np.full(len(a), np.inf)
        plane_squared_distance[inside] = signed_numerator[inside] ** 2 / normal_squared[inside]
        use_plane = plane_squared_distance < squared_distance
        squared_distance[use_plane] = plane_squared_distance[use_plane]
        closest_points[use_plane] = projection[use_plane]

        chunk_minimum_index = int(np.nanargmin(squared_distance))
        chunk_minimum_distance = float(squared_distance[chunk_minimum_index])
        if chunk_minimum_distance < minimum_squared_distance:
            minimum_squared_distance = chunk_minimum_distance
            minimum_point = closest_points[chunk_minimum_index].copy()

    if minimum_point is None:
        raise ValueError("Could not find a valid closest point on the mesh.")
    return float(np.sqrt(minimum_squared_distance) * 1000.0), np.asarray(minimum_point, dtype=float)



def compute_geometry_for_selected_run(dataset: Path, subject: str, run: str, known_point):
    surfaces = load_subject_surfaces(dataset, subject)
    hemisphere = "lh" if known_point[0] < 0 else "rh"
    pial_coordinates, pial_triangles = surfaces[hemisphere]
    scalp_coordinates, scalp_triangles = surfaces["scalp"]

    pial_surface_distance_mm, nearest_pial_point = point_to_mesh_distance(
        known_point, pial_coordinates, pial_triangles
    )
    scalp_surface_distance_mm, nearest_scalp_point = point_to_mesh_distance(
        known_point, scalp_coordinates, scalp_triangles
    )

    sensor_coordinates = load_subject_sensor_coordinates(dataset, subject)
    good_sensor_distance_mm, nearest_good_sensor_name, nearest_sensor_point = nearest_good_sensor_distance(
        dataset, subject, run, known_point, sensor_coordinates
    )

    geometry = {
        "subject": subject,
        "run": run,
        "expected_hemisphere": hemisphere,
        "surface_x_m": known_point[0],
        "surface_y_m": known_point[1],
        "surface_z_m": known_point[2],
        "pial_surface_distance_mm": pial_surface_distance_mm,
        "nearest_pial_x_m": nearest_pial_point[0],
        "nearest_pial_y_m": nearest_pial_point[1],
        "nearest_pial_z_m": nearest_pial_point[2],
        "scalp_surface_distance_mm": scalp_surface_distance_mm,
        "nearest_scalp_x_m": nearest_scalp_point[0],
        "nearest_scalp_y_m": nearest_scalp_point[1],
        "nearest_scalp_z_m": nearest_scalp_point[2],
        "nearest_good_eeg_sensor_distance_mm": good_sensor_distance_mm,
        "nearest_good_eeg_sensor": nearest_good_sensor_name,
        "nearest_sensor_x_m": nearest_sensor_point[0],
        "nearest_sensor_y_m": nearest_sensor_point[1],
        "nearest_sensor_z_m": nearest_sensor_point[2],
    }
    return geometry, surfaces


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def set_equal_3d_limits(axis, point_groups, margin=0.08):
    valid_groups = []
    for group in point_groups:
        group = np.asarray(group, dtype=float)
        if group.size == 0:
            continue
        if group.ndim == 1:
            group = group[None, :]
        mask = np.isfinite(group).all(axis=1)
        if mask.any():
            valid_groups.append(group[mask])
    if not valid_groups:
        raise ValueError("No finite 3-D points available to set axis limits.")

    points = np.vstack(valid_groups)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    centre = (lower + upper) / 2.0
    radius = max((upper - lower).max() / 2.0, 0.001) * (1.0 + margin)

    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1, 1, 1))



def add_surfaces_to_axis(axis, surfaces, surface_alpha):
    point_groups = []
    for name in ("lh", "rh"):
        coordinates, triangles = surfaces[name]
        axis.plot_trisurf(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            triangles=triangles,
            color="#C9C9C9",
            alpha=surface_alpha,
            linewidth=0,
            shade=True,
        )
        point_groups.append(coordinates)
    return point_groups



def save_figure1_source_geometry(geometry, surfaces, subject: str, run: str, output_png: Path, dpi: int, surface_alpha: float):
    stimulation = np.array([
        geometry["surface_x_m"],
        geometry["surface_y_m"],
        geometry["surface_z_m"],
    ], dtype=float)
    nearest_pial = np.array([
        geometry["nearest_pial_x_m"],
        geometry["nearest_pial_y_m"],
        geometry["nearest_pial_z_m"],
    ], dtype=float)
    nearest_scalp = np.array([
        geometry["nearest_scalp_x_m"],
        geometry["nearest_scalp_y_m"],
        geometry["nearest_scalp_z_m"],
    ], dtype=float)
    nearest_sensor = np.array([
        geometry["nearest_sensor_x_m"],
        geometry["nearest_sensor_y_m"],
        geometry["nearest_sensor_z_m"],
    ], dtype=float)

    views = [
        ("Lateral / oblique", 18, 125),
        ("Superior", 82, 90),
        ("Posterior / oblique", 20, 215),
    ]

    figure = plt.figure(figsize=(8.8, 15.8))

    for panel_index, (title, elev, azim) in enumerate(views, start=1):
        axis = figure.add_subplot(3, 1, panel_index, projection="3d")
        point_groups = add_surfaces_to_axis(axis, surfaces, surface_alpha)

        axis.scatter(*stimulation, color="#FFD92F", marker="*", s=360,
                     edgecolor="black", linewidth=1.2, depthshade=False, zorder=20)
        axis.scatter(*nearest_pial, color="#17BECF", marker="D", s=115,
                     edgecolor="black", linewidth=0.9, depthshade=False)
        axis.plot(*np.vstack([stimulation, nearest_pial]).T,
                  color="#17BECF", linewidth=2.0, linestyle=":")

        axis.scatter(*nearest_scalp, color="#E377C2", marker="s", s=118,
                     edgecolor="black", linewidth=0.9, depthshade=False)
        axis.plot(*np.vstack([stimulation, nearest_scalp]).T,
                  color="#E377C2", linewidth=2.0, linestyle="-.")

        axis.scatter(*nearest_sensor, color="#222222", marker="^", s=125,
                     edgecolor="white", linewidth=0.8, depthshade=False)
        axis.plot(*np.vstack([stimulation, nearest_sensor]).T,
                  color="#222222", linewidth=2.0, linestyle=(0, (4, 2)))

        point_groups.extend([
            stimulation[None, :],
            nearest_pial[None, :],
            nearest_scalp[None, :],
            nearest_sensor[None, :],
        ])
        set_equal_3d_limits(axis, point_groups, margin=0.02)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title, fontsize=14, pad=2)
        axis.set_axis_off()

    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFD92F",
               markeredgecolor="black", markersize=15,
               label="Known stimulation/source location"),
        Line2D([0], [0], marker="D", color="#17BECF", linestyle=":",
               markeredgecolor="black", markersize=8,
               label=f"Nearest pial point ({geometry['pial_surface_distance_mm']:.2f} mm)"),
        Line2D([0], [0], marker="s", color="#E377C2", linestyle="-.",
               markeredgecolor="black", markersize=8,
               label=f"Nearest scalp point ({geometry['scalp_surface_distance_mm']:.2f} mm)"),
        Line2D([0], [0], marker="^", color="#222222", linestyle="--",
               markerfacecolor="#222222", markersize=8,
               label=(
                   f"Nearest good EEG sensor {geometry['nearest_good_eeg_sensor']} "
                   f"({geometry['nearest_good_eeg_sensor_distance_mm']:.2f} mm)"
               )),
    ]
    figure.legend(
        handles=handles,
        title="Distances from the known source",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=1,
        frameon=False,
        fontsize=10.8,
        title_fontsize=11.2,
    )
    figure.suptitle(
        "Figure 1. Anatomical geometry of the known source",
        fontsize=19,
        y=0.99,
    )
    figure.text(
        0.5,
        0.965,
        f"{subject} {run} | The source lies inside the brain; distances are shown to the pial surface, scalp, and nearest usable EEG sensor",
        ha="center",
        va="top",
        fontsize=11.3,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14, hspace=0.02)
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_figure2_all_methods(run_table: pd.DataFrame, surfaces, subject: str, run: str, output_png: Path, dpi: int, surface_alpha: float):
    known = run_table[["known_x_m", "known_y_m", "known_z_m"]].iloc[0].to_numpy(float)
    views = [
        ("Lateral / oblique", 18, 125),
        ("Superior", 82, 90),
        ("Posterior / oblique", 20, 215),
    ]
    figure = plt.figure(figsize=(9.2, 16.4))

    for panel_index, (title, elev, azim) in enumerate(views, start=1):
        axis = figure.add_subplot(3, 1, panel_index, projection="3d")
        point_groups = add_surfaces_to_axis(axis, surfaces, surface_alpha)
        axis.scatter(*known, marker="*", s=360, color="#FFD92F",
                     edgecolor="black", linewidth=1.2, depthshade=False, zorder=20)
        point_groups.append(known[None, :])

        for row in run_table.itertuples(index=False):
            method = str(row.method)
            estimate = np.array([row.estimated_x_m, row.estimated_y_m, row.estimated_z_m], dtype=float)
            axis.scatter(*estimate, marker=METHOD_MARKERS[method], s=122,
                         color=METHOD_COLORS[method], edgecolor="white",
                         linewidth=0.9, depthshade=False, zorder=15)
            axis.plot(*np.vstack([known, estimate]).T, color=METHOD_COLORS[method],
                      linestyle="--", linewidth=1.1, alpha=0.72)
            point_groups.append(estimate[None, :])

        set_equal_3d_limits(axis, point_groups, margin=0.02)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title, fontsize=14, pad=2)
        axis.set_axis_off()

    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFD92F",
               markeredgecolor="black", markersize=15,
               label="Known stimulation/source location")
    ]
    for row in run_table.itertuples(index=False):
        method = str(row.method)
        handles.append(
            Line2D([0], [0], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                   markerfacecolor=METHOD_COLORS[method], markeredgecolor="white",
                   linestyle="--", linewidth=1.2, markersize=9,
                   label=f"{method} ({row.localization_error_mm:.2f} mm)")
        )

    figure.legend(
        handles=handles,
        title="Estimated source — localization error",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=False,
        fontsize=10.2,
        title_fontsize=11.0,
    )
    figure.suptitle(
        "Figure 2. Spatial comparison of inverse methods",
        fontsize=19,
        y=0.99,
    )
    figure.text(
        0.5,
        0.965,
        f"{subject} {run} | Star = known stimulation/source location; dashed lines show Euclidean localization error",
        ha="center",
        va="top",
        fontsize=11.3,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.18, hspace=0.02)
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_figure3_run_barplot(run_table: pd.DataFrame, output_png: Path, dpi: int):
    working = run_table[["subject", "run", "method", "localization_error_mm"]].copy()
    working["method"] = pd.Categorical(working["method"], METHOD_ORDER, ordered=True)
    working = working.sort_values("method")

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    bars = ax.bar(
        working["method"].astype(str),
        working["localization_error_mm"],
        color=[METHOD_COLORS[m] for m in working["method"].astype(str)],
    )
    ax.set_ylabel("Localization error (mm)")
    ax.set_xlabel("Inverse method")
    ax.set_title(
        f"Figure 3. Localization error for the representative run\n{working.iloc[0]['subject']} {working.iloc[0]['run']}"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    offset = max(0.3, float(working["localization_error_mm"].max()) * 0.012)
    for bar, value in zip(bars, working["localization_error_mm"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def save_color_box_distribution(
    long_table: pd.DataFrame,
    output_png: Path,
    title: str,
    subtitle: str,
    ylabel: str,
    dpi: int,
):
    """Create a presentation-style colored boxplot with no dots or violins."""
    fig, ax = plt.subplots(figsize=(15.2, 7.6))

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
            linewidth=1.2,
            legend=False,
            ax=ax,
        )
    else:
        data = [
            long_table.loc[
                long_table["method"].astype(str).eq(method),
                "localization_error_mm",
            ].dropna().to_numpy(float)
            for method in METHOD_ORDER
        ]
        boxes = ax.boxplot(
            data,
            positions=np.arange(1, len(METHOD_ORDER) + 1),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
        )
        for patch, method in zip(boxes["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_edgecolor("#404040")
            patch.set_alpha(0.90)
        for median in boxes["medians"]:
            median.set_color("#202020")
            median.set_linewidth(1.8)
        for collection_name in ("whiskers", "caps"):
            for line in boxes[collection_name]:
                line.set_color("#505050")
                line.set_linewidth(1.1)
        ax.set_xticks(np.arange(1, len(METHOD_ORDER) + 1), METHOD_ORDER)

    ax.set_title(f"{title}\n{subtitle}", pad=14)
    ax.set_xlabel("Inverse method")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()

    dataset = args.dataset.expanduser().resolve()
    original_grid_path = require_file(args.original_grid, "Original Script-09 grid")
    script10_top_path = require_file(args.script10_top, "Script-10 top configurations")
    script16_selected_path = require_file(args.script16_selected, "Script-16 selected configurations")
    script15_root = args.script15_root.expanduser().resolve()

    output_root, run_root, figure_dir, csv_dir = prepare_output(
        args.output_root, args.subject, args.run, args.overwrite
    )

    original_grid = load_original_grid(original_grid_path)
    additional_grid = load_additional_grids(script15_root)
    script10_selected = choose_script10_selected(script10_top_path)
    script16_selected = choose_script16_selected(script16_selected_path)

    selected_original = extract_original_selected_full_rows(original_grid, script10_selected)
    selected_additional = extract_additional_selected_full_rows(additional_grid, script16_selected)
    selected_all = pd.concat([selected_original, selected_additional], ignore_index=True, sort=False)
    selected_all, coordinate_mapping = identify_coordinate_columns(selected_all, "combined selected rows")
    selected_all = fill_known_coordinates_from_original_methods(selected_all)

    selected_run = validate_selected_run(selected_all, args.subject, args.run)
    validate_coordinate_completeness(selected_run)

    known_point = selected_run[["known_x_m", "known_y_m", "known_z_m"]].iloc[0].to_numpy(float)
    geometry, surfaces = compute_geometry_for_selected_run(dataset, args.subject, args.run, known_point)
    geometry_table = pd.DataFrame([geometry])

    overall_long = create_overall_run_values(original_grid, additional_grid)
    overall_long["method"] = pd.Categorical(overall_long["method"], METHOD_ORDER, ordered=True)
    selected_distribution = selected_all[["subject", "run", "method", "localization_error_mm"]].copy()
    selected_distribution["method"] = pd.Categorical(selected_distribution["method"], METHOD_ORDER, ordered=True)

    selected_run.to_csv(csv_dir / f"01_selected_run_all_methods_{args.subject}_{args.run}.csv", index=False)
    geometry_table.to_csv(csv_dir / f"02_source_geometry_{args.subject}_{args.run}.csv", index=False)
    selected_distribution.sort_values(["subject", "run", "method"]).to_csv(
        csv_dir / "03_selected_configuration_results_long.csv", index=False
    )
    overall_long.sort_values(["subject", "run", "method"]).to_csv(
        csv_dir / "04_overall_grid_run_values_long.csv", index=False
    )

    save_figure1_source_geometry(
        geometry,
        surfaces,
        args.subject,
        args.run,
        figure_dir / f"01_source_geometry_concept_{args.subject}_{args.run}.png",
        args.dpi,
        args.surface_alpha,
    )
    save_figure2_all_methods(
        selected_run,
        surfaces,
        args.subject,
        args.run,
        figure_dir / f"02_all_methods_spatial_comparison_{args.subject}_{args.run}.png",
        args.dpi,
        args.surface_alpha,
    )
    save_figure3_run_barplot(
        selected_run,
        figure_dir / f"03_selected_run_error_barplot_{args.subject}_{args.run}.png",
        args.dpi,
    )

    context = f"61 runs | 7 participants | 8 methods"
    save_color_box_distribution(
        selected_distribution,
        figure_dir / "04_selected_configuration_distribution_boxplot.png",
        "Figure 4. Selected fixed configuration across 61 runs",
        context,
        "Localization error (mm)",
        args.dpi,
    )
    save_color_box_distribution(
        overall_long,
        figure_dir / "05_overall_grid_distribution_boxplot.png",
        "Figure 5. Typical within-grid run behavior across 61 runs",
        context + " | one within-run grid median per method",
        "Within-run median localization error across grid (mm)",
        args.dpi,
    )

    print("=" * 78)
    print("SCRIPT 18 — PRESENTATION FIGURES")
    print("=" * 78)
    print(f"Selected run            : {args.subject} {args.run}")
    print(f"Root output             : {output_root}")
    print(f"Run-specific folder     : {run_root}")
    print()
    print("Detected coordinate sources:")
    for key, value in coordinate_mapping.items():
        print(f"  {key:<15} <- {', '.join(value)}")
    print()
    print("Selected-run localization errors:")
    print(
        selected_run[["method", "localization_error_mm"]]
        .sort_values("localization_error_mm")
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print()
    print("Geometry for Figure 1:")
    print(geometry_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print("Created figure files:")
    for path in sorted(figure_dir.iterdir()):
        print(f"  {path}")
    print()
    print("Created csv files:")
    for path in sorted(csv_dir.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
