#!/usr/bin/env python3
"""
Inspect anatomical geometry behind Localize-MI localization errors.

This script does not rerun an inverse solution. It combines the validated
run-method results from Script 05 with the released native-surface and MNI
SEEG coordinates. Its main question is whether large localization errors can
be explained by poor representation of the stimulation point on the cortical
source grid.

The MNI coordinates are retained for later atlas-based interpretation. Named
anatomical labels are intentionally not assigned here because the dataset does
not contain a cortical annotation or a released transform for converting an
arbitrary predicted surface vertex into MNI space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import h5py
import mne
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


METHOD_ORDER = ["MNE", "dSPM", "sLORETA", "eLORETA"]
METHOD_COLORS = {
    "MNE": "#4C78A8",
    "dSPM": "#D62728",
    "sLORETA": "#59A14F",
    "eLORETA": "#8E5A9E",
}


def parse_arguments():
    """Read paths and an optional run selected for the 3-D figure."""

    parser = argparse.ArgumentParser(
        description="Inspect Localize-MI anatomical localization geometry."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/Localize-MI"),
        help="Localize-MI BIDS root.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/tables/inverse_method_results.csv"),
        help="Validated result table created by Script 05.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/anatomical_inspection"),
        help="Directory for Script 07 tables and figures.",
    )
    parser.add_argument(
        "--subject",
        help="Optional subject for a run-specific 3-D figure, e.g. sub-05.",
    )
    parser.add_argument(
        "--run",
        help="Optional run for a run-specific 3-D figure, e.g. run-06.",
    )
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        metavar=("SUBJECT", "RUN"),
        help=(
            "Additional run-specific 3-D figure. This option can be repeated, "
            "for example: --case sub-01 run-01 --case sub-07 run-07."
        ),
    )
    parser.add_argument(
        "--only-selected-run",
        action="store_true",
        help=(
            "Process only --subject/--run for a quick test. Use a separate "
            "--output-dir so a test does not replace the complete tables."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of saved figures.",
    )
    return parser.parse_args()


def require_columns(table, required, table_name):
    """Stop with a clear message if an input table lacks required fields."""

    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def load_validated_results(path):
    """Load successful Script 05 results and validate one row per run-method."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Result table was not found: {path}")

    results = pd.read_csv(path)
    required = {
        "subject", "run", "method", "status", "stimulation_pair",
        "contact_1", "contact_2", "hemisphere", "peak_vertex",
        "peak_time_ms", "estimated_x_m", "estimated_y_m", "estimated_z_m",
        "known_x_m", "known_y_m", "known_z_m",
        "localization_distance_mm", "nearest_source_vertex",
        "nearest_source_x_m", "nearest_source_y_m", "nearest_source_z_m",
        "nearest_source_distance_mm", "geometric_excess_mm",
    }
    require_columns(results, required, "Script 05 result table")

    failed = results.loc[results["status"] != "PASS"]
    if not failed.empty:
        examples = failed[["subject", "run", "method", "status"]].head()
        raise ValueError(
            "Script 07 requires successful results only. Non-PASS rows include:\n"
            f"{examples.to_string(index=False)}"
        )

    if results.duplicated(["subject", "run", "method"]).any():
        raise ValueError("Duplicate subject-run-method results were found.")

    return results.copy()


def electrode_file(dataset, subject, coordinate_space):
    """Construct the released SEEG coordinate-table path."""

    return (
        dataset
        / "derivatives"
        / "epochs"
        / subject
        / "ieeg"
        / f"{subject}_task-seegstim_space-{coordinate_space}_electrodes.tsv"
    )


def contact_midpoint(path, contacts):
    """Calculate a bipolar contact midpoint from one coordinate table."""

    if not path.is_file():
        raise FileNotFoundError(f"Electrode table was not found: {path}")
    electrodes = pd.read_csv(path, sep="\t")
    require_columns(electrodes, {"name", "x", "y", "z"}, str(path))
    indexed = electrodes.set_index("name")
    missing = [contact for contact in contacts if contact not in indexed.index]
    if missing:
        raise ValueError(f"Contacts {missing} were not found in {path}")
    return indexed.loc[contacts, ["x", "y", "z"]].astype(float).mean().to_numpy()


def point_to_mesh_distance(point, coordinates, triangles, chunk_size=50000):
    """Calculate the exact Euclidean distance from a point to a triangle mesh.

    A nearest-vertex calculation can overestimate anatomical depth on a sparse
    mesh. This function therefore tests the triangular faces themselves. The
    returned distance is unsigned: it describes separation from the surface,
    not whether the point lies inside or outside it.
    """

    point = np.asarray(point, dtype=float)
    minimum_squared_distance = np.inf
    minimum_point = None

    for start in range(0, len(triangles), chunk_size):
        face_indices = triangles[start:start + chunk_size]
        a = coordinates[face_indices[:, 0]]
        b = coordinates[face_indices[:, 1]]
        c = coordinates[face_indices[:, 2]]

        # Distances to the three triangle edges also cover all triangle
        # vertices. They are used whenever the perpendicular projection falls
        # outside a triangular face.
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
            edge_distances.append(
                np.einsum("ij,ij->i", point - closest, point - closest)
            )
        stacked_edge_distances = np.stack(edge_distances, axis=0)
        stacked_edge_points = np.stack(edge_points, axis=0)
        closest_edge_index = np.argmin(stacked_edge_distances, axis=0)
        face_indices_in_chunk = np.arange(len(a))
        squared_distance = stacked_edge_distances[
            closest_edge_index, face_indices_in_chunk
        ]
        closest_points = stacked_edge_points[
            closest_edge_index, face_indices_in_chunk
        ].copy()

        # Test the perpendicular projection onto each non-degenerate triangle
        # plane. Barycentric coordinates determine whether that projection is
        # located inside the triangular face.
        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        normal_squared = np.einsum("ij,ij->i", normal, normal)
        valid_plane = normal_squared > np.finfo(float).eps
        signed_numerator = np.einsum("ij,ij->i", point - a, normal)
        projection = np.empty_like(a)
        projection[:] = np.nan
        projection[valid_plane] = (
            point
            - (
                signed_numerator[valid_plane]
                / normal_squared[valid_plane]
            )[:, None]
            * normal[valid_plane]
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
        valid_barycentric = (
            valid_plane
            & (np.abs(barycentric_denominator) > np.finfo(float).eps)
        )
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
        plane_squared_distance[inside] = (
            signed_numerator[inside] ** 2 / normal_squared[inside]
        )
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
    return (
        float(np.sqrt(minimum_squared_distance) * 1000.0),
        np.asarray(minimum_point, dtype=float),
    )


def load_subject_surfaces(dataset, subject):
    """Load left/right pial surfaces and the participant's outer scalp."""

    anat_dir = dataset / "derivatives" / "sourcemodelling" / subject / "anat"
    paths = {
        "lh": anat_dir / f"{subject}_hemi-L_pial.surf.gii",
        "rh": anat_dir / f"{subject}_hemi-R_pial.surf.gii",
        "scalp": anat_dir / f"{subject}_outer_skin.surf.gii",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required anatomical surfaces were not found:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    return {name: load_surface(path) for name, path in paths.items()}


def load_surface_transform_matrix(dataset, subject):
    """Read the same head-to-surface transform used by Script 04."""

    path = (
        dataset / "derivatives" / "sourcemodelling" / subject / "xfm"
        / f"{subject}_from-head_to-surface.h5"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Surface transform was not found: {path}")
    with h5py.File(path, "r") as file:
        transform = np.asarray(file["trans"][()], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4 x 4 transform in {path}.")
    return transform


def load_subject_sensor_coordinates(dataset, subject):
    """Load all EEG sensor coordinates once and transform them to surface space."""

    forward_path = (
        dataset / "derivatives" / "sourcemodelling" / subject / "fwd"
        / f"{subject}_fwd.fif"
    )
    if not forward_path.is_file():
        raise FileNotFoundError(f"Forward model was not found: {forward_path}")
    forward = mne.read_forward_solution(forward_path, verbose=False)
    transform = load_surface_transform_matrix(dataset, subject)
    names = [channel["ch_name"] for channel in forward["info"]["chs"]]
    coordinates = np.asarray(
        [channel["loc"][:3] for channel in forward["info"]["chs"]],
        dtype=float,
    )
    coordinates = mne.transforms.apply_trans(transform, coordinates)
    return dict(zip(names, coordinates))


def nearest_good_sensor_distance(
    dataset, subject, run, stimulation_midpoint, sensor_coordinates
):
    """Measure the stimulation midpoint's distance to the nearest good EEG sensor."""

    channel_path = (
        dataset / "derivatives" / "epochs" / subject / "eeg"
        / f"{subject}_task-seegstim_{run}_channels.tsv"
    )
    if not channel_path.is_file():
        raise FileNotFoundError(f"Channel table is missing: {channel_path}")

    channels = pd.read_csv(channel_path, sep="\t")
    require_columns(channels, {"name", "status"}, str(channel_path))
    good_names = set(
        channels.loc[
            channels["status"].astype(str).str.lower() != "bad", "name"
        ].astype(str)
    )
    good_items = [
        (name, coordinate)
        for name, coordinate in sensor_coordinates.items()
        if name in good_names
    ]
    good_coordinates = np.asarray(
        [coordinate for _, coordinate in good_items], dtype=float
    )
    if good_coordinates.size == 0:
        raise ValueError(f"No good EEG sensor coordinates found for {subject} {run}.")

    distances_mm = (
        np.linalg.norm(good_coordinates - stimulation_midpoint, axis=1)
        * 1000.0
    )
    nearest_index = int(np.argmin(distances_mm))
    return (
        float(distances_mm[nearest_index]),
        good_items[nearest_index][0],
        good_coordinates[nearest_index].copy(),
    )


def build_geometry_tables(results, dataset, checkpoint_file=None):
    """Create run-level and method-level native/MNI geometry tables."""

    dataset = dataset.expanduser().resolve()
    run_records = []
    surface_cache = {}
    sensor_cache = {}
    anatomical_distance_cache = {}

    # Run properties are repeated four times in the Script 05 table, so they
    # are reduced to one record before reading the SEEG files.
    run_rows = results.sort_values(["subject", "run", "method"]).drop_duplicates(
        ["subject", "run"]
    )

    total_runs = len(run_rows)
    for run_number, row in enumerate(run_rows.itertuples(index=False), start=1):
        print(
            f"[{run_number:02d}/{total_runs:02d}] {row.subject} {row.run} "
            f"{row.stimulation_pair}",
            flush=True,
        )
        contacts = [str(row.contact_1), str(row.contact_2)]
        surface_midpoint = contact_midpoint(
            electrode_file(dataset, row.subject, "surface"), contacts
        )
        mni_midpoint = contact_midpoint(
            electrode_file(dataset, row.subject, "MNI152NLin2009aSym"), contacts
        )

        csv_midpoint = np.array(
            [row.known_x_m, row.known_y_m, row.known_z_m], dtype=float
        )
        agreement_mm = float(np.linalg.norm(surface_midpoint - csv_midpoint) * 1000)
        if agreement_mm > 0.05:
            raise ValueError(
                f"Surface midpoint mismatch for {row.subject} {row.run}: "
                f"{agreement_mm:.4f} mm."
            )

        if row.subject not in surface_cache:
            surface_cache[row.subject] = load_subject_surfaces(dataset, row.subject)
            sensor_cache[row.subject] = load_subject_sensor_coordinates(
                dataset, row.subject
            )
        surfaces = surface_cache[row.subject]
        anatomical_key = (row.subject, tuple(contacts), row.hemisphere)
        if anatomical_key not in anatomical_distance_cache:
            pial_coordinates, pial_triangles = surfaces[row.hemisphere]
            scalp_coordinates, scalp_triangles = surfaces["scalp"]
            anatomical_distance_cache[anatomical_key] = (
                point_to_mesh_distance(
                    surface_midpoint, pial_coordinates, pial_triangles
                ),
                point_to_mesh_distance(
                    surface_midpoint, scalp_coordinates, scalp_triangles
                ),
            )
        (
            pial_result,
            scalp_result,
        ) = anatomical_distance_cache[anatomical_key]
        pial_surface_distance_mm, nearest_pial_point = pial_result
        scalp_surface_distance_mm, nearest_scalp_point = scalp_result
        (
            good_sensor_distance_mm,
            nearest_good_sensor_name,
            nearest_good_sensor_point,
        ) = nearest_good_sensor_distance(
            dataset,
            row.subject,
            row.run,
            surface_midpoint,
            sensor_cache[row.subject],
        )

        run_records.append({
            "subject": row.subject,
            "run": row.run,
            "stimulation_pair": row.stimulation_pair,
            "contact_1": contacts[0],
            "contact_2": contacts[1],
            "expected_hemisphere": row.hemisphere,
            "surface_x_m": surface_midpoint[0],
            "surface_y_m": surface_midpoint[1],
            "surface_z_m": surface_midpoint[2],
            "mni_x_m": mni_midpoint[0],
            "mni_y_m": mni_midpoint[1],
            "mni_z_m": mni_midpoint[2],
            "nearest_source_vertex": int(row.nearest_source_vertex),
            "nearest_source_x_m": row.nearest_source_x_m,
            "nearest_source_y_m": row.nearest_source_y_m,
            "nearest_source_z_m": row.nearest_source_z_m,
            "nearest_source_distance_mm": row.nearest_source_distance_mm,
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
            "nearest_sensor_x_m": nearest_good_sensor_point[0],
            "nearest_sensor_y_m": nearest_good_sensor_point[1],
            "nearest_sensor_z_m": nearest_good_sensor_point[2],
            "surface_midpoint_check_mm": agreement_mm,
            "mni_template": "ICBM 2009a Nonlinear Symmetric",
        })
        if checkpoint_file is not None:
            pd.DataFrame(run_records).to_csv(checkpoint_file, index=False)
        print(
            f"    grid={row.nearest_source_distance_mm:.2f} mm, "
            f"pial={pial_surface_distance_mm:.2f} mm, "
            f"scalp={scalp_surface_distance_mm:.2f} mm, "
            f"sensor={good_sensor_distance_mm:.2f} mm",
            flush=True,
        )

    run_geometry = pd.DataFrame(run_records).sort_values(["subject", "run"])

    method_geometry = results.copy()
    method_geometry["left_right_error_mm"] = (
        method_geometry["estimated_x_m"] - method_geometry["known_x_m"]
    ) * 1000
    method_geometry["anterior_posterior_error_mm"] = (
        method_geometry["estimated_y_m"] - method_geometry["known_y_m"]
    ) * 1000
    method_geometry["inferior_superior_error_mm"] = (
        method_geometry["estimated_z_m"] - method_geometry["known_z_m"]
    ) * 1000
    method_geometry["search_hemisphere_constrained"] = True

    method_geometry = method_geometry.merge(
        run_geometry[[
            "subject",
            "run",
            "mni_x_m",
            "mni_y_m",
            "mni_z_m",
            "pial_surface_distance_mm",
            "scalp_surface_distance_mm",
            "nearest_good_eeg_sensor_distance_mm",
        ]],
        on=["subject", "run"],
        how="left",
        validate="many_to_one",
    )
    return run_geometry, method_geometry


def save_depth_measurements_figure(run_geometry, output_file, dpi):
    """Compare source-grid, anatomical-depth, and sensor-distance measurements."""

    measurements = [
        (
            "nearest_source_distance_mm",
            "A. Cortical source-grid representation",
            "Nearest source vertex (mm)",
        ),
        (
            "pial_surface_distance_mm",
            "B. Distance to pial surface",
            "Nearest pial surface (mm)",
        ),
        (
            "scalp_surface_distance_mm",
            "C. Anatomical depth from scalp",
            "Nearest scalp surface (mm)",
        ),
        (
            "nearest_good_eeg_sensor_distance_mm",
            "D. Distance to a usable EEG sensor",
            "Nearest good EEG sensor (mm)",
        ),
    ]
    order = sorted(run_geometry["subject"].unique())
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))

    for axis, (column, title, ylabel) in zip(axes.flat, measurements):
        sns.boxplot(
            data=run_geometry,
            x="subject",
            y=column,
            order=order,
            color="#BDD7E7",
            width=0.55,
            showfliers=False,
            ax=axis,
        )
        sns.stripplot(
            data=run_geometry,
            x="subject",
            y=column,
            order=order,
            color="black",
            size=4.5,
            alpha=0.72,
            jitter=0.18,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Participant")
        axis.set_ylabel(ylabel)

    figure.suptitle("Geometric measurements of stimulation depth", y=1.01)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_distance_relation_figure(method_geometry, output_file, dpi):
    """Relate each depth measurement to inverse localization error."""

    measurements = [
        ("nearest_source_distance_mm", "A. Source-grid distance"),
        ("pial_surface_distance_mm", "B. Pial-surface distance"),
        ("scalp_surface_distance_mm", "C. Scalp-surface distance"),
        ("nearest_good_eeg_sensor_distance_mm", "D. Good-sensor distance"),
    ]
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(2, 2, figsize=(16, 13), sharey=True)

    for axis, (column, title) in zip(axes.flat, measurements):
        sns.scatterplot(
            data=method_geometry,
            x=column,
            y="localization_distance_mm",
            hue="method",
            hue_order=METHOD_ORDER,
            palette=METHOD_COLORS,
            s=62,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
            legend=False,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Distance (mm)")
        axis.set_ylabel("Localization error (mm)")
    handles = [Patch(facecolor=METHOD_COLORS[method], label=method) for method in METHOD_ORDER]
    figure.legend(
        handles=handles,
        title="Inverse method",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("Do source depth and geometry explain localization error?", y=0.99)
    figure.tight_layout(rect=(0, 0.12, 1, 0.96))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def create_depth_summary(run_geometry):
    """Summarize the four geometry measurements across the 61 runs."""

    labels = {
        "nearest_source_distance_mm": "nearest cortical source vertex",
        "pial_surface_distance_mm": "pial surface",
        "scalp_surface_distance_mm": "scalp surface",
        "nearest_good_eeg_sensor_distance_mm": "nearest good EEG sensor",
    }
    records = []
    for column, label in labels.items():
        values = run_geometry[column].astype(float)
        records.append({
            "measurement": label,
            "runs": int(values.notna().sum()),
            "mean_mm": values.mean(),
            "standard_deviation_mm": values.std(ddof=1),
            "median_mm": values.median(),
            "minimum_mm": values.min(),
            "maximum_mm": values.max(),
        })
    return pd.DataFrame(records)


def create_depth_error_correlations(method_geometry):
    """Calculate descriptive per-method depth/error rank correlations."""

    labels = {
        "nearest_source_distance_mm": "nearest cortical source vertex",
        "pial_surface_distance_mm": "pial surface",
        "scalp_surface_distance_mm": "scalp surface",
        "nearest_good_eeg_sensor_distance_mm": "nearest good EEG sensor",
    }
    records = []
    for method in METHOD_ORDER:
        subset = method_geometry.loc[method_geometry["method"] == method]
        for column, label in labels.items():
            result = spearmanr(
                subset[column],
                subset["localization_distance_mm"],
                nan_policy="omit",
            )
            records.append({
                "method": method,
                "depth_measurement": label,
                "runs": len(subset),
                "descriptive_spearman_rho": result.statistic,
            })
    return pd.DataFrame(records)


def save_directional_error_figure(method_geometry, output_file, dpi):
    """Show the direction, not only magnitude, of localization errors."""

    columns = {
        "left_right_error_mm": "Left–right (x)",
        "anterior_posterior_error_mm": "Posterior–anterior (y)",
        "inferior_superior_error_mm": "Inferior–superior (z)",
    }
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(14, 8))
    # Explicit positions keep gaps between methods, including with older
    # seaborn versions that do not support the boxplot "gap" argument.
    offsets = [-0.30, -0.10, 0.10, 0.30]
    for component_index, column in enumerate(columns):
        for method_index, method in enumerate(METHOD_ORDER):
            values = method_geometry.loc[
                method_geometry["method"] == method, column
            ].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            axis.boxplot(
                values,
                positions=[component_index + offsets[method_index]],
                widths=0.15,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": METHOD_COLORS[method], "edgecolor": "#404040"},
                medianprops={"color": "white", "linewidth": 1.8},
                whiskerprops={"color": "#404040"},
                capprops={"color": "#404040"},
            )
    axis.set_xticks(range(len(columns)), list(columns.values()))
    axis.set_xlim(-0.55, len(columns) - 0.45)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_title("Directional displacement of predicted sources")
    axis.set_xlabel("Coordinate direction in native surface space")
    axis.set_ylabel("Predicted minus stimulation coordinate (mm)")
    figure.text(
        0.5,
        0.12,
        "Positive: right, anterior, or superior; negative: left, posterior, or inferior",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    handles = [Patch(facecolor=METHOD_COLORS[method], label=method) for method in METHOD_ORDER]
    figure.legend(
        handles=handles,
        title="Inverse method",
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def load_surface(path):
    """Load a GIFTI surface and express its vertices in metres."""

    image = nib.load(path)
    coordinates = np.asarray(image.darrays[0].data, dtype=float)
    triangles = np.asarray(image.darrays[1].data, dtype=int)
    if np.nanmax(np.abs(coordinates)) > 1:
        coordinates = coordinates / 1000.0
    return coordinates, triangles


def set_equal_3d_limits(axis, point_groups):
    """Use equal physical scales for all three axes of a 3-D plot."""

    points = np.vstack(point_groups)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    centre = (lower + upper) / 2
    radius = max((upper - lower).max() / 2, 0.001)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def save_run_geometry_figure(
    dataset, run_geometry, method_geometry, subject, run, output_file, dpi
):
    """Plot stimulation, nearest source, and all method estimates together."""

    run_match = run_geometry.loc[
        (run_geometry["subject"] == subject) & (run_geometry["run"] == run)
    ]
    method_match = method_geometry.loc[
        (method_geometry["subject"] == subject) & (method_geometry["run"] == run)
    ]
    if len(run_match) != 1 or method_match.empty:
        raise ValueError(f"No complete result was found for {subject} {run}.")
    row = run_match.iloc[0]

    surfaces = []
    anat_dir = dataset / "derivatives" / "sourcemodelling" / subject / "anat"
    for hemisphere in ("L", "R"):
        path = anat_dir / f"{subject}_hemi-{hemisphere}_pial.surf.gii"
        if not path.is_file():
            raise FileNotFoundError(f"Pial surface was not found: {path}")
        surfaces.append(load_surface(path))

    scalp_path = anat_dir / f"{subject}_outer_skin.surf.gii"
    if not scalp_path.is_file():
        raise FileNotFoundError(f"Outer-scalp surface was not found: {scalp_path}")
    scalp_surface = load_surface(scalp_path)

    # A wide canvas reserves a dedicated column for the legend while giving
    # the anatomical panel most of the available height and width.
    figure = plt.figure(figsize=(19, 9.5))
    axis = figure.add_axes([0.015, 0.035, 0.72, 0.90], projection="3d")
    all_points = []
    for coordinates, triangles in surfaces:
        axis.plot_trisurf(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            triangles=triangles,
            color="lightgray",
            alpha=0.075,
            linewidth=0,
            shade=True,
        )
        all_points.append(coordinates)

    scalp_coordinates, scalp_triangles = scalp_surface
    axis.plot_trisurf(
        scalp_coordinates[:, 0],
        scalp_coordinates[:, 1],
        scalp_coordinates[:, 2],
        triangles=scalp_triangles,
        color="#D9C8B4",
        alpha=0.018,
        linewidth=0,
        shade=True,
    )
    # Do not use every scalp vertex to set the viewing limits. Inferior facial
    # and neck vertices would make the brain appear unnecessarily small. The
    # exact nearest scalp and sensor endpoints are added to the limits below.

    stimulation = row[["surface_x_m", "surface_y_m", "surface_z_m"]].to_numpy(float)
    nearest = row[
        ["nearest_source_x_m", "nearest_source_y_m", "nearest_source_z_m"]
    ].to_numpy(float)
    nearest_pial = row[
        ["nearest_pial_x_m", "nearest_pial_y_m", "nearest_pial_z_m"]
    ].to_numpy(float)
    nearest_scalp = row[
        ["nearest_scalp_x_m", "nearest_scalp_y_m", "nearest_scalp_z_m"]
    ].to_numpy(float)
    nearest_sensor = row[
        ["nearest_sensor_x_m", "nearest_sensor_y_m", "nearest_sensor_z_m"]
    ].to_numpy(float)
    axis.scatter(
        *stimulation,
        color="#FFD92F",
        marker="*",
        s=260,
        edgecolor="black",
        linewidth=1.0,
        depthshade=False,
        zorder=10,
    )
    axis.scatter(*nearest, color="#F4A259", s=95, edgecolor="black", depthshade=False)
    axis.plot(*np.vstack([stimulation, nearest]).T, color="#F4A259", linewidth=1.5)
    axis.scatter(
        *nearest_pial, color="#17BECF", marker="D", s=85,
        edgecolor="black", depthshade=False,
    )
    axis.plot(
        *np.vstack([stimulation, nearest_pial]).T,
        color="#17BECF", linewidth=2.0, linestyle=":",
    )
    axis.scatter(
        *nearest_scalp, color="#E377C2", marker="s", s=90,
        edgecolor="black", depthshade=False,
    )
    axis.plot(
        *np.vstack([stimulation, nearest_scalp]).T,
        color="#E377C2", linewidth=2.0, linestyle="-.",
    )
    axis.scatter(
        *nearest_sensor, color="#222222", marker="^", s=100,
        edgecolor="white", linewidth=0.7, depthshade=False,
    )
    axis.plot(
        *np.vstack([stimulation, nearest_sensor]).T,
        color="#222222", linewidth=2.0, linestyle=(0, (4, 2)),
    )

    for method in METHOD_ORDER:
        subset = method_match.loc[method_match["method"] == method]
        if subset.empty:
            continue
        estimate = subset.iloc[0][
            ["estimated_x_m", "estimated_y_m", "estimated_z_m"]
        ].to_numpy(float)
        distance = subset.iloc[0]["localization_distance_mm"]
        axis.scatter(
            *estimate,
            color=METHOD_COLORS[method],
            s=105,
            edgecolor="white",
            linewidth=0.7,
            depthshade=False,
        )
        axis.plot(
            *np.vstack([stimulation, estimate]).T,
            color=METHOD_COLORS[method],
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )
        all_points.append(estimate[None, :])

    set_equal_3d_limits(
        axis,
        all_points + [
            stimulation[None, :],
            nearest[None, :],
            nearest_pial[None, :],
            nearest_scalp[None, :],
            nearest_sensor[None, :],
        ],
    )
    millimetre_formatter = FuncFormatter(
        lambda value, position: f"{value * 1000:.0f}"
    )
    axis.xaxis.set_major_formatter(millimetre_formatter)
    axis.yaxis.set_major_formatter(millimetre_formatter)
    axis.zaxis.set_major_formatter(millimetre_formatter)
    axis.set_xlabel("X: left–right (mm)", labelpad=20)
    axis.set_ylabel("Y: posterior–anterior (mm)", labelpad=22)
    axis.set_zlabel("Z: inferior–superior (mm)", labelpad=18)
    axis.tick_params(axis="x", pad=7)
    axis.tick_params(axis="y", pad=7)
    axis.tick_params(axis="z", pad=7)
    axis.view_init(elev=22, azim=125)
    axis.set_title(
        f"Anatomical localization geometry\n"
        f"{subject} | {run} | {row['stimulation_pair']}",
        pad=10,
    )

    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFD92F",
               markeredgecolor="black", markersize=15,
               label="Stimulation midpoint"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F4A259",
               markeredgecolor="black", markersize=9,
               label=f"Nearest cortical source ({row['nearest_source_distance_mm']:.2f} mm)"),
        Line2D([0], [0], marker="D", color="#17BECF", linestyle=":",
               markeredgecolor="black", markersize=8,
               label=f"Nearest pial point ({row['pial_surface_distance_mm']:.2f} mm)"),
        Line2D([0], [0], marker="s", color="#E377C2", linestyle="-.",
               markeredgecolor="black", markersize=8,
               label=f"Nearest scalp point ({row['scalp_surface_distance_mm']:.2f} mm)"),
        Line2D([0], [0], marker="^", color="#222222", linestyle="--",
               markerfacecolor="#222222", markersize=8,
               label=(
                   f"Nearest good EEG sensor {row['nearest_good_eeg_sensor']} "
                   f"({row['nearest_good_eeg_sensor_distance_mm']:.2f} mm)"
               )),
    ]
    for method in METHOD_ORDER:
        subset = method_match.loc[method_match["method"] == method]
        if not subset.empty:
            distance = subset.iloc[0]["localization_distance_mm"]
            handles.append(
                Line2D([0], [0], marker="o", color=METHOD_COLORS[method],
                       markerfacecolor=METHOD_COLORS[method], markersize=9,
                       linestyle="--", label=f"{method}: {distance:.2f} mm")
            )
    axis.legend(
        handles=handles,
        title="Distances from stimulation midpoint",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.015),
        borderaxespad=0,
        framealpha=0.96,
        fontsize=11,
        title_fontsize=12,
    )
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main():
    """Build anatomical geometry tables and publication-oriented figures."""

    args = parse_arguments()
    if (args.subject is None) != (args.run is None):
        raise ValueError("--subject and --run must be supplied together.")
    requested_cases = []
    if args.subject is not None:
        requested_cases.append((args.subject, args.run))
    if args.case:
        requested_cases.extend(tuple(case) for case in args.case)
    requested_cases = list(dict.fromkeys(requested_cases))

    if args.only_selected_run and not requested_cases:
        raise ValueError(
            "--only-selected-run requires --subject/--run or at least one --case."
        )

    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_validated_results(args.results)
    if args.only_selected_run:
        selected = np.zeros(len(results), dtype=bool)
        for subject, run in requested_cases:
            selected |= (
                (results["subject"] == subject)
                & (results["run"] == run)
            ).to_numpy()
        results = results.loc[selected].copy()
        if results.empty:
            raise ValueError(
                f"No Script 05 results found for requested cases: {requested_cases}."
            )

    run_file = output_dir / "run_anatomical_geometry.csv"
    method_file = output_dir / "method_anatomical_geometry.csv"
    summary_file = output_dir / "depth_measurement_summary.csv"
    correlation_file = output_dir / "depth_error_correlations.csv"
    checkpoint_file = output_dir / "run_anatomical_geometry.partial.csv"

    print("ANATOMICAL LOCATION INSPECTION")
    print("------------------------------")
    print(f"Dataset             : {dataset}")
    print(f"Script 05 results   : {args.results.expanduser().resolve()}")
    print(f"Output directory    : {output_dir}")
    print(f"Runs to process     : {results[['subject', 'run']].drop_duplicates().shape[0]}")
    print(flush=True)

    run_geometry, method_geometry = build_geometry_tables(
        results,
        dataset,
        checkpoint_file=checkpoint_file,
    )

    run_geometry.to_csv(run_file, index=False)
    method_geometry.to_csv(method_file, index=False)
    depth_summary = create_depth_summary(run_geometry)
    depth_correlations = create_depth_error_correlations(method_geometry)
    depth_summary.to_csv(summary_file, index=False)
    depth_correlations.to_csv(correlation_file, index=False)
    if checkpoint_file.is_file():
        checkpoint_file.unlink()

    population_figures_created = len(run_geometry) > 1
    if population_figures_created:
        save_depth_measurements_figure(
            run_geometry,
            output_dir / "01_stimulation_depth_measurements.png",
            args.dpi,
        )
        save_distance_relation_figure(
            method_geometry,
            output_dir / "02_depth_vs_localization_error.png",
            args.dpi,
        )
        save_directional_error_figure(
            method_geometry,
            output_dir / "03_directional_localization_error.png",
            args.dpi,
        )

    run_figure_dir = output_dir / "run_figures"
    run_figures = []
    for subject, run in requested_cases:
        available = (
            (run_geometry["subject"] == subject)
            & (run_geometry["run"] == run)
        ).any()
        if not available:
            raise ValueError(
                f"Requested figure case {subject} {run} was not processed."
            )
        run_figure_dir.mkdir(parents=True, exist_ok=True)
        run_figure = run_figure_dir / f"anatomical_geometry_{subject}_{run}.png"
        save_run_geometry_figure(
            dataset,
            run_geometry,
            method_geometry,
            subject,
            run,
            run_figure,
            args.dpi,
        )
        run_figures.append(run_figure)

    print()
    print("ANATOMICAL LOCATION INSPECTION COMPLETE")
    print("---------------------------------------")
    print(f"Participants        : {run_geometry['subject'].nunique()}")
    print(f"Runs                : {len(run_geometry)}")
    print(f"Run-method results  : {len(method_geometry)}")
    print(f"Run geometry table  : {run_file}")
    print(f"Method table        : {method_file}")
    print(f"Depth summary       : {summary_file}")
    print(f"Depth correlations  : {correlation_file}")
    print(f"Figures             : {output_dir}")
    print(f"Population figures  : {'created' if population_figures_created else 'skipped (one run)'}")
    if run_figures:
        print(f"Run figures         : {run_figure_dir}")
    print()
    print("NOTE")
    print("----")
    print("Inverse peaks were searched within the expected hemisphere by Script 04.")
    print("Therefore, hemisphere agreement cannot be evaluated from this result table.")
    print("Pial distance is distance to the outer cortical sheet; it is not cortical thickness.")
    print("Spearman correlations are descriptive because runs are clustered by participant.")


if __name__ == "__main__":
    main()
