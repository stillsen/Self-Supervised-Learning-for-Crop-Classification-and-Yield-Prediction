# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-supervised learning for crop yield prediction across diversified cropping systems".
This script audits extreme y_hat predictions in y-y_hat CSV files.

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = r"D:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain"
DEFAULT_OUTPUT_DIR = r"D:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures"
DEFAULT_THRESHOLD = 1e6


MAIZE_PATCHES = {"50", "68", "20", "110", "74", "90"}
SUNFLOWER_PATCHES = {"76", "95", "115"}
SOY_PATCHES = {"65", "58", "19"}
LUPINE_PATCHES = {"89", "59", "119"}


def infer_ssl_sl(folder_name: str) -> str:
    return "SSL" if "VICReg" in folder_name else "SL"


def infer_architecture(folder_name: str) -> str:
    if "VICRegConvNext" in folder_name or "ConvNeXt" in folder_name:
        return "ConvNeXt"
    if "VICReg" in folder_name or "resnet18" in folder_name:
        return "resnet18"
    return "unknown"


def get_crop(field_id: str) -> str:
    if field_id in MAIZE_PATCHES:
        return "maize"
    if field_id in SUNFLOWER_PATCHES:
        return "sunflower"
    if field_id in SOY_PATCHES:
        return "soy"
    if field_id in LUPINE_PATCHES:
        return "lupine"
    return "unknown"


def parse_y_yhat_filename(file_name: str) -> Optional[Tuple[str, str, str]]:
    if not (file_name.startswith("y-y_hat_") and file_name.endswith(".csv")):
        return None

    stem = file_name[len("y-y_hat_") : -4]
    parts = stem.split("_")

    if len(parts) == 3:
        field_id, subset, fold = parts
    elif len(parts) == 4 and parts[1] == "July":
        field_id, subset, fold = parts[0], parts[2], parts[3]
    else:
        return None

    if not field_id.isdigit():
        return None
    if subset not in {"train", "val", "test"}:
        return None
    return field_id, subset, fold


def resolve_results_root(user_results_root: str) -> str:
    def count_y_yhat_files(path: str) -> int:
        if not os.path.isdir(path):
            return 0
        count = 0
        for folder_name in os.listdir(path):
            folder_path = os.path.join(path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name.startswith("Combined_Figures"):
                continue
            for file_name in os.listdir(folder_path):
                if file_name.startswith("y-y_hat_") and file_name.endswith(".csv"):
                    count += 1
        return count

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidate_paths = [
        r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain",
        user_results_root,
        os.path.join(project_root, "Output", "2024_SSL", "Results_Pub_Retrain"),
        os.path.join(project_root, "..", "Output", "2024_SSL", "Results_Pub_Retrain"),
        r"D:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain",
    ]
    best_path = None
    best_count = -1
    for candidate in dict.fromkeys(candidate_paths):
        candidate_abs = os.path.abspath(candidate)
        current_count = count_y_yhat_files(candidate_abs)
        if current_count > best_count:
            best_count = current_count
            best_path = candidate_abs
    if best_path is not None and best_count > 0:
        return best_path

    raise RuntimeError(
        "Could not locate a valid results root. Provide --results-root explicitly. "
        f"Given path does not exist: {user_results_root}"
    )


def resolve_output_dir(user_output_dir: str, resolved_results_root: str) -> str:
    if os.path.isdir(user_output_dir):
        return user_output_dir

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidate_paths = [
        os.path.join(resolved_results_root, "Combined_Figures"),
        os.path.join(project_root, "Output", "2024_SSL", "Results_Pub_Retrain", "Combined_Figures"),
        os.path.join(project_root, "..", "Output", "2024_SSL", "Results_Pub_Retrain", "Combined_Figures"),
        r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures",
        r"D:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures",
    ]
    for candidate in candidate_paths:
        candidate_abs = os.path.abspath(candidate)
        if os.path.isdir(candidate_abs):
            return candidate_abs

    # Fall back to path alongside the resolved results root.
    return os.path.join(resolved_results_root, "Combined_Figures")


def collect_file_level_audit(results_root: str, threshold_abs_yhat: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for folder_name in sorted(os.listdir(results_root)):
        folder_path = os.path.join(results_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.startswith("Combined_Figures"):
            continue

        ssl_sl = infer_ssl_sl(folder_name)
        architecture = infer_architecture(folder_name)

        for file_name in os.listdir(folder_path):
            parsed = parse_y_yhat_filename(file_name)
            if parsed is None:
                continue
            field_id, subset, fold = parsed

            csv_path = os.path.join(folder_path, file_name)
            current_df = pd.read_csv(csv_path)
            if "y_hat" not in current_df.columns:
                continue

            y_hat_values = pd.to_numeric(current_df["y_hat"], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(y_hat_values)
            affected_mask = finite_mask & (np.abs(y_hat_values) > threshold_abs_yhat)

            rows.append(
                {
                    "source_folder": folder_name,
                    "SSL/SL": ssl_sl,
                    "architecture": architecture,
                    "crop": get_crop(field_id),
                    "field_ID": field_id,
                    "set": subset,
                    "fold": fold,
                    "n_samples": int(y_hat_values.shape[0]),
                    "n_finite": int(finite_mask.sum()),
                    "n_affected": int(affected_mask.sum()),
                    "pct_affected": float(100.0 * affected_mask.sum() / y_hat_values.shape[0])
                    if y_hat_values.shape[0] > 0
                    else np.nan,
                    "max_abs_y_hat": float(np.nanmax(np.abs(y_hat_values))) if y_hat_values.size > 0 else np.nan,
                    "n_nan": int(np.isnan(y_hat_values).sum()),
                    "n_inf": int(np.isinf(y_hat_values).sum()),
                    "threshold_abs_yhat": float(threshold_abs_yhat),
                }
            )

    return pd.DataFrame(rows)


def aggregate_counts(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)[["n_samples", "n_finite", "n_affected", "n_nan", "n_inf"]]
        .sum()
        .reset_index()
    )
    grouped["pct_affected"] = np.where(
        grouped["n_samples"] > 0,
        100.0 * grouped["n_affected"] / grouped["n_samples"],
        np.nan,
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extreme y_hat predictions in y-y_hat CSV files.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT, help="Root directory containing experiment folders.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for audit CSV files.")
    parser.add_argument(
        "--threshold-abs-yhat",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Absolute y_hat threshold above which a sample is marked as affected.",
    )
    parser.add_argument(
        "--filter-ssl-sl",
        choices=["SSL", "SL", "all"],
        default="all",
        help="Optional filter on SSL/SL before writing outputs.",
    )
    parser.add_argument(
        "--filter-architecture",
        choices=["ConvNeXt", "resnet18", "unknown", "all"],
        default="all",
        help="Optional filter on architecture before writing outputs.",
    )
    args = parser.parse_args()

    results_root = resolve_results_root(args.results_root)
    output_dir = resolve_output_dir(args.output_dir, results_root)
    os.makedirs(output_dir, exist_ok=True)

    file_df = collect_file_level_audit(results_root, args.threshold_abs_yhat)
    if file_df.empty:
        raise RuntimeError(
            f"No y-y_hat files found for audit in resolved results root: {results_root}"
        )

    filtered_df = file_df.copy()
    if args.filter_ssl_sl != "all":
        filtered_df = filtered_df[filtered_df["SSL/SL"] == args.filter_ssl_sl]
    if args.filter_architecture != "all":
        filtered_df = filtered_df[filtered_df["architecture"] == args.filter_architecture]

    if filtered_df.empty:
        raise RuntimeError("No rows remain after applying filters.")

    by_crop_field_set = aggregate_counts(filtered_df, ["crop", "field_ID", "set"])
    by_crop_field = aggregate_counts(filtered_df, ["crop", "field_ID"])
    by_crop = aggregate_counts(filtered_df, ["crop"])
    by_ssl_arch_crop = aggregate_counts(filtered_df, ["SSL/SL", "architecture", "crop"])

    suffix = f"thr-{args.threshold_abs_yhat:g}_ssl-{args.filter_ssl_sl}_arch-{args.filter_architecture}"
    path_file_level = os.path.join(output_dir, f"extreme_yhat_audit_file-level_{suffix}.csv")
    path_crop_field_set = os.path.join(output_dir, f"extreme_yhat_audit_by-crop-field-set_{suffix}.csv")
    path_crop_field = os.path.join(output_dir, f"extreme_yhat_audit_by-crop-field_{suffix}.csv")
    path_crop = os.path.join(output_dir, f"extreme_yhat_audit_by-crop_{suffix}.csv")
    path_ssl_arch_crop = os.path.join(output_dir, f"extreme_yhat_audit_by-ssl-arch-crop_{suffix}.csv")

    filtered_df.to_csv(path_file_level, index=False)
    by_crop_field_set.sort_values(by=["crop", "field_ID", "set"]).to_csv(path_crop_field_set, index=False)
    by_crop_field.sort_values(by=["crop", "field_ID"]).to_csv(path_crop_field, index=False)
    by_crop.sort_values(by=["crop"]).to_csv(path_crop, index=False)
    by_ssl_arch_crop.sort_values(by=["SSL/SL", "architecture", "crop"]).to_csv(path_ssl_arch_crop, index=False)

    print("Saved file-level audit:")
    print(path_file_level)
    print("\nSaved by crop/field/set summary:")
    print(path_crop_field_set)
    print("\nSaved by crop/field summary:")
    print(path_crop_field)
    print("\nSaved by crop summary:")
    print(path_crop)
    print("\nSaved by SSL/architecture/crop summary:")
    print(path_ssl_arch_crop)

    print("\nQuick check (rows with any affected samples):")
    print(by_crop_field_set[by_crop_field_set["n_affected"] > 0].sort_values(by=["crop", "field_ID", "set"]).to_string(index=False))

    total_samples = int(filtered_df["n_samples"].sum())
    total_affected = int(filtered_df["n_affected"].sum())
    total_pct_affected = (100.0 * total_affected / total_samples) if total_samples > 0 else np.nan
    affected_files = int((filtered_df["n_affected"] > 0).sum())
    total_files = int(filtered_df.shape[0])

    print("\n=== Affected sample summary ===")
    print(f"Threshold (abs(y_hat) >): {args.threshold_abs_yhat:g}")
    print(f"Affected samples: {total_affected}/{total_samples} ({total_pct_affected:.4f}%)")
    print(f"Files with any affected samples: {affected_files}/{total_files}")

    by_crop_totals = by_crop[["crop", "n_affected", "n_samples", "pct_affected"]].sort_values(by="crop")
    print("\nAffected samples by crop:")
    print(by_crop_totals.to_string(index=False))


if __name__ == "__main__":
    main()
