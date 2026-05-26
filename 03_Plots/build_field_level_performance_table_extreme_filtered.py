import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


RESULTS_ROOT = r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain"
OUTPUT_DIR = r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures"
OUTPUT_FILE = "field_level_prediction_performance_SL_vs_SSL_pooled-folds.csv"
OUTPUT_FILE_TEST_ONLY = "field_level_prediction_performance_SL_vs_SSL_pooled-folds_test-only_sorted-by-crop.csv"
OUTPUT_FILE_TEST_ONLY_CLEAN = "field_level_prediction_performance_SL_vs_SSL_test-only_clean_sorted-by-ssl-crop-field.csv"
OUTPUT_FILE_TEST_AGGREGATED = "field_level_prediction_performance_SL_vs_SSL_test-only_aggregated_by-crop-and-management.csv"
OUTPUT_FILE_ALLSETS_AGGREGATED = "field_level_prediction_performance_SL_vs_SSL_aggregated_by-crop-and-management_train-val-test.csv"
INVENTORY_FILE = "model_folder_inventory_SL_SSL_for_field_table.csv"


MAIZE_PATCHES = {"50", "68", "20", "110", "74", "90"}
SUNFLOWER_PATCHES = {"76", "95", "115"}
SOY_PATCHES = {"65", "58", "19"}
LUPINE_PATCHES = {"89", "59", "119"}

CONVENTIONAL_MANAGEMENT = {"74", "90", "65", "89", "95_July"}
REDUCED_PESTICIDES_MANAGEMENT = {"50", "68", "58", "59", "76_July"}
REDUCED_PESTI_FLOWER_STRIPS_MANAGEMENT = {"20", "110", "19", "119", "115_July"}

YYHAT_PATTERN = re.compile(
    r"^y-y_hat_(?P<field_id>\d+)(?P<july_suffix>_July)?_(?P<subset>train|val|test)_(?P<fold>\d+)\.csv$"
)
LEGACY_METRIC_PATTERN = re.compile(
    r"^performance_metrics_f3_(?P<field_id>\d+)(?P<july_suffix>_July)?__(?P<subset>train|val|test)_\.csv$"
)


def infer_ssl_sl(folder_name: str) -> str:
    return "SSL" if "VICReg" in folder_name else "SL"


def infer_architecture(folder_name: str) -> str:
    if "VICRegConvNext" in folder_name or "ConvNeXt" in folder_name:
        return "ConvNeXt tiny"
    if "VICReg" in folder_name or "resnet18" in folder_name:
        return "ResNet18"
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


def get_management(field_id: str, folder_name: str) -> str:
    patch_key = f"{field_id}_July" if "July" in folder_name else field_id
    if patch_key in CONVENTIONAL_MANAGEMENT:
        return "Conventional"
    if patch_key in REDUCED_PESTICIDES_MANAGEMENT:
        return "Reduced Pesticides"
    if patch_key in REDUCED_PESTI_FLOWER_STRIPS_MANAGEMENT:
        return "Reduced Pesticides and Flower Strips"
    return "unknown"


def pearsons_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or y_pred.size < 2:
        return np.nan
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))
)


def discover_experiment_folders(results_root: str) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for folder_name in sorted(os.listdir(results_root)):
        folder_path = os.path.join(results_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.startswith("Combined_Figures"):
            continue
        yhat_files = [
            file_name
            for file_name in os.listdir(folder_path)
            if file_name.startswith("y-y_hat_") and file_name.endswith(".csv")
        ]
        if not yhat_files:
            continue
        records.append(
            {
                "folder_name": folder_name,
                "folder_path": folder_path,
                "ssl_sl": infer_ssl_sl(folder_name),
                "architecture": infer_architecture(folder_name),
                "n_y_yhat_files": len(yhat_files),
            }
        )
    return pd.DataFrame(records)


def _compute_metrics_from_pairs(pairs: List[Tuple[float, float]]) -> Dict[str, float]:
    pair_df = pd.DataFrame(pairs, columns=["y", "y_hat"])
    y_true = pair_df["y"].to_numpy(dtype=float)
    y_pred = pair_df["y_hat"].to_numpy(dtype=float)
    return {
        "Pearsons r": pearsons_r(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "n_pooled_pairs": int(y_true.shape[0]),
    }


def mad_custom(x: pd.Series) -> float:
    clean = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size <= 1:
        return 0.0
    median_val = np.median(clean)
    return float(np.median(np.abs(clean - median_val)))


def fmt_plus_minus(center: float, spread: float, digits: int = 2) -> str:
    center_val = pd.to_numeric(pd.Series([center]), errors="coerce").iloc[0]
    spread_val = pd.to_numeric(pd.Series([spread]), errors="coerce").iloc[0]
    if pd.isna(center_val) or pd.isna(spread_val):
        return ""
    return f"{float(center_val):.{digits}f} ± {float(spread_val):.{digits}f}"


def _suffix(base_name: str, metric_mode: str, threshold_abs_yhat: float) -> str:
    clean_name = base_name.replace(".csv", "")
    mode_suffix = "" if metric_mode == "pooled" else f"_{metric_mode}"
    return f"{clean_name}{mode_suffix}_extreme-filtered_thr-{threshold_abs_yhat:g}.csv"


def collect_field_level_metrics(
    experiment_df: pd.DataFrame,
    metric_mode: str,
    threshold_abs_yhat: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for _, experiment in experiment_df.iterrows():
        folder_name = str(experiment["folder_name"])
        folder_path = str(experiment["folder_path"])
        ssl_sl = str(experiment["ssl_sl"])
        architecture = str(experiment["architecture"])

        legacy_r_by_key: Dict[Tuple[str, str], float] = {}
        if metric_mode == "legacy":
            for file_name in os.listdir(folder_path):
                match_legacy = LEGACY_METRIC_PATTERN.match(file_name)
                if match_legacy is None:
                    continue
                field_id = match_legacy.group("field_id")
                subset = match_legacy.group("subset")
                legacy_csv_path = os.path.join(folder_path, file_name)
                legacy_df = pd.read_csv(legacy_csv_path)
                if "global_r" not in legacy_df.columns:
                    continue
                legacy_r = pd.to_numeric(legacy_df["global_r"], errors="coerce").iloc[0]
                legacy_r_by_key[(field_id, subset)] = float(legacy_r) if pd.notna(legacy_r) else np.nan

        grouped_pairs: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        grouped_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
        for file_name in os.listdir(folder_path):
            match = YYHAT_PATTERN.match(file_name)
            if match is None:
                continue
            field_id = match.group("field_id")
            subset = match.group("subset")
            key = (field_id, subset)
            if key not in grouped_pairs:
                grouped_pairs[key] = []
                grouped_counts[key] = {
                    "n_raw_pairs": 0,
                    "n_removed_nonfinite": 0,
                    "n_removed_affected": 0,
                }

            csv_path = os.path.join(folder_path, file_name)
            current_df = pd.read_csv(csv_path)
            if not {"y", "y_hat"}.issubset(set(current_df.columns)):
                continue

            y_values = pd.to_numeric(current_df["y"], errors="coerce").to_numpy(dtype=float)
            y_hat_values = pd.to_numeric(current_df["y_hat"], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(y_values) & np.isfinite(y_hat_values)
            affected_mask = finite_mask & (np.abs(y_hat_values) > threshold_abs_yhat)
            keep_mask = finite_mask & (~affected_mask)

            grouped_counts[key]["n_raw_pairs"] += int(y_values.shape[0])
            grouped_counts[key]["n_removed_nonfinite"] += int((~finite_mask).sum())
            grouped_counts[key]["n_removed_affected"] += int(affected_mask.sum())
            grouped_pairs[key].extend(zip(y_values[keep_mask].tolist(), y_hat_values[keep_mask].tolist()))

        keys_to_emit = set(grouped_pairs.keys())
        if metric_mode == "legacy":
            keys_to_emit = keys_to_emit.union(set(legacy_r_by_key.keys()))

        for (field_id, subset) in sorted(keys_to_emit):
            crop = get_crop(field_id)
            management = get_management(field_id, folder_name)

            pairs = grouped_pairs.get((field_id, subset), [])
            counts = grouped_counts.get(
                (field_id, subset),
                {"n_raw_pairs": 0, "n_removed_nonfinite": 0, "n_removed_affected": 0},
            )
            if pairs:
                metrics = _compute_metrics_from_pairs(pairs)
            else:
                metrics = {
                    "Pearsons r": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "n_pooled_pairs": 0,
                }

            if metric_mode == "legacy":
                metrics["Pearsons r"] = legacy_r_by_key.get((field_id, subset), np.nan)

            rows.append(
                {
                    "field_ID": field_id,
                    "crop": crop,
                    "management_practise": management,
                    "set": subset,
                    "SSL/SL": ssl_sl,
                    "architecture": architecture,
                    "Pearsons r": metrics["Pearsons r"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "n_pooled_pairs": int(metrics["n_pooled_pairs"]),
                    "n_raw_pairs": counts["n_raw_pairs"],
                    "n_removed_nonfinite": counts["n_removed_nonfinite"],
                    "n_removed_affected": counts["n_removed_affected"],
                    "threshold_abs_yhat": float(threshold_abs_yhat),
                    "source_folder": folder_name,
                    "metric_mode": metric_mode,
                }
            )

    return pd.DataFrame(rows)


def add_aggregated_columns(df: pd.DataFrame, metric_mode: str = "pooled") -> pd.DataFrame:
    out_df = df.copy()
    if out_df.empty:
        out_df["aggr_metrics_across_crops"] = pd.Series(dtype=object)
        out_df["aggr_metrics_across_management"] = pd.Series(dtype=object)
        return out_df

    crop_group_cols = ["SSL/SL", "architecture", "set", "crop"]
    mgmt_group_cols = ["SSL/SL", "architecture", "set", "management_practise"]
    metric_cols = ["Pearsons r", "MAE", "RMSE"]

    crop_agg = (
        out_df.groupby(crop_group_cols, dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Pearsons r": "crop_aggr_Pearsons_r",
                "MAE": "crop_aggr_MAE",
                "RMSE": "crop_aggr_RMSE",
            }
        )
    )

    mgmt_source_df = out_df
    if metric_mode == "legacy":
        mgmt_source_df = out_df[out_df["crop"] != "sunflower"]

    mgmt_agg = (
        mgmt_source_df.groupby(mgmt_group_cols, dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Pearsons r": "management_aggr_Pearsons_r",
                "MAE": "management_aggr_MAE",
                "RMSE": "management_aggr_RMSE",
            }
        )
    )

    out_df = out_df.merge(crop_agg, on=crop_group_cols, how="left")
    out_df = out_df.merge(mgmt_agg, on=mgmt_group_cols, how="left")

    out_df["aggr_metrics_across_crops"] = out_df.apply(
        lambda row: json.dumps(
            {
                "Pearsons r": row["crop_aggr_Pearsons_r"],
                "MAE": row["crop_aggr_MAE"],
                "RMSE": row["crop_aggr_RMSE"],
            }
        ),
        axis=1,
    )
    out_df["aggr_metrics_across_management"] = out_df.apply(
        lambda row: json.dumps(
            {
                "Pearsons r": row["management_aggr_Pearsons_r"],
                "MAE": row["management_aggr_MAE"],
                "RMSE": row["management_aggr_RMSE"],
            }
        ),
        axis=1,
    )
    return out_df


def build_test_aggregated_summary(df_test: pd.DataFrame, metric_mode: str = "pooled") -> pd.DataFrame:
    metric_cols = ["Pearsons r", "MAE", "RMSE"]

    crop_summary = (
        df_test.groupby(["SSL/SL", "architecture", "crop"], dropna=False)[metric_cols]
        .agg(
            mean_Pearsons_r=("Pearsons r", "mean"),
            sem_Pearsons_r=("Pearsons r", lambda x: x.sem() if len(x) > 1 else 0.0),
            median_Pearsons_r=("Pearsons r", "median"),
            mad_Pearsons_r=("Pearsons r", mad_custom),
            mean_MAE=("MAE", "mean"),
            sem_MAE=("MAE", lambda x: x.sem() if len(x) > 1 else 0.0),
            mean_RMSE=("RMSE", "mean"),
            sem_RMSE=("RMSE", lambda x: x.sem() if len(x) > 1 else 0.0),
        )
        .reset_index()
        .rename(columns={"crop": "group_value"})
    )
    crop_summary["aggregation_level"] = "crop"

    management_source_df = df_test
    if metric_mode == "legacy":
        management_source_df = df_test[df_test["crop"] != "sunflower"]

    management_summary = (
        management_source_df.groupby(["SSL/SL", "architecture", "management_practise"], dropna=False)[metric_cols]
        .agg(
            mean_Pearsons_r=("Pearsons r", "mean"),
            sem_Pearsons_r=("Pearsons r", lambda x: x.sem() if len(x) > 1 else 0.0),
            median_Pearsons_r=("Pearsons r", "median"),
            mad_Pearsons_r=("Pearsons r", mad_custom),
            mean_MAE=("MAE", "mean"),
            sem_MAE=("MAE", lambda x: x.sem() if len(x) > 1 else 0.0),
            mean_RMSE=("RMSE", "mean"),
            sem_RMSE=("RMSE", lambda x: x.sem() if len(x) > 1 else 0.0),
        )
        .reset_index()
        .rename(columns={"management_practise": "group_value"})
    )
    management_summary["aggregation_level"] = "management"

    summary_df = pd.concat([crop_summary, management_summary], ignore_index=True)
    summary_df["mean_Pearsons_r_pm_sem"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["mean_Pearsons_r"], row["sem_Pearsons_r"]), axis=1
    )
    summary_df["median_Pearsons_r_pm_mad"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["median_Pearsons_r"], row["mad_Pearsons_r"]), axis=1
    )
    summary_df["mean_MAE_pm_sem"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["mean_MAE"], row["sem_MAE"]), axis=1
    )
    summary_df["mean_RMSE_pm_sem"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["mean_RMSE"], row["sem_RMSE"]), axis=1
    )
    summary_df["_ssl_order"] = summary_df["SSL/SL"].map({"SSL": 0, "SL": 1}).fillna(99).astype(int)
    summary_df = summary_df.sort_values(by=["aggregation_level", "group_value", "_ssl_order", "architecture"])
    summary_df = summary_df[
        [
            "aggregation_level",
            "SSL/SL",
            "architecture",
            "group_value",
            "mean_Pearsons_r_pm_sem",
            "median_Pearsons_r_pm_mad",
            "mean_MAE_pm_sem",
            "mean_RMSE_pm_sem",
        ]
    ]
    return summary_df


def build_allsets_aggregated_summary(df_all_sets: pd.DataFrame, metric_mode: str = "pooled") -> pd.DataFrame:
    metric_cols = ["Pearsons r", "MAE", "RMSE"]

    crop_summary = (
        df_all_sets.groupby(["set", "SSL/SL", "architecture", "crop"], dropna=False)[metric_cols]
        .agg(
            mean_Pearsons_r=("Pearsons r", "mean"),
            sem_Pearsons_r=("Pearsons r", lambda x: x.sem() if len(x) > 1 else 0.0),
            median_Pearsons_r=("Pearsons r", "median"),
            mad_Pearsons_r=("Pearsons r", mad_custom),
            mean_MAE=("MAE", "mean"),
            sem_MAE=("MAE", lambda x: x.sem() if len(x) > 1 else 0.0),
            mean_RMSE=("RMSE", "mean"),
            sem_RMSE=("RMSE", lambda x: x.sem() if len(x) > 1 else 0.0),
        )
        .reset_index()
        .rename(columns={"crop": "group_value"})
    )
    crop_summary["aggregation_level"] = "crop"

    management_source_df = df_all_sets
    if metric_mode == "legacy":
        management_source_df = df_all_sets[df_all_sets["crop"] != "sunflower"]

    management_summary = (
        management_source_df.groupby(["set", "SSL/SL", "architecture", "management_practise"], dropna=False)[metric_cols]
        .agg(
            mean_Pearsons_r=("Pearsons r", "mean"),
            sem_Pearsons_r=("Pearsons r", lambda x: x.sem() if len(x) > 1 else 0.0),
            median_Pearsons_r=("Pearsons r", "median"),
            mad_Pearsons_r=("Pearsons r", mad_custom),
            mean_MAE=("MAE", "mean"),
            sem_MAE=("MAE", lambda x: x.sem() if len(x) > 1 else 0.0),
            mean_RMSE=("RMSE", "mean"),
            sem_RMSE=("RMSE", lambda x: x.sem() if len(x) > 1 else 0.0),
        )
        .reset_index()
        .rename(columns={"management_practise": "group_value"})
    )
    management_summary["aggregation_level"] = "management"

    summary_df = pd.concat([crop_summary, management_summary], ignore_index=True)
    summary_df["mean_Pearsons_r_pm_sem"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["mean_Pearsons_r"], row["sem_Pearsons_r"]), axis=1
    )
    summary_df["median_Pearsons_r_pm_mad"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["median_Pearsons_r"], row["mad_Pearsons_r"]), axis=1
    )
    summary_df["mean_MAE_pm_sem"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["mean_MAE"], row["sem_MAE"]), axis=1
    )
    summary_df["mean_RMSE_pm_sem"] = summary_df.apply(
        lambda row: fmt_plus_minus(row["mean_RMSE"], row["sem_RMSE"]), axis=1
    )
    summary_df["_ssl_order"] = summary_df["SSL/SL"].map({"SSL": 0, "SL": 1}).fillna(99).astype(int)
    summary_df = summary_df.sort_values(by=["set", "aggregation_level", "group_value", "_ssl_order", "architecture"])
    summary_df = summary_df[
        [
            "set",
            "aggregation_level",
            "SSL/SL",
            "architecture",
            "group_value",
            "mean_Pearsons_r_pm_sem",
            "median_Pearsons_r_pm_mad",
            "mean_MAE_pm_sem",
            "mean_RMSE_pm_sem",
        ]
    ]
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build field-level performance tables after removing samples with extreme abs(y_hat)."
    )
    parser.add_argument(
        "--metric-mode",
        choices=["pooled"],
        default="pooled",
        help="Metric computation mode (pooled only).",
    )
    parser.add_argument(
        "--results-root",
        default=RESULTS_ROOT,
        help="Root directory containing model folders with y-y_hat CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for generated summary tables.",
    )
    parser.add_argument(
        "--threshold-abs-yhat",
        type=float,
        default=1e6,
        help="Samples with abs(y_hat) > threshold are removed before metric computation.",
    )
    args = parser.parse_args()

    metric_mode = args.metric_mode
    os.makedirs(args.output_dir, exist_ok=True)

    experiment_df = discover_experiment_folders(args.results_root)
    if experiment_df.empty:
        raise RuntimeError(f"No valid experiment folders found in: {args.results_root}")

    inventory_output_path = os.path.join(
        args.output_dir, _suffix(INVENTORY_FILE, metric_mode, args.threshold_abs_yhat)
    )
    experiment_df.to_csv(inventory_output_path, index=False)

    metrics_df = collect_field_level_metrics(
        experiment_df=experiment_df,
        metric_mode=metric_mode,
        threshold_abs_yhat=args.threshold_abs_yhat,
    )
    if metrics_df.empty:
        raise RuntimeError("No field-level y-y_hat metrics could be computed.")

    final_df = add_aggregated_columns(metrics_df, metric_mode=metric_mode)
    final_df["_ssl_order"] = final_df["SSL/SL"].map({"SSL": 0, "SL": 1}).fillna(99).astype(int)
    final_df = final_df.sort_values(
        by=["crop", "_ssl_order", "architecture", "management_practise", "field_ID", "set", "source_folder"]
    )
    export_final_df = final_df.drop(columns=["_ssl_order"], errors="ignore")

    output_path = os.path.join(args.output_dir, _suffix(OUTPUT_FILE, metric_mode, args.threshold_abs_yhat))
    export_final_df.to_csv(output_path, index=False)
    test_only_output_path = os.path.join(
        args.output_dir, _suffix(OUTPUT_FILE_TEST_ONLY, metric_mode, args.threshold_abs_yhat)
    )
    final_df[final_df["set"] == "test"].sort_values(
        by=["crop", "_ssl_order", "architecture", "management_practise", "field_ID", "source_folder"]
    ).drop(columns=["_ssl_order"], errors="ignore").to_csv(test_only_output_path, index=False)
    test_only_clean_output_path = os.path.join(
        args.output_dir, _suffix(OUTPUT_FILE_TEST_ONLY_CLEAN, metric_mode, args.threshold_abs_yhat)
    )
    test_only_clean_df = (
        final_df[final_df["set"] == "test"]
        .sort_values(by=["crop", "_ssl_order", "architecture", "field_ID"])
        .drop(
            columns=[
                "_ssl_order",
                "crop_aggr_Pearsons_r",
                "crop_aggr_MAE",
                "crop_aggr_RMSE",
                "management_aggr_Pearsons_r",
                "management_aggr_MAE",
                "management_aggr_RMSE",
                "aggr_metrics_across_crops",
                "aggr_metrics_across_management",
            ],
            errors="ignore",
        )
    )
    test_only_clean_df.to_csv(test_only_clean_output_path, index=False)
    test_aggregated_output_path = os.path.join(
        args.output_dir, _suffix(OUTPUT_FILE_TEST_AGGREGATED, metric_mode, args.threshold_abs_yhat)
    )
    build_test_aggregated_summary(test_only_clean_df, metric_mode=metric_mode).to_csv(
        test_aggregated_output_path, index=False
    )
    allsets_aggregated_output_path = os.path.join(
        args.output_dir, _suffix(OUTPUT_FILE_ALLSETS_AGGREGATED, metric_mode, args.threshold_abs_yhat)
    )
    all_sets_clean_df = final_df.drop(
        columns=[
            "_ssl_order",
            "crop_aggr_Pearsons_r",
            "crop_aggr_MAE",
            "crop_aggr_RMSE",
            "management_aggr_Pearsons_r",
            "management_aggr_MAE",
            "management_aggr_RMSE",
            "aggr_metrics_across_crops",
            "aggr_metrics_across_management",
        ],
        errors="ignore",
    )
    build_allsets_aggregated_summary(all_sets_clean_df, metric_mode=metric_mode).to_csv(
        allsets_aggregated_output_path, index=False
    )

    removed_affected_total = int(final_df["n_removed_affected"].sum())
    removed_nonfinite_total = int(final_df["n_removed_nonfinite"].sum())
    raw_total = int(final_df["n_raw_pairs"].sum())
    kept_total = int(final_df["n_pooled_pairs"].sum())
    removed_total = removed_affected_total + removed_nonfinite_total
    removed_pct = (100.0 * removed_total / raw_total) if raw_total > 0 else np.nan

    print("\nSaved folder inventory for verification:")
    print(inventory_output_path)
    print("\nMetric mode:")
    print(metric_mode)
    print("\nExtreme-filtered threshold abs(y_hat) >:")
    print(args.threshold_abs_yhat)
    print("\nSaved field-level performance table:")
    print(output_path)
    print("\nSaved test-only (sorted-by-crop) performance table:")
    print(test_only_output_path)
    print("\nSaved test-only clean (sorted-by-ssl-crop-field) performance table:")
    print(test_only_clean_output_path)
    print("\nSaved test-only aggregated (crop + management) performance table:")
    print(test_aggregated_output_path)
    print("\nSaved train-val-test aggregated (crop + management) performance table:")
    print(allsets_aggregated_output_path)
    print("\nFiltering summary (all groups combined):")
    print(f"Kept pooled pairs: {kept_total}/{raw_total}")
    print(f"Removed affected pairs: {removed_affected_total}")
    print(f"Removed non-finite pairs: {removed_nonfinite_total}")
    print(f"Removed total percentage: {removed_pct:.4f}%")
    print("\nQuick verification:")
    print(final_df.groupby(["SSL/SL", "architecture"])["source_folder"].nunique())
    print("\nRows per set:")
    print(final_df["set"].value_counts().sort_index())
    print("\nFirst rows:")
    print(final_df.drop(columns=["_ssl_order"], errors="ignore").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
