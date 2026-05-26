"""
Paired one-sided SSL-vs-SL significance tests on extreme-filtered field-level metrics.

Hypotheses (per metric, paired across fields):
  - Pearson's r (and Fisher z): H0: SSL <= SL; H1: SSL > SL
  - MAE / RMSE (lower is better): H0: SSL >= SL; H1: SSL < SL

Non-significant results are reported as "no significant evidence that SSL is better"
(one-sided tests do not support a claim that SL is better).
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT_CSV = (
    r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures"
    r"\field_level_prediction_performance_SL_vs_SSL_pooled-folds_extreme-filtered_thr-1e+06.csv"
)
DEFAULT_OUTPUT_DIR = r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures"

H0_PEARSON = "SSL <= SL (Pearson r not higher under SSL)"
H1_PEARSON = "SSL > SL (Pearson r higher under SSL)"
H0_ERROR = "SSL >= SL (MAE/RMSE not lower under SSL)"
H1_ERROR = "SSL < SL (MAE/RMSE lower under SSL)"


def mad_custom(x: pd.Series) -> float:
    arr = x.dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))))


def fisher_z(r_values: np.ndarray) -> np.ndarray:
    clipped = np.clip(r_values, -0.999999, 0.999999)
    return np.arctanh(clipped)


def normality_check_shapiro(differences: np.ndarray) -> Dict[str, float]:
    clean = differences[np.isfinite(differences)]
    if clean.size < 3:
        return {"shapiro_w": np.nan, "shapiro_p": np.nan, "normality_ok_0_05": np.nan}
    shapiro_w, shapiro_p = stats.shapiro(clean)
    return {
        "shapiro_w": float(shapiro_w),
        "shapiro_p": float(shapiro_p),
        "normality_ok_0_05": bool(shapiro_p >= 0.05),
    }


def safe_wilcoxon_one_sided(
    x: np.ndarray, y: np.ndarray, alternative: str
) -> Dict[str, float]:
    clean_mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[clean_mask]
    y_clean = y[clean_mask]
    if x_clean.size == 0:
        return {"wilcoxon_statistic": np.nan, "p_value_wilcoxon": np.nan}
    try:
        w_stat, p_val = stats.wilcoxon(
            x_clean, y_clean, zero_method="wilcox", alternative=alternative
        )
        return {"wilcoxon_statistic": float(w_stat), "p_value_wilcoxon": float(p_val)}
    except ValueError:
        return {"wilcoxon_statistic": np.nan, "p_value_wilcoxon": np.nan}


def metric_higher_is_better(metric_name: str) -> bool:
    return metric_name.startswith("Pearsons r")


def to_ssl_improvement(diff_ssl_minus_sl: float, metric_name: str) -> float:
    if pd.isna(diff_ssl_minus_sl):
        return np.nan
    return float(diff_ssl_minus_sl) if metric_higher_is_better(metric_name) else float(-diff_ssl_minus_sl)


def hypothesis_labels(metric_name: str) -> Dict[str, str]:
    if metric_higher_is_better(metric_name):
        return {"h0": H0_PEARSON, "h1": H1_PEARSON}
    return {"h0": H0_ERROR, "h1": H1_ERROR}


def is_significant(row: pd.Series, alpha: float, significance_rule: str) -> bool:
    p_ttest = row.get("p_value_ttest", np.nan)
    p_wilcoxon = row.get("p_value_wilcoxon", np.nan)
    sig_ttest = bool(pd.notna(p_ttest) and float(p_ttest) < alpha)
    sig_wilcoxon = bool(pd.notna(p_wilcoxon) and float(p_wilcoxon) < alpha)

    if significance_rule == "ttest":
        return sig_ttest
    if significance_rule == "wilcoxon":
        return sig_wilcoxon
    if significance_rule == "both":
        return sig_ttest and sig_wilcoxon
    return sig_ttest or sig_wilcoxon


def direction_label_one_sided(mean_ssl_improvement: float, is_statistically_significant: bool) -> str:
    if is_statistically_significant and pd.notna(mean_ssl_improvement) and mean_ssl_improvement > 0:
        return "SSL better (one-sided, significant)"
    return "no significant evidence that SSL is better"


def paired_tests_for_group(
    group_df: pd.DataFrame, group_label: Dict[str, str], include_fisher: bool
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    sl_df = group_df[group_df["SSL/SL"] == "SL"].copy()
    ssl_df = group_df[group_df["SSL/SL"] == "SSL"].copy()
    merge_keys = ["field_ID", "crop", "management_practise", "set", "architecture"]
    paired = sl_df.merge(ssl_df, on=merge_keys, suffixes=("_SL", "_SSL"), how="inner")
    if paired.empty:
        return results

    r_sl = paired["Pearsons r_SL"].to_numpy(dtype=float)
    r_ssl = paired["Pearsons r_SSL"].to_numpy(dtype=float)

    diff_r_raw = r_ssl - r_sl
    t_r_raw, p_r_raw = stats.ttest_rel(
        r_ssl, r_sl, nan_policy="omit", alternative="greater"
    )
    hyp_r = hypothesis_labels("Pearsons r (raw)")
    results.append(
        {
            **group_label,
            "metric": "Pearsons r (raw)",
            "test_sidedness": "one-sided",
            "h0": hyp_r["h0"],
            "h1": hyp_r["h1"],
            "n_pairs": int(np.isfinite(diff_r_raw).sum()),
            "mean_ssl_minus_sl": float(np.nanmean(diff_r_raw)),
            "median_ssl_minus_sl": float(np.nanmedian(diff_r_raw)),
            "t_statistic": float(t_r_raw) if pd.notna(t_r_raw) else np.nan,
            "p_value_ttest": float(p_r_raw) if pd.notna(p_r_raw) else np.nan,
            **safe_wilcoxon_one_sided(r_ssl, r_sl, alternative="greater"),
            **normality_check_shapiro(diff_r_raw),
        }
    )

    z_sl = fisher_z(r_sl)
    z_ssl = fisher_z(r_ssl)
    if include_fisher:
        t_r, p_r = stats.ttest_rel(
            z_ssl, z_sl, nan_policy="omit", alternative="greater"
        )
        diff_r = z_ssl - z_sl
        hyp_f = hypothesis_labels("Pearsons r (Fisher z)")
        results.append(
            {
                **group_label,
                "metric": "Pearsons r (Fisher z)",
                "test_sidedness": "one-sided",
                "h0": hyp_f["h0"],
                "h1": hyp_f["h1"],
                "n_pairs": int(np.isfinite(diff_r).sum()),
                "mean_ssl_minus_sl": float(np.nanmean(diff_r)),
                "median_ssl_minus_sl": float(np.nanmedian(diff_r)),
                "t_statistic": float(t_r) if pd.notna(t_r) else np.nan,
                "p_value_ttest": float(p_r) if pd.notna(p_r) else np.nan,
                **safe_wilcoxon_one_sided(z_ssl, z_sl, alternative="greater"),
                **normality_check_shapiro(diff_r),
            }
        )

    for metric in ["MAE", "RMSE"]:
        sl_vals = paired[f"{metric}_SL"].to_numpy(dtype=float)
        ssl_vals = paired[f"{metric}_SSL"].to_numpy(dtype=float)
        t_stat, p_val = stats.ttest_rel(
            ssl_vals, sl_vals, nan_policy="omit", alternative="less"
        )
        diff = ssl_vals - sl_vals
        hyp = hypothesis_labels(metric)
        results.append(
            {
                **group_label,
                "metric": metric,
                "test_sidedness": "one-sided",
                "h0": hyp["h0"],
                "h1": hyp["h1"],
                "n_pairs": int(np.isfinite(diff).sum()),
                "mean_ssl_minus_sl": float(np.nanmean(diff)),
                "median_ssl_minus_sl": float(np.nanmedian(diff)),
                "t_statistic": float(t_stat) if pd.notna(t_stat) else np.nan,
                "p_value_ttest": float(p_val) if pd.notna(p_val) else np.nan,
                **safe_wilcoxon_one_sided(ssl_vals, sl_vals, alternative="less"),
                **normality_check_shapiro(diff),
            }
        )

    return results


def build_errorbar_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groupings = {
        "overall": ["set", "architecture", "SSL/SL"],
        "crop": ["set", "architecture", "SSL/SL", "crop"],
        "management": ["set", "architecture", "SSL/SL", "management_practise"],
    }
    metrics = ["Pearsons r", "MAE", "RMSE"]

    for level, cols in groupings.items():
        agg = df.groupby(cols, dropna=False)[metrics].agg(["mean", "sem", "median", mad_custom]).reset_index()
        agg.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in agg.columns.to_flat_index()
        ]
        agg = agg.rename(
            columns={
                "Pearsons r_mad_custom": "Pearsons r_mad",
                "MAE_mad_custom": "MAE_mad",
                "RMSE_mad_custom": "RMSE_mad",
            }
        )
        agg["aggregation_level"] = level
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired one-sided SSL-vs-SL significance analysis "
            "(H1: SSL better) using extreme-filtered field-level input."
        )
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Filtered field-level performance CSV path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--output-prefix",
        default="ssl_vs_sl_significance_extreme-filtered_thr-1e+06_one-sided",
        help="Prefix used for generated CSV files.",
    )
    parser.add_argument(
        "--set",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Run tests for one set only or all sets.",
    )
    parser.add_argument(
        "--include-fisher",
        action="store_true",
        help="Include Pearson's r tested in Fisher-z space (default: off).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold alpha (default: 0.05).",
    )
    parser.add_argument(
        "--significance-rule",
        choices=["either", "both", "ttest", "wilcoxon"],
        default="either",
        help="How significance is decided for direction label (default: either).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    if args.set != "all":
        df = df[df["set"] == args.set].copy()

    required_cols = {
        "field_ID",
        "crop",
        "management_practise",
        "set",
        "SSL/SL",
        "architecture",
        "Pearsons r",
        "MAE",
        "RMSE",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    all_results = []
    for (subset, architecture), grp in df.groupby(["set", "architecture"], dropna=False):
        all_results.extend(
            paired_tests_for_group(
                grp,
                {"set": subset, "architecture": architecture},
                include_fisher=args.include_fisher,
            )
        )
    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        raise RuntimeError("No paired SSL/SL rows found for significance tests.")

    result_df["mean_ssl_improvement"] = result_df.apply(
        lambda row: to_ssl_improvement(row["mean_ssl_minus_sl"], row["metric"]), axis=1
    )
    result_df["median_ssl_improvement"] = result_df.apply(
        lambda row: to_ssl_improvement(row["median_ssl_minus_sl"], row["metric"]), axis=1
    )
    result_df["is_significant"] = result_df.apply(
        lambda row: is_significant(
            row,
            alpha=args.alpha,
            significance_rule=args.significance_rule,
        ),
        axis=1,
    )
    result_df["direction_if_significant"] = result_df.apply(
        lambda row: direction_label_one_sided(row["mean_ssl_improvement"], row["is_significant"]),
        axis=1,
    )
    result_df = result_df.sort_values(by=["set", "architecture", "metric"])

    errorbar_df = build_errorbar_summary(df).sort_values(
        by=["aggregation_level", "set", "architecture", "SSL/SL"]
    )

    tests_path = os.path.join(args.output_dir, f"{args.output_prefix}_paired_tests.csv")
    summary_path = os.path.join(args.output_dir, f"{args.output_prefix}_errorbar_summary_sem_mad.csv")
    result_df.to_csv(tests_path, index=False)
    errorbar_df.to_csv(summary_path, index=False)

    print("Saved paired one-sided significance tests (H1: SSL better):")
    print(tests_path)
    print("\nSaved SEM + MAD summary:")
    print(summary_path)
    if "n_removed_affected" in df.columns:
        removed_aff = int(df["n_removed_affected"].sum())
        removed_nonfinite = int(df.get("n_removed_nonfinite", pd.Series([0] * len(df))).sum())
        print("\nInput filtering info from field-level table:")
        print(f"Removed affected pairs (summed rows): {removed_aff}")
        print(f"Removed non-finite pairs (summed rows): {removed_nonfinite}")
    print("\nPreview paired tests:")
    print(result_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
