# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-supervised learning for crop yield prediction across diversified cropping systems".
Paired one-sided intra-site transfer significance tests (validation vs test).

Test-vs-validation (per SSL/SL approach):
  - Pearson's r: H0: test >= val; H1: test < val (significant generalization drop)
  - MAE / RMSE: H0: test <= val; H1: test > val (significant error increase on test)

SSL-vs-SL on intrinsic deltas (test - val), paired across fields:
  - Pearson delta: H0: delta_SSL <= delta_SL; H1: delta_SSL > delta_SL
  - MAE/RMSE delta: H0: delta_SSL >= delta_SL; H1: delta_SSL < delta_SL
    (SSL shows smaller deterioration from validation to test)

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
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

H0_TEST_VAL_R = "test >= val (Pearson r not lower on test)"
H1_TEST_VAL_R = "test < val (Pearson r lower on test)"
H0_TEST_VAL_ERR = "test <= val (MAE/RMSE not higher on test)"
H1_TEST_VAL_ERR = "test > val (MAE/RMSE higher on test)"
H0_SSL_DELTA_R = "delta_SSL <= delta_SL (SSL intra-site Pearson delta not larger)"
H1_SSL_DELTA_R = "delta_SSL > delta_SL (SSL intra-site Pearson delta larger)"
H0_SSL_DELTA_ERR = "delta_SSL >= delta_SL (SSL intra-site MAE/RMSE delta not smaller)"
H1_SSL_DELTA_ERR = "delta_SSL < delta_SL (SSL intra-site MAE/RMSE delta smaller)"


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


def pearson_degradation_label_one_sided(mean_delta: float, is_statistically_significant: bool) -> str:
    if is_statistically_significant and pd.notna(mean_delta) and mean_delta < 0:
        return "test < val on average (one-sided, significant degradation)"
    return "no significant evidence of test < val (Pearson r degradation)"


def error_increase_label_one_sided(mean_delta: float, is_statistically_significant: bool) -> str:
    if is_statistically_significant and pd.notna(mean_delta) and mean_delta > 0:
        return "test > val on average (one-sided, significant error increase)"
    return "no significant evidence of test > val (MAE/RMSE increase)"


def paired_tests_test_vs_val_raw(
    r_test: np.ndarray, r_val: np.ndarray, group_label: Dict[str, str]
) -> Dict[str, object]:
    diff_raw = r_test - r_val
    t_stat, p_val = stats.ttest_rel(
        r_test, r_val, nan_policy="omit", alternative="less"
    )
    return {
        **group_label,
        "metric": "Pearsons r (raw)",
        "test_sidedness": "one-sided",
        "h0": H0_TEST_VAL_R,
        "h1": H1_TEST_VAL_R,
        "n_pairs": int(np.sum(np.isfinite(diff_raw))),
        "mean_test_minus_val": float(np.nanmean(diff_raw)),
        "median_test_minus_val": float(np.nanmedian(diff_raw)),
        "t_statistic": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p_value_ttest": float(p_val) if pd.notna(p_val) else np.nan,
        **safe_wilcoxon_one_sided(r_test, r_val, alternative="less"),
        **normality_check_shapiro(diff_raw),
    }


def paired_tests_test_vs_val_fisher(
    r_test: np.ndarray, r_val: np.ndarray, group_label: Dict[str, str]
) -> Dict[str, object]:
    z_test = fisher_z(r_test)
    z_val = fisher_z(r_val)
    diff_z = z_test - z_val
    t_stat, p_val = stats.ttest_rel(
        z_test, z_val, nan_policy="omit", alternative="less"
    )
    return {
        **group_label,
        "metric": "Pearsons r (Fisher z)",
        "test_sidedness": "one-sided",
        "h0": H0_TEST_VAL_R,
        "h1": H1_TEST_VAL_R,
        "n_pairs": int(np.sum(np.isfinite(diff_z))),
        "mean_test_minus_val": float(np.nanmean(diff_z)),
        "median_test_minus_val": float(np.nanmedian(diff_z)),
        "t_statistic": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p_value_ttest": float(p_val) if pd.notna(p_val) else np.nan,
        **safe_wilcoxon_one_sided(z_test, z_val, alternative="less"),
        **normality_check_shapiro(diff_z),
    }


def paired_tests_maermse(
    metric: str, v_test: np.ndarray, v_val: np.ndarray, group_label: Dict[str, str]
) -> Dict[str, object]:
    diff = v_test - v_val
    t_stat, p_val = stats.ttest_rel(
        v_test, v_val, nan_policy="omit", alternative="greater"
    )
    return {
        **group_label,
        "metric": metric,
        "test_sidedness": "one-sided",
        "h0": H0_TEST_VAL_ERR,
        "h1": H1_TEST_VAL_ERR,
        "n_pairs": int(np.sum(np.isfinite(diff))),
        "mean_test_minus_val": float(np.nanmean(diff)),
        "median_test_minus_val": float(np.nanmedian(diff)),
        "t_statistic": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p_value_ttest": float(p_val) if pd.notna(p_val) else np.nan,
        **safe_wilcoxon_one_sided(v_test, v_val, alternative="greater"),
        **normality_check_shapiro(diff),
    }


def per_approach_paired_analysis(
    merged_tv: pd.DataFrame, architecture: str, ssl_sl: str, include_fisher: bool
) -> List[Dict[str, object]]:
    label = {"architecture": architecture, "SSL/SL": ssl_sl}
    r_t = merged_tv["Pearsons r_test"].to_numpy(dtype=float)
    r_v = merged_tv["Pearsons r_val"].to_numpy(dtype=float)
    results: List[Dict[str, object]] = [paired_tests_test_vs_val_raw(r_t, r_v, label)]
    if include_fisher:
        results.append(paired_tests_test_vs_val_fisher(r_t, r_v, label))

    for m in ["MAE", "RMSE"]:
        results.append(
            paired_tests_maermse(
                m,
                merged_tv[f"{m}_test"].to_numpy(dtype=float),
                merged_tv[f"{m}_val"].to_numpy(dtype=float),
                label,
            )
        )
    return results


def merge_test_val_wide(df_tv: pd.DataFrame) -> pd.DataFrame:
    """Inner-merge field-level validation and test rows."""
    keys = ["field_ID", "crop", "management_practise", "architecture", "SSL/SL"]
    d_test = df_tv[df_tv["set"].astype(str).str.lower() == "test"].drop(columns=["set"]).copy()
    d_val = df_tv[df_tv["set"].astype(str).str.lower() == "val"].drop(columns=["set"]).copy()
    merged = d_test.merge(d_val, on=keys, how="inner", suffixes=("_test", "_val"))
    if merged.empty:
        return merged
    rt = merged["Pearsons r_test"].to_numpy(dtype=float)
    rv = merged["Pearsons r_val"].to_numpy(dtype=float)
    merged["diff_Pearsons_r"] = rt - rv
    merged["diff_Pearsons_z"] = fisher_z(rt) - fisher_z(rv)
    merged["diff_MAE"] = merged["MAE_test"].to_numpy(dtype=float) - merged["MAE_val"].to_numpy(dtype=float)
    merged["diff_RMSE"] = merged["RMSE_test"].to_numpy(dtype=float) - merged["RMSE_val"].to_numpy(
        dtype=float
    )
    return merged


def ssl_vs_sl_on_intrinsic_deltas(
    merged_wide: pd.DataFrame, architecture: str, include_fisher: bool
) -> List[Dict[str, object]]:
    """
    One-sided paired tests: H1 favors SSL (larger Pearson delta or smaller MAE/RMSE delta).
    """
    merge_align = ["field_ID", "crop", "management_practise", "architecture"]

    subset = merged_wide[merged_wide["architecture"].astype(str) == str(architecture)].copy()
    cols = merge_align + [
        "diff_Pearsons_r",
        "diff_Pearsons_z",
        "diff_MAE",
        "diff_RMSE",
        "SSL/SL",
    ]
    sub = subset[cols]

    ssl_part = (
        sub[sub["SSL/SL"] == "SSL"]
        .drop(columns=["SSL/SL"])
        .rename(
            columns={
                "diff_Pearsons_r": "diff_Pearsons_r_SSL",
                "diff_Pearsons_z": "diff_Pearsons_z_SSL",
                "diff_MAE": "diff_MAE_SSL",
                "diff_RMSE": "diff_RMSE_SSL",
            }
        )
    )
    sl_part = (
        sub[sub["SSL/SL"] == "SL"]
        .drop(columns=["SSL/SL"])
        .rename(
            columns={
                "diff_Pearsons_r": "diff_Pearsons_r_SL",
                "diff_Pearsons_z": "diff_Pearsons_z_SL",
                "diff_MAE": "diff_MAE_SL",
                "diff_RMSE": "diff_RMSE_SL",
            }
        )
    )
    comp = ssl_part.merge(sl_part, on=merge_align, how="inner")
    results: List[Dict[str, object]] = []
    if comp.empty:
        return results

    label = {"architecture": architecture, "comparison": "delta_SSL_minus_delta_SL (paired strata)"}

    def one_row(
        col_ssl: str,
        col_sl: str,
        metric_label: str,
        ttest_alternative: str,
        h0: str,
        h1: str,
    ) -> Dict[str, object]:
        d_ssl_vec = comp[col_ssl].to_numpy(dtype=float)
        d_sl_vec = comp[col_sl].to_numpy(dtype=float)
        delta_pair = d_ssl_vec - d_sl_vec
        t_stat, p_val = stats.ttest_rel(
            d_ssl_vec, d_sl_vec, nan_policy="omit", alternative=ttest_alternative
        )
        return {
            **label,
            "metric": metric_label,
            "test_sidedness": "one-sided",
            "h0": h0,
            "h1": h1,
            "n_pairs": int(np.sum(np.isfinite(delta_pair))),
            "mean_delta_ssl_minus_sl": float(np.nanmean(delta_pair)),
            "median_delta_ssl_minus_sl": float(np.nanmedian(delta_pair)),
            "t_statistic": t_stat if pd.notna(t_stat) else np.nan,
            "p_value_ttest": float(p_val) if pd.notna(p_val) else np.nan,
            **safe_wilcoxon_one_sided(d_ssl_vec, d_sl_vec, alternative=ttest_alternative),
            **normality_check_shapiro(delta_pair),
        }

    results.append(
        one_row(
            "diff_Pearsons_r_SSL",
            "diff_Pearsons_r_SL",
            "intra-site delta Pearson r (raw), SSL - SL",
            "greater",
            H0_SSL_DELTA_R,
            H1_SSL_DELTA_R,
        )
    )
    if include_fisher:
        results.append(
            one_row(
                "diff_Pearsons_z_SSL",
                "diff_Pearsons_z_SL",
                "intra-site delta Pearson r (Fisher z), SSL - SL",
                "greater",
                H0_SSL_DELTA_R,
                H1_SSL_DELTA_R,
            )
        )

    results.append(
        one_row(
            "diff_MAE_SSL",
            "diff_MAE_SL",
            "intra-site delta MAE (test - val), SSL - SL",
            "less",
            H0_SSL_DELTA_ERR,
            H1_SSL_DELTA_ERR,
        )
    )
    results.append(
        one_row(
            "diff_RMSE_SSL",
            "diff_RMSE_SL",
            "intra-site delta RMSE (test - val), SSL - SL",
            "less",
            H0_SSL_DELTA_ERR,
            H1_SSL_DELTA_ERR,
        )
    )
    return results


def ssl_better_transfer_label_pearson_one_sided(
    mean_delta_ssl_minus_sl: float, is_statistically_significant: bool
) -> str:
    if is_statistically_significant and pd.notna(mean_delta_ssl_minus_sl) and mean_delta_ssl_minus_sl > 0:
        return "SSL shows larger Pearson r intra-site deltas (one-sided, significant)"
    return "no significant evidence that SSL intra-site Pearson deltas exceed SL"


def ssl_better_maermse_delta_one_sided(
    mean_delta_ssl_minus_sl: float, is_statistically_significant: bool
) -> str:
    if is_statistically_significant and pd.notna(mean_delta_ssl_minus_sl) and mean_delta_ssl_minus_sl < 0:
        return "SSL shows smaller deterioration on test vs val (one-sided, significant)"
    return "no significant evidence that SSL intra-site MAE/RMSE deltas are smaller than SL"


def build_delta_sem_mad(merged_wide: pd.DataFrame) -> pd.DataFrame:
    metrics = ["diff_Pearsons_r", "diff_Pearsons_z", "diff_MAE", "diff_RMSE"]
    groupings = [
        ("overall", ["architecture", "SSL/SL"]),
        ("crop", ["architecture", "SSL/SL", "crop"]),
        ("management", ["architecture", "SSL/SL", "management_practise"]),
    ]
    frames: List[pd.DataFrame] = []
    for level, gcols in groupings:
        gcols_actual = [c for c in gcols if c in merged_wide.columns]
        agg = merged_wide.groupby(gcols_actual, dropna=False)[metrics].agg(["mean", "sem", "median", mad_custom])
        agg = agg.reset_index()
        agg.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col for col in agg.columns.to_flat_index()
        ]
        agg["aggregation_level"] = level
        frames.append(agg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def direction_ssl_vs_sl_on_delta_one_sided(metric: str, mean_mu: float, significant: bool) -> str:
    if not significant or pd.isna(mean_mu):
        return ""
    mstr = metric
    if "MAE" in mstr or "RMSE" in mstr:
        return ssl_better_maermse_delta_one_sided(mean_mu, True)
    return ssl_better_transfer_label_pearson_one_sided(mean_mu, True)


def direction_test_vs_val_one_sided(metric: str, mean_delta: float, significant: bool) -> str:
    if not significant or pd.isna(mean_delta):
        return ""
    if metric.startswith("Pearsons r"):
        return pearson_degradation_label_one_sided(mean_delta, True)
    if metric in ("MAE", "RMSE"):
        return error_increase_label_one_sided(mean_delta, True)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired one-sided intra-site transfer tests (test vs val degradation; "
            "SSL-vs-SL delta comparisons favoring SSL)."
        )
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Filtered field-level CSV path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--output-prefix",
        default="intra_site_transfer_significance_extreme-filtered_thr-1e+06_one-sided",
        help="CSV filename prefix.",
    )
    parser.add_argument(
        "--include-fisher",
        action="store_true",
        help="Analyze Pearson correlations in Fisher-z space (paired across val/test).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold alpha.")
    parser.add_argument(
        "--significance-rule",
        choices=["either", "both", "ttest", "wilcoxon"],
        default="either",
        help="How to combine paired t-tests and Wilcoxon results.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

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

    df_tv = df[df["set"].astype(str).str.lower().isin(("test", "val"))].copy()
    merged_wide = merge_test_val_wide(df_tv)
    if merged_wide.empty:
        raise RuntimeError("No overlapping test/validation rows after inner merge.")

    per_approach_rows: List[Dict[str, object]] = []
    for arch, grp in merged_wide.groupby("architecture", dropna=False):
        for ssl_sl, sub in grp.groupby("SSL/SL", dropna=False):
            per_approach_rows.extend(per_approach_paired_analysis(sub, str(arch), str(ssl_sl), args.include_fisher))

    per_df = pd.DataFrame(per_approach_rows)
    per_df["is_significant"] = per_df.apply(
        lambda r: is_significant(r, args.alpha, args.significance_rule), axis=1
    )
    per_df["direction_if_significant"] = per_df.apply(
        lambda row: direction_test_vs_val_one_sided(
            str(row["metric"]), row["mean_test_minus_val"], row["is_significant"]
        ),
        axis=1,
    )
    per_df = per_df.sort_values(by=["architecture", "SSL/SL", "metric"])

    delta_summary = build_delta_sem_mad(merged_wide)

    per_path = os.path.join(args.output_dir, f"{args.output_prefix}_test_minus_val_paired_tests.csv")
    delta_path = os.path.join(args.output_dir, f"{args.output_prefix}_intrinsic_delta_summary_sem_mad.csv")
    per_df.to_csv(per_path, index=False)
    delta_summary.sort_values(by=["aggregation_level", "architecture", "SSL/SL"]).to_csv(delta_path, index=False)

    print("Saved paired one-sided test-vs-val inference:")
    print(per_path)
    print("\nSaved descriptive delta summary (SEM + MAD):")
    print(delta_path)

    ssl_vs_rows: List[Dict[str, object]] = []
    for arch, _grp in merged_wide.groupby("architecture", dropna=False):
        ssl_vs_rows.extend(ssl_vs_sl_on_intrinsic_deltas(merged_wide, str(arch), args.include_fisher))
    comp_df = pd.DataFrame(ssl_vs_rows)
    if not comp_df.empty:
        comp_df["is_significant"] = comp_df.apply(
            lambda r: is_significant(r, args.alpha, args.significance_rule), axis=1
        )
        comp_df["direction_if_significant"] = [
            direction_ssl_vs_sl_on_delta_one_sided(str(m), float(mu) if pd.notna(mu) else np.nan, bool(sig))
            for m, mu, sig in zip(
                comp_df["metric"], comp_df["mean_delta_ssl_minus_sl"], comp_df["is_significant"]
            )
        ]
        comp_path = os.path.join(
            args.output_dir, f"{args.output_prefix}_ssl_vs_sl_on_intrinsic_delta_paired.csv"
        )
        comp_df.sort_values(by=["architecture", "metric"]).to_csv(comp_path, index=False)
        print("\nSaved paired one-sided SSL-vs-SL tests on stratified deltas (H1: SSL better transfer):")
        print(comp_path)

    if "n_removed_affected" in df.columns:
        print("\nInput filtering counts (summed across rows):", int(df["n_removed_affected"].sum()))

    print("\nPreview per-approach tests:")
    print(per_df.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
