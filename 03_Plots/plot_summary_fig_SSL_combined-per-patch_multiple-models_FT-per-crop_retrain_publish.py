# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-supervised learning for crop yield prediction across diversified cropping systems".
This script generates publication-quality plots comparing model performance across different architectures, crop types, and management practices.
It creates three types of visualizations:
1. Performance by crop type (Pearson's r correlation)
2. Performance by management practice
3. Combined performance comparison across train/val/test sets

The script processes performance metrics from CSV files, calculates statistics (mean, SEM, median, MAD),
and creates publication-ready plots with proper formatting, annotations, and error bars.

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

import argparse
# import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


DEFAULT_FILTERED_FIELD_TABLE = (
    r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures"
    r"\field_level_prediction_performance_SL_vs_SSL_pooled-folds_extreme-filtered_thr-1e+06.csv"
)


def mad_custom(x):
    """Calculate Median Absolute Deviation (MAD) for a given array.
    Returns 0.0 for groups with one or no valid data points."""
    if len(x) > 1:
        return np.median(np.abs(x - np.median(x)))
    else:
        return 0.0


def build_r_agg(show_median_mad):
    """Build groupby aggregation dict for Pearson's r summaries."""
    agg = {
        "mean": "mean",
        "sem": lambda x: x.sem() if len(x) > 1 else 0,
    }
    if show_median_mad:
        agg["median"] = "median"
        agg["mad"] = lambda x: mad_custom(x)
    return agg

def get_summary_glob(filtered_field_table_path):
    """Load already extreme-filtered field-level metrics and reshape for plotting."""
    if not os.path.exists(filtered_field_table_path):
        raise FileNotFoundError(f"Filtered field-level table not found: {filtered_field_table_path}")

    filtered_df = pd.read_csv(filtered_field_table_path)
    required_cols = {
        "field_ID",
        "crop",
        "management_practise",
        "set",
        "SSL/SL",
        "architecture",
        "Pearsons r",
    }
    missing_cols = required_cols.difference(filtered_df.columns)
    if missing_cols:
        raise ValueError(
            f"Filtered field-level table is missing required columns: {sorted(missing_cols)}"
        )

    ssl_sl_to_dataset = {"SSL": "SSL", "SL": "pointDS"}
    architecture_for_plot = (
        filtered_df["architecture"]
        .astype(str)
        .replace({"ConvNeXt tiny": "ConvNeXt", "ResNet18": "resnet18"})
    )
    result_df_glob = pd.DataFrame(
        {
            "r": pd.to_numeric(filtered_df["Pearsons r"], errors="coerce"),
            "crop_type": filtered_df["crop"].astype(str),
            "architecture": architecture_for_plot,
            "set": filtered_df["set"].astype(str),
            "dataset": filtered_df["SSL/SL"].map(ssl_sl_to_dataset).fillna("pointDS"),
            "fields": "14",
            "sort": "0",
            "patch_no": filtered_df["field_ID"].astype(str),
            "management": filtered_df["management_practise"].astype(str),
        }
    )
    return result_df_glob

def main():
    """Main function to generate performance comparison plots for different model architectures
    and management types across train/val/test sets."""
    parser = argparse.ArgumentParser(
        description="Generate SSL/SL summary figures from extreme-filtered field-level table."
    )
    parser.add_argument(
        "--filtered-field-table",
        default=DEFAULT_FILTERED_FIELD_TABLE,
        help="Path to field_level_prediction_performance_*_extreme-filtered*.csv.",
    )
    parser.add_argument(
        "--figure-path",
        default=r"E:\Projects\PatchCROP\Output\2024_SSL\Results_Pub_Retrain\Combined_Figures",
        help="Output directory for generated figures and summary CSVs.",
    )
    parser.add_argument(
        "--output-tag",
        default="extreme-filtered",
        help="Tag appended to all output filenames to mark filtered outputs.",
    )
    parser.add_argument(
        "--no-median-mad",
        action="store_true",
        help="Disable median and MAD in figures (mean ± SEM only).",
    )
    args = parser.parse_args()

    show_median_mad = not args.no_median_mad
    figure_path = args.figure_path
    output_tag_suffix = "_" + str(args.output_tag).strip().replace(" ", "-")
    os.makedirs(figure_path, exist_ok=True)

    # Set plot styling parameters
    # font_size = 11
    # title_size = 12
    # label_size = 11
    # text_size = 10    

    # Generate plots for each dataset split
    for s in ['train', 'val', 'test']:
        font_size = 15
        title_size = 15
        label_size = 15
        text_size = 15

        plt.rc('font', size=font_size)
        plt.rc('axes', titlesize=title_size)
        plt.rc('axes', labelsize=label_size)
        plt.rc('xtick', labelsize=label_size)
        plt.rc('ytick', labelsize=label_size)
        plt.rc('legend', fontsize=label_size)
        plt.rc('figure', titlesize=title_size)
        # Create figure for crop type aggregation
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13.5, 9), sharey=True)
        plt.subplots_adjust(wspace=0.05)

        for _ in [0]:
            result_df_glob = get_summary_glob(args.filtered_field_table)
            df_set = result_df_glob[result_df_glob['set'] == s]
            i = 0  # Initialize counter variable

            # Process data for 14 fields
            for n in ['14']:
                # Filter data for line plot
                df_lineplot = df_set[(df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt')) & 
                                     ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'SSL'))) &
                                    (df_set['fields'] == n)]

                # Aggregate performance metrics by crop type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(
                    ['architecture', 'dataset', 'crop_type']
                )['r'].agg(**build_r_agg(show_median_mad)).reset_index()

                # Save aggregated data
                df_local_agg_agg_mean_sem_mad.to_csv(
                    os.path.join(figure_path, "crop_performance_" + s + output_tag_suffix + ".csv")
                )

                # Separate SSL and supervised data
                ssl_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] == 'SSL']
                supervised_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] != 'SSL']

                # Calculate y-axis limits
                global_min = df_local_agg_agg_mean_sem_mad['mean'].min() - df_local_agg_agg_mean_sem_mad['sem'].max()
                global_max = df_local_agg_agg_mean_sem_mad['mean'].max() + df_local_agg_agg_mean_sem_mad['sem'].max()

                # Plot supervised methods
                ax = axes
                offset = {
                    ('resnet18', 'pointDS'): -0.1,
                    ('ConvNeXt', 'pointDS'): -0.05,
                    ('resnet18', 'SSL'): 0.0,
                    ('ConvNeXt', 'SSL'): 0.15
                }

                for architecture, dataset, color, label_suffix in [
                    ('ConvNeXt', 'pointDS', '#808080', 'ConvNeXt tiny'),
                    ('resnet18', 'pointDS', '#808080', 'ResNet18')]:
                    
                    group = supervised_data[(supervised_data['architecture'] == architecture) &
                                          (supervised_data['dataset'] == dataset)]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + offset[(architecture, dataset)] for x in range(len(group['crop_type']))]

                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.03 if (architecture == 'resnet18' and dataset == 'pointDS') else 0.02
                        x_offset = -0.16
                        ax.text(x + x_offset, mean + y_offset, f'{mean:.2f}±{sem:.2f}', 
                               ha='center', fontsize=text_size, color=color)

                # Configure plot appearance
                ax.set_ylabel('Pearson\'s r', fontsize=title_size)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=font_size)
                if i == 0: ax.legend(fontsize=label_size)

                # Plot SSL methods
                for architecture, color, label_suffix in [
                    ('ConvNeXt', '#029386', 'SSL - ConvNeXt tiny'),
                    ('resnet18', '#029386', 'SSL - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'ConvNeXt' else 0) 
                              for x in range(len(group['crop_type']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.02 if architecture == 'resnet18' else 0.02
                        x_offset = 0.16
                        ax.text(x + x_offset, mean + y_offset, f'{mean:.2f}±{sem:.2f}', 
                               ha='center', fontsize=text_size, color=color)

                # Final plot configuration
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=title_size)
                if i == 0: ax.legend(fontsize=label_size)

                # Set x-tick labels for crop types
                ax.set_xticks(range(len(group['crop_type'])))
                ax.set_xticklabels([label.capitalize() for label in group['crop_type']], 
                                 rotation=45, fontsize=label_size)

                plt.sca(ax)
                plt.tight_layout()

            # Save the crop type figure
            fig.savefig(
                os.path.join(figure_path, "r_vs_crop-type_" + "14" + "_mean-sem_" + s + output_tag_suffix + ".png")
            )
            fig.savefig(
                os.path.join(figure_path, "r_vs_crop-type_" + "14" + "_mean-sem_" + s + output_tag_suffix + ".tiff")
            )

        # Create figure for management type aggregation
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13.5, 9), sharey=True)
        plt.subplots_adjust(wspace=0.05)

        for _ in [0]:
            result_df_glob = get_summary_glob(args.filtered_field_table)
            df_set = result_df_glob[result_df_glob['set'] == s]
            i = 0  # Initialize counter variable

            # Process data for 14 fields
            for n in ['14']:
                # Filter data for line plot
                df_lineplot = df_set[(df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt')) & 
                                     ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'SSL'))) &
                                    (df_set['fields'] == n)]

                # Aggregate performance metrics by management type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(
                    ['architecture', 'dataset', 'management']
                )['r'].agg(**build_r_agg(show_median_mad)).reset_index()

                # Save aggregated data
                df_local_agg_agg_mean_sem_mad.to_csv(
                    os.path.join(figure_path, "management_performance_" + s + output_tag_suffix + ".csv")
                )

                # Separate SSL and supervised data
                ssl_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] == 'SSL']
                supervised_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] != 'SSL']

                # Calculate y-axis limits
                global_min = df_local_agg_agg_mean_sem_mad['mean'].min() - df_local_agg_agg_mean_sem_mad['sem'].max()
                global_max = df_local_agg_agg_mean_sem_mad['mean'].max() + df_local_agg_agg_mean_sem_mad['sem'].max()

                # Plot supervised methods
                ax = axes
                offset = {
                    ('resnet18', 'pointDS'): -0.1,
                    ('ConvNeXt', 'pointDS'): -0.05,
                    ('resnet18', 'SSL'): 0.0,
                    ('ConvNeXt', 'SSL'): 0.15
                }

                for architecture, dataset, color, label_suffix in [
                    ('ConvNeXt', 'pointDS', '#808080', 'ConvNeXt tiny'),
                    ('resnet18', 'pointDS', '#808080', 'ResNet18')]:
                    
                    group = supervised_data[(supervised_data['architecture'] == architecture) &
                                          (supervised_data['dataset'] == dataset)]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + offset[(architecture, dataset)] for x in range(len(group['management']))]

                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.03 if (architecture == 'resnet18' and dataset == 'pointDS') else 0.02
                        x_offset = -0.16
                        ax.text(x + x_offset, mean + y_offset, f'{mean:.2f}±{sem:.2f}', 
                               ha='center', fontsize=text_size, color=color)

                # Configure plot appearance
                ax.set_ylabel('Pearson\'s r', fontsize=title_size)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=font_size)
                if i == 0: ax.legend(fontsize=label_size)

                # Plot SSL methods
                for architecture, color, label_suffix in [
                    ('ConvNeXt', '#029386', 'SSL - ConvNeXt tiny'),
                    ('resnet18', '#029386', 'SSL - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'ConvNeXt' else 0) 
                              for x in range(len(group['management']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.02 if architecture == 'resnet18' else 0.02
                        x_offset = 0.16
                        ax.text(x + x_offset, mean + y_offset, f'{mean:.2f}±{sem:.2f}', 
                               ha='center', fontsize=text_size, color=color)

                # Final plot configuration
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=title_size)
                if i == 0: ax.legend(fontsize=label_size)

                # Set x-tick labels for management types
                ax.set_xticks(range(len(group['management'])))
                ax.set_xticklabels([label for label in group['management']], 
                                 rotation=45, fontsize=label_size)

                plt.sca(ax)
                plt.tight_layout()

            # Save the management type figure
            fig.savefig(
                os.path.join(figure_path, "r_vs_management_" + "14" + "_mean-sem_" + s + output_tag_suffix + ".png")
            )
            fig.savefig(
                os.path.join(figure_path, "r_vs_management_" + "14" + "_mean-sem_" + s + output_tag_suffix + ".tiff")
            )

        # Create combined figure for crop type and management type
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(13.5, 18))
        plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing between subplots

        for _ in [0]:
            result_df_glob = get_summary_glob(args.filtered_field_table)
            df_set = result_df_glob[result_df_glob['set'] == s]
            i = 0  # Initialize counter variable

            # Process data for 14 fields
            for n in ['14']:
                # Filter data for line plot
                df_lineplot = df_set[(df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt')) & 
                                     ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'SSL'))) &
                                    (df_set['fields'] == n)]

                # Top subplot: Crop type performance
                ax = axes[0]
                # Aggregate performance metrics by crop type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(
                    ['architecture', 'dataset', 'crop_type']
                )['r'].agg(**build_r_agg(show_median_mad)).reset_index()

                # Separate SSL and supervised data
                ssl_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] == 'SSL']
                supervised_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] != 'SSL']

                # Plot supervised methods
                offset = {
                    ('resnet18', 'pointDS'): -0.1,
                    ('ConvNeXt', 'pointDS'): -0.05,
                    ('resnet18', 'SSL'): 0.0,
                    ('ConvNeXt', 'SSL'): 0.15
                }

                for architecture, dataset, color, label_suffix in [
                    ('ConvNeXt', 'pointDS', '#808080', 'SL - ConvNeXt tiny'),
                    ('resnet18', 'pointDS', '#808080', 'SL - ResNet18')]:
                    
                    group = supervised_data[(supervised_data['architecture'] == architecture) &
                                          (supervised_data['dataset'] == dataset)]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + offset[(architecture, dataset)] for x in range(len(group['crop_type']))]

                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # No text annotations in combined figure to keep readability.

                # Plot SSL methods
                for architecture, color, label_suffix in [
                    ('ConvNeXt', '#029386', 'SSL - ConvNeXt tiny'),
                    ('resnet18', '#029386', 'SSL - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'ConvNeXt' else 0) 
                              for x in range(len(group['crop_type']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # No text annotations in combined figure to keep readability.

                # Configure top subplot appearance
                ax.set_ylabel('Pearson\'s r', fontsize=label_size+2)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=font_size+2)
                if i == 0: ax.legend(fontsize=label_size)

                # Set x-tick labels for crop types
                ax.set_xticks(range(len(group['crop_type'])))
                ax.set_xticklabels(
                    [
                        'Soybean' if str(lbl).capitalize() == 'Soy' else str(lbl).capitalize()
                        for lbl in group['crop_type']
                    ],
                    rotation=45,
                    fontsize=label_size + 2,
                )

                # Add 'a' label
                ax.text(-0.1, 1.05, 'a', transform=ax.transAxes, fontsize=20, fontweight='bold')

                # Bottom subplot: Management type performance
                ax = axes[1]
                # Aggregate performance metrics by management type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(
                    ['architecture', 'dataset', 'management']
                )['r'].agg(**build_r_agg(show_median_mad)).reset_index()

                # Separate SSL and supervised data
                ssl_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] == 'SSL']
                supervised_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] != 'SSL']

                # Plot supervised methods
                for architecture, dataset, color, label_suffix in [
                    ('ConvNeXt', 'pointDS', '#808080', 'SL - ConvNeXt tiny'),
                    ('resnet18', 'pointDS', '#808080', 'SL - ResNet18')]:
                    
                    group = supervised_data[(supervised_data['architecture'] == architecture) &
                                          (supervised_data['dataset'] == dataset)]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + offset[(architecture, dataset)] for x in range(len(group['management']))]

                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # No text annotations in combined figure to keep readability.

                # Plot SSL methods
                for architecture, color, label_suffix in [
                    ('ConvNeXt', '#029386', 'SSL - ConvNeXt tiny'),
                    ('resnet18', '#029386', 'SSL - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'ConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'ConvNeXt' else 0) 
                              for x in range(len(group['management']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # No text annotations in combined figure to keep readability.

                # Configure bottom subplot appearance
                ax.set_ylabel('Pearson\'s r', fontsize=label_size+2)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=font_size+2)

                # Set x-tick labels for management types
                ax.set_xticks(range(len(group['management'])))
                ax.set_xticklabels([label.replace('Reduced Pesticides and Flower Strips', 'Reduced Pesticides\nand Flower Strips') for label in group['management']], 
                                 rotation=45, fontsize=label_size+2)

                # Add 'b' label
                ax.text(-0.1, 1.05, 'b', transform=ax.transAxes, fontsize=20, fontweight='bold')

                plt.tight_layout()

            # Save the combined figure
            fig.savefig(
                os.path.join(
                    figure_path,
                    "r_vs_crop-and-management_" + "14" + "_mean-sem_" + s + output_tag_suffix + ".png",
                )
            )
            fig.savefig(
                os.path.join(
                    figure_path,
                    "r_vs_crop-and-management_" + "14" + "_mean-sem_" + s + output_tag_suffix + ".tiff",
                )
            )

        #########################################################################################
        #########################################################################################
            # font_size = 11
            # title_size = 12
            # label_size = 11
            # text_size = 10    
            font_size = 12
            title_size = 12
            label_size = 12
            text_size = 12

            plt.rc('font', size=font_size)
            plt.rc('axes', titlesize=title_size)
            plt.rc('axes', labelsize=label_size)
            plt.rc('xtick', labelsize=label_size)
            plt.rc('ytick', labelsize=label_size)
            plt.rc('legend', fontsize=label_size)
            plt.rc('figure', titlesize=title_size)
        # Create figure for modeling approach comparison
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 9), sharey=True)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.5, 9), sharey=True)
        plt.subplots_adjust(wspace=0.05)  # Adjust the spacing between subplots to be tight

        for s in ['test', 'val', 'train']:
            for _ in [0]:
                result_df_glob = get_summary_glob(args.filtered_field_table)
                df_set = result_df_glob[result_df_glob['set'] == s]
                i_plots = 1

                if s == 'test':
                    ax = axes[2]
                elif s == 'val':
                    ax = axes[1]
                else:
                    ax = axes[0]

                for n in ['14']:
                    # Data for local aggregation (excluding 'all-no-distinction', median and std of 'r')
                    df_local_agg = df_set[(df_set['crop_type'] != 'all-no-distinction') &
                                        ((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt')) &
                                        ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'SSL')) &
                                        (df_set['fields'] == n)]

                    # Aggregate performance metrics
                    df_local_agg_agg_mean_sem_mad = df_local_agg.groupby(
                        ['architecture', 'dataset']
                    )['r'].agg(**build_r_agg(show_median_mad)).reset_index()

                    # Save aggregated data
                    df_local_agg_agg_mean_sem_mad.to_csv(
                        os.path.join(figure_path, "modeling_approach_performance_" + s + output_tag_suffix + ".csv")
                    )

                    i = 1
                    order = [('resnet18', 'pointDS'), ('ConvNeXt', 'pointDS'), ('resnet18', 'SSL'), ('ConvNeXt', 'SSL')]
                    for (architecture, dataset) in order:
                        if dataset == 'pointDS':
                            color = '#808080'
                        elif dataset == 'SSL':
                            color = '#029386'
                        else:
                            color = '#870981'

                        # Jitter the points
                        r = df_local_agg[(df_local_agg['architecture'] == architecture) & (df_local_agg['dataset'] == dataset)].r
                        jit_x = [i] * len(r) + 0.1 * np.random.rand(len(r)) - 0.05

                        ax.scatter(jit_x, r, color=color, edgecolors="black", alpha=0.5, s=100)

                        group_stats = df_local_agg_agg_mean_sem_mad[
                            (df_local_agg_agg_mean_sem_mad['architecture'] == architecture)
                            & (df_local_agg_agg_mean_sem_mad['dataset'] == dataset)
                        ]
                        y_sem = group_stats['sem']
                        y_mean = group_stats['mean']
                        mean_x = [i - 0.05 if show_median_mad else i] * len(y_mean)

                        if i == 1:
                            ax.errorbar(
                                mean_x, y_mean, yerr=y_sem, fmt='o', color='black', capsize=0,
                                label='Mean ± SEM', zorder=2, markersize=14, linewidth=2,
                            )
                        else:
                            ax.errorbar(
                                mean_x, y_mean, yerr=y_sem, fmt='o', color='black', capsize=0,
                                label='', zorder=2, markersize=14, linewidth=2,
                            )

                        if show_median_mad:
                            y_median = group_stats['median']
                            y_mad = group_stats['mad']
                            if i == 1:
                                ax.errorbar(
                                    [i + 0.05] * len(y_mean), y_median, yerr=y_mad, fmt='o',
                                    color='#F08030', capsize=0, label='Median ± MAD', zorder=2,
                                    markersize=14, linewidth=2,
                                )
                            else:
                                ax.errorbar(
                                    [i + 0.05] * len(y_mean), y_median, yerr=y_mad, fmt='o',
                                    color='#F08030', capsize=0, label='', zorder=2,
                                    markersize=14, linewidth=2,
                                )

                        # Annotate mean and SEM values
                        for x, mean, sem in zip([i] * len(y_mean), y_mean, y_sem):
                            x = x + 0.1
                            if i <= 5:
                                ax.text(x + 0.4, mean + 0.01, f'{mean:.2f}±{sem:.2f}', ha='center', fontsize=text_size, color='black')
                            else:
                                if i_plots == 0:
                                    ax.text(x - 0.4, mean - 0.09, f'{mean:.2f}±{sem:.2f}', ha='center', fontsize=text_size, color='black')
                                elif i_plots == 2:
                                    ax.text(x - 0.4, mean + 0.06, f'{mean:.2f}±{sem:.2f}', ha='center', fontsize=text_size, color='black')
                                else:
                                    ax.text(x - 0.4, mean + 0.01, f'{mean:.2f}±{sem:.2f}', ha='center', fontsize=text_size, color='black')

                        if show_median_mad:
                            y_median = group_stats['median']
                            y_mad = group_stats['mad']
                            # Annotate median and MAD values
                            for x, median, mean, mad in zip([i] * len(y_median), y_median, y_mean, y_mad):
                                x = x + 0.1
                                if i <= 5:
                                    ax.text(x + 0.4, mean + 0.05, f'{median:.2f}±{mad:.2f}', ha='center', fontsize=text_size, color='#F08030')
                                else:
                                    if i_plots == 0:
                                        ax.text(x - 0.4, mean - 0.04, f'{median:.2f}±{mad:.2f}', ha='center', fontsize=text_size, color='#F08030')
                                    elif i_plots == 2:
                                        ax.text(x - 0.4, mean + 0.11, f'{median:.2f}±{mad:.2f}', ha='center', fontsize=text_size, color='#F08030')
                                    else:
                                        ax.text(x - 0.4, mean + 0.05, f'{median:.2f}±{mad:.2f}', ha='center', fontsize=text_size, color='#F08030')
                        i += 1

                    ax.axhline(0, color='grey', lw=1)  # Add a line at r=0 for reference
                    ax.grid(True, which='both', linestyle='--', linewidth=1)
                    if s == 'test': ax.legend(fontsize=text_size)
                    if s == 'train': ax.set_ylabel('Pearson\'s r', fontsize=label_size+2)

                    # Set x-tick labels
                    labels = ['SL \nResNet18', 'SL \nConvNeXt tiny', 'SSL \nResNet18', 'SSL \nConvNeXt tiny']
                    plt.sca(ax)
                    plt.xticks(ticks=[1, 2, 3, 4], labels=labels, rotation=45, fontsize=label_size+2)

                    ax.set_title(s.capitalize() + ' Set', fontsize=label_size+2)

                    plt.tight_layout()

        # Save the modeling approach figure
        fig.savefig(
            os.path.join(
                figure_path,
                "aggregated_r_across_rop-types_" + "14" + "_train-val-test" + output_tag_suffix + ".png",
            )
        )
        fig.savefig(
            os.path.join(
                figure_path,
                "aggregated_r_across_rop-types_" + "14" + "_train-val-test" + output_tag_suffix + ".tiff",
            )
        )

if __name__ == '__main__':
    main()

