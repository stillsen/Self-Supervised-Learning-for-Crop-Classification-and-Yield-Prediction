# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
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

import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

def mad_custom(x: np.ndarray) -> float:
    """
    Calculate Median Absolute Deviation (MAD) for a given array.
    Returns 0.0 for groups with one or no valid data points.
    
    Args:
        x: Input array of values
        
    Returns:
        float: MAD value or 0.0 for small groups
    """
    if len(x) > 1:
        return np.median(np.abs(x - np.median(x)))
    else:
        return 0.0

def get_summary_glob(root_dir: str, summary_path: str) -> pd.DataFrame:
    """
    Load or create a summary DataFrame containing model performance metrics.
    The DataFrame includes metrics for different architectures, datasets, and management types.
    
    Args:
        root_dir: Root directory containing model output files
        summary_path: Path to save/load the summary CSV file
        
    Returns:
        pd.DataFrame: Summary DataFrame with performance metrics
    """
    # Check if summary file already exists
    summary_file_path = os.path.join(root_dir, summary_path)
    if os.path.exists(summary_file_path):
        print('Loading existing summary file: {}'.format(summary_file_path))
        return pd.read_csv(summary_file_path, dtype={
            'r': 'float64',
            'crop_type': 'object',
            'architecture': 'object',
            'set': 'object',
            'dataset': 'object',
            'fields': 'object',
            'sort': 'object',
            'patch_no': 'object',
            'management': 'object'})

    # Initialize DataFrame with required columns
    result_df_glob = pd.DataFrame(columns=['r', 'crop_type', 'architecture', 'set', 'dataset', 'fields', 'sort', 'patch_no', 'management'])
    
    # Define patch IDs for different crop types
    maize_patches = ['50', '68', '20', '110', '74', '90']
    sunflower_patches = ['76', '95', '115']
    soy_patches = ['65', '58', '19']
    lupine_patches = ['89', '59', '119']

    # Define management types for different patches
    conventional_management = ['74', '90', '65', '89', '95_July']
    reduced_pesticides_management = ['50', '68', '58', '59', '76_July']
    reduced_pesti_flower_strps_management = ['20', '110', '19', '119', '115_July']

    # Process each subdirectory in the root directory
    for subdir in os.listdir(root_dir):
        if 'Combined_Figures' in subdir or 'E500' in subdir:
            continue

        # Determine dataset and architecture from subdirectory name
        dataset = 'pointDS' if 'pointDS' in subdir else 'swDS' if 'swDS' in subdir else 'SSL'
        architecture = 'resnet18' if 'resnet18' in subdir else 'ConvNeXt' if 'ConvNeXt' in subdir and not ('VICRegConvNext' in subdir) else 'VICReg' if 'VICReg' in subdir and not ('VICRegConvNext' in subdir) else 'VICRegConvNeXt'
        fields = '4' if '4' in subdir and not ('14' in subdir) else '14' if '14' in subdir else ''

        subdir_path = os.path.join(root_dir, subdir)
        print(subdir_path)
        
        if os.path.isdir(subdir_path):
            # Process each CSV file in the subdirectory
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                if csv_file.startswith('performance_metrics_f3_'):
                    # Determine crop type and management based on patch number
                    crop_type = ''
                    patch_no = ''
                    management = ''
                    
                    if subdir.startswith('P_'):
                        patch_no = subdir.split('_')[1]
                        crop_type = 'lupine' if patch_no in lupine_patches else 'maize' if patch_no in maize_patches else 'soy' if patch_no in soy_patches else 'sunflower' if patch_no in sunflower_patches else ''
                        management = 'Conventional' if patch_no in conventional_management else 'Reduced Pesticides' if patch_no in reduced_pesticides_management else 'Reduced Pesticides and Flower Strips' if patch_no in reduced_pesti_flower_strps_management else ''
                    else:
                        idx_from_back = -2 if 'July' in csv_file else -1
                        patch_no = csv_file.split('__')[0].split('_')[idx_from_back]
                        if patch_no != 'f3':
                            crop_type = 'lupine' if patch_no in lupine_patches else 'maize' if patch_no in maize_patches else 'soy' if patch_no in soy_patches else 'sunflower' if patch_no in sunflower_patches else ''
                            management = 'Conventional' if patch_no in conventional_management else 'Reduced Pesticides' if patch_no in reduced_pesticides_management else 'Reduced Pesticides and Flower Strips' if patch_no in reduced_pesti_flower_strps_management else ''
                        else:
                            crop_type = 'all-no-distinction'
                            patch_no = 'global'

                    # Determine dataset split (train/val/test)
                    subset = 'train' if '_train_' in csv_file else 'val' if '_val_' in csv_file else 'test' if '_test_' in csv_file else ''
                    
                    if subset:
                        csv_path = os.path.join(subdir_path, csv_file)
                        csv_df = pd.read_csv(csv_path)
                        result_df_glob.loc[subdir +'---'+ csv_file] = [
                            csv_df['global_r'].values[0], crop_type, architecture, subset, 
                            dataset, fields, '0', patch_no, management
                        ]

    # Save the processed DataFrame
    result_df_glob.to_csv(summary_file_path, encoding='utf-8')
    return result_df_glob

def main() -> None:
    """
    Main function to generate performance comparison plots for different model architectures
    and management types across train/val/test sets.
    
    Creates three types of visualizations:
    1. Performance by crop type
    2. Performance by management practice
    3. Combined performance comparison across train/val/test sets
    
    Each plot includes:
    - Mean and SEM error bars
    - Median and MAD error bars
    - Individual data points
    - Proper formatting for publication
    """
    # Define paths and parameters
    root_dirs = ['/media/stillsen/Megalodon/Projects/PatchCROP/Output/2024_SSL/Results_Pub_Retrain/']
    figure_path = '/media/stillsen/Megalodon/Projects/PatchCROP/Output/2024_SSL/Results_Pub_Retrain/Combined_Figures'
    summary_paths = ['summary_glob_retrain.csv']

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

        for root_dir, summary_path in zip(root_dirs, summary_paths):
            result_df_glob = get_summary_glob(root_dir, summary_path)
            df_set = result_df_glob[result_df_glob['set'] == s]
            i = 0  # Initialize counter variable

            # Process data for 14 fields
            for n in ['14']:
                # Filter data for line plot
                df_lineplot = df_set[(df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt')) & 
                                     ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'swDS'))) &
                                    (df_set['fields'] == n) |
                                    (df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'VICReg') | (df_set['architecture'] == 'VICRegConvNeXt')) & 
                                     (df_set['dataset'] == 'SSL')) &
                                    (df_set['fields'] == n)]

                # Aggregate performance metrics by crop type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(['architecture', 'dataset', 'crop_type'])['r'].agg(
                    mean='mean',
                    sem=lambda x: x.sem() if len(x) > 1 else 0,
                    median='median',
                    mad=lambda x: mad_custom(x)
                ).reset_index()

                # Save aggregated data
                df_local_agg_agg_mean_sem_mad.to_csv(os.path.join(figure_path, 'crop_performance_'+ s + '.csv'))

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
                    ('VICReg', 'SSL'): 0.0,
                    ('VICRegConvNeXt', 'SSL'): 0.15
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
                    ('VICRegConvNeXt', '#029386', 'VICReg - ConvNeXt tiny'),
                    ('VICReg', '#029386', 'VICReg - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'VICRegConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'VICRegConvNeXt' else 0) 
                              for x in range(len(group['crop_type']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.02 if (architecture == 'resnet18' or architecture == 'VICReg') else 0.02
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
            fig.savefig(os.path.join(figure_path, 'r_vs_crop-type_' + '14' + '_mean-sem_' + s + '.png'))
            fig.savefig(os.path.join(figure_path, 'r_vs_crop-type_' + '14' + '_mean-sem_' + s + '.tiff'))

        # Create figure for management type aggregation
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13.5, 9), sharey=True)
        plt.subplots_adjust(wspace=0.05)

        for root_dir, summary_path in zip(root_dirs, summary_paths):
            result_df_glob = get_summary_glob(root_dir, summary_path)
            df_set = result_df_glob[result_df_glob['set'] == s]
            i = 0  # Initialize counter variable

            # Process data for 14 fields
            for n in ['14']:
                # Filter data for line plot
                df_lineplot = df_set[(df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt')) & 
                                     ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'swDS'))) &
                                    (df_set['fields'] == n) |
                                    (df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'VICReg') | (df_set['architecture'] == 'VICRegConvNeXt')) & 
                                     (df_set['dataset'] == 'SSL')) &
                                    (df_set['fields'] == n)]

                # Aggregate performance metrics by management type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(['architecture', 'dataset', 'management'])['r'].agg(
                    mean='mean',
                    sem=lambda x: x.sem() if len(x) > 1 else 0,
                    median='median',
                    mad=lambda x: mad_custom(x)
                ).reset_index()

                # Save aggregated data
                df_local_agg_agg_mean_sem_mad.to_csv(os.path.join(figure_path, 'management_performance_'+ s + '.csv'))

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
                    ('VICReg', 'SSL'): 0.0,
                    ('VICRegConvNeXt', 'SSL'): 0.15
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
                    ('VICRegConvNeXt', '#029386', 'VICReg - ConvNeXt tiny'),
                    ('VICReg', '#029386', 'VICReg - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'VICRegConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'VICRegConvNeXt' else 0) 
                              for x in range(len(group['management']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.02 if (architecture == 'resnet18' or architecture == 'VICReg') else 0.02
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
            fig.savefig(os.path.join(figure_path, 'r_vs_management_' + '14' + '_mean-sem_' + s + '.png'))
            fig.savefig(os.path.join(figure_path, 'r_vs_management_' + '14' + '_mean-sem_' + s + '.tiff'))

        # Create combined figure for crop type and management type
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(13.5, 18))
        plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing between subplots

        for root_dir, summary_path in zip(root_dirs, summary_paths):
            result_df_glob = get_summary_glob(root_dir, summary_path)
            df_set = result_df_glob[result_df_glob['set'] == s]
            i = 0  # Initialize counter variable

            # Process data for 14 fields
            for n in ['14']:
                # Filter data for line plot
                df_lineplot = df_set[(df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt')) & 
                                     ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'swDS'))) &
                                    (df_set['fields'] == n) |
                                    (df_set['crop_type'] != 'all-no-distinction') &
                                    (((df_set['architecture'] == 'VICReg') | (df_set['architecture'] == 'VICRegConvNeXt')) & 
                                     (df_set['dataset'] == 'SSL')) &
                                    (df_set['fields'] == n)]

                # Top subplot: Crop type performance
                ax = axes[0]
                # Aggregate performance metrics by crop type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(['architecture', 'dataset', 'crop_type'])['r'].agg(
                    mean='mean',
                    sem=lambda x: x.sem() if len(x) > 1 else 0,
                    median='median',
                    mad=lambda x: mad_custom(x)
                ).reset_index()

                # Separate SSL and supervised data
                ssl_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] == 'SSL']
                supervised_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] != 'SSL']

                # Plot supervised methods
                offset = {
                    ('resnet18', 'pointDS'): -0.1,
                    ('ConvNeXt', 'pointDS'): -0.05,
                    ('VICReg', 'SSL'): 0.0,
                    ('VICRegConvNeXt', 'SSL'): 0.15
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
                               ha='center', fontsize=text_size+1, color=color)

                # Plot SSL methods
                for architecture, color, label_suffix in [
                    ('VICRegConvNeXt', '#029386', 'VICReg - ConvNeXt tiny'),
                    ('VICReg', '#029386', 'VICReg - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'VICRegConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'VICRegConvNeXt' else 0) 
                              for x in range(len(group['crop_type']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.02 if (architecture == 'resnet18' or architecture == 'VICReg') else 0.02
                        x_offset = 0.16
                        ax.text(x + x_offset, mean + y_offset, f'{mean:.2f}±{sem:.2f}', 
                               ha='center', fontsize=text_size+1, color=color)

                # Configure top subplot appearance
                ax.set_ylabel('Pearson\'s r', fontsize=label_size+2)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=font_size+2)
                if i == 0: ax.legend(fontsize=label_size)

                # Set x-tick labels for crop types
                ax.set_xticks(range(len(group['crop_type'])))
                ax.set_xticklabels([label.capitalize() for label in group['crop_type']], 
                                 rotation=45, fontsize=label_size+2)

                # Add 'a' label
                ax.text(-0.1, 1.05, 'a', transform=ax.transAxes, fontsize=20, fontweight='bold')

                # Bottom subplot: Management type performance
                ax = axes[1]
                # Aggregate performance metrics by management type
                df_local_agg_agg_mean_sem_mad = df_lineplot.groupby(['architecture', 'dataset', 'management'])['r'].agg(
                    mean='mean',
                    sem=lambda x: x.sem() if len(x) > 1 else 0,
                    median='median',
                    mad=lambda x: mad_custom(x)
                ).reset_index()

                # Separate SSL and supervised data
                ssl_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] == 'SSL']
                supervised_data = df_local_agg_agg_mean_sem_mad[df_local_agg_agg_mean_sem_mad['dataset'] != 'SSL']

                # Plot supervised methods
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

                    # Add mean and SEM annotations with reduced x-offset
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.013 if (architecture == 'resnet18' and dataset == 'pointDS') else 0.02
                        x_offset = -0.12  # Reduced from -0.16
                        ax.text(x + x_offset, mean + y_offset, f'{mean:.2f}±{sem:.2f}', 
                               ha='center', fontsize=text_size+1, color=color)

                # Plot SSL methods
                for architecture, color, label_suffix in [
                    ('VICRegConvNeXt', '#029386', 'VICReg - ConvNeXt tiny'),
                    ('VICReg', '#029386', 'VICReg - ResNet18')]:
                    
                    group = ssl_data[ssl_data['architecture'] == architecture]
                    line_style = 'dashed' if architecture == 'VICRegConvNeXt' else 'dotted'
                    x_dodged = [x + 0.05 * (1 if architecture == 'VICRegConvNeXt' else 0) 
                              for x in range(len(group['management']))]
                    
                    ax.plot(x_dodged, group['mean'], label=f'{label_suffix}', 
                           linestyle=line_style, linewidth=3.0, marker='o', color=color)
                    ax.errorbar(x_dodged, group['mean'], yerr=group['sem'], 
                              fmt='o', color=color, capsize=5)

                    # Add mean and SEM annotations with reduced x-offset
                    for x, mean, sem in zip(x_dodged, group['mean'], group['sem']):
                        y_offset = -0.012 if (architecture == 'resnet18' or architecture == 'VICReg') else 0.012
                        x_offset = 0.12  # Reduced from 0.16
                        ax.text(x + x_offset, mean + y_offset, f'{mean:.2f}±{sem:.2f}', 
                               ha='center', fontsize=text_size+1, color=color)

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
            fig.savefig(os.path.join(figure_path, 'r_vs_crop-and-management_' + '14' + '_mean-sem_' + s + '.png'))
            fig.savefig(os.path.join(figure_path, 'r_vs_crop-and-management_' + '14' + '_mean-sem_' + s + '.tiff'))

        #########################################################################################
        #########################################################################################
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
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.5, 9), sharey=True)
        plt.subplots_adjust(wspace=0.05)  # Adjust the spacing between subplots to be tight

        for s in ['test', 'val', 'train']:
            for root_dir, summary_path in zip(root_dirs, summary_paths):
                result_df_glob = get_summary_glob(root_dir, summary_path)
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
                                        ((df_set['architecture'] == 'resnet18') | (df_set['architecture'] == 'ConvNeXt') | (df_set['architecture'] == 'VICReg') | (df_set['architecture'] == 'VICRegConvNeXt')) &
                                        ((df_set['dataset'] == 'pointDS') | (df_set['dataset'] == 'swDS') | (df_set['dataset'] == 'SSL')) &
                                        (df_set['fields'] == n)]

                    # Aggregate performance metrics
                    df_local_agg_agg_mean_sem_mad = df_local_agg.groupby(['architecture', 'dataset'])['r'].agg(
                        mean='mean',
                        sem=lambda x: x.sem() if len(x) > 1 else 0,
                        median='median',
                        mad=lambda x: mad_custom(x)
                    ).reset_index()

                    # Save aggregated data
                    df_local_agg_agg_mean_sem_mad.to_csv(os.path.join(figure_path, 'modeling_approach_performance_'+ s + '.csv'))

                    i = 1
                    order = [('resnet18', 'pointDS'), ('ConvNeXt', 'pointDS'), ('VICReg', 'SSL'), ('VICRegConvNeXt', 'SSL')]
                    for (architecture, dataset) in order:
                        if dataset == 'pointDS':
                            color = '#808080'
                        elif dataset == 'swDS':
                            color = '#870981'
                        else:
                            color = '#029386'

                        # Jitter the points
                        r = df_local_agg[(df_local_agg['architecture'] == architecture) & (df_local_agg['dataset'] == dataset)].r
                        jit_x = [i] * len(r) + 0.1 * np.random.rand(len(r)) - 0.05

                        ax.scatter(jit_x, r, color=color, edgecolors="black", alpha=0.5, s=100)

                        y_sem = df_local_agg_agg_mean_sem_mad[(df_local_agg_agg_mean_sem_mad['architecture'] == architecture) & (df_local_agg_agg_mean_sem_mad['dataset'] == dataset)]['sem']
                        y_mean = df_local_agg_agg_mean_sem_mad[(df_local_agg_agg_mean_sem_mad['architecture'] == architecture) & (df_local_agg_agg_mean_sem_mad['dataset'] == dataset)]['mean']
                        y_median = df_local_agg_agg_mean_sem_mad[(df_local_agg_agg_mean_sem_mad['architecture'] == architecture) & (df_local_agg_agg_mean_sem_mad['dataset'] == dataset)]['median']
                        y_mad = df_local_agg_agg_mean_sem_mad[(df_local_agg_agg_mean_sem_mad['architecture'] == architecture) & (df_local_agg_agg_mean_sem_mad['dataset'] == dataset)]['mad']

                        if i == 1:
                            ax.errorbar([i + 0.05] * len(y_mean), y_median, yerr=y_mad, fmt='o', color='#F08030', capsize=0, label='Median ± MAD', zorder=2, markersize=14, linewidth=2)
                            ax.errorbar([i - 0.05] * len(y_mean), y_mean, yerr=y_sem, fmt='o', color='black', capsize=0, label='Mean ± SEM', zorder=2, markersize=14, linewidth=2)
                        else:
                            ax.errorbar([i + 0.05] * len(y_mean), y_median, yerr=y_mad, fmt='o', color='#F08030', capsize=0, label='', zorder=2, markersize=14, linewidth=2)
                            ax.errorbar([i - 0.05] * len(y_mean), y_mean, yerr=y_sem, fmt='o', color='black', capsize=0, label='', zorder=2, markersize=14, linewidth=2)

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
                    labels = ['ResNet18', 'ConvNeXt tiny', 'VICReg \nResNet18', 'VICReg \nConvNeXt tiny']
                    plt.sca(ax)
                    plt.xticks(ticks=[1, 2, 3, 4], labels=labels, rotation=45, fontsize=label_size+2)

                    ax.set_title(s.capitalize() + ' Set', fontsize=label_size+2)

                    plt.tight_layout()

        # Save the modeling approach figure
        fig.savefig(os.path.join(figure_path, 'aggregated_r_across_rop-types_' + '14' + '_train-val-test_' + '.png'))
        fig.savefig(os.path.join(figure_path, 'aggregated_r_across_rop-types_' + '14' + '_train-val-test_' + '.tiff'))

if __name__ == '__main__':
    main()
