# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script generates publication-quality plots of SSL training and fine-tuning losses for different architectures and crop types.

The script:
1. Processes training statistics from CSV files for each fold
2. Handles both SSL pre-training (VICReg Loss) and fine-tuning (RMSE) phases
3. Creates subplots for each crop type and fold
4. Adds best epoch markers and proper formatting
5. Saves plots in both PNG and TIFF formats

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports
import os
from typing import List, Tuple, Dict, Any

# Libs
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define paths
output_root = '/media/stillsen/Megalodon/PatchCROP/Output/2024_SSL/Results_Pub_Retrain/'
fig_root = '/media/stillsen/Megalodon/PatchCROP/Output/2024_SSL/Results_Pub_Retrain/Combined_Figures/'
os.makedirs(fig_root, exist_ok=True)

# Define architectures and their corresponding directories
archs = ['VICRegConvNext', 'VICReg']
all_output_dirs = [
    ['Combined_SCV_large_July_VICRegConvNext_SCV_E500lightly-VICReg_kernel_size_224_DS-combined_SCV_large_July_ConvNext-back_14fields_tuning/',
     'Combined_SCV_large_lupine_VICRegConvNext_SCV_E2000_kernel_size_224_DS-combined_SCV_large_lupine_14fields_lupine/',
     'Combined_SCV_large_maize_VICRegConvNext_SCV_E2000_kernel_size_224_DS-combined_SCV_large_maize_14fields_maize/',
     'Combined_SCV_large_soy_VICRegConvNext_SCV_E2000_kernel_size_224_DS-combined_SCV_large_soy_14fields_soy/',
     'Combined_SCV_large_sunflower_July_VICRegConvNext_SCV_E2000_kernel_size_224_DS-combined_SCV_large_sunflower_July_14fields_sunflower/'],
    
    ['Combined_SCV_large_July_VICReg_SCV_E500lightly-VICReg_kernel_size_224_DS-combined_SCV_large_July_resnet-back_14fields_tuning/',
     'Combined_SCV_large_lupine_VICReg_SCV_E2000_kernel_size_224_DS-combined_SCV_large_lupine_14fields_lupine/',
     'Combined_SCV_large_maize_VICReg_SCV_E2000_kernel_size_224_DS-combined_SCV_large_maize_14fields_maize/',
     'Combined_SCV_large_soy_VICReg_SCV_E2000_kernel_size_224_DS-combined_SCV_large_soy_14fields_soy/',
     'Combined_SCV_large_sunflower_July_VICReg_SCV_E2000_kernel_size_224_DS-combined_SCV_large_sunflower_July_14fields_sunflower/']
]

row_labels = ['SSL', 'SSL-FT-lupine', 'SSL-FT-maize', 'SSL-FT-soy', 'SSL-FT-sunflower']
num_folds = 4
title_size = 15
label_size = 13
text_size = 13

def process_and_plot_data(df: pd.DataFrame, ax: plt.Axes, is_ssl: bool = False, show_legend: bool = False) -> None:
    """
    Process training data and create a plot for a single subplot.
    
    Args:
        df: DataFrame containing training statistics
        ax: Matplotlib axis to plot on
        is_ssl: Whether this is SSL pre-training phase (default: False)
        show_legend: Whether to show the legend (default: False)
    
    The function:
    1. Converts MSE to RMSE for fine-tuning phase
    2. Sorts data by epochs
    3. Plots training and validation curves
    4. Adds best epoch marker
    5. Sets axis labels and limits based on phase (SSL or fine-tuning)
    """
    if not is_ssl:  # For domain tuning, convert to RMSE
        df['val_loss'] = df['val_loss'] ** (1/2)
        df['train_loss'] = df['train_loss'] ** (1/2)
    
    df_sorted = df.sort_values(by='epochs')
    ax.plot(df_sorted['epochs'], df_sorted['train_loss'], label='Train')
    ax.plot(df_sorted['epochs'], df_sorted['val_loss'], label='Val')
    
    # Add vertical line for best epoch
    ax.axvline(x=df['best_epoch'][0], color='red', linestyle='dashed')
    ax.text(x=df['best_epoch'][0]-3, y=ax.get_ylim()[1]*1.01, 
            s=str(df['best_epoch'][0]), color='red')
    
    ax.set_xlabel('epochs', fontsize=label_size)
    if is_ssl:
        ax.set_ylabel('VICReg Loss', fontsize=label_size)
        ax.set_xlim([0, 500])  # SSL training was for 500 epochs
    else:
        ax.set_ylabel('RMSE', fontsize=label_size)
        ax.set_xlim([0, 2000])  # Fine-tuning was for 2000 epochs
    
    if show_legend:
        ax.legend(fontsize=label_size)

# Create figures for each architecture
for arch_idx, arch in enumerate(archs):
    # Use GridSpec for better control over subplot layout
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
    axes = gs.subplots()
    
    # fig.suptitle(f'{arch} Training Curves', fontsize=16)
    
    output_dirs = all_output_dirs[arch_idx]
    
    for row, (output_dir, row_label) in enumerate(zip(output_dirs, row_labels)):
        full_output_dir = os.path.join(output_root, output_dir)
        
        for k in range(num_folds):
            # Determine which suffix to use based on row
            suffix = '_self-supervised' if row == 0 else '_domain-tuning'
            
            # Read and plot data
            csv_path = os.path.join(full_output_dir, f'training_statistics_f{k}{suffix}.csv')
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                df.rename(columns={'Unnamed: 0': 'epochs'}, inplace=True)
                
                # Show legend only in the first subplot (row=0, k=0)
                process_and_plot_data(df, axes[row, k], is_ssl=(row == 0), show_legend=(row==0 and k==0))
                
                # Set titles
                if row == 0:
                    axes[row, k].set_title(f'Fold {k}', fontsize=title_size)
                if k == 0:
                    axes[row, k].set_ylabel(f'{row_label}\n{"VICReg Loss" if row == 0 else "RMSE"}', 
                                          fontsize=label_size, labelpad=10)
    
    # Use tight_layout with specific padding
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95], h_pad=2, w_pad=2)
    fig.savefig(os.path.join(fig_root, f'{arch}_combined_training_curves.png'), 
                dpi=400, bbox_inches='tight')
    fig.savefig(os.path.join(fig_root, f'{arch}_combined_training_curves.tiff'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
