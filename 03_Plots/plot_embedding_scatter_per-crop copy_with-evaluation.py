# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script analyzes and visualizes the learned embeddings from the SSL model, evaluating their quality and interpretability.

The script:
1. Processes embeddings from different splits (train/val/test)
2. Performs UMAP dimensionality reduction with parameter optimization
3. Evaluates embedding quality using multiple metrics:
   - Trustworthiness
   - Reconstruction error
   - Cophenetic correlation
   - Neighborhood preservation
4. Analyzes cluster quality using KNN classification
5. Creates publication-quality visualizations:
   - Scatter plots with crop type coloring
   - Scatter plots with management practice coloring
   - Thumbnail overlays of representative images
   - KNN-based nearest neighbor visualizations
6. Generates comprehensive evaluation reports

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports

# Libs
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import sklearn.metrics
import pandas as pd
import matplotlib.offsetbox as osb
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.manifold import trustworthiness

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp
from PIL import Image
from matplotlib.colors import to_rgba
from umap import UMAP
import time
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# Own modules
from PatchCROPDataModule import PatchCROPDataModule#, KFoldLoop#, SpatialCVModel
from directory_listing import output_dirs, data_dirs, input_files_rgb




# Function to get color for a crop type
def get_crop_color(crop_type):
    """
    Get a consistent color for a crop type using a colorblind-friendly colormap.
    
    Args:
        crop_type: Name of the crop type (e.g., 'lupine', 'maize', etc.)
    
    Returns:
        Color value from the colormap for the given crop type
    """
    # Define consistent color mapping for crops
    ctype = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    taxonomic_groups_to_color = {
        '0': 1/10, '1': 2/10, '2': 3/10,
        '3': 4/10, '4': 5/10, '5': 6/10, 
        '6': 7/10, '7': 8/10, '8': 9/10
    }

    # Create a mapping from crop names to taxonomic groups
    crop_to_taxonomic = {
        'lupine': '0',
        'maize': '1',
        'sunflower': '2',
        'soy': '3'
    }

    # Get a colorblind-friendly colormap
    # Options: 'Set2', 'Set3', 'Paired', 'Dark2', 'Accent'
    cmap = plt.cm.get_cmap('Set2', 8)  # Set2 is colorblind-friendly
    
    if crop_type in crop_to_taxonomic:
        tax_group = crop_to_taxonomic[crop_type]
        color_value = taxonomic_groups_to_color[tax_group]
        return cmap(color_value)
    return 'gray'  # fallback color

def get_management_color(patch_id):
    """
    Get a consistent color for a management practice using a colorblind-friendly colormap.
    
    Args:
        patch_id: ID of the patch/field
    
    Returns:
        Color value from the colormap for the given management practice
    """
    # Define management practice groups
    conventional_management = ['74', '90', '65', '89', '95_July']
    reduced_pesticides_management = ['50', '68', '58', '59', '76_July']
    reduced_pesti_flower_strps_management = ['20', '110', '19', '119', '115_July']
    
    # Create a mapping from management practices to colors
    management_to_color = {
        'conventional': 0.2,
        'reduced_pesticides': 0.5,
        'reduced_pesti_flower_strps': 0.8
    }
    
    # Get a different colorblind-friendly colormap for management practices
    cmap = plt.colormaps['Dark2']
    
    if str(patch_id) in conventional_management:
        return cmap(management_to_color['conventional'])
    elif str(patch_id) in reduced_pesticides_management:
        return cmap(management_to_color['reduced_pesticides'])
    elif str(patch_id) in reduced_pesti_flower_strps_management:
        return cmap(management_to_color['reduced_pesti_flower_strps'])
    return 'gray'  # fallback color

def get_scatter_plot_with_thumbnails(embeddings_2d, img_stack, crop_types, color, fig, kernel_size):
    """
    Creates a figure with two subplots: a scatter plot on the left and a scatter plot with thumbnails overlaid on the right.
    
    Args:
        embeddings_2d: 2D UMAP embeddings
        img_stack: Stack of images corresponding to embeddings
        crop_types: List of crop types for each embedding
        color: Optional color array for points
        fig: Optional figure to plot on
        kernel_size: Size of image thumbnails
    
    Returns:
        Tuple of (figure, list of colors used for points)
    """
    # Initialize empty figure
    fig = plt.figure(figsize=(8.27, 4))
    
    ax_scatter = fig.add_subplot(1,2,1)
    ax_scatter_with_thumbnails = fig.add_subplot(1,2,2)
    
    # Create color array based on crop types using the new color scheme
    crop_colors = [get_crop_color(crop_type) for crop_type in crop_types]
    
    # Scatter plot on the left
    ax_scatter.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='.', c=crop_colors, alpha=0.8)
    
    # Add legend manually
    unique_crop_types = np.unique(crop_types)
    for crop_type in unique_crop_types:
        color = get_crop_color(crop_type)
        ax_scatter.scatter([], [], marker='.', c=[color], label=crop_type)
    ax_scatter.legend(fontsize=13)

    # Scatter plot with thumbnails on the right
    ax_scatter_with_thumbnails.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='.', c=crop_colors, alpha=0.8)
    ax_scatter_with_thumbnails.set_yticklabels([])
    ax_scatter_with_thumbnails.set_ylabel("")

    # Overlay thumbnails on the scatter plot
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)

    for i in iterator:
        # Only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 0.2:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    for idx in shown_images_idx:
        thumbnail_size = int(rcp["figure.figsize"][0] * 4.0)
        img = img_stack[idx].astype(np.uint8)
        img = Image.fromarray(img)
        img = functional.resize(img, thumbnail_size)
        
        # Use the same color coding as in the scatter plot
        frame_color = get_crop_color(crop_types[idx])

        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.1,
            frameon=True,
            bboxprops=dict(edgecolor=frame_color, linewidth=2),
            boxcoords="data",
        )
        ax_scatter_with_thumbnails.add_artist(img_box)

    # Adjust the aspect ratio and limits for both plots to match
    ax_scatter.set_aspect(1.0 / ax_scatter.get_data_ratio(), adjustable="box")
    ax_scatter_with_thumbnails.set_aspect(1.0 / ax_scatter_with_thumbnails.get_data_ratio(), adjustable="box")

    # Set x and y labels with new font sizes
    ax_scatter.set_xlabel("UMAP Component 1", fontsize=11)
    ax_scatter_with_thumbnails.set_xlabel("UMAP Component 1", fontsize=11)
    ax_scatter.set_ylabel("UMAP Component 2", fontsize=11)

    return fig, crop_colors


def plot_nearest_neighbors_3x3(example_img_ids, imgs, embeddings, crop_type_name, color, per_crop_accuracy=None):
    """
    Plots the example image and its eight nearest neighbors in a 3x3 grid.
    
    Args:
        example_img_ids: Index of the example image
        imgs: Stack of all images
        embeddings: Embeddings for all images
        crop_type_name: Name of the crop type
        color: Color to use for the frame
        per_crop_accuracy: Optional accuracy for this crop type
    
    Returns:
        Figure containing the visualization
    """
    n_subplots = 9
    fig = plt.figure(figsize=(4, 6))
    distances = embeddings - embeddings[example_img_ids]
    distances = np.power(distances, 2).sum(-1).squeeze()
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)

        if plot_offset == 0:
            title = ''
            if per_crop_accuracy is not None:
                title = f"\t \t \t \t $\mathbf{{{crop_type_name.capitalize()}}}$ \t Accuracy: {per_crop_accuracy:.2f}%"

            title += f"\nExample"
            ax.set_title(title, fontsize=16)
            img = imgs[example_img_ids].astype(np.uint8)
            img = Image.fromarray(img)
            plt.imshow(img)
        else:
            ax.set_title(f"d={distances[plot_idx]:.3f}", fontsize=16)
            img = imgs[plot_idx].astype(np.uint8)
            img = Image.fromarray(img)
            plt.imshow(img)
        plt.axis("off")
    return fig


def plot_nearest_neighbors_2x3(example_img_ids, imgs, embeddings, crop_type_name, color, per_crop_accuracy=None, k=5):
    """
    Plots the example image and its k-1 nearest neighbors in a 2x3 grid.
    
    Args:
        example_img_ids: Index of the example image
        imgs: Stack of all images
        embeddings: Embeddings for all images
        crop_type_name: Name of the crop type
        color: Color to use for the frame
        per_crop_accuracy: Optional accuracy for this crop type
        k: Number of neighbors to show (default: 5)
    
    Returns:
        Figure containing the visualization
    """
    # Always use 5 neighbors to fit in 2x3 grid (6 total spaces)
    n_subplots = 6  # Fixed size for 2x3 grid
    fig, axes = plt.subplots(3, 2, figsize=(4, 6))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Calculate distances and get nearest neighbors
    distances = embeddings - embeddings[example_img_ids]
    distances = np.power(distances, 2).sum(-1).squeeze()
    nearest_neighbors = np.argsort(distances)[:n_subplots]  # Always get 5 neighbors + example

    # Format title with crop type name and accuracy
    title = ''
    if per_crop_accuracy is not None:
        title = f"$\mathbf{{{crop_type_name.capitalize()}}}$  Accuracy: {per_crop_accuracy:.2f}%"

    # Position the title at the top center of the figure
    fig.suptitle(title, fontsize=28, ha='center')

    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        if plot_offset >= n_subplots:  # Safety check
            break
            
        ax = axes[plot_offset]  # Get the correct subplot axis

        if plot_offset == 0:
            ax.set_title(f"Example", fontsize=15)
            img = imgs[example_img_ids].astype(np.uint8)
            img = Image.fromarray(img)
            ax.imshow(img)
        else:
            img = imgs[plot_idx].astype(np.uint8)
            img = Image.fromarray(img)
            ax.imshow(img)
        ax.axis("off")

    # Draw a colored frame around the entire figure
    fig.patch.set_edgecolor(color)
    fig.patch.set_linewidth(10)

    return fig

def create_combined_visualization(embeddings_2d, img_stack, crop_types, crop_type_names_list, example_image_ids, per_crop_accuracy, kernel_size, best_k, split_name, patch_ids):
    """
    Creates a combined figure with scatter plots at the top (2/3) and four crop-specific KNN plots at the bottom (1/3).
    
    Args:
        embeddings_2d: 2D UMAP embeddings
        img_stack: Stack of images corresponding to embeddings
        crop_types: List of crop types for each embedding
        crop_type_names_list: List of unique crop type names
        example_image_ids: List of example image indices for each crop type
        per_crop_accuracy: Dictionary of accuracies for each crop type
        kernel_size: Size of image thumbnails
        best_k: Best k value for KNN
        split_name: Name of the data split (train/val/test)
        patch_ids: List of patch IDs
    
    Returns:
        Figure containing the combined visualization
    """
    # Create a figure with adjusted height ratio and reduced margins
    fig = plt.figure(figsize=(12, 11))
    
    # Create a grid with 2/3 - 1/3 ratio and minimal spacing
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.2, left=0.04, top=0.99)
    
    # Get initial plot positions to align labels
    fig.canvas.draw()
    
    # Top grid for scatter plots - split vertically
    top_gs = gs[0].subgridspec(2, 2, height_ratios=[1, 1], wspace=0.01, hspace=0.01)
    
    # Add subplots for scatter plots
    ax_crop = fig.add_subplot(top_gs[0, 0])  # Upper left - crop type
    ax_management = fig.add_subplot(top_gs[1, 0])  # Lower left - management
    ax_thumbnails = fig.add_subplot(top_gs[:, 1])  # Right side - thumbnails
    
    # Create color arrays based on crop types
    crop_colors = [get_crop_color(crop_type) for crop_type in crop_types]
    
    # Load the image-to-patch mapping
    image_to_patch = torch.load(os.path.join(this_output_dir, 'global_patch_ids_' + split_name + '.pt'))
    
    # Create management colors using the actual mapping
    management_colors = [get_management_color(patch_id) for patch_id in image_to_patch]
    
    # Crop type scatter plot (upper left)
    scatter_crop = ax_crop.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 marker='.', c=crop_colors, alpha=0.6, s=150)
    
    # Add legend for crop types
    unique_crop_types = np.unique(crop_types)
    for crop_type in unique_crop_types:
        color = get_crop_color(crop_type)
        label = 'Soybean' if crop_type == 'soy' else crop_type.capitalize()
        ax_crop.scatter([], [], marker='.', c=[color], alpha=0.6, label=label, s=150)
    ax_crop.legend(fontsize=12)
    
    # Management practice scatter plot (lower left)
    scatter_management = ax_management.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                             marker='.', c=management_colors, alpha=0.6, s=150)
    
    # Add legend for management practices with correct colors
    management_labels = ['Conventional', 'Reduced', 'Reduced + Flower']
    management_colors_legend = [
        get_management_color('74'),  # conventional
        get_management_color('50'),  # reduced pesticides
        get_management_color('20')   # reduced pesticides + flower strips
    ]
    
    for label, color in zip(management_labels, management_colors_legend):
        ax_management.scatter([], [], marker='.', c=[color], 
                            alpha=0.6, label=label, s=150)
    ax_management.legend(fontsize=12)
    
    # Scatter plot with thumbnails (right)
    ax_thumbnails.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         marker='.', c=crop_colors, alpha=0.6, s=150)
    
    # Add thumbnails to right plot
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    
    for i in iterator:
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 0.2:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
        
        thumbnail_size = int(rcp["figure.figsize"][0] * 7.2)
        img = img_stack[i].astype(np.uint8)
        img = Image.fromarray(img)
        img = functional.resize(img, thumbnail_size)
        
        frame_color = get_crop_color(crop_types[i])
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[i],
            pad=0.1,
            frameon=True,
            bboxprops=dict(edgecolor=frame_color, linewidth=2),
            boxcoords="data",
        )
        ax_thumbnails.add_artist(img_box)
    
    # Set labels for all subplots
    for ax in [ax_crop, ax_management, ax_thumbnails]:
        ax.set_xlabel("UMAP Component 1", fontsize=11)
        if ax != ax_thumbnails:  # Only add y-label to left plots
            ax.set_ylabel("UMAP Component 2", fontsize=11)

    # Bottom grid for KNN plots with minimal spacing
    bottom_gs = gs[1].subgridspec(1, 4, wspace=0.01)
    
    # Store the axes and their properties for later frame drawing
    knn_plots = []
    
    # First pass: Create all plots and collect their information
    for i, (crop_type_name, ex_img) in enumerate(zip(crop_type_names_list, example_image_ids)):
        ax_knn = fig.add_subplot(bottom_gs[i])
        
        # Calculate distances and get nearest neighbors
        distances = embeddings_2d - embeddings_2d[ex_img]
        distances = np.power(distances, 2).sum(-1).squeeze()
        nearest_neighbors = np.argsort(distances)[:6]
        
        # Remove x and y axis tick labels
        ax_knn.set_xticklabels([])
        ax_knn.set_yticklabels([])
        
        # Create a tighter 2x3 grid within this subplot
        inner_gs = bottom_gs[i].subgridspec(3, 2, wspace=0.01, hspace=0.01)
        
        # Format title with crop type name and accuracy
        name = crop_type_name.capitalize() if crop_type_name != 'soy' else 'Soybean'
        title = f"$\mathbf{{{name}}}$  Acc.: {per_crop_accuracy[crop_type_name]/100:.2f}"
        
        # Set the title with minimal padding
        ax_knn.set_title(title, fontsize=10, pad=7, y=1.05)
        
        # Plot the example and neighbors
        for plot_offset, plot_idx in enumerate(nearest_neighbors):
            if plot_offset >= 6:
                break
                
            row = plot_offset // 2
            col = plot_offset % 2
            
            ax_img = fig.add_subplot(inner_gs[row, col])
            
            if plot_offset == 0:
                img = img_stack[ex_img].astype(np.uint8)
                img = Image.fromarray(img)
                ax_img.imshow(img)
                ax_img.set_title("Example", fontsize=10, pad=5, y=1.01)
            else:
                img = img_stack[plot_idx].astype(np.uint8)
                img = Image.fromarray(img)
                ax_img.imshow(img)
            
            ax_img.set_xticklabels([])
            ax_img.set_yticklabels([])
            ax_img.axis("off")
        
        # Store the axis and its properties for later
        knn_plots.append({
            'ax': ax_knn,
            'crop_type_name': crop_type_name,
            'per_crop_accuracy': per_crop_accuracy[crop_type_name]
        })
    
    # Draw the figure once to ensure all elements are properly positioned
    fig.canvas.draw()
    
    # Get positions for label alignment
    scatter_bbox = ax_crop.get_position()
    first_knn_bbox = knn_plots[0]['ax'].get_position()
    
    # Add subplot labels aligned with plot tops
    fig.text(0.015, scatter_bbox.y1, 'a', fontsize=12, fontweight='bold')
    fig.text(0.015, first_knn_bbox.y1 + 0.02, 'b', fontsize=12, fontweight='bold')
    
    # Second pass: Add the frames with aligned positions
    frame_top = None
    for plot_info in knn_plots:
        ax_knn = plot_info['ax']
        crop_type_name = plot_info['crop_type_name']
        
        # Get the frame color for this crop type
        frame_color = get_crop_color(crop_type_name)
        
        # Get the exact position of the title and subplot
        title_bbox = ax_knn.title.get_window_extent()
        subplot_bbox = ax_knn.get_window_extent()
        
        # Convert bounding boxes to figure coordinates
        title_bbox = title_bbox.transformed(fig.transFigure.inverted())
        subplot_bbox = subplot_bbox.transformed(fig.transFigure.inverted())
        
        # Store or use the frame top position to align all frames
        if frame_top is None:
            frame_top = title_bbox.y1 + 0.01  # Small padding above title
        
        # Create rectangle with aligned top position
        rect = plt.Rectangle(
            (subplot_bbox.x0, subplot_bbox.y0-0.01),
            subplot_bbox.width,
            frame_top - subplot_bbox.y0,
            transform=fig.transFigure,
            fill=False,
            edgecolor=frame_color,
            linewidth=2,
            zorder=900
        )
        fig.add_artist(rect)
        
        # Ensure title is drawn on top of the frame
        ax_knn.title.set_zorder(1000)
        
        # Remove the default spines
        for spine in ax_knn.spines.values():
            spine.set_visible(False)
    
    return fig

def evaluate_umap_quality(original_embeddings, umap_embeddings):
    """
    Evaluate UMAP embedding quality using multiple metrics.
    
    Args:
        original_embeddings: Original high-dimensional embeddings
        umap_embeddings: 2D UMAP embeddings
    
    Returns:
        Dictionary containing quality metrics:
        - trustworthiness: Measure of local structure preservation
        - reconstruction_error: Error in distance preservation
        - cophenetic_correlation: Correlation of distance matrices
        - neighborhood_preservation: Preservation of k-nearest neighbors
    """
    metrics = {
        'trustworthiness': trustworthiness(
            original_embeddings, 
            umap_embeddings, 
            n_neighbors=30
        ),
        'reconstruction_error': reconstruction_error(
            original_embeddings, 
            umap_embeddings
        ),
        'cophenetic_correlation': cophenetic_correlation(
            original_embeddings, 
            umap_embeddings
        ),
        'neighborhood_preservation': neighborhood_preservation(
            original_embeddings, 
            umap_embeddings,
            k=10
        )
    }
    return metrics

def reconstruction_error(original, embedded):
    """
    Calculate reconstruction error between original and embedded spaces.
    
    Args:
        original: Original high-dimensional embeddings
        embedded: Lower-dimensional embeddings
    
    Returns:
        Mean squared error between normalized distance matrices
    """
    # Calculate pairwise distances in both spaces
    dist_original = pairwise_distances(original)
    dist_embedded = pairwise_distances(embedded)
    
    # Normalize distances
    dist_original = dist_original / dist_original.max()
    dist_embedded = dist_embedded / dist_embedded.max()
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((dist_original - dist_embedded) ** 2)
    return reconstruction_error

def cophenetic_correlation(original, embedded):
    """
    Calculate cophenetic correlation between original and embedded spaces.
    
    Args:
        original: Original high-dimensional embeddings
        embedded: Lower-dimensional embeddings
    
    Returns:
        Spearman correlation between distance matrices
    """
    # Calculate pairwise distances
    dist_original = pairwise_distances(original).flatten()
    dist_embedded = pairwise_distances(embedded).flatten()
    
    # Calculate correlation
    correlation, _ = spearmanr(dist_original, dist_embedded)
    return correlation

def neighborhood_preservation(original, embedded, k=10):
    """
    Calculate neighborhood preservation score.
    
    Args:
        original: Original high-dimensional embeddings
        embedded: Lower-dimensional embeddings
        k: Number of neighbors to consider (default: 10)
    
    Returns:
        Mean proportion of preserved neighbors
    """
    # Find k-nearest neighbors in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k).fit(original)
    indices_orig = nbrs_orig.kneighbors(original, return_distance=False)
    
    # Find k-nearest neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=k).fit(embedded)
    indices_embedded = nbrs_embedded.kneighbors(embedded, return_distance=False)
    
    # Calculate preservation
    preservation = np.mean([
        len(set(orig) & set(emb)) / k 
        for orig, emb in zip(indices_orig, indices_embedded)
    ])
    return preservation

def evaluate_cluster_quality(embeddings_2d, labels):
    """
    Evaluate cluster quality using KNN classifier with different k values.
    
    Args:
        embeddings_2d: 2D embeddings to evaluate
        labels: True labels for classification
    
    Returns:
        Dictionary containing results for each k value:
        - accuracy: Classification accuracy
        - report: Detailed classification report
        - cv_scores: Cross-validation scores
    """
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_2d, labels, test_size=0.2, random_state=42
    )
    
    k_values = [3, 5, 7, 10]
    results = {}
    
    for k in k_values:
        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Get predictions
        y_pred = knn.predict(X_test)
        
        # Calculate accuracy and get detailed report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(knn, embeddings_2d, labels, cv=5)
        
        results[k] = {
            'accuracy': accuracy,
            'report': report,
            'cv_scores': cv_scores
        }
    
    return results

def evaluate_umap_knn(embeddings, labels, umap_params):
    """
    Evaluate UMAP parameters using KNN classifier.
    
    Args:
        embeddings: Original embeddings
        labels: True labels
        umap_params: UMAP parameters to evaluate
    
    Returns:
        KNN classification accuracy
    """
    # Create UMAP embeddings with given parameters
    umap_2d = UMAP(
        n_components=2,
        random_state=42,
        **umap_params
    )
    embeddings_2d = umap_2d.fit_transform(embeddings)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_2d, labels, test_size=0.2, random_state=42
    )
    
    # Train and evaluate KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    
    return score

def get_management_label(patch_id):
    """
    Convert patch ID to management practice label.
    
    Args:
        patch_id: ID of the patch/field
    
    Returns:
        Management practice label (conventional, reduced_pesticides, reduced_pesti_flower_strps, or unknown)
    """
    conventional_management = ['74', '90', '65', '89', '95_July']
    reduced_pesticides_management = ['50', '68', '58', '59', '76_July']
    reduced_pesti_flower_strps_management = ['20', '110', '19', '119', '115_July']
    
    if str(patch_id) in conventional_management:
        return 'conventional'
    elif str(patch_id) in reduced_pesticides_management:
        return 'reduced_pesticides'
    elif str(patch_id) in reduced_pesti_flower_strps_management:
        return 'reduced_pesti_flower_strps'
    return 'unknown'

data_root = '/media/stillsen/Megalodon/Projects/PatchCROP/2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_from_HPC/'

this_output_dir = '/media/stillsen/Megalodon/Projects/PatchCROP/Output/2024_SSL/Results_Pub_Retrain/Combined_SCV_large_July_VICRegConvNext_SCV_E500lightly-VICReg_kernel_size_224_DS-combined_SCV_large_July_ConvNext-back_14fields_tuning'

ctype= ['0','1','2','3','4','5','6','7','8']
taxonomic_groups_to_color = {'0': 1/10, '1': 2/10, '2': 3/10,
                             '3': 4/10,
                             '4': 5/10, '5': 6/10, '6':7/10, '7':8/10, '8':9/10}
cc = [taxonomic_groups_to_color[x] for x in ctype]
cmap = plt.cm.get_cmap('tab10', 9)
colors = cmap(cc)

# patch_no = 'combined_SCV_large'

num_folds = 4

# this_output_dir = output_dirs[patch_no]
test_patch_name = ''

seed = 42
stride = 60
kernel_size = 224
dataset_type = 'swDS'

augmentation = False
features = 'RGB'
batch_size = 1
validation_strategy = 'SCV'
num_samples_per_fold = None
fake_labels = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('working on device %s' % device)
if device == 'cpu':
    workers = os.cpu_count()
else:
    workers = 1
    print('\twith {} workers'.format(workers))

patch_ids = [74, 90, 50, 68, 20, 110, '76_July', '95_July', '115_July', 65, 19, 89, 59, 119]
combined_dataset = {'train': np.empty((0, 224, 224, 3)),
                    'val': np.empty((0, 224, 224, 3)),
                    'test': np.empty((0, 224, 224, 3)),
                    }
sets = [
        'train',
        'val',
        'test',
        ]

thumbnails = True

# Create subfolder for results
results_dir = os.path.join(this_output_dir, 'embedding_analysis')
os.makedirs(results_dir, exist_ok=True)

for s in sets:
    print(f"\nProcessing {s} split...")
    
    fig = None
    crop_type_names = torch.load(os.path.join(this_output_dir, 'global_crop_type_names_' + s + '.pt'))
    embeddings = torch.load(os.path.join(this_output_dir, 'global_embeddings_' + s + '.pt'))
    image_set = torch.load(os.path.join(this_output_dir, 'global_imgs_' + s + '.pt'))

    # Convert embeddings to numpy array if needed
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    # Define example image IDs based on the split
    if s == 'train':
        example_image_ids = [150, 2652+10, 3910+10, 4760+10]  # All Folds
    elif s == 'val':
        example_image_ids = [150, 1176+10, 1764+10, 2142+10]
    elif s == 'test':
        example_image_ids = [150, 1250+10, 1880+10, 2285+10]

    # Convert crop type names to categorical numerical values
    numerical_crop_types = pd.Categorical(crop_type_names)

    # UMAP parameter grid search
    param_grid = {
        'min_dist': [0.05, 0.1, 0.2, 0.3],
        'n_neighbors': [15, 30, 50, 70]
    }

    # Record UMAP start time
    umap_start_time = time.time()

    best_score = 0
    best_params = None

    for min_dist in param_grid['min_dist']:
        for n_neighbors in param_grid['n_neighbors']:
            params = {'min_dist': min_dist, 'n_neighbors': n_neighbors}
            score = evaluate_umap_knn(embeddings, numerical_crop_types, params)
            print(f"min_dist={min_dist}, n_neighbors={n_neighbors}: accuracy={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_params = params

    print(f"\nBest UMAP parameters: {best_params}")
    print(f"Best KNN accuracy: {best_score:.3f}")

    # Create UMAP embeddings with best parameters
    umap_2d = UMAP(
        n_components=2,
        random_state=seed,
        **best_params
    )
    embeddings_2d = umap_2d.fit_transform(embeddings)

    # Calculate UMAP processing time
    umap_time = time.time() - umap_start_time

    # Evaluate UMAP quality
    print("Evaluating UMAP quality metrics...")
    umap_quality_metrics = evaluate_umap_quality(embeddings, embeddings_2d)

    # Load patch IDs
    patch_ids = torch.load(os.path.join(this_output_dir, 'global_patch_ids_' + s + '.pt'))
    
    # Convert patch IDs to management practice labels
    management_labels = [get_management_label(pid) for pid in patch_ids]
    management_labels = pd.Categorical(management_labels)
    
    # Evaluate cluster quality for both crop types and management practices
    knn_results_crop = evaluate_cluster_quality(embeddings_2d, numerical_crop_types)
    knn_results_management = evaluate_cluster_quality(embeddings_2d, management_labels)
    
    # Find best k for both
    best_k_crop = None
    best_accuracy_crop = 0
    for k, results in knn_results_crop.items():
        if results['accuracy'] > best_accuracy_crop:
            best_accuracy_crop = results['accuracy']
            best_k_crop = k

    best_k_management = None
    best_accuracy_management = 0
    for k, results in knn_results_management.items():
        if results['accuracy'] > best_accuracy_management:
            best_accuracy_management = results['accuracy']
            best_k_management = k

    # Calculate per-crop accuracy using best k
    X_train, X_test, y_train, y_test = train_test_split(embeddings_2d, numerical_crop_types, test_size=0.2, random_state=42)
    knn_crop = KNeighborsClassifier(n_neighbors=best_k_crop, weights='distance')
    knn_crop.fit(X_train, y_train)
    y_pred_crop = knn_crop.predict(X_test)

    # Calculate management practice accuracy
    X_train_management, X_test_management, y_train_management, y_test_management = train_test_split(
        embeddings_2d, management_labels, test_size=0.2, random_state=42
    )
    knn_management = KNeighborsClassifier(n_neighbors=best_k_management, weights='distance')
    knn_management.fit(X_train_management, y_train_management)
    y_pred_management = knn_management.predict(X_test_management)

    # Define labels for reports
    crop_type_names_list = ['maize', 'sunflower', 'soy', 'lupine']
    management_names_list = ['conventional', 'reduced_pesticides', 'reduced_pesti_flower_strps']
    
    # Get classification reports
    classification_report_crop = classification_report(y_test, y_pred_crop, output_dict=True, target_names=crop_type_names_list)
    classification_report_management = classification_report(y_test_management, y_pred_management, output_dict=True, target_names=management_names_list)
    
    # Calculate per-crop and per-management accuracies
    per_crop_accuracy = {crop_type: classification_report_crop[crop_type]['precision'] * 100 
                        for crop_type in crop_type_names_list}
    per_management_accuracy = {management: classification_report_management[management]['precision'] * 100 
                             for management in management_names_list}

    # Calculate mean accuracies
    mean_accuracy_crop = sum(per_crop_accuracy.values()) / len(per_crop_accuracy)
    mean_accuracy_management = sum(per_management_accuracy.values()) / len(per_management_accuracy)

    # Save metrics with split name clearly indicated
    metrics_file = os.path.join(results_dir, f'embedding_metrics_{s}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Metrics for {s.upper()} split\n")
        f.write("======================\n\n")
        
        f.write(f"UMAP Processing Metrics:\n")
        f.write(f"----------------------\n")
        f.write(f"Number of samples: {len(embeddings)}\n")
        f.write(f"Processing time: {umap_time:.2f} seconds\n")
        f.write(f"Processing time per sample: {umap_time/len(embeddings):.4f} seconds\n\n")
        
        f.write(f"UMAP Parameters:\n")
        f.write(f"----------------------\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best KNN accuracy: {best_score:.3f}\n\n")
        
        f.write(f"UMAP Quality Metrics:\n")
        f.write(f"----------------------\n")
        for metric_name, value in umap_quality_metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
        f.write("\n")
        
        f.write(f"Crop Type Classification Metrics:\n")
        f.write(f"----------------------\n")
        f.write(f"Best k value: {best_k_crop}\n")
        f.write(f"Per-crop accuracy: {per_crop_accuracy}\n")
        f.write(f"Mean accuracy across all crop types: {mean_accuracy_crop:.2f}%\n\n")
        
        f.write(f"Management Practice Classification Metrics:\n")
        f.write(f"----------------------\n")
        f.write(f"Best k value: {best_k_management}\n")
        f.write(f"Per-management accuracy: {per_management_accuracy}\n")
        f.write(f"Mean accuracy across all management practices: {mean_accuracy_management:.2f}%\n\n")
        
        # Add detailed classification reports
        f.write("\nDetailed Crop Type Classification Report:\n")
        f.write("---------------------------\n")
        f.write(classification_report(y_test, y_pred_crop, target_names=crop_type_names_list))
        
        f.write("\nDetailed Management Practice Classification Report:\n")
        f.write("---------------------------\n")
        f.write(classification_report(y_test_management, y_pred_management, target_names=management_names_list))

    # Create and save scatter plot visualization first to get the color scheme
    print("Creating scatter plot visualization...")
    fig, plot_colors = get_scatter_plot_with_thumbnails(
        embeddings_2d=embeddings_2d,
        img_stack=image_set,
        crop_types=crop_type_names,
        color=None,
        fig=None,
        kernel_size=kernel_size
    )
    fig.savefig(os.path.join(results_dir, f'embedding_scatter_{s}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create a mapping of crop types to their colors
    unique_crop_types = np.unique(crop_type_names)
    color_mapping = {crop_type: get_crop_color(crop_type) for crop_type in unique_crop_types}

    # Create and save KNN visualizations using the same color scheme
    for crop_type_name, ex_img in zip(crop_type_names_list, example_image_ids):
        # Get the color for this crop type using the global color map
        frame_color = get_crop_color(crop_type_name)
        fig = plot_nearest_neighbors_2x3(
            example_img_ids=ex_img,
            imgs=image_set,
            embeddings=embeddings_2d,
            crop_type_name=crop_type_name,
            per_crop_accuracy=per_crop_accuracy[crop_type_name],
            color=frame_color,
            k=best_k_crop
        )
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f'embedding_KNN_{s}_{crop_type_name}_4.png'), dpi=100)
        plt.close(fig)
        
    # Create and save the combined visualization
    print(f"Creating combined visualization for {s} split...")
    combined_fig = create_combined_visualization(
        embeddings_2d=embeddings_2d,
        img_stack=image_set,
        crop_types=crop_type_names,
        crop_type_names_list=crop_type_names_list,
        example_image_ids=example_image_ids,
        per_crop_accuracy=per_crop_accuracy,
        kernel_size=kernel_size,
        best_k=best_k_crop,
        split_name=s,
        patch_ids=patch_ids
    )
    combined_fig.savefig(os.path.join(results_dir, f'embedding_combined_{s}_with_management.png'), dpi=300, bbox_inches='tight')
    plt.close(combined_fig)

    print(f"Completed processing {s} split")

print("Analysis complete!")


