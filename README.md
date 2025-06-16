# Self-Supervised Learning for Crop Classification and Yield Prediction

## Author

**Stefan Stiller**  
Leibniz-Centre for Agricultural Landscape Research (ZALF) e.V.  
Email: stefan [dot] stiller [at] zalf [dot] de, stillsen [at] gmail [dot] com  
ORCID: [0009-0004-7468-1678](https://orcid.org/0009-0004-7468-1678)

## Description

This repository contains the code for the study "Self-Supervised Learning Advances Crop Classification and Yield Prediction". The project aims to train, and evaluate a convolutional neural network with self-supervised learning for classifying crop types and predicting yields from RGB drone imagery across the four summer crops: lupine, sunflower, soy and maize. 

We pre-train the self-supervised model using VICReg across unlabeled images of all crop types across the entire fields. Then we exchange the projector for a prediction head and fine-tune specifically for each crop type by freezing the backbone. For file tuning, we use a dataset that clips images around each yield point's location. This setup includes fourfold spatial cross-validation and hyperparameter tuning. 

We evaluate self-supervised learning at two levels:
- At the level of embeddings
- At the level of the yield prediction. 

We evaluated the embeddings for differentiating between crop types. For that, we generated embeddings for each test image and reprojected them into two-dimensional space using Uniform Manifold Approximation and Projection (UMAP). We visually evalutated the emergence of clusters with respect to crop type in the 2D projected embeddings, and quantitatively using k-nearest neighbor classification. 

We evaluated yield prediction using Pearson's correlation coefficient between predicted and observed yield using a field-level prediction performance indicator. 

For a detailed description and results, see the publication or reach out for contact.

## Data Requirements

The code assumes the following data structure for each sample:
```python
self.data_set = (
    feature_kernel_tensor,  # PIL Image or numpy array of shape (H, W, C)
    label_kernel_tensor,    # numpy array of yield values
    crop_types,            # numpy array of crop type annotations
    dates                  # numpy array of image capture dates
)
```

Note: The data preprocessing scripts in `01_Data_Preprocessing/` are not included in this repository. If you need assistance with data preparation or have questions about the required data format, please contact the authors via email.

## Project Structure

```
.
├── 01_Data_Preprocessing/     # Data preparation scripts (not included)
├── 02_Modeling/              # Model training and evaluation scripts
│   ├── 01_train_lightly-SSL-VICReg_*.py  # SSL training scripts
│   ├── 02_predict_*.py       # Prediction and embedding generation
│   ├── PatchCROPDataModule.py           # Data handling module
│   ├── TransformTensorDataset.py        # Custom dataset class
│   ├── RGBYieldRegressor_Trainer.py    # Model training utilities
│   ├── TuneYieldRegressor.py           # Hyperparameter tuning
│   └── ModelSelection_and_Validation.py # Model evaluation
└── 03_Plots/                 # Visualization and analysis scripts
    ├── plot_all_losses_*.py  # Training loss visualization
    ├── plot_summary_fig_*.py # Model performance comparison
    └── plot_embedding_*.py   # Embedding visualization
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ssl-crop-yield-prediction.git
cd ssl-crop-yield-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Model Training

#### Self-Supervised Learning (SSL)
```bash
# Train SSL model with ResNet18 backbone
python 02_Modeling/01_train_lightly-SSL-VICReg_resnet18_SCV_Combined-SCV-14_SSL-Loss-together.py

# Train SSL model with ConvNeXt backbone
python 02_Modeling/01_train_lightly-SSL-VICReg_ConvNext_SCV_Combined-SCV-14_SSL-Loss-together.py
```

#### Fine-tuning
```bash
# Fine-tune ResNet18 model
python 02_Modeling/01_train_lightly-SSL-VICReg_resnet_SCV_Combined-SCV-14_SSL-Loss-together_FT-for-crop-type.py

# Fine-tune ConvNeXt model
python 02_Modeling/01_train_lightly-SSL-VICReg_ConvNext_SCV_Combined-SCV-14_SSL-Loss-together_FT-for-crop-type.py
```

### 2. Prediction and Analysis
```bash
# Generate embeddings
python 02_Modeling/02_predict_embeddings_combined-per-patch.py

# Make yield predictions
python 02_Modeling/02_predict_labels_SCV_combined-per-patch_for-crop.py
```

### 3. Visualization
```bash
# Plot training losses
python 03_Plots/plot_all_losses_SSL_publish.py
python 03_Plots/plot_all_losses_SL_publish.py

# Generate performance comparison plots
python 03_Plots/plot_summary_fig_SSL_combined-per-patch_multiple-models_FT-per-crop_retrain_publish.py

# Visualize embeddings
python 03_Plots/plot_embedding_scatter_per-crop.py
```

## Key Features

- Self-supervised learning with VICReg for crop yield prediction
- Support for multiple backbone architectures (ResNet18, ConvNeXt tiny)
- Spatial cross-validation for reduced overfitting and enhanced generalization
- Comprehensive visualization for model analysis
- Crop-type specific fine-tuning
- Embedding analysis and visualization

## Citation

If you use this code in your research, please cite our paper:
```
@article{your-paper-citation}
```

## License

This project is licensed under the GNU GPLv3 License - see the LICENSE file for details. 