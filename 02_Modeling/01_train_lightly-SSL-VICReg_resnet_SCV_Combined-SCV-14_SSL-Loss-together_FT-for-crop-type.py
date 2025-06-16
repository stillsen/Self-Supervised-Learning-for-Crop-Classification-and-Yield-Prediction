# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script implements fine-tuning of the VICReg model with ResNet18 backbone for crop-specific yield prediction.

The script:
1. Sets up data loading and preprocessing for each crop type
2. Loads pre-trained VICReg model with ResNet18 backbone
3. Performs fine-tuning for each crop type (maize, sunflower, soy, lupine)
4. Uses spatial cross-validation for model evaluation
5. Handles model training and hyperparameter tuning

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports
import os
import random
import time
import gc

# Libs
import numpy as np
from datetime import date

from lightly.loss import VICRegLoss
from lightly.transforms.vicreg_transform import VICRegTransform
import torch
from torch import nn

import warnings

# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from ModelSelection_and_Validation import ModelSelection_and_Validation
from directory_listing import output_dirs, data_dirs, input_files_rgb

import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support

def setup_multiprocessing() -> int:
    """
    Set up multiprocessing configuration for training.
    
    This function:
    1. Checks for available CUDA devices
    2. Configures the number of workers based on available GPUs
    3. Sets up the multiprocessing start method to 'spawn' for CUDA compatibility
    
    Returns:
        int: Number of workers to use for data loading
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f'Number of available CUDA devices: {num_gpus}')
        workers = num_gpus * 4  # A common heuristic is to use 4 workers per GPU
        print(f"Using {workers} workers")
        # Only set the start method if it hasn't been set already
        if not mp.get_start_method(allow_none=True):
            try:
                mp.set_start_method('spawn')
                print("Multiprocessing start method set to 'spawn'")
            except RuntimeError:
                print("Unable to set start method to 'spawn' as it has already been set")
        
        # Confirmation check
        current_method = mp.get_start_method()
        print(f"Current multiprocessing start method: {current_method}")
        
        if current_method != 'spawn':
            print("Warning: The start method is not 'spawn'. This may cause issues with CUDA.")
    else:
        workers = os.cpu_count()
        print(f"CUDA not available. Using {workers} CPU workers.")
    
    return workers


def main() -> None:
    """
    Main fine-tuning function for VICReg with ResNet18 backbone.
    
    This function:
    1. Sets up random seeds for reproducibility
    2. Configures multiprocessing
    3. Sets up hyperparameters and model configuration
    4. Iterates through each crop type for fine-tuning
    5. Initializes data modules and performs model fine-tuning
    """
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set up multiprocessing
    workers = setup_multiprocessing()

    # Your existing code continues here, starting with:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('working on device %s' % device)

    ## HYPERPARAMETERS
    num_epochs = 2000
    tune_epochs = 100

    lr = None
    momentum = 0.9
    wd = None
    classes = 1
    batch_size = None
    num_folds = 4
    min_delta = 0.01  # aka 1%
    patience = 10
    min_epochs = num_epochs
    duplicate_trainset_ntimes = 1

    stride = 60
    dataset_type = 'pointDS'
    compute_singular_loss = True

    kernel_size = 224
    architecture = 'VICReg'
    augmentation = True
    tune_fc_only = False
    pretrained = False
    features = 'RGB'
    num_samples_per_fold = None  # subsamples? -> None means do not subsample but take whole fold
    validation_strategy = 'SCV'  # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
    fake_labels = False
    training_response_normalization = False
    criterion = nn.MSELoss(reduction='mean')

    ############################################################################
    patch_ids = ['combined_SCV_large_maize', 'combined_SCV_large_sunflower_July', 'combined_SCV_large_soy', 'combined_SCV_large_lupine']
    crop_type_names = ['maize', 'sunflower', 'soy', 'lupine']
    ############################################################################

    for patch_no, crop_type_name in zip(patch_ids, crop_type_names):
        this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + '_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no) + '_14fields_' + crop_type_name

        if not os.path.exists(this_output_dir):
            print('creating: \t {}'.format(this_output_dir))
            os.mkdir(this_output_dir)
        else:
            warnings.warn("{} directory exists. WILL OVERRIDE.".format(this_output_dir))

        img_date = date(2020, 7, 3) if isinstance(patch_no, str) and 'July' in patch_no else date(2020, 8, 6)

        print('Setting up data in {}'.format(data_dirs[patch_no]))
        datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                         patch_id=patch_no,
                                         this_output_dir=this_output_dir,
                                         seed=seed,
                                         data_dir=data_dirs[patch_no],
                                         stride=stride,
                                         kernel_size=kernel_size,
                                         workers=workers,
                                         augmented=augmentation,
                                         input_features=features,
                                         batch_size=batch_size,
                                         validation_strategy=validation_strategy,
                                         fake_labels=fake_labels,
                                         img_date=img_date
                                         )
        print('\t preparing data \n\tkernel_size: {}\tstride: {}'.format(kernel_size, stride))
        datamodule.prepare_data(num_samples=num_samples_per_fold, dataset_type=dataset_type)

        print('\t splitting data into folds')
        models = ModelSelection_and_Validation(num_folds=num_folds,
                                               pretrained=pretrained,
                                               this_output_dir=this_output_dir,
                                               datamodule=datamodule,
                                               datamodule_SSL_pointDS=None,
                                               training_response_normalization=training_response_normalization,
                                               validation_strategy=validation_strategy,
                                               patch_no=patch_no,
                                               seed=seed,
                                               num_epochs=num_epochs,
                                               patience=patience,
                                               min_delta=min_delta,
                                               min_epochs=min_epochs,
                                               momentum=momentum,
                                               architecture=architecture,
                                               only_tune_hyperparameters=False,
                                               device=device,
                                               workers=workers,
                                               dataset_type=dataset_type,
                                               tune_epochs=tune_epochs,
                                               tune_by_downstream_performance=False,
                                               )
        models.train_and_tune_lightlySSL(start=1 if crop_type_name == 'maize' else 0, 
                                         SSL_type=architecture, 
                                         domain_training_enabled=True)


if __name__ == '__main__':
    print("Starting main function", flush=True)
    freeze_support()
    main()