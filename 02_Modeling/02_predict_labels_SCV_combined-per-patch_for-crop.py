# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script generates yield predictions for each crop type using trained models.

The script:
1. Loads trained models for each fold and crop type
2. Generates yield predictions for training and validation data
3. Evaluates prediction performance using R² score
4. Saves predictions and performance metrics for analysis
5. Handles multiple architectures (ResNet18, ConvNext, VICReg, VICRegConvNext)

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports
import os, random
from datetime import date

# Libs
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
from torch.utils.data.dataloader import DataLoader
from torch import nn
from collections import OrderedDict

# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from directory_listing import output_dirs, data_dirs, input_files_rgb

import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support


def setup_multiprocessing() -> int:
    """
    Set up multiprocessing configuration for prediction.
    
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
    Main function for generating yield predictions for each crop type.
    
    This function:
    1. Sets up random seeds for reproducibility
    2. Configures multiprocessing
    3. Sets up hyperparameters and model configuration
    4. Iterates through each architecture and crop type
    5. Loads trained models and generates predictions
    6. Evaluates and saves prediction performance
    
    The function handles multiple architectures (ResNet18, ConvNext, VICReg, VICRegConvNext)
    and processes each crop type (maize, sunflower, soy, lupine) separately.
    """
    # Add debug message at start
    print("Initializing main function with configurations...")
    
    seed = 42
    # seed_everything(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # HYPERPARAMETERS
    num_epochs = 2000
    # num_epochs = 1000

    num_epochs_finetuning = 10
    lr = 0.001
    lr_finetuning = 0.0001
    momentum = 0.9
    wd = 0.0005 
    classes = 1
    batch_size = 1
    num_folds = 4
    min_delta = 0.001
    patience = 10
    min_epochs = num_epochs
    repeat_trainset_ntimes = 1


    stride = 60
    kernel_size = 224

    dataset_type='pointDS'

    compute_singular_loss = True


    architectures = ['resnet18', 'ConvNeXt', 'VICReg', 'VICRegConvNext']


    strategy = None
    augmentation = False
    tune_fc_only = True
    pretrained = False
    features = 'RGB'

    num_samples_per_fold = None

    validation_strategy = 'SCV'  # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
    file_suffix = ''
    criterion = nn.MSELoss(reduction='mean')

    fake_labels = False
    training_response_normalization = False

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('working on device %s' % device)
    workers = setup_multiprocessing()

    # Define which architectures were trained for each crop
    crop_architecture_mapping = {
        'maize': ['VICReg', 'VICRegConvNext'],
        'sunflower': ['resnet18', 'ConvNeXt', 'VICReg', 'VICRegConvNext'],
        'soy': ['VICReg', 'VICRegConvNext'],
        'lupine': ['VICReg', 'VICRegConvNext']
    }

    # Add debug message before the loops
    print(f"Starting processing for architectures: {architectures}")
    
    for architecture in architectures:
        print(f"\nProcessing architecture: {architecture}")

        sets = [
            'train', 'val',
                'test',
                ]
        # sets = ['val']
        global_pred = dict()
        global_y = dict()
        local_r2 = dict()
        local_r = dict()
        global_r2 = dict()
        global_r = dict()

        patch_no = None

        patch_ids_combined = [
            'combined_SCV_large_maize',
            'combined_SCV_large_sunflower_July',
            'combined_SCV_large_soy',
            'combined_SCV_large_lupine'
        ]
        crop_type_names = [
            'maize',
            'sunflower',
            'soy',
            'lupine'
        ]
        suffixes = [
            '_per-crop-maize',
            '_per-crop-sunflower',
            '_per-crop-soy',
            '_per-crop-lupine'
        ]

        patch_ids = [
            [50, 68, 20, 110, 74, 90], # 'maize'
            ['76_July', '95_July', '115_July'], #'sunflower'
            # [95, 115], #'sunflower'
            [65, 19], #'soy'
            [89, 59, 119] #'lupine'
                    ]
        ############################################################################
        for patch_nos, crop_type_name, patch_no_combined, suffix in zip(patch_ids, crop_type_names, patch_ids_combined, suffixes):
            # Check if this architecture was trained for this crop
            if architecture not in crop_architecture_mapping[crop_type_name]:
                print(f"Skipping {architecture} for {crop_type_name} - not trained for this crop type")
                continue
                
            print(f"\nProcessing crop type: {crop_type_name}")
            print(f"Patch numbers: {patch_nos}")
            print(f"Combined patch ID: {patch_no_combined}")
            
            # Move this line before the if-elif block
            this_output_dir = None
            
            if architecture == 'VICReg':
                print("Setting up VICReg output directory")
                this_output_dir = output_dirs[patch_no_combined] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + '_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no_combined) + '_14fields_' + crop_type_name
            elif architecture == 'VICRegConvNext':
                print("Setting up VICRegConvNext output directory")
                this_output_dir = output_dirs[patch_no_combined] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + '_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no_combined) + '_14fields_' + crop_type_name
            elif architecture == 'ConvNeXt':
                print("Setting up ConvNeXt output directory")
                if dataset_type == 'swDS':
                    this_output_dir = output_dirs[patch_no_combined] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + '_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no_combined) + '_ConvNeXt_14fields_swDS_singular-loss'
                else:
                    this_output_dir = output_dirs[patch_no_combined] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + '_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no_combined) + '_ConvNeXt_14fields_pointDS_singular-loss'
            else:  # ResNet18
                print("Setting up ResNet18 output directory")
                if dataset_type == 'swDS':
                    this_output_dir = output_dirs[patch_no_combined] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + '_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no_combined) + '_resnet18_14fields_swDS_singular-loss'
                else:
                    this_output_dir = output_dirs[patch_no_combined] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + '_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no_combined) + '_resnet18_14fields_pointDS_singular-loss'

            if this_output_dir is None:
                print(f"ERROR: Failed to set output directory for architecture {architecture}")
                continue

            print(f"Output directory set to: {this_output_dir}")

            checkpoint_paths = ['model_f0.ckpt',
                                'model_f1.ckpt',
                                'model_f2.ckpt',
                                'model_f3.ckpt',
                                ]
            SSL = False
            fname = this_output_dir.split('/')[-1]
            if 'SSL' in fname or 'SimCLR' in fname or 'VICReg' in fname or 'SimSiam' in fname or 'VICRegL' in fname or 'VICRegLConvNext' in fname:
                checkpoint_paths = ['model_f0_domain-tuning.ckpt',
                                    'model_f1_domain-tuning.ckpt',
                                    'model_f2_domain-tuning.ckpt',
                                    'model_f3_domain-tuning.ckpt',
                                    ]
                SSL = True
                strategy = 'domain-tuning'

            checkpoint_paths = [this_output_dir + '/' + cp for cp in checkpoint_paths]

            for patch_no in patch_nos:
                print(f"\nProcessing individual patch: {patch_no}")
                
                # Add validation for patch_no
                if patch_no not in data_dirs:
                    print(f"ERROR: Invalid patch_no {patch_no}. Skipping...")
                    continue

                img_date = date(2020, 7, 3) if isinstance(patch_no, str) and 'July' in patch_no else date(2020, 8, 6)
                print('Setting up data in {} for date {}'.format(data_dirs[patch_no], img_date))
                
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
                file_suffix = str(patch_no)+'_'

                # set up model wrapper and input data
                model_wrapper = RGBYieldRegressor_Trainer(
                    pretrained=pretrained,
                    tune_fc_only=tune_fc_only,
                    architecture=architecture,
                    criterion=criterion,
                    device=device,
                    workers=workers,
                )
                for s in sets:
                    global_pred[s] = []
                    global_y[s] = []
                    local_r2[s] = []
                    local_r[s] = []

                    for k in range(num_folds):
                        print('predictions for set: {} fold: {}'.format(s, k))
                        # load data
                        # combined_datamodule.setup_fold(fold=k, dataset_type=dataset_type)
                        datamodule.setup_fold(fold=k, dataset_type=dataset_type)
                        dataloaders_dict = {
                                            'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(),
                                            'test': datamodule.test_dataloader(),
                                            }


                        if not compute_singular_loss:
                            model_wrapper.compute_singular_loss = compute_singular_loss
                            model_wrapper.mean_yield_per_crop = combined_datamodule.mean_yield_per_crop
                            model_wrapper.std_yield_per_crop = combined_datamodule.std_yield_per_crop

                        # set dataloaders
                        model_wrapper.set_dataloaders(dataloaders=dataloaders_dict)
                        # reinitialize FC layer for domain prediction, if SSL
                        if SSL:
                            if architecture == 'SimSiam' or architecture == 'VICRegL' or architecture == 'VICRegLConvNext' or architecture == 'VICReg' or architecture == 'VICRegConvNext':
                                model_wrapper.model.SSL_training = False
                            model_wrapper.reinitialize_fc_layers()
                        # load weights and skip rest of the method if already trained
                        model_loaded = model_wrapper.load_model_if_exists(model_dir=this_output_dir, strategy=strategy, k=k)
                        if not model_loaded: raise FileNotFoundError
                        print('model loaded')

                        # make prediction
                        # for each fold store labels and predictions
                        local_preds, local_labels = model_wrapper.predict(phase=s)

                        # save labels and predictions for each fold
                        torch.save(local_preds, os.path.join(this_output_dir, 'y_hat_' + file_suffix + s + '_' + str(k) + '.pt'))
                        torch.save(local_labels, os.path.join(this_output_dir, 'y_' + file_suffix + s + '_' + str(k) + '.pt'))

                        # for debugging, save labels and predictions in df
                        y_yhat_df = pd.DataFrame({'y': local_labels, 'y_hat': local_preds})
                        y_yhat_df.to_csv(os.path.join(this_output_dir, 'y-y_hat_{}{}_{}.csv'.format(file_suffix, s, k)), encoding='utf-8')


if __name__ == '__main__':
    print("Starting main function", flush=True)
    freeze_support()
    main()