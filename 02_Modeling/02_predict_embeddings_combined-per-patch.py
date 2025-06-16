# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script generates embeddings from trained models for internal sets (train and validation).

The script:
1. Loads trained models for each fold
2. Generates embeddings for training and validation data
3. Saves embeddings for downstream analysis
4. Handles both ResNet18 and ConvNext backbone models

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

from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer
from directory_listing import output_dirs, data_dirs, input_files_rgb

# Add to existing imports at the top
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
    Main function for generating embeddings from trained models.
    
    This function:
    1. Sets up random seeds for reproducibility
    2. Configures multiprocessing
    3. Sets up hyperparameters and model configuration
    4. Loads trained models for each fold
    5. Generates and saves embeddings for training and validation data
    """
    seed = 42
    # seed_everything(seed)
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
    num_epochs = 500
    num_epochs_finetuning = 10
    lr = 0.001 # (Krizhevsky et al.2012)
    lr_finetuning = 0.0001
    momentum = 0.9 # (Krizhevsky et al.2012)
    wd = 0.0005 # (Krizhevsky et al.2012)
    classes = 1
    # batch_size = 16
    batch_size = 1
    num_folds = 4
    min_delta = 0.001
    patience = 10
    min_epochs = num_epochs
    repeat_trainset_ntimes = 1

    patch_no = 'combined_SCV_large'
    # patch_no = 'combined_SCV'

    # dataset_type = 'pointDS'
    dataset_type = 'swDS'
    kernel_size = 224
    stride = 60
    # add_n_black_noise_img_to_train = 1500
    # add_n_black_noise_img_to_val = 500

    # validation_strategy = 'RCV'
    validation_strategy = 'SCV'  # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation

    architecture = 'VICRegConvNext'
    # architecture = 'VICReg'

    # strategy = 'domain-tuning'
    strategy = None

    augmentation = False
    tune_fc_only = True
    pretrained = False
    features = 'RGB'
    # 'GRN' | 'black' | 'RGB+'
    num_samples_per_fold = None

    fake_labels = False
    training_response_normalization = False
    criterion = nn.MSELoss(reduction='mean')

    # file_suffix = str(patch_no)+'_'
    file_suffix = ''
    # Detect if we have a GPU available

    # ResNet
    # this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICReg_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no) + '_resnet-back_4fields_s60'
    # this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICReg_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no)+ '_resnet-back_14fields'
    # this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICReg_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no)+ '_resnet-back_14fields_Dim1024'

    # ConvNext
    # this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICReg_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no)+ '_ConvNext-back_4fields_s60'
    # this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICReg_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no)+ '_ConvNext-back_14fields'
    # this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICReg_kernel_size_' + str(kernel_size) + '_DS-' + str(patch_no)+ '_ConvNext-back_14fields_Dim1024'
    this_output_dir = '/beegfs/stiller/PatchCROP_all/Output/2024_SSL/Results_Pub_Retrain/Combined_SCV_large_July_VICRegConvNext_SCV_E500lightly-VICReg_kernel_size_224_DS-combined_SCV_large_July_ConvNext-back_14fields_tuning'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sets = ['train', 'val',
            'test',
            ]
    # sets = ['val']

    patch_no = None
    patch_ids = [74, 90, 50, 68, 20, 110, '76_July', '95_July', '115_July', 65, 19, 89, 59, 119]
    # patch_ids = [ 68, 20, 110, 76, 95, 115, 65, 19, 89, 59, 119]

    global_preds = {'train': [], 'val': [], 'test': []}
    global_labels = {'train': [], 'val': [], 'test': []}
    global_patch_ids = {'train': [], 'val': [], 'test': []}
    global_crop_type_ids = {'train': [], 'val': [], 'test': []}
    global_crop_type_names_ids = {'train': [], 'val': [], 'test': []}
    combined_dataset = {'train': np.empty((0, 224, 224, 3)),
                        'val': np.empty((0, 224, 224, 3)),
                        'test': np.empty((0, 224, 224, 3)),
                        }
    for patch_no in patch_ids:

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

        checkpoint_paths = ['model_f0.ckpt',
                            'model_f1.ckpt',
                            'model_f2.ckpt',
                            'model_f3.ckpt',
                            ]
        SSL = False
        if 'SSL' in this_output_dir or 'SimCLR' in this_output_dir or 'VICReg' in this_output_dir or 'SimSiam' in this_output_dir or 'VICRegL' in this_output_dir or 'VICRegLConvNext' in this_output_dir:
            checkpoint_paths = ['model_f0_self-supervised.ckpt',
                                'model_f1_self-supervised.ckpt',
                                'model_f2_self-supervised.ckpt',
                                'model_f3_self-supervised.ckpt',
                                ]
            SSL = True
            strategy = 'self-supervised'

        checkpoint_paths = [this_output_dir+'/'+cp for cp in checkpoint_paths]

        # for s in sets:
        # for k in range(3,num_folds):
        for k in range(0,1):
            local_preds = None
            local_labels = None
            local_patch_ids = None
            local_crop_type_ids = None
            local_crop_type_names_ids = None


            # load data
            datamodule.setup_fold(fold=k, dataset_type=dataset_type)
            dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(),
                                'test': datamodule.test_dataloader(),
                                }

            # set up model wrapper and input data
            model_wrapper = RGBYieldRegressor_Trainer(
                pretrained=pretrained,
                tune_fc_only=tune_fc_only,
                architecture=architecture,
                criterion=criterion,
                device=device,
                workers=workers,
            )
            # set dataloaders
            model_wrapper.set_dataloaders(dataloaders=dataloaders_dict)
            # reinitialize FC layer for domain prediction, if SSL
            if SSL:
                if architecture == 'SimSiam' or architecture == 'VICRegL' or architecture == 'VICRegLConvNext' or architecture == 'VICReg' or architecture == 'VICRegConvNext':
                    model_wrapper.model.SSL_training = True
                    model_wrapper.reinitialize_fc_layers()
                    model_wrapper.disable_all_but_fc_grads()

            # load weights and skip rest of the method if already trained
            model_loaded = model_wrapper.load_model_if_exists(model_dir=this_output_dir, strategy=strategy, k=k)
            if not model_loaded: raise FileNotFoundError

            for s in sets:
                print('Generating embeddings:: Patch-ID:{}, Fold:{}, Set:{}'.format(patch_no,k,s))

                # make prediction
                # for each fold store labels and predictions
                local_preds, local_labels = model_wrapper.predict(phase=s, predict_embeddings=True)
                local_patch_ids = np.repeat(patch_no, len(local_labels)).tolist()
                # if s != 'train':
                #     local_crop_type_ids = np.tile(datamodule.crop_type, (len(local_labels),1)).tolist()
                #     local_crop_type_names_ids = np.repeat(datamodule.crop_type_name, len(local_labels)).tolist()
                # else:
                if patch_no in [50, 68, 20, 110, 74, 90]:
                    crop_type_name = 'maize'
                    crop_type = [1, 0, 0, 0, 0]
                elif patch_no in ['76_July', '95_July', '115_July']:
                    crop_type_name = 'sunflower'
                    crop_type = [0, 1, 0, 0, 0]
                elif patch_no in [65, 58, 19]:
                    crop_type_name = 'soy'
                    crop_type = [0, 0, 1, 0, 0]
                elif patch_no in [89, 59, 119]:
                    crop_type_name = 'lupine'
                    crop_type = [0, 0, 0, 1, 0]
                elif patch_no in [81, 73, 49, 39, 13, 21]:
                    crop_type_name = 'oats'
                    crop_type = [0, 0, 0, 0, 1]
                local_crop_type_ids = np.tile(crop_type, (len(local_labels), 1)).tolist()
                local_crop_type_names_ids = np.repeat(crop_type_name, len(local_labels)).tolist()
                print('Crop Type Name: {}'.format(crop_type_name))

                global_preds[s].extend(local_preds)
                global_labels[s].extend(local_labels)
                global_patch_ids[s].extend(local_patch_ids)
                global_crop_type_ids[s].extend(local_crop_type_ids)
                global_crop_type_names_ids[s].extend(local_crop_type_names_ids)

                if s == 'train':
                    combined_dataset['train'] = np.concatenate((combined_dataset['train'], datamodule.data_set[0][datamodule.train_idxs, :, :, :]), axis=0)
                elif s == 'val':
                    combined_dataset['val'] = np.concatenate((combined_dataset['val'], datamodule.data_set[0][datamodule.val_idxs, :, :, :]), axis=0)
                else:
                    combined_dataset['test'] = np.concatenate((combined_dataset['test'], datamodule.data_set[0][datamodule.test_idxs, :, :, :]), axis=0)

    for s in sets:
        # save labels and predictions for each fold
        torch.save(global_preds[s], os.path.join(this_output_dir, 'global_embeddings_' + file_suffix + s + '.pt'))
        torch.save(global_labels[s], os.path.join(this_output_dir, 'global_yield_' + file_suffix + s + '.pt'))
        torch.save(global_patch_ids[s], os.path.join(this_output_dir, 'global_patch_ids_' + file_suffix + s + '.pt'))
        torch.save(global_crop_type_ids[s], os.path.join(this_output_dir, 'global_crop_type_ids_' + file_suffix + s + '.pt'))
        torch.save(global_crop_type_names_ids[s], os.path.join(this_output_dir, 'global_crop_type_names_' + file_suffix + s + '.pt'))
        torch.save(combined_dataset[s], os.path.join(this_output_dir, 'global_imgs_' + file_suffix + s + '.pt'), pickle_protocol=4)

if __name__ == '__main__':
    print("Starting main function", flush=True)
    freeze_support()
    main()