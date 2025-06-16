# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script implements the data loading and preprocessing pipeline for UAV imagery and yield data.

The module supports two dataset types:
- swDS (sliding window dataset): Extracts fixed-size patches using a sliding window approach
- pointDS (point-centered dataset): Extracts patches centered on yield measurement points

For both dataset types, overlapping samples between train, validation, and test sets are removed.
Data splitting is managed through index files stored in the indices directory.

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports
import os
import warnings
import random
import math
from datetime import date

# Libs
import pandas as pd
from osgeo import gdal
import numpy as np
from typing import Optional
import geopandas as gpd
import glob

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset, TensorDataset
from torchvision.datasets import VisionDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from PIL import Image
# from memory_profiler import profile

from shapely.geometry import Polygon
import matplotlib.colors as mcolors

# Own modules
from TransformTensorDataset import TransformTensorDataset, CustomVisionDataset, LightyWrapper
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

# Add near the top of the file with other imports
import torch.multiprocessing as mp

class PatchCROPDataModule:
    """
    Data module for processing UAV imagery and yield data for crop yield prediction.
    
    This class handles:
    1. Loading and preprocessing of UAV imagery and yield data
    2. Data normalization and augmentation
    3. Generation of training samples using sliding window or point-centered approaches
    4. Spatial cross-validation for model evaluation
    5. Data loading for training, validation, and testing
    
    The module expects data in the following format:
    self.data_set = (
        feature_kernel_tensor,  # PIL Image or numpy array of shape (H, W, C)
        label_kernel_tensor,    # numpy array of yield values
        crop_types,            # numpy array of crop type annotations
        dates                  # numpy array of image capture dates
    )
    
    Attributes:
        train_dataset (Optional[Dataset]): Training dataset
        test_dataset (Optional[Dataset]): Test dataset
        train_fold (Optional[Dataset]): Current training fold
        val_fold (Optional[Dataset]): Current validation fold
        test_fold (Optional[Dataset]): Current test fold
    """
    
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None
    test_fold: Optional[Dataset] = None

    def __init__(self,
                 input_files: dict,
                 patch_id,
                 this_output_dir,
                 seed,
                 data_dir: str = './',
                 batch_size: int = 20,
                 stride: int = 30,
                 kernel_size:int = 224,
                 workers: int = 0,
                 input_features: str = 'RGB',
                 augmented: bool = False,
                 validation_strategy: str = 'SCV',
                 fake_labels: bool = False,
                 img_date: date = date(2020, 8, 6)
                 ):
        super().__init__()
        self.flowerstrip_patch_ids = [12, 13, 19, 20, 21, 105, 110, 114, 115, '115_July', 119]
        self.patch_ids = ['12', '13', '19', '20', '21', '39', '40', '49', '50', '51', '58', '59', '60', '65', '66', '68', '73', '74', '76', '76_July', '81', '89', '90', '95', '95_July', '96', '102', '105', '110', '114',
                          '115', '115_July', '119']

        self.data_dir = data_dir
        # name of the label file
        self.input_files = input_files
        self.batch_size = batch_size
        self.stride = stride
        self.kernel_size = kernel_size
        self.workers = workers
        self.augmented = augmented
        self.patch_id = patch_id
        self.input_features = input_features  # options: a) RGB - red, green, blue; b) GRN - green, red edge, near infrared; c) RGBRN- red, green, blue, red edge, near infrafred
        self.validation_strategy = validation_strategy
        self.fake_labels = fake_labels
        self.training_response_standardizer = None
        self.fold = None
        self.this_output_dir = this_output_dir
        self.seed = seed
        self.val_idxs = None
        self.train_idxs = None
        self.img_date = img_date
        self.mean_yield_per_crop, self.std_yield_per_crop = dict(), dict()
        self.channel_mean = []
        self.channel_std = []
        self.trainset_transforms = None
        self.testset_transforms = None
        # set crop type as one-hot encoding for [maize, sunflower, soy, lupin, oats]
        if self.img_date.year == 2020: #[74, 90, 50, 68, 20, 110, 76, 95, 115, 65, 19, 89, 59, 119]
            if patch_id in [50, 68, 20, 110, 74, 90]:
                self.crop_type_name = 'maize'
                self.crop_type = [1, 0, 0, 0, 0]
            elif patch_id in [76, 95, 115] or patch_id in ['76_July', '95_July', '115_July']:
                self.crop_type = 'sunflower'
                self.crop_type = [0, 1, 0, 0, 0]
            elif patch_id in [65, 58, 19]:
                self.crop_type = 'soy'
                self.crop_type = [0, 0, 1, 0, 0]
            elif patch_id in [89, 59, 119]:
                self.crop_type = 'lupine'
                self.crop_type = [0, 0, 0, 1, 0]
            elif patch_id in [81, 73, 49, 39, 13, 21]:
                self.crop_type = 'oats'
                self.crop_type = [0, 0, 0, 0, 1]
        else:
            raise NotImplementedError("Crop types other than for year 2020 are not implemented yet.")

        # Datesets
        self.data_set = None
        self.setup_done = False

        # Add this new attribute
        self.worker_init_fn = self._worker_init_fn

    def _get_field_bounds(self, fhandle=None):
        """
        Calculates the field bounds (origin and bottom-right corner) from a raster file.

        Parameters:
        - fhandle: file handle to the raster file

        Returns:
        - field_bounds: A tuple containing the coordinates of the origin (top-left corner)
                        and the bottom-right corner of the raster (field).
        """
        if fhandle is None:
            label, feature = next(iter(self.input_files.items()))
            # load and concat features

            print('reading features {}'.format(os.path.join(self.data_dir, feature[0])))
            fhandle = gdal.Open(os.path.join(self.data_dir, feature[0]), gdal.GA_ReadOnly)


        # Get the geotransform array
        geotransform = fhandle.GetGeoTransform()

        # Extract the origin (top-left corner)
        origin_x, origin_y = geotransform[0], geotransform[3]

        # Get raster dimensions
        raster_x_size = fhandle.RasterXSize
        raster_y_size = fhandle.RasterYSize

        # Calculate the pixel width and height
        pixel_width, pixel_height = geotransform[1], geotransform[5]

        # Calculate the coordinates of the bottom-right corner
        bottom_right_x = origin_x + (pixel_width * raster_x_size)
        bottom_right_y = origin_y + (pixel_height * raster_y_size)

        # Define field bounds as a tuple
        field_bounds = (origin_x, origin_y, bottom_right_x, bottom_right_y)
        resolution = (geotransform[1], abs(geotransform[5]))

        f = torch.tensor(fhandle.ReadAsArray()[:3, :, :], dtype=torch.float)
        # Reorder the tensor to H,W,C
        f = np.transpose(f, (1, 2, 0)).numpy().astype(np.uint8)
        # feature_matrix = feature_matrix.movedim(1, 3).numpy()
        # Normalize or scale the tensor values to [0, 255] if not already
        if f.max() <= 1.0:
            feature_matrix = (f * 255).astype(np.uint8)

        return field_bounds, resolution, f.shape, f

    def prepare_data(self, num_samples: int = None, dataset_type='swDS') -> None:  # would actually make more sense to put it to setup, but there it is called as several processes
        # load the data
        # normalize
        # split
        # oversample
        # load feature label combinations

        if self.fake_labels:
            kfold_dir = os.path.join(self.data_dir, 'kfold_set_fakelabels_s{}_ks{}_{}'.format(self.stride, self.kernel_size, str(self.img_date)))
        else:
            if self.input_features == "black":
                kfold_dir = os.path.join(self.data_dir, 'kfold_set_origlabels_s{}_ks{}_black_{}'.format(self.stride, self.kernel_size, str(self.img_date)))
            else:
                kfold_dir = os.path.join(self.data_dir, 'kfold_set_origlabels_s{}_ks{}_{}'.format(self.stride, self.kernel_size, str(self.img_date)))

        dataset_type = '' if dataset_type == 'swDS' else '_' + dataset_type
        # print('trying to load data in {}'.format(kfold_dir))
        if not self.setup_done:
            loading_patch_id = self.patch_id if 'July' not in str(self.patch_id) else self.patch_id.split('_July')[0]
            print('trying to load dataset'+'Patch_ID_' + str(loading_patch_id) + dataset_type + '.pt')
            if self.input_features == "RGB" or self.input_features == "black":
                file_name = 'Patch_ID_' + str(loading_patch_id) + dataset_type + '.pt'
            elif self.input_features == "GRN":
                file_name = 'Patch_ID_' + str(loading_patch_id) + '_grn' + dataset_type + '.pt'
            elif self.input_features == "RGBRN":
                file_name = 'Patch_ID_' + str(loading_patch_id) + '_RGBRN' + dataset_type + '.pt'
            
            print('loading {} in {}'.format(kfold_dir, file_name))
            # check if already build and load
            if os.path.isfile(os.path.join(kfold_dir, file_name)):
                try:
                    f = torch.load(os.path.join(kfold_dir, file_name))
                except FileNotFoundError:
                    print("Dataset type {} not implemented, try 'swDS' or 'pointDS'.")
                self.data_set = f
            else:
                # otherwise build oversampled k-fold set
                print('no data found, start generating data')
                for label, features in self.input_files.items():
                    label_matrix = torch.tensor([])
                    feature_matrix = torch.tensor([])
                    # load label
                    print('reading labels {}'.format(os.path.join(self.data_dir, label)))
                    label_matrix = torch.tensor(gdal.Open(os.path.join(self.data_dir, label), gdal.GA_ReadOnly).ReadAsArray(), dtype=torch.float)
                    # label_matrix = gdal.Open(os.path.join(self.data_dir, label), gdal.GA_ReadOnly).ReadAsArray()
                    # load and concat features
                    for idx, feature in enumerate(features):
                        # if RGB drop alpha channel, else all
                        print('reading features {}'.format(os.path.join(self.data_dir, feature)))
                        fhandle = gdal.Open(os.path.join(self.data_dir, feature), gdal.GA_ReadOnly)

                        # The field bounds can be represented as a tuple of the origin and bottom-right corner
                        field_bounds, resolution, _, _ = self._get_field_bounds(fhandle=fhandle)

                        if 'soda' in feature or 'Soda' in feature:  # RGB
                            print(feature)
                            f = torch.tensor(fhandle.ReadAsArray()[:3, :, :], dtype=torch.float)  # only first 3 channels -> not alpha channel
                        else:  # multichannel
                            f = torch.tensor(fhandle.ReadAsArray(), dtype=torch.float)
                        # concat feature tensor, e.g. RGB tensor and NDVI tensor
                        if f.dim() == 2:
                            feature_matrix = torch.cat((feature_matrix, f.unsqueeze(0)), 0)
                        else:
                            feature_matrix = torch.cat((feature_matrix, f), 0)

                    # create point dataset
                    # creates self.point_data_set
                    if dataset_type == '_pointDS':
                        print('\tgenerating point dataset')
                        self.generate_point_dataset(feature_matrix=feature_matrix, field_bounds=field_bounds, resolution=resolution)
                    elif dataset_type == '_swDS' or dataset_type == '':
                        # sliding window  -> n samples of 224x224 images
                        # creates self.data_set
                        print('\tgenerating sliding window dataset')
                        self.generate_sliding_window_augmented_ds(label_matrix, feature_matrix)

                # save data_set
                if not os.path.isdir(kfold_dir):
                    os.mkdir(kfold_dir)
                file_name = kfold_dir + '/' + self.data_dir.split('/')[-1] + dataset_type + '.pt'
                torch.save(self.data_set, file_name)
            self.setup_done = True
        # # drop all negatives
        # Assuming self.data_set is a tuple (feature_kernel_tensor, label_kernel_tensor, crop_types, dates)
        feature_kernel_tensor, label_kernel_tensor, crop_types, dates = self.data_set

        # Create a mask based on the condition
        mask = label_kernel_tensor >= 0

        # Apply the mask to each element
        filtered_feature_kernel_tensor = feature_kernel_tensor[mask]
        filtered_label_kernel_tensor = label_kernel_tensor[mask]
        filtered_crop_types = crop_types[mask]
        filtered_dates = dates[mask]

        if torch.is_tensor(filtered_crop_types):
            filtered_crop_types = filtered_crop_types.numpy()

        self.mask = mask
        
        # Recreate the data_set tuple with the filtered data
        self.data_set = (filtered_feature_kernel_tensor, filtered_label_kernel_tensor, filtered_crop_types, filtered_dates)

    def generate_sliding_window_augmented_ds(self, label_matrix, feature_matrix, lower_bound: int = 327) -> None:
        '''
        :param label_matrix: kriging interpolated yield maps
        :param feature_matrix: stack of feature maps, such as rgb remote sensing, multi-channel remote sensing or dsm or ndvi
        both are required to have same dimension, but have different size according to whether the patch is a flower strip
        or not
        :return: None
        generate sliding window samples over whole patch and save as self.data_set
        '''
        # length= 2322/1935

        upper_bound_x = feature_matrix.shape[2] - lower_bound
        if self.patch_id in self.flowerstrip_patch_ids:
            lower_bound = 273
            upper_bound_y = feature_matrix.shape[1] - lower_bound + 1
        else:
            upper_bound_y = feature_matrix.shape[1] - lower_bound

        # clip to tightest inner rectangle
        feature_arr = feature_matrix[:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]

        # scale to 0...1
        if self.input_features == 'RGB':
            # feature_arr = feature_arr / 255 # --> deprecated, normalization is now done in setup_fold()->transformer
            pass
        elif self.input_features == 'black':
            # blacken one quarter of the image
            mask = torch.zeros(3, 2323, 2323)
            quarter_size = 2323 // 2  # Assuming you want to blacken the top-left quarter
            mask[:, :quarter_size, :quarter_size] = 1
            # Generate random values close to zero to multiply with the original tensor
            random_values = torch.rand(3, 2323, 2323) * 0.1

            # Apply the mask to the original tensor
            feature_arr = feature_arr * (1 - mask) + (feature_arr * mask * random_values).round()

            # LABEL MISSING!!!!!
            warnings.warn("Labels not zeroed! -> only use for SSL training!")

        elif self.input_features == 'GRN':
            c1_min = feature_arr[0, :, :].min()
            c2_min = feature_arr[1, :, :].min()
            c3_min = feature_arr[2, :, :].min()
            c1_max = feature_arr[0, :, :].max()
            c2_max = feature_arr[1, :, :].max()
            c3_max = feature_arr[2, :, :].max()

            feature_arr[0, :, :] = ((feature_arr[0, :,
                                     :] - c1_min) / c1_max) * 255  # normalization to [0..1] *255 to first get it to RGB range, second use normalization is now done in setup_fold()->transformer
            feature_arr[1, :, :] = ((feature_arr[1, :, :] - c2_min) / c2_max) * 255
            feature_arr[2, :, :] = ((feature_arr[2, :, :] - c3_min) / c3_max) * 255
        elif self.input_features == 'RGBRN':
            c4_min = feature_arr[3, :, :].min()
            c5_min = feature_arr[4, :, :].min()
            c4_max = feature_arr[3, :, :].max()
            c5_max = feature_arr[4, :, :].max()

            feature_arr[3, :, :] = ((feature_arr[3, :,
                                     :] - c4_min) / c4_max) * 255  # normalization to [0..1] *255 to first get it to RGB range, second use normalization is now done in setup_fold()->transformer
            feature_arr[4, :, :] = ((feature_arr[4, :, :] - c5_min) / c5_max) * 255

        if self.fake_labels:
            # three channel pixel-wise average -> but this is still in range [0..255]
            label_arr = (feature_arr[0, :, :] + feature_arr[1, :, :] + feature_arr[2, :, :]) / 3
            # normalize to range [0..255]
            label_arr = label_arr / 255
        else:
            label_arr = label_matrix[lower_bound:upper_bound_y, lower_bound:upper_bound_x]  # label matrix un-normalized
            if self.input_features == 'black':
                # blacken one quarter of the image
                mask = torch.zeros(2323, 2323)
                quarter_size = 2323 // 2  # Assuming you want to blacken the top-left quarter
                mask[:quarter_size, :quarter_size] = 1
                # Generate random values close to zero to multiply with the original tensor
                random_values = torch.rand(2323, 2323) * 0.1

                # Apply the mask to the original tensor
                label_arr = label_arr * (1 - mask)+ label_arr * mask * random_values

        # set sizes
        self.data_set_row_extend = feature_arr.shape[2]
        self.data_set_column_extend = feature_arr.shape[1]

        # calculating start positions of kernels
        # x and y are need to able to take different values, as patches with flowerstrips are smaller
        x_size = feature_arr.shape[2]
        y_size = feature_arr.shape[1]
        possible_shifts_x = int((x_size - self.kernel_size) / self.stride)
        x_starts = np.array(list(range(possible_shifts_x))) * self.stride
        possible_shifts_y = int((y_size - self.kernel_size) / self.stride)
        y_starts = np.array(list(range(possible_shifts_y))) * self.stride

        # loop over start postitions and save kernel as separate image
        feature_kernel_tensor = None
        label_kernel_tensor = None
        for y in y_starts:
            print('y: {}'.format(y))
            if y == y_starts[int(len(y_starts) / 4)]:
                print("........... 25%")
            if y == y_starts[int(len(y_starts) / 2)]:
                print("........... 50%")
            if y == y_starts[int(len(y_starts) * 3 / 4)]:
                print("........... 75%")
            for x in x_starts:
                # shift kernel over image and extract kernel part
                # only take RGB value
                feature_kernel_img = feature_arr[:, y:y + self.kernel_size, x:x + self.kernel_size]
                label_kernel_img = label_arr[y:y + self.kernel_size, x:x + self.kernel_size]
                if x == 0 and y == 0:
                    feature_kernel_tensor = feature_kernel_img.unsqueeze(0)
                    label_kernel_tensor = label_kernel_img.mean().unsqueeze(0)
                else:
                    feature_kernel_tensor = torch.cat((feature_kernel_tensor, feature_kernel_img.unsqueeze(0)), 0)
                    label_kernel_tensor = torch.cat((label_kernel_tensor, label_kernel_img.mean().unsqueeze(0)), 0)
        # create tensor with n dates, and one with n crop_types
        n = len(label_kernel_tensor)
        dates = torch.tensor([self.img_date.year, self.img_date.month, self.img_date.day]).repeat(n, 1)
        crop_types = torch.tensor(self.crop_type).repeat(n, 1).numpy()

        # self.data_set = (feature_kernel_tensor, label_kernel_tensor, crop_types, dates)
        self.data_set = (feature_kernel_tensor.movedim(1, 3).numpy(), label_kernel_tensor.numpy(), crop_types, dates.numpy())

    def _compute_mean_std(self):
        transforms = A.Compose([  # assumption by albumentations: image is in HWC
            # self.normalize,
            ToTensorV2(),  # convert to CHW
        ])
        if self.patch_id == 'combined_SCV' or self.patch_id == 'combined_SCV_large' :
            for s in ['train', 'val']:
                if s == 'train':
                    data_set = TransformTensorDataset((
                                self.data_set[0][self.train_idxs, :, :, :],
                                self.data_set[1][self.train_idxs],
                                self.data_set[2][self.train_idxs],
                                self.data_set[3][self.train_idxs]
                            ),transform=transforms)
                else:
                    data_set = TransformTensorDataset((
                        self.data_set[0][self.val_idxs, :, :, :],
                        self.data_set[1][self.val_idxs],
                        self.data_set[2][self.val_idxs],
                        self.data_set[3][self.val_idxs]
                    ), transform=transforms)
                # Iterate over the entire dataset to compute mean and std
                all_yields = []
                all_crop_types = []

                num_channels = 3
                num_pixels = 0
                sum_channels = torch.zeros(num_channels)
                sum_squares_channels = torch.zeros(num_channels)

                for i in range(len(data_set)):
                    x0, labels, crop_types, _ = data_set[i]
                    all_yields.append(labels)
                    all_crop_types.append(crop_types)

                    num_pixels += x0.numel() / num_channels  # Total number of pixels per channel

                    # Sum of pixel values per channel
                    sum_channels += x0.sum(dim=[1, 2])

                    # Sum of squared pixel values per channel
                    sum_squares_channels += (x0 ** 2).sum(dim=[1, 2])

                # all_yields = np.concatenate(all_yields)
                # all_crop_types = np.concatenate(all_crop_types)
                all_yields = np.array(all_yields)
                all_crop_types = np.array(all_crop_types)

                # Compute mean for each channel
                self.channel_mean = sum_channels / num_pixels

                # Compute std for each channel
                self.channel_std = torch.sqrt(sum_squares_channels / num_pixels - self.channel_mean ** 2)

                # Calculate mean and std for each crop type
                mean_yield_per_crop = []
                std_yield_per_crop = []

                for crop_type_index in range(all_crop_types.shape[1]):
                    crop_type_mask = all_crop_types[:, crop_type_index].astype(bool) # that is the wrong dimension from the crop type mask!!! or?
                    yield_per_crop = all_yields[crop_type_mask]
                    if any(crop_type_mask):
                        mean_yield_per_crop.append(np.mean(yield_per_crop))
                        std_yield_per_crop.append(np.std(yield_per_crop))
                    else:
                        mean_yield_per_crop.append(0.)
                        std_yield_per_crop.append(1.)

                self.mean_yield_per_crop[s], self.std_yield_per_crop[s] = mean_yield_per_crop, std_yield_per_crop


    # def setup_fold(self, fold: int = 0, testset_transforms=None, trainset_transforms=None, dataset_type='swDS') -> None:
    def setup_fold(self, fold: int = 0, dataset_type='swDS', indices_dir=None) -> None:
        # print('\tSetting up fold specific datasets...')
        # self.set_transforms(testset_transforms, trainset_transforms)
        self.fold = fold
        if dataset_type == 'swDS':
            self.setup_fold_swDS(fold=fold, indices_dir=indices_dir)
        elif dataset_type == 'pointDS':
            self.setup_fold_point_ds(fold=fold, indices_dir=indices_dir)
        else: raise NotImplementedError("Setup fold not implemented for {}".format(dataset_type))

        # Adjust indices after filtering out negative labels
        if hasattr(self, 'mask'):
            # Create a mapping from old indices to new indices
            cumsum = np.cumsum(self.mask) - 1  # -1 to get 0-based index
            old_to_new = {i: cumsum[i] for i in range(len(self.mask)) if self.mask[i]}
            
            # Update train indices
            self.train_idxs = np.array([old_to_new[idx] for idx in self.train_idxs if idx in old_to_new])
            
            # Update validation indices
            self.val_idxs = np.array([old_to_new[idx] for idx in self.val_idxs if idx in old_to_new])
            
            # Update test indices if they exist
            if hasattr(self, 'test_idxs'):
                self.test_idxs = np.array([old_to_new[idx] for idx in self.test_idxs if idx in old_to_new])

        self._compute_mean_std()

    def set_transforms(self, trainset_transforms, testset_transforms=None):
        self.trainset_transforms = trainset_transforms
        self.testset_transforms = testset_transforms

    def setup_fold_swDS(self, fold: int = 0, indices_dir=None) -> None:
        '''
        Set up validation and train data set using self.data_set and apply normalization for both and augmentations for the train set if
        self.augmented. Overlap samples between train and test set are discarded.

        :return: None
        '''
        if not self.load_subset_indices(k=fold, indices_dir=indices_dir):
            # Splitting into datasets according to validation strategy
            # Part 1) Compute subset indexes
            if self.validation_strategy == 'SCV' or self.validation_strategy == 'SCV_no_test':  # spatial cross validation with or without test set
                print('splitting for SCV')
                # ranges for x and y
                # blocks:
                # 0 1
                # 2 3
                # test val train   | fold
                #  0    1   {2,3}  |  0
                #  1    3   {0,2}  |  1
                #  3    2   {0,1}  |  2
                #  2    0   {1,3}  |  3

                # sliding window size
                window_size = self.kernel_size
                stride = self.stride
                # length= 2322/1935
                x_size = y_size = 2322

                # number of sliding window samples in any given row
                samples_per_row = samples_per_col = math.floor((x_size - window_size) / stride)

                x_def = y_def = np.arange(samples_per_row)
                # center lines to split the patch into folds at
                center_x = center_y = x_size / 2
                # last row/col index such that samples do not overlap between folds in quadrant 0/2
                buffer_to_x = buffer_to_y = math.floor((math.floor(center_x) - window_size) / stride)
                # first row/col index such that samples do not overlap between folds in quadrant 1/3
                buffer_from_x = buffer_from_y = math.ceil(math.floor(center_x) / stride)
                # last row/col index such that samples CAN OVERLAP between folds in quadrant 0/2
                overlap_to_x = overlap_to_y = math.floor((math.floor(center_x)) / stride)
                # first row/col index such that samples CAN OVERLAP between folds in quadrant 1/3
                overlap_from_x = overlap_from_y = math.ceil(math.floor(center_x) / stride)
                if self.patch_id in self.flowerstrip_patch_ids:
                    y_size = 1935
                    center_y = y_size / 2
                    samples_per_col = math.floor((y_size - window_size) / stride)
                    y_def = np.arange(samples_per_col)
                    buffer_to_y = math.floor((math.floor(center_y) - window_size) / stride)
                    buffer_from_y = math.ceil(math.floor(center_y) / stride)
                    overlap_to_y = math.floor((math.floor(center_y)) / stride)
                    # buffer_from_y = math.ceil(math.floor(center_y) / stride)
                    overlap_from_y = math.ceil(math.floor(center_y) / stride)

                debug_test_map = [np.zeros((1, y_size, x_size)),
                                  np.zeros((1, y_size, x_size)),
                                  np.zeros((1, y_size, x_size)),
                                  np.zeros((1, y_size, x_size)),
                                  ]
                debug_val_map = [np.zeros((1, y_size, x_size)),
                                 np.zeros((1, y_size, x_size)),
                                 np.zeros((1, y_size, x_size)),
                                 np.zeros((1, y_size, x_size)),
                                 ]
                debug_train_map = [np.zeros((1, y_size, x_size)),
                                   np.zeros((1, y_size, x_size)),
                                   np.zeros((1, y_size, x_size)),
                                   np.zeros((1, y_size, x_size)),
                                   ]
                # no overlap quadrants
                quadrant_0 = [(x, y) for x in np.arange(buffer_to_x) for y in np.arange(buffer_to_y)]
                quadrant_1 = [(x, y) for x in np.arange(buffer_from_x, samples_per_row) for y in np.arange(buffer_to_y)]
                quadrant_2 = [(x, y) for x in np.arange(buffer_to_x) for y in np.arange(buffer_from_y, samples_per_col)]
                quadrant_3 = [(x, y) for x in np.arange(buffer_from_x, samples_per_row) for y in np.arange(buffer_from_y, samples_per_col)]

                # overlap quadrants
                overlap_quadrant_0 = [(x, y) for x in np.arange(overlap_to_x) for y in np.arange(overlap_to_y)]
                overlap_quadrant_1 = [(x, y) for x in np.arange(overlap_from_x, samples_per_row) for y in np.arange(overlap_to_y)]
                overlap_quadrant_2 = [(x, y) for x in np.arange(overlap_to_x) for y in np.arange(overlap_from_y, samples_per_col)]
                overlap_quadrant_3 = [(x, y) for x in np.arange(overlap_from_x, samples_per_row) for y in np.arange(overlap_from_y, samples_per_col)]

                half_23 = [(x, y) for x in np.arange(samples_per_row) for y in np.arange(buffer_from_y, samples_per_col)]
                half_02 = [(x, y) for x in np.arange(buffer_to_x) for y in np.arange(samples_per_col)]
                half_01 = [(x, y) for x in np.arange(samples_per_row) for y in np.arange(buffer_to_y)]
                half_13 = [(x, y) for x in np.arange(buffer_from_x, samples_per_row) for y in np.arange(samples_per_col)]

                if self.validation_strategy == 'SCV':  # spatial cross validation with test set
                    test_range = [quadrant_0,  # quadrant 0
                                  quadrant_1,  # quadrant 1
                                  quadrant_3,  # quadrant 3
                                  quadrant_2,  # quadrant 2
                                  ]
                    val_range = [quadrant_1,  # quadrant 1
                                 quadrant_3,  # quadrant 3
                                 quadrant_2,  # quadrant 2
                                 quadrant_0,  # quadrant 0
                                 ]
                    train_range = [half_23,
                                   half_02,
                                   half_01,
                                   half_13,
                                   ]

                    test_sw_idxs = []
                    val_sw_idxs = []
                    train_sw_idxs = []

                    for y in y_def:
                        for x in x_def:
                            idx = (x + y * samples_per_row)
                            if (x, y) in test_range[fold]:
                                test_sw_idxs.append(idx)
                                debug_test_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                            elif (x, y) in val_range[fold]:
                                val_sw_idxs.append(idx)
                                debug_val_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                            elif (x, y) in train_range[fold]:
                                train_sw_idxs.append(idx)
                                debug_train_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                    self.save_subset_indices(k=fold, train_indices=train_sw_idxs, val_indices=val_sw_idxs, test_indices=test_sw_idxs)
                elif self.validation_strategy == 'SCV_no_test':  # spatial cross validation no test set
                    threequarter_230 = list(set(half_02 + half_23))
                    threequarter_021 = list(set(half_02 + half_01))
                    threequarter_013 = list(set(half_01 + half_13))
                    threequarter_132 = list(set(half_13 + half_23))

                    val_range = [quadrant_1,  # quadrant 1
                                 quadrant_3,  # quadrant 3
                                 quadrant_2,  # quadrant 2
                                 quadrant_0,  # quadrant 0
                                 ]
                    train_range = [threequarter_230,
                                   threequarter_021,
                                   threequarter_013,
                                   threequarter_132,
                                   ]

                    test_sw_idxs = []
                    val_sw_idxs = []
                    train_sw_idxs = []

                    for y in y_def:
                        for x in x_def:
                            idx = (x + y * samples_per_row)
                            if (x, y) in val_range[fold]:
                                val_sw_idxs.append(idx)
                                debug_val_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                            elif (x, y) in train_range[fold]:
                                train_sw_idxs.append(idx)
                                debug_train_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                # debug images to verify correct splitting and amount sample
                # create folders
                print('creating debug fold images')
                dir = os.path.join(self.data_dir, 'Test_Figs')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                # val
                m = np.max(debug_val_map[fold])
                debug_val_map[fold] = debug_val_map[fold]  # /m
                fig, ax = plt.subplots()
                c = plt.imshow(debug_val_map[fold][0, :, :], cmap='jet')
                fig.colorbar(c, ax=ax)
                ax.xaxis.tick_top()
                plt.tight_layout()
                plt.savefig(os.path.join(dir, 'val_folds_{}_{}_{}.png'.format(fold, stride, self.validation_strategy)))
                # train
                m = np.max(debug_train_map[fold])
                debug_train_map[fold] = debug_train_map[fold]  # / m
                fig, ax = plt.subplots()
                c = plt.imshow(debug_train_map[fold][0, :, :], cmap='jet')
                fig.colorbar(c, ax=ax)
                ax.xaxis.tick_top()
                plt.tight_layout()
                plt.savefig(os.path.join(dir, 'train_folds_{}_{}_{}.png'.format(fold, stride, self.validation_strategy)))
                # test
                if self.validation_strategy == 'SCV':
                    m = np.max(debug_test_map[fold])
                    debug_test_map[fold] = debug_test_map[fold]  # / m
                    fig, ax = plt.subplots()
                    c = plt.imshow(debug_test_map[fold][0, :, :], cmap='jet')
                    fig.colorbar(c, ax=ax)
                    ax.xaxis.tick_top()
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, 'test_folds_{}_{}_{}.png'.format(fold, stride, self.validation_strategy)))
                if self.validation_strategy == 'SCV':
                    self.save_subset_indices(k=fold, train_indices=train_sw_idxs, val_indices=val_sw_idxs, test_indices=test_sw_idxs)
                else:
                    self.save_subset_indices(k=fold, train_indices=train_sw_idxs, val_indices=val_sw_idxs)
            elif self.validation_strategy == 'SHOV':  # spatial hold out validation
                # train_idxs = np.arange(0,61)
                # val_idxs = np.arange(61, 81)
                # train_idxs = np.arange(0, 234)
                # val_idxs = np.arange(252, 324)
                # train_idxs = np.arange(0, 989)
                # val_idxs = np.arange(1027, 1369)
                # stride 30
                train_sw_idxs = np.arange(0, 3294)
                val_sw_idxs = np.arange(3847, 4761)
                self.save_subset_indices(k=fold, train_indices=train_sw_idxs, val_indices=val_sw_idxs)
            elif self.validation_strategy == 'RCV':  # random CV
                kf = KFold(n_splits=4, random_state=self.seed, shuffle=True)
                idxs = np.arange(len(self.data_set[0]))
                k = 0
                for train_sw_idxs_iter , val_sw_idxs_iter in kf.split(X=idxs):
                    self.save_subset_indices(k=k, train_indices=train_sw_idxs_iter, val_indices=val_sw_iter)
                    if k == fold:
                        val_sw_idxs = val_sw_iter
                        train_sw_idxs = train_sw_iter
                    k += 1
            # set as global parameter
            self.val_idxs = val_sw_idxs
            self.train_idxs = train_sw_idxs
            if self.validation_strategy == 'SCV':
                self.test_idxs = test_sw_idxs

        print('\tIndices for splitting in train/val/test built ...')

    def load_subset_indices(self, k, indices_dir=None, dataset_type='swDS'):
        stride_token = ''
        if self.stride != 30:
            stride_token = '_s'+str(self.stride)
        dataset_type = '' if dataset_type == 'swDS' else '_' + dataset_type

        # Create path to centralized indices directory - going up one level 
        if indices_dir is None:
            indices_dir = os.path.join(os.path.dirname(self.this_output_dir), 'Indices', str(self.patch_id))
        
        val_filename = os.path.join(indices_dir, 'val_df_' + str(k) + self.validation_strategy + stride_token + dataset_type + '_pID' + str(self.patch_id) + '.csv')
        print('loading subset indices e.g. {}'.format(val_filename))
        train_filename = os.path.join(indices_dir, 'train_df_' + str(k) + self.validation_strategy + stride_token + dataset_type + '_pID' + str(self.patch_id) + '.csv')
        
        if os.path.exists(val_filename):
            self.val_idxs = pd.read_csv(val_filename, usecols=['val_indices'])['val_indices'].values
            self.train_idxs = pd.read_csv(train_filename, usecols=['train_indices'])['train_indices'].values
            if self.validation_strategy == 'SCV':
                test_filename = os.path.join(indices_dir, 'test_df_' + str(k) + self.validation_strategy + stride_token + dataset_type + '_pID' + str(self.patch_id) + '.csv')
                self.test_idxs = pd.read_csv(test_filename, usecols=['test_indices'])['test_indices'].values
            return True
        return False

    def save_subset_indices(self, k, train_indices, val_indices, test_indices=None, dataset_type='swDS'):
        stride_token = ''
        if self.stride != 30:
            stride_token = '_s' + str(self.stride)
        dataset_type = '' if dataset_type == 'swDS' else '_'+dataset_type

        # Create path to centralized indices directory - going up one level
        indices_dir = os.path.join(os.path.dirname(self.this_output_dir), 'Indices', str(self.patch_id))
        
        # Create directories if they don't exist
        os.makedirs(indices_dir, exist_ok=True)

        train_df = pd.DataFrame({'train_indices': train_indices})
        val_df = pd.DataFrame({'val_indices': val_indices})
        
        train_df.to_csv(os.path.join(indices_dir, 'train_df_' + str(k) + self.validation_strategy + stride_token + dataset_type + '_pID' + str(self.patch_id) + '.csv'), encoding='utf-8')
        val_df.to_csv(os.path.join(indices_dir, 'val_df_' + str(k) + self.validation_strategy + stride_token + dataset_type + '_pID' + str(self.patch_id) + '.csv'), encoding='utf-8')
        
        if self.validation_strategy == 'SCV':
            test_df = pd.DataFrame({'test_indices': test_indices})
            test_df.to_csv(os.path.join(indices_dir, 'test_df_' + str(k) + self.validation_strategy + stride_token + dataset_type + '_pID' + str(self.patch_id) + '.csv'), encoding='utf-8')

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def custom_collate(self, batch):
        """
        Optimized collate function that handles both standard and Lightly datasets.
        """
        try:
            # Convert and stack data
            data = torch.stack([
                item[0] if isinstance(item[0], torch.Tensor)
                else torch.from_numpy(item[0]) 
                for item in batch
            ])
            
            # Handle scalar labels specially
            labels = torch.tensor([
                item[1].item() if isinstance(item[1], torch.Tensor)
                else float(item[1])
                for item in batch
            ], dtype=torch.float32)
            
            # Convert and stack crop types and dates
            crop_types = torch.stack([
                item[2] if isinstance(item[2], torch.Tensor)
                else torch.from_numpy(item[2])
                for item in batch
            ])
            
            dates = torch.stack([
                item[3] if isinstance(item[3], torch.Tensor)
                else torch.from_numpy(item[3])
                for item in batch
            ])
            
            return data, labels, crop_types, dates
        except Exception as e:
            print(f"Collate error with batch shapes: {[type(item[1]) for item in batch]}")
            raise e

    def worker_init_fn(self, worker_id):
        """
        Minimal worker initialization with essential setup.
        """
        # Set worker-specific random seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        if torch.cuda.is_available():
            # Simple GPU assignment without extra overhead
            torch.cuda.set_device(worker_id % torch.cuda.device_count())

    def train_dataloader(self, shuffle:bool=True) -> DataLoader:
        # print(f"train_idxs type: {type(self.train_idxs)}, contents: {self.train_idxs}")
        print('loading training set with {} samples'.format(len(self.data_set[1][self.train_idxs])))
        drop_last = False if len(self.train_idxs) % self.batch_size == 0 else True
        
        # CUDA setup
        cuda_kwargs = {}
        if torch.cuda.is_available():
            cuda_kwargs = {
                'pin_memory': True,
                'pin_memory_device': 'cuda'
            }
            stream = torch.cuda.Stream()
        
        if self.trainset_transforms == None:
            if self.augmented:
                trainset_transforms = A.Compose([
                    A.RandomBrightnessContrast(),
                    A.ColorJitter(brightness=(0.8, 1), contrast=(0.8, 1), saturation=(0.8, 1), hue=(-0.5, 0.5), p=0.2),
                    A.RandomRotate90(),
                    A.VerticalFlip(),
                    A.HorizontalFlip(),
                    A.Transpose(),
                    ToTensorV2(),
                ])
            else:
                trainset_transforms = A.Compose([
                    ToTensorV2(),
                ])
                
            dataset = TransformTensorDataset(
                (
                    self.data_set[0][self.train_idxs],
                    self.data_set[1][self.train_idxs],
                    self.data_set[2][self.train_idxs],
                    self.data_set[3][self.train_idxs]
                ),
                transform=trainset_transforms
            )
        else:
            # Lightly dataset handling
            dataset = LightyWrapper(
                LightlyDataset.from_torch_dataset(
                    CustomVisionDataset(
                        (
                            self.data_set[0][self.train_idxs],
                            self.data_set[1][self.train_idxs]
                        )
                    ),
                    transform=self.trainset_transforms
                ),
                self.data_set[2][self.train_idxs],
                self.data_set[3][self.train_idxs]
            )
        
        # Optimize DataLoader settings
        base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(
            base_sampler,
            batch_size=self.batch_size,
            drop_last=drop_last
        )
        
        loader_kwargs = {
            'batch_sampler': batch_sampler,
            'num_workers': min(4, os.cpu_count() - 1),  # Limit workers to reduce contention
            'persistent_workers': True,
            'prefetch_factor': 2,
            'collate_fn': self.custom_collate,
            'worker_init_fn': self.worker_init_fn,
        }
        
        if torch.cuda.is_available():
            loader_kwargs.update({
                'pin_memory': True,
                'pin_memory_device': 'cuda'
            })
        
        return DataLoader(dataset, **loader_kwargs)

    def val_dataloader(self, shuffle:bool=True) -> DataLoader:
        print('loading val set with {} samples'.format(len(self.data_set[1][self.val_idxs])))
        drop_last = False
        
        # CUDA setup
        cuda_kwargs = {}
        if torch.cuda.is_available():
            cuda_kwargs = {
                'pin_memory': True,
                'pin_memory_device': 'cuda'
            }
            stream = torch.cuda.Stream()
        
        if self.trainset_transforms is None:
            testset_transforms = self.testset_transforms
            if self.testset_transforms is None:
                testset_transforms = A.Compose([
                    ToTensorV2(),
                ])
                
            dataset = TransformTensorDataset(
                (
                    self.data_set[0][self.val_idxs],
                    self.data_set[1][self.val_idxs],
                    self.data_set[2][self.val_idxs],
                    self.data_set[3][self.val_idxs],
                ),
                transform=testset_transforms
            )
            
            # For validation, typically use SequentialSampler
            base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(
                base_sampler,
                batch_size=self.batch_size,
                drop_last=drop_last
            )
            
            loader_kwargs = {
                'batch_sampler': batch_sampler,
                'num_workers': min(self.workers, os.cpu_count() - 1),  # Leave one CPU free
                'persistent_workers': True,
                'prefetch_factor': 3,  # Increased from 2
                'collate_fn': self.custom_collate,
                'worker_init_fn': self.worker_init_fn,
            }
            
            if torch.cuda.is_available():
                loader_kwargs.update({
                    'pin_memory': True,
                    'pin_memory_device': 'cuda'
                })
            
            return DataLoader(dataset, **loader_kwargs)
        else:
            dataset = LightyWrapper(
                LightlyDataset.from_torch_dataset(
                    CustomVisionDataset(
                        (
                            self.data_set[0][self.val_idxs],
                            self.data_set[1][self.val_idxs],
                        ),
                    ),
                    transform=self.trainset_transforms
                ),
                self.data_set[2][self.val_idxs],
                self.data_set[3][self.val_idxs],
            )
            
            # Create sampler based on shuffle parameter
            base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(
                base_sampler,
                batch_size=self.batch_size,
                drop_last=drop_last
            )
            
            loader_kwargs = {
                'batch_sampler': batch_sampler,
                'num_workers': min(self.workers, os.cpu_count() - 1),  # Leave one CPU free
                'persistent_workers': True,
                'prefetch_factor': 3,  # Increased from 2
                'collate_fn': self.custom_collate,
                'worker_init_fn': self.worker_init_fn,
            }
            
            if torch.cuda.is_available():
                loader_kwargs.update({
                    'pin_memory': True,
                    'pin_memory_device': 'cuda'
                })
            
            return DataLoader(dataset, **loader_kwargs)

    def test_dataloader(self, shuffle:bool=False) -> DataLoader:
        # print('loading test set with {} samples'.format(len(self.data_set[1][self.test_sw_ds_idxs].numpy())))
        print('loading test set with {} samples'.format(len(self.data_set[1][self.test_idxs])))

        testset_transforms = self.testset_transforms
        if self.testset_transforms is None:
            testset_transforms = A.Compose([  # assumption by albumentations: image is in HWC
                # self.normalize,
                ToTensorV2(),  # convert to CHW
            ])
        return DataLoader(TransformTensorDataset(
            (
                self.data_set[0][self.test_idxs],
                self.data_set[1][self.test_idxs],
                self.data_set[2][self.test_idxs],
                self.data_set[3][self.test_idxs],
            ),
            transform=testset_transforms),
            shuffle=shuffle,
            num_workers=self.workers,
            batch_size=self.batch_size,
            worker_init_fn=self._worker_init_fn,
            multiprocessing_context='spawn' if self.workers > 0 else None,
            persistent_workers=True if self.workers > 0 else False)


    def all_dataloader(self, shuffle:bool=False) -> DataLoader:
        if self.data_set is None:
            raise RuntimeError("self.data_set was set to None in datamodule.setup_fold to save memory in hyper parameter tuning. Please enable if you want to use self.all_dataloader().")
        print('loading whole patch data set with {} samples'.format(len(self.data_set)))
        transformer = A.Compose([
                                 # self.normalize,
                                 ToTensorV2(),
                                 ])
        return DataLoader(TransformTensorDataset(
            (
                self.data_set[0],
                self.data_set[1],
                self.data_set[2],
                self.data_set[3],
            ),
            transform=transformer),
            shuffle=shuffle,
            num_workers=self.workers,
            batch_size=self.batch_size)

    def create_debug_samples(self, n=20):
        '''
        Save n samples from each data set to folder, for purposes of testing if augmentations are correctly applied
        :param n: number of samples saved to self.data_dir/Test_Figs for train/val/test set
        :return:
        '''
        # create folders
        dir = os.path.join(self.data_dir, 'Test_Figs')
        if not os.path.exists(dir):
            os.mkdir(dir)

        # load dataset loaders with batch_size = 1
        orig_bs = self.batch_size
        self.batch_size = 1
        dataloaders = {'train': self.train_dataloader(),
                       'val': self.val_dataloader(),
                       # 'test': self.test_dataloader()
                       }

        for phase in ['train', 'val']:
            print('\tSaving Debug Figures for {}'.format(phase))
            i = 0
            c1 = None
            c2 = None
            c3 = None
            if self.input_features == 'RGBRN':
                c4 = None
                c5 = None
            response = None
            for inputs, labels in dataloaders[phase]:
                # save sample figures
                if i < n:
                    plt.imshow(inputs.squeeze().movedim(0, 2).numpy())
                    plt.savefig(os.path.join(dir, 'fold{}_{}_{}_{}.png'.format(self.fold, phase, i, str(labels.squeeze().item())[:5])))
                    i += 1
        self.batch_size = orig_bs


    def _load_yield_points(self, ):
        """
        function for loading yield points from a shapefile.

        Returns geopandas dataframe
        """
        # Construct the search pattern to match all '.shp' files in the directory
        pattern = os.path.join(self.data_dir, '*.shp')
        # Use glob.glob to find all files matching the pattern
        shapefile_paths = glob.glob(pattern)
        df = gpd.read_file(shapefile_paths[0])
        df = df.to_crs("EPSG:25833")
        return df

    def _load_field_shapes(self, ):
        """
        function for loading field/patch shapefile.

        sets geopandas dataframe with shape
        """
        # Construct the search pattern to match all '.shp' files in the directory
        pattern = os.path.join(str.rsplit(self.data_dir, '/', 1)[0], '*.shp')
        # Use glob.glob to find all files matching the pattern
        shapefile_paths = glob.glob(pattern)
        print('loading shapefile: {}'.format(shapefile_paths[0]))
        df = gpd.read_file(shapefile_paths[0])
        df = df.to_crs("EPSG:25833")
        self.field_shapes = df
        self.field_shape = df.loc[df['Patch_id_1']==self.patch_id]

    def _geo_to_pixel(self, geo_point, origin, resolution):
        """
        Convert geographical coordinates to pixel coordinates.

        Parameters:
        - geo_point: Tuple of (longitude, latitude) for the yield point.
        - origin: Tuple of (longitude, latitude) of the image's top-left corner.
        - resolution: Tuple of (x_resolution, y_resolution) in meters per pixel.

        Returns:
        - Tuple of (x_pixel, y_pixel) representing the point's pixel coordinates.
        """
        x_pixel = int((geo_point[0] - origin[0]) / resolution[0])
        y_pixel = int((origin[1] - geo_point[1]) / resolution[1])  # Assuming north-up orientation
        return x_pixel, y_pixel

    def _calculate_bbox_geo(self, geo_point, image_size, feature_matrix_shape, origin, resolution):
        """
        Calculate the bounding box for a 224x224 image centered at the yield point
        with conversion from geographical to pixel coordinates.

        Parameters:
        - point_geo: Tuple of (longitude, latitude) for the yield point.
        - image_size, feature_matrix_shape, origin, resolution: as previously described.

        Returns:
        - bbox: Tuple (x_min, y_min, x_max, y_max) representing the bounding box in pixel coordinates.
        """
        try:
            x_center, y_center = self._geo_to_pixel((geo_point['X'], geo_point['Y']), origin, resolution)
        except KeyError:
            x_center, y_center = self._geo_to_pixel((geo_point['x'], geo_point['y']), origin, resolution)
        half_width, half_height = image_size[0] // 2, image_size[1] // 2

        # Ensure the bbox does not go beyond the feature matrix boundaries
        x_min = max(x_center - half_width, 0)
        y_min = max(y_center - half_height, 0)
        x_max = min(x_center + half_width, feature_matrix_shape[1])
        y_max = min(y_center + half_height, feature_matrix_shape[0])

        return (x_min, y_min, x_max, y_max)

    def _pad_to_size(self, image, target_height=224, target_width=224):
        # Calculate padding amounts
        pad_height = target_height - image.shape[0]
        pad_width = target_width - image.shape[1]

        # Apply padding as needed (this example applies padding equally on both sides)
        image_padded = np.pad(image, ((pad_height // 2, pad_height - pad_height // 2), (pad_width // 2, pad_width - pad_width // 2), (0, 0)), mode='constant', constant_values=0)
        return image_padded

    def _extract_image(self, feature_matrix, bbox):
        """
        Extract the sub-image from the feature matrix using the bounding box.

        Parameters:
        - feature_matrix: The large RGB image of the field.
        - bbox: Tuple (x_min, y_min, x_max, y_max) representing the bounding box.

        Returns:
        - sub_image: The extracted sub-image.
        """
        x_min, y_min, x_max, y_max = bbox
        sub_image = feature_matrix[y_min:y_max, x_min:x_max]
        return sub_image

    def _generate_debug_images(self, feature_matrix, yield_points, bbox_list, set_list=None, n=20, fold=0):
        """
        Generates and saves the debug images.

        Parameters:
        - feature_matrix: The background image matrix.
        - yield_points: List of yield point coordinates (in pixel space).
        - bbox_list: List of bounding box coordinates for each yield point.
        - set_list: List of subsets for each yield point defining the color.
        - n: Number of points to plot (for limiting the number of debug images generated).
        """
        # create folders
        output_dir = os.path.join(self.data_dir, 'Test_Figs')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Define a color map for different subsets
        color_map = {
            'train': 'deepskyblue',
            'val': 'blue',
            'test': 'deeppink',
            'discard': 'darksalmon',  # Optional, if you have 'discard' as a subset
        }

        # Debug Image 1: Feature matrix with yield points and bounding boxes
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(feature_matrix)

        for i, (point, bbox) in enumerate(zip(yield_points, bbox_list)):
            if set_list:
                subset = set_list[i]
                color = color_map.get(subset, 'black')  # Use 'black' as default color if subset not in color_map
            else:
                color = 'black'  # Default color if set_list is not provided

            # Mark yield point as a dot
            ax.plot(point[0], point[1], 'ro', markersize=5, color=color)
            # Annotate the point with its index
            ax.text(point[0], point[1], str(i), color='white', fontsize=12)

            # Draw bounding box
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Draw box for sliding window dataset
        lower_bound_x = 327
        lower_bound_y = 273 if self.patch_id in self.flowerstrip_patch_ids else 327
        upper_bound_x = feature_matrix.shape[0] - lower_bound_x
        if self.patch_id in self.flowerstrip_patch_ids:
            upper_bound_y = feature_matrix.shape[1] - lower_bound_y + 1
        else:
            upper_bound_y = feature_matrix.shape[1] - lower_bound_y

        rect = patches.Rectangle((lower_bound_x, upper_bound_y), upper_bound_x - lower_bound_x, -1*(upper_bound_y-lower_bound_y), linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rect)

        # Calculate the midpoint of the feature_matrix
        mid_x = feature_matrix.shape[1] // 2  # Midpoint in X direction
        mid_y = feature_matrix.shape[0] // 2  # Midpoint in Y direction

        # Draw horizontal and vertical lines at the midpoint
        ax.axhline(y=mid_y, color='yellow', linestyle='--')  # Horizontal line
        ax.axvline(x=mid_x, color='yellow', linestyle='--')  # Vertical line

        plt.axis('off')
        plt.savefig(f"{output_dir}/debug_image_all_points_"+str(fold)+".png", bbox_inches='tight')
        plt.close(fig)

        # Debug Image 2: Extracted image for each yield point
        for i, (point, bbox) in enumerate(zip(yield_points, bbox_list)):
            if i < n:

                extracted_img = feature_matrix[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                fig, ax = plt.subplots(1, figsize=(5, 5))
                ax.imshow(extracted_img)
                # Mark the center yield point
                center_x, center_y = (bbox[2] - bbox[0]) // 2, (bbox[3] - bbox[1]) // 2
                ax.plot(center_x, center_y, 'ro', markersize=5)
                # Annotate the center point with its index
                ax.text(center_x, center_y, str(i), color='white', fontsize=12)

                plt.axis('off')
                plt.savefig(f"{output_dir}/extracted_image_{i}_"+str(fold)+".png", bbox_inches='tight')
                plt.close(fig)

    def _generate_clipped_rs_images(self, feature_matrix, yield_points, bbox_list, set_list=None, n=20, yield_values=None):
        """
        Generates and saves the images.

        Parameters:
        - feature_matrix: The background image matrix.
        - yield_points: List of yield point coordinates (in pixel space).
        - bbox_list: List of bounding box coordinates for each yield point.
        - set_list: List of subsets for each yield point defining the color.
        - n: Number of points to plot (for limiting the number of debug images generated).
        """
        # create folders
        output_dir = os.path.join(self.data_dir, 'Test_Figs')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Define a color map for different subsets
        color_map = {
            'train': 'deepskyblue',
            'val': 'blue',
            'test': 'deeppink',
            'discard': 'darksalmon',  # Optional, if you have 'discard' as a subset
        }

        # Debug Image 1: Feature matrix with yield points and bounding boxes
        fig, ax = plt.subplots(1, figsize=(10, 10))
        field_mask = np.any(feature_matrix != [0, 0, 0], axis=-1)
        # Create a new image array filled with black pixels
        new_image = np.zeros_like(feature_matrix)

        field_mask_black = np.any(feature_matrix == [0, 0, 0], axis=-1)

        # Set field pixels to white
        new_image[field_mask] = [255, 255, 255]
        # Set black pixels to grey
        new_image[field_mask_black] = [200, 200, 200]
        ax.imshow(new_image)

        feature_matrix_1 = feature_matrix
        feature_matrix_1[field_mask_black] = [200, 200, 200]

        for i, (point, bbox) in enumerate(zip(yield_points, bbox_list)):

            # Mark yield point as a dot
            ax.plot(point[0], point[1], 'ro', markersize=5, color='#6f30a0')

            # Draw bounding box
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1.5, edgecolor='#6f30a0', facecolor='none')
            ax.add_patch(rect)

            # Extract the clipped image from feature_matrix
            # bbox format is assumed to be [x_min, y_min, x_max, y_max]
            clipped_image = feature_matrix_1[bbox[1]:bbox[3], bbox[0]:bbox[2]]


            # Set the extent for the clipped image
            extent = [bbox[0], bbox[2], bbox[1], bbox[3]]  # left, right, bottom, top

            # Optionally, adjust the limits of the plot to ensure everything is visible
            ax.set_xlim([0, feature_matrix_1.shape[1]])
            ax.set_ylim([feature_matrix_1.shape[0], 0])

            # Add the clipped image to the plot
            ax.imshow(np.fliplr(np.rot90(np.rot90(clipped_image))), extent=extent)

        plt.axis('off')
        plt.savefig(f"{output_dir}/clipped_rs_images_1.png", bbox_inches='tight')
        plt.close(fig)

        # Debug Image 2: yield points with values
        fig, ax = plt.subplots(1, figsize=(10, 10))

        ax.imshow(new_image)

        # Normalize yield values to 0-1 for color mapping
        norm = Normalize(vmin=min(yield_values), vmax=max(yield_values))
        cmap = plt.get_cmap('RdYlGn')  # Red to Green color map
        mapper = ScalarMappable(norm=norm, cmap=cmap)

        for i, (point, bbox, yield_value) in enumerate(zip(yield_points, bbox_list, yield_values)):
            # Determine color from yield value
            color = mapper.to_rgba(yield_value)
            # Mark yield point as a dot
            ax.plot(point[0], point[1], 'ro', markersize=8, color=color)
            # # Draw bounding box
            # rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='blue', facecolor='none')
            # ax.add_patch(rect)

        plt.axis('off')
        plt.savefig(f"{output_dir}/yield_points_5.png", bbox_inches='tight')
        plt.close(fig)

    def generate_point_dataset(self, feature_matrix, field_bounds, resolution):
        """
        Generates the point dataset from yield points, self.point_data_set.

        Parameters:
        - feature_matrix: The RGB image of the field.

        """
        # Reorder the tensor to H,W,C
        feature_matrix = np.transpose(feature_matrix, (1, 2, 0)).numpy().astype(np.uint8)
        # feature_matrix = feature_matrix.movedim(1, 3).numpy()
        # Normalize or scale the tensor values to [0, 255] if not already
        if feature_matrix.max() <= 1.0:
            feature_matrix = (feature_matrix * 255).astype(np.uint8)
        origin = (field_bounds[0], field_bounds[1])

        self.point_dataset = None
        # get yield point geo dataframe
        yield_points = self._load_yield_points()
        # set patch/field shape
        self._load_field_shapes()
        bbox_list = []
        pixel_points = []  # To store pixel coordinates of yield points for debug images

        label_tensor = None
        img_array = None
        for idx, point in yield_points.iterrows():
            try:
                point_geo = (point['X'], point['Y'])
            except KeyError:
                point_geo = (point['x'], point['y'])
            x_pixel, y_pixel = self._geo_to_pixel(point_geo, origin, resolution)
            pixel_points.append((x_pixel, y_pixel))

            bbox = self._calculate_bbox_geo(point, (self.kernel_size, self.kernel_size), feature_matrix.shape, origin, resolution)
            bbox_list.append(bbox)

            image = self._extract_image(feature_matrix, bbox)
            # Apply padding before expanding dimensions and concatenating
            image_padded = self._pad_to_size(image, self.kernel_size, self.kernel_size)
            if idx == 0:
                label_tensor = torch.tensor(point['yield_smc']).unsqueeze(0)
                img_array = np.expand_dims(image_padded, axis=0)
            else:
                # Concatenate along the first axis (axis=0)
                label_tensor = torch.cat((label_tensor, torch.tensor(point['yield_smc']).unsqueeze(0)), 0) # yield_smc -> adjusted to standard moisture content
                # img_array = np.concatenate((img_array, image), axis=0)
                img_array = np.concatenate((img_array, np.expand_dims(image_padded, axis=0)), axis=0)

        n = len(label_tensor)
        dates = torch.tensor([self.img_date.year, self.img_date.month, self.img_date.day]).repeat(n, 1)
        crop_types = torch.tensor(self.crop_type).repeat(n, 1)

        self.data_set = (img_array, label_tensor, crop_types, dates)

        print('point dataset generation done')

    def _assign_to_spatial_blocks_and_subsets(self, yield_points, field_bounds, fold_k):
        """
        Extends the method to assign each yield point not just to a spatial block but also
        to a subset (train, val, test, discard) based on the spatial block and fold.

        Parameters:
        - yield_points: DataFrame containing yield points with 'X' and 'Y' coordinates.
        - field_bounds: Tuple of (min_x, min_y, max_x, max_y) defining the field's bounding box.
        - fold_k: The current fold (0 to 3) for which the assignment is made.
        """
        min_x, min_y, max_x, max_y = field_bounds
        mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        # Define a function to determine the block based on coordinates
        def assign_block(x, y):
            if x < mid_x and y >= mid_y:
                return 0  # Bottom-left block
            elif x >= mid_x and y >= mid_y:
                return 1  # Bottom-right block
            elif x < mid_x and y < mid_y:
                return 2  # Top-left block
            else:
                return 3  # Top-right block

        # Define block to subset assignment for each fold
        # This needs to be defined based on your cross-validation strategy
        # Example strategy:
        block_to_subset_map = {
            0: {0: 'test', 1: 'val', 2: 'train', 3: 'train'},
            1: {0: 'train', 1: 'test', 2: 'train', 3: 'val'},
            2: {0: 'train', 1: 'train', 2: 'val', 3: 'test'},
            3: {0: 'val', 1: 'train', 2: 'test', 3: 'train'},
        }

        # Assign each point to a block
        try:
            yield_points['block'] = yield_points.apply(lambda row: assign_block(row['X'], row['Y']), axis=1)
        except KeyError:
            yield_points['block'] = yield_points.apply(lambda row: assign_block(row['x'], row['y']), axis=1)
        # Assign each point to a subset based on its block and the current fold
        yield_points['subset'] = yield_points['block'].apply(lambda block: block_to_subset_map[fold_k][block])

        return yield_points

    def _mark_points_extending_beyond_subsets(self, yield_points, bbox_list, subset_boundaries, origin, resolution):
        """
        Marks yield points for discard if their surrounding image polygon intersects
        with any polygon of subsets that the point does not belong to.
        """
        # Convert subset boundaries from geo-coordinates to pixel coordinates and create polygons
        pixel_subset_polygons = {}
        for subset, bounds in subset_boundaries.items():
            min_x_geo, min_y_geo, max_x_geo, max_y_geo = bounds
            # Convert corners to pixel coordinates
            corners_pixel = [self._geo_to_pixel((x, y), origin, resolution) for x, y in [(min_x_geo, min_y_geo), (max_x_geo, min_y_geo), (max_x_geo, max_y_geo), (min_x_geo, max_y_geo)]]
            # Create a polygon for the subset
            pixel_subset_polygons[subset] = Polygon(corners_pixel)

        for idx, (point, bbox) in enumerate(zip(yield_points.itertuples(), bbox_list)):
            point_subset = point.subset
            # Create a polygon for the yield point's surrounding image
            image_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])

            if idx == 126 or idx == 100:
                print()
            # Check for intersections with polygons of subsets the point does NOT belong to
            for subset, polygon in pixel_subset_polygons.items():
                if subset != point_subset and image_polygon.intersects(polygon):
                    # Mark the point for discard
                    yield_points.at[idx, 'subset'] = 'discard'
                    break  # No need to check other subsets once an intersection is found

        return yield_points
    def _get_subset_boundaries(self, field_bounds, fold_k):
        """
        Calculates the subset boundaries for a given fold based on the field bounds and the specific schema.

        Parameters:
        - field_bounds: Tuple of (min_x, min_y, max_x, max_y) defining the field's bounding box.
        - fold_k: The current fold (0 to 3).

        Returns:
        - subset_boundaries: A dictionary with keys 'train', 'val', 'test' and their corresponding
                             pixel-coordinate boundaries (min_x, min_y, max_x, max_y).
        """
        min_x, min_y, max_x, max_y = field_bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        # Define block boundaries within the field
        block_boundaries = {
            0: (min_x, min_y, mid_x, mid_y),
            1: (mid_x, min_y, max_x, mid_y),
            2: (min_x, mid_y, mid_x, max_y),
            3: (mid_x, mid_y, max_x, max_y),
        }

        # Fold-specific block to subset assignment based on the provided schema
        fold_assignments = {
            0: {'test': [0], 'val': [1], 'train': [2, 3]},
            1: {'test': [1], 'val': [3], 'train': [0, 2]},
            2: {'test': [3], 'val': [2], 'train': [0, 1]},
            3: {'test': [2], 'val': [0], 'train': [1, 3]},
        }

        # Initialize dictionary to hold subset boundaries
        subset_boundaries = {'train': None, 'val': None, 'test': None}

        # Assign boundaries to subsets based on the current fold's assignment
        for subset, blocks in fold_assignments[fold_k].items():
            # For each subset, calculate the combined boundaries of its assigned blocks
            subset_bounds = [block_boundaries[block] for block in blocks]
            # Calculate min and max boundaries across all assigned blocks
            min_x_subset = min(bounds[0] for bounds in subset_bounds)
            # min_y_subset = min(bounds[1] for bounds in subset_bounds)
            min_y_subset = max(bounds[1] for bounds in subset_bounds)
            max_x_subset = max(bounds[2] for bounds in subset_bounds)
            # max_y_subset = max(bounds[3] for bounds in subset_bounds)
            max_y_subset = min(bounds[3] for bounds in subset_bounds)
            # Update subset boundaries
            subset_boundaries[subset] = (min_x_subset, min_y_subset, max_x_subset, max_y_subset)

        return subset_boundaries

    def setup_fold_point_ds(self, fold: int = 0, indices_dir=None) -> None:
        """
        Setup the datasets for fold k of the 4-fold spatial cross-validation.

        Parameters:
        - fold_k: The current fold (0 to 3)
        """
        if self.validation_strategy != 'SCV': raise NotImplementedError("Point dataset is only implemented for SCV.")

        # # since swDS contains float and pointDS ints, we convert the set in a sloppy way to prevent the need to redo everything
        self.data_set = (self.data_set[0].astype(np.float32), self.data_set[1], self.data_set[2], self.data_set[3])

        if not self.load_subset_indices(k=fold, dataset_type='pointDS', indices_dir=indices_dir):
            field_bounds, resolution, shape, feature_matrix = self._get_field_bounds()
            origin = (field_bounds[0], field_bounds[1])

            # get yield point geo dataframe
            yield_points = self._load_yield_points()
            # Assign each point to a spatial block
            yield_points = self._assign_to_spatial_blocks_and_subsets(yield_points=yield_points, field_bounds=field_bounds, fold_k=fold)

            # get bounding boxes of yield points
            bbox_list = []
            for idx, point in yield_points.iterrows():
                bbox = self._calculate_bbox_geo(point, (self.kernel_size, self.kernel_size), shape, origin, resolution)
                bbox_list.append(bbox)
            # get bounding boxes of fields
            subset_boundaries = self._get_subset_boundaries(field_bounds=field_bounds, fold_k=fold)
            # mark partially spatially overlapping bounding boxes of yield points with another subset
            yield_points = self._mark_points_extending_beyond_subsets(yield_points=yield_points, bbox_list=bbox_list, subset_boundaries=subset_boundaries, origin=origin, resolution=resolution)

            # Extract indices for each subset
            train_indices = yield_points[yield_points['subset'] == 'train'].index.tolist()
            val_indices = yield_points[yield_points['subset'] == 'val'].index.tolist()
            test_indices = yield_points[yield_points['subset'] == 'test'].index.tolist()

            # creating debug images
            pixel_points = []  # To store pixel coordinates of yield points for debug images
            yield_values= []
            set_list = []
            for idx, point in yield_points.iterrows():
                try:
                    point_geo = (point['X'], point['Y'])
                except KeyError:
                    point_geo = (point['x'], point['y'])
                x_pixel, y_pixel = self._geo_to_pixel(point_geo, origin, resolution)
                pixel_points.append((x_pixel, y_pixel))
                yield_values.append(point['yield'])
                set_label = point['subset']
                set_list.append(set_label)
            self._generate_clipped_rs_images(feature_matrix, pixel_points, bbox_list, set_list=set_list, yield_values=yield_values)
            self._generate_debug_images(feature_matrix, pixel_points, bbox_list, set_list=set_list, fold=fold)

            self.val_idxs = val_indices
            self.train_idxs = train_indices
            if self.validation_strategy == 'SCV':
                self.test_idxs = test_indices
                # save indices
                self.save_subset_indices(k=fold, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, dataset_type='pointDS')

    # Add this new method near other helper methods
    def _worker_init_fn(self, worker_id):
        """Initialize worker with proper CUDA device."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and torch.cuda.is_available():
            device_id = worker_id % torch.cuda.device_count()
            try:
                torch.cuda.set_device(device_id)
            except RuntimeError as e:
                print(f"Warning: Could not set CUDA device for worker {worker_id}: {e}")
