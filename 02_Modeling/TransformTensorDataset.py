# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script provides custom dataset classes for handling image data in the yield prediction pipeline, supporting transformations,
augmentations, and metadata handling for both supervised and self-supervised learning approaches.

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

from torch.utils.data.dataset import Dataset, ConcatDataset, TensorDataset
from torchvision.datasets.vision import VisionDataset
from lightly.data import LightlyDataset
from typing import Any, Callable, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch

class TransformTensorDataset(Dataset):
    """
    TensorDataset with support for transforms and augmentations.
    
    This class extends PyTorch's Dataset to handle tensor data with optional transformations.
    It supports converting between tensors and numpy arrays for compatibility with various transforms.
    
    Attributes:
        data (tuple): Tuple containing input data, labels, crop types, and dates
        transform (Optional[Callable]): Optional transform to apply to the input data
    """
    def __init__(self, data: tuple, transform: Optional[Callable] = None) -> None:
        """
        Initialize the dataset.
        
        Args:
            data (tuple): Tuple containing (input_data, labels, crop_types, dates)
            transform (Optional[Callable]): Optional transform to apply to the input data
        """
        self.data = data
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any, Any, Any]:
        """
        Get a single data sample.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, Any, Any, Any]: Tuple containing (transformed_input, label, crop_type, date)
        """
        x = self.data[0][index]

        if self.transform:
            # Convert tensor to numpy array before transformation
            if torch.is_tensor(x):
                x = x.cpu().numpy()
            # Apply transformation
            transformed = self.transform(image=x)
            x = transformed['image']
            # Convert back to tensor if needed
            if not torch.is_tensor(x):
                x = torch.from_numpy(x).float()

        y = self.data[1][index]
        crop_type = self.data[2][index]
        date = self.data[3][index]

        return x, y, crop_type, date

    def __len__(self) -> int:
        """
        Get the total number of samples.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data[0])

class LightyWrapper(Dataset):
    """
    Wrapper for Lightly dataset to support additional metadata.
    
    This class extends the Lightly dataset to include crop types and dates alongside
    the standard image and label data.
    
    Attributes:
        lightly_dataset (Dataset): Base Lightly dataset
        crop_types (List): List of crop types
        dates (List): List of dates
    """
    def __init__(self, lightly_dataset: Dataset, crop_types: List, dates: List) -> None:
        """
        Initialize the wrapper.
        
        Args:
            lightly_dataset (Dataset): Base Lightly dataset
            crop_types (List): List of crop types
            dates (List): List of dates
        """
        self.lightly_dataset = lightly_dataset
        self.crop_types = crop_types
        self.dates = dates

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Get a single data sample.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Tuple[Any, Any, Any, Any]: Tuple containing (images, label, crop_type, date)
        """
        imgs, label, _ = self.lightly_dataset[index]
        crop_type = self.crop_types[index]
        date = self.dates[index]

        return imgs, label, crop_type, date

    def __len__(self) -> int:
        """
        Get the total number of samples.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.lightly_dataset)


class CustomVisionDataset(VisionDataset):
    """
    Custom VisionDataset implementation for PIL images.
    
    This class extends torchvision's VisionDataset to handle PIL images with
    optional transformations.
    
    Attributes:
        dataset (tuple): Tuple containing input images and labels
        transform (Optional[Callable]): Optional transform to apply to the images
    """
    def __init__(self, pil_imgs: tuple, transform: Optional[Callable] = None) -> None:
        """
        Initialize the dataset.
        
        Args:
            pil_imgs (tuple): Tuple containing (images, labels)
            transform (Optional[Callable]): Optional transform to apply to the images
        """
        super().__init__(root=None, transform=transform)
        self.dataset = pil_imgs

    def __getitem__(self, index: int) -> Tuple[Image.Image, Any]:
        """
        Get a single data sample.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Tuple[Image.Image, Any]: Tuple containing (transformed_image, label)
        """
        x = Image.fromarray(np.uint8(self.dataset[0][index]))

        if self.transform:
            x = self.transform(x)

        y = self.dataset[1][index]

        return x, y

    def __len__(self) -> int:
        """
        Get the total number of samples.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset[0])
