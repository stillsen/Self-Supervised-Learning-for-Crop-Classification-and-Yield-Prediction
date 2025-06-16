# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script implements hyperparameter tuning for yield prediction models using Ray Tune, supporting both supervised and self-supervised learning approaches.
It handles model training, validation, and optimization with support for mixed precision training and early stopping.

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports
import os

# Libs
from copy import deepcopy
from typing import Optional, Dict, Any, Tuple

from ray import tune
from collections import OrderedDict

import torch
from torch import nn
from lightly.transforms.vicreg_transform import VICRegTransform

# Own modules
from RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer

# Add these imports at the top
import multiprocessing
import torch.multiprocessing as mp

# Set start method to spawn
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Add near the top of the file, after imports
from functools import lru_cache
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


class TuneYieldRegressor(tune.Trainable):
    """
    Custom Ray Tune Trainable class for hyperparameter tuning of yield prediction models.
    
    This class handles:
    1. Model setup and configuration
    2. Training and validation steps
    3. Checkpoint management
    4. Support for both supervised and self-supervised learning
    5. Mixed precision training
    6. Early stopping
    """

    @staticmethod
    def safe_worker_init_fn(self, worker_id: int) -> None:
        """
        Static method for safe worker initialization in data loading.
        
        Args:
            worker_id (int): ID of the worker process
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if torch.cuda.is_available():
                device_id = worker_id % torch.cuda.device_count()
                try:
                    torch.cuda.set_device(device_id)
                except RuntimeError as e:
                    print(f"Warning: Could not set CUDA device for worker {worker_id}: {e}")

    def setup(self,
              config: Dict[str, Any],
              workers: Optional[int] = None,
              datamodule: Optional[Any] = None,
              datamodule_SSL_pointDS: Optional[Any] = None,
              momentum: Optional[float] = None,
              patch_no: Optional[int] = None,
              architecture: Optional[str] = None,
              tune_fc_only: Optional[bool] = None,
              pretrained: Optional[bool] = None,
              criterion: Optional[Any] = None,
              device: Optional[torch.device] = None,
              training_response_standardizer: Optional[Any] = None,
              state_dict: Optional[Dict[str, Any]] = None,
              SSL: Optional[str] = None,
              tune_by_downstream_performance: bool = False,
              ) -> None:
        """
        Initialize the tuning process with model configuration and data.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary for hyperparameters
            workers (Optional[int]): Number of worker processes
            datamodule (Optional[Any]): Data module for training
            datamodule_SSL_pointDS (Optional[Any]): Data module for SSL point dataset
            momentum (Optional[float]): Momentum for optimizers
            patch_no (Optional[int]): Patch number for identification
            architecture (Optional[str]): Model architecture to use
            tune_fc_only (Optional[bool]): Whether to only tune fully connected layers
            pretrained (Optional[bool]): Whether to use pretrained models
            criterion (Optional[Any]): Loss function
            device (Optional[torch.device]): Device to use for training
            training_response_standardizer (Optional[Any]): Standardizer for training responses
            state_dict (Optional[Dict[str, Any]]): Initial model state
            SSL (Optional[str]): SSL framework to use
            tune_by_downstream_performance (bool): Whether to tune by downstream performance
        """
        print('tuning hyper parameters on patch {}'.format(patch_no))

        self.epoch = 0
        self.SSL = SSL
        self.latest_downstream_performance = None
        self.architecture = architecture
        self.device = device
        self.workers = workers
        self._tune_by_downstream_performance = tune_by_downstream_performance

        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device=device)
            torch.cuda.synchronize()

        # Load dataset only once for the first trial
        self._load_dataset(datamodule)
        # Adjust batch size for each trial
        self.datamodule.set_batch_size(batch_size=int(config['batch_size']))

        # set VICReg transforms and tune min_scale
        if (state_dict is None) and (architecture == 'VICReg' or architecture == 'VICRegConvNext'):
            self.datamodule.set_transforms(
                trainset_transforms=VICRegTransform(input_size=datamodule.kernel_size,
                                                    hf_prob=0.5,
                                                    vf_prob=0.5,
                                                    rr_prob=0.5,
                                                    min_scale=config['transforms_min_scale'],
                                                    cj_prob=0.2,
                                                    cj_bright=0.1,
                                                    cj_contrast=0.1,
                                                    cj_hue=0.1,
                                                    cj_sat=0.1,
                                                    normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]}, )
            )

        # Initialize trainer based on architecture and SSL mode
        if architecture == 'SimSiam' or architecture == 'VICRegL' or architecture == 'VICRegLConvNext' or architecture == 'VICReg' or architecture == 'VICRegConvNext':
            # For SSL architectures, check if we need hidden_dim and output_dim
            if SSL is not None:
                # SSL mode - use dimensions from config if available, otherwise use defaults
                hidden_dim = config.get('hidden_dim', 2048)  # default value
                output_dim = config.get('output_dim', 8192)  # default value
                self._setup_trainer(
                    SSL=SSL,
                    architecture=architecture,
                    config=config,
                    criterion=criterion,
                    device=device,
                    pretrained=pretrained,
                    tune_fc_only=tune_fc_only,
                    workers=workers,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
            else:
                # Non-SSL mode - don't pass hidden_dim and output_dim
                self._setup_trainer(
                    SSL=SSL,
                    architecture=architecture,
                    config=config,
                    criterion=criterion,
                    device=device,
                    pretrained=pretrained,
                    tune_fc_only=tune_fc_only,
                    workers=workers
                )
        else:
            # For non-SSL architectures
            self._setup_trainer(
                SSL=SSL,
                architecture=architecture,
                config=config,
                criterion=criterion,
                device=device,
                pretrained=pretrained,
                tune_fc_only=tune_fc_only,
                workers=workers
            )

        # build and update dataloaders
        self._setup_dataloader()

        # pre-trained model incoming, i.e. domain-tuning
        if state_dict is not None:
            if architecture == 'SimSiam' or architecture == 'VICRegL' or architecture == 'VICRegLConvNext' or architecture == 'VICReg' or architecture == 'VICRegConvNext':
                if SSL is None:
                    self.model_wrapper.model.SSL_training = False

            self.model_wrapper.model.load_state_dict(state_dict, strict=False)
            self.model_wrapper.reinitialize_fc_layers()
            self.model_wrapper.disable_all_but_fc_grads()

        # update criterion
        self.model_wrapper.set_criterion(criterion=criterion)

        if (SSL is not None) and (SSL == 'VICReg' or SSL == 'VICRegConvNext'):
            self.datamodule_SSL_pointDS = datamodule_SSL_pointDS
            self.model_wrapper.criterion.mu_param = config['mu_lambda_param']
            self.model_wrapper.criterion.lambda_param = config['mu_lambda_param']

        # update optimizer to new hyper parameter set
        self.model_wrapper.set_optimizer()
        # data parallelize and send model to device
        self.model_wrapper.parallize_and_to_device()

    def _setup_trainer(self, 
                      SSL: Optional[str], 
                      architecture: str, 
                      config: Dict[str, Any], 
                      criterion: Any, 
                      device: torch.device, 
                      pretrained: bool, 
                      tune_fc_only: bool, 
                      workers: int, 
                      hidden_dim: Optional[int] = None, 
                      output_dim: Optional[int] = None) -> None:
        """
        Set up the model trainer with specified configuration.
        
        Args:
            SSL (Optional[str]): SSL framework to use
            architecture (str): Model architecture
            config (Dict[str, Any]): Configuration dictionary
            criterion (Any): Loss function
            device (torch.device): Device to use
            pretrained (bool): Whether to use pretrained model
            tune_fc_only (bool): Whether to only tune fully connected layers
            workers (int): Number of worker processes
            hidden_dim (Optional[int]): Hidden dimension for SSL
            output_dim (Optional[int]): Output dimension for SSL
        """
        self.model_wrapper = RGBYieldRegressor_Trainer(
            pretrained=pretrained,
            tune_fc_only=tune_fc_only,
            architecture=architecture,
            criterion=criterion,
            device=device,
            workers=workers,
            SSL=SSL,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        # update hyperparameters
        self.model_wrapper.set_hyper_parameters(
            lr=config['lr'],
            wd=config['wd'],
            batch_size=config['batch_size'],
            T_0=config['T_0'],
            T_mult=config['T_mult']
        )

    def _load_dataset(self, datamodule: Any) -> None:
        """
        Load and initialize the dataset.
        
        Args:
            datamodule (Any): Data module to load
        """
        if not hasattr(self, 'datamodule'):
            self.datamodule = datamodule

    def _setup_dataloader(self) -> None:
        """
        Set up data loaders for training and validation.
        """
        self.model_wrapper.set_dataloaders(dataloaders={'train': self.datamodule.train_dataloader(), 'val': self.datamodule.val_dataloader(), })

    def _evaluate_downstream_model(self) -> float:
        """
        Evaluate the model's downstream performance.
        
        Returns:
            float: Best validation loss achieved
        """
        SSL = None

        # initialize trainer and architecture
        criterion = nn.MSELoss(reduction='mean')
        eval_model_wrapper = RGBYieldRegressor_Trainer(
            pretrained=False,
            tune_fc_only=True,
            architecture=self.architecture,
            criterion=criterion,
            # device="cpu",
            device=self.device,
            workers=self.workers,
            SSL=SSL,
        )

        self.datamodule_SSL_pointDS.augmented = True
        self.datamodule_SSL_pointDS.set_batch_size(batch_size=64)

        # update hyperparameters
        eval_model_wrapper.set_hyper_parameters(lr=0.00005, wd=0.00007, batch_size=64, T_0=1, T_mult=2)

        if not self.compute_singular_loss:
            eval_model_wrapper.mean_yield_per_crop = self.datamodule_SSL_pointDS.mean_yield_per_crop
            eval_model_wrapper.std_yield_per_crop = self.datamodule_SSL_pointDS.std_yield_per_crop

        eval_model_wrapper.set_dataloaders(dataloaders={'train': self.datamodule_SSL_pointDS.train_dataloader(), 'val': self.datamodule_SSL_pointDS.val_dataloader(), })

        eval_model_wrapper.set_criterion(criterion=criterion)
        # update optimizer to new hyper parameter set
        eval_model_wrapper.set_optimizer()
        # data parallelize and send model to device
        eval_model_wrapper.parallize_and_to_device()
        eval_model_wrapper.SSL = SSL

        state_dict = deepcopy(self.model_wrapper.model.state_dict())
        eval_model_wrapper.model.load_state_dict(state_dict)
        # temporarily model SSL to CPU to avoid two conflicting model with AMP gradscaler on a single GPU
        self.model_wrapper.model.to("cpu")

        eval_model_wrapper.model.SSL_training = False
        # freeze conv layers and reinitialize FC layer
        eval_model_wrapper.reinitialize_fc_layers()
        eval_model_wrapper.disable_all_but_fc_grads()

        # train eval_model_wrapper
        eval_model_wrapper.train(min_delta=0.0001,
                                 num_epochs=250,
                                 min_epochs=250, )

        self.model_wrapper.model.to(self.device)
        return eval_model_wrapper.best_loss

    def compute_combined_metric(self, ssl_loss: float, downstream_performance: Optional[float]) -> float:
        """
        Compute combined metric from SSL loss and downstream performance.
        
        Args:
            ssl_loss (float): SSL training loss
            downstream_performance (Optional[float]): Downstream task performance
            
        Returns:
            float: Combined metric value
        """
        if downstream_performance is not None:
            return 0.3 * ssl_loss + 0.7 * downstream_performance  # Adjust weights as needed
        else:
            return ssl_loss  # Use SSL loss alone if downstream performance is not available

    def step(self) -> Dict[str, float]:
        """
        Perform one training step and evaluate validation performance.
        
        Returns:
            Dict[str, float]: Dictionary containing training and validation losses
        """
        # try:
        train_loss = self.model_wrapper.train_step(epoch=self.epoch)
        self.epoch = self.epoch + 1

        # evaluate validation performance
        if self.SSL is None:  # if downstream or no SSL setting
            val_loss = self.model_wrapper.test(phase='val')
        else:  # if hyper-param tuning in SSL
            if self._tune_by_downstream_performance:  # with respect to downstream performance
                SSL_val_loss = self.model_wrapper.test(phase='val')
                if (self.epoch == 9) or (self.epoch % 30 == 0):  # eval every 30 epochs
                    print('------------------ assessing downstream performance ---------------')
                    downstream_val_loss = self._evaluate_downstream_model()
                    self.latest_downstream_performance = downstream_val_loss
                    print('ssl-loss: {}, downstream-loss: {}'.format(SSL_val_loss, downstream_val_loss))
                val_loss = self.compute_combined_metric(SSL_val_loss, self.latest_downstream_performance)
            else:  # with respect to pre-text task
                val_loss = self.model_wrapper.test(phase='val')

        return {"train_loss": train_loss, "val_loss": val_loss}

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoint
            
        Returns:
            str: Path to saved checkpoint
        """
        checkpoint_path = os.path.join(checkpoint_dir, "tuning_model.pth")
        torch.save(self.model_wrapper.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint (str): Path to checkpoint file
        """
        self.model_wrapper.model.load_state_dict(torch.load(checkpoint))

