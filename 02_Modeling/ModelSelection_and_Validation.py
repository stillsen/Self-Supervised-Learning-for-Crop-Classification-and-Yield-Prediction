# -*- coding: utf-8 -*-
"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script handles model selection, hyperparameter tuning, and validation for both supervised and self-supervised learning approaches.
It implements spatial cross-validation, early stopping, and model checkpointing.

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports
import time, gc, os, shutil

import pandas as pd

import torch
import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune import Stopper

import numpy as np
import pickle
from collections import OrderedDict

from lightly.loss import NTXentLoss, VICRegLoss, NegativeCosineSimilarity, VICRegLLoss
from lightly.transforms.vicreg_transform import VICRegTransform

import torch.nn as nn
# Own modules
from TuneYieldRegressor import TuneYieldRegressor
from RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer
from RGBYieldRegressor_Trainer import SimCLR

# Custom stopper that combines NaN handling and maximum iteration stopping
class CombinedNaNAndIterationStopper(Stopper):
    """
    Custom stopper that combines NaN detection and maximum iteration limits.
    
    This stopper will stop a trial if either:
    1. The specified metric becomes NaN
    2. The maximum number of iterations is reached
    """
    
    def __init__(self, metric: str, max_iterations: int):
        """
        Initialize the stopper.
        
        Args:
            metric (str): Name of the metric to monitor
            max_iterations (int): Maximum number of iterations before stopping
        """
        self.metric = metric
        self.max_iterations = max_iterations

    def __call__(self, trial_id: str, result: dict) -> bool:
        """
        Check if the trial should be stopped.
        
        Args:
            trial_id (str): Unique identifier for the trial
            result (dict): Current trial results
            
        Returns:
            bool: True if the trial should be stopped
        """
        # Stop if the metric is NaN
        metric_value = result.get(self.metric, None)
        if metric_value is not None and np.isnan(metric_value):
            print(f"Stopping trial {trial_id} due to NaN value in {self.metric}.")
            return True

        # Stop if the number of iterations exceeds the maximum allowed
        current_iteration = result.get("training_iteration", 0)
        if current_iteration >= self.max_iterations:
            print(f"Stopping trial {trial_id} because it reached {self.max_iterations} iterations.")
            return True

        return False

    def stop_all(self) -> bool:
        """
        Check if all trials should be stopped.
        
        Returns:
            bool: Always returns False as we don't want to stop all trials
        """
        return False

class ModelSelection_and_Validation:
    """
    Handles model selection, hyperparameter tuning, and validation.
    
    This class manages:
    1. Hyperparameter optimization using Ray Tune
    2. Model training and validation across folds
    3. Support for both supervised and self-supervised learning
    4. Early stopping and model checkpointing
    5. Integration with data modules for training and validation
    """
    
    def __init__(self,
                 num_folds: int,
                 pretrained: bool,
                 this_output_dir: str,
                 datamodule: Any,
                 training_response_normalization: bool,
                 validation_strategy: str,
                 patch_no: int,
                 seed: int,
                 num_epochs: int,
                 patience: int,
                 min_delta: float,
                 min_epochs: int,
                 momentum: float,
                 architecture: str,
                 only_tune_hyperparameters: bool,
                 device: str,
                 workers: int,
                 tune_epochs: int = 100,
                 # SSL_transforms=None,
                 datamodule_SSL_pointDS: Optional[Any] = None,
                 dataset_type: str = 'swDS',
                 tune_by_downstream_performance: bool = False,
                 ):
        """
        Initialize the model selection and validation handler.
        
        Args:
            num_folds (int): Number of cross-validation folds
            pretrained (bool): Whether to use pretrained models
            this_output_dir (str): Output directory for results
            datamodule (Any): Data module for training/validation
            training_response_normalization (bool): Whether to normalize training responses
            validation_strategy (str): Validation strategy to use
            patch_no (int): Patch number for identification
            seed (int): Random seed for reproducibility
            num_epochs (int): Maximum number of training epochs
            patience (int): Patience for early stopping
            min_delta (float): Minimum change for early stopping
            min_epochs (int): Minimum number of epochs to train
            momentum (float): Momentum for optimizers
            architecture (str): Model architecture to use
            only_tune_hyperparameters (bool): Whether to only tune hyperparameters
            device (str): Device to use for training
            workers (int): Number of worker processes
            tune_epochs (int): Number of epochs for hyperparameter tuning
            datamodule_SSL_pointDS (Optional[Any]): Data module for SSL point dataset
            dataset_type (str): Type of dataset to use
            tune_by_downstream_performance (bool): Whether to tune by downstream performance
        """
        self.num_folds = num_folds
        self.pretrained = pretrained
        self.this_output_dir = this_output_dir
        self.datamodule = datamodule
        self.datamodule_SSL_pointDS = datamodule_SSL_pointDS
        self.training_response_normalization = training_response_normalization
        self.validation_strategy = validation_strategy
        self.patch_no = patch_no
        self.seed = seed
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.momentum = momentum
        self.architecture = architecture
        self.model_wrapper = None
        self.only_tune_hyperparameters = only_tune_hyperparameters
        self.device = device
        self.workers = workers
        self.tune_epochs = tune_epochs
        self.dataset_type = dataset_type
        self.tune_by_downstream_performance = tune_by_downstream_performance
        self.hidden_dim = 8192
        self.output_dim = 8192

        # init with None
        self.lr = self.wd = self.batch_size = self.T_0 = self.T_mult = self.transforms_min_scale = self.mu_lambda_param = None

    def load_hyperparameters_if_exist(self, k: int, strategy: str = '') -> bool:
        """
        Load hyperparameters from file if they exist.
        
        Args:
            k (int): Fold number
            strategy (str): Strategy identifier
            
        Returns:
            bool: True if hyperparameters were loaded successfully
        """
        # for SSL pretraining, add training phase
        if strategy != '':
            strategy = '_' + strategy
        
        # load tuned hyper params if exist
        filename = os.path.join(self.this_output_dir, 'hyper_df_' + str(k) + strategy + '.csv')
        print(f"Loading hyperparameters for fold {k} and strategy '{strategy}' in '{filename}'")
        if os.path.exists(filename):
            if ('self-supervised' in strategy) and (self.architecture in ['VICRegLConvNext', 'VICRegConvNext', 'VICReg', 'VICRegL']):
                column_names = ['lr', 'wd', 'batch_size', 'T_0', 'T_mult', 'transforms_min_scale', 'mu_lambda_param', 'hidden_dim', 'output_dim']
            else:
                column_names = ['lr', 'wd', 'batch_size', 'T_0', 'T_mult']
            df = pd.read_csv(filename, usecols=column_names)
            self.lr = float(df['lr'][0])
            self.wd = float(df['wd'][0])
            self.batch_size = int(df['batch_size'][0])
            self.T_0 = int(df['T_0'][0])
            self.T_mult = int(df['T_mult'][0])
            if ('self-supervised' in strategy) and (self.architecture in ['VICRegLConvNext', 'VICRegConvNext', 'VICReg', 'VICRegL']):
                self.transforms_min_scale = float(df['transforms_min_scale'][0])
                self.mu_lambda_param = float(df['mu_lambda_param'][0])
                self.hidden_dim = int(df['hidden_dim'][0])
                self.output_dim = int(df['output_dim'][0])
            
            print(f"Hyperparameters loaded successfully for fold {k} and strategy '{strategy}':")
            print(f"Learning rate: {self.lr}")
            print(f"Weight decay: {self.wd}")
            print(f"Batch size: {self.batch_size}")
            print(f"T_0: {self.T_0}")
            print(f"T_mult: {self.T_mult}")
            if ('self-supervised' in strategy) and (self.architecture in ['VICRegLConvNext', 'VICRegConvNext', 'VICReg', 'VICRegL']):
                print(f"Transforms min scale: {self.transforms_min_scale}")
                print(f"Mu lambda param: {self.mu_lambda_param}")
                print(f"Hidden dim: {self.hidden_dim}")
                print(f"Output dim: {self.output_dim}")
            
            return True
        else:
            print(f"No hyperparameters found for fold {k} and strategy '{strategy}'")
            return False

    def save_hyperparameters(self, k: int, strategy: str = '') -> None:
        """
        Save hyperparameters to file.
        
        Args:
            k (int): Fold number
            strategy (str): Strategy identifier
        """
        # for SSL pretraining, add training phase
        if strategy != '':
            strategy = '_' + strategy

        if ('self-supervised' in strategy) and (self.architecture in ['VICRegLConvNext', 'VICRegConvNext', 'VICReg', 'VICRegL']):
            hyper_df = pd.DataFrame({'lr': [self.lr],
                                     'wd': [self.wd],
                                     'batch_size': [self.batch_size],
                                     'T_0': [self.T_0],
                                     'T_mult': [self.T_mult],
                                     'transforms_min_scale': [self.transforms_min_scale],
                                     'mu_lambda_param': [self.mu_lambda_param],
                                     'hidden_dim': [self.hidden_dim],
                                     'output_dim': [self.output_dim],
                                     })
        else:
            hyper_df = pd.DataFrame({'lr': [self.lr],
                                 'wd': [self.wd],
                                 'batch_size': [self.batch_size],
                                 'T_0': [self.T_0],
                                 'T_mult': [self.T_mult],
                                 })

        hyper_df.to_csv(os.path.join(self.this_output_dir, 'hyper_df_' + str(k) + strategy + '.csv'), encoding='utf-8')

    def get_state_dict_if_exists(self, this_output_dir: str, k: int, strategy: str = '') -> Optional[Dict[str, Any]]:
        """
        Load model state dict if it exists.
        
        Args:
            this_output_dir (str): Output directory
            k (int): Fold number
            strategy (str): Strategy identifier
            
        Returns:
            Optional[Dict[str, Any]]: Model state dict if it exists, None otherwise
        """
        state_dict = None
        if strategy != '':
            strategy = '_' + strategy

        if os.path.exists(os.path.join(this_output_dir, 'model_f' + str(k) + strategy + '.ckpt')):
            print('initializing model with pretrained weights')
            checkpoint_path = os.path.join(this_output_dir, 'model_f' + str(k) + strategy + '.ckpt')
            if torch.cuda.is_available():
                state_dict = torch.load(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        return state_dict

    def tune_hyperparameters(self, datamodule: Any, tune_fc_only: bool, criterion: Any, state_dict: Optional[Dict[str, Any]] = None, datamodule_SSL_pointDS: Optional[Any] = None, SSL: Optional[str] = None, tune_name: str = None, patch_no: int = None, seed: int = 42) -> Tuple[float, ...]:
        """
        Tune hyperparameters using Ray Tune.
        
        Args:
            datamodule (Any): Data module for training
            tune_fc_only (bool): Whether to only tune fully connected layers
            criterion (Any): Loss function
            state_dict (Optional[Dict[str, Any]]): Initial model state
            datamodule_SSL_pointDS (Optional[Any]): Data module for SSL point dataset
            SSL (Optional[str]): SSL framework to use
            tune_name (str): Name for the tuning run
            patch_no (int): Patch number
            seed (int): Random seed
            
        Returns:
            Tuple[float, ...]: Best hyperparameters found
        """
        # Function to perform cleanup of checkpoints
        def cleanup_checkpoints(directory):
            if os.path.exists(directory):
                print(f"Cleaning up checkpoints in {directory}...")
                shutil.rmtree(directory)  # Remove the directory and all its contents
                print(f"Cleanup complete. {directory} has been deleted.")

        # tune_name[strategy] = 'Tuning_{}_{}_all_bayes_L1_ALB_TL-FC_f{}_{}_{}'.format(self.architecture, validation_strategy, k, patch_no, strategy)
        if self.architecture == 'resnet18' or self.architecture == 'baselinemodel':
            param_space = {
                "lr": tune.loguniform(1e-6, 1e-1),
                "wd": tune.uniform(0, 5 * 1e-3),
                "batch_size": tune.choice([4, 8, 16, 32, 64, 128, 256, 512]),
                "T_0": tune.choice([1, 10, 50, 100, 200]),
                "T_mult": tune.choice([1, 2]),
            }
        elif self.architecture == 'ConvNeXt':
            param_space = {
                "lr": tune.loguniform(1e-6, 1e-1),
                "wd": tune.uniform(0, 5 * 1e-3),
                # "batch_size": tune.choice([32]),
                "batch_size": tune.choice([8, 16, 32, 64]),
                "T_0": tune.choice([1, 10, 50, 100, 200]),
                "T_mult": tune.choice([1, 2]),
            }
        elif self.architecture == 'SimCLR':
            param_space = {
                "lr": tune.loguniform(1e-6, 1e-1),
                "wd": tune.uniform(0, 5 * 1e-3),
                "batch_size": tune.choice([64, 128, 256, 512, 1024]),
                "T_0": tune.choice([1, 10, 50, 100, 200]),
                "T_mult": tune.choice([1, 2]),
            }
        elif (SSL is not None) and (self.architecture == 'VICReg' or self.architecture == 'VICRegL'):  # pre-training
            param_space = {
                "lr": tune.loguniform(1e-5, 1e-2),  # initially for the next 4 below: "lr": tune.loguniform(1e-6, 1e-1),
                "wd": tune.uniform(1e-4, 5 * 1e-3),  # initially for the next 4 below: "wd": tune.uniform(0, 5 * 1e-3)
                "batch_size": tune.choice([64, 128, 256]),
                "T_0": tune.choice([10, 20, 50, ]),
                "T_mult": tune.choice([2]),
                "transforms_min_scale": tune.choice([0.1, 0.3, 0.5]),
                "mu_lambda_param": tune.choice([1.25, 2.5, 5, 10]),
                "hidden_dim": tune.choice([512, 768, 1024, ]),
                "output_dim": tune.choice([512, 768, 1024, ]),
            }
        elif (SSL is None) and (self.architecture == 'VICReg' or self.architecture == 'VICRegL'):  # downstream
            param_space = {
                "lr": tune.loguniform(1e-5, 1e-2),
                "wd": tune.uniform(1e-4, 5 * 1e-3),
                "batch_size": tune.choice([32, 64, 128, ]),
                "T_0": tune.choice([10, 20, 50, ]),
                "T_mult": tune.choice([2]),
            }
        elif (SSL is not None) and (self.architecture == 'VICRegLConvNext' or self.architecture == 'VICRegConvNext'):  # pre-training
            param_space = {
                "lr": tune.loguniform(1e-5, 1e-2),
                "wd": tune.uniform(1e-4, 5 * 1e-3),
                "batch_size": tune.choice([32, 64]),
                "T_0": tune.choice([10, 20, 50, ]),
                "T_mult": tune.choice([2]),
                "transforms_min_scale": tune.choice([0.1, 0.3, 0.5]),
                "mu_lambda_param": tune.choice([1.25, 2.5, 5, 10]),
                "hidden_dim": tune.choice([512, 768, 1024, ]),
                "output_dim": tune.choice([512, 768, 1024, ]),
            }
        elif (SSL is None) and (self.architecture == 'VICRegLConvNext' or self.architecture == 'VICRegConvNext'):  # downstream
            param_space = {
                "lr": tune.loguniform(1e-5, 1e-2),
                "wd": tune.uniform(1e-4, 5 * 1e-3),
                "batch_size": tune.choice([16, 32, 64]),
                "T_0": tune.choice([10, 20, 50, ]),
                "T_mult": tune.choice([2]),
            }
        elif self.architecture == 'SimSiam':
            param_space = {
                "lr": tune.loguniform(1e-6, 1e-1),
                "wd": tune.uniform(0, 5 * 1e-3),
                "batch_size": tune.choice([64, 128, 256, 512]),
                "T_0": tune.choice([1, 10, 50, 100, 200]),
                "T_mult": tune.choice([1, 2]),
            }



        max_t = 50
        num_trials = 75
        reduction_factor = 4

        bohb_hyperband = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=max_t,  # defines the maximum number of iterations for the scheduler. max_t=100 suggests that you expect significant progress within 100 epochs
            reduction_factor=reduction_factor,  # controls how aggressively trials are reduced. A factor of 4 is quite aggressive, which means after every bracket, only the top 25% of trials proceed
            stop_last_trials=False,
        )
        # Bayesian Optimization HyperBand -> terminates bad trials + Bayesian Optimization
        bohb_search = TuneBOHB(metric='val_loss',
                               mode='min',
                               seed=seed, )
        bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)

        # setup and init tmp dir

        tmp_log_dir = '/scratch/stiller/Ray/'

        if not os.path.exists(tmp_log_dir):
            os.mkdir(tmp_log_dir)
        ray.init(_temp_dir=tmp_log_dir,
                 ignore_reinit_error=True)

        try:
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(
                        TuneYieldRegressor,
                        momentum=self.momentum,
                        patch_no=patch_no,
                        architecture=self.architecture,
                        tune_fc_only=tune_fc_only,
                        pretrained=self.pretrained,
                        datamodule=datamodule,
                        datamodule_SSL_pointDS=datamodule_SSL_pointDS,
                        criterion=criterion,
                        device=self.device,
                        workers=self.workers,
                        training_response_standardizer=datamodule.training_response_standardizer,
                        state_dict=state_dict,
                        SSL=SSL,
                        tune_by_downstream_performance=self.tune_by_downstream_performance,
                    ),
                    {"cpu": 1,
                     "gpu": torch.cuda.device_count()}),

                param_space=param_space,
                tune_config=tune.TuneConfig(
                    metric='val_loss',
                    mode='min',
                    search_alg=bohb_search,
                    scheduler=bohb_hyperband,
                    num_samples=num_trials,
                ),
                run_config=ray.air.config.RunConfig(
                    checkpoint_config=ray.air.config.CheckpointConfig(num_to_keep=1, ),
                    failure_config=ray.air.config.FailureConfig(max_failures=5),
                    stop=CombinedNaNAndIterationStopper(metric="val_loss", max_iterations=20),  # Custom combined stopper
                    name=tune_name,
                    local_dir=this_output_dir, ),
            )

            analysis = tuner.fit()
            file_name = tune_name + '.analysis'
            file_path = os.path.join(self.this_output_dir, file_name)
            # Save the ResultGrid object to disk
            with open(file_path, 'wb') as f:
                pickle.dump(analysis, f)

            best_result = analysis.get_best_result(metric="val_loss", mode="min")
            lr = best_result.config['lr']
            wd = best_result.config['wd']
            batch_size = best_result.config['batch_size']
            T_0 = best_result.config['T_0']
            T_mult = best_result.config['T_mult']

            # Initialize these variables with default values
            transforms_min_scale = None
            mu_lambda_param = None
            hidden_dim = None
            output_dim = None

            if SSL is not None:  # pre-training
                if self.architecture in ['VICRegLConvNext', 'VICRegConvNext', 'VICReg', 'VICRegL']:
                    transforms_min_scale = best_result.config.get('transforms_min_scale')
                    mu_lambda_param = best_result.config.get('mu_lambda_param')
                    hidden_dim = best_result.config.get('hidden_dim')
                    output_dim = best_result.config.get('output_dim')

            print('\tbest config: ', analysis.get_best_result(metric="val_loss", mode="min"))
        except Exception as e:
            print("Exception {}: ".format(e))
        finally:
            print('finished hyper-param tuning')
            ray.shutdown()
            cleanup_checkpoints(tmp_log_dir)

        if SSL is not None:  # pre-training
            return lr, wd, batch_size, T_0, T_mult, transforms_min_scale, mu_lambda_param, hidden_dim, output_dim
        else:
            return lr, wd, batch_size, T_0, T_mult

    def train_and_tune_OneStrategyModel(self, tune_fc_only: bool, criterion: Any, start: int = 0, max_fold: Optional[int] = None) -> None:
        """
        Train and tune a model using a single strategy.
        
        Args:
            tune_fc_only (bool): Whether to only tune fully connected layers
            criterion (Any): Loss function
            start (int): Starting fold number
            max_fold (Optional[int]): Maximum fold number
        """
        # Set max_fold to num_folds if not specified
        max_fold = self.num_folds if max_fold is None else min(max_fold + 1, self.num_folds)
        
        for k in range(start, max_fold):
            print('#' * 60)
            print('Fold: {}'.format(k))
            # reinitialize hyperparams for this fold
            self.lr = self.wd = self.batch_size = self.T_0 = self.T_mult = None

            # initialize trainer and architecture
            self.model_wrapper = RGBYieldRegressor_Trainer(
                pretrained=self.pretrained,
                tune_fc_only=tune_fc_only,
                architecture=self.architecture,
                criterion=criterion,
                device=self.device,
                workers=self.workers,
                this_output_dir=self.this_output_dir,
            )
            # load weights and skip rest of the method if already trained
            if self.model_wrapper.load_model_if_exists(model_dir=self.this_output_dir, strategy=None, k=k):
                continue

            self._tune_OneFold_OneStrategyModel(tune_fc_only=tune_fc_only,
                                                 criterion=criterion,
                                                 k=k,
                                                 state_dict=None)

            self._train_OneFold_OneStrategyModel(tune_fc_only=tune_fc_only,
                                                 criterion=criterion,
                                                 k=k,
                                                 state_dict=None)

    def train_and_tune_lightlySSL(self, start: int = 0, SSL_type: str = 'VICReg', domain_training_enabled: bool = True) -> None:
        """
        Train and tune a model using self-supervised learning.
        
        Args:
            start (int): Starting fold number
            SSL_type (str): Type of SSL framework to use
            domain_training_enabled (bool): Whether to enable domain training
        """
        training_strategy = ['self-supervised','domain-tuning']
        training_strategy_params = {
            training_strategy[0]: {
                'tune_fc_only': False,
                'augmentation': True,
                'lr': None,
                'wd': None,
                'batch_size': None,
                'T_0': None,
                'T_mult': None
            },
            training_strategy[1]: {
                'tune_fc_only': True,
                'augmentation': True,
                'lr': None,
                'wd': None,
                'batch_size': None,
                'T_0': None,
                'T_mult': None
            }
        }
        tune_name = dict()
        tune_name['self-supervised'] = ''
        tune_name['domain-tuning'] = ''

        if SSL_type == 'SimCLR':
            criterion = {training_strategy[0]: NTXentLoss(),
                         training_strategy[1]: nn.MSELoss(reduction='mean'),}
        elif SSL_type == 'VICReg' or SSL_type == 'VICRegConvNext':
            criterion = {training_strategy[0]: VICRegLoss(),
                         training_strategy[1]: nn.MSELoss(reduction='mean'), }
        elif SSL_type == 'SimSiam':
            criterion = {training_strategy[0]: NegativeCosineSimilarity(),
                         training_strategy[1]: nn.MSELoss(reduction='mean'), }
        elif SSL_type == 'VICRegL' or SSL_type == 'VICRegLConvNext':
            criterion = {training_strategy[0]: VICRegLLoss(),
                         training_strategy[1]: nn.MSELoss(reduction='mean'), }

        print(f"Training with {SSL_type} SSL", flush=True)

        SSL = SSL_type
        for k in range(start, self.num_folds):
            print('#' * 60)
            print('Fold: {}'.format(k))
            # reinitialize hyperparams for this fold
            self.lr = self.wd = self.batch_size = self.T_0 = self.T_mult = None

            selected_training_strategies = training_strategy if domain_training_enabled else training_strategy[:-1]
            
            for i, strategy in enumerate(selected_training_strategies):
                print(f'{"#" * 60}\nFold: {k}, Strategy: {strategy}')
                
                if i == 0:  # Self-supervised phase
                    # Tune hyperparameters first
                    self._tune_OneFold_OneStrategyModel(
                        tune_fc_only=training_strategy_params[strategy]['tune_fc_only'],
                        criterion=criterion[strategy],
                        k=k,
                        state_dict=None,
                        strategy=strategy,
                        SSL=SSL_type
                    )
                    
                    # Initialize model with tuned parameters
                    if self.hidden_dim is not None and self.output_dim is not None:
                        print(f'Model trainer with embedding dimensions, hidden: {self.hidden_dim} output: {self.output_dim}')
                        self.model_wrapper = RGBYieldRegressor_Trainer(
                            pretrained=self.pretrained,
                            tune_fc_only=training_strategy_params['self-supervised']['tune_fc_only'],
                            architecture=self.architecture,
                            criterion=criterion['self-supervised'],
                            device=self.device,
                            workers=self.workers,
                            SSL=SSL_type,
                            this_output_dir=self.this_output_dir,
                            k=k,
                            hidden_dim=self.hidden_dim,
                            output_dim=self.output_dim,
                        )
                    else:
                        raise ValueError("hidden_dim and output_dim not set after tuning")
                else:
                    if strategy == 'domain-tuning':
                        # get and initialize model with pre-trained, self-supervised weights
                        state_dict = self.get_state_dict_if_exists(this_output_dir=self.this_output_dir, k=k, strategy='self-supervised')
                        if state_dict is not None:
                            self._prepare_model_for_domain_tuning(state_dict, SSL_type)
                            self.model_wrapper.set_criterion(criterion=criterion['domain-tuning'])
                            self.datamodule.augmented = True
                            SSL = None
                            print('setting SSL from {} to {} for hyper-parameter tuning'.format(str(self.model_wrapper.SSL), str(SSL)))
                            self.model_wrapper.SSL = SSL
                        else:
                            print("Warning: No pretrained weights found!")
                    # change architecture difference in label for training
                    self._tune_OneFold_OneStrategyModel(tune_fc_only=training_strategy_params[strategy]['tune_fc_only'],
                                                        criterion=criterion[strategy],
                                                        k=k,
                                                        state_dict=state_dict,
                                                        strategy=strategy,
                                                        # SSL_transforms=SSL_transforms,
                                                        SSL=SSL)
                i += 1

                if strategy == 'domain-tuning':
                    self.model_wrapper.set_criterion(criterion=criterion['domain-tuning'])
                    self.datamodule.augmented = True
                    SSL = None
                elif strategy == 'self-supervised':
                    self.model_wrapper.set_criterion(criterion=criterion['self-supervised'])
                    SSL = SSL_type
                print('setting SSL from {} to {} for training'.format(str(self.model_wrapper.SSL), str(SSL)))
                self.model_wrapper.SSL = SSL
                # load weights and skip rest of the method if already trained
                if self.model_wrapper.load_model_if_exists(model_dir=self.this_output_dir, strategy=strategy, k=k):
                    continue

                # load pretrained model weights for hyper parameter tuning
                if strategy == 'domain-tuning':
                    # get and initialize model with pre-trained, self-supervised weights
                    state_dict = self.get_state_dict_if_exists(this_output_dir=self.this_output_dir, k=k, strategy='self-supervised')
                    
                    if state_dict is not None:
                        print("Loading pretrained weights from self-supervised model...")
                        self._prepare_model_for_domain_tuning(state_dict, SSL_type)
                    else:
                        print("Warning: No pretrained weights found!")
                else:
                    state_dict = None
                    if SSL_type in ['SimSiam', 'VICRegL', 'VICRegLConvNext', 'VICReg', 'VICRegConvNext']:
                        self.model_wrapper.model.SSL_training = True

                self._train_OneFold_OneStrategyModel(tune_fc_only=training_strategy_params[strategy]['tune_fc_only'],
                                                     criterion=criterion[strategy],
                                                     k=k,
                                                     state_dict=state_dict,
                                                     strategy=strategy,
                                                     # SSL_transforms=SSL_transforms,
                                                     SSL=SSL)

    def _tune_OneFold_OneStrategyModel(self, tune_fc_only: bool, criterion: Any, k: int, state_dict: Optional[Dict[str, Any]] = None, strategy: str = '', SSL: Optional[str] = None) -> None:
        """
        Tune hyperparameters for a single fold and strategy.
        
        Args:
            tune_fc_only (bool): Whether to only tune fully connected layers
            criterion (Any): Loss function
            k (int): Fold number
            state_dict (Optional[Dict[str, Any]]): Initial model state
            strategy (str): Strategy identifier
            SSL (Optional[str]): SSL framework to use
        """
        # sample data into folds according to provided in validation strategy
        print('Attempting hyper-parameter tuning for fold {} and strategy {}'.format(k, strategy))
        print(f"Setting up data for fold {k} and strategy '{strategy}'")
        self.datamodule.setup_fold(
                                    fold=k,
                                    # trainset_transforms=SSL_transforms,
                                    dataset_type=self.dataset_type,
        )

        # tune hyper parameters #######################
        tune_name = 'Tuning_{}_{}_f{}{}_{}'.format(self.architecture, self.validation_strategy, k, strategy, self.patch_no)

        # load hyper params if already tuned
        if not self.load_hyperparameters_if_exist(k=k, strategy=strategy):
            datamodule_SSL_pointDS = None
            if (SSL is not None):
                if self.tune_by_downstream_performance:
                    datamodule_SSL_pointDS = self.datamodule_SSL_pointDS
                    datamodule_SSL_pointDS.setup_fold(
                        fold=k,
                        # trainset_transforms=SSL_transforms,
                        dataset_type='pointDS',
                    )
                self.lr, self.wd, self.batch_size, self.T_0, self.T_mult, self.transforms_min_scale, self.mu_lambda_param, self.hidden_dim, self.output_dim = self.tune_hyperparameters(
                                                                        datamodule=self.datamodule,
                                                                        datamodule_SSL_pointDS=datamodule_SSL_pointDS,
                                                                        tune_fc_only=tune_fc_only,
                                                                        this_output_dir=self.this_output_dir,
                                                                        patch_no=self.patch_no,
                                                                        seed=self.seed,
                                                                        criterion=criterion,
                                                                        tune_name=tune_name,
                                                                        state_dict=state_dict,
                                                                        SSL=SSL
                                                                        )
            else:
                self.lr, self.wd, self.batch_size, self.T_0, self.T_mult = self.tune_hyperparameters(
                    datamodule=self.datamodule,
                    datamodule_SSL_pointDS=datamodule_SSL_pointDS,
                    tune_fc_only=tune_fc_only,
                    this_output_dir=self.this_output_dir,
                    patch_no=self.patch_no,
                    seed=self.seed,
                    criterion=criterion,
                    tune_name=tune_name,
                    state_dict=state_dict,
                    SSL=SSL
                )
            self.save_hyperparameters(k=k, strategy=strategy)

    def async_save_checkpoint(self, state_dict, filepath):
        """Asynchronously save checkpoint using a separate process"""
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            executor.submit(torch.save, state_dict, filepath)

    def _train_OneFold_OneStrategyModel(self, tune_fc_only: bool, criterion: Any, k: int, state_dict: Optional[Dict[str, Any]] = None, strategy: str = '', SSL: Optional[str] = None) -> None:
        """
        Train a model for a single fold and strategy.
        
        Args:
            tune_fc_only (bool): Whether to only tune fully connected layers
            criterion (Any): Loss function
            k (int): Fold number
            state_dict (Optional[Dict[str, Any]]): Initial model state
            strategy (str): Strategy identifier
            SSL (Optional[str]): SSL framework to use
        """
        if strategy != '':
            strategy = '_' + strategy
        # self.lr, self.wd, self.batch_size = 0.01, 0.01, 32
        if not self.only_tune_hyperparameters:
            ###################### Training #########################
            # set hyper parameters for model wrapper and data module
            print('Hyper-parameter: lr={}, wd={}, bs={}, T_0={}, T_mult={}, transform_min_scale={}, mu=lamba={}, hidden_dim={}, output_dim={}'.format(self.lr, self.wd, self.batch_size, self.T_0, self.T_mult, self.transforms_min_scale, self.mu_lambda_param, self.hidden_dim, self.output_dim))
            self.datamodule.set_batch_size(batch_size=self.batch_size)
            if 'self-supervised' in strategy:
                print('setting trainset_transforms with min_scale={}'.format(self.transforms_min_scale))
                self.datamodule.set_transforms(
                    trainset_transforms=VICRegTransform(input_size=self.datamodule.kernel_size,
                                                        # require invariance to flips and rotations
                                                        hf_prob=0.5,
                                                        vf_prob=0.5,
                                                        rr_prob=0.5,
                                                        # important parameter
                                                        min_scale=self.transforms_min_scale,
                                                        # use a weak color jitter for invariance w.r.t small color changes
                                                        cj_prob=0.2,
                                                        cj_bright=0.1,
                                                        cj_contrast=0.1,
                                                        cj_hue=0.1,
                                                        cj_sat=0.1,
                                                        normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]},
                                                        )
                )
                if self.architecture == 'VICReg' or self.architecture == 'VICRegConvNext':
                    self.model_wrapper.criterion.mu_param = self.mu_lambda_param
                    self.model_wrapper.criterion.lambda_param = self.mu_lambda_param
            self.model_wrapper.set_hyper_parameters(lr=self.lr, wd=self.wd, batch_size=self.batch_size, T_0=self.T_0, T_mult=self.T_mult)

            # build and set data loader dict
            self.model_wrapper.set_dataloaders(dataloaders={'train': self.datamodule.train_dataloader(), 'val': self.datamodule.val_dataloader(), })

            # update optimizer to new hyper parameter set
            self.model_wrapper.set_optimizer()

            # data parallelize and send model to device:: only if not already done -> if strategy '' or 'self-supervised'
            if strategy == '' or 'self-supervised' in strategy:
                self.model_wrapper.parallize_and_to_device()

            # measure training time
            self.model_wrapper.start_timer()

            # Train and evaluate
            print('training fold {} for {} epochs'.format(k, self.num_epochs))
            self.model_wrapper.k = k
            self.model_wrapper.train(patience=self.patience,
                                min_delta=self.min_delta,
                                num_epochs=self.num_epochs,
                                min_epochs=self.min_epochs,
                                )


            # save best model
            # torch.save(self.model_wrapper.model.state_dict(), os.path.join(self.this_output_dir, 'model_f' + str(k) + strategy + '.ckpt'))
            self.async_save_checkpoint(self.model_wrapper.model.state_dict(), 
                                     os.path.join(self.this_output_dir, f'model_f{k}{strategy}.ckpt'))

            run_time = self.model_wrapper.end_timer_and_get_time('\nEnd training for fold {}'.format(k))

            # save training statistics
            df = pd.DataFrame({'train_loss': self.model_wrapper.train_mse_history,
                               'val_loss': self.model_wrapper.test_mse_history,
                               'best_epoch': self.model_wrapper.best_epoch,
                               'best_loss': self.model_wrapper.best_loss,
                               'time': run_time,
                               })
            df.to_csv(os.path.join(self.this_output_dir, 'training_statistics_f' + str(k) + strategy + '.csv'), encoding='utf-8')

            # save learning rates
            df = pd.DataFrame({'lrs': self.model_wrapper.lrs,
                               })
            df.to_csv(os.path.join(self.this_output_dir, 'lrs_f' + str(k) + strategy + '.csv'), encoding='utf-8')

    def _prepare_model_for_domain_tuning(self, state_dict, SSL_type):
        """
        Prepares the model for domain tuning by loading pretrained weights and setting up layers.
        
        Args:
            state_dict: The state dictionary from the self-supervised model
            SSL_type: The type of SSL being used (e.g., 'VICReg', 'SimSiam', etc.)
        
        Returns:
            bool: True if weights were loaded successfully, False otherwise
        """
        if state_dict is not None:
            print("Loading pretrained weights from self-supervised model...")
            
            # Create new state dict with only backbone weights
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                # Only include backbone parameters
                if ('backbone' in key and 'projection_head' not in key and 
                    'prediction_head' not in key):
                    # Keep the module.backbone. prefix or add it if missing
                    if key.startswith('module.backbone.'):
                        name = key  # Keep as is
                    elif key.startswith('backbone.'):
                        name = f'module.{key}'  # Add module prefix
                    elif key.startswith('module.'):
                        name = f'module.backbone.{key[7:]}'  # Add backbone after module
                    else:
                        name = f'module.backbone.{key}'  # Add full prefix
                    
                    new_state_dict[name] = value
            
            print(f"Attempting to load {len(new_state_dict)} backbone layers...")
            
            try:
                # Load the filtered state dict
                missing_keys, unexpected_keys = self.model_wrapper.model.load_state_dict(new_state_dict, strict=False)
                print(f"Successfully loaded pretrained weights:")
                print(f"Missing keys: {len(missing_keys)}")
                if len(missing_keys) > 0:
                    print(f"\tMissing keys: {missing_keys}")
                    # Filter out projection/prediction head keys from missing keys count
                    actual_missing = [k for k in missing_keys if 'backbone' in k]
                    if actual_missing:
                        print(f"\tActual missing backbone keys: {actual_missing}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0:
                    print(f"\tUnexpected keys: {unexpected_keys}")
                print(f"Loaded {len(new_state_dict)} backbone layers")
            except Exception as e:
                print(f"Error loading state dict: {e}")
                print("Current model keys:", list(self.model_wrapper.model.state_dict().keys())[:5])
                print("Attempting to load keys:", list(new_state_dict.keys())[:5])
                raise
            
            if SSL_type in ['SimSiam', 'VICRegL', 'VICRegLConvNext', 'VICReg', 'VICRegConvNext']:
                self.model_wrapper.model.SSL_training = False
            
            # freeze conv layers and reinitialize FC layer
            self.model_wrapper.reinitialize_fc_layers()
            self.model_wrapper.disable_all_but_fc_grads()
            return True
        else:
            print("Warning: No pretrained weights found!")
            return False
