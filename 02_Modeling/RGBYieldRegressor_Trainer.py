"""
Part of the self-supervised learning for crop yield prediction study entitled "Self-Supervised Learning Advances Crop Classification and Yield Prediction".
This script implements the training framework for crop yield prediction using VICReg self-supervised learning.

The framework supports:
- VICReg self-supervised learning with ResNet18 or ConvNeXt backbones
- Mixed precision training
- Early stopping and model checkpointing
- Learning rate scheduling
- Evaluation and prediction functionality

For license information, see LICENSE file in the repository root.
For citation information, see CITATION.cff file in the repository root.
"""

# Built-in/Generic Imports
import math
import time
import gc
import os
from copy import deepcopy
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict, Any

# Libs
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet
from torchvision.models.convnext import ConvNeXt
from torchvision.models.convnext import LayerNorm2d
from torchvision.models.convnext import CNBlockConfig
from functools import partial

from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead

from lars import LARS

# Own modules
from BaselineModel import BaselineModel

class VICReg(nn.Module):
    """
    VICReg model implementation for self-supervised learning.
    
    This class implements the VICReg architecture with support for both SSL pre-training
    and yield prediction fine-tuning.
    
    Args:
        backbone (nn.Module): The backbone network (e.g., ResNet18, ConvNeXt)
        num_filters (int): Number of filters in the final layer of the backbone
        prediction_head (bool): If True, use prediction head for yield prediction
        hidden_dim (int): Hidden dimension for projection head
        output_dim (int): Output dimension for projection head
    """
    
    def __init__(self, 
                 backbone: nn.Module, 
                 num_filters: int, 
                 prediction_head: bool = False, 
                 hidden_dim: int = 2048, 
                 output_dim: int = 8192) -> None:
        super().__init__()
        self.SSL_training = not prediction_head
        self.backbone = backbone
        self.num_filters = num_filters
        
        # Ensure we have valid dimensions
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2048
        self.output_dim = output_dim if output_dim is not None else 8192
        
        # Initialize both heads as None first
        self.prediction_head = None
        self.projection_head = None
        
        if prediction_head:
            # For yield prediction
            self.prediction_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.num_filters, 1))
        else:
            # For SSL pretraining
            self.projection_head = BarlowTwinsProjectionHead(
                num_filters, 
                self.hidden_dim,
                self.output_dim
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Model output (either projection or prediction)
        """
        x = self.backbone(x).flatten(start_dim=1)
        if self.SSL_training:
            if self.projection_head is None:
                raise AttributeError("projection_head not initialized but SSL_training is True")
            z = self.projection_head(x)
            output = z
        else:
            if self.prediction_head is None:
                raise AttributeError("prediction_head not initialized but SSL_training is False")
            h = self.prediction_head(x)
            output = h
        return output


class RGBYieldRegressor_Trainer:
    """
    Trainer class for crop yield prediction using VICReg self-supervised learning.
    
    This class handles the training, evaluation, and prediction pipeline for crop yield
    prediction using VICReg self-supervised learning with different backbone architectures.
    
    Args:
        device (torch.device): Device to run the model on
        workers (int): Number of worker processes for data loading
        pretrained (bool): Whether to use pretrained backbone
        tune_fc_only (bool): Whether to only tune the final layer
        architecture (str): Model architecture ('resnet18' or 'ConvNeXt')
        criterion (Optional[nn.Module]): Loss function
        SSL (Optional[str]): Self-supervised learning method ('VICReg' or 'VICRegConvNext')
        prediction_head (bool): Whether to use prediction head for yield prediction
        hidden_dim (int): Hidden dimension for projection head
        output_dim (int): Output dimension for projection head
        mean_yield_per_crop (Optional[Dict]): Mean yield per crop type for normalization
        std_yield_per_crop (Optional[Dict]): Standard deviation of yield per crop type
        this_output_dir (Optional[str]): Output directory for saving models
        k (Optional[int]): Cross-validation fold number
    """
    
    def __init__(self, 
                 device: torch.device,
                 workers: int,
                 pretrained: bool = False,
                 tune_fc_only: bool = False,
                 architecture: str = 'resnet18',
                 criterion: Optional[nn.Module] = None,
                 SSL: Optional[str] = None,
                 prediction_head: bool = False,
                 hidden_dim: int = 2048,
                 output_dim: int = 8192,
                 mean_yield_per_crop: Optional[Dict] = None,
                 std_yield_per_crop: Optional[Dict] = None,
                 this_output_dir: Optional[str] = None,
                 k: Optional[int] = None) -> None:
        self.architecture = architecture
        self.pretrained = pretrained
        self.lr = None
        self.momentum = 0.9
        self.wd = None
        self.batch_size = None
        self.SSL = SSL
        self.prediction_head = prediction_head
        self.mean_yield_per_crop, self.std_yield_per_crop = mean_yield_per_crop, std_yield_per_crop
        self.T_0 = None
        self.T_mult = None
        self.this_output_dir = this_output_dir
        self.k = k
        self.device = device
        self.workers = workers
        self.lrs = []
        self.scaler = None
        self.use_mixed_precision = False

        num_target_classes = 1
        if self.architecture == 'resnet18':
            # init resnet18 with FC exchanged
            if pretrained:
                self.model = models.resnet18(weights='DEFAULT')
            else:
                self.model = models.resnet18(weights=None)
            num_filters = self.model.fc.in_features
            self.num_filters = num_filters
            self.model.fc = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_filters, num_target_classes))

        elif self.architecture == 'ConvNeXt':
            # init ConvNext with FC exchanged
            if pretrained:
                self.model = models.convnext_tiny(weights='DEFAULT')
            else:
                self.model = models.convnext_tiny(weights=None)
            self.num_filters = 768
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            lastconv_output_channels = 768
            self.model.classifier = nn.Sequential(
                norm_layer(lastconv_output_channels), nn.Flatten(1),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_filters, num_target_classes))

        elif self.architecture == 'VICReg' or self.architecture == 'VICRegConvNext':
            # init resnet18 or convnext with VICReg head
            if self.architecture == 'VICReg':
                if pretrained:
                    raise NotImplementedError('pretrained VICReg not implemented')
                else:
                    resnet = models.resnet18(weights=None)
                    self.num_filters = resnet.fc.in_features
                    backbone = nn.Sequential(*list(resnet.children())[:-1])
            else:  # VICRegConvNext
                if pretrained:
                    raise NotImplementedError('pretrained VICRegConvNext not implemented')
                else:
                    convnext = models.convnext_tiny(weights=None)
                    self.num_filters = 768
                    backbone = nn.Sequential(*list(convnext.children())[:-1])
                
            # Ensure we have valid dimensions
            hidden_dim = hidden_dim if hidden_dim is not None else 2048
            output_dim = output_dim if output_dim is not None else 8192
            
            # Create VICReg model with explicit prediction_head flag
            self.model = VICReg(
                backbone=backbone,
                num_filters=self.num_filters,
                prediction_head=(SSL is None),  # True if not in SSL mode
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )



        # disable gradient computation for conv layers
        if pretrained and tune_fc_only:
            self.disable_all_but_fc_grads()

        if criterion is None:
            self.set_criterion()
        else:
            self.set_criterion(criterion=criterion)

    def print_children_require_grads(self) -> None:
        """Print gradient requirements for all model children."""
        children = [child for child in self.model.children()]
        i = 0
        for child in children:
            params = child.parameters()
            for param in params:
                print('child_{} grad required: {}'.format(i, param.requires_grad))
            i += 1

    def enable_grads(self) -> None:
        """Enable gradients for all model parameters."""
        if isinstance(self.model, BaselineModel):
            self.model.blocks['block_0'].requires_grad_(True)
            self.model.blocks['block_1'].requires_grad_(True)
            self.model.blocks['block_2'].requires_grad_(True)
            self.model.blocks['block_3'].requires_grad_(True)
            self.model.blocks['block_4'].requires_grad_(True)
            self.model.blocks['block_5'].requires_grad_(True)
            self.model.blocks['block_6'].requires_grad_(True)
        elif isinstance(self.model, DenseNet):
            self.model.features.requires_grad_(True)
        elif isinstance(self.model, (ResNet, VICReg, ConvNeXt)):
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = True

    def reinitialize_fc_layers(self) -> None:
        """
        Reinitialize the final layer for yield prediction.
        For VICReg, this means reinitializing the prediction_head when in prediction mode.
        """
        if self.architecture in ['VICReg', 'VICRegConvNext']:
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            if not model.SSL_training and hasattr(model, 'prediction_head'):
                final_layer = model.prediction_head[-1]
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)
                print("Reinitialized VICReg prediction head for yield prediction")

    def disable_all_but_fc_grads(self) -> None:
        """
        Freeze all layers except the final prediction layer.
        For VICReg, freeze backbone and enable only prediction_head gradients.
        """
        if self.architecture in ['VICReg', 'VICRegConvNext']:
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            model.SSL_training = False
            
            if model.prediction_head is None:
                model.prediction_head = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(model.num_filters, 1)
                )
            
            for param in model.backbone.parameters():
                param.requires_grad = False
                
            for param in model.prediction_head.parameters():
                param.requires_grad = True
                    
            print("VICReg: Froze backbone, enabled prediction head gradients")
            
        elif self.architecture in ['resnet18', 'baselinemodel']:
            children = [child for child in self.model.children()]
            for child in children[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
            for child in children[-1:]:
                for param in child.parameters():
                    param.requires_grad = True
                    
        else:
            raise ValueError(f"Architecture {self.architecture} not supported for gradient freezing")

    def set_dataloaders(self, dataloaders: Dict[str, Any]) -> None:
        """
        Set the dataloaders for training, validation, and testing.
        
        Args:
            dataloaders (Dict[str, Any]): Dictionary containing dataloaders for each phase
        """
        self.dataloaders = dataloaders

    def set_hyper_parameters(self, 
                           lr: float, 
                           wd: float, 
                           batch_size: int, 
                           momentum: float = 0.9, 
                           T_0: int = 50, 
                           T_mult: int = 2) -> None:
        """
        Set hyperparameters for training.
        
        Args:
            lr (float): Learning rate
            wd (float): Weight decay
            batch_size (int): Batch size
            momentum (float): Momentum for SGD
            T_0 (int): Initial restart period for cosine annealing
            T_mult (int): Multiplier for restart period
        """
        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.batch_size = batch_size
        self.T_0 = T_0
        self.T_mult = T_mult

    def set_optimizer(self, scheduler: bool = True) -> None:
        """
        Set up the optimizer and learning rate scheduler.
        
        Args:
            scheduler (bool): Whether to use learning rate scheduler
        """
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.wd
        )

        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult
            )
        else:
            self.scheduler = None
            
        print('using optimizer: {}; with scheduler: {}'.format(self.optimizer, self.scheduler))

    def set_criterion(self, criterion: nn.Module = nn.L1Loss(reduction='mean')) -> None:
        """
        Set the loss function.
        
        Args:
            criterion (nn.Module): Loss function to use
        """
        self.criterion = criterion

    def start_timer(self, device: Optional[torch.device] = None) -> None:
        """
        Start timing execution and clear memory.
        
        Args:
            device (Optional[torch.device]): Device to clear memory for
        """
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device=device)
            torch.cuda.synchronize()
        self.start_time = time.time()

    def end_timer_and_get_time(self, local_msg: str = '') -> float:
        """
        End timing and print execution statistics.
        
        Args:
            local_msg (str): Message to print with timing information
            
        Returns:
            float: Total execution time in seconds
        """
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()
        print("\n" + local_msg)
        print("Total execution time = {} sec".format(end_time - self.start_time))
        print('Max memory used by tensors = {} bytes'.format(torch.cuda.max_memory_allocated()))
        return end_time - self.start_time

    def set_mixed_precision(self, scaler: GradScaler) -> None:
        """
        Enable mixed precision training.
        
        Args:
            scaler (GradScaler): Gradient scaler for mixed precision training
        """
        self.scaler = scaler
        self.use_mixed_precision = True

    def train_step(self, epoch: int) -> float:
        """
        Perform one epoch of training.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average loss for the epoch
        """
        phase = 'train'
        running_loss = 0.0
        
        self.model.train()
        avg_loss = 0.0
        avg_output_std = 0.0
        iters = len(self.dataloaders[phase])

        for batch_idx, batch in enumerate(self.dataloaders[phase]):
            self.model.zero_grad()
            
            if self.SSL is None:
                x0, labels, crop_types, _ = batch
                x0 = x0.to(self.device)
                labels = labels.to(self.device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(enabled=self.use_mixed_precision):
                        outputs = self.model(x0)
                        loss = self.criterion(torch.flatten(outputs), labels.data)
                del x0, labels

            elif self.SSL in ['VICReg', 'VICRegConvNext']:
                (x0, x1), _, _, _ = batch
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(enabled=self.use_mixed_precision):
                        z0 = self.model(x0)
                        z1 = self.model(x1)
                        loss = self.criterion(z0, z1)
                del x0, x1

            running_loss += loss.item()

            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.SSL in ['VICReg', 'VICRegConvNext']:
                output = z0.detach()
                output = torch.nn.functional.normalize(output, dim=1)
                output_std = torch.std(output, 0)
                output_std = output_std.mean()
                
                w = 0.9
                avg_loss = w * avg_loss + (1 - w) * loss.item()
                avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

                out_dim = self.model.output_dim if not isinstance(self.model, nn.DataParallel) else self.model.module.output_dim
                collapse_level = max(0.0, 1 - math.sqrt(out_dim) * avg_output_std)
                del z0, z1

        epoch_loss = running_loss / (len(self.dataloaders[phase].dataset) / self.batch_size)

        if self.scheduler is not None:
            self.scheduler.step(epoch)
            self.lrs.append(self.scheduler.get_last_lr())

        if self.SSL in ['VICReg', 'VICRegConvNext']:
            print(
                f"Epoch-Loss = {epoch_loss:.2f} | "
                f"Mov_Avg_Loss = {avg_loss:.2f} | "
                f"Collapse Level: {collapse_level:.2f} / 1.00"
            )
        else:
            print('{} Avg Epoch-Loss: {:.4f}'.format(phase, epoch_loss))
            
        return epoch_loss

    def test(self, phase: str = 'test') -> float:
        """
        Evaluate model on the specified dataset.
        
        Args:
            phase (str): Dataset phase to evaluate on ('test' or 'val')
            
        Returns:
            float: Average loss on the dataset
        """
        running_loss = 0.0
        epoch_loss = 0.0

        self.model.to(self.device)
        self.model.eval()
        
        for batch_idx, batch in enumerate(self.dataloaders[phase]):
            if self.SSL is None:
                x0, labels, crop_types, _ = batch
                x0 = x0.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(x0)
                    loss = self.criterion(torch.flatten(outputs), labels.data)

            elif self.SSL in ['VICReg', 'VICRegConvNext']:
                (x0, x1), _, _, _ = batch
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                with torch.no_grad():
                    z0 = self.model(x0)
                    z1 = self.model(x1)
                    loss = self.criterion(z0, z1)
                    
            running_loss += loss.item()

        epoch_loss = running_loss / (len(self.dataloaders[phase].dataset)/self.batch_size)
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        return epoch_loss

    def predict(self, 
               phase: str = 'test', 
               predict_embeddings: bool = False, 
               crop_type: Optional[List[int]] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Generate predictions for the specified dataset.
        
        Args:
            phase (str): Dataset phase to predict on
            predict_embeddings (bool): Whether to predict embeddings instead of yield
            crop_type (Optional[List[int]]): List of crop types to predict for
            
        Returns:
            Tuple[np.ndarray, List[float]]: Predictions and ground truth labels
        """
        self.model.to(self.device)
        self.model.eval()

        local_preds = []
        local_labels = []

        if not predict_embeddings:
            for batch_idx, batch in enumerate(self.dataloaders[phase]):
                inputs, labels, crop_types, _ = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                crop_types = crop_types.to(self.device)

                with torch.no_grad():
                    y_hat = torch.flatten(self.model(inputs))

                local_preds.extend(y_hat.detach().cpu().numpy())
                local_labels.extend(labels.detach().cpu().numpy())

        else:
            for batch_idx, batch in enumerate(self.dataloaders[phase]):
                inputs, labels, crop_types, _ = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                crop_types = crop_types.to(self.device)

                with torch.no_grad():
                    y_hat = self.model(inputs).flatten(start_dim=1)

                local_preds.append(y_hat)
                local_labels.extend(labels.detach().cpu().numpy())

            local_preds = torch.cat(local_preds, dim=0)
            local_preds = local_preds.cpu().numpy()

        return local_preds, local_labels

    def train(self, 
             patience: int = 5, 
             min_delta: float = 0.01, 
             num_epochs: int = 200, 
             min_epochs: int = 200, 
             avg_epochs: int = 5) -> None:
        """
        Train the model with early stopping.
        
        Args:
            patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change in loss to be considered improvement
            num_epochs (int): Maximum number of epochs to train
            min_epochs (int): Minimum number of epochs to train
            avg_epochs (int): Number of epochs to average for early stopping
        """
        since = time.time()
        test_mse_history = []
        train_mse_history = []

        best_model_wts = deepcopy(self.model.state_dict())
        best_loss = np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        self.lrs = []

        self.model.to(self.device)
        self.model.zero_grad()
        print("min delta: {}".format(min_delta))
        
        recent_losses = []
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    train_epoch_loss = self.train_step(epoch=epoch)
                    train_mse_history.append(train_epoch_loss)
                if phase == 'val':
                    test_epoch_loss = self.test(phase='val')
                    test_mse_history.append(test_epoch_loss)

                    print('Epoch: {}; Epoch Loss: {}, Best Loss: {}'.format(epoch, test_epoch_loss, best_loss))

                    recent_losses.append(test_epoch_loss)
                    if len(recent_losses) > avg_epochs:
                        recent_losses.pop(0)

                    avg_recent_loss = sum(recent_losses) / len(recent_losses)

                    if test_epoch_loss < best_loss:
                        print('\tsaving best model')
                        best_loss = test_epoch_loss
                        best_model_wts = deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        epochs_without_improvement = 0
                    else:
                        if test_epoch_loss < (avg_recent_loss - min_delta):
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1

            if epochs_without_improvement >= patience and epoch >= min_epochs:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val MSE: {:4f}'.format(best_loss))

        self.optimizer_state_dict = self.optimizer.state_dict()
        self.model.load_state_dict(best_model_wts)
        self.test_mse_history = test_mse_history
        self.train_mse_history = train_mse_history
        self.best_epoch = best_epoch
        self.best_loss = best_loss

    def load_model_if_exists(self, 
                           model_dir: Optional[str] = None, 
                           strategy: Optional[str] = None, 
                           filename: Optional[str] = None, 
                           k: Optional[int] = None) -> bool:
        """
        Load model if exists and validate the loading was successful.
        
        Args:
            model_dir (Optional[str]): Directory containing the model
            strategy (Optional[str]): Training strategy used
            filename (Optional[str]): Specific model filename
            k (Optional[int]): Cross-validation fold number
            
        Returns:
            bool: True if model was loaded successfully
        """
        model_loaded = False
        min_layers_threshold = 0.5

        if filename is None:
            if strategy is not None:
                checkpoint_path = os.path.join(model_dir, 'model_f' + str(k) + '_' + strategy + '.ckpt')
            else:
                checkpoint_path = os.path.join(model_dir, 'model_f' + str(k) + '.ckpt')
        else:
            checkpoint_path = os.path.join(model_dir, filename)

        print(f'\tTry loading trained model from: {checkpoint_path}')
        print(f'\tFile exists: {os.path.exists(checkpoint_path)}')
        print(f'\tFile size: {os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else "N/A"} bytes')
        
        if os.path.exists(checkpoint_path):
            try:
                if torch.cuda.is_available():
                    state_dict = torch.load(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                
                print(f"\tState dict keys: {list(state_dict.keys())[:5]}...")
                
                model_dict = self.model.state_dict()
                print(f"\tModel dict keys: {list(model_dict.keys())[:5]}...")
                
                state_dict_has_module = any(k.startswith('module.') for k in state_dict.keys())
                model_dict_has_module = any(k.startswith('module.') for k in model_dict.keys())
                
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if state_dict_has_module and not model_dict_has_module:
                        name = k[7:] if k.startswith('module.') else k
                    elif not state_dict_has_module and model_dict_has_module:
                        name = f'module.{k}' if not k.startswith('module.') else k
                    else:
                        name = k
                    new_state_dict[name] = v
                
                filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
                
                total_layers = len(model_dict)
                loaded_layers = len(filtered_dict)
                loading_percentage = loaded_layers / total_layers
                
                print(f"Loaded {loaded_layers} / {total_layers} layers ({loading_percentage:.1%})")
                
                if loading_percentage < min_layers_threshold:
                    print(f"WARNING: Only {loading_percentage:.1%} of layers loaded - below threshold of {min_layers_threshold:.1%}")
                    print("\tMissing keys:", set(model_dict.keys()) - set(filtered_dict.keys()))
                    return False
                
                print("\tUpdating model state dict...")
                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict, strict=False)
                
                if self.workers > 1 and not isinstance(self.model, nn.DataParallel):
                    print("\tSetting up DataParallel...")
                    self.model = nn.DataParallel(self.model)
                
                model_loaded = True
                print("Model loaded successfully")
                
            except Exception as e:
                print(f"ERROR loading model: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                return False
        else:
            print(f"Checkpoint file not found: {checkpoint_path}")
            return False

        return model_loaded

    def parallize_and_to_device(self) -> None:
        """Set up data parallelization and move model to device."""
        if self.workers > 1:
            if not isinstance(self.model, nn.DataParallel):
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
