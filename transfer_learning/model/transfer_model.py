'''
Transfer learning version of ResNet3D model

@author: AI Assistant
'''

import os
import sys
import numpy as np
from collections import OrderedDict

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"Critical dependency missing: {str(e)}")
    print("Please ensure you are in the correct Python environment with all dependencies installed.")
    sys.exit(1)

from .modules import ResidualNet3D

class TransferResNet3D(nn.Module):
    '''
    Transfer learning version of ResNet3D that supports:
    - Loading pretrained weights
    - Freezing specific layers
    - Fine-tuning selected parts of the network
    
    Args:
        input_dims = [h,w,c] - list or tuple containing dimensions of each input slice
        width_f = width factor for network (default=1)
        pretrained_path = path to pretrained weights file (optional)
        freeze_features = whether to freeze the feature extraction layers (default=True)
        num_fc_layers = number of FC layers in new classification head (default=3)
        fc_dims = list of FC layer dimensions (default=[512,512,1])
    '''
    def __init__(self, input_dims, width_f=1, pretrained_path=None, 
                 freeze_features=True, num_fc_layers=3, fc_dims=[512,512,1]):
        super(TransferResNet3D, self).__init__()
        
        # Initialize feature extractor
        self.features = ResidualNet3D(z=1, width_f=width_f)
        
        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
            
        # Freeze feature layers if specified
        if freeze_features:
            self._freeze_features()
            
        # Calculate flattened feature dimensions
        input_size = [input_dims[2], input_dims[0], input_dims[1]]
        self.flat_ftrs = self._get_flat_fts(input_dims)
        
        # Create new fully connected layers
        fc_layers = []
        in_dim = self.flat_ftrs
        
        for i in range(num_fc_layers-1):
            fc_layers.extend([
                ('fc{}'.format(i+1), nn.Linear(in_dim, fc_dims[i])),
                ('relu{}'.format(i+1), nn.ReLU(inplace=True))
            ])
            in_dim = fc_dims[i]
            
        # Add final layer
        fc_layers.append(('fc{}'.format(num_fc_layers), 
                         nn.Linear(in_dim, fc_dims[-1])))
        
        self.fc = nn.Sequential(OrderedDict(fc_layers))

    def _load_pretrained_weights(self, weights_path):
        """Load pretrained weights and handle module prefixes"""
        checkpoint = torch.load(weights_path, map_location='cpu')
        new_checkpoint = OrderedDict()
        
        # Handle module prefix in state dict
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                name = k[7:]  # Remove 'module.' prefix
            else:
                name = k
            new_checkpoint[name] = v
            
        # Load only feature extractor weights
        feature_dict = OrderedDict()
        for k, v in new_checkpoint.items():
            if k.startswith('features.'):
                feature_dict[k] = v
                
        self.features.load_state_dict(feature_dict)

    def _freeze_features(self):
        """Freeze all parameters in feature extractor"""
        for param in self.features.parameters():
            param.requires_grad = False
            
    def unfreeze_features(self, layers=None):
        """
        Unfreeze specific layers in feature extractor
        Args:
            layers: List of layer names to unfreeze, or None to unfreeze all
        """
        if layers is None:
            # Unfreeze all feature layers
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.features.named_parameters():
                if any(layer in name for layer in layers):
                    param.requires_grad = True

    def _get_flat_fts(self, in_size):
        """Calculate flattened feature dimensions"""
        f = self.features(torch.ones(1,1,*in_size))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x_cnn = x.view(x.size()[0], -1)
        x = self.fc(x_cnn)
        return x, x_cnn