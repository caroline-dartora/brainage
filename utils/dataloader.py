#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: GM
@edited: CD
"""

import numpy as np
from torch.utils.data import Dataset
import torch

class mri_dset(Dataset):
    '''
    Custom dataset class for mri images during training and validation.
    '''
    def __init__(self, df, partition=None,
                 input_transform=None, is_training=False,
                 sample_weights=None):
        self.df = df
        if partition is not None:
            self.df = self.df.query('partition==@partition')
        self.is_training = is_training
        self.input_transform = input_transform
        self.sample_weights = sample_weights
        if sample_weights is not None:
            self.sample_weights = sample_weights[:len(self.df)]
            # Normalize weights
            self.sample_weights = self.sample_weights / self.sample_weights.sum()

    def __getitem__(self, index):
        subj = self.df.iloc[index]
        path = subj['path_registered']
        img = self.input_transform(path)
        return img, subj['age_at_scan'], subj['uid'], subj['guid']

    def __len__(self):
        return len(self.df)

    def get_weighted_sampler(self):
        """Returns a WeightedRandomSampler if sample_weights are provided"""
        if self.sample_weights is not None:
            # Ensure weights are positive and normalized
            weights = np.maximum(self.sample_weights, 1e-6)
            weights = weights / weights.sum()
            weights = torch.as_tensor(weights, dtype=torch.float64)
            return torch.utils.data.WeightedRandomSampler(
                weights,
                len(weights),
                replacement=True
            )
        return None

