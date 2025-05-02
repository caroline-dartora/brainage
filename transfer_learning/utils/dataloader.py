#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset utilities for transfer learning

@author: GM
@edited: CD
"""
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    """Dataset class for MRI images during training and validation"""
    def __init__(self, df, partition=None, input_transform=None, is_training=False):
        """
        Args:
            df: DataFrame with columns ['uid', 'path_registered', 'age_at_scan', 'partition']
            partition: Which partition to use ('train', 'dev', 'test')
            input_transform: Transform to apply to input images
            is_training: Whether this is for training
        """
        self.df = df
        if partition is not None:
            self.df = self.df.query('partition==@partition')
        self.is_training = is_training
        self.input_transform = input_transform

    def __getitem__(self, index):
        """Get a single item from the dataset"""
        subj = self.df.iloc[index]
        path = subj['path_registered']
        img = self.input_transform(path)
        return img, subj['age_at_scan'], subj['uid']

    def __len__(self):
        """Get dataset size"""
        return len(self.df)