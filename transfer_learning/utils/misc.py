#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility classes and functions for transfer learning

@author: GM
@edited: CD
"""
import numpy as np 
import pandas as pd
import copy

class StoreOutput:
    """Accumulate outputs and labels and compute performance metrics"""
    def __init__(self):
        self.predictions = []
        self.chronological_age = []
        self.uids = []
        
    def update(self, pred, age, uid):
        self.predictions.extend(pred.detach().to('cpu').numpy())
        self.chronological_age.extend(age.detach().to('cpu').numpy())
        self.uids.extend(uid)
        
    def mae(self):
        """Calculate mean absolute error"""
        return np.mean(np.abs(np.array(self.predictions) - np.array(self.chronological_age)))
    
    def get_df(self):
        """Return results as pandas DataFrame"""
        return pd.DataFrame({
            'uid': self.uids,
            'age_at_scan': self.chronological_age,
            'predicted_age': self.predictions
        })

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' for metrics like loss/MAE, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None
        
    def __call__(self, value, epoch, model_state=None):
        """Returns True if training should stop"""
        if (self.mode == 'min' and value < self.best_value - self.min_delta) or \
           (self.mode == 'max' and value > self.best_value + self.min_delta):
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            if model_state is not None:
                self.best_state = copy.deepcopy(model_state)
        else:
            self.counter += 1
            
        return self.counter >= self.patience

    def load_best_state(self):
        """Return the best model state"""
        return self.best_state