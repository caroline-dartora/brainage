#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-validation transfer learning training script for brain age prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error

from model.transfer_model import TransferResNet3D
from transforms.load_transform import load_transforms
from utils.misc import EarlyStopping, StoreOutput
from utils.dataloader import MRIDataset

def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-validation transfer learning for brain age prediction'
    )
    parser.add_argument('--input-csv', required=True,
                        help='Path to CSV with image paths and ages')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save results')
    parser.add_argument('--pretrained-weights', required=True,
                        help='Path to pretrained model weights')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--k-folds', type=int, default=10,
                        help='Number of folds for cross-validation (default: 10)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience in epochs (default: 7)')
    parser.add_argument('--min-delta', type=float, default=0.01,
                        help='Minimum improvement for early stopping (default: 0.01)')
    parser.add_argument('--unfreeze-layers', nargs='*',
                        help='Feature layers to unfreeze')
    parser.add_argument('--fc-layers', type=int, default=3,
                        help='Number of FC layers (default: 3)')
    parser.add_argument('--comment', default='',
                        help='Comment to add to output directory name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configuration
    cfg = {
        'img_dim': [160, 192, 160],
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # Setup
    t = time.localtime()
    time_str = '%2d%02d%02d_%02d.%02d.%02d' % (t.tm_year, t.tm_mon, t.tm_mday,
                                               t.tm_hour, t.tm_min, t.tm_sec)
    
    print('Fixing random seed')
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Load data
    print(f'Creating dataset from {args.input_csv}')
    df = pd.read_csv(args.input_csv)
    
    # Setup cross-validation
    kf = StratifiedGroupKFold(n_splits=args.k_folds)
    
    # Initialize transforms
    transforms_train = load_transforms(cfg, random_chance=0.7)
    transforms_val = load_transforms(cfg, random_chance=0)
    
    # Setup output directory
    results_dir = os.path.join(
        args.output_dir,
        time_str + '_' + args.comment.replace(' ', '_')
    )
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(results_dir)
    
    # Create directories for splits and predictions
    splits_dir = os.path.join(results_dir, 'splits')
    pred_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Cross-validation loop
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(df, df['Project'], df['uid']), 1
    ):
        print(f'\nFold {fold}/{args.k_folds}')
        
        # Split data
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        train_df['partition'] = 'train'
        val_df['partition'] = 'dev'
        fold_df = pd.concat([train_df, val_df])
        
        # Save fold split
        fold_df.to_csv(
            os.path.join(splits_dir, f'split_fold_{fold}.csv'),
            index=False
        )
        
        # Create datasets
        train_dataset = MRIDataset(
            fold_df, 
            partition='train',
            input_transform=transforms_train,
            is_training=True
        )
        val_dataset = MRIDataset(
            fold_df,
            partition='dev',
            input_transform=transforms_val,
            is_training=False
        )
        
        # Create dataloaders
        train_loader = monai.data.ThreadDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True
        )
        val_loader = monai.data.ThreadDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )
        
        # Initialize models for this fold
        models = {
            f'model_{i}': TransferResNet3D(
                input_dims=cfg['img_dim'],
                width_f=3,
                pretrained_path=args.pretrained_weights,
                freeze_features=True,
                num_fc_layers=args.fc_layers
            ) for i in range(5)
        }
        
        # Unfreeze layers if specified
        if args.unfreeze_layers:
            print(f'Unfreezing layers: {args.unfreeze_layers}')
            for model in models.values():
                model.unfreeze_features(args.unfreeze_layers)
        
        # Move models to device and setup parallel processing
        optimizers = {}
        schedulers = {}
        early_stoppers = {}
        
        for name, model in models.items():
            print(f'{name} trainable parameters: {count_parameters(model)}')
            model = model.to(cfg['device'])
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            models[name] = model
            
            optimizers[name] = torch.optim.Adam(
                model.parameters(),
                lr=args.lr
            )
            schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizers[name],
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
            early_stoppers[name] = EarlyStopping(
                patience=args.patience,
                min_delta=args.min_delta,
                mode='min'
            )
        
        criterion = torch.nn.L1Loss()
        
        # Training loop
        best_mae = float('inf')
        for epoch in range(args.epochs):
            print(f'\nEpoch {epoch+1}/{args.epochs}')
            
            # Training phase
            for name, model in models.items():
                model.train()
            train_metrics = {name: StoreOutput() for name in models}
            
            for i, (images, ages, uids) in enumerate(train_loader):
                images = images.to(cfg['device'])
                ages = ages.float().to(cfg['device'])
                
                # Train all models
                for name, model in models.items():
                    optimizers[name].zero_grad()
                    predictions, _ = model(images)
                    loss = criterion(predictions.squeeze(), ages)
                    
                    loss.backward()
                    optimizers[name].step()
                    
                    train_metrics[name].update(
                        predictions.squeeze(), ages, uids
                    )
            
            # Validation phase
            for name, model in models.items():
                model.eval()
            val_metrics = {name: StoreOutput() for name in models}
            
            with torch.no_grad():
                for images, ages, uids in val_loader:
                    images = images.to(cfg['device'])
                    ages = ages.float().to(cfg['device'])
                    
                    for name, model in models.items():
                        predictions, _ = model(images)
                        val_metrics[name].update(
                            predictions.squeeze(), ages, uids
                        )
            
            # Calculate metrics
            train_maes = {
                name: metrics.mae() 
                for name, metrics in train_metrics.items()
            }
            val_maes = {
                name: metrics.mae()
                for name, metrics in val_metrics.items()
            }
            
            # Log metrics
            for name in models:
                writer.add_scalars(
                    f'mae_fold_{fold}/{name}',
                    {
                        'train': train_maes[name],
                        'val': val_maes[name]
                    },
                    epoch
                )
                writer.add_scalar(
                    f'lr_fold_{fold}/{name}',
                    optimizers[name].param_groups[0]['lr'],
                    epoch
                )
                
                # Update learning rate
                schedulers[name].step(val_maes[name])
                
                # Save predictions
                val_metrics[name].get_df().to_csv(
                    os.path.join(
                        pred_dir,
                        f'fold_{fold}_{name}_epoch_{epoch}.csv'
                    ),
                    index=False
                )
            
            # Calculate ensemble predictions
            ensemble_preds = []
            for name in models:
                df = val_metrics[name].get_df()
                ensemble_preds.append(df['predicted_age'].values)
            
            ensemble_df = val_metrics['model_0'].get_df().copy()
            ensemble_df['predicted_age'] = np.mean(ensemble_preds, axis=0)
            ensemble_mae = mean_absolute_error(
                ensemble_df['age_at_scan'],
                ensemble_df['predicted_age']
            )
            
            writer.add_scalar(
                f'mae_fold_{fold}/ensemble',
                ensemble_mae,
                epoch
            )
            
            # Save ensemble predictions
            ensemble_df.to_csv(
                os.path.join(
                    pred_dir,
                    f'fold_{fold}_ensemble_epoch_{epoch}.csv'
                ),
                index=False
            )
            
            # Early stopping check
            should_stop = []
            for name in models:
                if early_stoppers[name](
                    val_maes[name],
                    epoch,
                    models[name].state_dict()
                ):
                    should_stop.append(True)
                    models[name].load_state_dict(
                        early_stoppers[name].load_best_state()
                    )
                else:
                    should_stop.append(False)
            
            if all(should_stop):
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Save final models
        for name, model in models.items():
            torch.save(
                model.state_dict(),
                os.path.join(results_dir, f'fold_{fold}_{name}.pth')
            )
        
        # Store fold results
        fold_metrics.append({
            'fold': fold,
            'final_ensemble_mae': ensemble_mae,
            'best_model_mae': min(val_maes.values()),
            'epochs_trained': epoch + 1
        })
    
    # Save overall cross-validation results
    cv_results = pd.DataFrame(fold_metrics)
    cv_results.to_csv(
        os.path.join(results_dir, 'cv_results.csv'),
        index=False
    )
    
    print('\nCross-validation completed!')
    print(f'Average ensemble MAE: {cv_results.final_ensemble_mae.mean():.4f}')
    print(f'Standard deviation: {cv_results.final_ensemble_mae.std():.4f}')
    
    writer.close()

if __name__ == '__main__':
    main()