#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for transfer learning brain age prediction

@author: CD
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import monai

from model.transfer_model import TransferResNet3D
from transforms.load_transform import load_transforms
from utils.misc import EarlyStopping, StoreOutput
from utils.dataloader import MRIDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Transfer learning for brain age prediction from T1-weighted MRI'
    )
    parser.add_argument('--input-csv', required=True,
                        help='Path to CSV with image paths and ages')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save outputs')
    parser.add_argument('--pretrained-weights', required=True,
                        help='Path to pretrained model weights')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Patience for early stopping (default: 7)')
    parser.add_argument('--min-delta', type=float, default=0.01,
                        help='Minimum improvement for early stopping (default: 0.01)')
    parser.add_argument('--unfreeze-layers', nargs='*',
                        help='Feature layers to unfreeze (default: None)')
    parser.add_argument('--fc-layers', type=int, default=3,
                        help='Number of FC layers (default: 3)')
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Evaluate test set after training')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Device and configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {'img_dim': [160, 192, 160], 'device': device}
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print(f'Loading data from {args.input_csv}')
    df = pd.read_csv(args.input_csv)
    
    # Setup transforms
    transforms_train = load_transforms(cfg, random_chance=0.7)
    transforms_val = load_transforms(cfg, random_chance=0)
    
    # Create datasets
    train_dataset = MRIDataset(
        df, partition='train',
        input_transform=transforms_train,
        is_training=True
    )
    val_dataset = MRIDataset(
        df, partition='dev',
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
    
    # Create test dataset if needed
    if args.evaluate_test:
        test_dataset = MRIDataset(
            df, partition='test',
            input_transform=transforms_val,
            is_training=False
        )
        test_loader = monai.data.ThreadDataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )
    
    # Setup output directory
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'transfer_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    
    # Initialize model
    print('Initializing model...')
    model = TransferResNet3D(
        input_dims=cfg['img_dim'],
        width_f=3,
        pretrained_path=args.pretrained_weights,
        freeze_features=True,
        num_fc_layers=args.fc_layers
    )
    
    if args.unfreeze_layers:
        print(f'Unfreezing layers: {args.unfreeze_layers}')
        model.unfreeze_features(args.unfreeze_layers)
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1,
        patience=5, verbose=True
    )
    criterion = torch.nn.L1Loss()
    early_stopper = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='min'
    )
    
    # Training loop
    best_mae = float('inf')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = StoreOutput()
        
        for i, (images, ages, uids) in enumerate(train_loader):
            images = images.to(device)
            ages = ages.float().to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(images)
            loss = criterion(predictions.squeeze(), ages)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_metrics.update(predictions.squeeze(), ages, uids)
            
            if i % 10 == 0:
                print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        train_mae = train_metrics.mae()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = StoreOutput()
        
        with torch.no_grad():
            for images, ages, uids in val_loader:
                images = images.to(device)
                ages = ages.float().to(device)
                
                predictions, _ = model(images)
                loss = criterion(predictions.squeeze(), ages)
                
                val_loss += loss.item()
                val_metrics.update(predictions.squeeze(), ages, uids)
        
        val_loss /= len(val_loader)
        val_mae = val_metrics.mae()
        
        # Log metrics
        print(f'Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}')
        print(f'Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}')
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step(val_mae)
        
        # Save predictions
        train_metrics.get_df().to_csv(
            os.path.join(output_dir, f'train_pred_epoch{epoch}.csv'),
            index=False
        )
        val_metrics.get_df().to_csv(
            os.path.join(output_dir, f'val_pred_epoch{epoch}.csv'),
            index=False
        )
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, 'best_model.pth')
            )
        
        # Early stopping
        if early_stopper(val_mae, epoch, model.state_dict()):
            print('Early stopping triggered')
            model.load_state_dict(early_stopper.load_best_state())
            break
    
    # Final test evaluation if requested
    if args.evaluate_test:
        model.eval()
        test_metrics = StoreOutput()
        
        with torch.no_grad():
            for images, ages, uids in test_loader:
                images = images.to(device)
                ages = ages.float().to(device)
                
                predictions, _ = model(images)
                test_metrics.update(predictions.squeeze(), ages, uids)
        
        test_mae = test_metrics.mae()
        print(f'\nTest MAE: {test_mae:.4f}')
        
        test_metrics.get_df().to_csv(
            os.path.join(output_dir, 'test_predictions.csv'),
            index=False
        )
    
    writer.close()
    print('Training completed!')

if __name__ == '__main__':
    main()