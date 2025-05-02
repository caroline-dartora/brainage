#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: GM
@edit: CD
#-----------------------------------------

         Brain AGE v2 - hold-out

#-----------------------------------------

Age prediction of brain images with convolutional neural networks.

---
Before using this script, please be sure that you have preprocessed your images using the "brain_age_trainer_preprocessing.py" script with:

python brain_age_trainer_preprocessing.py --input-csv your_csv_file.csv --output-csv your_new_csv_file.csv --output-dir /path/to/folder/to/plave/registered/images

After that, the files will be ready to be a input in this script.
---

This script briefly:
    1. Takes processed T1-weighted native MRI image in .nii.gz or .nii format.
    2. Apply augmentation on the training set;
    3. Applies curriculum learning to gradually increase training difficulty
    4. Parameter tuning is realized with 5 convolutional neural network with 20 epochs with SGD and initial LR of lambda=0.002 that decreased with a factor of 10 every 5 epochs

The input .csv file needs to contain, at least, the follow columns:
    'indx': a sample order of the images in decrescent order to be able to follow changes of indexed data.
    'Project': the name of the original project of the image. It will be necessary in the stratification of groups.
    'uid': the individual number of identification of subject.
    'path_registered': the path for the registered image.
    'age_at_scan': the chronological age of the subject in the image acquisition time.
    'partition': a column that need to be fullfilled with the word 'train', 'dev', and/or 'test', depending on your data splits (at least 'train' and 'dev' are mandatory)


"""
import os
import numpy as np
from utils.dataloader import mri_dset
from transforms.load_transform import load_transforms
import pandas as pd
import matplotlib.pyplot as plt
from model.model import ResNet3D
import time
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import copy
import utils.misc
from sklearn.metrics import mean_squared_error, mean_absolute_error
import monai
from monai.data import DataLoader, ThreadDataLoader
from utils.misc import EarlyStopping

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%% variables
parser = argparse.ArgumentParser(description='Training of model for brain age predictions from T1-weighted nifti images')

parser.add_argument('--input-csv', default='../brain_age/data/your_data.csv', help='Path to csv file with paths to img files and labels (chronological age)')
parser.add_argument('--output-dir', default='/path/to/brain_age/output_dir', help='Path to directory where output folder is created.')
parser.add_argument('--evaluate-test-set', dest='evaluate_test_set', action='store_false',help='If you have images labeled "test" in you csv file and want evaluate it after training')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-bs', '--batch-size', default=20, type=int,metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-epochs', default=20, type=int,metavar='N', help='number of epochs (default: 20)')
parser.add_argument('--comment', default='test_public_script', help='Add comment to training session for outputdir')
parser.add_argument('--print-frequency', default=10, type=int, metavar='N',help='num batches to process before printing prediction error')
parser.add_argument('--curriculum-pace', default=0.2, type=float, help='How quickly to increase difficulty (default: 0.2)')
parser.add_argument('--patience', default=5, type=int, help='Number of epochs to wait before early stopping (default: 5)')
parser.add_argument('--min-delta', default=0.01, type=float, help='Minimum change in MAE to qualify as an improvement (default: 0.01)')
parser.set_defaults(evaluate_test_set=True)
args = parser.parse_args()

cfg = {
        'img_dim':[160,192,160],
        'device':torch.device('cuda'),
        }

#%%
t = time.localtime()
time_str = '%2d%02d%02d_%02d.%02d.%02d' % (t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)

print('fixing rand seed')
np.random.seed(0)
torch.random.manual_seed(0)

#%% Create datasets and data loaders
print('Creating dataset from %s' % args.input_csv)
print('Images in column path_registered are assumed to have been registered ')
df = pd.read_csv(args.input_csv,usecols=['uid','path_registered','age_at_scan','partition'])

# Initialize curriculum learning
sample_difficulty = utils.misc.SampleDifficulty(df[df['partition'] == 'train'])

# transforms
transforms_train = load_transforms(cfg,random_chance=.7)
transforms_test = load_transforms(cfg,random_chance=0)

# datasets for training, development and test sets
dset_training=mri_dset(df,
                       partition='train',
                       is_training=True,
                       input_transform=transforms_train,
                       sample_weights=sample_difficulty.get_weights()
                       )

dset_dev=mri_dset(df,
                       partition='dev',
                       is_training=False,
                       input_transform=transforms_test,
                       )

loader_training = monai.data.ThreadDataLoader(
        dset_training, batch_size=args.batch_size,
        sampler=dset_training.get_weighted_sampler(),
        num_workers=10,
        pin_memory=True,
        )

loader_dev = monai.data.ThreadDataLoader(
        dset_dev, batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        )

if args.evaluate_test_set:
    dset_test=mri_dset(df,
                           partition='test',
                           is_training=False,
                           input_transform=transforms_test,
                           )
    loader_test = monai.data.ThreadDataLoader(
            dset_test, batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            )

#%% Test dset and dataloader, get img needed for initializing network
index=0
img,label,uid = dset_training.__getitem__(index)
for img_tmp in [img,]:
    ix=32
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img_tmp[ix,:,:])
    plt.subplot(1,3,2)
    plt.imshow(img_tmp[:,ix,:])
    plt.subplot(1,3,3)
    plt.imshow(img_tmp[:,:,ix]);plt.colorbar()
    plt.show(block=False)
    print([img_tmp.min(),img_tmp.mean(),img_tmp.max()])
    plt.pause(3)
    plt.close()


#%% Initialize models and optimizers
results_dir=os.path.join(args.output_dir,time_str + '_'+args.comment.replace(' ','_').replace(',','_').replace('[','').replace(']','').replace('__','_'))
writer = SummaryWriter(results_dir,comment='')
writer.add_text('comment',args.comment)
print('Output directory created in %s' % results_dir)


loss_func = torch.nn.L1Loss() # optimize for mean absolute error

models ={ # create five models and use as ensemble predictions
       'ResNet3D_3x_0':ResNet3D(img.shape,width_f=3),
       'ResNet3D_3x_1':ResNet3D(img.shape,width_f=3),
       'ResNet3D_3x_2':ResNet3D(img.shape,width_f=3),
       'ResNet3D_3x_3':ResNet3D(img.shape,width_f=3),
       'ResNet3D_3x_4':ResNet3D(img.shape,width_f=3),
       }
optimizers = {}
schedulers= {}
for key in models.keys():
    print([key,count_parameters(models[key])])
    models[key] = models[key].to(cfg['device'])
    models[key]= torch.nn.DataParallel(models[key], device_ids=range(torch.cuda.device_count()))
    optimizers[key] = torch.optim.SGD(models[key].parameters(), lr=args.lr,momentum=.9)
    schedulers[key] = torch.optim.lr_scheduler.StepLR(optimizers[key], step_size=5, gamma=0.1)

#%% Train network
# Initialize early stopping for each model
early_stoppers = {key: EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='min')
                 for key in models.keys()}

for epoch in range(args.epochs):
    # Update curriculum learning for this epoch
    sample_difficulty.update_epoch(epoch)
    if epoch > 0:  # After first epoch, use prediction errors to determine difficulty
        sample_difficulty.update_difficulty(method='prediction_error')

    # Update dataset weights
    dset_training.sample_weights = sample_difficulty.get_weights(args.curriculum_pace)
    loader_training = monai.data.ThreadDataLoader(
        dset_training, batch_size=args.batch_size,
        sampler=dset_training.get_weighted_sampler(),
        num_workers=10,
        pin_memory=True,
    )

    phases = ['train','dev']
    eval_test_set = args.evaluate_test_set and epoch==(args.epochs-1) # eval test set at the end of training
    if eval_test_set:
        phases.append('test')

    ratings = {}
    for phase in phases:
        ratings[phase] = {}
        for key in models.keys():
            ratings[phase][key] = utils.misc.StoreOutput()

    print('--- starting training epoch ' + str(epoch) + ' ---')
    phase='train'
    start=time.time()
    for i,(img,age,uid) in enumerate(loader_training):
        age=age.type(torch.FloatTensor).to(cfg['device'])
        # since dataloading is the bottleneck we train all models at once
        for key in models.keys():
            loss=0
            start_model=time.time()
            models[key].train()
            tmp_pred,_ = models[key](img.detach().to(cfg['device']))

            loss += loss_func(tmp_pred.squeeze(),age)
            optimizers[key].zero_grad()
            loss.backward()

            optimizers[key].step()
            ratings[phase][key].update(tmp_pred.squeeze(),age,uid)
            if i%args.print_frequency==0:
                print([epoch,i,len(loader_training),key,loss.detach().to('cpu')])

        # Update sample difficulties based on predictions
        predictions = []
        for key in models.keys():
            df_tmp = ratings[phase][key].get_df()
            predictions.append(df_tmp['predicted_age'].to_numpy())
        mean_predictions = np.mean(predictions, axis=0)
        sample_difficulty.update_predictions(df_tmp['uid'].values, mean_predictions)

    #%%
    # -------------------------- evaluate dev set ----------------------
    phase='dev'
    with torch.no_grad():
        for i,(img,age,uid) in enumerate(loader_dev):
            age=age.type(torch.FloatTensor).to(cfg['device'])
            for key in models.keys():
                models[key].eval()
                tmp_pred,_ = models[key](img.detach().to(cfg['device']))
                ratings[phase][key].update(tmp_pred.squeeze_(1),age,uid)

    print('finished epoch %d' % epoch)

    #%%
    # -------------------------- evaluate test set ----------------------
    if eval_test_set:
        print('Evaluating test set')
        phase='test'
        with torch.no_grad():
            for i,(img,age,uid) in enumerate(loader_test):
                age=age.type(torch.FloatTensor).to(cfg['device'])
                for key in models.keys():
                    models[key].eval()
                    tmp_pred,_ = models[key](img.detach().to(cfg['device']))
                    ratings[phase][key].update(tmp_pred.squeeze_(1),age,uid)

    #%% Compute epoch statistics and add to tensorboard summary writer
    maes = {} # dictionary with MAE
    for phase in phases:
        predictions = [] # for calculating ensemble predictions
        for key in models.keys():
            mae = ratings[phase][key].mae()
            maes[phase+'_'+key] = mae

            df_tmp = ratings[phase][key].get_df()
            predictions.append(df_tmp['predicted_age'].to_numpy())
            correlation = np.corrcoef(df_tmp['predicted_age'],df_tmp['age_at_scan'])[0][1] # pearson correlation
            lims = [df_tmp['age_at_scan'].min()-3,df_tmp['age_at_scan'].max()+3] # x and y lims for plotting

            # generate scatterplots and add to tensorboard
            fig = plt.figure(figsize=(12,8))
            plt.scatter(df_tmp['age_at_scan'],df_tmp['predicted_age'],alpha=1)
            tstr= '%s - MAE: %.2f - rho: %.3f (%s)'% (phase,mae,correlation,key)
            plt.title(tstr,fontsize=16)
            plt.plot(lims,lims,'k:');plt.grid();
            plt.xlim(lims);plt.ylim(lims);
            plt.xlabel('Chronological age',fontsize=16)
            plt.ylabel('Predicted age',fontsize=16)
            writer.add_figure('predictions/'+phase+'_'+key,fig,epoch)
            plt.close()
            fname = os.path.join(results_dir,'predictions_'+phase+'_'+key+'.csv')
            ratings[phase][key].save_df(fname)

        # calculate ensemble predictions
        df_tmp['predicted_age'] = np.mean(predictions,axis=0)
        fname = os.path.join(results_dir,'predictions_'+phase+'_ensemble.csv')
        df_tmp.to_csv(fname,index=False)
        maes[phase + '_ensemble'] = mean_absolute_error(df_tmp['age_at_scan'], df_tmp['predicted_age'])

    writer.add_scalars('mae',maes,epoch)

    # add learning rates to tensorboard and update learning rate scheduler
    lrs = {}
    for key in models.keys(): # loop over models
        lrs[key] = optimizers[key].param_groups[0]['lr']
        schedulers[key].step()

    writer.add_scalars('lr',lrs,epoch)

    # save model weights and curriculum learning stats
    for key in models.keys():
        fname=results_dir+'/'+key+'.pth'
        print('saving weights in %s' % fname)
        torch.save(models[key].to('cpu').state_dict(), fname)
        models[key].to(cfg['device'])

    # Save curriculum learning statistics
    curriculum_stats = pd.DataFrame({
        'epoch': [epoch],
        'mean_difficulty': [sample_difficulty.df['difficulty'].mean()],
        'max_difficulty': [sample_difficulty.df['difficulty'].max()],
        'min_difficulty': [sample_difficulty.df['difficulty'].min()],
        'mean_weight': [sample_difficulty.df['weight'].mean()]
    })
    curriculum_stats.to_csv(os.path.join(results_dir, 'curriculum_stats.csv'),
                           mode='a', header=not os.path.exists(os.path.join(results_dir, 'curriculum_stats.csv')),
                           index=False)

    # Check early stopping criteria after dev set evaluation
    phase = 'dev'
    should_stop = []
    for key in models.keys():
        mae = ratings[phase][key].mae()
        should_stop.append(early_stoppers[key](mae, epoch, models[key].state_dict()))

    # If all models signal to stop, end training
    if all(should_stop):
        print(f'Early stopping triggered at epoch {epoch}')
        # Load best weights for each model
        for key in models.keys():
            models[key].load_state_dict(early_stoppers[key].load_best_state())
        break
