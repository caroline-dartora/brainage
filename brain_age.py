"""
@author: GM
@edited: CD

Age prediction of brain images convolutional neural networks. The script briefly:
    1. Takes a single unprocessed T1-weighted native MRI image in .nii.gz or .nii format.
    2. Uses FSL Flirt for a rigid registration (AC-PC alignment) and interpolation
    to 1x1x1mm3 voxel size.
    3. An ensemble of convolutional neural networks  predicts the age of the indivual,
    4. A .csv file 'uid'.csv is created with the average predicted age from the ensemble models.

The model was trained on more than 15000 images from the following cohorts: UK biobank, ADNI, AIBL and GENIC.

Method and results are detailed in paper: #TODO

To run:
python3 brain_age.py  --input-file /path/to/img.nii.gz --uid output_file_name_prefix --model-dir /path/to/trained/model/weights --output-dir /path/to/output/dir

"""
import pandas as pd
import torch
import argparse
import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

from collections import OrderedDict
import glob
import datetime

from utils.misc import native_to_tal_fsl
from model.model import ResNet3D
from transforms.load_transform import load_transforms

def setup_logging(log_file, log_level):
    """Configure logging to both file and console with the specified level."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up logging format with additional context
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format)
    
    # Create file handler that appends to the unified log file
    file_handler = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=5, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Get root logger and add handlers
    logger = logging.getLogger('brain_age')
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers = []
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set logger's level
    logger.setLevel(log_level)
    
    return logger

# Argument parser setup
parser = argparse.ArgumentParser(description='Brain age prediction from unprocessed T1-weighted nifti images')
parser.add_argument('--model-dir', type=str,default='/path/to/the/path/with/trained/model/weights', help='Path to directory containing the folders trained network weights')
parser.add_argument('--input-file', default='/path/to/your/input/nifti/registered/file/my_registered_nifti.nii.gz', help='Absoulute path to the input MRI file in (file assumed to be in .nii or .nii.gz format)')
parser.add_argument('--log-file', type=str, default='brain_age.log', help='Path to the unified log file. If not absolute, will be relative to output-dir')
parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
parser.add_argument('--output-dir', default='/path/to/brain_age/output_dir', help='Path to directory where all output files. Directory will be created if it doesn\'t exist')
parser.add_argument('--uid', type=str,help='Chosen unique id for output files that will located at output-dir/{uid_mni_dof_6.nii,uid.csv,uid_coronal.jpg}')
parser.add_argument('--no-new-registration', dest='registration', action='store_false',help='If a previous AC/PC-alignment exists (file output_folder/uid_mni_dof_6.nii) then setting this flag will use previous registration. If there is no previous transform, the transform will be performed anyway.')
parser.set_defaults(registration=True)
args = parser.parse_args()

# Set up logging with unified log file
if not os.path.isabs(args.log_file):
    args.log_file = os.path.join(args.output_dir, args.log_file)

<<<<<<< Updated upstream
args.device =  torch.device('cpu')
timestamp= '{:%Y-%m-%d_%H_%M_%S}'.format(datetime.datetime.now())
fname= os.path.join(args.output_dir,args.uid + '_info.log')
=======
logger = setup_logging(args.log_file, getattr(logging, args.log_level.upper()))

if not args.uid:
    args.uid = os.path.basename(args.input_file)
    args.uid = args.uid[:args.uid.find('.nii')]
    logger.info('uid not specified. Automatically setting uid to %s', args.uid)

logger.info('---- Started age prediction ----')
logger.info('Input file: %s', args.input_file)
logger.info('Output files: %s', os.path.join(args.output_dir, args.uid + '*'))
logger.info('Model directory: %s', args.model_dir)
logger.info('Force new registration: %s', str(args.registration))
logger.info('Log file: %s', args.log_file)
logger.info('Log level: %s', args.log_level)

args.device = torch.device('cpu')
logger.debug('Using device: %s', args.device)
>>>>>>> Stashed changes

# Check that input parameters are OK
if not os.path.exists(args.input_file):
    logger.error('input-file not specified or does not exist')
    raise ValueError('input-file not specified or does not exist')

if not args.output_dir:
    logger.error('output-dir not specified')
    raise ValueError('output-dir not specified')

if '.nii' not in os.path.basename(args.input_file):
    logger.error('input-file %s should be in .nii or .nii.gz format', args.input_file)
    raise ValueError(f'input-file {args.input_file} should be in .nii or .nii.gz format')

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    logger.info('Created output directory: %s', args.output_dir)

# Perform registration using FSL
if args.registration:
    logger.info('Performing rigid registration of input image to MNI template (AC-PC alignment)')
else:
    logger.info('Using existing registration if found, otherwise performing new registration')

dof = 6  # degrees of freedom of transform
try:
    native_to_tal_fsl(args.input_file, force_new_transform=args.registration, dof=dof, output_folder=args.output_dir, guid=args.uid)
    logger.info('Registration completed successfully')
except Exception as e:
    logger.error('Registration failed: %s', str(e))
    raise

tal_path = os.path.join(args.output_dir, args.uid + '_mni_dof_' + str(dof) + '.nii')
logger.debug('Talairach path: %s', tal_path)

# Log parameters in output csv file
rating_dict = OrderedDict()
rating_dict['uid'] = [args.uid]
rating_dict['model-dir'] = [args.model_dir]

# Load transforms and parameters
logger.info('Loading transforms and parameters')
params = {'ac_pc': True, 'img_dim': [160, 192, 160]}  # img_dim is dimensions used for training
transforms_test = load_transforms(params, random_chance=0)

# Load image with transforms
logger.info('Loading and transforming image')
try:
    img = transforms_test(tal_path).unsqueeze(0)
    logger.debug('Image loaded and transformed successfully')
except Exception as e:
    logger.error('Failed to load or transform image: %s', str(e))
    raise

# Initialize models
logger.info('Initializing models')
model_paths = np.sort(glob.glob(args.model_dir + '/*.pth'))
if len(model_paths) == 0:
    logger.error('No model weights found in directory: %s', args.model_dir)
    raise ValueError(f'No model weights found in directory: {args.model_dir}')

models = {}
for i, m in enumerate(model_paths):
    logger.debug('Initializing model %d from %s', i, m)
    models[i] = ResNet3D(np.array(params['img_dim'])//2, width_f=3)

# Load weights and run prediction
logger.info('Loading weights and running predictions')
predicted_ages = np.zeros(len(models))
<<<<<<< Updated upstream
for i,key in enumerate(models.keys()):
    # loading weights
    model_checkpoint = torch.load(model_paths[i],map_location='cpu')
    new_model_checkpoint = OrderedDict()
    for k, v in model_checkpoint.items():
        name = k[7:] # remove module. from items in model_checkpoint
        new_model_checkpoint[name] = v
    models[key].load_state_dict(new_model_checkpoint)
    models[key] = models[key].to('cpu')
    # evaluating model
    with torch.no_grad():
        models[key].eval()
        tmp,_ = models[key](img)
        predicted_ages[i] = tmp.detach().numpy()

print('---- Ages predicted from each individual model ---')
print(predicted_ages)
print('--'*20)
#%% Save
=======

for i, key in enumerate(models.keys()):
    try:
        # loading weights
        logger.debug('Loading weights for model %d', i)
        model_checkpoint = torch.load(model_paths[i], map_location='cpu')
        new_model_checkpoint = OrderedDict()
        for k, v in model_checkpoint.items():
            name = k[7:]  # remove module. from items in model_checkpoint
            new_model_checkpoint[name] = v
        models[key].load_state_dict(new_model_checkpoint)
        models[key] = models[key].to('cpu')
        
        # evaluating model
        with torch.no_grad():
            models[key].eval()
            tmp, _ = models[key](img)
            predicted_ages[i] = tmp.detach().numpy()
            logger.debug('Model %d prediction: %.2f years', i, predicted_ages[i])
    except Exception as e:
        logger.error('Failed to process model %d: %s', i, str(e))
        raise

logger.info('Ages predicted from individual models: %s', predicted_ages)

# Save results
>>>>>>> Stashed changes
rating_dict['predicted_age_mean'] = predicted_ages.mean()
rating_dict['predicted_age_std'] = predicted_ages.std()
csv_name = os.path.join(args.output_dir, args.uid + '.csv')

logger.info('Saving results: mean age %.3f years (std: %.3f) to %s', 
            rating_dict['predicted_age_mean'], 
            rating_dict['predicted_age_std'],
            csv_name)

try:
    pd.DataFrame(rating_dict).to_csv(csv_name, index=False)
    logger.info('Results saved successfully')
except Exception as e:
    logger.error('Failed to save results: %s', str(e))
    raise

logger.info('Brain age prediction completed successfully')
