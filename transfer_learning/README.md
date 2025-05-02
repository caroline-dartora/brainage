# Transfer Learning for Brain Age Prediction

This guide explains how to use the transfer learning setup for brain age prediction from T1-weighted MRI scans.

## Prerequisites

- Python 3.7+
- Required packages (install using `pip install -r requirements.txt`):
  - PyTorch ≥1.9.0
  - MONAI ≥0.9.0
  - NumPy ≥1.19.2
  - Pandas ≥1.2.0
  - TensorBoard ≥2.5.0 
  - NiBabel ≥3.2.0
  - scikit-learn ≥0.24.0

## Data Preparation

1. Prepare your data in a CSV file with the following columns:
   - `uid`: Unique identifier for each subject
   - `path_registered`: Full path to the T1-weighted MRI scan (NIfTI format)
   - `age_at_scan`: Subject's age at time of scan
   - `partition`: One of ['train', 'dev', 'test'] for data splitting

Example CSV format:
```csv
uid,path_registered,age_at_scan,partition
sub-001,/path/to/sub-001_T1w.nii.gz,45.2,train
sub-002,/path/to/sub-002_T1w.nii.gz,67.8,dev
sub-003,/path/to/sub-003_T1w.nii.gz,23.4,test
```

2. Ensure all MRI scans are:
   - In NIfTI format (.nii or .nii.gz)
   - Pre-processed (bias field correction, registration to standard space)
   - Of consistent dimensions (ideally [160,192,160])

## Training

### Basic Training

For basic transfer learning with all feature layers frozen:

```bash
python train.py \
  --input-csv /path/to/your/data.csv \
  --output-dir /path/to/save/results \
  --pretrained-weights /path/to/pretrained/model.pth \
  --batch-size 4 \
  --epochs 30
```

### Advanced Training Options

Fine-tune specific layers while training:

```bash
python train.py \
  --input-csv /path/to/your/data.csv \
  --output-dir /path/to/save/results \
  --pretrained-weights /path/to/pretrained/model.pth \
  --batch-size 4 \
  --epochs 30 \
  --lr 0.0002 \
  --unfreeze-layers resblock5 resblock6 \
  --fc-layers 3 \
  --patience 7 \
  --min-delta 0.01
```

### Command Line Arguments

- `--input-csv`: Path to CSV file containing data information (required)
- `--output-dir`: Directory to save results (required)
- `--pretrained-weights`: Path to pretrained model weights (required)
- `--batch-size`: Batch size for training (default: 4)
- `--epochs`: Number of epochs to train (default: 30)
- `--lr`: Learning rate (default: 0.0002)
- `--patience`: Early stopping patience in epochs (default: 7)
- `--min-delta`: Minimum improvement for early stopping (default: 0.01)
- `--unfreeze-layers`: Space-separated list of feature layers to unfreeze
- `--fc-layers`: Number of fully connected layers (default: 3)
- `--evaluate-test`: Flag to evaluate test set after training

## Outputs

The training script creates a timestamped directory under your specified output directory containing:

- `best_model.pth`: Weights of the best performing model
- `train_pred_epoch{N}.csv`: Training predictions for each epoch
- `val_pred_epoch{N}.csv`: Validation predictions for each epoch
- `test_predictions.csv`: Test set predictions (if --evaluate-test used)
- TensorBoard logs with:
  - Training/validation loss
  - Mean Absolute Error (MAE)
  - Learning rate changes

## Monitoring Training

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir /path/to/output/dir
```

## Tips for Best Results

1. **Data Quality**:
   - Ensure consistent preprocessing across all scans
   - Check for outliers in your age distribution
   - Verify image registration quality

2. **Training Strategy**:
   - Start with frozen features and train only the FC layers
   - If needed, gradually unfreeze and fine-tune deeper layers
   - Use early stopping to prevent overfitting

3. **Hyperparameter Tuning**:
   - Adjust learning rate based on training stability
   - Increase batch size if you have sufficient GPU memory
   - Modify FC layer architecture based on your dataset size

## Troubleshooting

1. **Out of Memory Errors**:
   - Reduce batch size
   - Use fewer workers in the data loader
   - Check input image dimensions

2. **Poor Performance**:
   - Verify data preprocessing quality
   - Try unfreezing more layers
   - Adjust learning rate
   - Increase model width factor

3. **Slow Training**:
   - Ensure data is on fast storage
   - Increase number of workers in data loader
   - Use GPU if available

## Example Use Cases

### Basic Transfer Learning

```bash
python train.py \
  --input-csv data/subjects.csv \
  --output-dir results/transfer \
  --pretrained-weights models/pretrained.pth \
  --batch-size 4
```

### Fine-tuning for New Domain

```bash
python train.py \
  --input-csv data/subjects.csv \
  --output-dir results/transfer \
  --pretrained-weights models/pretrained.pth \
  --batch-size 4 \
  --lr 0.0001 \
  --unfreeze-layers resblock4 resblock5 resblock6 \
  --epochs 50
```