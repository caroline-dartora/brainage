# Changelog

## [1.0.0] - 2025-04-30

### Added Curriculum Learning Implementation

#### Common Changes (Both Scripts)
- Added new `--curriculum-pace` command line argument (default 0.2) to control difficulty progression rate
- Added `SampleDifficulty` class in `utils/misc.py` to manage:
  - Initial difficulty based on age distinctness
  - Dynamic difficulty updates based on prediction errors
  - Gradual introduction of harder samples through weighted sampling
  - Sample weight calculation based on current epoch and difficulty

#### Changes to brain_age_trainer_holdout.py
- Added curriculum learning initialization for training set
- Modified training loop to:
  - Update sample difficulties each epoch
  - Track and save curriculum learning statistics
  - Use weighted sampling in data loader
  - Update difficulties based on ensemble predictions
- Added curriculum stats saving to `curriculum_stats.csv`

#### Changes to brain_age_trainer_crossvalidation.py
- Added per-fold curriculum learning initialization
- Modified cross-validation training to:
  - Reset curriculum learning for each fold
  - Track fold-specific difficulty progression
  - Use weighted sampling independently per fold
  - Update difficulties based on ensemble predictions per fold
- Added fold-specific curriculum stats saving (`curriculum_stats_cv_{fold}.csv`)

### Added Training Optimizations
- Added age-weighted loss to handle data imbalances across age groups
- Implemented early stopping to prevent overfitting

### Modified Files
1. brain_age_trainer_holdout.py:
   - Added curriculum learning argument and initialization
   - Updated training loop for curriculum progression
   - Added statistics tracking and storage
   - Added early stopping configuration arguments
   - Integrated age-weighted loss function
   - Modified training loop to support early stopping
   - Added restoration of best model weights when stopping

2. brain_age_trainer_crossvalidation.py:
   - Added curriculum learning argument and per-fold initialization
   - Updated cross-validation loop for curriculum progression
   - Added fold-specific statistics tracking
   - Added early stopping configuration arguments
   - Integrated age-weighted loss function
   - Added per-fold early stopping tracking
   - Modified training loop to handle early stopping per fold
   - Added restoration of best model weights per fold

3. utils/misc.py:
   - Added SampleDifficulty class for curriculum management
   - Implemented difficulty scoring methods
   - Added sample weight calculation
   - Added AgeWeightedLoss class with configurable binning and smoothing
   - Added EarlyStopping class with customizable patience and improvement thresholds
   - Added support for both minimization and maximization metrics

4. utils/dataloader.py:
   - Updated mri_dset class to support weighted sampling
   - Added sample weight handling

### Key Features
- Progressive difficulty increase during training
- Two difficulty metrics:
  1. Age distinctness (initial metric)
  2. Prediction error (used after first epoch)
- Configurable curriculum pace
- Independent curriculum progression for each cross-validation fold
- Comprehensive statistics tracking
- Weighted sampling based on sample difficulty

#### Age-Weighted Loss
- Dynamically computes weights based on age distribution
- Configurable number of age bins and smoothing factor
- Automatically normalizes weights to maintain scale
- Integrates with existing L1 loss function

#### Early Stopping
- Monitors validation MAE with configurable patience
- Supports minimum improvement thresholds
- Tracks and restores best model weights
- Independent tracking per model in ensemble
- Per-fold stopping for cross-validation

### Usage
To use curriculum learning, add the optional `--curriculum-pace` argument:
```
--curriculum-pace 0.2  # Default, slower difficulty progression
--curriculum-pace 0.5  # Faster difficulty progression
```

Lower values result in slower introduction of difficult samples, while higher values accelerate the introduction of harder examples.
